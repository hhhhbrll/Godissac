"""手写渲染器：把答案文本以手写字体 + 逐字扰动写入 PDF 指定位置。

坐标约定（接口契约）：
空位用 (page, x, y, width, height, type) 描述：
- x, y   = 空位左端/下缘的坐标（pt，PDF 坐标系，y 向下增大）
- width  = 空位可用宽度
- height = 空位可用高度（单行空位约等于字号；多行答题区=首尾横线间距）
- type   = "underline" | "bracket" | "grid" | "block"

渲染位置规则（按正常写作业习惯）：
- 单行下划线/横线（underline）：字贴着下线写，基线 = y - BASELINE_GAP
- 括号（bracket）：字在括号内垂直居中
- 多行答题区（block/多条横线）：从最后一条横线开始往上写，最后一行贴着下线
  字高占行间距的75%，行距 = 字号 × 1.18（符合"两线之间靠下75%区域书写"）
- 田字格（grid）：字在格子内居中
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz

from .perturb import PerturbEngine, PerturbParams

BASELINE_GAP = 0.5   # 基线与下划线的间距 (pt)：横线穿过字的底部，符合手写习惯
MIN_FONT_SIZE = 6.0   # 自适应缩放下限：再小不可读，宁可溢出
GLYPH_H_EM = 0.88     # 字面高/em 比（ASCENT 880/1000）
FILL_RATIO = 0.75     # 字高占行间距的比例（用户需求 3/4）
LINE_SPACING = 1.18   # 多行行高系数


@dataclass
class BlankSpec:
    """一个待填空位（模块二 VLM 识别的输出即此格式）。"""

    page: int
    x: float          # 空位左端 x (pt)
    y: float          # 下划线/空位下缘 y (pt)
    width: float      # 空位可用宽度 (pt)
    answer: str       # 答案文本
    id: int = 0       # 空位编号
    height: float = 0.0   # 空位可用高度 (pt)
    type: str = "underline"  # 空位类型
    inline_chars: list = None  # inline句读：原文字符 [(char, x0, x1, y_top, y_bot)]
    segments: list = None      # block答题区：[(page, x, w, [横线y...])]（支持跨页续写）


@dataclass
class RenderResult:
    blank: BlankSpec
    font_size: float
    text_width: float
    overflowed: bool  # 答案在最小字号下仍超出空位宽度
    n_lines: int = 1  # 实际渲染行数（多行模式）


class HandwriteRenderer:
    def __init__(
        self,
        font_path: str | Path,
        params: PerturbParams | None = None,
        base_size: float = 13.0,
        fallback_font: str | Path | None = None,
    ):
        self.font_path = str(font_path)
        self.base_size = base_size
        self.font = fitz.Font(fontfile=self.font_path)  # 仅用于度量字宽
        self.fallback_path = str(fallback_font) if fallback_font else None
        self.fallback_font = fitz.Font(fontfile=self.fallback_path) if fallback_font else None
        self.engine = PerturbEngine(params)
        self.fallback_used: set[str] = set()   # 本次渲染中回退到 fallback 的字

    # ---------- 对外主接口 ----------

    def render(self, doc: fitz.Document, blank: BlankSpec, cover: bool = False) -> RenderResult:
        """把一个空位的答案渲染到 doc 的对应页面，返回渲染信息。

        cover: 是否先用白底覆盖空位区域。默认 False——印刷括号/答题横线
        保留在卷面上，手写字压在其上（更接近真实作业观感）。
        """
        page = doc[blank.page]
        text = blank.answer
        if not text.strip():
            return RenderResult(blank, 0.0, 0.0, False)

        btype = blank.type or "underline"

        if cover and blank.width > 0 and blank.height > 0 and btype != "inline":
            self._white_cover(page, blank)

        # 句读题（inline）：在原文字间插入"/"，不覆盖原文
        if btype == "inline":
            return self._render_inline(page, blank)

        # 答题横线区（block）：字贴着每条横线写（有 segments 时按横线逐行排）
        if btype == "block" and blank.segments:
            return self._render_rules_block(doc, blank)

        # 括号/方框/田字格：字号=括号高×1.1（打印实测×0.95偏小；学生字
        # 常顶满括号上下沿），水平居中，垂直坐进括号。
        # 超容量（压缩后仍装不下）：不缩字号不截断——装得下的主段进空位，
        # 溢出段在横线下方用小一号字补写（学生"写不下补下面"的自然习惯）。
        # 田字格（grid）除外：格子对位严格，超长直接截断。
        if btype in ("bracket", "box", "grid"):
            size = blank.height * 1.1 if blank.height > 0 else self.base_size
            size = min(size, self.base_size * 1.25)
            cap = max(1, round(blank.width / size))  # 括号容纳字数
            if len(text) > cap and btype != "grid":
                main, rest = text[:cap], text[cap:]
                text_w = self.font.text_length(main, fontsize=size)
                x_start = blank.x + (blank.width - text_w) / 2  # 居中
                y_center = blank.y - blank.height / 2
                baseline = y_center + 0.38 * size
                self._render_line(page, main, x_start, baseline, size)
                # 溢出段：横线下方，小一号，从空位左缘起写
                sub = size * 0.8
                base2 = blank.y + sub * 0.88 + 1.5
                self._render_line(page, rest, blank.x, base2, sub)
                return RenderResult(blank, size, blank.width, True)
            if len(text) > cap:
                text = text[-cap:]  # grid 兜底截断取尾部（语义核心多在后）
            text_w = self.font.text_length(text, fontsize=size)
            x_start = blank.x + (blank.width - text_w) / 2  # 居中
            y_center = blank.y - blank.height / 2
            baseline = y_center + 0.38 * size
            self._render_line(page, text, x_start, baseline, size)
            return RenderResult(blank, size, text_w, False)

        # 判断是否多行（block/多条横线答题区 或 height 足够大）
        is_multi = btype == "block" or (blank.height > self.base_size * 1.8)
        if is_multi and blank.height > self.base_size:
            size, lines = self._layout_multiline(text, blank)
            # 从最后一条线往上写：最后一行 baseline = blank.y - size*0.88（字底部贴着下缘）
            line_h = size * LINE_SPACING
            n = len(lines)
            last_baseline = blank.y - size * 0.88
            overflow = False
            rendered = 0
            for li, line in enumerate(lines):
                baseline = last_baseline - (n - 1 - li) * line_h
                rendered += self._render_line(page, line, blank.x, baseline, size)
            return RenderResult(blank, size, blank.width, overflow, n)

        # 单行模式（underline/单横线）：基准字号贴线写。
        # 超长不缩字号：装不下的部分写在横线下方（小一号），
        # 比挤压字体更自然（学生"写不下补在下面"的习惯）。
        size = self.base_size
        text_w = self.font.text_length(text, fontsize=size)
        if text_w <= blank.width:
            # 装得下：正常贴线写
            if btype == "grid" and blank.height > 0:
                # 田字格：垂直居中
                y_top = blank.y - blank.height
                baseline = (y_top + blank.y) / 2
            else:
                # 下划线：baseline = 下缘 - 字底部偏移（让字底部贴着下划线）
                baseline = blank.y - size * 0.88 - BASELINE_GAP
            self._render_line(page, text, blank.x, baseline, size)
            return RenderResult(blank, size, text_w, False)
        # 装不下：逐字累计找出横线上装得下的主段
        fit_n, w_acc = 0, 0.0
        for ch in text:
            cw = self.font.text_length(ch, fontsize=size)
            if w_acc + cw > blank.width:
                break
            w_acc += cw
            fit_n += 1
        if fit_n == 0:
            # 空位太窄连一个字都放不下（异常小空位）：退回自适应缩字号
            size = self._fit_size(text, blank.width)
            baseline = blank.y - size * 0.88 - BASELINE_GAP
            self._render_line(page, text, blank.x, baseline, size)
            return RenderResult(blank, size, blank.width, True)
        main, rest = text[:fit_n], text[fit_n:]
        baseline = blank.y - size * 0.88 - BASELINE_GAP
        self._render_line(page, main, blank.x, baseline, size)
        if rest:
            sub = size * 0.8
            base2 = blank.y + sub * 0.88 + 1.5
            self._render_line(page, rest, blank.x, base2, sub)
        return RenderResult(blank, size, blank.width, True)

    # ---------- 内部 ----------

    def _white_cover(self, page: fitz.Page, blank: BlankSpec) -> None:
        """用白底覆盖空位区域（消除印刷括号/下划线、红圈等）。

        关键：括号空位必须覆盖**括号本身**+**括号外侧少许**，否则左右括号
        仍可见。坐标约定：blank.y 是下缘（数值大），blank.height 是上下缘差。
        blank.x 是左缘，blank.width 是宽度。

        红圈清除：M8 流程先在原图上画红圈+白底圆，把坐标转回原 PDF 后，
        红圈仍可能残留 —— 现用稍大的白底矩形彻底抹掉。

        【关键防御】绝对不能把整个页面（w>300pt 或 h>50pt）的题目文字擦掉：
        block 类空位若 width 异常大（如 merge 误判的整页宽），只覆盖实际有意义的
        范围（高度按 block 高度，宽度按 block 实际宽度）。
        """
        x0 = blank.x
        x1 = blank.x + blank.width
        y_top = blank.y - blank.height   # 上缘
        y_bot = blank.y                  # 下缘

        # 按类型扩展覆盖范围（精准控制：不擦除题目文字）
        btype = blank.type or "underline"
        if btype == "bracket":
            # 括号：左扩 6pt（盖住左括号"（"和可能存在的红圈）、
            # 右扩 8pt（盖住右括号"）"）、上下扩展 6pt
            x0 -= 6
            x1 += 8
            y_top -= 6
            y_bot += 6
        elif btype == "underline":
            # 下划线：左右扩 2pt，上下扩 4pt（盖住下划线及附近红圈）
            x0 -= 2
            x1 += 2
            y_top -= 4
            y_bot += 4
        elif btype == "block":
            # 大答题区：上下边界各扩 4pt，左右边界各扩 2pt
            # 【防御性】宽度限制：若 width > 300pt，几乎肯定是 merge 误判，
            # 改成只覆盖 height 区域（避免擦除整页题目文字）
            if blank.width > 300:
                # 限制宽度：从中心向两边各取 min(width/2, 200) pt
                cx = (x0 + x1) / 2
                half = min(blank.width / 2, 200)
                x0 = cx - half
                x1 = cx + half
            x0 -= 2
            x1 += 2
            y_top -= 4
            y_bot += 4
        else:
            # grid/box：保守扩 3pt
            x0 -= 3
            x1 += 3
            y_top -= 3
            y_bot += 3

        rect = fitz.Rect(x0, y_top, x1, y_bot)
        # 用白色填充（无边框），overlay=True 让它盖在文字之上
        page.draw_rect(rect, color=None, fill=(1, 1, 1), width=0, overlay=True)

    # ---------- 内部 ----------

    def _render_inline(self, page, blank: BlankSpec) -> RenderResult:
        """句读题：在原文字符间插入"/"（手写体），不覆盖原文。

        blank.answer 形如 "每风月清朗/则焚香操弄数曲/弄罢复歌以诗词/而使子弟和之"。
        用答案片段在原文 inline_chars 序列中做子串匹配，片段末尾字后插入"/"。
        """
        chars = blank.inline_chars or []
        if not chars or "/" not in blank.answer:
            return RenderResult(blank, 0.0, 0.0, False)
        seq = [c for c in chars if c[0].strip()]
        if len(seq) < 2:
            return RenderResult(blank, 0.0, 0.0, False)
        orig = "".join(c[0] for c in seq)

        # 答案按"/"切片段，逐片段在原文中定位，记录每个片段末尾的原文下标
        fragments = [f for f in blank.answer.split("/") if f.strip()]
        if len(fragments) < 2:
            # 答案只有 0~1 个片段（如纯"/"或单段无斜杠）：无插入点
            # （曾因空 fragments 走兜底等分导致 n=-1 除零崩溃）
            return RenderResult(blank, 0.0, 0.0, False)
        insert_after: list[int] = []  # seq 下标：该字之后插"/"
        for frag in fragments[:-1]:  # 最后一段后面不插
            frag = frag.strip()
            pos = orig.find(frag)
            if pos == -1:
                # 容错：去掉首字再试（LLM可能多/少字）
                pos = orig.find(frag[1:]) if len(frag) > 2 else -1
                if pos != -1:
                    pos += 1
            if pos != -1:
                insert_after.append(pos + len(frag) - 1)
        # 兜底：匹配失败则按片段数等距
        if not insert_after:
            n = len(fragments) - 1
            step = len(seq) / (n + 1)
            insert_after = [int((k + 1) * step) - 1 for k in range(n)]

        size = min(self.base_size, (seq[0][4] - seq[0][3]) * 1.1)
        # 手写字体常缺"/"字形 → 回退字体（否则 insert_text 静默丢字）
        font_path = self.font_path
        font_name = "HWR"
        if not self.font.has_glyph(ord("/")) and self.fallback_font is not None:
            font_path = self.fallback_path
            font_name = "HWRFB"
            self.fallback_used.add("/")
        # 斜杠墨迹中心相对笔落点的偏移（em比）：实测渲染像素，不硬编码
        # （morph 缩放围绕笔落点，墨迹中心 = 笔落点 + dx_em*sx*size）
        dx_em = self._slash_ink_dx(font_path)
        sx = 0.6  # 横向压缩避免压到右字
        rendered = 0
        for idx in sorted(set(insert_after)):
            if idx >= len(seq) - 1 and idx == len(seq) - 1:
                continue
            a = seq[idx]
            b = seq[min(idx + 1, len(seq) - 1)]
            x_t = (a[2] + b[1]) / 2  # 两字中间（目标墨迹中心）
            y_base = a[4] + size * 0.1  # 基线略低于字底，斜杠贯穿
            x_pen = x_t - dx_em * sx * size
            t = self.engine.transform("/", size, self.font.text_length("/", fontsize=size))
            point = fitz.Point(x_pen, y_base + t.dy)
            morph = fitz.Matrix(sx, 1).prerotate(t.rotation)
            page.insert_text(point, "/", fontsize=t.size, fontname=font_name,
                             fontfile=font_path,
                             color=(t.ink, t.ink, t.ink), morph=(point, morph))
            rendered += 1
        return RenderResult(blank, size, 0, False, rendered)

    def _slash_ink_dx(self, font_path: str) -> float:
        """实测"/"墨迹中心相对笔落点的水平偏移（em 比），按字体缓存。"""
        cache = getattr(self, "_slash_dx_cache", None)
        if cache is None:
            cache = self._slash_dx_cache = {}
        if font_path in cache:
            return cache[font_path]
        import numpy as np
        doc = fitz.open()
        pg = doc.new_page(width=60, height=60)
        pg.insert_text(fitz.Point(20, 40), "/", fontsize=40, fontname="M",
                       fontfile=font_path)
        pix = pg.get_pixmap(dpi=72)  # 1px = 1pt
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        ink = arr[:, :, 0] < 128
        doc.close()
        dx = 0.5
        if ink.any():
            ys, xs = np.where(ink)
            dx = (xs.mean() - 20) / 40.0
        cache[font_path] = dx
        return dx

    def _render_rules_block(self, doc: fitz.Document, blank: BlankSpec) -> RenderResult:
        """答题横线区：答案逐行写在每条横线上（字底贴线，学生习惯）。

        segments: [(page, x, w, [横线y...]), ...]，跨页续写段在后。
        行数超出横线数时缩小字号；仍超出则截断并标记溢出。
        """
        segs = blank.segments
        # 展开为 [(page, x, w, rule_y)] 列表（阅读顺序）
        slots = [(pg, x, w, ry) for pg, x, w, rules in segs for ry in rules]
        n_rules = len(slots)
        if n_rules == 0:
            return RenderResult(blank, 0.0, 0.0, False)

        # 行距 = 段内相邻横线间距的最小值（缺省 15.5pt）
        gaps = []
        for _pg, _x, _w, rules in segs:
            for a, b in zip(rules, rules[1:]):
                gaps.append(b - a)
        gap = min(gaps) if gaps else 15.5

        # 字号：行距的 82%（字高占行距八成），封顶 base_size
        size = min(self.base_size, gap * 0.82)
        min_w = min(w for _pg, x, w, _ry in segs) - 4
        lines = self._wrap(blank.answer, min_w, size)
        # 行数超出横线数 → 缩字号（底线 8pt）
        while len(lines) > n_rules and size > 8.0:
            size *= 0.92
            lines = self._wrap(blank.answer, min_w, size)

        overflow = len(lines) > n_rules
        for i, line in enumerate(lines):
            if i >= n_rules:
                break
            pg, x, w, ry = slots[i]
            page = doc[pg]
            baseline = ry - size * 0.12   # 字底微贴横线
            self._render_line(page, line, x + 2, baseline, size)
        return RenderResult(blank, size, blank.width, overflow, len(lines))

    def _layout_multiline(self, text: str, blank: BlankSpec
                          ) -> tuple[float, list[str]]:
        """多行排版（简答题）。

        字号策略：字高占行间距的75%（FILL_RATIO），行高=字号×1.18。
        从最后一行开始往上排，行数装不下时缩字号。
        """
        # 目标行高 = 空高 / 行数（先假设n行）；初始字号按空高自适应
        # 先试基准字号
        size = self.base_size
        lines = self._wrap(text, blank.width, size)
        # 若行数超出空位则缩小字号直至装下（或到底线）
        for _ in range(24):
            line_h = size * LINE_SPACING
            if len(lines) * line_h <= blank.height + 1:
                break
            size *= 0.92
            if size <= MIN_FONT_SIZE:
                size = MIN_FONT_SIZE
                lines = self._wrap(text, blank.width, size)
                break
            lines = self._wrap(text, blank.width, size)
        # 如果文本短（只有一两行），可以适当放大字号填满空间
        if len(lines) <= 2 and size < self.base_size * 1.2:
            max_size_by_h = blank.height * FILL_RATIO / GLYPH_H_EM / LINE_SPACING * LINE_SPACING
            candidate = min(self.base_size * 1.2, max_size_by_h, size * 1.1)
            test_lines = self._wrap(text, blank.width, candidate)
            if len(test_lines) * candidate * LINE_SPACING <= blank.height + 1:
                size = candidate
                lines = test_lines
        return size, lines

    def _wrap(self, text: str, max_w: float, size: float) -> list[str]:
        """按字符宽度断行（中文场景逐字断行；首个超宽字符独行防死循环）。"""
        lines: list[str] = []
        cur = ""
        for ch in text:
            if self.font.text_length(cur + ch, fontsize=size) <= max_w or not cur:
                cur += ch
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
        return lines

    def _render_line(self, page, text: str, x0: float, y_base: float,
                     size: float) -> int:
        """渲染单行（逐字扰动），返回渲染字符数。"""
        x = x0
        n = 0
        for ch in text:
            # 缺字回退：主字体无该字形时用 fallback（如文楷）顶替
            font_path = self.font_path
            measure_font = self.font
            font_name = "HWR"  # 主字体名
            if ch.strip() and not self.font.has_glyph(ord(ch)) and self.fallback_font is not None:
                font_path = self.fallback_path
                measure_font = self.fallback_font
                font_name = "HWRFB"  # 回退字体名（必须与主字体名不同）
                self.fallback_used.add(ch)   # 留档：扩容字库的数据依据
            base_adv = measure_font.text_length(ch, fontsize=size)
            t = self.engine.transform(ch, size, base_adv)
            if ch.strip():  # 空白字符只占位不落笔
                point = fitz.Point(x, y_base + t.dy)
                rot = fitz.Matrix(1, 0, 0, 1, 0, 0).prerotate(t.rotation)
                page.insert_text(
                    point,
                    ch,
                    fontsize=t.size,
                    fontname=font_name,
                    fontfile=font_path,
                    color=(t.ink, t.ink, t.ink),
                    morph=(point, rot),  # 以字符自身落点为轴心旋转
                )
                n += 1
            x += t.advance
        return n

    def _fit_size(self, text: str, max_width: float) -> float:
        natural = self.font.text_length(text, fontsize=self.base_size)
        if natural <= max_width:
            return self.base_size
        return max(self.base_size * max_width / natural, MIN_FONT_SIZE)
