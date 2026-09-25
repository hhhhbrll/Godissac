"""确定性空位检测：坐标由程序精确计算（不用 VLM 猜）。

设计原则：
- 空位类型：underline（下划线/横线）、bracket（括号）、grid（田字格）、block（大答题区）
- 坐标 100% 来自 PDF 文本/绘图提取 或 CV 图像检测，精度 < 1pt / < 2px
- VLM 只负责"看题目 + 按编号给答案"，完全不接触坐标

返回的空位统一格式：
{
  "id": 序号(从1开始, 阅读顺序: 先上后下, 先左后右),
  "type": "underline" | "bracket" | "grid" | "block",
  "x": 左端x (pt),
  "y": 空位下缘y (pt, PDF坐标系),
  "w": 宽度 (pt),
  "h": 高度 (pt, 可写区域高度),
  "context": "空位前后的题目文字(供VLM理解)",
  "label_pos": (x, y) 标注编号的位置（在空位旁边画数字用）
}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pymupdf as fitz
import re

BlankType = Literal["underline", "bracket", "grid", "block"]


@dataclass
class Blank:
    id: int
    type: BlankType
    x: float
    y: float       # 下缘（下划线/最后一条横线的y）
    w: float
    h: float       # 可写高度（单行空位h≈字号；多行答题区h=两线/多线总间距）
    context: str = ""
    # 供渲染器用的派生字段
    answer: str = ""
    confidence: float = 1.0

    def to_spec(self) -> dict:
        """转为渲染器契约格式。"""
        return {
            "id": self.id,
            "page": self.page if hasattr(self, "page") else 0,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "width": round(self.w, 1),
            "height": round(self.h, 1),
            "context": self.context,
            "answer": self.answer,
            "confidence": self.confidence,
            "type": self.type,
        }


@dataclass
class PageBlanks:
    page_index: int
    blanks: list[Blank] = field(default_factory=list)
    # 标注了编号的预览图（给VLM看）
    labeled_image_path: Path | None = None
    page_w: float = 595.0
    page_h: float = 842.0


# ========== PDF 检测 ==========

def _group_horizontal_lines(lines: list[tuple[float, float, float, float]],
                            tol_y: float = 2.0, tol_gap: float = 6.0
                            ) -> list[tuple[float, float, float, float]]:
    """把同一水平线上的小线段合并成完整横线。
    输入/输出: [(x1, y, x2, y)]（y是线的y坐标）。
    """
    if not lines:
        return []
    # 按y聚类
    lines = sorted(lines, key=lambda l: (l[1], l[0]))
    groups: list[list[tuple]] = []
    for l in lines:
        placed = False
        for g in groups:
            gy = g[0][1]
            if abs(l[1] - gy) <= tol_y:
                g.append(l)
                placed = True
                break
        if not placed:
            groups.append([l])
    merged = []
    for g in groups:
        y = g[0][1]
        # 组内按x1排序，合并重叠或间隙<tol_gap的线段
        segs = sorted([(l[0], l[2]) for l in g])
        kept: list[list[float]] = []
        for sx1, sx2 in segs:
            if not kept:
                kept = [[sx1, sx2]]
            elif sx1 - kept[-1][1] <= tol_gap:
                kept[-1][1] = max(kept[-1][1], sx2)
            else:
                kept.append([sx1, sx2])
        for x1, x2 in kept:
            if x2 - x1 >= 8:
                merged.append((x1, y, x2, y))
    return merged


def _extract_pdf_rects(page: fitz.Page) -> list[tuple[float, float, float, float]]:
    """从 PDF 绘图指令中提取矩形（作为答题格/田字格边界）。
    返回 [(x0, y0, x1, y1)]（PDF pt）。

    PyMuPDF 的 drawings 按"路径"组织。闭合矩形有两种来源：
    (a) items 中含 're' 命令
    (b) 路径被画成"上下左右四段直线"，items 全是 'l'，但路径级 rect
        字段记录了外接矩形。

    关键陷阱：很多 PDF 编辑器会生成 fill=(1,1,1)（白底）的隐式矩形来防止
    文字穿透——它的边界恰好就是答题格的边界。因此不能简单剔除 fill-only
    矩形，只能剔除"有颜色填充"的（白底当无填充处理）。

    进一步陷阱：这些白底矩形经常是**整版内容同宽**（x0≈左边距，x1≈右边距），
    它们不是答题格，而是页面的防穿透层。需要按"是否超过60%页面宽"过滤：
    真正答题格远窄于页面文字区。
    """
    rects = []
    seen: set[tuple[float, float, float, float]] = set()
    page_w = page.rect.width
    # 内容区宽度阈值：宽于页面60%几乎都是防穿透层
    max_answer_w = page_w * 0.6
    for path in page.get_drawings():
        # 颜色过滤：红色批改线忽略
        color = path.get("color")
        if color and len(color) == 3:
            r, g, b = color
            if r > 0.7 and g < 0.5 and b < 0.5:
                continue
        # 填充色过滤：白底/无填充都保留；非白有颜色填充剔除（色块/底纹）
        fill = path.get("fill")
        stroke = path.get("stroke")
        if fill is not None and stroke is None:
            # 有色填充才算"色块"——白色 (1,1,1) 视为无填充
            if any(abs(c - 1.0) > 0.05 for c in fill):
                continue
        # 优先从 're' item 抓
        for item in path.get("items", []):
            if item[0] == "re":
                rc = item[1]
                if rc.height >= 2 and rc.width >= 8:
                    # 过滤：宽度超过60%页面宽的矩形（防穿透白底）
                    if rc.width >= max_answer_w:
                        continue
                    if rc.height >= 30:
                        continue
                    key = (round(rc.x0, 2), round(rc.y0, 2),
                           round(rc.x1, 2), round(rc.y1, 2))
                    if key not in seen:
                        seen.add(key)
                        rects.append((rc.x0, rc.y0, rc.x1, rc.y1))
        # 再从路径级 rect 抓（闭合线段矩形）
        pr = path.get("rect")
        if pr is not None and pr.height >= 2 and pr.width >= 8:
            if pr.width >= max_answer_w:
                continue
            if pr.height >= 30:
                continue
            key = (round(pr.x0, 2), round(pr.y0, 2),
                   round(pr.x1, 2), round(pr.y1, 2))
            if key not in seen:
                seen.add(key)
                rects.append((pr.x0, pr.y0, pr.x1, pr.y1))
    return rects


def _extract_pdf_lines(page: fitz.Page) -> list[tuple[float, float, float, float]]:
    """从 PDF 绘图指令中提取水平线段（下划线、答题横线、田字格线）。"""
    h_lines = []
    for path in page.get_drawings():
        # 填充矩形忽略（色块、底纹）
        if path.get("fill") and not path.get("stroke"):
            continue
        color = path.get("color")
        # 红色/彩色线忽略（教师批改标记）
        if color and len(color) == 3:
            r, g, b = color
            if r > 0.7 and g < 0.5 and b < 0.5:
                continue
        width = path.get("width", 0.5)
        for item in path.get("items", []):
            if item[0] == "l":  # line
                p1, p2 = item[1], item[2]
                x1, y1, x2, y2 = p1.x, p1.y, p2.x, p2.y
                if abs(y1 - y2) <= 1.0 and abs(x2 - x1) >= 5:  # 近似水平
                    h_lines.append((min(x1, x2), (y1 + y2)/2, max(x1, x2), (y1 + y2)/2))
            elif item[0] == "re":  # rectangle
                rect = item[1]
                x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                # 矩形的上下边也可能是横线（田字格外框）
                if abs(y1 - y0) < 2:  # 扁矩形=粗线
                    h_lines.append((x0, (y0+y1)/2, x1, (y0+y1)/2))
    return _group_horizontal_lines(h_lines)


def _extract_text_words(page: fitz.Page) -> list[dict]:
    """提取页面文字词（带bbox）。"""
    words = []
    for w in page.get_text("words"):
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        words.append({
            "text": text,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "cx": (x0 + x1) / 2,
            "cy": (y0 + y1) / 2,
        })
    return words


def _extract_char_spans(page: fitz.Page) -> list[dict]:
    """提取字符级位置（用于精确括号定位）。

    用 get_text("rawdict") 获取每个字符的独立 bbox。
    返回 [{x0, y0, x1, y1, c}]（PDF pt）。
    """
    char_spans = []
    raw = page.get_text("rawdict")
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    bbox = ch.get("bbox", (0, 0, 0, 0))
                    origin = ch.get("origin", (0, 0))
                    c = ch.get("c", "")
                    char_spans.append({
                        "c": c,
                        "x0": bbox[0],
                        "y0": bbox[1],
                        "x1": bbox[2],
                        "y1": bbox[3],
                        "origin_x": origin[0],
                        "origin_y": origin[1],
                    })
    # 按位置排序
    char_spans.sort(key=lambda c: (round(c["y0"] / 5), c["x0"]))
    return char_spans


def _find_bracket_blanks(page: fitz.Page, words: list[dict],
                         lines: list[tuple]) -> list[Blank]:
    """识别括号空位：（  ）、【 】、[ ]、( ) 中间为空白。

    策略：用字符级位置精确找每个括号的左右边界，然后按顺序配对。
    关键：括号必须按出现顺序一一配对（第一个左括号配第一个右括号），
    而不是贪心找最近的右括号。
    
    过滤规则：
    - 过滤选择题格式：()、（）内含 A/B/C/D 或 单字母A/B/C/D
    - 过滤标题括号（如页眉页脚、注释）
    - 过滤太宽的括号（可能是多行答题区，应识别为 block）
    """
    blanks = []
    left_chars = {"（", "【", "[", "(", "「"}
    right_chars = {"）", "】", "]", ")", "」"}
    char_spans = _extract_char_spans(page)

    # 按行聚类（同一行：y0 差 < 8pt）
    rows: dict[int, list[dict]] = {}
    for ch in char_spans:
        row_key = round(ch["y0"] / 8)
        rows.setdefault(row_key, []).append(ch)

    for row_key in sorted(rows.keys()):
        row = sorted(rows[row_key], key=lambda c: c["x0"])
        lefts = [c for c in row if c["c"] in left_chars]
        rights = [c for c in row if c["c"] in right_chars]

        # 配对策略：最近距离优先
        # 对每个左括号，找最近且在右侧的右括号
        used_rights: set[int] = set()
        for lc in lefts:
            best = None
            best_dist = float("inf")
            for ri, rc in enumerate(rights):
                if ri in used_rights:
                    continue
                if rc["x0"] <= lc["x1"]:
                    continue  # 右括号在左括号左侧
                dist = rc["x0"] - lc["x1"]
                if dist < best_dist:
                    best_dist = dist
                    best = ri
                    best_rc = rc
            if best is None:
                continue
            x1 = lc["x1"]
            x2 = best_rc["x0"]
            gap = x2 - x1
            # 过滤：括号间距 < 8pt（太近，是嵌套括号或引号）
            if gap < 8:
                continue
            # 过滤：括号太宽（> 60pt）可能是大题区，不是简单括号空位
            if gap > 60:
                continue
            y_top = min(lc["y0"], best_rc["y0"]) - 2
            y_bot = max(lc["y1"], best_rc["y1"]) + 2

            # 【核心增强】提取括号前最近的【加点字】——
            # 关键：不能是「①」「②」这类圈号，也不能是「（」括号本身
            # 加点字通常是汉字，距离括号 1-3 字内
            left_chars_set = {"（", "【", "[", "(", "「"}
            right_chars_set = {"）", "】", "]", ")", "」"}
            dot_char = ""
            # 找左括号前最近的【有效字符】（是汉字/字，距括号 1-2 个字符宽）
            candidates = []
            for ch in row:
                if ch["x1"] <= lc["x1"] + 1:
                    c = ch.get("c", "").strip()
                    if c and c not in left_chars_set and c not in right_chars_set:
                        if c.isdigit():
                            continue
                        if c in {"，", "。", "；", "：", "、", "?", "!", "?", "!", "·", "．", "．"}:
                            continue
                        candidates.append((ch["x0"], c))
            # 优先取最近汉字
            cjk_candidates = [(x, c) for x, c in candidates
                              if '\u4e00' <= c <= '\u9fff']
            if cjk_candidates:
                cjk_candidates.sort(reverse=True)
                dot_char = cjk_candidates[0][1]
            elif candidates:
                candidates.sort(reverse=True)
                dot_char = candidates[0][1]

            ctx = _make_context(row, x1, x2, dot_char)
            # 过滤：明显非空位标题（如页眉页脚括号）
            if "二模" in ctx or "注释" in ctx or ctx.startswith("（") or len(ctx) < 5:
                used_rights.add(best)
                continue
            # 过滤：（X 分）大题号占位 — 找括号前是数字+圆点（题号）、括号内是"X 分"格式
            # 例：14.解释下列加点词在句中的意思。（2 分）→ 这里的括号是分值占位
            if re.search(r'[（(]\s*\d+\s*分\s*[）)]', ctx):
                used_rights.add(best)
                continue
            # 过滤：题号位置（括号前是题号格式 "N." 或 "（N）"）
            if re.search(r'^\s*\d+\.', ctx) or re.search(r'[）)]\s*[（(]', ctx):
                used_rights.add(best)
                continue
            # 过滤：选择题格式检测（括号内含 A/B/C/D 或单独字母）
            # 提取括号内的文字
            inner_chars = [c["c"] for c in row if lc["x1"] < c["x0"] < best_rc["x0"]]
            inner_text = "".join(inner_chars).strip()
            # 如果是选择题格式（单个字母 A/B/C/D 或 A. B. C. D. 格式），跳过
            if inner_text.upper() in ("A", "B", "C", "D", "A.", "B.", "C.", "D."):
                used_rights.add(best)
                continue
            # 【关键修复 2】判断括号是否为大题答题括号
            # 大题区括号特征：context 含题号或括号超宽
            is_answer_block = _is_answer_block_area(ctx, x2 - x1, [])
            blank_type = "block" if is_answer_block else "bracket"
            blanks.append(Blank(
                id=0, type=blank_type,
                x=x1, y=y_bot,
                w=x2 - x1,
                h=max(y_bot - y_top, 8.0),
                context=ctx,
            ))
            used_rights.add(best)
    return blanks


def _find_underline_blanks(page: fitz.Page, words: list[dict],
                           lines: list[tuple]) -> list[Blank]:
    """识别下划线空位：水平横线，找出线上方无文字覆盖的空白段（即要填空的地方）。

    逻辑：对每条横线，找到其上方紧邻的一行文字（y在[ly-20, ly+2]范围内且x重叠），
    这些文字的bbox会覆盖线的一部分；未被覆盖的连续段就是空位。
    若一条线完全没有上方文字（如页眉姓名栏、简答题独立横线），整条线都是空位。

    过滤：
    - 横线宽度 > 页面60% 的视为页面装饰横线（页眉/页脚/章节分隔），跳过

    【关键修复 4】大题答题横线特殊处理：
    - 整条横线被题目文字覆盖时（如17-19题的横线），
      应识别为整条 block 答题区，而不是空白段
    """
    blanks = []
    page_w = page.rect.width
    # 排除"超宽装饰线"（页眉页脚），但**不能**把大题答题横线排除掉
    # 大题答题横线宽 447.8pt，比 357 还宽。
    # 改用绝对阈值：宽于 500pt 才视为装饰线
    max_answer_w = 500
    for lx1, ly, lx2, _ in lines:
        lw = lx2 - lx1
        if lw < 10:
            continue
        # 全宽装饰线跳过
        if lw >= max_answer_w:
            continue

        # 【关键修复 4】宽度>200pt 的大题答题横线：直接识别为整条 block
        # 这是关键修复！17-19题答题横线宽 447.8pt，被原"找空白段"逻辑
        # 完全覆盖（因为整条横线紧贴题目下方），导致识别失败。
        # 现在直接跳过覆盖检测，整条识别为 block。
        if lw > 200:
            # 找最近的题号作为 context
            ctx = _find_question_header(words, ly)
            if not ctx:
                ctx = _get_context(words, lx1, ly - 25, lx2, ly)
            # 高度 = 到下一条横线的距离
            below_lines = [l for l in lines if l[1] > ly + 4 and l[0] < lx2 and l[2] > lx1]
            if below_lines:
                h = below_lines[0][1] - ly - 2
            else:
                h = 22
            blanks.append(Blank(
                id=0, type="block",
                x=lx1, y=ly, w=lw, h=h,
                context=ctx,
            ))
            continue

        # 找这条线上方/附近的文字（在横线书写区域附近）
        nearby_words = [
            w for w in words
            if w["y1"] < ly + 3 and w["y0"] > ly - 25
            and w["x1"] > lx1 - 5 and w["x0"] < lx2 + 5
        ]
        # 计算被文字覆盖的区间
        covered: list[tuple[float, float]] = []
        for w in nearby_words:
            # 文字下方投影到线上：水平方向重叠
            cx0 = max(lx1, w["x0"])
            cx1 = min(lx2, w["x1"])
            if cx1 > cx0:
                covered.append((cx0, cx1))
        # 合并覆盖区间
        covered.sort()
        merged_cov: list[tuple[float, float]] = []
        for s, e in covered:
            if merged_cov and s <= merged_cov[-1][1] + 3:
                merged_cov[-1] = (merged_cov[-1][0], max(merged_cov[-1][1], e))
            else:
                merged_cov.append((s, e))
        # 反推出空白段（空位）
        blank_segments = []
        cursor = lx1
        for s, e in merged_cov:
            if s - cursor > 8:  # 空白段至少8pt
                blank_segments.append((cursor, s))
            cursor = e
        if lx2 - cursor > 8:
            blank_segments.append((cursor, lx2))

        # 计算可写高度：到下一条横线的距离 或 到上方文字的距离（默认18pt）
        below_lines = [l for l in lines if l[1] > ly + 4 and l[0] < lx2 and l[2] > lx1]
        next_line_y = min((l[1] for l in below_lines), default=ly + 20)
        h_default = min(next_line_y - ly - 2, 22)
        # 找上方最近文字的y_top，计算该空位的h
        for sx1, sx2 in blank_segments:
            # 该段上方的文字
            above = [w for w in nearby_words
                     if w["x1"] > sx1 - 2 and w["x0"] < sx2 + 2]
            if above:
                y_top_above = min(w["y0"] for w in above) - 1
                h = ly - y_top_above
                if h < 8:
                    h = h_default
            else:
                h = h_default
            ctx = _get_context(words, sx1, ly - h, sx2, ly)
            # 【关键修复 1】判断是否为大题答题区：
            # 特征：上下文含"分）"题号 或 宽度超大
            is_answer_block = _is_answer_block_area(ctx, sx2 - sx1, nearby_words)
            blank_type = "block" if is_answer_block else "underline"
            blanks.append(Blank(
                id=0, type=blank_type,
                x=sx1, y=ly, w=sx2 - sx1, h=h,
                context=ctx,
            ))
    return blanks


def _find_question_header(words: list, y: float, below: bool = False) -> str:
    """找指定y位置上方（或下方）最近的题号文字。

    below=False: 找 y 上方最近题号（"14."、"15." 等）
    below=True: 找 y 下方最近题号
    """
    import re
    question_pattern = re.compile(r'^\s*(\d{1,2})\.')

    candidates = []
    for w in words:
        text = w["text"].strip()
        if question_pattern.match(text):
            if below and w["cy"] > y:
                candidates.append((w["cy"] - y, text))
            elif not below and w["cy"] < y:
                candidates.append((y - w["cy"], text))

    if candidates:
        candidates.sort()
        return candidates[0][1]
    return ""


def _is_answer_block_area(ctx: str, width: float, nearby_words: list) -> bool:
    """判断下划线空位是否实际是大题答题区。

    判定条件（满足任一即为大题区）：
    1. context 中包含 "分）" "分)" 等分值标记（典型大题题号）
    2. 宽度 > 200pt（典型大题答题区宽度）
    3. 上方文字明显是题号（"14." / "15."等格式）
    """
    import re
    # 条件1: 含分值标记（最可靠的特征）
    if re.search(r'[（(]\s*\d+\s*分\s*[)）]', ctx):
        return True
    # 条件2: 宽度超大
    if width > 200:
        return True
    # 条件3: 上方紧邻文字是题号格式
    if nearby_words:
        nearby_texts = [w["text"] for w in nearby_words[:3]]
        nearby_str = "".join(nearby_texts)
        if re.search(r'^\s*\d+\.', nearby_str):
            return True
    return False


def _find_rect_blocks(rects: list[tuple], words: list[dict]) -> list[Blank]:
    """从一组矩形里识别"连续等高矩形群"，合并成大题答题区(block)。

    特征：上下紧贴、宽度一致、x起点和终点对齐 → 老师用 Word/Excel 画的答题格。
    例：文言文 PDF 第 2/3 页 y=[118.8,134.4][134.4,150.0]... 这种连续矩形。
    """
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: (r[1], r[0]))
    clusters: list[list[tuple]] = []
    for r in rects:
        placed = False
        for c in clusters:
            top = c[0]; bot = c[-1]
            if abs(r[1] - bot[3]) > 1.5:  # 必须竖直紧贴
                continue
            if abs((r[2] - r[0]) - (top[2] - top[0])) > 2:  # 宽度一致
                continue
            if abs(r[0] - top[0]) > 2 or abs(r[2] - top[2]) > 2:  # 左右对齐
                continue
            c.append(r)
            placed = True
            break
        if not placed:
            clusters.append([r])
    out = []
    for c in clusters:
        if len(c) >= 3:
            x1 = c[0][0]; x2 = c[0][2]
            y1 = c[0][1]
            y2 = c[-1][3]
            ctx = ""
            if words:
                above = [w for w in words
                         if w["y1"] < y1 + 2 and w["cx"] > x1 - 30 and w["cx"] < x2 + 30]
                ctx = _get_context(words, x1, y1 - 20, x2, y1)
            out.append(Blank(
                id=0, type="block",
                x=x1, y=y2, w=x2 - x1, h=y2 - y1,
                context=ctx,
            ))
    return out


def _merge_multiline_blocks(blanks: list[Blank], words: list[dict],
                            lines: list[tuple]) -> list[Blank]:
    """把连续多条水平横线（间距相近、左右对齐）合并成一个多行答题区(block)。
    同样把**上下对齐的多个 bracket** 合并为大答题区（如简答题横线用括号标注字数）。

    【关键修复 3】对大题答题横线（≥2条相近且宽度>200pt）合并为单一大block
    """
    # ① underline 合并
    underline_blanks = [b for b in blanks if b.type == "underline"]
    other_blanks = [b for b in blanks if b.type not in ("underline", "bracket")]
    bracket_blanks = [b for b in blanks if b.type == "bracket"]

    merged: list[Blank] = list(other_blanks)

    # 【关键修复 5】对所有 block（包括原本的和大线转的）合并相邻的
    # 按 y 分组：相邻的 block 合并为一个
    block_blanks = [b for b in merged if b.type == "block"]
    non_block_blanks = [b for b in merged if b.type != "block"]

    if block_blanks:
        block_blanks.sort(key=lambda b: (round(b.y / 5), b.x))
        merged_blocks: list[Blank] = []
        for b in block_blanks:
            merged_into_existing = False
            for existing in merged_blocks:
                # 检查是否相邻：y 差距 < 30pt, x 范围有重叠
                if abs(b.y - existing.y) < 30:
                    # 横向重叠：取并集
                    x1 = min(b.x, existing.x)
                    x2 = max(b.x + b.w, existing.x + existing.w)
                    y1 = min(b.y - b.h if b.h else b.y, existing.y - existing.h if existing.h else existing.y)
                    y2 = max(b.y, existing.y)
                    # 上下文取较长的
                    ctx = existing.context if len(existing.context) > len(b.context) else b.context
                    existing.x = x1
                    existing.y = y2
                    existing.w = x2 - x1
                    existing.h = y2 - y1
                    existing.context = ctx
                    merged_into_existing = True
                    break
            if not merged_into_existing:
                merged_blocks.append(b)
        merged = non_block_blanks + merged_blocks

    # --- underline 聚类 ---
    if underline_blanks:
        underline_blanks.sort(key=lambda b: b.y)
        clusters: list[list[Blank]] = []
        for b in underline_blanks:
            placed = False
            for c in clusters:
                top = c[0]
                bot = c[-1]
                if b.y - bot.y > 40:
                    continue
                overlap = min(b.x + b.w, bot.x + bot.w) - max(b.x, bot.x)
                min_w = min(b.w, bot.w)
                if overlap < min_w * 0.4:
                    continue
                c.append(b)
                placed = True
                break
            if not placed:
                clusters.append([b])
        for c in clusters:
            if len(c) >= 2:
                # 【关键修复 3】2条横线 + 宽度>200 也合并为大block
                x1 = min(b.x for b in c)
                x2 = max(b.x + b.w for b in c)
                y1 = c[0].y - (c[1].y - c[0].y) * 0.9 if len(c) >= 2 else c[0].y
                y2 = c[-1].y
                # 判断是否为真正大题（宽度>200 或 context有分值标记）
                ctx = _get_context(words, x1, y1 - 20, x2, y1)
                is_big_block = (x2 - x1) > 200 or _is_answer_block_area(ctx, x2 - x1, [])

                if is_big_block and len(c) >= 2:
                    # 合并为单一大答题区（覆盖整个空白）
                    merged.append(Blank(
                        id=0, type="block",
                        x=x1, y=y2, w=x2 - x1, h=y2 - y1,
                        context=ctx,
                    ))
                elif len(c) >= 3:
                    # 原来逻辑:≥3条合并
                    merged.append(Blank(
                        id=0, type="block",
                        x=x1, y=y2, w=x2 - x1, h=y2 - y1,
                        context=ctx,
                    ))
                else:
                    merged.extend(c)
            else:
                merged.extend(c)

    # --- bracket 聚类（≥3 个上下对齐 → block）---
    if bracket_blanks:
        bracket_blanks.sort(key=lambda b: b.y)
        bclusters: list[list[Blank]] = []
        for b in bracket_blanks:
            placed = False
            for c in bclusters:
                top = c[0]
                bot = c[-1]
                if b.y - bot.y > 40:   # 间距太大，不连续
                    continue
                # x 对齐：左右两端接近（≥40% 重叠）
                overlap = min(b.x + b.w, bot.x + bot.w) - max(b.x, bot.x)
                min_w = min(b.w, bot.w)
                if overlap < min_w * 0.4:
                    continue
                c.append(b)
                placed = True
                break
            if not placed:
                bclusters.append([b])
        for c in bclusters:
            if len(c) >= 3:
                x1 = min(b.x for b in c)
                x2 = max(b.x + b.w for b in c)
                y1 = c[0].y - c[0].h
                y2 = c[-1].y
                ctx = _get_context(words, x1, y1 - 20, x2, y1)
                merged.append(Blank(
                    id=0, type="block",
                    x=x1, y=y2, w=x2 - x1, h=y2 - y1,
                    context=ctx,
                ))
            else:
                merged.extend(c)

    merged.sort(key=lambda b: (round(b.y / 5), b.x))  # 按行排序，再按x
    return merged


def _get_context(words: list[dict], x1: float, y1: float,
                 x2: float, y2: float) -> str:
    """取空位周围的文字作为题目上下文（供VLM理解这是什么题）。

    策略：找 blank 下缘（y2）上方最近的一行文字（cy < y2），
    提取该行所有文字拼接。
    """
    y2_float = float(y2)
    above = [w for w in words if w["cy"] < y2_float - 1]
    if not above:
        return ""
    nearest = max(above, key=lambda w: w["cy"])
    nearest_cy = nearest["cy"]
    row = [w for w in above if abs(w["cy"] - nearest_cy) < 2]
    row.sort(key=lambda w: w["x0"])
    text = "".join(w["text"] for w in row)
    if len(text) >= 2:
        return text[-80:] if len(text) > 80 else text
    return ""


def _make_context(row: list[dict], x1: float, x2: float, dot_char: str = "") -> str:
    """根据一行字符和空位范围，生成 VLM 可见的题目上下文。

    row 是 char_spans（来自 _extract_char_spans），每个元素有 c 字段表示字符。

    策略：把括号内的空位区域用【__】标出，
    并在末尾标注加点字（用→箭头）。
    """
    if not row:
        return ""

    parts = []
    in_blank = False
    for ch in row:
        x0_ch, x1_ch = ch["x0"], ch["x1"]
        t = ch.get("c", ch.get("text", ""))
        entering = x0_ch <= x1 and x1_ch > x1
        leaving = x0_ch >= x2 and x1_ch > x2
        if entering:
            in_blank = True
        if leaving:
            in_blank = False
        if in_blank:
            parts.append("【__】")
        else:
            parts.append(t)

    result = "".join(parts)
    if len(result) > 50:
        result = result[-50:]
    if dot_char:
        result = result + " ->加点:「" + dot_char + "」"
    return result


def detect_pdf_blanks(pdf_path: str | Path, page_index: int = 0) -> PageBlanks:
    """检测单页PDF的所有空位（精确坐标）。"""
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    pw, ph = page.rect.width, page.rect.height
    lines = _extract_pdf_lines(page)
    rects = _extract_pdf_rects(page)
    words = _extract_text_words(page)
    blanks = []
    # 1) 矩形群（答题格）→ block
    blanks.extend(_find_rect_blocks(rects, words))
    # 2) 括号空位（一般PDF少用，先收集）
    blanks.extend(_find_bracket_blanks(page, words, lines))
    # 3) 下划线空位
    blanks.extend(_find_underline_blanks(page, words, lines))
    # 4) 连续横线（≥3 条）合并为大答题区
    blanks = _merge_multiline_blocks(blanks, words, lines)
    # 分配id（从上到下、从左到右）
    blanks.sort(key=lambda b: (round(b.y / 4), b.x))
    for i, b in enumerate(blanks, 1):
        b.id = i
    doc.close()
    pb = PageBlanks(page_index=page_index, blanks=blanks, page_w=pw, page_h=ph)
    for b in pb.blanks:
        b.page = page_index
    return pb


def detect_pdf_all_pages(pdf_path: str | Path,
                           enable_brackets: bool = False) -> list[PageBlanks]:
    """检测 PDF 所有页的空位。

    enable_brackets: 是否启用括号空位检测。默认 False ——
        多数教材/试卷中的 `（）` 是题目中的注释性括号而非空位，
        全开会引入大量误判。后续可按需对单页/单类样本开启。
    """
    doc = fitz.open(pdf_path)
    out = []
    for i in range(len(doc)):
        doc_i = fitz.open(pdf_path)
        page = doc_i[i]
        pw, ph = page.rect.width, page.rect.height
        lines = _extract_pdf_lines(page)
        rects = _extract_pdf_rects(page)
        words = _extract_text_words(page)
        blanks = []
        blanks.extend(_find_rect_blocks(rects, words))
        if enable_brackets:
            blanks.extend(_find_bracket_blanks(page, words, lines))
        blanks.extend(_find_underline_blanks(page, words, lines))
        blanks = _merge_multiline_blocks(blanks, words, lines)
        blanks.sort(key=lambda b: (round(b.y / 4), b.x))
        for j, b in enumerate(blanks, 1):
            b.id = j
        pb = PageBlanks(page_index=i, blanks=blanks, page_w=pw, page_h=ph)
        for b in pb.blanks:
            b.page = i
        out.append(pb)
        doc_i.close()
    doc.close()
    return out


# ========== 图像检测（照片/扫描件） ==========

def detect_image_blanks(image_path: str | Path,
                        a4_w: float = 595.0, a4_h: float = 842.0,
                        zoom: float = 2.0) -> PageBlanks:
    """用OpenCV检测照片/扫描图中的横线空位。
    坐标直接返回PDF pt（zoom=2时 1pt=2px）。
    """
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化（深色线/字）
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # 横向形态学核：检测横线
    kx = max(30, int(w * 0.04))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    # 提取轮廓得到线段
    contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if hh > 4 or ww < 20:  # 太粗/太短的不是横线
            continue
        # 转为pt坐标
        lx1 = x / zoom
        lx2 = (x + ww) / zoom
        ly = (y + hh / 2) / zoom
        lines.append((lx1, ly, lx2, ly))
    # 合并同一行的小线段
    lines = _group_horizontal_lines(lines, tol_y=1.5/zoom, tol_gap=8/zoom)
    # 构建简单的blanks（图像模式下括号等复杂结构交给VLM辅助，但横线是准的）
    blanks = []
    # 按y排序
    lines.sort(key=lambda l: l[1])
    for i, (lx1, ly, lx2, _) in enumerate(lines):
        # 估计高度：到下一条线的距离，或默认20pt
        if i + 1 < len(lines):
            hh = lines[i+1][1] - ly - 2
        else:
            hh = 20
        if hh < 5:
            hh = 14
        blanks.append(Blank(
            id=0, type="underline",
            x=lx1, y=ly, w=lx2 - lx1, h=hh,
            context="",
        ))
    blanks = _merge_multiline_blocks(blanks, [], lines)
    blanks.sort(key=lambda b: (round(b.y / 4), b.x))
    for i, b in enumerate(blanks, 1):
        b.id = i
    pb = PageBlanks(page_index=0, blanks=blanks, page_w=a4_w, page_h=a4_h)
    for b in pb.blanks:
        b.page = 0
    return pb


# ========== 标注编号（给VLM看） ==========

def draw_labels_on_image(page_blanks: PageBlanks, base_image_path: Path,
                          output_path: Path, zoom: float = 2.0) -> Path:
    """在已A4化的PNG（统一zoom=2）上标注空位编号，生成VLM输入图。
    编号画在空位左侧红色圆圈中，空位范围用蓝色框标记。
    """
    img = cv2.imdecode(np.fromfile(str(base_image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {base_image_path}")
    font = cv2.FONT_HERSHEY_SIMPLEX
    for b in page_blanks.blanks:
        lx = int(b.x * zoom)
        ly = int(b.y * zoom)
        # 编号位置：空位左侧 8pt，垂直方向在空位上部
        label_x = max(4, lx - 24)
        label_y = ly - int(b.h * zoom * 0.3)
        # 画红色圆圈底 + 白色数字
        cv2.circle(img, (label_x + 10, label_y - 6), 11, (0, 0, 210), -1)
        text = str(b.id)
        tx = label_x + (7 if len(text) == 1 else 3)
        cv2.putText(img, text, (tx, label_y - 1),
                    font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        # 蓝色框标记空位范围（1px细线）
        x1 = int(b.x * zoom)
        y1 = int((b.y - b.h) * zoom)
        x2 = int((b.x + b.w) * zoom)
        y2 = int(b.y * zoom)
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 100, 0), 1)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("标注图编码失败")
    output_path.write_bytes(buf.tobytes())
    page_blanks.labeled_image_path = output_path
    return output_path
