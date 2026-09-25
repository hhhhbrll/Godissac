"""文本优先空位提取：从 PDF 文本层提取空位坐标 + 重建标记全文。

设计哲学（M9 文本优先架构）：
- 坐标 100% 来自 PDF 数据本身（字符 bbox / 绘图指令），不用 VLM 猜
- 括号空位：字符流中找 `（ 空白 ）`，bbox = 左括号右边 → 右括号左边
- 横线空位：drawings 提取水平线段，插入字符流对应位置
- 大题答题区：连续多条横线合并为 block
- 全文（跨页）重建为带【N】标记的纯文本 → 交给文本 LLM 作答
  （LLM 能看到完整文章+题目+参考译文，对齐率100%）

输出契约（与渲染器对齐）：
blank = {no, page, type, x, y(下缘), w, h, prev(前文), after(后文)}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz

# 标记格式（避免与原文注释编号 ①②③ 冲突）
MARK_FMT = "【{}】"


@dataclass
class Blank:
    no: int              # 全局连续编号（从1开始）
    page: int
    type: str            # bracket / underline / block / inline(句读画线)
    x: float             # 左端
    y: float             # 下缘（括号底部 / 横线 y）
    w: float
    h: float
    prev: str = ""       # 空位前12字（对齐校验用）
    after: str = ""      # 空位后8字
    answer: str = ""
    confidence: float = 1.0
    mark_offset: int = -1  # 【N】标记在标记全文中的字符偏移（分批切片用）
    inline_chars: list = None  # inline专用：该画线行原文字符 [(char, x0, x1, y_top, y_bot)]
    segments: list = None      # block专用：答题区各段 [(page, x, w, [横线y...])]（支持跨页续写）

    def to_spec(self) -> dict:
        spec = {
            "id": self.no,
            "page": self.page,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "width": round(self.w, 1),
            "height": round(self.h, 1),
            "type": self.type,
            "context": (self.prev + "（）" + self.after)[-30:],
            "answer": self.answer,
            "confidence": self.confidence,
        }
        if self.inline_chars:
            spec["inline_chars"] = [
                [c, round(a, 1), round(b, 1), round(t, 1), round(d, 1)]
                for c, a, b, t, d in self.inline_chars
            ]
        if self.segments:
            spec["segments"] = [
                {"page": p, "x": round(x, 1), "width": round(w, 1),
                 "rules": [round(r, 1) for r in rs]}
                for p, x, w, rs in self.segments
            ]
        return spec


@dataclass
class PageData:
    page_index: int
    chars: list = field(default_factory=list)      # [(char, bbox)] 阅读顺序
    brackets: list = field(default_factory=list)   # [(li, ri, bbox)]
    lines: list = field(default_factory=list)      # 横线 [(x1, y, x2, y)]
    insert_at: dict = field(default_factory=dict)  # {stream_idx: blank引用} 标记插入点


# ---------------------------------------------------------------------------
# 基础提取
# ---------------------------------------------------------------------------

def _extract_chars(page: fitz.Page) -> list[tuple[str, tuple]]:
    """按阅读顺序提取 (char, bbox)。rawdict 的块顺序即 PDF 内容流顺序。"""
    chars = []
    raw = page.get_text("rawdict")
    for block in raw["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                for ch in span["chars"]:
                    chars.append((ch["c"], ch["bbox"]))
    return chars


def _extract_lines(page: fitz.Page) -> list[tuple[float, float, float, float]]:
    """从绘图指令提取水平横线 [(x1, y, x2, y)]。

    Word 导出 PDF 的横线是纯填充细矩形（fill=黑、stroke=None），
    不能当色块滤掉——只有"厚矩形/非黑填充"才是装饰色块。
    """
    h_lines = []
    for path in page.get_drawings():
        fill = path.get("fill")
        if fill and not path.get("stroke"):
            is_black = all(v < 0.3 for v in fill)
            if not is_black:
                continue  # 彩色色块忽略
        for item in path.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) <= 1.0 and abs(p2.x - p1.x) >= 10:
                    h_lines.append((min(p1.x, p2.x), (p1.y + p2.y) / 2,
                                    max(p1.x, p2.x), (p1.y + p2.y) / 2))
            elif item[0] == "re":
                r = item[1]
                if abs(r.y1 - r.y0) < 2 and r.x1 - r.x0 >= 10:
                    h_lines.append((r.x0, (r.y0 + r.y1) / 2, r.x1, (r.y0 + r.y1) / 2))
    # 同y聚类合并（多条短线拼成长线）
    h_lines.sort(key=lambda l: (l[1], l[0]))
    merged: list[list] = []
    for l in h_lines:
        if merged and abs(l[1] - merged[-1][0][1]) <= 2.0 and l[0] - merged[-1][0][2] <= 6:
            m0 = merged[-1]
            m0[0] = (m0[0][0], m0[0][1], max(m0[0][2], l[2]), m0[0][3])
            m0.append(l)
        else:
            merged.append([l])
    return [g[0] for g in merged]


def _find_bracket_blanks(chars: list) -> list[tuple[int, int, tuple]]:
    """找空括号 `（ ... ）`（含半角）。返回 [(左括号idx, 右括号idx, bbox)]。"""
    lefts = {"（", "("}
    rights = {"）", ")"}
    blanks = []
    i = 0
    while i < len(chars):
        if chars[i][0] in lefts:
            j = i + 1
            inner = ""
            while j < len(chars) and chars[j][0] not in rights:
                inner += chars[j][0]
                j += 1
            # 空括号判定：内部只有空白（或极少不可见字符）；
            # 长度放宽到 12（上海卷等宽括号内含 6+ 空格）
            if j < len(chars) and inner.strip() == "" and j - i <= 12:
                # bbox：左括号右边 → 右括号左边
                x1 = chars[i][1][2]
                x2 = chars[j][1][0]
                y1 = min(chars[i][1][1], chars[j][1][1])
                y2 = max(chars[i][1][3], chars[j][1][3])
                if x2 - x1 >= 2:  # 至少能塞一个字
                    blanks.append((i, j, (x1, y1, x2, y2)))
                    i = j + 1
                    continue
        i += 1
    return blanks


# ---------------------------------------------------------------------------
# 横线空位 → 大题区合并 + 文本流插入点
# ---------------------------------------------------------------------------

def _is_text_underline(chars: list, line: tuple) -> list:
    """判定横线是否为"文字下方的画线"（句读题画线部分）。

    返回压线字符列表 [(idx, char, bbox)]；空列表 = 非画线（是填空横线）。
    特征：线上方文字底部贴近横线且水平覆盖过半。
    """
    lx1, ly, lx2, _ = line
    on_line = []
    for idx, (c, b) in enumerate(chars):
        if b[3] < ly - 3 or b[3] > ly + 2.5:  # 文字底部须贴近横线
            continue
        if b[2] < lx1 - 2 or b[0] > lx2 + 2:  # 无水平重叠
            continue
        if c.strip():
            on_line.append((idx, c, b))
    if not on_line:
        return []
    segs = sorted((max(lx1, b[0]), min(lx2, b[2])) for _, _, b in on_line)
    covered = 0.0
    cur_end = -1e9
    for s, e in segs:
        if e <= cur_end:
            continue
        covered += e - max(s, cur_end)
        cur_end = e
    return on_line if covered >= (lx2 - lx1) * 0.5 else []


def _find_para_markers(chars: list) -> list[tuple[int, float]]:
    """提取段落标记 ①②③… 的 (段号, y顶)。"""
    circled = "①②③④⑤⑥⑦⑧⑨⑩"
    markers = []
    for c, b in chars:
        if c in circled:
            markers.append((circled.index(c) + 1, b[1]))
    return markers


def _find_line_questions(chars: list) -> dict[int, str]:
    """提取引用画线句的题目：{段号: 'judu'(句读,标在原文) | 'fanyi'(翻译,写答题线)}。

    依据题目文字如"第②段画线部分…用/标识"（句读）/"第③段画浪线句子翻译成…"（翻译）。
    """
    import re
    text = "".join(c for c, _ in chars)
    circled = "①②③④⑤⑥⑦⑧⑨⑩"
    result: dict[int, str] = {}
    for m in re.finditer(r"第([①②③④⑤⑥⑦⑧⑨⑩\d一二三四五]+)段.{0,6}?(波浪线|画浪线|浪线|画线|划线)", text):
        raw = m.group(1)
        if raw in circled:
            para = circled.index(raw) + 1
        elif raw.isdigit():
            para = int(raw)
        else:
            cn = "一二三四五".find(raw)
            para = cn + 1 if cn >= 0 else None
        if para is None:
            continue
        # 窄窗口判定题型：翻译关键词优先（避免被相邻句读题的"标识"污染）
        qtext = text[m.end(): m.end() + 30]
        if any(k in qtext for k in ("翻译", "译成", "译文")):
            result[para] = "fanyi"
        elif any(k in qtext for k in ("句读", "断句", "标识", "用/")):
            result[para] = "judu"
    return result


def _para_of(page: int, y_top: float,
             global_markers: list[tuple[int, int, float]]) -> int | None:
    """(page, y) 所属段号：全局（跨页）其上方最近的段落标记。"""
    best = None  # (page, y, no)
    for mp, no, my in global_markers:
        if mp < page or (mp == page and my <= y_top + 2):
            if best is None or (mp, my) > (best[0], best[1]):
                best = (mp, my, no)
    return best[2] if best else None


def _find_judu_restatements(pd: PageData) -> list[dict]:
    """句读题题干下方的"重述句" → inline 空位。

    句读题（"…有四处需加句读，请用'/'标识出来。（X分）"）通常在题干后
    重述待断句的原文——学生的"/"应标在这句重述句上（题干下方），
    而非文章原文的画线处（用户明确要求）。

    返回 [{bbox, insert_idx, type:'inline', inline_chars, sort, restatement:True}]
    """
    import re
    text = "".join(c for c, _ in pd.chars)
    results = []
    for m in re.finditer(r"句读|断句", text):
        # 题干结尾的分值"（X 分）"之后即重述句
        tail = text[m.end(): m.end() + 30]
        mm = re.search(r"（\s*\d+\s*分\s*）", tail)
        if not mm:
            continue
        start = m.end() + mm.end()
        # 重述句范围：到下一题号（如"17."）或第一个句号（含）为止
        seg = text[start: start + 90]
        endm = re.search(r"\d{1,2}\s*\.", seg)
        sent = seg[: endm.start()] if endm else seg
        pm = re.search(r"。", sent)
        if pm:
            sent = sent[: pm.end()]
        if len(sent.strip()) < 8:  # 无重述句（该卷要求标在原文）
            continue
        chars = [(i, c, b) for i in range(start, start + len(sent))
                 for c, b in [pd.chars[i]] if c.strip()]
        if len(chars) < 8:
            continue
        ordered = sorted(chars, key=lambda t: (t[2][1], t[2][0]))
        lx1 = min(t[2][0] for t in ordered)
        lx2 = max(t[2][2] for t in ordered)
        y_top = min(t[2][1] for t in ordered)
        y_bot = max(t[2][3] for t in ordered)
        results.append({
            "bbox": (lx1, y_top, lx2, y_bot),
            "insert_idx": ordered[0][0],
            "type": "inline",
            "inline_chars": [(c, b[0], b[2], b[1], b[3]) for _, c, b in ordered],
            "sort": (ordered[0][2][1], ordered[0][2][0]),
            "restatement": True,
        })
    return results


def _find_line_blanks(pd: PageData, base_size: float = 13.0,
                      global_markers: list | None = None,
                      global_questions: dict | None = None) -> list[dict]:
    """横线 → 空位。

    - 句读题：答案("/")标在题干下方的重述句上（_find_judu_restatements）；
      文章原文的画线不再作答（用户要求：答题位置在题干下方，非文中）。
      无重述句的卷子才回退到原文画线。
    - 翻译题波浪线 → 跳过（答案写下方答题横线）
    - 空白答题横线：聚簇（簇内不隔题干文字）→ block 空位，记录每条横线 y（字贴线写）
    返回 [{bbox, insert_idx, type, inline_chars?/rules?, orphan?}]
    """
    lines = sorted(pd.lines, key=lambda l: (l[1], l[0])) if pd.lines else []

    para_markers = global_markers or [
        (pd.page_index, no, y) for no, y in _find_para_markers(pd.chars)
    ]
    page_q = global_questions or {}

    result = []
    fill_lines = []
    # 句读重述句：题干下方重述的待断句（答案"/"标在这里）
    restatements = _find_judu_restatements(pd)
    result.extend(restatements)
    # 压线分组：句读句可能跨多行，同段行距≤25pt 合并为一个 inline 空位
    # （仅当本页无重述句时才回退到原文画线；有重述句则原文画线不作答）
    judu_groups: list = []  # [(para, [lines], [on_line])]
    for l in lines:
        on_line = _is_text_underline(pd.chars, l)
        if not on_line:
            fill_lines.append(l)
            continue
        para = _para_of(pd.page_index, l[1], para_markers)
        if page_q.get(para) == "fanyi":
            continue  # 翻译题波浪线：答案写答题横线，原文不标
        if restatements:
            continue  # 句读答案在题干下方重述句，原文画线不作答
        # 原文画线只有题干明说"用/断句"（judu）才作答——翻译画线、
        # 【甲乙丙丁】标示线、小说引用线等一律不建空位（学生写在答题纸上）
        if page_q.get(para) != "judu":
            continue
        if (judu_groups and judu_groups[-1][0] == para
                and 0 < l[1] - judu_groups[-1][1][-1][1] <= 25):
            judu_groups[-1][1].append(l)
            judu_groups[-1][2].extend(on_line)
        else:
            judu_groups.append((para, [l], list(on_line)))

    for _para, glines, gchars in judu_groups:
        # 剔除括号字符（括号另有空位）与不可见字符；
        # 边缘标点（如前导"，""""）会让标记插到标点前，被 LLM 误读成
        # "词语后的括号空位"（如"琴【22】，"）——必须让标记紧贴句首字
        keep = [(i, c, b) for i, c, b in gchars
                if c.strip() and c not in "（）()【】"]
        if not keep:
            continue
        lx1 = min(l[0] for l in glines)
        lx2 = max(l[2] for l in glines)
        ordered = sorted(keep, key=lambda t: (t[2][1], t[2][0]))
        y_top = min(b[1] for _, _, b in keep)
        first_idx = ordered[0][0]  # 阅读顺序第一个字（标记紧贴句首字）
        result.append({
            "bbox": (lx1, y_top, lx2, glines[-1][1]),
            "insert_idx": first_idx,
            "type": "inline",
            "inline_chars": [(c, b[0], b[2], b[1], b[3]) for _, c, b in ordered],
            # 排序键用首条画线的y（≈文字行基线）而非字符顶：避免排到
            # 同行前方括号之前导致标记编号乱序（LLM会因此答错题型）
            "sort": (glines[0][1], ordered[0][1]),
        })

    # 空白答题横线聚簇：y间距≤40、x重叠≥40%、且两线之间不隔题干文字
    def _text_between(y_top: float, y_bot: float, x1: float, x2: float) -> bool:
        for c, b in pd.chars:
            if (c.strip() and b[3] > y_top + 2 and b[1] < y_bot - 2
                    and b[2] > x1 + 2 and b[0] < x2 - 2):
                return True
        return False

    clusters: list[list] = []
    for l in fill_lines:
        if clusters:
            prev = clusters[-1][-1]
            overlap = min(l[2], prev[2]) - max(l[0], prev[0])
            if (0 < prev[1] < l[1] <= prev[1] + 40
                    and overlap >= 0.4 * min(l[2] - l[0], prev[2] - prev[0])
                    and not _text_between(prev[1], l[1],
                                          min(l[0], prev[0]), max(l[2], prev[2]))):
                clusters[-1].append(l)
                continue
        clusters.append([l])

    for cl in clusters:
        x1 = min(l[0] for l in cl)
        x2 = max(l[2] for l in cl)
        rule_ys = [l[1] for l in cl]
        gap = rule_ys[1] - rule_ys[0] if len(rule_ys) > 1 else 15.5
        y1 = rule_ys[0] - gap          # 首线上方留一行
        y2 = rule_ys[-1]
        insert_idx = _find_insert_idx(pd.chars, cl[0])
        orphan = insert_idx is None and rule_ys[0] < 200
        if orphan:
            # 页顶跨页答题区：标记插到本页阅读顺序第一个字符前
            vis = [(b[1], b[0], i) for i, (c, b) in enumerate(pd.chars) if c.strip()]
            insert_idx = min(vis)[2] if vis else 0
        if insert_idx is not None:
            result.append({
                "bbox": (x1, y1, x2, y2),
                "insert_idx": insert_idx,
                "type": "block",
                "rules": rule_ys,
                "orphan": orphan,
            })
    return result


def _find_insert_idx(chars: list, line: tuple) -> int | None:
    """横线标记在字符流的插入位置。

    判定顺序：
    ① 行内填空线（默写题"映阶碧草自春色，___"）：横线与文字同基线，
       左邻字符距横线左端极近（≤15pt）→ 标记插左邻字符后
    ② 大题答题区：线上方 0~45pt 内最近的文字行，取该行行尾字符
       （答题线距题干常有 15pt 留白，容差 45pt）
    """
    lx1, ly, lx2 = line[0], line[1], line[2]
    # ① 行内横线：同一基线（y 中心差 ≤ 6）且紧贴横线左端的字符
    same_row = [(chars[i][1][2], i) for i, (c, b) in enumerate(chars)
                if c.strip() and b[2] <= lx1 + 3 and lx1 - b[2] <= 15
                and abs((b[1] + b[3]) / 2 - ly) <= 6]
    if same_row:
        return max(same_row)[1]

    # ② 答题区：上方文字行行尾
    near = [(b[3], idx) for idx, (c, b) in enumerate(chars)
            if c.strip() and 0 <= ly - b[3] <= 45]
    if near:
        # 最近的文字行（字底最贴近横线）
        nearest_bottom = max(t[0] for t in near)
        row = [idx for bottom, idx in near if nearest_bottom - bottom <= 3.5]
        # 行内最右字符（优先横线水平范围内的）
        in_span = [i for i in row if chars[i][1][2] <= lx2 + 3]
        if in_span:
            return max(in_span, key=lambda i: chars[i][1][2])
        return max(row, key=lambda i: chars[i][1][2])
    return None


def _gap_to_prev_text(chars: list, line: tuple, default: float = 20.0) -> float:
    """横线上方最近文字的底部到横线的距离（可写高度）。"""
    lx1, ly, lx2, _ = line
    tops = [b[1] for c, b in chars
            if b[3] < ly + 2 and b[1] > ly - 40
            and b[2] > lx1 - 5 and b[0] < lx2 + 5]
    if tops:
        h = ly - min(tops)
        if 8 <= h <= 36:
            return h
    return default


# ---------------------------------------------------------------------------
# 主入口：提取全部空位 + 标记全文
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: str | Path) -> tuple[str, list[Blank], int]:
    """PDF → (标记全文, 空位列表[全局编号], 页数)。

    标记全文：所有页拼接，空位处嵌【N】。
    """
    doc = fitz.open(pdf_path)
    pages_data: list[PageData] = []

    for pi, page in enumerate(doc):
        pd = PageData(page_index=pi)
        pd.chars = _extract_chars(page)
        pd.brackets = _find_bracket_blanks(pd.chars)
        pd.lines = _extract_lines(page)
        pages_data.append(pd)

    # 全局（跨页）段落标记 + 题目归属，供画线句语义判定
    global_markers = [
        (pd.page_index, no, y)
        for pd in pages_data for no, y in _find_para_markers(pd.chars)
    ]
    # 题目归属按页隔离（多篇文章段号会冲突：第1篇③段翻译 vs 第2篇③段句读）；
    # 无题干的续页沿用最近前页的归属
    q_by_page: dict[int, dict] = {}
    last_q: dict = {}
    for pd in pages_data:
        pq = _find_line_questions(pd.chars)
        if pq:
            last_q = pq
        q_by_page[pd.page_index] = last_q

    # 收集所有空位候选：括号（直接有流内位置）+ 横线（有插入点）
    candidates = []

    for pd in pages_data:
        for (li, ri, bbox) in pd.brackets:
            candidates.append({
                "page": pd.page_index,
                "sort": (bbox[1], bbox[0]),
                "kind": "bracket",
                "li": li, "ri": ri, "bbox": bbox,
                "pd": pd,
            })
        for lb in _find_line_blanks(pd, global_markers=global_markers,
                                     global_questions=q_by_page.get(pd.page_index)):
            candidates.append({
                "page": pd.page_index,
                "sort": lb.get("sort") or (lb["bbox"][1], lb["bbox"][0]),
                "kind": lb["type"],
                "insert_idx": lb["insert_idx"],
                "bbox": lb["bbox"],
                "pd": pd,
                "inline_chars": lb.get("inline_chars"),
                "rules": lb.get("rules"),
                "orphan": lb.get("orphan", False),
            })

    # 跨页合并：页顶孤儿答题区（如第2页开头的续写横线）并入上一页最后一个 block
    merged_candidates: list = []
    for c in candidates:
        if c.get("orphan") and c["page"] > 0:
            prev_block = None
            for pc in merged_candidates:
                if pc["page"] == c["page"] - 1 and pc["kind"] == "block":
                    prev_block = pc
            if prev_block is not None:
                prev_block.setdefault("extra_segs", []).append(
                    (c["page"], c["bbox"][0], c["bbox"][2] - c["bbox"][0], c["rules"]))
                continue  # 不单独成空位（标记仍留在上一页的插入点）
        merged_candidates.append(c)
    candidates = merged_candidates

    # 全局编号（按页→y→x 阅读顺序）
    candidates.sort(key=lambda c: (c["page"], round(c["sort"][0] / 8), c["sort"][0], c["sort"][1]))

    blanks: list[Blank] = []
    # 标记插入映射：{page_idx: {stream_idx: [no, ...]}}
    insert_map: dict[int, dict[int, list]] = {}
    # 括号覆盖字符集合：{page_idx: set(stream_idx)}
    skip_map: dict[int, set] = {}

    for no, c in enumerate(candidates, 1):
        pd = c["pd"]
        bbox = c["bbox"]
        segments = None
        if c["kind"] == "block":
            segs = [(pd.page_index, bbox[0], bbox[2] - bbox[0], c.get("rules") or [])]
            for extra in c.get("extra_segs", []):
                segs.append(extra)
            segments = segs
        b = Blank(
            no=no, page=c["page"], type=c["kind"],
            x=bbox[0], y=bbox[3], w=bbox[2] - bbox[0], h=bbox[3] - bbox[1],
            inline_chars=c.get("inline_chars"),
            segments=segments,
        )
        blanks.append(b)
        if c["kind"] == "bracket":
            # 括号：左括号处放标记（左括号本身删除），右括号与内部空白删除
            li, ri = c["li"], c["ri"]
            insert_map.setdefault(pd.page_index, {}).setdefault(li, []).append(no)
            skip = skip_map.setdefault(pd.page_index, set())
            skip.update(range(li, ri + 1))
        else:
            # 横线/大题区/句读画线：在插入点放标记（原文保留）
            insert_map.setdefault(pd.page_index, {}).setdefault(c["insert_idx"], []).append(no)

    # 重建标记全文（跨页拼接，页间加分隔符）
    # 句读空位用专属标记【N=断句】：与括号空位【N】区分，避免 LLM 把
    # 句首标记误读成"词语后的括号空位"而答成词语释义
    # 括号空位标记【N·K字】：K=括号能容纳的字数（宽/正文字号），
    # 让 LLM 按实际宽度作答，避免窄括号塞长答案导致渲染字号被压缩
    type_by_no = {b.no: b.type for b in blanks}
    cap_by_no = {b.no: max(1, round(b.w / 10.4)) for b in blanks if b.type == "bracket"}
    offset_by_no: dict[int, int] = {}
    parts = []
    base = 0  # 当前页 parts 起始的全局偏移
    for pd in pages_data:
        skip = skip_map.get(pd.page_index, set())
        imap = insert_map.get(pd.page_index, {})
        buf = []
        cur = 0  # 页内偏移
        for idx, (ch, bbox) in enumerate(pd.chars):
            if idx in imap:
                for no in imap[idx]:
                    if type_by_no.get(no) == "inline":
                        mark = f"【{no}=断句】"
                    elif type_by_no.get(no) == "bracket":
                        mark = f"【{no}·{cap_by_no.get(no, 1)}字】"
                    else:
                        mark = MARK_FMT.format(no)
                    offset_by_no[no] = base + cur
                    cur += len(mark)
                    buf.append(mark)
            if idx in skip:
                continue
            buf.append(ch)
            cur += 1
        parts.append("".join(buf))
        base += len(parts[-1]) + 1  # 页间 \n
    marked_text = "\n".join(parts)

    # 回填标记偏移（供篇章分批切片）
    for b in blanks:
        b.mark_offset = offset_by_no.get(b.no, -1)

    # 回填 prev/after（供人工核对与对齐校验）
    doc2 = fitz.open(pdf_path)
    for b in blanks:
        pd = pages_data[b.page]
        li = None
        for idx, nos in insert_map.get(b.page, {}).items():
            if b.no in nos:
                li = idx
                break
        if li is not None:
            b.prev = "".join(c for c, _ in pd.chars[max(0, li - 12):li])
            b.after = "".join(c for c, _ in pd.chars[li + 1:li + 9])
    doc.close()
    doc2.close()

    return marked_text, blanks, len(pages_data)


def has_text_layer(pdf_path: str | Path) -> bool:
    """PDF 是否有可用文本层（每页平均字符数 > 50 视为有）。"""
    doc = fitz.open(pdf_path)
    try:
        total = 0
        for page in doc:
            total += len(page.get_text("text").strip())
        return total > 50 * len(doc)
    finally:
        doc.close()
