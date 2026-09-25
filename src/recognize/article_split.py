"""篇章切分：大批量作业按"篇章"分组喂给 LLM。

背景：整册作业（10+页、多篇文言文）一次性发送：
- 输出分批时每批重发全文，成本翻倍；长上下文注意力稀释，答案质量下降

切分原则（用户需求：不能按页硬切）：
- 篇章是原子单位（原文+译文+题目一体，上下文完整）
- 篇章边界在页中间时，按字符偏移切——边界页上半（上篇题目）归前批、
  下半（下篇原文）归后批，正是"第七页前后传两次"的精准实现

边界判据（阅读顺序扫描空位序列）：
- 篇尾信号：空位是 block/inline 大题（或 prev 含"分）"的题目型括号）
- 篇头信号：其后连续 ≥3 个原文型 bracket（文言正文填空密集），
  且其 prev 不含"分）"（非题号行）
两信号相邻 → 篇章边界。

批组装：篇章按顺序累积成批（≤MAX_CHARS 字 或 ≤MAX_BLANKS 空位），
超预算开新批；单篇超预算独立成批（输入不切篇，输出侧仍按 45 空位
分批见 text_llm.answer_homework）。

用法：
    from src.recognize.article_split import split_by_article
    batches = split_by_article(marked_text, blanks)  # [(text, [no...]), ...]
"""

from __future__ import annotations

MAX_CHARS = 6000     # 每批字符预算（≈1万 token 输入）
MAX_BLANKS = 200      # 每批空位数软预算（输出侧另有45/调用限制）
BRACKET_RUN = 6       # 篇头信号：连续原文 bracket 数（真篇头 run≥32，
                      # 题目区引用原文的括号 run≤4，如"（2）而皇后…"小题群）
TAIL_GAP = 40         # 批起点越过上批末空位标记的字符数（标记+题干尾巴）


def _is_question_text(prev: str) -> bool:
    """prev 是题号/分值/小题号行片段（题目语言）而非文言正文。"""
    return ("分）" in prev or "分)" in prev or "（）（" in prev
            or prev.startswith("）（") or prev.startswith("()("))


def detect_articles(blanks) -> list[tuple[int, int, int]]:
    """空位序列 → [(篇首空位下标, 篇尾空位下标, 篇首文本切点偏移)]。"""
    if not blanks:
        return []
    n = len(blanks)
    arts: list[tuple[int, int, int]] = []
    cur_start = 0
    for i in range(n):
        b = blanks[i]
        boundary = False
        # 篇尾信号：block 大题（inline 句读标记在原文区，不是篇尾信号）
        if b.type == "block" or (
                b.type == "bracket" and _is_question_text(b.prev)):
            # 篇头信号：其后连续 ≥BRACKET_RUN 个原文型 bracket
            run = 0
            j = i + 1
            while j < n and blanks[j].type == "bracket" and not _is_question_text(blanks[j].prev):
                run += 1
                j += 1
            boundary = run >= BRACKET_RUN
        if boundary or i == n - 1:
            head = blanks[cur_start].mark_offset
            arts.append((cur_start, i, head if head >= 0 else 0))
            cur_start = i + 1
    return arts


def split_by_article(marked_text: str, blanks) -> list[tuple[str, list[int]]]:
    """→ [(批文本, 空位编号列表)]。每批文本含完整篇章。"""
    if not blanks:
        return [(marked_text, [])]
    arts = detect_articles(blanks)
    if not arts or any(b.mark_offset < 0 for b in blanks):
        return [(marked_text, [b.no for b in blanks])]

    # 每篇精确文本范围（起点=上篇末空位后TAIL_GAP，终点=下篇篇头空位前）
    n_arts = len(arts)
    art_ranges: list[tuple[int, int]] = []
    for k, (a0, a1, _) in enumerate(arts):
        if k == 0:
            start = 0
        else:
            prev_end = blanks[arts[k - 1][1]].mark_offset
            start = min(max(0, prev_end + TAIL_GAP),
                        blanks[a0].mark_offset - 1)
        if k + 1 < n_arts:
            end = blanks[arts[k + 1][0]].mark_offset
        else:
            end = len(marked_text)
        art_ranges.append((start, max(start + 1, end)))

    # 篇章按预算组装成批（篇原子不切）
    groups: list[list[tuple[int, int, int]]] = []
    cur: list[tuple[int, int, int]] = []
    cur_chars = cur_blanks = 0
    for a, (a_start, a_end) in zip(arts, art_ranges):
        n_blanks = a[1] - a[0] + 1
        a_len = a_end - a_start
        if cur and (cur_chars + a_len > MAX_CHARS or cur_blanks + n_blanks > MAX_BLANKS):
            groups.append(cur)
            cur, cur_chars, cur_blanks = [], 0, 0
        cur.append(a)
        cur_chars += max(a_len, 0)
        cur_blanks += n_blanks
    if cur:
        groups.append(cur)

    out: list[tuple[str, list[int]]] = []
    for gi, grp in enumerate(groups):
        start = art_ranges[arts.index(grp[0])][0]
        if gi + 1 < len(groups):
            nxt_first = groups[gi + 1][0]
            end = art_ranges[arts.index(nxt_first)][0]
        else:
            end = len(marked_text)
        end = max(start + 1, end)
        nos = [blanks[k].no for a in grp for k in range(a[0], a[1] + 1)]
        out.append((marked_text[start:end], nos))
    return out
