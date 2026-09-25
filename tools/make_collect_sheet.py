"""生成手写字体采集模板 PDF（田字格 + 四角定位锚点 + 提示字）。

用法：
    python tools/make_collect_sheet.py [字数 | @清单.json] [free|trace] [追加字串]

两种模式：
- free（默认）：格子右上角印小号灰色提示字，用户在格内自由书写。
  字形结构 100% 出自本人，仅提示"该写什么"。
- trace：格中央印大号浅灰提示字，用户覆盖描摹。字形更统一，
  但结构会向模板字体倾斜（不推荐给追求还原真实笔迹的场景）。

字源：
- 数字（如 300）：取高频字表前 N 字。
- @清单.json：读补录清单（list 或 {"top1000": [...]}），如 output/m4/recollect_list.json。
- 追加字串：第3参数直接给出要点名的字符（如 "升仆"），去重后追加到末尾。

产出（prefix 由字源决定：默认 collect / @json 清单名去 _list）：
    samples/{prefix}_sheet.pdf      打印用模板
    samples/{prefix}_chars.json     格子→汉字的映射清单（分割端读取）
"""

import json
import math
import sys
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fontforge_lib.charlist import load_freq_chars
from src.fontforge_lib.template import (
    A4_H, A4_W, CELL, CELLS_PER_PAGE, COLS, GRID_LEFT, GRID_TOP, MARKER,
    MARKER_INSET, ROWS, marker_centers_pt,
)

FONT = ROOT / "fonts" / "LXGWWenKai-Regular.ttf"

TITLE = "手写字体采集模板"
INSTRUCTIONS = {
    "free": "对照每格右上角的灰色小字，用黑色签字笔在格内自由书写；写大些、居中、不要出格",
    "trace": "请用黑色签字笔，在浅灰提示字上覆盖书写；字要居中、大小尽量一致",
}


def draw_sheet(doc: fitz.Document, chars: list[str], page_no: int, total_pages: int,
               mode: str = "free"):
    page = doc.new_page(width=A4_W, height=A4_H)

    def put(x, y, text, size, color, rotate=0):
        page.insert_text((x, y), text, fontsize=size, fontname="F",
                         fontfile=str(FONT), color=color, rotate=rotate)

    # 页眉
    font = fitz.Font(fontfile=str(FONT))
    tw = font.text_length(TITLE, fontsize=20)
    put((A4_W - tw) / 2, 42, TITLE, 20, (0, 0, 0))
    instruction = INSTRUCTIONS[mode]
    iw = font.text_length(instruction, fontsize=9.5)
    put((A4_W - iw) / 2, 66, instruction, 9.5, (0.25, 0.25, 0.25))

    # 四角定位锚点（纯黑实心方块）
    for cx, cy in marker_centers_pt():
        page.draw_rect(
            fitz.Rect(cx - MARKER / 2, cy - MARKER / 2, cx + MARKER / 2, cy + MARKER / 2),
            color=None, fill=(0, 0, 0),
        )

    # 田字格 + 浅灰提示字
    for r in range(ROWS):
        for c in range(COLS):
            idx = page_no * CELLS_PER_PAGE + r * COLS + c
            if idx >= len(chars):
                continue
            ch = chars[idx]
            x0 = GRID_LEFT + c * CELL
            y0 = GRID_TOP + r * CELL
            rect = fitz.Rect(x0, y0, x0 + CELL, y0 + CELL)
            # 外框（印刷灰必须足够浅，确保与黑色笔迹可靠分离）
            page.draw_rect(rect, color=(0.75, 0.75, 0.75), width=0.5)
            # 虚线十字（田字格中线）
            dash = "[2 2] 0"
            mid_x, mid_y = x0 + CELL / 2, y0 + CELL / 2
            page.draw_line(fitz.Point(x0, mid_y), fitz.Point(x0 + CELL, mid_y),
                           color=(0.82, 0.82, 0.82), width=0.4, dashes=dash)
            page.draw_line(fitz.Point(mid_x, y0), fitz.Point(mid_x, y0 + CELL),
                           color=(0.82, 0.82, 0.82), width=0.4, dashes=dash)
            # 提示字
            if mode == "trace":
                # 大号浅灰字居中（覆盖描摹）
                size = CELL * 0.62
                cw = font.text_length(ch, fontsize=size)
                put(x0 + (CELL - cw) / 2, y0 + CELL / 2 + size * 0.36, ch, size,
                    (0.78, 0.78, 0.78))
            else:
                # 小号灰字在右上角（仅提示写什么，不描摹）
                size = CELL * 0.20
                cw = font.text_length(ch, fontsize=size)
                put(x0 + CELL - cw - 3, y0 + size + 2.5, ch, size,
                    (0.78, 0.78, 0.78))

    # 页脚
    foot = f"第 {page_no + 1} / {total_pages} 页"
    fw = font.text_length(foot, fontsize=9)
    put((A4_W - fw) / 2, A4_H - 20, foot, 9, (0.3, 0.3, 0.3))


def load_chars_from_json(path: Path) -> list[str]:
    """清单JSON：{"top1000": [...], ...} 或 [字符, ...]"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        chars = data.get("top1000")
        if chars is None:
            chars = next((v for v in data.values() if isinstance(v, list)), [])
    else:
        chars = data
    return [c for c in chars if isinstance(c, str) and len(c) == 1]


def main(spec: str = "300", mode: str = "free", extra: str = ""):
    if spec.startswith("@"):
        src = Path(spec[1:])
        chars = load_chars_from_json(src)
        stem = src.stem
        prefix = stem[:-5] if stem.endswith("_list") else stem
    else:
        chars = load_freq_chars(int(spec))
        prefix = "collect"
    for ch in extra:
        if len(ch) == 1 and ch not in chars:
            chars.append(ch)
    out_pdf = ROOT / "samples" / f"{prefix}_sheet.pdf"
    out_json = ROOT / "samples" / f"{prefix}_chars.json"
    total_pages = math.ceil(len(chars) / CELLS_PER_PAGE)
    doc = fitz.open()
    for p in range(total_pages):
        draw_sheet(doc, chars, p, total_pages, mode=mode)
    doc.save(out_pdf, garbage=4, deflate=True)
    out_json.write_text(
        json.dumps({"chars": chars, "pages": total_pages,
                    "cols": COLS, "rows": ROWS, "cell": CELL},
                   ensure_ascii=False, indent=0),
        encoding="utf-8",
    )
    print(f"已生成 {out_pdf}（{total_pages} 页，{len(chars)} 字，模式: {mode}）")
    print(f"清单 {out_json}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    spec = args[0] if args else "300"
    mode = args[1] if len(args) > 1 and args[1] in ("free", "trace") else "free"
    extra = args[2] if len(args) > 2 else ""
    main(spec, mode, extra)
