"""生成续写字对照清单 PDF：与采集表同布局（12列×16行、同页序），带行列号。

对着采集表书写时用：清单上 行列号 = 采集表上格子位置。
写错的字按 "页,行,列" 登记到 {prefix}_remap.json 的 void/extra。

用法：
    python tools/make_ref_list.py [chars_json]   # 默认 samples/expansion_chars.json

产出：
    samples/{prefix}_reflist.pdf
"""

import json
import sys
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fontforge_lib.template import COLS, ROWS

FONT = ROOT / "fonts" / "LXGWWenKai-Regular.ttf"

# 表格几何（pt）
LEFT, TOP = 60.0, 100.0
LABEL_W = 42.0          # 行号列宽
CELL_W = 38.0           # 字格宽
CELL_H = 40.0           # 字格高
HEAD_H = 26.0           # 列号表头高
CHAR_SIZE = 20          # 清单字号（大而清晰）
HEAD_SIZE = 9


def draw_page(doc: fitz.Document, chars: list[str], page_no: int, total_pages: int,
              title: str):
    page = doc.new_page(width=595, height=842)
    font = fitz.Font(fontfile=str(FONT))

    def put(x, y, text, size, color=(0, 0, 0), fontname="F"):
        page.insert_text((x, y), text, fontsize=size, fontname=fontname,
                         fontfile=str(FONT), color=color)

    # 标题
    tw = font.text_length(title, fontsize=16)
    put((595 - tw) / 2, 48, title, 16)
    sub = f"第 {page_no} / {total_pages} 页（与采集表页序一致）"
    sw = font.text_length(sub, fontsize=10)
    put((595 - sw) / 2, 68, sub, 10, (0.3, 0.3, 0.3))

    # 列号表头
    for c in range(COLS):
        cx = LEFT + LABEL_W + c * CELL_W
        num = str(c + 1)
        nw = font.text_length(num, fontsize=HEAD_SIZE)
        put(cx + (CELL_W - nw) / 2, TOP + HEAD_H - 8, num, HEAD_SIZE, (0.35, 0.35, 0.35))
        page.draw_line(fitz.Point(cx, TOP), fitz.Point(cx, TOP + HEAD_H),
                       color=(0.85, 0.85, 0.85), width=0.4)
    page.draw_line(fitz.Point(LEFT + LABEL_W + COLS * CELL_W, TOP),
                   fitz.Point(LEFT + LABEL_W + COLS * CELL_W, TOP + HEAD_H),
                   color=(0.85, 0.85, 0.85), width=0.4)

    # 行：行号 + 该行12字
    for r in range(ROWS):
        idx0 = page_no * COLS * ROWS + r * COLS
        if idx0 >= len(chars):
            break
        y0 = TOP + HEAD_H + r * CELL_H
        # 行号 + 横线
        lab = f"行{r + 1}"
        lw = font.text_length(lab, fontsize=10)
        put(LEFT + (LABEL_W - lw) / 2, y0 + CELL_H / 2 + 3.5, lab, 10, (0.35, 0.35, 0.35))
        page.draw_line(fitz.Point(LEFT, y0), fitz.Point(LEFT + LABEL_W + COLS * CELL_W, y0),
                       color=(0.85, 0.85, 0.85), width=0.4)
        # 该行的字（每5列留窄缝分组，便于快速定位）
        for c in range(COLS):
            idx = idx0 + c
            if idx >= len(chars):
                break
            ch = chars[idx]
            cx = LEFT + LABEL_W + c * CELL_W
            cw = font.text_length(ch, fontsize=CHAR_SIZE)
            put(cx + (CELL_W - cw) / 2, y0 + CELL_H / 2 + CHAR_SIZE * 0.36, ch, CHAR_SIZE)
            if c % 5 == 4 and c < COLS - 1:   # 第5/10列右侧分组线（更醒目）
                page.draw_line(fitz.Point(cx + CELL_W, y0),
                               fitz.Point(cx + CELL_W, y0 + CELL_H),
                               color=(0.7, 0.7, 0.7), width=0.6)
    # 底框线
    n_rows = min(ROWS, -(-(len(chars) - page_no * COLS * ROWS) // COLS))
    y_end = TOP + HEAD_H + max(n_rows, 1) * CELL_H
    page.draw_line(fitz.Point(LEFT, y_end), fitz.Point(LEFT + LABEL_W + COLS * CELL_W, y_end),
                   color=(0.85, 0.85, 0.85), width=0.4)

    foot = "清单行列号 = 采集表格子位置；写错登记 {prefix}_remap.json: 页,行,列"
    fw = font.text_length(foot, fontsize=8.5)
    put((595 - fw) / 2, 822, foot, 8.5, (0.45, 0.45, 0.45))


def main(chars_json: Path):
    prefix = chars_json.stem.removesuffix("_chars")
    chars = json.loads(chars_json.read_text(encoding="utf-8"))["chars"]
    total_pages = -(-len(chars) // (COLS * ROWS))
    doc = fitz.open()
    title = f"续写字对照清单（{prefix}，共 {len(chars)} 字）"
    for p in range(total_pages):
        draw_page(doc, chars, p, total_pages, title)
    out = chars_json.parent / f"{prefix}_reflist.pdf"
    doc.save(out, garbage=4, deflate=True)
    print(f"已生成 {out}（{total_pages} 页，{len(chars)} 字，"
          f"布局与采集表一致：{COLS}列×{ROWS}行/页）")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "samples" / "expansion_chars.json")
    main(Path(arg))
