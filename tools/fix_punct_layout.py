"""M4-9 标点排版规范修复：尺寸+位置+笔宽三合一。

问题：标点真迹按"bbox 填满字面框"归一化 → 句号变居中大圆蛋、引号占满
整格。中文排版规范（GB/T 15834）要求标点只占全角位的特定角落。

方案：每类标点定义目标区域（em units，baseline 原点），
- line 模式（，；：？！""''《》〈〉【】（）——～）：墨迹等比缩放到目标
  区域后，笔宽校正到与汉字一致（54.7em ↔ 8.22px@148字面）——等比缩小
  会让笔画等比变细，校正后视觉粗细与汉字协调（同一支笔）
- dot 模式（。、…·）：实心墨点无笔宽概念，bbox 直接缩放到规范尺寸
- 其余（数字/字母/数学符号）：保持现状（居中全尺寸真迹，无问题）

处理后再用绝对位置矢量化（不做 bbox 居中）替换 TTF 字形。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
from potrace import Bitmap as PotaBitmap
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = open(ROOT / "output" / "m4" / "fix_punct.log", "w", encoding="utf-8")
_bp = print
def print(*a, **k):
    _bp(*a, **k, file=log, flush=True)

from src.fontforge_lib.segment import page_cells
from src.fontforge_lib.template import COLS
from merge_m4 import stroke_width_px   # 距离变换笔宽测量

FONT = ROOT / "output" / "myhand_full.ttf"
BACKUP = ROOT / "output" / "m4" / "myhand_full_备份_标点排版前.ttf"
CHARS_JSON = ROOT / "samples" / "punct_chars.json"
PHOTO = ROOT / "samples" / "punct_p1.jpg"
LXGW = ROOT / "fonts" / "LXGWWenKai-Regular.ttf"
OUT_DIR = ROOT / "output" / "m4"

# 画布：180px 代表 em 宽 1000 / em 高 [descent -120, ascent 880]
CANVAS = 180
ASCENT = 880

# 目标笔宽（px @ 180画布）：汉字 8.22px@148字面 → 177px 字面 → 8.22*177/148
TARGET_SW = 8.22 * (CANVAS * 0.984) / 148

# 排版规范目标区域：(x0, x1, y0, y1, mode)，em units，y 从 baseline 向上
LAYOUT: dict[str, tuple[float, float, float, float, str]] = {
    # 左下角：句读类
    "。": (60, 310, 30, 280, "dot"),
    "，": (15, 290, 10, 360, "line"),
    "、": (30, 260, 40, 330, "line"),
    "；": (15, 280, 15, 420, "line"),
    "：": (70, 310, 70, 390, "line"),
    # 左侧：问叹
    "？": (20, 520, 40, 820, "line"),
    "！": (40, 370, 40, 820, "line"),
    # 右上角：引号
    "\u201c": (470, 950, 560, 900, "line"),   # "
    "\u201d": (470, 950, 560, 900, "line"),   # "
    "\u2018": (470, 820, 560, 900, "line"),   # '
    "\u2019": (470, 820, 560, 900, "line"),   # '
    # 中部：括号/书名号（开贴右、闭贴左，紧挨内容）
    "（": (350, 900, 60, 860, "line"),
    "）": (100, 650, 60, 860, "line"),
    "《": (300, 900, 100, 780, "line"),
    "》": (100, 700, 100, 780, "line"),
    "〈": (330, 870, 140, 720, "line"),
    "〉": (130, 670, 140, 720, "line"),
    "【": (300, 880, 100, 800, "line"),
    "】": (120, 700, 100, 800, "line"),
    # 字面重心线：连接/省略/波浪/间隔
    "—": (50, 950, 400, 470, "line"),
    "…": (200, 800, 350, 620, "dot"),
    "～": (100, 900, 350, 620, "line"),
    "·": (430, 580, 410, 560, "dot"),
}


def correct_sw(ink: np.ndarray, target_px: float) -> np.ndarray:
    """迭代膨胀/腐蚀把笔宽调到目标（与 merge_m4 同策略）。"""
    u = ink.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(8):
        cur = stroke_width_px(u > 0)
        if cur <= 0:
            break
        if cur < target_px * 0.92:
            u = cv2.dilate(u, k, iterations=1)
        elif cur > target_px * 1.08:
            u = cv2.erode(u, k, iterations=1)
            u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k, iterations=1)
        else:
            break
    return u > 0


def place_glyph(ink: np.ndarray, x0: float, x1: float, y0: float, y1: float,
                mode: str) -> np.ndarray:
    """原始墨迹 → 等比缩放放进目标区域（画布 180px，y 向下）。"""
    ys, xs = np.where(ink)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 1000 / CANVAS
    tx0, tx1 = x0 / s, x1 / s              # em → 画布 px
    ty_top = (ASCENT - y1) / s             # em y1(上) → 画布顶侧 y
    ty_bot = (ASCENT - y0) / s             # em y0(下) → 画布底侧 y
    tw, th = tx1 - tx0, ty_bot - ty_top
    scale = min(tw / w, th / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop.astype(np.uint8), (nw, nh),
                         interpolation=cv2.INTER_NEAREST) > 0
    canvas = np.zeros((CANVAS, CANVAS), dtype=bool)
    # 等比缩放只受宽度约束时（宽扁形墨迹），高度可能超出目标区——
    # 以目标区垂直中心为锚放置，并裁剪到画布内（引号宽墨迹场景）
    ox = int(round(tx0 + (tw - nw) / 2))
    oy = int(round(ty_top + (th - nh) / 2))
    cy_lo, cy_hi = 0, CANVAS
    if oy < 0:            # 顶部越界：贴顶
        oy = 0
    elif oy + nh > CANVAS:  # 底部越界：贴底
        oy = CANVAS - nh
    ox = max(0, min(ox, CANVAS - nw))
    # 目标区域本身在画布外（不应发生，防御）
    oy = max(cy_lo, min(oy, cy_hi - min(nh, CANVAS)))
    nh_eff = min(nh, CANVAS - oy)
    nw_eff = min(nw, CANVAS - ox)
    canvas[oy:oy + nh_eff, ox:ox + nw_eff] = resized[:nh_eff, :nw_eff]
    if mode == "line":
        canvas = correct_sw(canvas, TARGET_SW)
    return canvas


def cell_to_font_contours_abs(cell_ink: np.ndarray) -> list[list[tuple]]:
    """绝对位置矢量化：画布 180px → em [0,1000]×[-120,880]，y 翻转。

    与 trace.cell_to_font_contours 的区别：不做 bbox 居中归一化，
    墨迹在画布中的位置即字形在 em 中的位置（标点排版的关键）。
    """
    if not cell_ink.any():
        return []
    PAD = 4
    arr = np.pad(~cell_ink, PAD, constant_values=True)
    path = PotaBitmap(arr).trace()

    def fx(px: float) -> float:
        return (px - PAD) * 1000 / CANVAS

    def fy(py: float) -> float:
        return ASCENT - (py - PAD) * 1000 / CANVAS

    contours: list[list[tuple]] = []
    for curve in path:
        cmds: list[tuple] = []
        sp = curve.start_point
        cmds.append(("M", fx(sp.x), fy(sp.y)))
        for seg in curve.segments:
            ep = seg.end_point
            if hasattr(seg, "c1"):
                cmds.append(("C", fx(seg.c1.x), fy(seg.c1.y),
                             fx(seg.c2.x), fy(seg.c2.y),
                             fx(ep.x), fy(ep.y)))
            else:
                cmds.append(("L", fx(ep.x), fy(ep.y)))
        cmds.append(("Z",))
        contours.append(cmds)
    return contours


def _draw(pen, contours):
    for seq in contours:
        for cmd in seq:
            if cmd[0] == "M":
                pen.moveTo((cmd[1], cmd[2]))
            elif cmd[0] == "L":
                pen.lineTo((cmd[1], cmd[2]))
            elif cmd[0] == "C":
                pen.curveTo((cmd[1], cmd[2]), (cmd[3], cmd[4]), (cmd[5], cmd[6]))
            elif cmd[0] == "Z":
                pen.closePath()


def replace_glyphs(glyphs: dict[str, list]) -> list[str]:
    font = TTFont(str(FONT), lazy=False)
    cmap = font.getBestCmap()
    order = set(font.getGlyphOrder())
    glyf, hmtx = font["glyf"], font["hmtx"]
    replaced = []
    for ch, contours in glyphs.items():
        gname = cmap.get(ord(ch))
        exists = gname is not None and gname in order
        if not exists:
            gname = f"uni{ord(ch):04X}"
        tt = TTGlyphPen(None)
        cu = Cu2QuPen(tt, max_err=1.5, reverse_direction=False)
        _draw(cu, contours)
        glyf.glyphs[gname] = tt.glyph()
        if not exists:
            font.glyphOrder.append(gname)
            for sub in font["cmap"].tables:
                if sub.isUnicode():
                    sub.cmap[ord(ch)] = gname
            cmap[ord(ch)] = gname
            order.add(gname)
        xs = [cmd[1] for seq in contours for cmd in seq if cmd[0] in ("M", "L")]
        xs += [cmd[5] for seq in contours for cmd in seq if cmd[0] == "C"]
        lsb = int(round(min(xs))) if xs else 0
        adv = hmtx.metrics.get(gname, (1000, 0))[0]
        hmtx.metrics[gname] = (adv, lsb)
        replaced.append(ch)
    font["maxp"].numGlyphs = len(glyf.glyphs)
    font.save(str(FONT))
    return replaced


def make_compare(chs: list[str]):
    """对比图：上排=修复前（居中大尺寸）、下排=修复后（规范位置）。"""
    doc = fitz.open()
    page = doc.new_page(width=842, height=595)
    page.insert_text((30, 40), "标点排版修复：上排=修复前(居中放大) 下排=修复后(规范位置+笔宽统一)",
                     fontsize=11, fontname="F", fontfile=str(LXGW))
    sample = "".join(chs)
    for row, fp in enumerate([BACKUP, FONT]):
        for i in range(0, len(sample), 24):
            page.insert_text((30, 90 + row * 200 + (i // 24) * 44),
                             sample[i:i + 24], fontsize=32,
                             fontname=f"H{row}", fontfile=str(fp))
    doc.save(OUT_DIR / "punct_layout_对比.pdf", garbage=4, deflate=True)
    doc[0].get_pixmap(dpi=150).save(OUT_DIR / "punct_layout_对比.png")


def main():
    print("=== M4-9 标点排版规范修复 ===")
    chars = json.loads(CHARS_JSON.read_text(encoding="utf-8"))["chars"]
    todo = [c for c in chars if c in LAYOUT]
    print(f"清单 {len(chars)} 字符，其中排版标点 {len(todo)}: {''.join(todo)}")

    print("[1] 照片切格取原始墨迹...")
    cells = page_cells(PHOTO, verbose=False)
    inks = {}
    for i, ch in enumerate(chars):
        r, c = divmod(i, COLS)
        ink = cells.get((r, c))
        if ink is not None and ink.any():
            inks[ch] = ink
    print(f"  有效墨迹 {len(inks)}/{len(chars)}")

    print("[2] 缩放到规范区域 + 笔宽校正 + 绝对位置矢量化...")
    glyphs = {}
    for ch in todo:
        if ch not in inks:
            print(f"  [!] {ch} 无墨迹，跳过")
            continue
        x0, x1, y0, y1, mode = LAYOUT[ch]
        try:
            canvas = place_glyph(inks[ch], x0, x1, y0, y1, mode)
            contours = cell_to_font_contours_abs(canvas)
        except Exception as e:
            import traceback
            print(f"  [!] {ch} 处理失败: {e}")
            traceback.print_exc(file=log)
            continue
        if contours:
            glyphs[ch] = contours
        else:
            print(f"  [!] {ch} 矢量化失败")
    print(f"  生成字形 {len(glyphs)}/{len(todo)}")

    print("[3] 备份并替换 TTF 字形...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FONT, BACKUP)
    replaced = replace_glyphs(glyphs)
    print(f"  已替换 {len(replaced)} 字（备份: {BACKUP.name}）")

    print("[4] 前后对比图...")
    make_compare(todo)
    print(f"\n完成: {FONT}")
    print(f"对比图: {OUT_DIR / 'punct_layout_对比.png'}")
    print("后续: python tools/closed_loop_test.py 重跑闭环")


if __name__ == "__main__":
    main()
