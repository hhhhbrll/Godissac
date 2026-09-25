# -*- coding: utf-8 -*-
"""引号排版校准：按文楷实际字面尺寸（双引号251×197、单引号121×197 em）
重排4个弯引号，修复"双引号看起来像单引号"的观感问题。

问题：原 LAYOUT 区域 y 560-900（340高）使引号渲染高达 0.34em（文楷仅
0.197em，1.8倍过高），双引号两笔与单引号同宽同高，小字号下糊成单引号。

方案：区域改为 y 620-820（200高，对齐文楷 617-814），
双引号区 x 460-940，单引号区 x 500-780。仅重排4个引号字形。
"""
import sys
from pathlib import Path

import numpy as np
from potrace import Bitmap as PotaBitmap
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import cv2  # noqa: E402
from src.fontforge_lib.segment import page_cells  # noqa: E402

FONT = ROOT / "output" / "myhand_full.ttf"
PHOTO = ROOT / "samples" / "punct_p1.jpg"
CHARS_JSON = ROOT / "samples" / "punct_chars.json"
BACKUP = ROOT / "output" / "m4" / "myhand_full_备份_引号排版前.ttf"

import json  # noqa: E402
import shutil  # noqa: E402

CANVAS = 180
ASCENT = 880
TARGET_SW = 8.22 * (CANVAS * 0.984) / 148

# 新排版区域 (x0, x1, y0, y1)——y 对齐文楷引号字面 617-814
QUOTE_LAYOUT = {
    "\u201c": (460, 940, 620, 820),   # “ 双开
    "\u201d": (460, 940, 620, 820),   # ” 双闭
    "\u2018": (500, 780, 620, 820),   # ' 单开
    "\u2019": (500, 780, 620, 820),   # ' 单闭
}

from merge_m4 import stroke_width_px  # noqa: E402


def correct_sw(ink, target_px):
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


def place_glyph(ink, x0, x1, y0, y1):
    """等比缩放进目标区域（画布180px，y向下）。"""
    ys, xs = np.where(ink)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 1000 / CANVAS
    tx0, tx1 = x0 / s, x1 / s
    ty_top = (ASCENT - y1) / s
    ty_bot = (ASCENT - y0) / s
    tw, th = tx1 - tx0, ty_bot - ty_top
    scale = min(tw / w, th / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop.astype(np.uint8), (nw, nh),
                         interpolation=cv2.INTER_NEAREST) > 0
    canvas = np.zeros((CANVAS, CANVAS), dtype=bool)
    ox = int(round(tx0 + (tw - nw) / 2))
    oy = int(round(ty_top + (th - nh) / 2))
    oy = max(0, min(oy, CANVAS - nh))
    ox = max(0, min(ox, CANVAS - nw))
    canvas[oy:oy + nh, ox:ox + nw] = resized[:min(nh, CANVAS - oy),
                                             :min(nw, CANVAS - ox)]
    return correct_sw(canvas, TARGET_SW)


def cell_to_font_contours_abs(cell_ink):
    if not cell_ink.any():
        return []
    PAD = 4
    arr = np.pad(~cell_ink, PAD, constant_values=True)
    path = PotaBitmap(arr).trace()

    def fx(px):
        return (px - PAD) * 1000 / CANVAS

    def fy(py):
        return ASCENT - (py - PAD) * 1000 / CANVAS

    contours = []
    for curve in path:
        cmds = [("M", fx(curve.start_point.x), fy(curve.start_point.y))]
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
                pen.curveTo((cmd[1], cmd[2]), (cmd[3], cmd[4]),
                            (cmd[5], cmd[6]))
            elif cmd[0] == "Z":
                pen.closePath()


def main():
    chars = json.loads(CHARS_JSON.read_text(encoding="utf-8"))["chars"]
    print("切格取原始墨迹...")
    cells = page_cells(str(PHOTO), verbose=False)
    inks = {}
    for i, ch in enumerate(chars):
        if ch in QUOTE_LAYOUT:
            r, c = divmod(i, 12)
            ink = cells.get((r, c))
            if ink is not None and ink.any():
                inks[ch] = ink
    print(f"引号墨迹: {len(inks)}/4")

    glyphs = {}
    for ch, (x0, x1, y0, y1) in QUOTE_LAYOUT.items():
        if ch not in inks:
            print(f"  [!] {ch} 无墨迹")
            continue
        canvas = place_glyph(inks[ch], x0, x1, y0, y1)
        contours = cell_to_font_contours_abs(canvas)
        if contours:
            glyphs[ch] = contours
            print(f"  {ch} U+{ord(ch):04X}: {len(contours)} 轮廓")

    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        shutil.copy2(FONT, BACKUP)
        print(f"备份: {BACKUP.name}")

    font = TTFont(str(FONT), lazy=False)
    cmap = font.getBestCmap()
    glyf, hmtx = font["glyf"], font["hmtx"]
    for ch, contours in glyphs.items():
        gname = cmap.get(ord(ch))
        tt = TTGlyphPen(None)
        cu = Cu2QuPen(tt, max_err=1.5, reverse_direction=False)
        _draw(cu, contours)
        glyf.glyphs[gname] = tt.glyph()
        xs = [cmd[1] for seq in contours for cmd in seq if cmd[0] in ("M", "L")]
        xs += [cmd[5] for seq in contours for cmd in seq if cmd[0] == "C"]
        adv = hmtx.metrics.get(gname, (1000, 0))[0]
        hmtx.metrics[gname] = (adv, int(round(min(xs))))
    font.save(str(FONT))
    print(f"已替换 {len(glyphs)} 个引号字形: {FONT}")


if __name__ == "__main__":
    main()
