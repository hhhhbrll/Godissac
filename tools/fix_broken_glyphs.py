"""碎字修复：墨量比过低的字从 results/ 原始生成位图重做。

背景：初版 merge 时部分生成字 IoU 校验后仍漏入字库（如"兄"字形碎裂
成碎片、墨量仅为文楷 0.22）——笔画加粗救不了碎字（碎片只会变粗）。

流程：
1. 全字库渲染扫描：墨量/文楷 < THRESH 的字为碎字候选
2. 从 output/m4/results/ 读该字原始位图（云端 FontDiffuser 生成图）
3. 统一画布 → 迭代膨胀到目标笔宽（与 thicken_font 一致 0.055em）
4. potrace 重矢量化写回 glyf

用法：
    python tools/fix_broken_glyphs.py --dry-run
    python tools/fix_broken_glyphs.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from src.fontforge_lib.trace import cell_to_font_contours  # noqa: E402

FONT = ROOT / "output" / "myhand_full.ttf"
WENKAI = ROOT / "fonts" / "LXGWWenKai-Regular.ttf"
RESULTS = ROOT / "output" / "m4" / "results"
BACKUP = ROOT / "output" / "m4" / "myhand_full_备份_碎字修复前.ttf"

THRESH = 0.45      # 墨量比下限（低于=碎字）
TARGET_EM = 0.055  # 目标笔宽（与 thicken_font 一致）
CANVAS = 200       # 统一画布高（px）
MIN_RUN = 20       # 原始位图最小墨迹像素（防空图）


def ink_ratio(ch: str, f_our, f_wk) -> float:
    def ink(font):
        img = Image.new("L", (220, 220), 255)
        ImageDraw.Draw(img).text((55, 55), ch, font=font, fill=0)
        return (np.array(img) < 128).sum()
    w = ink(f_wk)
    return ink(f_our) / w if w else 1.0


def load_raw_ink(ch: str) -> np.ndarray | None:
    """results/ 原始 PNG → 二值墨迹图。"""
    stem = None
    for p in RESULTS.glob("*.png"):
        try:
            real = p.stem.encode("cp437").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            real = p.stem
        if real == ch:
            stem = p
            break
    if stem is None:
        return None
    gray = cv2.imdecode(np.frombuffer(stem.read_bytes(), np.uint8),
                         cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (bw < 128).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)
    return ink > 0


def thicken(ink: np.ndarray, tgt: float) -> np.ndarray:
    u = ink.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(10):
        dist = cv2.distanceTransform(u, cv2.DIST_L2, 5)
        cur = 4.0 * float(dist[u > 0].mean()) if u.any() else 0
        if cur <= 0:
            break
        if cur < tgt * 0.92:
            u = cv2.dilate(u, k, iterations=1)
        elif cur > tgt * 1.08:
            u = cv2.erode(u, k, iterations=1)
        else:
            break
    return u > 0


def contours_to_glyph(contours):
    tt = TTGlyphPen(None)
    cu = Cu2QuPen(tt, max_err=1.5, reverse_direction=False)
    for contour in contours:
        for cmd in contour:
            if cmd[0] == "M":
                cu.moveTo((cmd[1], cmd[2]))
            elif cmd[0] == "L":
                cu.lineTo((cmd[1], cmd[2]))
            elif cmd[0] == "C":
                cu.curveTo((cmd[1], cmd[2]), (cmd[3], cmd[4]), (cmd[5], cmd[6]))
            elif cmd[0] == "Z":
                cu.closePath()
    g = tt.glyph()
    xs = [pt[1] for c in contours for cmd in c if cmd[0] in ("M", "L")
          for pt in [(cmd[0], cmd[1])]]
    xs += [cmd[5] for c in contours for cmd in c if cmd[0] == "C"]
    lsb = int(round(min(xs))) if xs else 0
    return g, lsb


def main(dry_run: bool = False) -> None:
    import json as _json
    hand = set(_json.loads((ROOT / "output" / "m4" / "handwritten_chars.json")
                           .read_text(encoding="utf-8"))["chars"])
    f_our = ImageFont.truetype(str(FONT), 100)
    f_wk = ImageFont.truetype(str(WENKAI), 100)

    font = TTFont(str(FONT), lazy=False)
    cmap = font.getBestCmap()
    glyf, hmtx = font["glyf"], font["hmtx"]

    # 1. 扫描碎字
    broken = []
    for cp in sorted(cmap.keys()):
        if not (0x4E00 <= cp <= 0x9FA5):
            continue
        ch = chr(cp)
        if ch in hand:
            continue  # 手写字不走 results（那是生成字位图库）
        r = ink_ratio(ch, f_our, f_wk)
        if r < THRESH:
            broken.append((ch, r))
    print(f"碎字候选 {len(broken)} 个（墨量比<{THRESH}）:")
    print("  " + " ".join(f"{c}({r:.2f})" for c, r in broken[:60]))
    if len(broken) > 60:
        print(f"  ... 共 {len(broken)}")
    if dry_run:
        return

    # 2. 从 results 重做
    fixed, no_src = [], []
    for ch, r in broken:
        ink = load_raw_ink(ch)
        if ink is None or ink.sum() < MIN_RUN:
            no_src.append(ch)
            continue
        # 统一画布：墨迹 bbox 高度 → CANVAS
        ys, xs = np.where(ink)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        s = CANVAS / max(h, 1)
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        sub = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        sub = cv2.resize(sub.astype(np.uint8), (nw, nh),
                         interpolation=cv2.INTER_NEAREST) > 0
        # 目标笔宽（px）：字面高≈0.88em → 0.055em 笔宽
        tgt = CANVAS / 0.88 * TARGET_EM
        sub = thicken(sub, tgt)
        contours = cell_to_font_contours(sub)
        if not contours:
            no_src.append(ch)
            continue
        g, lsb = contours_to_glyph(contours)
        glyf[cmap[ord(ch)]] = g
        adv = hmtx.metrics.get(cmap[ord(ch)], (1000, 0))[0]
        hmtx.metrics[cmap[ord(ch)]] = (adv, lsb)
        fixed.append(ch)

    print(f"\n修复 {len(fixed)} 字: {''.join(fixed[:80])}")
    if no_src:
        print(f"无原始位图（保持现状）: {len(no_src)} 字: {''.join(no_src[:40])}")
    if fixed:
        if not BACKUP.exists():
            shutil.copy2(FONT, BACKUP)
            print(f"备份: {BACKUP.name}")
        font.save(str(FONT))
        print(f"已保存: {FONT}")
        # 复检
        f_new = ImageFont.truetype(str(FONT), 100)
        ratios = [ink_ratio(c, f_new, f_wk) for c in fixed]
        print(f"复检墨量比: min={min(ratios):.2f} mean={np.mean(ratios):.2f}")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
