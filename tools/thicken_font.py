"""全字库笔画加粗：统一到 target em 笔宽（打印可读）。

背景：手写字 trace 后笔宽仅 0.037em（文楷 0.063em 的 60%），生成字
0.046em。打印实测笔画过细显浅（用户反馈"过细以至于模糊不清"）。
之前 thin_gen_glyphs 把生成字减细向手写看齐方向错了——打印场景下
整库都太细，应统一加粗到印刷体量级。

流程（每字独立闭环定标）：
1. fitz 渲染当前字形 → 墨迹位图
2. 距离变换测当前笔宽 → 迭代膨胀到 TARGET_EM（±8%）
3. potrace 重矢量化（与 merge 管线同一 trace，字面框一致）
4. 写回 glyf + hmtx.lsb

用法：
    python tools/thicken_font.py --sample    # 8字对比图，不改字体
    python tools/thicken_font.py --dry-run   # 只报告
    python tools/thicken_font.py             # 备份后全量写回
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.fontforge_lib.trace import cell_to_font_contours  # noqa: E402

FONT = ROOT / "output" / "myhand_full.ttf"
BACKUP = ROOT / "output" / "m4" / "myhand_full_备份_加粗前.ttf"

TARGET_EM = 0.055      # 目标笔宽（em 比，文楷 0.063 的 88%）
RENDER_SIZE = 100      # 渲染字号 pt
RENDER_DPI = 150
CANVAS = 400           # 渲染画布 px


def target_px() -> float:
    return TARGET_EM * RENDER_SIZE * RENDER_DPI / 72.0


def stroke_width_px(ink: np.ndarray) -> float:
    if not ink.any():
        return 0.0
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    return 4.0 * float(dist[ink].mean())


def render_ink(ch: str, fontfile: str) -> np.ndarray | None:
    doc = fitz.open()
    pg = doc.new_page(width=CANVAS, height=CANVAS)
    pg.insert_text((CANVAS * 0.15, CANVAS * 0.8), ch, fontsize=RENDER_SIZE,
                   fontname="H", fontfile=fontfile)
    pix = pg.get_pixmap(dpi=RENDER_DPI)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    doc.close()
    ink = (arr[:, :, 0] < 128)
    if not ink.any():
        return None
    # 裁剪到墨迹 bbox（去除大片空白，加速后续操作）
    ys, xs = np.where(ink)
    y0, y1 = max(ys.min() - 5, 0), min(ys.max() + 6, ink.shape[0])
    x0, x1 = max(xs.min() - 5, 0), min(xs.max() + 6, ink.shape[1])
    return ink[y0:y1, x0:x1]


def thicken(ink: np.ndarray, tgt: float) -> np.ndarray:
    u = ink.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(8):
        cur = stroke_width_px(u > 0)
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


def main(sample: bool = False, dry_run: bool = False) -> None:
    tgt = target_px()
    font = TTFont(str(FONT), lazy=False)
    cmap = font.getBestCmap()
    glyf, hmtx = font["glyf"], font["hmtx"]

    if sample:
        from PIL import Image
        pick = list('中亮创惊举兄羽齐永历')
        tiles = []
        for ch in pick:
            ink = render_ink(ch, str(FONT))
            if ink is None:
                continue
            tiles.append((ch, ink, thicken(ink, tgt)))
        H = 150
        img = Image.new('RGB', (9 * (H + 8) + 8, 2 * (H + 8) + 8), (255, 255, 255))
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        f = None
        for i, (ch, old, new) in enumerate(tiles):
            for row, ink in enumerate((old, new)):
                sub = Image.fromarray((~ink).astype(np.uint8) * 255).convert('RGB')
                sub = sub.resize((H, H))
                img.paste(sub, (8 + i * (H + 8), 8 + row * (H + 8)))
        img.save(str(ROOT / "_tmp_thicken_sample.png"))
        print(f"上=当前 下=加粗到{TARGET_EM}em → _tmp_thicken_sample.png")
        print("字:", ''.join(t[0] for t in tiles))
        return

    done = skip = 0
    before_sw, after_sw = [], []
    for cp in sorted(cmap.keys()):
        # 只加粗 CJK 统一表意区：标点/数字/字母由 fix-punct 精调过排版，
        # 重 trace 会破坏 GB/T 15834 位置；部首/扩展区字罕见不冒险
        if not (0x4E00 <= cp <= 0x9FA5):
            continue
        ch = chr(cp)
        g = glyf[cmap[cp]]
        if g is None or g.numberOfContours <= 0:
            continue
        ink = render_ink(ch, str(FONT))
        if ink is None:
            skip += 1
            continue
        cur = stroke_width_px(ink)
        if cur <= 0:
            skip += 1
            continue
        before_sw.append(cur)
        if dry_run:
            continue
        new = thicken(ink, tgt)
        after_sw.append(stroke_width_px(new))
        contours = cell_to_font_contours(new)
        if not contours:
            skip += 1
            continue
        g2, lsb = contours_to_glyph(contours)
        glyf[cmap[cp]] = g2
        adv = hmtx.metrics.get(cmap[cp], (1000, 0))[0]
        hmtx.metrics[cmap[cp]] = (adv, lsb)
        done += 1
        if done % 300 == 0:
            print(f"  进度 {done} ...")

    if dry_run:
        print(f"[dry-run] 目标笔宽 {tgt:.1f}px；样本当前均值 "
              f"{np.mean(before_sw):.1f}px")
        return
    print(f"写回 {done} 字（跳过 {skip}）")
    print(f"笔宽: {np.mean(before_sw):.1f} → {np.mean(after_sw):.1f} px "
          f"(目标 {tgt:.1f})")
    if not BACKUP.exists():
        shutil.copy2(FONT, BACKUP)
        print(f"备份: {BACKUP.name}")
    font.save(str(FONT))
    print(f"已保存: {FONT}")


if __name__ == "__main__":
    main(sample="--sample" in sys.argv, dry_run="--dry-run" in sys.argv)
