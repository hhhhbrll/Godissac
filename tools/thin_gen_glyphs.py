"""生成字笔画减细：gen_cp 原始位图 → 更细目标归一化 → 矢量化写回 glyf。

背景：生成字笔画粗细均匀、笔锋钝（马克笔感），与真迹的提按变化不协调，
用户观感"笔墨深浅差异大"。侵蚀实验确认减细 ~15% 后视觉更贴近真迹。

流程（只动生成字，真迹不碰）：
1. 读 output/m4/gen_cp/U+XXXX.png 原始位图（合并前的云端生成图缓存）
2. normalize_stroke_width(ink, TARGET_SW) 减细
3. cell_to_font_contours 矢量化（与 merge_m4 同一管线，字面框一致）
4. 写回 myhand_full.ttf 的 glyf + hmtx.lsb

用法：
    python tools/thin_gen_glyphs.py --sample   # 只渲染8字对比图，不改字体
    python tools/thin_gen_glyphs.py            # 备份后全量写回
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.fontforge_lib.trace import cell_to_font_contours  # noqa: E402

FONT = ROOT / "output" / "myhand_full.ttf"
GEN_CP = ROOT / "output" / "m4" / "gen_cp"
HAND_JSON = ROOT / "output" / "m4" / "handwritten_chars.json"
BACKUP = ROOT / "output" / "m4" / "myhand_full_备份_减细前.ttf"

TARGET_SW = 7.0   # 原 8.22；减细 ~15%（侵蚀实验确认的视觉和谐点）


def stroke_width_px(ink: np.ndarray) -> float:
    if not ink.any():
        return 0.0
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    return 4.0 * float(dist[ink].mean())


def normalize_stroke_width(ink: np.ndarray, target_sw: float) -> np.ndarray:
    from src.fontforge_lib.pipeline import normalize_ink
    if not ink.any() or target_sw <= 0:
        return ink
    u = normalize_ink(ink).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(6):
        cur = stroke_width_px(u > 0)
        if cur <= 0:
            break
        if cur < target_sw * 0.92:
            u = cv2.dilate(u, k, iterations=1)
        elif cur > target_sw * 1.08:
            u = cv2.erode(u, k, iterations=1)
            u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k, iterations=1)
        else:
            break
    return u > 0


def decode_png(raw: bytes) -> np.ndarray | None:
    from src.fontforge_lib.segment import _cluster_components
    gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (bw < 128).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)
    return _cluster_components(ink > 0, gap=30)


def load_gen_inks() -> dict[str, np.ndarray]:
    inks = {}
    for png in sorted(GEN_CP.glob("U*.png")):
        try:
            cp = int(png.stem[1:], 16)
        except ValueError:
            continue
        if cp < 0x2E80:
            continue
        ink = decode_png(png.read_bytes())
        if ink is not None and ink.any():
            inks[chr(cp)] = ink
    return inks


def contours_to_glyph(contours):
    """轮廓命令 → (glyph, lsb)，与 buildfont 同参数。"""
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


def main(sample: bool = False) -> None:
    hand = set(json.loads(HAND_JSON.read_text(encoding="utf-8"))["chars"])
    inks = load_gen_inks()
    gens = {c: i for c, i in inks.items() if c not in hand}
    print(f"gen_cp 原始位图 {len(inks)}，其中生成字 {len(gens)}")

    if sample:
        import random
        from PIL import Image, ImageDraw, ImageFont
        random.seed(3)
        pick = random.sample(sorted(gens), 8)
        f_old = ImageFont.truetype(str(FONT), 100)
        # 新管线位图直接画
        H = 130
        img = Image.new("RGB", (8 * 120 + 20, 2 * H + 10), (255, 255, 255))
        d = ImageDraw.Draw(img)
        for i, ch in enumerate(pick):
            d.text((10 + i * 120, 5), ch, font=f_old, fill=0)
        for i, ch in enumerate(pick):
            thin = normalize_stroke_width(gens[ch], TARGET_SW)
            sub = Image.fromarray((~thin).astype(np.uint8) * 255).resize((120, 120))
            img.paste(sub, (10 + i * 120, H + 5))
        img.save(str(ROOT / "_tmp_thin_sample.png"))
        print("上=当前字体 下=减细位图 → _tmp_thin_sample.png")
        print("字:", "".join(pick))
        return

    font = TTFont(str(FONT), lazy=False)
    glyf, hmtx, cmap = font["glyf"], font["hmtx"], font.getBestCmap()
    done, skip = 0, 0
    for ch, ink in gens.items():
        gname = cmap.get(ord(ch))
        if gname is None:
            skip += 1
            continue
        thin = normalize_stroke_width(ink, TARGET_SW)
        contours = cell_to_font_contours(thin)
        if not contours:
            skip += 1
            continue
        g, lsb = contours_to_glyph(contours)
        glyf[gname] = g
        adv = hmtx.metrics.get(gname, (1000, 0))[0]
        hmtx.metrics[gname] = (adv, lsb)
        done += 1
    print(f"写回 {done} 字（跳过 {skip}）")
    if not BACKUP.exists():
        import shutil
        shutil.copy2(FONT, BACKUP)
        print(f"备份: {BACKUP.name}")
    font.save(str(FONT))
    print(f"已保存: {FONT}")


if __name__ == "__main__":
    main(sample="--sample" in sys.argv)
