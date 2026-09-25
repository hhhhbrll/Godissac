"""字形污迹手术 v3：字顶/字底孤立墨迹全库排查清除。

v2 只查字底（顶部误删过34字部首）。v3 新增字顶检测，安全门：
- **隔离门**（核心）：污迹与主体轮廓的垂直 gap ≥ 120（合法部首/顶点
  都贴着主体：立 gap=41、音 gap=13；污迹悬空：古 gap=253、亮 gap=339）
- 位置门：字顶带（cy > 70%高度）或字底带（cy < 30%高度）
- 尺寸门：h ≤ 200 且 bbox 面积 ≤ 主体 35%
- 覆盖门：当前墨量/文楷 < 0.55（字确实被污迹拉小了才动刀）
- 放大门：删除后字面高度回缩 ≥ 8%（污迹确实撑大了 bbox）

用法：
    python tools/fix_glyph_noise_v3.py --dry-run
    python tools/fix_glyph_noise_v3.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pymupdf as fitz
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT / "output" / "myhand_full.ttf"
WENKAI = ROOT / "fonts" / "LXGWWenKai-Regular.ttf"
BACKUP = ROOT / "output" / "m4" / "myhand_full_备份_污迹手术v3前.ttf"

# 字面框（与 trace.py 一致）
MARGIN, BASE_GAP, ASCENT, EM = 8, 8, 880, 1000
BOX_W, BOX_H = EM - MARGIN - MARGIN, ASCENT - MARGIN - BASE_GAP
BOX_CX = MARGIN + BOX_W / 2
BOX_CY = BASE_GAP + BOX_H / 2

ISOLATE_GAP = 120    # 污迹与非污迹轮廓的垂直隔离下限（em千分单位）
MAX_NOISE_H = 200    # 污迹轮廓最大高度
MAX_AREA_RATIO = 0.35
COV_LIMIT = 0.55     # 墨量比上限（字确实偏小才动刀）
SHRINK_H = 1.10      # 删除后高度回缩比下限

# 排除表（VLM复核为正常/误检风险的字）
EXCLUDE = {"厂"}


def contour_ranges(glyph):
    ends = glyph.endPtsOfContours
    ranges, start = [], 0
    for e in ends:
        ranges.append((start, e))
        start = e + 1
    return ranges


def contour_bbox(glyph, r):
    pts = glyph.coordinates[r[0]:r[1] + 1]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def vgap_between(a, b):
    """两 bbox 垂直 gap：重叠=0。"""
    if a[3] >= b[1] and b[3] >= a[1]:
        return 0
    return b[1] - a[3] if a[3] < b[1] else a[1] - b[3]


def find_noise(glyph):
    """两阶段判定：
    1. 候选门：位置（顶/底带）+ 尺寸（h/面积）
    2. 隔离门：与非候选轮廓的垂直 gap ≥ ISOLATE_GAP
       （合法顶点/部首都贴着邻近笔画：立=41 音=13；污迹悬空：古=253 亮=339）
    3. 放大门：删除后字面高度回缩 ≥ SHRINK_H
    """
    if glyph.numberOfContours <= 1:
        return []
    ranges = contour_ranges(glyph)
    boxes = [contour_bbox(glyph, r) for r in ranges]
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    main_i = max(range(len(boxes)), key=lambda i: areas[i])
    gy0 = min(b[1] for b in boxes)
    gy1 = max(b[3] for b in boxes)
    gh = gy1 - gy0
    if gh <= 0:
        return []
    # 阶段1：候选
    cands = []
    for i, b in enumerate(boxes):
        if i == main_i:
            continue
        h = b[3] - b[1]
        cy = (b[1] + b[3]) / 2
        in_top = cy > gy0 + 0.7 * gh
        in_bot = cy < gy0 + 0.3 * gh
        if not (in_top or in_bot):
            continue
        if h > MAX_NOISE_H or areas[i] > MAX_AREA_RATIO * areas[main_i]:
            continue
        cands.append(i)
    if not cands:
        return []
    # 阶段2：隔离门（与非候选轮廓比）
    noise = []
    for i in cands:
        gap = min(vgap_between(boxes[i], boxes[j])
                  for j in range(len(boxes)) if j != i and j not in cands)
        if gap >= ISOLATE_GAP:
            noise.append(i)
    if not noise:
        return []
    # 阶段3：放大门
    keep = [i for i in range(len(boxes)) if i not in noise]
    if not keep:
        return []
    ah = max(boxes[i][3] for i in keep) - min(boxes[i][1] for i in keep)
    if ah <= 0 or max(gh, 1) / ah < SHRINK_H:
        return []
    return noise


def fix_glyph(glyph, noise):
    """删除污迹轮廓并重缩放到字面框。"""
    ranges = contour_ranges(glyph)
    keep = [i for i in range(len(ranges)) if i not in noise]
    coords, flags, ends = [], [], []
    for i in keep:
        r = ranges[i]
        coords.extend(glyph.coordinates[r[0]:r[1] + 1])
        flags.extend(glyph.flags[r[0]:r[1] + 1])
        ends.append(len(coords) - 1)
    if not coords:
        return False
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return False
    s = min(BOX_W / bw, BOX_H / bh)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    new_coords = [(round((x - cx) * s + BOX_CX), round((y - cy) * s + BOX_CY))
                  for x, y in coords]
    from array import array
    from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
    from fontTools.ttLib.tables.ttProgram import Program
    glyph.coordinates = GlyphCoordinates(new_coords)
    glyph.flags = array("B", flags)
    glyph.endPtsOfContours = ends
    glyph.numberOfContours = len(ends)
    prog = Program()
    prog.fromBytecode(b"")
    glyph.program = prog
    glyph.xMin, glyph.yMin = min(p[0] for p in new_coords), min(p[1] for p in new_coords)
    glyph.xMax, glyph.yMax = max(p[0] for p in new_coords), max(p[1] for p in new_coords)
    return True


# ---------- 墨量比 ----------
_wk_cache: dict[str, int] = {}


def wenkai_ink(ch: str) -> int:
    if ch in _wk_cache:
        return _wk_cache[ch]
    if not hasattr(wenkai_ink, "_font"):
        wenkai_ink._font = ImageFont.truetype(str(WENKAI), 100)
    img = Image.new("L", (140, 140), 255)
    ImageDraw.Draw(img).text((70, 70), ch, font=wenkai_ink._font,
                             fill=0, anchor="mm")
    n = int((np.array(img) < 128).sum())
    _wk_cache[ch] = n
    return n


def our_ink(ch: str) -> int:
    doc = fitz.open()
    pg = doc.new_page(width=140, height=140)
    pg.insert_text((20, 110), ch, fontsize=100, fontname="H",
                   fontfile=str(FONT))
    pix = pg.get_pixmap(dpi=72)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    n = int((arr[:, :, 0] < 128).sum())
    doc.close()
    return n


def main(dry_run: bool = False) -> None:
    font = TTFont(str(FONT), lazy=False)
    glyf = font["glyf"]
    hmtx = font["hmtx"]
    cmap = font.getBestCmap()
    fixed, skipped = [], []
    for cp, gname in cmap.items():
        if not (0x3400 <= cp <= 0x9FFF):  # 仅汉字
            continue
        if chr(cp) in EXCLUDE:
            continue
        glyph = glyf[gname]
        if glyph is None or glyph.numberOfContours <= 0:
            continue
        ch = chr(cp)
        try:
            noise = find_noise(glyph)
            if not noise:
                continue
            wk = wenkai_ink(ch)
            cov = our_ink(ch) / wk if wk else 1.0
            if cov >= COV_LIMIT:
                skipped.append((ch, cov))
                continue
            if fix_glyph(glyph, noise):
                fixed.append((ch, cov))
                adv = hmtx.metrics.get(gname, (1000, 0))[0]
                hmtx.metrics[gname] = (adv, int(glyph.xMin))
        except Exception as e:  # noqa: BLE001
            print(f"  [!] {ch} 处理失败: {e}")
    print(f"手术修复: {len(fixed)} 个")
    for ch, cov in fixed:
        print(f"  {ch}  墨量比={cov:.2f}")
    if skipped:
        print(f"几何命中但墨量正常（未动）: {len(skipped)} 个")
        for ch, cov in skipped[:20]:
            print(f"  {ch}  墨量比={cov:.2f}")
    if dry_run:
        print("[dry-run] 未修改字体")
        return
    if fixed:
        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        if not BACKUP.exists():
            shutil.copy2(FONT, BACKUP)
            print(f"备份: {BACKUP.name}")
        font.save(str(FONT))
        print(f"已保存: {FONT}")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
