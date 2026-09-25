"""字体底噪手术 v2：扫描 myhand_full.ttf 全部汉字，剔除"底部孤立墨迹"。

问题场景：采集照片中字形下方有污点/底栏墨迹 → bbox 被拉长 → 等比缩放后
字整体变小（用户反馈"弟""危""号"等字异常偏小的根因）。

v1 教训：gap≥150 的"完全脱开"判定过严——"危/号"的污迹 gap 仅 63-87
被漏掉；且坐标方向写反导致从未生效。

v2 判定（y-up，em=1000，glyf 原始坐标顶大底小）：
- **只检测字底带**（v2.1 教训：照片污迹全部来自采集底栏，只在字底；
  顶部带检测曾误删 34 字的合法部首——宀厂广亠丁等，如历丢厂头、
  安丢宝盖、府丢广头。部首轮廓 y 高，污迹轮廓 y 低，方向不能搞反）
- 几何门：轮廓整体位于字形底部 30% 带内，
  且不侵入主体纵向范围（与其余轮廓的最近边 gap ≥ -60），
  且 h ≤ 170、bbox 面积 ≤ 主体 25%
- 放大门：删除后字面高度回缩比 ≥ 1.18（污迹拉长 bbox 高度）
- 覆盖门：当前渲染覆盖率 < 文楷的 55%（字确实偏小才动刀）
三门齐过才手术；合法底笔（心/灬 等点在主体纵向范围内）放大门不过，
正常字覆盖门不过，双重保险。

修复：删除噪声轮廓 → 剩余轮廓重新等比缩放到标准字面框（984x864，
中心 500/440，与 trace.py 归一化一致）。

用法：
    python tools/fix_glyph_noise.py --dry-run    # 只报告不修改
    python tools/fix_glyph_noise.py             # 备份后修复
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pymupdf as fitz
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT / "output" / "myhand_full.ttf"
WENKAI = ROOT / "fonts" / "LXGWWenKai-Regular.ttf"
BACKUP = ROOT / "output" / "m4" / "myhand_full_备份_底噪手术前.ttf"

# 字面框（与 trace.py 一致）
MARGIN, BASE_GAP, ASCENT, EM = 8, 8, 880, 1000
BOX_W, BOX_H = EM - MARGIN - MARGIN, ASCENT - MARGIN - BASE_GAP
BOX_CX = MARGIN + BOX_W / 2
BOX_CY = BASE_GAP + BOX_H / 2

MAX_NOISE_H = 170       # 噪声轮廓最大高度
MAX_AREA_RATIO = 0.25   # 噪声 bbox 面积 / 主体 bbox 面积
BAND = 0.7              # 噪声顶边须位于字形底部 (1-BAND) 带内
OVERLAP_TOL = 60        # 噪声顶边允许高出其余轮廓底边的量
GROW_H = 1.18           # 删除后字面高度回缩比下限（底噪只拉长高度不拉宽度；
                        # 合法底点心/小 高度比≈1.08-1.13，污迹字≈1.22+）
COV_LIMIT = 0.55        # 当前覆盖率/文楷 上限（字确实偏小才动刀）


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


def transform_coords(coords, s, cx, cy):
    return [(round((x - cx) * s + BOX_CX), round((y - cy) * s + BOX_CY))
            for x, y in coords]


def find_noise(glyph):
    """返回应删除的轮廓下标列表。y-up 坐标（TrueType 标准，顶大底小）。"""
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
    noise = []
    for i, b in enumerate(boxes):
        if i == main_i:
            continue
        h = b[3] - b[1]
        # 几何门：整体在字底带（y-up：底带=轮廓顶边 ≤ ymin + 30%高度）。
        # 顶部不检测（合法部首在顶部，误删风险高，见文件头 v2.1 教训）
        in_bot = b[3] <= gy0 + (1 - BAND) * gh
        if not in_bot:
            continue
        # 不侵入主体纵向范围：与其余轮廓最近边的 gap ≥ -OVERLAP_TOL
        rest = min(boxes[j][1] for j in range(len(boxes)) if j != i)
        gap = rest - b[3]
        if gap < -OVERLAP_TOL:
            continue
        if h > MAX_NOISE_H or areas[i] > MAX_AREA_RATIO * areas[main_i]:
            continue
        noise.append(i)
    if not noise:
        return []
    # 放大门：删除后字面高度回缩比
    keep = [i for i in range(len(boxes)) if i not in noise]
    if not keep:
        return []
    ah = max(boxes[i][3] for i in keep) - min(boxes[i][1] for i in keep)
    if ah <= 0:
        return []
    if max(gh, 1) / ah < GROW_H:
        return []
    return noise


def fix_glyph(glyph, noise):
    """按 noise 下标删除轮廓并重缩放到字面框。返回 True=已修改。"""
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
    new_coords = transform_coords(coords, s, cx, cy)
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
    nxs = [p[0] for p in new_coords]
    nys = [p[1] for p in new_coords]
    glyph.xMin, glyph.yMin = min(nxs), min(nys)
    glyph.xMax, glyph.yMax = max(nxs), max(nys)
    return True


# ---------- 覆盖率（渲染像素 vs 文楷） ----------

_wk_pix_cache: dict[str, int] = {}


def wenkai_ink(ch: str) -> int:
    if ch in _wk_pix_cache:
        return _wk_pix_cache[ch]
    from PIL import Image, ImageDraw, ImageFont
    if not hasattr(wenkai_ink, "_font"):
        wenkai_ink._font = ImageFont.truetype(str(WENKAI), 100)
    img = Image.new("L", (140, 140), 255)
    ImageDraw.Draw(img).text((70, 70), ch, font=wenkai_ink._font, fill=0,
                             anchor="mm")
    n = int((np.array(img) < 128).sum())
    _wk_pix_cache[ch] = n
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
        # 仅汉字区（标点/符号的底部点画是字形本体，如！？÷，绝不能动）
        if not (0x3400 <= cp <= 0x9FFF):
            continue
        glyph = glyf[gname]
        if glyph is None or glyph.numberOfContours <= 0:
            continue
        ch = chr(cp)
        try:
            noise = find_noise(glyph)
            if not noise:
                continue
            # 覆盖门：字确实偏小才动刀
            wk = wenkai_ink(ch)
            cov = our_ink(ch) / wk if wk else 1.0
            if cov >= COV_LIMIT:
                skipped.append((ch, cov))
                continue
            if fix_glyph(glyph, noise):
                fixed.append((ch, cov))
                adv = hmtx.metrics.get(gname, (1000, 0))[0]
                hmtx.metrics[gname] = (adv, int(glyph.xMin))
        except Exception as e:  # noqa: BLE001 单字失败不阻断
            import traceback
            traceback.print_exc()
            print(f"  [!] {ch}({gname}) 处理失败: {e}")
    print(f"手术修复: {len(fixed)} 个")
    for ch, cov in fixed:
        print(f"  {ch}  覆盖率={cov:.2f}")
    if skipped:
        print(f"几何命中但覆盖率正常（未动）: {len(skipped)} 个")
        for ch, cov in skipped:
            print(f"  {ch}  覆盖率={cov:.2f}")
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
