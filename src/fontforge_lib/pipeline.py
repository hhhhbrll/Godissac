"""M3 端到端管线：采集照片(多张,按页序) → 个人手写 TTF + 质检报告。

用法：
    python -m src.fontforge_lib.pipeline 照片1.png 照片2.png ...

产出：
    output/myhand.ttf          个人字体
    output/myhand_质检.png     用新字体渲染的样张
    output/myhand_cells/       每格原始墨迹与矢量化回显对比图（抽查）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

from .buildfont import build_ttf
from .segment import page_cells
from .template import ASCENT, CELL, CELL_PX, CELLS_PER_PAGE, COLS, EM, ROWS
from .trace import cell_to_font_contours

ROOT = Path(__file__).resolve().parents[2]
CHARS_JSON = ROOT / "samples" / "collect_chars.json"
OUT_DIR = ROOT / "output"


def load_remap(path: Path | None = None) -> tuple[set, dict]:
    """读取作废格与补写格登记表（默认 samples/collect_remap.json）。

    格式（页/行/列均为 1-based）：
    {
      "void":  [[1, 3, 5], ...],              # 作废格：写错的字所在格
      "extra": {"2, 16, 3": "鹏", ...}        # 补写格：正确字写在哪个空格
    }
    """
    p = path or (ROOT / "samples" / "collect_remap.json")
    if not p.exists():
        return set(), {}
    data = json.loads(p.read_text(encoding="utf-8"))
    void = {tuple(int(v) for v in item) for item in data.get("void", [])}
    extra = {tuple(int(t) for t in k.split(",")): v for k, v in data.get("extra", {}).items()}
    return void, extra


def collect_glyphs(photos: list[str], chars_json: Path | None = None,
                   remap_json: Path | None = None) -> tuple[dict[str, list], list[str], dict[str, np.ndarray]]:
    """多页照片 → {汉字: 轮廊} + 原始墨迹（供IoU质检）。返回 (glyphs, missing, cells_by_char)。

    chars_json/remap_json：默认用初次采集的 collect_chars/collect_remap；
    补录流程传入 samples/recollect_chars.json 与 samples/recollect_remap.json。
    """
    chars = json.loads((chars_json or CHARS_JSON).read_text(encoding="utf-8"))["chars"]
    void, extra = load_remap(remap_json)
    extra_chars = set(extra.values())
    glyphs: dict[str, list] = {}
    missing: list[str] = []
    cells_by_char: dict[str, np.ndarray] = {}
    for page_no, photo in enumerate(photos):
        print(f"处理 {photo} (第{page_no + 1}页)")
        cells = page_cells(photo)
        for (r, c), ink in cells.items():
            key = (page_no + 1, r + 1, c + 1)  # 1-based (页,行,列)
            idx = page_no * CELLS_PER_PAGE + r * COLS + c
            if key in void:
                # 作废格：若未提供补写，该字记为缺失
                if idx < len(chars) and chars[idx] not in extra_chars:
                    missing.append(chars[idx])
                continue
            ch = chars[idx] if idx < len(chars) else extra.get(key)
            if ch is None:
                continue
            if not ink.any():
                missing.append(ch)
                continue
            contours = cell_to_font_contours(ink)
            if contours:
                glyphs[ch] = contours
                cells_by_char[ch] = ink
    for key, ch in extra.items():
        if ch not in glyphs:
            print(f"  ⚠ 补写格 {key}({ch}) 为空或矢量化失败，该字仍缺失")
    return glyphs, missing, cells_by_char


def glyph_bitmap(font_path: Path, ch: str, px: int = CELL_PX) -> np.ndarray:
    """把新字体里的单个字渲染回墨迹位图，用于 IoU 对比。"""
    doc = fitz.open()
    page = doc.new_page(width=CELL, height=CELL)
    page.insert_text((0, ASCENT / EM * CELL), ch, fontsize=CELL,
                     fontname="F", fontfile=str(font_path), color=(0, 0, 0))
    zoom = px / CELL
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr[:, :, 0] < 128


def normalize_ink(ink: np.ndarray, canvas: int = 180, target: int = 148) -> np.ndarray:
    """墨迹按 bbox 等比缩放至 target、居中到 canvas（与 trace.py 归一化一致的位图版）。

    IoU 质检用：字体字形与原始墨迹各自归一化后对比，检验纯形状保真度，
    排除书写位置/大小偏差的干扰。
    """
    ys, xs = np.where(ink)
    if len(xs) == 0:
        return ink
    crop = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = crop.shape
    s = target / max(w, h, 1)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    resized = cv2.resize(crop.astype(np.uint8), (nw, nh),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    out = np.zeros((canvas, canvas), dtype=bool)
    ox, oy = (canvas - nw) // 2, (canvas - nh) // 2
    out[oy : oy + nh, ox : ox + nw] = resized
    return out


def iou_qc(font_path: Path, glyphs: dict, cells_by_char: dict[str, np.ndarray],
           sample: int = 30, seed: int = 7) -> float:
    """抽样对比：矢量字渲染回位图 vs 原始墨迹（双方均 bbox 归一化），返回平均 IoU。"""
    rng = np.random.default_rng(seed)
    keys = sorted(glyphs)
    picked = [k for k in rng.choice(keys, size=min(sample, len(keys)), replace=False)]
    ious = []
    for ch in picked:
        a = normalize_ink(glyph_bitmap(font_path, ch))
        b = normalize_ink(cells_by_char[ch])
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        if union:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def preview(font_path: Path, glyphs: dict):
    """用新字体渲染一张样张（只含已采集字）。"""
    have = "".join(sorted(glyphs))
    sentence = "".join(ch for ch in "我们都是中国人大小多少今天上学读书写字一起很好平安快乐" if ch in have)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 100), "个人手写字体样张（合成数据验证）", fontsize=16,
                     fontname="F", fontfile=str(ROOT / "fonts" / "LXGWWenKai-Regular.ttf"))
    page.insert_text((60, 170), sentence[:24], fontsize=28, fontname="H", fontfile=str(font_path))
    page.insert_text((60, 240), sentence[24:48], fontsize=28, fontname="H", fontfile=str(font_path))
    page.insert_text((60, 330), "常用字集：", fontsize=12,
                     fontname="F", fontfile=str(ROOT / "fonts" / "LXGWWenKai-Regular.ttf"))
    for i in range(0, len(have), 40):
        page.insert_text((60, 360 + (i // 40) * 20), have[i : i + 40], fontsize=11,
                         fontname="H", fontfile=str(font_path))
    doc[0].get_pixmap(dpi=150).save(OUT_DIR / "myhand_质检.png")


def main(photos: list[str]):
    OUT_DIR.mkdir(exist_ok=True)
    glyphs, missing, cells_by_char = collect_glyphs(photos)
    print(f"\n采集完成: {len(glyphs)} 字有效" + (f"，空格子 {len(missing)} 个: {''.join(missing[:20])}" if missing else ""))

    # 边框污染探针（灰度判别版）：贴边墨迹灰度>130 才是格线混入；
    # 真实笔画贴边（字写大了）灰度<130，无害，bbox 归一化会处理
    from .segment import extract_cells, find_marker_centers, load_gray, warp_to_template
    from .template import CELL, GRID_LEFT, GRID_TOP, SCALE

    pollution = []
    for page_no, photo in enumerate(photos):
        warped = warp_to_template(load_gray(photo), find_marker_centers(load_gray(photo)))
        cells_p = extract_cells(warped)
        chars = json.loads(CHARS_JSON.read_text(encoding="utf-8"))["chars"]
        void, extra = load_remap()
        extra_cells = {k: v for k, v in extra.items()}
        for (r, c), ink in cells_p.items():
            key = (page_no + 1, r + 1, c + 1)
            idx = page_no * CELLS_PER_PAGE + r * COLS + c
            if key in void:
                continue
            ch = chars[idx] if idx < len(chars) else extra_cells.get(key)
            if ch is None or ch not in cells_by_char or not ink.any():
                continue
            edge = np.zeros_like(ink)
            edge[0:3, :] |= ink[0:3, :]
            edge[-3:, :] |= ink[-3:, :]
            edge[:, 0:3] |= ink[:, 0:3]
            edge[:, -3:] |= ink[:, -3:]
            if edge.any():
                gx = int((GRID_LEFT + c * CELL) * SCALE)
                gy = int((GRID_TOP + r * CELL) * SCALE)
                vals = warped[gy : gy + 180, gx : gx + 180][edge]
                if (vals < 130).mean() <= 0.5:  # 贴边部分多为浅灰 → 格线
                    pollution.append(ch)
    if pollution:
        print(f"⚠ 边框污染: {len(pollution)} 字混入格线: {''.join(pollution[:20])}")
    else:
        print("边框污染探针: 0（无格线混入）✓")

    # 两种绕向都构建，IoU 高者胜出（自动解决 TrueType 填充方向问题）
    best, best_iou, best_path = None, -1, None
    for rev in (False, True):
        p = OUT_DIR / ("myhand_rev.ttf" if rev else "myhand.ttf")
        build_ttf(glyphs, p, reverse=rev)
        iou = iou_qc(p, glyphs, cells_by_char)
        print(f"绕向 reverse={rev}: 平均IoU={iou:.3f}")
        if iou > best_iou:
            best, best_iou, best_path = rev, iou, p
    final = OUT_DIR / "myhand.ttf"
    if best_path != final:
        best_path.replace(final)
    (OUT_DIR / "myhand_rev.ttf").unlink(missing_ok=True)

    print(f"\n最终字体: {final}（reverse={best}, 平均IoU={best_iou:.3f}）")
    print("IoU 解读: >0.85 优秀 | 0.7~0.85 可用 | <0.7 需检查")
    preview(final, glyphs)
    print(f"样张: {OUT_DIR / 'myhand_质检.png'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
