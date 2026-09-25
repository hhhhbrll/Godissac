"""白纸手写补字：无田字格/无锚点的照片 → 提取墨迹 → 合并入 myhand_full.ttf。

适用场景：用户在白纸上补写少数几个字（如"弟、居、/"），拍照后直接合并。
按墨迹的水平分组（左→右）与字符表顺序一一对应。

用法：
    python tools/merge_freehand.py "samples/补录：弟、居、左斜杠.jpg" "弟居/"

窄字形处理："/"（句读斜杠）等窄字形若按常规归一化会占满整字面框，
插入字间会压到相邻字——自动做 x 收窄（保持高度、压扁宽度，斜杠变陡）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fontforge_lib.trace import cell_to_font_contours
from merge_recollect import FONT, replace_glyphs

OUT_DIR = ROOT / "output" / "m4"

# 窄字形 x 收窄目标（字面框宽的比例）
NARROW_CHARS = {"/"}
NARROW_MAX_W = 420   # 窄字形最大宽度（font units, em=1000）
NARROW_MAX_H = 760   # 窄字形最大高度（不占满行高，句读"/"略小于正文）


def extract_groups(gray: np.ndarray) -> list[np.ndarray]:
    """白纸照片 → 按水平分组提取墨迹位图（True=墨）。

    暗照片处理：光照归一化（除以大核背景）→ Otsu 二值化 → 去小碎点
    （碎片会变成大量碎轮廓，字形会呈斑点状）。
    """
    # 光照归一化：消除拍摄亮度不均
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121)))
    norm = cv2.divide(gray, bg, scale=255)
    _, bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # 去小碎点（<250px 的孤立墨点多为纸面噪点/笔画碎片）
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, 4] >= 250
    bw = keep[labels].astype(np.uint8) * 255
    comps = [(stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4])
             for i in range(1, n) if stats[i, 4] >= 250]
    if not comps:
        raise RuntimeError("照片中未检出墨迹")
    comps.sort(key=lambda t: t[0])
    # 水平分组：相邻组件 x 间隙 > 200px 视为不同字
    groups: list[list] = []
    for c in comps:
        if groups and c[0] - max(g[0] + g[2] for g in groups[-1]) < 200:
            groups[-1].append(c)
        else:
            groups.append([c])
    inks = []
    for g in groups:
        x0 = max(0, min(c[0] for c in g) - 25)
        x1 = min(gray.shape[1], max(c[0] + c[2] for c in g) + 25)
        y0 = max(0, min(c[1] for c in g) - 25)
        y1 = min(gray.shape[0], max(c[1] + c[3] for c in g) + 25)
        inks.append(bw[y0:y1, x0:x1] > 0)
    return inks


def scale_narrow(contours: list[list[tuple]]) -> list[list[tuple]]:
    """窄字形（/等）收窄：y 压到 NARROW_MAX_H，x 压到 NARROW_MAX_W 以内。"""
    xs, ys = [], []
    for seq in contours:
        for cmd in seq:
            if cmd[0] in ("M", "L"):
                xs.append(cmd[1]); ys.append(cmd[2])
            elif cmd[0] == "C":
                xs.append(cmd[5]); ys.append(cmd[6])
    if not xs:
        return contours
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    bw, bh = x1 - x0, y1 - y0
    if bh <= 0:
        return contours
    sy = min(1.0, NARROW_MAX_H / bh)
    sx = min(1.0, NARROW_MAX_W / max(bw, 1), sy)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    out = []
    for seq in contours:
        nseq = []
        for cmd in seq:
            if cmd[0] in ("M", "L"):
                nseq.append((cmd[0], 500 + (cmd[1] - cx) * sx,
                             440 + (cmd[2] - cy) * sy))
            elif cmd[0] == "C":
                nseq.append((cmd[0],
                             500 + (cmd[1] - cx) * sx, 440 + (cmd[2] - cy) * sy,
                             500 + (cmd[3] - cx) * sx, 440 + (cmd[4] - cy) * sy,
                             500 + (cmd[5] - cx) * sx, 440 + (cmd[6] - cy) * sy))
            else:
                nseq.append(cmd)
        out.append(nseq)
    return out


def main(photo: str, chars: str):
    gray = cv2.imdecode(np.fromfile(photo, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        sys.exit(f"无法读取照片: {photo}")
    inks = extract_groups(gray)
    chars = chars.strip()
    print(f"检出墨迹分组 {len(inks)} 个，字符表 {len(chars)} 字")
    if len(inks) != len(chars):
        sys.exit(f"分组数({len(inks)})与字符数({len(chars)})不一致，请检查照片")

    glyphs = {}
    for ch, ink in zip(chars, inks):
        contours = cell_to_font_contours(ink)
        if not contours:
            print(f"  [!] {ch!r} 矢量化失败，跳过")
            continue
        if ch in NARROW_CHARS:
            contours = scale_narrow(contours)
        glyphs[ch] = contours
        ys, xs = np.where(ink)
        print(f"  {ch!r}: 墨迹 {xs.max()-xs.min()}x{ys.max()-ys.min()}px → "
              f"{len(contours)} 轮廓" + ("（窄字形收窄）" if ch in NARROW_CHARS else ""))

    if not glyphs:
        sys.exit("无有效字形")
    replaced, failed = replace_glyphs(glyphs)
    print(f"已生效 {len(replaced)} 字: {''.join(replaced)}")
    if failed:
        print(f"失败: {''.join(failed)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
