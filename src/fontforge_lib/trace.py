"""墨迹位图 → 字体轮廓（potrace 纯Python版）。

potracer 坐标约定（实验标定）：
- 输入数组 True=白/背景，False=墨（注意与我们的墨迹bool图相反）
- 输出坐标 (x=列, y=行)，图像坐标系 y 向下
- BezierSegment 有 c1/c2/end_point；CornerSegment 只有 end_point

归一化映射（中文字体标准做法）：
定义字面框（em 内文字实际填充的正方形区域）：
- 水平 [MARGIN, EM-MARGIN]，中心在 EM/2=500
- 垂直 [BASE_GAP, ASCENT-MARGIN]，即 baseline 上方留少量边距到 ascent 下方留边距
- 字面框中心 y = (BASE_GAP + ASCENT-MARGIN)/2 ≈ 450（视觉中心偏上，非数学中心 380）
墨迹 bbox 等比缩放至装进字面框（最长边对齐），在框内水平+垂直居中。
所有字形严格在 ascent 区域内，不穿越 baseline，避免"字往下掉"。

底部噪声剔除：采集时常有底栏横线/污点（用户手抄时残留、拍照时光斑），
会让 bbox 上下被拉长，等比缩放后字偏小。先用连通域去噪，再限制底部墨迹不能
离字符主体太远（否则视为底栏剔除）。
"""

from __future__ import annotations

import cv2
import numpy as np
from potrace import Bitmap as PotaBitmap

from .template import ASCENT, EM

PAD = 4            # potrace 边界留白
MARGIN = 8         # 字身边距（左右/顶部留白，小=字大）
BASE_GAP = 8       # 基线留白（底部少量间隙避免笔画贴边）
FILL_RATIO = 1.0   # 字面框填充率（填满，不留余量）

# 字面框边界
_BOX_L = MARGIN
_BOX_R = EM - MARGIN
_BOX_B = BASE_GAP
_BOX_T = ASCENT - MARGIN
_BOX_W = _BOX_R - _BOX_L
_BOX_H = _BOX_T - _BOX_B
_BOX_CX = (_BOX_L + _BOX_R) / 2
_BOX_CY = (_BOX_B + _BOX_T) / 2  # 视觉中心，约 445


def _clean_bottom_noise(cell_ink: np.ndarray) -> np.ndarray:
    """剔除采集时残留的底栏墨迹/污点。

    启发式：
    1. 取最下方 15% 行的墨迹像素集合
    2. 如果最下方有"明显孤立"的小连通域（与上方字符不连，且宽度横跨大），
       视为底栏横线剔除
    3. 否则保留（可能是字符本来的底部笔画如"长"、"上"）

    返回清理后的墨迹图（与输入相同 shape/dtype）。
    """
    h, w = cell_ink.shape
    if h < 20:
        return cell_ink
    bottom_strip = cell_ink[int(h*0.85):, :]
    # 1) 整个底栏都是一条横线的情况（横向连通域宽度 > 70% 单元格宽，且不在上方）
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bottom_strip.astype(np.uint8), connectivity=8
    )
    if n_labels <= 1:
        return cell_ink
    # 取最大连通域
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return cell_ink
    largest_label = 1 + int(areas.argmax())
    largest_w = stats[largest_label, cv2.CC_STAT_WIDTH]
    largest_h = stats[largest_label, cv2.CC_STAT_HEIGHT]
    largest_area = stats[largest_label, cv2.CC_STAT_AREA]
    # 条件：宽度 ≥ 70% cell 宽 AND 高度 ≤ 3px AND 面积 ≤ 宽度×2（典型横线特征）
    if largest_w >= 0.7 * w and largest_h <= 3 and largest_area <= largest_w * 2:
        # 检查这个底栏是否与上方主字符相连
        above = cell_ink[:int(h*0.85), :]
        n2, _, stats2, _ = cv2.connectedComponentsWithStats(
            above.astype(np.uint8), connectivity=8
        )
        # 主连通域 = 上面部分面积最大的那个
        if n2 > 1:
            main_above_area = max(stats2[1:, cv2.CC_STAT_AREA])
            # 底栏连通域面积 / 上方主字符面积 < 0.3 → 视为底噪剔除
            if largest_area < 0.3 * main_above_area:
                cleaned = cell_ink.copy()
                cleaned[int(h*0.85):, :][labels == largest_label] = False
                return cleaned
    return cell_ink


def cell_to_font_contours(cell_ink: np.ndarray) -> list[list[tuple]]:
    """墨迹bool图(True=墨) → 归一化后的字体轮廓命令列表。

    每条轮廓为命令序列：("M",x,y) / ("L",x,y) / ("C",c1x,c1y,c2x,c2y,x,y) / ("Z",)
    坐标为 font units（y 向上）。
    """
    cell_ink = _clean_bottom_noise(cell_ink)
    ys, xs = np.where(cell_ink)
    if len(xs) == 0:
        return []

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1

    # 等比缩放：使墨迹 bbox 装进字面框（最长边对齐，FILL_RATIO 留余量）
    s = FILL_RATIO * min(_BOX_W / max(bw, 1), _BOX_H / max(bh, 1))
    # 墨迹 bbox 中心 → 字面框中心（视觉中心偏上）
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    def fx(px: float) -> float:
        return (px - cx) * s + _BOX_CX

    def fy(py: float) -> float:
        # 图像 y 向下，字体 y 向上；墨迹中心映射到字面框视觉中心
        return _BOX_CY - (py - cy) * s

    # potracer: False=墨；四周补白(True)
    arr = np.pad(~cell_ink, PAD, constant_values=True)
    path = PotaBitmap(arr).trace()

    contours: list[list[tuple]] = []
    for curve in path:
        cmds: list[tuple] = []
        sp = curve.start_point
        cmds.append(("M", fx(sp.x - PAD), fy(sp.y - PAD)))
        for seg in curve.segments:
            ep = seg.end_point
            if hasattr(seg, "c1"):
                cmds.append(("C",
                             fx(seg.c1.x - PAD), fy(seg.c1.y - PAD),
                             fx(seg.c2.x - PAD), fy(seg.c2.y - PAD),
                             fx(ep.x - PAD), fy(ep.y - PAD)))
            else:
                cmds.append(("L", fx(ep.x - PAD), fy(ep.y - PAD)))
        cmds.append(("Z",))
        contours.append(cmds)
    return contours
