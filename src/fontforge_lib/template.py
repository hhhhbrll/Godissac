"""采集模板几何常量（pt）——生成端与分割端共享的唯一真源。

任何一端改动此文件，另一端自动同步。
"""

A4_W, A4_H = 595.0, 842.0

# 网格：12列 × 16行，每页 192 格
COLS, ROWS = 12, 16
CELL = 45.0
GRID_W, GRID_H = COLS * CELL, ROWS * CELL   # 540 × 720
GRID_LEFT = (A4_W - GRID_W) / 2             # 27.5
GRID_TOP = 90.0                             # 页眉区之后
CELLS_PER_PAGE = COLS * ROWS                # 192

# 四角定位锚点（黑色实心方块，透视校正用）
MARKER = 16.0        # 边长
MARKER_INSET = 14.0  # 距页缘


def marker_centers_pt() -> list[tuple[float, float]]:
    """四角锚点中心坐标（pt），顺序：左上、右上、左下、右下。"""
    x = MARKER_INSET + MARKER / 2
    y = MARKER_INSET + MARKER / 2
    return [
        (x, y),
        (A4_W - x, y),
        (x, A4_H - y),
        (A4_W - x, A4_H - y),
    ]


# 透视校正后的标准页面尺寸（px/pt）
SCALE = 4.0
PAGE_PX = (int(A4_W * SCALE), int(A4_H * SCALE))
CELL_PX = int(CELL * SCALE)  # 180

# 字体度量（font units, em=1000）
EM = 1000
ASCENT = 880
DESCENT = -120
