"""照片→标准页面→格子墨迹。

管线：定位四角锚点 → 透视校正到模板坐标系 → 按已知网格切格
→ 墨迹/印刷灰分离 → 连通域去噪。

坐标契约：所有输出均为模板坐标系（pt×SCALE，见 template.py）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .template import (
    A4_H, A4_W, CELL, CELLS_PER_PAGE, CELL_PX, COLS, GRID_LEFT, GRID_TOP,
    MARKER, PAGE_PX, ROWS, SCALE, marker_centers_pt,
)


def load_gray(path: str | Path) -> np.ndarray:
    """读图为灰度图（Windows 中文路径安全：np.fromfile + imdecode）。"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def find_marker_centers(gray: np.ndarray) -> np.ndarray:
    """检测四角锚点中心（照片像素坐标）。顺序：左上、右上、左下、右下。

    策略：每个角只在该角外扩 25% 的象限带内选"最近的黑组件"，
    组件不可复用；最终做四边形凸性/朝向校验。

    阴影鲁棒：若某角在默认阈值 90 下无候选，依次降阈（70/55/40）重试
    ——拍摄阴影可把右下角纸底压到 120 以下，锚点组件随之变暗，
    降阈后锚点仍是最黑最大且长宽比规整的方块，可与阴影底区分。
    """
    h, w = gray.shape

    def _find_at(threshold: int) -> np.ndarray | None:
        dark = (gray < threshold).astype(np.uint8)
        # 轻度闭运算连接锚点内部可能的反光断裂
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dark, connectivity=8)

        img_area = h * w
        min_area = max(200.0, 0.00008 * img_area)  # 排除噪点，但容忍透视缩放

        quads = [  # (角点, 质心允许区间)
            ((0, 0), lambda cx, cy: cx < 0.25 * w and cy < 0.25 * h),
            ((w, 0), lambda cx, cy: cx > 0.75 * w and cy < 0.25 * h),
            ((0, h), lambda cx, cy: cx < 0.25 * w and cy > 0.75 * h),
            ((w, h), lambda cx, cy: cx > 0.75 * w and cy > 0.75 * h),
        ]
        used_labels: set[int] = set()
        centers = []
        for (cx, cy), in_band in quads:
            best, best_d, best_label = None, np.inf, -1
            for i in range(1, n):
                if i in used_labels:
                    continue
                x, y, bw, bh, area = stats[i]
                if area < min_area or area > 0.05 * img_area:
                    continue
                if not 0.4 <= bw / max(bh, 1) <= 2.5:
                    continue
                if not in_band(centroids[i][0], centroids[i][1]):
                    continue
                # 只认"很黑"的组件（锚点是纯黑实心）
                mx = int(np.clip(centroids[i][0], 0, w - 1))
                my = int(np.clip(centroids[i][1], 0, h - 1))
                if gray[my, mx] > threshold:
                    continue
                d = np.hypot(centroids[i][0] - cx, centroids[i][1] - cy)
                if d < best_d:
                    best_d, best, best_label = d, centroids[i], i
            if best is None:
                return None   # 本阈值下有角缺锚点 → 整体重试更低阈值
            used_labels.add(best_label)
            centers.append(best)
        try:
            _validate_quad(np.float32(centers), w, h)
        except RuntimeError:
            return None
        return np.float32(centers)

    for th in (90, 70, 55, 40):
        centers = _find_at(th)
        if centers is not None:
            if th != 90:
                print(f"  [锚点] 默认阈值无解，降阈 {th} 检出（拍摄阴影）")
            return centers
    raise RuntimeError("四角锚点检测失败：默认与降阈均无有效候选，"
                       "请检查四角锚点是否拍全/被遮挡")


def _validate_quad(pts: np.ndarray, w: int, h: int) -> None:
    """校验四点构成凸四边形且顺序正确（TL,TR,BL,BR）。"""
    tl, tr, bl, br = pts
    # 顺序正确：TL 在上、BL 在下……即 tl.y < bl.y, tr.y < br.y, tl.x < tr.x
    if not (tl[1] < bl[1] and tr[1] < br[1] and tl[0] < tr[0] and bl[0] < br[0]):
        raise RuntimeError(f"锚点顺序异常: {pts.tolist()}")
    # 凸性：边向量叉积同号
    order = [tl, tr, br, bl]
    crosses = []
    for i in range(4):
        a, b, c = order[i], order[(i + 1) % 4], order[(i + 2) % 4]
        v1, v2 = b - a, c - b
        crosses.append(float(v1[0] * v2[1] - v1[1] * v2[0]))
    if not (all(c > 0 for c in crosses) or all(c < 0 for c in crosses)):
        raise RuntimeError(f"锚点四边形非凸: {pts.tolist()}")
    # 边长均衡：长边/短边 < 4（页面宽高比 1.41，加畸变余量）
    sides = [float(np.linalg.norm(order[(i + 1) % 4] - order[i])) for i in range(4)]
    if max(sides) / max(min(sides), 1) > 4:
        raise RuntimeError(f"锚点四边形边长失衡: {sides}")


def warp_to_template(gray: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """透视校正到标准页面尺寸 PAGE_PX。界外区域填白（默认填黑会伪装成墨迹）。"""
    dst = np.float32([(x * SCALE, y * SCALE) for x, y in marker_centers_pt()])
    M = cv2.getPerspectiveTransform(centers.astype(np.float32), dst)
    return cv2.warpPerspective(gray, M, PAGE_PX, flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def flatten_illumination(warped: np.ndarray) -> np.ndarray:
    """背景光照归一化：大核闭运算估计纸张亮度场（低频阴影），逐像素相除压平。

    解决拍照阴影导致的整格误判墨迹：阴影区纸底可跌至 105-130，
    绝对阈值 gray<130 会把纸当墨。归一化后纸底恢复均匀高亮，
    墨迹/格线作为高频细节保持相对对比度，原双阈值逻辑继续适用。
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121))
    bg = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, k)
    return cv2.divide(warped, bg, scale=255)


def extract_cells(warped: np.ndarray, ratio: float = 0.62, abs_ink: float = 130.0,
                  min_area: int = 12) -> dict[tuple[int, int], np.ndarray]:
    """切格 + 墨迹分离 + 去噪 + 连通域聚类。返回 {(row,col): 墨迹bool图(180×180)}。

    先做光照归一化（压平阴影），墨迹判定双条件（满足其一即墨）：
    - 相对阈值：norm < 格子纸底p70 × ratio —— 跟随局部光照，抓正常笔迹
    - 绝对黑度：norm < abs_ink —— 兜底抓"下笔极轻"的浅字迹
    （归一化后纸底≈255、格线≈190+，abs_ink=130 仍与格线保持安全距离）

    连通域聚类：保留最大组件及其邻近组件（汉字的多点结构如"心""小"），
    丢弃远离主体的散点（小灰字残留/拍照噪点），防止 bbox 被撑大导致字形压扁。
    """
    warped = flatten_illumination(warped)
    page_paper = float(np.percentile(warped, 95))
    cells = {}
    for r in range(ROWS):
        for c in range(COLS):
            x = int((GRID_LEFT + c * CELL) * SCALE)
            y = int((GRID_TOP + r * CELL) * SCALE)
            cell_raw = warped[y : y + CELL_PX, x : x + CELL_PX]
            paper = max(float(np.percentile(cell_raw, 70)), page_paper * 0.8)
            cell = (cell_raw < paper * ratio) | (cell_raw < abs_ink)
            # 连通域去噪（先去极小噪点）
            n, labels, stats, _ = cv2.connectedComponentsWithStats(
                cell.astype(np.uint8), connectivity=8)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    cell[labels == i] = False
            # 连通域聚类：以最大组件为种子，合并邻近组件（距离<18px≈格子10%）
            cell = _cluster_components(cell, gap=18)
            cells[(r, c)] = cell
    return cells


def _cluster_components(mask: np.ndarray, gap: int = 25) -> np.ndarray:
    """连通域聚类：保留主体组件及其邻近/大面积组件，丢弃远离主体的散点。

    策略：
    - 最大连通域作为种子（主体）
    - 面积 >= 主体面积 3% 的组件一律保留（大组件不可能是噪点/小灰字残留）
    - 小组件：与任一已保留组件 bbox 距离 < gap 才纳入（属于该字笔画）
    - 极小组件（面积 < 主体 0.5%）即使邻近也丢弃（拍照噪点）

    实现为 numpy 向量化（组件两两距离矩阵 + 迭代传播），避免
    Python 多重循环——M4 批量处理 3286 张扩散生成图时为性能关键。
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    comps = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    comps.sort(key=lambda t: -t[1])
    main_area = comps[0][1]
    # 大组件(≥3%)直接保留；极小(<0.5%)直接丢弃；中等组件看邻近关系
    cand = [i for i, area in comps if area >= main_area * 0.005]
    big = {i for i, area in comps if area >= main_area * 0.03}
    big.add(comps[0][0])  # 种子：最大组件必留

    # 所有候选组件（含大组件）的两两 bbox 间隙矩阵
    boxes = np.array([[stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                       stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH],
                       stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]]
                      for i in cand], dtype=np.float64)  # [x0,y0,x1,y1]
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    dx = np.maximum(0, np.maximum(x0[None, :] - x1[:, None],
                                  x0[:, None] - x1[None, :]))
    dy = np.maximum(0, np.maximum(y0[None, :] - y1[:, None],
                                  y0[:, None] - y1[None, :]))
    near = np.hypot(dx, dy) < gap                  # (m, m) bool

    # 迭代传播：初始保留=大组件，与保留集相邻的中等组件逐轮纳入
    keep = np.array([i in big for i in cand], dtype=bool)
    while True:
        new = keep | near[:, keep].any(axis=1)
        if (new == keep).all():
            break
        keep = new

    out = np.zeros_like(mask)
    for k in range(len(cand)):
        if keep[k]:
            out[labels == cand[k]] = True
    return out


def _bbox_distance(a: tuple, b: tuple) -> float:
    """两矩形 bbox 的最短距离（重叠/接触返回 0）。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    # 水平距离
    dx = max(0, max(ax - (bx + bw), bx - (ax + aw)))
    dy = max(0, max(ay - (by + bh), by - (ay + ah)))
    return float(np.hypot(dx, dy))


def page_cells(photo_path: str | Path, verbose: bool = True) -> dict[tuple[int, int], np.ndarray]:
    """单张照片 → 格子墨迹字典。"""
    gray = load_gray(photo_path)
    centers = find_marker_centers(gray)
    warped = warp_to_template(gray, centers)
    cells = extract_cells(warped)
    if verbose:
        filled = sum(1 for v in cells.values() if v.any())
        print(f"  {Path(photo_path).name}: 锚点✓ 校正✓ 非空格子 {filled}/{CELLS_PER_PAGE}")
    return cells
