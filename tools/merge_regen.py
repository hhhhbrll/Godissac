"""M4-8 应用重生成结果：results_regen.zip → 就地替换/新增 myhand_full.ttf。

与 merge_m4 主管线的区别：不做结构校验丢弃——这些字当前回退文楷（最差
状态），任何生成版本都是风格一致性提升；但逐字报告膨胀IoU供人工复核，
明显异常(<0.15)的字会在日志中列出。

笔画宽度归一化与主管线同度量（距离变换，目标值读自 merge_report.json）。

用法：
    python tools/merge_regen.py results_regen.zip
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # 引入 tools 内模块

# 内建日志（PowerShell 终端会吞 UTF-8 输出）
(ROOT / "output" / "m4").mkdir(parents=True, exist_ok=True)
_log = open(ROOT / "output" / "m4" / "merge_regen.log", "w", encoding="utf-8", buffering=1)
_real_print = print
def print(*a, **k):  # noqa: F811
    _real_print(*a, **k, file=_log)
    try:
        _real_print(*a, **k)
    except UnicodeEncodeError:
        pass

from src.fontforge_lib.segment import _cluster_components
from src.fontforge_lib.trace import cell_to_font_contours
from src.fontforge_lib.pipeline import normalize_ink
from merge_recollect import FONT, OUT_DIR, replace_glyphs  # 就地字形手术

REPORT = ROOT / "output" / "m4" / "merge_report.json"
BACKUP = OUT_DIR / "myhand_full_备份_regen.ttf"
LXGW = ROOT / "fonts" / "LXGWWenKai-Regular.ttf"

DIL_K = 5


def true_stem(info) -> str | None:
    name = info.filename
    if not name.endswith(".png"):
        return None
    if info.flag_bits & 0x800:
        return Path(name).stem
    try:
        return Path(name.encode("cp437").decode("utf-8")).stem
    except (UnicodeDecodeError, UnicodeEncodeError):
        return Path(name).stem


def stroke_width_px(ink: np.ndarray) -> float:
    if not ink.any():
        return 0.0
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    return 4.0 * float(dist[ink].mean())


def normalize_stroke_width(ink: np.ndarray, target_sw: float) -> np.ndarray:
    """与 merge_m4 v3 相同：距离变换测笔画宽 + 迭代膨胀/腐蚀。"""
    if not ink.any() or target_sw <= 0:
        return ink
    norm = normalize_ink(ink)
    u = norm.astype(np.uint8)
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


def load_regen_inks(zip_path: Path) -> dict[str, np.ndarray]:
    """zip → {字符: 二值化墨迹图}（与 merge_m4.load_gen_inks 相同处理）。"""
    z = zipfile.ZipFile(zip_path)
    inks: dict[str, np.ndarray] = {}
    for info in z.infolist():
        stem = true_stem(info)
        if stem is None or len(stem) != 1 or not ("\u4e00" <= stem <= "\u9fff"):
            continue
        raw = z.read(info)
        gray = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  [损坏] {stem} PNG解码失败，跳过")
            continue
        big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = (bw < 128).astype(bool)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, k_open) > 0
        ink = _cluster_components(ink, gap=30)
        if ink.any():
            inks[stem] = ink
    return inks


def dilated_iou(ink: np.ndarray, ch: str) -> float:
    """膨胀IoU（对照文楷渲染）——仅报告用，不丢弃。"""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(LXGW), 100)
    img = Image.new("L", (128, 128), 255)
    ImageDraw.Draw(img).text((64, 64), ch, font=font, fill=0, anchor="mm")
    ref = np.array(img) < 128
    a, b = normalize_ink(ink), normalize_ink(ref)
    k = np.ones((DIL_K, DIL_K), np.uint8)
    ad = cv2.dilate(a.astype(np.uint8), k) > 0
    bd = cv2.dilate(b.astype(np.uint8), k) > 0
    union = np.logical_or(ad, bd).sum()
    return float(np.logical_and(ad, bd).sum() / union) if union else 0.0


def main(zip_path: str):
    zp = Path(zip_path)
    if not zp.exists():
        sys.exit(f"未找到 {zp}")
    if not FONT.exists():
        sys.exit(f"缺少字库 {FONT}")
    if not REPORT.exists():
        sys.exit("缺少 merge_report.json（笔画宽度目标值来源）")
    target_sw = json.loads(REPORT.read_text(encoding="utf-8")).get(
        "stroke_width_target", 0)
    if target_sw <= 0:
        sys.exit("merge_report.json 无 stroke_width_target，请先跑 v3 版 merge_m4.py")

    print("=== M4-8 应用重生成结果 ===")
    print(f"[1] 提取 {zp.name}...")
    inks = load_regen_inks(zp)
    print(f"  有效字形 {len(inks)}")
    if not inks:
        sys.exit("无有效字形，中止")

    print(f"[2] 笔画宽度归一化（目标 {target_sw:.2f}px）...")
    for ch, ink in inks.items():
        inks[ch] = normalize_stroke_width(ink, target_sw)

    print("[3] 矢量化 + 膨胀IoU复核（仅报告，不丢弃）...")
    # 断点续跑：矢量化进度存盘（沙箱对长进程有 ~10 分钟限制）
    import pickle
    ckpt = OUT_DIR / "regen_vecs.pkl"
    glyphs = {}
    ious: dict[str, float] = {}
    if ckpt.exists():
        with open(ckpt, "rb") as f:
            st = pickle.load(f)
        glyphs, ious = st["glyphs"], st["ious"]
        print(f"  [续跑] 已矢量化 {len(glyphs)} 字")
    n_new = 0
    for ch, ink in inks.items():
        if ch in glyphs:
            continue
        contours = cell_to_font_contours(ink)
        if contours:
            glyphs[ch] = contours
            ious[ch] = dilated_iou(ink, ch)
            n_new += 1
        if n_new and n_new % 300 == 0:
            with open(ckpt, "wb") as f:
                pickle.dump({"glyphs": glyphs, "ious": ious}, f)
            print(f"  进度 {len(glyphs)}/{len(inks)}（已存盘）")
    with open(ckpt, "wb") as f:
        pickle.dump({"glyphs": glyphs, "ious": ious}, f)
    print(f"  可用字形 {len(glyphs)}/{len(inks)}")
    vals = np.array(list(ious.values()))
    if len(vals):
        print(f"  膨胀IoU: mean {vals.mean():.3f} med {float(np.median(vals)):.3f}"
              f" min {vals.min():.3f}")
    suspect = sorted(ch for ch, v in ious.items() if v < 0.15)
    if suspect:
        print(f"  [!] 明显异常(<0.15) {len(suspect)} 字，建议后续手写补录: "
              f"{''.join(suspect)}")

    print("[4] 备份并就地替换字形...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FONT, BACKUP)
    replaced, failed = replace_glyphs(glyphs)
    print(f"  已生效 {len(replaced)} 字（备份: {BACKUP.name}）")
    if failed:
        print(f"  [!] 失败: {''.join(failed)}")

    print(f"\n完成: {FONT}")
    if ckpt.exists():
        ckpt.unlink()
        print("  [清理] regen_vecs.pkl")
    print("后续: python tools/closed_loop_test.py 重跑闭环渲染")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
