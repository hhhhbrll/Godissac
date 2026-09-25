"""M4-3 v2 合并管线：云端生成位图 → 完整个人字体（修复版）。

修复三项用户反馈：
A. 结构校验过严 → 膨胀IoU + 跳过ASCII字母数字
B. 笔画粗细不统一 → 笔画宽度归一化（测量→膨胀/细化）
C. 字号偏小 → 字面框放满 + 渲染器基准字号调大（见renderer.py）
"""

import json
import sys
import zipfile
from pathlib import Path

sys.stdout = sys.stderr = open(
    Path(__file__).resolve().parent.parent / "output" / "m4" / "merge_run.log",
    "w", encoding="utf-8", buffering=1)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fontforge_lib.buildfont import build_ttf
from src.fontforge_lib.pipeline import (collect_glyphs, iou_qc, normalize_ink)
from src.fontforge_lib.segment import _cluster_components
from src.fontforge_lib.trace import cell_to_font_contours

# 云端生成结果包：优先 v2（多参考均值重生成，results_v2.zip），兼容旧版
_ZIP_CANDIDATES = [ROOT / "results_v2.zip", ROOT / "results.zip"]
ZIP = next((p for p in _ZIP_CANDIDATES if p.exists()), ROOT / "results.zip")
LXGW = ROOT / "fonts" / "LXGWWenKai-Regular.ttf"
TARGETS = ROOT / "output" / "m4" / "upload" / "target_chars.txt"
GEN_CP_DIR = ROOT / "output" / "m4" / "gen_cp"
OUT_FONT = ROOT / "output" / "myhand_full.ttf"
OUT_DIR = ROOT / "output"

# 膨胀IoU阈值（容忍细笔画偏移）：>=0.25 保留（生成字普遍偏移大，过严会误杀）
DIL_K = 5
DIL_GOOD, DIL_BAD = 0.40, 0.25


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
    """平均笔画宽度(px)：距离变换。宽 w 的矩形笔画线性剖面均值=w/4，
    故墨迹内距离均值×4 即平均笔画宽。对笔画交叉/端点有一致性偏置，
    但目标与测量用同一估计器，偏差相消。"""
    if not ink.any():
        return 0.0
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    return 4.0 * float(dist[ink].mean())


def measure_sw(ink: np.ndarray) -> float:
    """统一尺度下的笔画宽度：bbox 归一化(148/180)后测量，消除字大小/分辨率差异。"""
    return stroke_width_px(normalize_ink(ink))


def normalize_stroke_width(ink: np.ndarray, target_sw: float) -> np.ndarray:
    """把笔画宽度调整到目标值（距离变换测量 + 迭代膨胀/腐蚀）。

    v3 修正：旧版按"墨迹密度"归一化有系统性缺陷——密度=墨迹面积/bbox面积，
    与笔画数强相关：多笔画字(墨/壤)密度天然高被过度腐蚀变细，少笔画字
    (仆/升)密度低被膨胀变粗，字间粗细反而更不均。笔画宽度直接度量笔粗，
    与笔画数无关，才是与"同一支笔"对应的正确目标量。

    调整在 bbox 归一化后的统一尺度(148字面/180画布)上进行，与手写格子
    分辨率一致；返回归一化位图，直接矢量化（trace 会再归一化，无害）。
    """
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
            u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k, iterations=1)  # 防断笔
        else:
            break
    return u > 0


def load_gen_inks() -> dict[str, np.ndarray]:
    """云端生成位图 → {字符: 二值化墨迹图}。

    输入源优先级：
    1. results_v2.zip / results.zip（云端原始包，若存在）
    2. output/m4/gen_cp/ 目录（上次运行从 zip 解出的 U+XXXX.png，
       zip 被清理后仍可从缓存重跑）
    """
    inks: dict[str, np.ndarray] = {}
    GEN_CP_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP.exists():
        targets = set(TARGETS.read_text(encoding="utf-8").split())
        z = zipfile.ZipFile(ZIP)
        for info in z.infolist():
            stem = true_stem(info)
            if stem is None or len(stem) != 1:
                continue
            if stem not in targets:
                continue
            cp = ord(stem)
            if cp < 0x2E80:   # ASCII/标点：手动录入，跳过
                continue
            raw = z.read(info)
            (GEN_CP_DIR / f"U{cp:04X}.png").write_bytes(raw)
            ink = _decode_gen_png(raw)
            if ink is not None and ink.any():
                inks[stem] = ink
        return inks

    # zip 已清理 → 从 gen_cp 缓存目录直读
    print("  [提示] 云端zip不存在，从 gen_cp 缓存目录读取")
    for png in sorted(GEN_CP_DIR.glob("U*.png")):
        try:
            cp = int(png.stem[1:], 16)
        except ValueError:
            continue
        if cp < 0x2E80:
            continue
        ch = chr(cp)
        ink = _decode_gen_png(png.read_bytes())
        if ink is not None and ink.any():
            inks[ch] = ink
    return inks


def _decode_gen_png(raw: bytes) -> np.ndarray | None:
    """单个云端生成PNG → 二值墨迹图（2x放大+Otsu+去噪）。"""
    gray = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8),
                        cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (bw < 128).astype(bool)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, k_open) > 0
    ink = _cluster_components(ink, gap=30)
    return ink


def dilate_mask(mask: np.ndarray, k: int = DIL_K) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8),
                      np.ones((k, k), np.uint8), iterations=1) > 0


def struct_check_dilated(inks: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    """膨胀IoU：容错细笔画错位。"""
    font = ImageFont.truetype(str(LXGW), 100)
    results = []
    for ch, ink in inks.items():
        img = Image.new("L", (128, 128), 255)
        ImageDraw.Draw(img).text((64, 64), ch, font=font, fill=0, anchor="mm")
        ref = np.array(img) < 128
        a, b = normalize_ink(ink), normalize_ink(ref)
        ad, bd = dilate_mask(a), dilate_mask(b)
        union = np.logical_or(ad, bd).sum()
        iou = np.logical_and(ad, bd).sum() / union if union else 0.0
        results.append((ch, iou))
    return sorted(results, key=lambda t: t[1])


def collect_all_hand() -> tuple[dict[str, list], list[str], dict[str, np.ndarray]]:
    """合并所有手写批次：首批300 + 扩展700（samples/expansion_p*.jpg）+
    可选离群补录（samples/recollect_p*.jpg）。后批覆盖先批。"""
    batches: list[tuple[list[str], Path, Path]] = [
        (["samples/page1.jpg", "samples/page2.jpg"],
         ROOT / "samples" / "collect_chars.json",
         ROOT / "samples" / "collect_remap.json"),
    ]
    for prefix in ("expansion", "recollect"):
        # 兼容两种命名：expansion_p1.jpg 或直接 1.jpg（扩展批次恰好4页时）
        photos = sorted(str(p) for p in (ROOT / "samples").glob(f"{prefix}_p*.jpg"))
        if not photos and prefix == "expansion":
            cand = [ROOT / "samples" / f"{i}.jpg" for i in range(1, 5)]
            if all(p.exists() for p in cand):
                photos = [str(p) for p in cand]
        if photos:
            batches.append((photos,
                            ROOT / "samples" / f"{prefix}_chars.json",
                            ROOT / "samples" / f"{prefix}_remap.json"))
    glyphs: dict[str, list] = {}
    missing: list[str] = []
    cells: dict[str, np.ndarray] = {}
    for photos, chars_json, remap_json in batches:
        g, m, c = collect_glyphs(photos, chars_json, remap_json)
        print(f"  批次 {chars_json.stem}: {len(g)} 有效字形"
              + (f"，缺失 {len(m)} 字" if m else ""))
        glyphs.update(g)
        missing.extend(m)
        cells.update(c)
    return glyphs, missing, cells


def main():
    import pickle
    ckpt_input = ROOT / "output" / "m4" / "merge_input.pkl"
    ckpt_vecs = ROOT / "output" / "m4" / "merge_vecs.pkl"

    # 断点续跑①：步骤1-4结果存在则直接加载，跳过重算（沙箱杀进程后重启免重跑）
    if ckpt_input.exists():
        print("=== M4 v2 合并管线（续跑模式） ===")
        with open(ckpt_input, "rb") as f:
            st = pickle.load(f)
        gen_inks = st["gen_inks"]; hand_glyphs = st["hand_glyphs"]
        cells_by_char = st["cells_by_char"]; drop = st["drop"]
        scores = st["scores"]; low = st["low"]
        target_sw = st["target_sw"]
        before = np.array(st["before"]); after = np.array(st["after"])
        print(f"[恢复] 生成字 {len(gen_inks)} | 手写 {len(hand_glyphs)} | "
              f"丢弃 {len(drop)} | 待复核 {len(low)}")
    else:
        print("=== M4 v2 合并管线 ===\n[1] 提取云端生成图...")
        gen_inks = load_gen_inks()
        print(f"有效生成字形(CJK): {len(gen_inks)}")

        print("[2] 测手写字平均笔画宽度...")
        hand_glyphs, missing, cells_by_char = collect_all_hand()
        hand_sw = [measure_sw(cells_by_char[c])
                   for c in hand_glyphs if c in cells_by_char]
        hand_sw = [v for v in hand_sw if v > 0.5]
        target_sw = float(np.median(hand_sw))
        print(f"  手写字 {len(hand_sw)} 个有效样本，笔画宽度中位 {target_sw:.2f}px"
              f" (mean {np.mean(hand_sw):.2f})")

        print("[3] 生成字笔画宽度归一化...")
        before, after = [], []
        for ch, ink in gen_inks.items():
            before.append(measure_sw(ink))
            gen_inks[ch] = normalize_stroke_width(ink, target_sw)
            after.append(stroke_width_px(gen_inks[ch]))
        before, after = np.array(before), np.array(after)
        print(f"  归一化前: mean {before.mean():.2f} std {before.std():.2f}"
              f"  min {before.min():.2f} max {before.max():.2f}")
        print(f"  归一化后: mean {after.mean():.2f} std {after.std():.2f}"
              f"  min {after.min():.2f} max {after.max():.2f}")

        print("[4] 结构校验（膨胀IoU）...")
        scores = struct_check_dilated(gen_inks)
        vals = [s for _, s in scores]
        bad = [c for c, s in scores if s < DIL_BAD]
        low = [c for c, s in scores if DIL_BAD <= s < DIL_GOOD]
        print(f"  膨胀IoU: mean {np.mean(vals):.3f} med {np.median(vals):.3f}"
              f" min {min(vals):.3f}")
        print(f"  疑坏(<{DIL_BAD}): {len(bad)} 字")
        print(f"  待复核(<{DIL_GOOD}): {len(low)} 字")
        drop = set(bad)
        with open(ckpt_input, "wb") as f:
            pickle.dump({"gen_inks": gen_inks, "hand_glyphs": hand_glyphs,
                         "cells_by_char": cells_by_char, "drop": drop,
                         "scores": scores, "low": low, "target_sw": target_sw,
                         "before": before.tolist(), "after": after.tolist()}, f)
        print(f"  [存档] 步骤1-4状态 → {ckpt_input.name}")

    print("[5] 生成字矢量化+合并...")
    # 断点续跑②：矢量化进度（每300字存一次盘）
    vecs: dict[str, list] = {}
    vec_cells: dict[str, np.ndarray] = {}
    if ckpt_vecs.exists():
        with open(ckpt_vecs, "rb") as f:
            st = pickle.load(f)
        vecs, vec_cells = st["vecs"], st["vec_cells"]
        print(f"  [续跑] 已矢量化 {len(vecs)} 字")
    glyphs = dict(hand_glyphs)
    glyphs.update(vecs)
    cells_by_char.update(vec_cells)
    n_new = 0
    for n_done, (ch, ink) in enumerate(gen_inks.items(), 1):
        if ch in drop or ch in glyphs:
            continue
        contours = cell_to_font_contours(ink)
        if contours:
            vecs[ch] = contours
            vec_cells[ch] = ink
            glyphs[ch] = contours
            cells_by_char[ch] = ink
            n_new += 1
        if n_new and n_new % 300 == 0:
            with open(ckpt_vecs, "wb") as f:
                pickle.dump({"vecs": vecs, "vec_cells": vec_cells}, f)
            print(f"  进度 {len(vecs)}/{len(gen_inks) - len(drop)}（已存盘）")
    with open(ckpt_vecs, "wb") as f:
        pickle.dump({"vecs": vecs, "vec_cells": vec_cells}, f)
    gen_count = len(glyphs) - len(hand_glyphs)
    print(f"合并总计 {len(glyphs)} 字（手写 {len(hand_glyphs)} + 生成 {gen_count}）")

    print("[6] 构建TTF+质检...")
    build_ttf(glyphs, OUT_FONT, reverse=False)
    iou = iou_qc(OUT_FONT, glyphs, cells_by_char, sample=60)
    print(f"整库 IoU 抽样60: {iou:.3f}")

    # 真迹字清单（export_m4 据此渲染参考图、计算生成目标缺口）
    (ROOT / "output" / "m4" / "handwritten_chars.json").write_text(
        json.dumps({"chars": sorted(hand_glyphs)}, ensure_ascii=False, indent=0),
        encoding="utf-8")

    report = {
        "gen_valid": len(gen_inks),
        "gen_dropped_bad": sorted(drop),
        "gen_review_low": low,
        "hand_count": len(hand_glyphs),
        "total": len(glyphs),
        "stroke_width_target": round(target_sw, 2),
        "stroke_width_before": {"mean": round(float(before.mean()), 2),
                                "std": round(float(before.std()), 2)},
        "stroke_width_after": {"mean": round(float(after.mean()), 2),
                               "std": round(float(after.std()), 2)},
        "iou_sample": round(iou, 3),
    }
    (ROOT / "output" / "m4" / "merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    preview(scores, gen_inks, drop)
    print(f"\n完成: {OUT_FONT}")

    # 全部成功，清理断点存档（下次重跑从头开始）
    for p in (ckpt_input, ckpt_vecs):
        if p.exists():
            p.unlink()
            print(f"  [清理] {p.name}")


def preview(scores, gen_inks, drop):
    import pymupdf as fitz
    doc = fitz.open()
    page = doc.new_page(width=842, height=595)
    page.insert_text((40, 40), "M4 v2 随机48个生成字（笔画宽度归一化后）",
                     fontsize=12, fontname="F", fontfile=str(LXGW))
    rng = np.random.default_rng(7)
    pool = [c for c, _ in scores if c not in drop]
    picked = rng.choice(pool, size=min(48, len(pool)), replace=False)
    text = "".join(picked)
    for i in range(0, len(text), 24):
        page.insert_text((40, 100 + (i // 24) * 40), text[i:i + 24],
                         fontsize=30, fontname="H", fontfile=str(OUT_FONT))
    doc[0].get_pixmap(dpi=150).save(OUT_DIR / "m4_full_质检.png")


if __name__ == "__main__":
    main()
