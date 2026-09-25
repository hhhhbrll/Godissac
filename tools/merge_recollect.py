"""M4-5 手写补录合并：手写照片 → 就地替换 myhand_full.ttf 中的对应字形。

用法：
    python tools/merge_recollect.py 照片1.jpg 照片2.jpg ... [--chars samples/xxx_chars.json]

批次由 --chars 决定（默认 samples/recollect_chars.json，即227字离群补录表）：
    --chars samples/recollect_chars.json   227字离群补录（路线：最小补录）
    --chars samples/expansion_chars.json   700字扩展采集（路线A：top1000全覆盖）

配套文件（均按前缀自动推导，缺省可不存在）：
    samples/{prefix}_chars.json    格子→汉字映射（make_collect_sheet.py 生成）
    samples/{prefix}_remap.json    写错字登记（格式同 collect_remap.json）：
        {"void": [[1,3,5]], "extra": {"2,16,3": "鹏"}}

流程：
    [1] 照片切格+矢量化 → 补录字形（100% 真笔迹，不做任何笔画归一化）
    [2] 备份字库 → 就地替换对应字形（优先级：后批手写 > 先前手写/AI生成）
    [3] IoU 质检（矢量保真度）
    [4] 生成替换前后对比图 output/m4/{prefix}_对比.png
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 内建日志（PowerShell 终端会吞 UTF-8 输出）
(ROOT / "output" / "m4").mkdir(parents=True, exist_ok=True)
_log = open(ROOT / "output" / "m4" / "merge_recollect.log", "w", encoding="utf-8", buffering=1)
_real_print = print
def print(*a, **k):  # noqa: F811
    _real_print(*a, **k, file=_log)
    try:
        _real_print(*a, **k)
    except UnicodeEncodeError:
        pass

import numpy as np
import pymupdf as fitz
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fontforge_lib.pipeline import collect_glyphs, glyph_bitmap, normalize_ink

FONT = ROOT / "output" / "myhand_full.ttf"
CHARS_JSON = ROOT / "samples" / "recollect_chars.json"
OUT_DIR = ROOT / "output" / "m4"
LXGW = ROOT / "fonts" / "LXGWWenKai-Regular.ttf"

IOU_BAD = 0.75   # 单字保真度低于此值 → 建议重写或重拍


def draw_contours(pen, contours: list[list[tuple]]):
    for seq in contours:
        for cmd in seq:
            if cmd[0] == "M":
                pen.moveTo((cmd[1], cmd[2]))
            elif cmd[0] == "L":
                pen.lineTo((cmd[1], cmd[2]))
            elif cmd[0] == "C":
                pen.curveTo((cmd[1], cmd[2]), (cmd[3], cmd[4]), (cmd[5], cmd[6]))
            elif cmd[0] == "Z":
                pen.closePath()


def replace_glyphs(glyphs: dict[str, list]) -> tuple[list[str], list[str]]:
    """就地替换 myhand_full.ttf 中的字形；字库缺字则新增（字库曾被结构校验
    丢弃部分生成字，手写补录正好补上）。返回 (生效字, 失败字)。"""
    font = TTFont(str(FONT), lazy=False)
    cmap = font.getBestCmap()
    order = set(font.getGlyphOrder())
    glyf, hmtx = font["glyf"], font["hmtx"]
    replaced, failed = [], []
    for ch, contours in glyphs.items():
        gname = cmap.get(ord(ch))
        exists = gname is not None and gname in order
        if not exists:
            gname = f"uni{ord(ch):04X}"
        tt = TTGlyphPen(None)
        cu = Cu2QuPen(tt, max_err=1.5, reverse_direction=False)
        draw_contours(cu, contours)
        try:
            glyph = tt.glyph()
        except Exception:
            failed.append(ch)
            continue
        if not exists:   # 新增：登记字形顺序 + cmap
            font.glyphOrder.append(gname)
            for sub in font["cmap"].tables:
                if sub.isUnicode():
                    sub.cmap[ord(ch)] = gname
            cmap[ord(ch)] = gname
            order.add(gname)
        glyf.glyphs[gname] = glyph
        # lsb 应等于字形 xMin（TrueType 规范，否则渲染器平移字形）
        xs = [cmd[1] for seq in contours for cmd in seq if cmd[0] in ("M", "L")]
        xs += [cmd[5] for seq in contours for cmd in seq if cmd[0] == "C"]
        lsb = int(round(min(xs))) if xs else 0
        adv = hmtx.metrics.get(gname, (1000, 0))[0]
        hmtx.metrics[gname] = (adv, lsb)
        replaced.append(ch)
    font["maxp"].numGlyphs = len(glyf.glyphs)
    font.save(str(FONT))
    return replaced, failed


def qc(replaced: list[str], cells_by_char: dict) -> list[str]:
    """逐字 IoU：新字库渲染回位图 vs 照片原始墨迹（均 bbox 归一化）。"""
    ious = {}
    for ch in replaced:
        a = normalize_ink(glyph_bitmap(FONT, ch))
        b = normalize_ink(cells_by_char[ch])
        union = np.logical_or(a, b).sum()
        ious[ch] = float(np.logical_and(a, b).sum() / union) if union else 0.0
    vals = np.array(list(ious.values()))
    bad = sorted(ch for ch, v in ious.items() if v < IOU_BAD)
    print(f"  补录字 IoU: mean {vals.mean():.3f} med {float(np.median(vals)):.3f}"
          f" min {vals.min():.3f}")
    if bad:
        print(f"  [!] 低保真(<{IOU_BAD}) {len(bad)} 字，建议重写重拍: {''.join(bad)}")
    return bad


def make_compare(replaced: list[str], backup: Path, prefix: str):
    """对比图：每行上排=替换前、下排=替换后，同列同字。"""
    priority = [c for c in "升评仆" if c in replaced]
    ordered = priority + [c for c in replaced if c not in priority]
    doc = fitz.open()
    page, y = None, 0
    per_line, size = 16, 26
    for i0 in range(0, len(ordered), per_line):
        if page is None or y > 740:
            page = doc.new_page(width=595, height=842)
            page.insert_text((40, 52), "上排 = 替换前（AI生成）    下排 = 替换后（本人手写补录）",
                             fontsize=13, fontname="F", fontfile=str(LXGW))
            y = 105
        line = "".join(ordered[i0:i0 + per_line])
        page.insert_text((40, y), line, fontsize=size, fontname="O", fontfile=str(backup))
        page.insert_text((40, y + size + 14), line, fontsize=size, fontname="N", fontfile=str(FONT))
        y += size * 2 + 54
    doc.save(OUT_DIR / f"{prefix}_对比.pdf", garbage=4, deflate=True)
    doc[0].get_pixmap(dpi=150).save(OUT_DIR / f"{prefix}_对比.png")


def main(photos: list[str], chars_json: Path):
    prefix = chars_json.stem.removesuffix("_chars")   # recollect / expansion / ...
    remap_json = chars_json.parent / f"{prefix}_remap.json"
    backup = OUT_DIR / f"myhand_full_备份_{prefix}.ttf"

    print("=== M4-5 手写补录合并 ===")
    if not chars_json.exists():
        raise SystemExit(f"缺少批次清单 {chars_json}，先用 make_collect_sheet.py 生成采集表")
    if not FONT.exists():
        raise SystemExit(f"缺少字库 {FONT}，请先完成 M4 合并管线")

    print(f"[1] 照片切格 + 矢量化（批次: {prefix}）...")
    glyphs, missing, cells_by_char = collect_glyphs(photos, chars_json, remap_json)
    total = len(json.loads(chars_json.read_text(encoding="utf-8"))["chars"])
    print(f"  有效补录字形 {len(glyphs)}/{total}")
    if missing:
        print(f"  [!] 空格/矢量化失败 {len(missing)} 字: {''.join(missing[:40])}")
    if not glyphs:
        raise SystemExit("无有效补录字形，中止")

    print("[2] 备份并就地替换字形...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FONT, backup)
    replaced, failed = replace_glyphs(glyphs)
    print(f"  已生效 {len(replaced)} 字（含字库缺字新增；备份: {backup.name}）")
    if failed:
        print(f"  [!] 矢量化失败（跳过）: {''.join(failed)}")

    # 维护真迹字清单（export_m4 据此决定参考图范围与生成目标缺口）
    manifest = OUT_DIR / "handwritten_chars.json"
    if manifest.exists():
        prev = json.loads(manifest.read_text(encoding="utf-8"))
        prev_chars = set(prev["chars"] if isinstance(prev, dict) else prev)
    else:
        seed = json.loads((ROOT / "samples" / "collect_chars.json")
                          .read_text(encoding="utf-8"))["chars"]
        prev_chars = set(seed)   # 首批300字本就是真迹
    all_chars = sorted(prev_chars | set(replaced))
    manifest.write_text(json.dumps({"chars": all_chars}, ensure_ascii=False, indent=0),
                        encoding="utf-8")
    print(f"  真迹清单累计 {len(all_chars)} 字 → {manifest.name}")

    print("[3] IoU 质检...")
    qc(replaced, cells_by_char)

    print("[4] 生成替换前后对比图...")
    make_compare(replaced, backup, prefix)

    print(f"\n完成: {FONT}（{len(replaced)} 字补录生效）")
    print(f"对比图: {OUT_DIR / (prefix + '_对比.png')}")
    print("后续: python tools/closed_loop_test.py 重跑闭环渲染查看整体效果")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="手写补录合并（就地替换字库字形）")
    ap.add_argument("photos", nargs="+", help="手写采集表照片路径（按页序）")
    ap.add_argument("--chars", default=str(CHARS_JSON),
                    help="批次清单JSON（默认 %(default)s）")
    args = ap.parse_args()
    main(args.photos, Path(args.chars))
