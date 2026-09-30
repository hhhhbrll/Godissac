"""渲染已审核的答案文件：m2_review.json → 答案层 PDF + 合并版 PDF。

用法：
    python -m src.render.render_answered output/m2_review.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf as fitz

from .perturb import PerturbParams
from .renderer import BlankSpec, HandwriteRenderer

ROOT = Path(__file__).resolve().parents[2]
# 默认字体：优先完整字库，否则300字版，最后回退文楷
_FONT_CANDIDATES = [
    ROOT / "assets" / "fonts" / "myhand_full.ttf",
    ROOT / "output" / "myhand_full.ttf",
    ROOT / "output" / "myhand.ttf",
    ROOT / "fonts" / "LXGWWenKai-Regular.ttf",
]
FONT = next((p for p in _FONT_CANDIDATES if p.exists()), _FONT_CANDIDATES[-1])
FALLBACK = ROOT / "fonts" / "LXGWWenKai-Regular.ttf"
# 渲染风格配置（用户可在 config/render_style.json 手动调整，也可用软件菜单）
STYLE_PATH = ROOT / "config" / "render_style.json"
_DEFAULT_STYLE = {"print_fade": 0.1, "ink_darken": 0.016}


def _load_style() -> dict:
    """读渲染风格：print_fade=题目变淡(0~0.6)，ink_darken=手写加深(0~0.08)。"""
    style = dict(_DEFAULT_STYLE)
    if STYLE_PATH.exists():
        try:
            style.update(json.loads(STYLE_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    style["print_fade"] = min(max(float(style.get("print_fade", 0)), 0.0), 0.6)
    style["ink_darken"] = min(max(float(style.get("ink_darken", 0)), 0.0), 0.08)
    return style


def _fade_print(doc: fitz.Document, fade: float) -> None:
    """题目变淡：每页整页叠加半透明白层（文本/图片统一变淡，减轻打印痕迹）。
    须在渲染手写答案之前调用——手写字保持原墨色，与淡题目形成对比。"""
    if fade <= 0:
        return
    for page in doc:
        page.draw_rect(page.rect, color=None, fill=(1, 1, 1),
                       fill_opacity=fade, width=0, overlay=True)


def _subset(doc: fitz.Document) -> None:
    """字体子集化（减小体积）。图片型文档（JPG 作业照片）不支持，降级跳过。"""
    try:
        doc.subset_fonts()
    except Exception:
        pass


def main(review_path: str):
    data = json.loads(Path(review_path).read_text(encoding="utf-8"))
    src_pdf = Path(data["source"])
    if not src_pdf.exists():
        raise FileNotFoundError(f"原始作业文件不存在: {src_pdf}")
    blanks = []
    for b in data["blanks"]:
        spec = {k: b[k] for k in ("id", "page", "x", "y", "width", "answer", "height")}
        spec["type"] = b.get("type", "underline")
        if b.get("inline_chars"):
            spec["inline_chars"] = [tuple(c) for c in b["inline_chars"]]
        if b.get("segments"):
            spec["segments"] = [(s["page"], s["x"], s["width"], s["rules"])
                                for s in b["segments"]]
        blanks.append(BlankSpec(**spec))

    # 字号 14pt（打印实测 13pt 偏小；墨色同步加深见 perturb.py）
    # 产物目录：优先本次作业的 run 目录（output/runs/<作业名>_<时间戳>/），
    # 识别与渲染产物聚在一起；无 run_dir 时退回 output/
    out_dir = Path(data.get("run_dir", "")) if data.get("run_dir") else (ROOT / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    # 产物名优先用原始作业文件名（拍照作业的渲染底版叫 page_000.pdf 不友好）
    orig = data.get("original", str(src_pdf))
    stem = Path(orig).stem

    # 渲染风格（题目变淡/手写加深，见 config/render_style.json）
    style = _load_style()
    params = PerturbParams()
    if style["ink_darken"] > 0:
        # 手写墨色整体加深：灰度值下移，下限钳到 0（全黑）
        params.ink_min = max(0.0, params.ink_min - style["ink_darken"])
        params.ink_max = max(params.ink_min, params.ink_max - style["ink_darken"])

    # ① 答案层（透明，与题目分离；不受 print_fade 影响）
    src = fitz.open(src_pdf)
    layer = fitz.open()
    for p in src:
        layer.new_page(width=p.rect.width, height=p.rect.height)
    renderer = HandwriteRenderer(FONT, params, base_size=14,
                                  fallback_font=FALLBACK)
    results = [renderer.render(layer, b) for b in blanks]
    _subset(layer)
    layer.save(out_dir / f"{stem}_答案层.pdf", garbage=4, deflate=True)

    # ② 合并版：先淡题目再写手写——手写墨色不受白层影响，对比突出
    merged = fitz.open(src_pdf)
    _fade_print(merged, style["print_fade"])
    for b in blanks:
        renderer.render(merged, b)
    _subset(merged)
    merged.save(out_dir / f"{stem}_完成版.pdf", garbage=4, deflate=True)

    doc = fitz.open(out_dir / f"{stem}_完成版.pdf")
    doc[0].get_pixmap(dpi=150).save(out_dir / f"{stem}_完成版_预览.png")

    overflow = [r for r in results if r.overflowed]
    print(f"渲染完成: {out_dir / (stem + '_答案层.pdf')}")
    print(f"          {out_dir / (stem + '_完成版.pdf')}")
    if renderer.fallback_used:
        fb = "".join(sorted(renderer.fallback_used))
        print(f"注意: {len(renderer.fallback_used)} 个字回退文楷（字库未覆盖）: {fb}")
        (out_dir / "fallback_chars.log").write_text(fb, encoding="utf-8")
    if overflow:
        print(f"注意: {len(overflow)} 个空位在最小字号下仍溢出: "
              + ", ".join(f"#{r.blank.id}" for r in overflow))


if __name__ == "__main__":
    main(sys.argv[1])
