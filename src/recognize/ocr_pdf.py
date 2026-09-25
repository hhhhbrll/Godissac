"""图片 → OCR → 重建带文本层 PDF → 走 M9 文本优先管线。

背景：照片/扫描件无文本层，旧 CV+VLM 降级方案定位精度差、耗时
（每页 VLM 看图作答 30-60s）、context 弱导致题型误判（如排序题
答成"串"）。qwen-vl-ocr 转写质量高（含 ____ 空线标记），
重建 PDF 后空位坐标 100% 自洽（字符 bbox 由我们自己排）。

流程：
1. 图片文件夹 → 智能排序（convert.sort_photos_by_page_order）
2. 逐页 qwen-vl-ocr 转写（纯文本，含 ____ 与（）空位标记）
3. 重建 A4 PDF：文楷字体流式排版；____ 渲染为横线（fill 矩形，
   M9 管线的 _extract_lines 能识别）
4. 返回 PDF 路径 → pipeline 走 process_pdf_textfirst

OCR 失败（无 key/网络断）→ pipeline 自动降级 CV+VLM。

用法（pipeline 内部）：
    from src.recognize.ocr_pdf import rebuild_pdf_from_images
    pdf = rebuild_pdf_from_images(folder_or_image, out_dir)
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[2]

# 排版参数
PAGE_W, PAGE_H = 595.0, 842.0          # A4 pt
MARGIN_X, MARGIN_TOP, MARGIN_BOT = 52.0, 58.0, 48.0
FONT_SIZE = 10.5
LINE_GAP = 8.0                          # 行间距（pt）
UNDERLINE_RUN = 4                       # OCR 下划线最小字符数（____ = 4 个）
FALLBACK_FONT = ROOT / "fonts" / "LXGWWenKai-Regular.ttf"


def _ocr_page(img_bgr: np.ndarray, api_key: str) -> str:
    """单页图片 → qwen-vl-ocr 纯文本。

    后处理：紧凑空括号"（）"展开为"（    ）"——OCR 常丢失括号内空格，
    紧贴括号重建后宽度为 0 会被空位检测拒绝。
    """
    from dashscope import MultiModalConversation

    h, w = img_bgr.shape[:2]
    s = 1500 / max(h, w)
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)))
    tmp_dir = Path(tempfile.mkdtemp(prefix="god_ocr_"))
    tmp = tmp_dir / "page.jpg"
    try:
        cv2.imwrite(str(tmp), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        resp = MultiModalConversation.call(
            model="qwen-vl-ocr", api_key=api_key,
            messages=[{"role": "user", "content": [
                {"image": f"file://{tmp.resolve().as_posix()}"}]}],
            timeout=120)
        parts = []
        for seg in resp.output.choices[0].message.content:
            if "text" in seg:
                parts.append(seg["text"])
        text = "\n".join(parts)
        text = text.replace("（）", "（    ）").replace("()", "(    )")
        return text
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _load_key() -> str | None:
    import re as _re
    env = ROOT / ".env"
    if not env.exists():
        return None
    m = _re.search(r"^\s*DASHSCOPE_API_KEY\s*=\s*(\S+)",
                   env.read_text(encoding="utf-8"), _re.M)
    return m.group(1) if m else None


def _render_page(doc: fitz.Document, text: str, font: fitz.Font) -> None:
    """一段 OCR 文本 → A4 页（流式排版，____ → 横线）。"""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    fontname = "OCR"
    page.insert_font(fontname=fontname, fontfile=str(FALLBACK_FONT))
    y = MARGIN_TOP
    max_w = PAGE_W - MARGIN_X * 2

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            y += LINE_GAP * 0.6
            continue
        # 切 ____ 段：文本/横线交替渲染
        segs = re.split(r"(_{2,})", line)
        # 自动换行：累计宽度超行宽时折行（简单按字符流）
        x = MARGIN_X
        for seg in segs:
            if not seg:
                continue
            if re.fullmatch(r"_{2,}", seg):
                # 横线段：宽度 = 字符数 × 半字宽，下限 55pt（OCR 的
                # 下划线数量不可靠，4 个 _ 常代表一整句默写空位≈5字宽）
                w = max(len(seg) * FONT_SIZE * 0.55, 55.0)
                if x + w > PAGE_W - MARGIN_X:  # 折行
                    x = MARGIN_X
                    y += FONT_SIZE + LINE_GAP
                page.draw_rect(fitz.Rect(x, y - 2.5, x + w, y - 2.0),
                               color=None, fill=(0, 0, 0))
                x += w
                continue
            # 文本段：逐字排（空格占位不渲染——括号内空隙需要真实宽度）
            for ch in seg:
                if ch == " " or ch == "\u3000":
                    x += FONT_SIZE * (0.5 if ch == " " else 1.0)
                    continue
                if not ch.strip():
                    continue
                cw = font.text_length(ch, fontsize=FONT_SIZE)
                if x + cw > PAGE_W - MARGIN_X:
                    x = MARGIN_X
                    y += FONT_SIZE + LINE_GAP
                if y > PAGE_H - MARGIN_BOT:
                    return
                page.insert_text(fitz.Point(x, y), ch,
                                 fontsize=FONT_SIZE, fontname=fontname)
                x += cw
        y += FONT_SIZE + LINE_GAP
        if y > PAGE_H - MARGIN_BOT:
            break


def rebuild_pdf_from_images(src: str | Path, out_dir: str | Path,
                             pages: list[tuple] | None = None) -> Path:
    """图片（文件夹/单张）→ OCR → 重建 PDF。

    pages: normalize() 返回的 [(idx, png_path, render_pdf)]（已智能排序
    + A4 归一化）；None 时内部对 src 归一化。
    返回重建 PDF 路径（out_dir/rebuilt.pdf）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(src)

    api_key = _load_key()
    if not api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，无法 OCR")

    # 页图（智能排序 + A4 归一化）
    if pages is None:
        from .convert import normalize
        pages = normalize(src, out_dir / "pages")
    if not pages:
        raise RuntimeError(f"无图片页: {src}")

    doc = fitz.open()
    font = fitz.Font(fontfile=str(FALLBACK_FONT))
    for i, png, _rpdf in pages:
        raw = cv2.imdecode(np.fromfile(str(png), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        if raw is None:
            print(f"  [跳过] 无法读取: {Path(png).name}")
            continue
        print(f"  OCR 第 {i+1} 页...")
        text = _ocr_page(raw, api_key)
        _render_page(doc, text, font)
    pdf = out_dir / "rebuilt.pdf"
    doc.save(str(pdf), garbage=4, deflate=True)
    doc.close()
    print(f"  重建 PDF: {pdf}（{len(fitz.open(pdf))} 页）")
    return pdf
