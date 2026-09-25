"""输入归一化：PDF / JPG / Word → 统一为 A4 尺寸 PNG（zoom=2, 144dpi）+ 渲染底版 PDF。

统一输出规格（简化坐标映射）：
- VLM 用 PNG：统一为 A4 尺寸 1190×1684px（zoom=2, 1pt=2px），白底
  PDF：直接栅格化；照片：等比缩放贴到A4页居中（留2%边距）
- 渲染用 PDF：标准A4尺寸，PDF直接用原文件，照片生成A4居中版
- 坐标完全统一：pt = px / 2，无需额外 scale/offset 换算
- 照片增强已取消（用户要求：请自行扫描文件后再上传）
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

ZOOM = 2.0  # 全局统一：1pt = 2px
A4_W, A4_H = 595.0, 842.0  # pt
A4_PX_W, A4_PX_H = int(A4_W * ZOOM), int(A4_H * ZOOM)  # 1190 × 1684 px


def pdf_to_page_images(pdf_path: str | Path, out_dir: str | Path,
                        zoom: float = ZOOM):
    """PDF 每页渲染为 PNG（统一zoom=2）。返回 [(page_idx, png_path, render_pdf_path)]。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    results = []
    mat = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        png = out_dir / f"page_{i:03d}.png"
        page.get_pixmap(matrix=mat, alpha=False).save(png)
        results.append((i, png, str(pdf_path)))
    doc.close()
    return results


def _place_image_on_a4(img_bgr: np.ndarray, landscape: bool = False) -> np.ndarray:
    """把一张图等比缩放贴到A4白底页（zoom=2像素尺寸），返回A4大小的BGR图像。"""
    pw, ph = (A4_PX_H, A4_PX_W) if landscape else (A4_PX_W, A4_PX_H)
    h, w = img_bgr.shape[:2]
    margin = 0.02
    s = min(pw * (1 - margin) / w, ph * (1 - margin) / h)
    new_w, new_h = int(w * s), int(h * s)
    if (new_w, new_h) != (w, h):
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = img_bgr
    # 白底
    canvas = np.full((ph, pw, 3), 255, dtype=np.uint8)
    ox = (pw - new_w) // 2
    oy = (ph - new_h) // 2
    canvas[oy:oy+new_h, ox:ox+new_w] = resized
    return canvas


def _make_a4_pdf_from_image(img_bgr: np.ndarray, out_pdf: Path,
                             landscape: bool = False) -> None:
    """从BGR图像生成A4 PDF（渲染底版用）。"""
    pw, ph = (A4_H, A4_W) if landscape else (A4_W, A4_H)
    doc = fitz.open()
    page = doc.new_page(width=pw, height=ph)
    # 临时保存图片为PNG再插入
    tmp = out_pdf.with_suffix(".tmp.png")
    ok, buf = cv2.imencode(".png", img_bgr)
    tmp.write_bytes(buf.tobytes())
    h, w = img_bgr.shape[:2]
    # 图像按等比贴到A4
    margin = 0.02
    s = min(pw * (1 - margin) / (w / ZOOM), ph * (1 - margin) / (h / ZOOM))
    disp_w, disp_h = (w / ZOOM) * s, (h / ZOOM) * s
    ox, oy = (pw - disp_w) / 2, (ph - disp_h) / 2
    page.insert_image(fitz.Rect(ox, oy, ox + disp_w, oy + disp_h), filename=str(tmp))
    tmp.unlink(missing_ok=True)
    doc.save(str(out_pdf), garbage=4, deflate=True)
    doc.close()


def image_to_page_image(img_path: str | Path, out_dir: str | Path):
    """单张照片 → A4尺寸PNG（VLM用）+ A4 PDF（渲染用）。
    返回 [(0, png_path, render_pdf_path)]。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(img_path)
    raw = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError(f"无法读取图片: {src}")
    # 不增强（用户要求取消）
    h, w = raw.shape[:2]
    landscape = w > h
    a4_img = _place_image_on_a4(raw, landscape=landscape)
    png = out_dir / "page_000.png"
    ok, buf = cv2.imencode(".png", a4_img)
    if not ok:
        raise RuntimeError(f"PNG 编码失败: {png}")
    png.write_bytes(buf.tobytes())
    render_pdf = out_dir / "page_000.pdf"
    _make_a4_pdf_from_image(raw, render_pdf, landscape=landscape)
    return [(0, png, str(render_pdf))]


def _natural_key(p: Path):
    """自然排序键：p2 < p10。"""
    import re
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", p.name)]


# ---------------------------------------------------------------------------
# 未标序照片智能排序（VLM 识别页码+首题号）
# ---------------------------------------------------------------------------

_PAGE_PROMPT_ROI = (
    "图中从上到下是某试卷一页的三个裁剪区域：①页眉（含页码，形如"
    "'第X页/共N页'或'X'）②正文开头（含第一个题号，如'23.'）③页脚。\n"
    "请输出一行：页码:X;首题号:Y\n"
    "页码=①或③中最大的数字编号（如'第43页'输出43；没有页码输出0）；"
    "首题号=②中第一个题目编号数字（没有输出0）。只输出这一行。")

_PAGE_PROMPT_FULL = (
    "这是一份试卷的一页照片。请找出页眉或页脚的页码数字"
    "（形如'第X页/共N页'或'X'），以及这页最上方第一个题目的编号。\n"
    "输出一行：页码:X;首题号:Y（找不到的项输出0）。只输出这一行。")


def _vlm_page_info(img_bgr: np.ndarray, api_key: str, roi: bool) -> tuple[int, int]:
    """VLM 识别 (页码, 首题号)。roi=True 用页眉/正文顶/页脚特写条。"""
    import re
    import tempfile
    from dashscope import MultiModalConversation

    if roi:
        h = img_bgr.shape[0]
        head = img_bgr[:int(h * 0.09)]
        top = img_bgr[int(h * 0.09):int(h * 0.22)]
        foot = img_bgr[int(h * 0.90):]
        strips = [s for s in (head, top, foot) if s.size]
        maxw = max(s.shape[1] for s in strips)
        padded = []
        for s in strips:
            if s.shape[1] < maxw:
                pad = np.full((s.shape[0], maxw - s.shape[1], 3), 255, np.uint8)
                s = cv2.hconcat([s, pad])
            padded.append(s)
        sep = np.full((6, maxw, 3), 128, np.uint8)
        combo = padded[0]
        for s in padded[1:]:
            combo = cv2.vconcat([combo, sep, s])
        s2 = 1000 / combo.shape[1]
        combo = cv2.resize(combo, (1000, int(combo.shape[0] * s2)))
        work = combo
        prompt = _PAGE_PROMPT_ROI
    else:
        h, w = img_bgr.shape[:2]
        s = 900 / max(h, w)
        work = cv2.resize(img_bgr, (int(w * s), int(h * s))) if s < 1 else img_bgr
        prompt = _PAGE_PROMPT_FULL

    tmp = Path(tempfile.mkdtemp(prefix="god_pg_")) / "page.jpg"
    try:
        cv2.imwrite(str(tmp), work, [cv2.IMWRITE_JPEG_QUALITY, 75])
        resp = MultiModalConversation.call(
            model="qwen-vl-plus", api_key=api_key,
            messages=[{"role": "user", "content": [
                {"image": f"file://{tmp.resolve().as_posix()}"},
                {"text": prompt}]}],
            timeout=60)
        text = "".join(seg.get("text", "") for seg in
                       resp.output.choices[0].message.content)
    finally:
        import shutil as _sh
        _sh.rmtree(tmp.parent, ignore_errors=True)
    m = re.search(r"页码[:：]?\s*(\d+).*?首题号[:：]?\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    nums = re.findall(r"\d+", text)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    return 0, 0


def _load_dashscope_key() -> str | None:
    import re
    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.exists():
        return None
    m = re.search(r"^\s*DASHSCOPE_API_KEY\s*=\s*(\S+)",
                  env.read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else None


def sort_photos_by_page_order(photos: list[Path]) -> list[Path]:
    """未标序照片按卷面页码+题号排出阅读顺序。

    策略（两轮 VLM 融合，单轮各有盲区）：
    - ROI 特写（页眉/正文顶/页脚裁剪条）：高页码识别准
    - 全图：低页码与题号识别准
    - 排序键 = (页码, 首题号)：页码为主序，页码相同（如多页都标"第1页"
      的合订卷）按题号；VLM 失败/无 key 回退文件名序。
    """
    api_key = _load_dashscope_key()
    if not api_key:
        print("  [提示] 未配置 DASHSCOPE_API_KEY，照片按文件名顺序处理")
        return photos

    print(f"  识别 {len(photos)} 张照片的页码顺序...")
    infos: list[tuple[int, int]] = []
    for p in photos:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            infos.append((0, 0))
            continue
        try:
            pg, qn = _vlm_page_info(img, api_key, roi=True)
            if pg <= 0:  # ROI 失败 → 全图补
                pg2, qn2 = _vlm_page_info(img, api_key, roi=False)
                pg = pg2 if pg2 > 0 else 0
                qn = qn if qn > 0 else qn2
            infos.append((pg, qn))
        except Exception as e:  # noqa: BLE001 单页失败不阻断
            print(f"    [!] {p.name[:12]}... 页码识别失败({e})，按 0 处理")
            infos.append((0, 0))

    order = sorted(range(len(photos)), key=lambda i: infos[i])
    seq = " ".join(f"{infos[i][0]}/{infos[i][1]}" for i in order)
    print(f"  页序(页码/题号): {seq}")
    # 有效性：页码全 0 = 识别整体失败 → 回退
    if all(pg == 0 for pg, _ in infos):
        print("  [提示] 页码识别全部失败，回退文件名顺序")
        return photos
    return [photos[i] for i in order]


def photos_to_pages(dir_path: str | Path, out_dir: str | Path):
    """文件夹（多张作业照片）→ 每张一页：A4 PNG 列表 + 合并 A4 PDF。

    照片顺序：文件名乱序（如微信导出的 hash 名）时用 VLM 识别页码排序；
    识别失败回退自然文件名序。
    返回 [(page_idx, png_path, render_pdf_path)]。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(dir_path)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    photos = sorted((p for p in src_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in exts),
                    key=_natural_key)
    if not photos:
        raise ValueError(f"文件夹中没有作业照片（支持 {sorted(exts)}）: {src_dir}")
    # 文件名含明显序号（p1/p2 或 1.jpg 2.jpg）直接用；hash 名才 VLM 排序
    import re
    named = all(re.search(r"\d", p.stem) and
                len(p.stem) <= 12 for p in photos)
    if not named:
        photos = sort_photos_by_page_order(photos)
    doc = fitz.open()
    entries = []
    for i, src in enumerate(photos):
        raw = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            print(f"  [跳过] 无法读取: {src.name}")
            continue
        h, w = raw.shape[:2]
        landscape = w > h
        a4_img = _place_image_on_a4(raw, landscape=landscape)
        png = out_dir / f"page_{i:03d}.png"
        ok, buf = cv2.imencode(".png", a4_img)
        if not ok:
            raise RuntimeError(f"PNG 编码失败: {png}")
        png.write_bytes(buf.tobytes())
        # 追加到合并PDF
        pw, ph = (A4_H, A4_W) if landscape else (A4_W, A4_H)
        page = doc.new_page(width=pw, height=ph)
        tmp = out_dir / f"_tmp_{i:03d}.png"
        okt, buf2 = cv2.imencode(".png", raw)
        tmp.write_bytes(buf2.tobytes())
        margin = 0.02
        s = min(pw * (1 - margin) / (w / ZOOM), ph * (1 - margin) / (h / ZOOM))
        disp_w, disp_h = (w / ZOOM) * s, (h / ZOOM) * s
        ox, oy = (pw - disp_w) / 2, (ph - disp_h) / 2
        page.insert_image(fitz.Rect(ox, oy, ox + disp_w, oy + disp_h), filename=str(tmp))
        tmp.unlink(missing_ok=True)
        entries.append((len(entries), png))
        print(f"  第{len(entries)}页: {src.name} → {'横向' if landscape else '纵向'}A4")
    if not entries:
        raise ValueError("没有可用的照片页")
    render_pdf = out_dir / "pages_merged.pdf"
    doc.save(str(render_pdf), garbage=4, deflate=True)
    doc.close()
    return [(i, png, str(render_pdf)) for (i, png) in entries]


def _docx_to_pdf_word_com(doc_path: Path, out_dir: Path) -> Path:
    """Microsoft Word COM 自动化 → PDF（Windows+Office 环境，自带文本层）。"""
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    word = None
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(str(doc_path.resolve()), ReadOnly=True)
        try:
            pdf = out_dir / (doc_path.stem + ".pdf")
            doc.ExportAsFixedFormat(OutputFileName=str(pdf),
                                    ExportFormat=17)  # wdExportFormatPDF
        finally:
            doc.Close(SaveChanges=0)
        return pdf
    finally:
        if word is not None:
            word.Quit()
            pythoncom.CoUninitialize()


def docx_to_pdf(doc_path: str | Path, out_dir: str | Path):
    """Word(.doc/.docx) → PDF → 页图。

    转换器优先级：LibreOffice(soffice，跨平台) → Microsoft Word COM
    (Windows+Office，ExportAsFixedFormat 自带文本层可直接走 M9 文本优先管线)。
    """
    import shutil
    import subprocess
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(doc_path)
    soffice = shutil.which("soffice") or shutil.which("soffice.exe")
    if soffice:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf",
             "--outdir", str(out_dir), str(src)],
            check=True, timeout=180,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        pdf = out_dir / (src.stem + ".pdf")
        if pdf.exists():
            return pdf_to_page_images(pdf, out_dir)
        raise ValueError(f"LibreOffice 转换失败: {src.name}")

    # Windows + Microsoft Office：Word COM 转换
    try:
        pdf = _docx_to_pdf_word_com(src, out_dir)
    except ImportError:
        raise ValueError(
            "Word 文档转换需要 LibreOffice(soffice) 或 Microsoft Word"
            "(pip install pywin32)，两者都不可用；"
            "也可先在 Word 里另存为 PDF 再运行")
    except Exception as e:  # noqa: BLE001 COM 错误（Word 弹窗/文件损坏等）
        raise ValueError(f"Word 转换失败（{e}）；可先在 Word 里另存为 PDF 再运行")
    return pdf_to_page_images(pdf, out_dir)


def normalize(file_path: str | Path, out_dir: str | Path):
    """统一入口：按扩展名分发。
    返回 [(page_idx, png_path, render_pdf_path)]：
    - png_path: 统一zoom=2的A4尺寸PNG（VLM用），pt = px / 2
    - render_pdf_path: 渲染底版PDF路径
    """
    file_path = Path(file_path)
    if file_path.is_dir():
        return photos_to_pages(file_path, out_dir)
    suf = file_path.suffix.lower()
    if suf == ".pdf":
        return pdf_to_page_images(file_path, out_dir)
    if suf in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        return image_to_page_image(file_path, out_dir)
    if suf in (".doc", ".docx"):
        return docx_to_pdf(file_path, out_dir)
    raise ValueError(f"不支持的格式: {suf}")
