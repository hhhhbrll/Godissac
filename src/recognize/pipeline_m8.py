"""M8 重构版管线：程序精确定位空位 + VLM 纯 OCR 读题作答 + 位置直接注入答案。

架构：
1. PDF/照片 → 程序精确检测空位坐标（已有，100% 精准）
2. 在 PNG 上用顺序 1,2,3... 标注空位（永远从 1 开始，不依赖 PDF 检测的 id）
3. VLM 看标注图，按图上编号作答（不看任何外部 id）
4. 答案按位置顺序直接注入 blanks[]，不用 key 查找（消除歧义）
5. 渲染时 blanks[i] 本身就有坐标，直接用

关键设计：blank_detector 的 blank.id 只用于调试展示，
管线内部流通用 0-based 位置索引，不再有任何 id 转换。
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Literal

import pymupdf as fitz

# 加载配置
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = ROOT / "config" / "prompt.txt"

# ------------------------------------------------------------------
# VLM 调用（支持百炼和中转站）
# ------------------------------------------------------------------

def _load_vendor() -> tuple[str, str, Literal["qwen"] | Literal["relay"] | Literal["zhipu"], str | None]:
    """返回 (api_key, model, vendor, base_url)。"""
    from .vlclient import load_vendor_config, resolve_vendor_key
    cfg = load_vendor_config()
    vendor, api_key, base_url = resolve_vendor_key(cfg)
    model = cfg.get("model")
    return api_key, model, vendor, base_url


def _call_vlm(image_path: Path, blank_count: int,
              contexts: list[str], api_key: str, model: str,
              vendor: Literal["qwen"] | Literal["relay"] | Literal["zhipu"],
              base_url: str | None, timeout: int = 180,
              offset: int = 0) -> list[str]:
    """调用 VLM，返回 answers[0..n-1]（按图上编号 1..n 的顺序）。

    offset: 本批空位在原图的起始编号（用于 prompt 提示对应圆圈编号）
    """
    # 构造强约束的文言文释义题 prompt
    ctx_lines = []
    for i, ctx in enumerate(contexts):
        # 解析空位宽度信息（来自 _make_context 的子字符串）
        # 我们不传宽度，但通过 context 让模型知道答案长度
        ctx_lines.append(f"[{i+1+offset}] {ctx}")
    ctx_text = "\n".join(ctx_lines)
    instruction = (
        f"这是一张高中语文文言文作业图片，红色圆圈标注了多个待填空位。\n"
        f"本批次你需要回答圆圈编号 {offset+1} ~ {offset+blank_count} 共 {blank_count} 个空位。\n\n"
        f"【题型说明】「加点字词解释」题——用括号（）表示空位，需要填入【加点字】的现代汉语释义。\n"
        f"【答题规则】（严格执行）\n"
        f"1. 绝对不能填加点字本身！\n"
        f"   例: 环植（·竹）之，加点字是「植」，应答「种植」或「种竹」，绝不能答「植」\n"
        f"   例: 视（·）纷华势利，加点字是「视」，应答「看」或「看待」，绝不能答「视」\n"
        f"   例: 直谅多闻（·），加点字是「闻」，应答「见闻」或「听说」，绝不能答「闻」\n"
        f"2. 每个空位都标注了「加点:「X」」告诉你哪个字是加点字。答案必须针对这个字。\n"
        f"3. 【关键】答案长度根据括号宽度调整：\n"
        f"   - 括号很窄（仅够1字）：答1字（如「家」「多」「贤」）\n"
        f"   - 括号较宽（够2字）：答2字（如「种植」「家庭」）\n"
        f"   - 宁可答1字也不要答3字以上\n"
        f"4. 文言文常用字义（1-3字）：\n"
        f"   视=看；泊=淡泊；辄=往往；多=多；承=承受/继承；训=教诲；甫=才；冠=成年；\n"
        f"   闻=见闻；咏=吟咏；殖=种植；率=大多；蚤=早；启=打开；涕=眼泪；落=衰落；\n"
        f"   惟=思考；和=唱和；拏=妻子；馁=饥饿；遑=闲暇；恤=体恤；嫡=嫡亲；嗣=继承\n"
        f"5. block（大题答题区）填完整句子，翻译题用现代汉语\n"
        f"6. 注释里的括号（'【注释】'内）和大题分值占位（'（X 分）'）不是空位，忽略\n"
        f"7. 不确定时填 \"\"\n\n"
        f"【每个空位的题目文字】（按圆圈编号顺序）\n"
        f"{ctx_text}\n\n"
        f"【输出格式】严格只输出 JSON 数组（绝对不输出其他文字）：\n"
        f'["第{offset+1}个空的答案", "第{offset+2}个空的答案", ...]\n'
        f"数组长度必须 = {blank_count}，按编号 {offset+1} ~ {offset+blank_count} 顺序对应。"
    )

    if vendor in ("relay", "zhipu"):
        return _call_relay(image_path, instruction, api_key, base_url, model, timeout)
    else:
        return _call_dashscope(image_path, instruction, api_key, model, timeout)


def _call_relay(image_path: Path, instruction: str,
                api_key: str, base_url: str, model: str, timeout: int) -> list[str]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    img_bytes = image_path.read_bytes()
    import base64 as _b64
    b64_img = _b64.b64encode(img_bytes).decode("utf-8")
    ext = image_path.suffix.lower().lstrip(".")
    mime = f"image/{ext}" if ext in ("jpg", "jpeg", "png", "gif", "webp") else "image/png"

    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}},
                        {"type": "text", "text": instruction},
                    ],
                }],
                timeout=timeout,
            )
            text = resp.choices[0].message.content
            return _parse_answers_list(text)
        except Exception as e:
            last_err = e
            print(f"  ⚠ Relay 调用失败(第{attempt+1}次): {e} — {5*(attempt+1)}秒后重试")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Relay 连续 3 次调用失败: {last_err}")


def _call_dashscope(image_path: Path, instruction: str,
                    api_key: str, model: str, timeout: int) -> list[str]:
    from dashscope import MultiModalConversation
    url = f"file://{image_path.as_posix()}"
    last_err = None
    for attempt in range(3):
        try:
            resp = MultiModalConversation.call(
                model=model,
                api_key=api_key,
                messages=[{"role": "user", "content": [{"image": url}, {"text": instruction}]}],
                timeout=timeout,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"API 返回 {resp.status_code}: {getattr(resp, 'message', '')}")
            text = "".join(seg.get("text", "") for seg in
                           resp.output.choices[0].message.content)
            return _parse_answers_list(text)
        except Exception as e:
            last_err = e
            print(f"  ⚠ DashScope 调用失败(第{attempt+1}次): {e} — {5*(attempt+1)}秒后重试")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"DashScope 连续 3 次调用失败: {last_err}")


def _parse_answers_list(text: str) -> list[str]:
    """从 VLM 响应中解析 answers 列表，返回 [ans1, ans2, ...]。

    支持格式：
      {"answers": ["ans1", "ans2", ...]}
      {"answers": {"1": "ans1", "2": "ans2", ...}}  (旧格式，转换为列表)
      ["ans1", "ans2", ...]  (数组直接返回)
    """
    text = text.strip()
    # 找 JSON
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            text = text[s:e+1]
        else:
            s, e = text.find("["), text.rfind("]")
            if s != -1 and e > s:
                text = text[s:e+1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    # 列表格式
    if isinstance(data, list):
        return [str(a) for a in data]

    # 字典格式
    if isinstance(data, dict):
        if "answers" in data:
            raw = data["answers"]
            if isinstance(raw, list):
                return [str(a) for a in raw]
            if isinstance(raw, dict):
                # 转为顺序列表
                keys = sorted(raw.keys(), key=lambda k: int(k) if k.isdigit() else 0)
                return [str(raw.get(k, "")) for k in keys]
        return []

    return []


# ------------------------------------------------------------------
# 管线核心
# ------------------------------------------------------------------

def process(
    src: Path,
    run_dir: Path,
    enable_brackets: bool = False,
    model_spec: str | None = None,
    prompt_path: str | None = None,
) -> tuple[str, list[dict]]:
    """M8 主管道：检测空位 → 编号图 → VLM 作答 → 注入答案 → blanks spec。

    返回 (render_src, blanks_spec_list)
    """
    from .blank_detector import detect_pdf_all_pages, detect_image_blanks, draw_labels_on_image
    from .convert import normalize, ZOOM, A4_W, A4_H, pdf_to_page_images
    from .vlclient import load_vendor_config, resolve_vendor_key

    cfg = load_vendor_config()
    vendor, api_key, base_url = resolve_vendor_key(cfg, model_spec)
    model = cfg.get("model") or ("qwen3.7-plus" if vendor == "qwen" else "qwen3-vl-plus")
    print(f"[后端] vendor={vendor}  model={model}  base_url={base_url}")

    # 复制 prompt（避免 shutil.copy2 中文路径问题）
    run_prompt = run_dir / "prompt.txt"
    src_prompt = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
    try:
        run_prompt.write_bytes(src_prompt.read_bytes())
    except Exception:
        pass

    suf = src.suffix.lower()

    # === 分发：检测空位 ===
    if suf == ".pdf":
        pages_blanks = detect_pdf_all_pages(src, enable_brackets=enable_brackets)
        total = sum(len(pb.blanks) for pb in pages_blanks)
        print(f"程序检测到 {total} 个空位（坐标精度 < 0.5pt）")
        if total == 0:
            return str(src), []

        work = run_dir / "pages"
        work.mkdir(parents=True, exist_ok=True)
        page_pngs = pdf_to_page_images(src, work)
        labeled_pngs: list[Path] = []
        for (page_idx, png, _), pb in zip(page_pngs, pages_blanks):
            if not pb.blanks:
                labeled_pngs.append(png)
                continue
            labeled = work / f"page_{page_idx:03d}_labeled.png"
            # 画标注：永远用 0-based 顺序 1,2,3...
            _draw_sequential_labels(pb, png, labeled, zoom=ZOOM)
            labeled_pngs.append(labeled)

        # === VLM 逐页作答（分批：每批 ≤8 个空位，模型视野更聚焦）===
        all_blanks: list[dict] = []
        global_idx = 0  # 0-based 全局位置（用于注入答案）
        render_src = str(src)
        BATCH_SIZE = 8
        for pb, labeled_png in zip(pages_blanks, labeled_pngs):
            if not pb.blanks:
                continue
            n = len(pb.blanks)
            contexts = [b.context for b in pb.blanks]
            print(f"\n=== 第 {pb.page_index+1} 页：VLM 作答（{n} 个空位，分 {(n+BATCH_SIZE-1)//BATCH_SIZE} 批）===")

            # 分批调用 VLM
            answers = []
            for batch_start in range(0, n, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, n)
                batch_n = batch_end - batch_start
                batch_ctx = contexts[batch_start:batch_end]
                batch_answers = _call_vlm(labeled_png, batch_n, batch_ctx,
                                          api_key, model, vendor, base_url,
                                          offset=batch_start)
                answers.extend(batch_answers)
                print(f"  批 [{batch_start+1}~{batch_end}/{n}] 完成")

            for i, b in enumerate(pb.blanks):
                ans = answers[i].strip() if i < len(answers) else ""
                spec = {
                    "id": b.id,
                    "page": pb.page_index,
                    "x": round(b.x, 1),
                    "y": round(b.y, 1),
                    "width": round(b.w, 1),
                    "height": round(b.h, 1),
                    "type": b.type,
                    "context": b.context,
                    "answer": ans,
                    "confidence": 1.0,
                }
                all_blanks.append(spec)
                ans_disp = ans if ans else "(空)"
                print(f"  #{b.id:02d}(位{i+1}/{n}) [{b.type:9s}] "
                      f"x={b.x:6.1f} y={b.y:6.1f} ans='{ans_disp[:40]}'")
                global_idx += 1

        return render_src, all_blanks

    elif suf in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        work = run_dir / "pages"
        work.mkdir(parents=True, exist_ok=True)
        pages = normalize(src, work)
        render_src = pages[0][2]
        all_blanks: list[dict] = []
        global_idx = 0
        for page_idx, png, _ in pages:
            pb = detect_image_blanks(png, a4_w=A4_W, a4_h=A4_H, zoom=ZOOM)
            if not pb.blanks:
                continue
            labeled = work / f"page_{page_idx:03d}_labeled.png"
            _draw_sequential_labels(pb, png, labeled, zoom=ZOOM)
            n = len(pb.blanks)
            contexts = [b.context for b in pb.blanks]
            print(f"\n=== 第 {page_idx+1} 页：VLM 作答（{n} 个空位）===")
            answers = _call_vlm(labeled, n, contexts, api_key, model, vendor, base_url)
            for i, b in enumerate(pb.blanks):
                ans = answers[i].strip() if i < len(answers) else ""
                spec = {
                    "id": b.id, "page": page_idx,
                    "x": round(b.x, 1), "y": round(b.y, 1),
                    "width": round(b.w, 1), "height": round(b.h, 1),
                    "type": b.type, "context": b.context,
                    "answer": ans, "confidence": 1.0,
                }
                all_blanks.append(spec)
                global_idx += 1
        return render_src, all_blanks

    elif suf in (".doc", ".docx"):
        work = run_dir / "pages"
        work.mkdir(parents=True, exist_ok=True)
        from .convert import normalize as _norm
        _norm(src, work)
        pdf_path = work / (src.stem + ".pdf")
        if not pdf_path.exists():
            for f in work.glob("*.pdf"):
                pdf_path = f; break
        return process(pdf_path, run_dir, enable_brackets, model_spec, prompt_path)

    elif src.is_dir():
        work = run_dir / "pages"
        work.mkdir(parents=True, exist_ok=True)
        pages = normalize(src, work)
        render_src = pages[0][2]
        all_blanks: list[dict] = []
        global_idx = 0
        for page_idx, png, _ in pages:
            pb = detect_image_blanks(png, a4_w=A4_W, a4_h=A4_H, zoom=ZOOM)
            if not pb.blanks:
                continue
            labeled = work / f"page_{page_idx:03d}_labeled.png"
            _draw_sequential_labels(pb, png, labeled, zoom=ZOOM)
            n = len(pb.blanks)
            contexts = [b.context for b in pb.blanks]
            print(f"\n=== 第 {page_idx+1} 页：VLM 作答（{n} 个空位）===")
            answers = _call_vlm(labeled, n, contexts, api_key, model, vendor, base_url)
            for i, b in enumerate(pb.blanks):
                ans = answers[i].strip() if i < len(answers) else ""
                spec = {
                    "id": b.id, "page": page_idx,
                    "x": round(b.x, 1), "y": round(b.y, 1),
                    "width": round(b.w, 1), "height": round(b.h, 1),
                    "type": b.type, "context": b.context,
                    "answer": ans, "confidence": 1.0,
                }
                all_blanks.append(spec)
                global_idx += 1
        return render_src, all_blanks

    else:
        raise ValueError(f"不支持的格式: {suf}")


def _draw_sequential_labels(pb, base_image_path: Path, output_path: Path, zoom: float = 2.0):
    """用 1,2,3... 顺序标注（永远从 1 开始，不依赖 blank.id）。"""
    import cv2, numpy as np
    img = cv2.imdecode(
        np.fromfile(str(base_image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {base_image_path}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    for seq, b in enumerate(pb.blanks, 1):  # seq = 1,2,3,...
        lx = int(b.x * zoom)
        ly = int(b.y * zoom)
        label_x = max(4, lx - 24)
        label_y = ly - int(b.h * zoom * 0.3)
        # 红色圆圈 + 白色数字
        cv2.circle(img, (label_x + 10, label_y - 6), 11, (0, 0, 210), -1)
        text = str(seq)
        tx = label_x + (7 if len(text) == 1 else 3)
        cv2.putText(img, text, (tx, label_y - 1), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        # 蓝色框
        x1 = int(b.x * zoom)
        y1 = int((b.y - b.h) * zoom)
        x2 = int((b.x + b.w) * zoom)
        y2 = int(b.y * zoom)
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 100, 0), 1)

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("标注图编码失败")
    output_path.write_bytes(buf.tobytes())
    pb.labeled_image_path = output_path


def save_output(render_src: str, blanks: list[dict], original: Path, run_dir: Path) -> Path:
    """保存 m2_review.json 到 run_dir 和 output/m2_review.json。"""
    data = {
        "source": render_src,
        "original": str(original),
        "run_dir": str(run_dir),
        "pages": max((b["page"] for b in blanks), default=0) + 1,
        "blanks": blanks,
    }
    out1 = run_dir / "m2_review.json"
    out2 = ROOT / "output" / "m2_review.json"
    for out in (out1, out2):
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out1


# ------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------

def main():
    import sys as _sys
    args = _sys.argv[1:]
    interactive = False
    prompt_path = None
    model_spec = None
    file_arg = None
    enable_brackets = True  # 文言文作业默认开启括号检测
    for a in args:
        if a in ("-i", "--interactive"):
            interactive = True
        elif a.startswith("--prompt="):
            prompt_path = a.split("=", 1)[1]
        elif a.startswith("-m=") or a.startswith("--model="):
            model_spec = a.split("=", 1)[1]
        elif a == "--brackets":
            enable_brackets = True
        elif not a.startswith("-"):
            file_arg = a

    if not file_arg:
        file_arg = "samples/demo_homework.pdf"

    src = Path(file_arg)
    if not src.exists():
        _sys.exit(f"文件不存在: {src}")

    print(f"处理文件: {src}")

    # 交互模式
    if interactive:
        run_dir_tmp = ROOT / "output" / "runs" / "_tmp_interact"
        run_dir_tmp.mkdir(parents=True, exist_ok=True)
        tmp_prompt = run_dir_tmp / "prompt.txt"
        if DEFAULT_PROMPT_PATH.exists():
            shutil.copy2(DEFAULT_PROMPT_PATH, tmp_prompt)
        print(f"\n提示词文件: {tmp_prompt}")
        print(f"按 C 编辑，回车继续...")
        while True:
            c = input("> ").strip().lower()
            if c == "c":
                import os as _os
                try:
                    _os.startfile(str(tmp_prompt))
                except Exception:
                    pass
                input("保存后回车继续...")
            elif c == "q":
                _sys.exit("已取消")
            else:
                break
        prompt_path = str(tmp_prompt)

    stem = src.stem if src.is_file() else src.name
    run_dir = ROOT / "output" / "runs" / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    render_src, blanks = process(src, run_dir, enable_brackets=enable_brackets,
                                 model_spec=model_spec, prompt_path=prompt_path)
    out = save_output(render_src, blanks, src, run_dir)

    print(f"\n=== 完成 ===")
    print(f"待审核: {out}")
    print(f"共 {len(blanks)} 个空位")
    print(f"\n可执行: python god.py render")


if __name__ == "__main__":
    main()
