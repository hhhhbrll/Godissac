"""M9 主管线（文本优先）：作业文件 → 文本层空位提取 → 文本LLM作答 → 待审核JSON。

架构（核心突破：坐标与作答彻底解耦）：
1. PDF 有文本层 → 括号/横线坐标从字符 bbox 与绘图指令提取（100%精准）
2. 重建全文并嵌入【N】空位标记（跨页拼接，LLM 可见完整文章+译文）
3. 文本 LLM 按编号作答（对齐率100%，不会漂移）
4. 答案注入空位 → m2_review.json → 渲染器

无文本层（照片/扫描件）→ 降级旧方案（CV检测+VLM）。

用法：
    python -m src.recognize.pipeline <作业文件> [-i] [--prompt=路径] [-m 后端:模型]
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _write_review(render_src: str, original: Path, run_dir: Path,
                  pages: int, blanks_specs: list[dict]) -> Path:
    data = {
        "source": render_src,
        "original": str(original),
        "run_dir": str(run_dir),
        "pages": pages,
        "blanks": blanks_specs,
    }
    out1 = run_dir / "m2_review.json"
    out2 = ROOT / "output" / "m2_review.json"
    out2.parent.mkdir(parents=True, exist_ok=True)
    for out in (out1, out2):
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out1


# ---------------------------------------------------------------------------
# 文本优先管线（PDF）
# ---------------------------------------------------------------------------

def process_pdf_textfirst(src: Path, run_dir: Path,
                          prompt_path: str | None = None,
                          model_spec: str | None = None,
                          interactive: bool = False) -> Path:
    from .text_extract import extract_pdf
    from .text_llm import answer_homework, refine_answers

    t0 = time.time()
    marked_text, blanks, pages = extract_pdf(src)
    print(f"文本层提取：{pages} 页，{len(blanks)} 个空位（坐标来自字符bbox，100%精准）")
    if not blanks:
        print("⚠ 未检测到空位（可能无文本层或无括号/横线）")
        return _write_review(str(src), src, run_dir, pages, [])

    # 展示检测结果
    for b in blanks[:12]:
        print(f"  【{b.no:2d}】[{b.type:9s}] p{b.page+1} x={b.x:6.1f} y={b.y:6.1f} w={b.w:5.1f}  前文'{b.prev[-10:]}'")
    if len(blanks) > 12:
        print(f"  ... 其余 {len(blanks)-12} 个略")

    # 标记文本存档（调试+用户核对）
    (run_dir / "marked_text.txt").write_text(marked_text, encoding="utf-8")

    # 提示词：复制到 run 目录供用户修改
    from .text_llm import load_prompt as _lp
    from .text_llm import DEFAULT_PROMPT_PATH
    run_prompt = run_dir / "prompt.txt"
    src_prompt = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
    try:
        run_prompt.write_bytes(src_prompt.read_bytes())
    except Exception:
        run_prompt.write_text(_lp(), encoding="utf-8")
    print(f"\n本次任务提示词: {run_prompt}")

    if interactive:
        import os
        print(f"\n{'='*60}")
        print(f"回车 = 用当前提示词开始作答 | C = 打开编辑 prompt.txt | Q = 取消")
        while True:
            c = input("> ").strip().lower()
            if c == "q":
                sys.exit("已取消")
            if c == "c":
                try:
                    os.startfile(str(run_prompt))
                except Exception:
                    print(f"请手动打开: {run_prompt}")
                input("修改保存后回车继续...")
            break

    # 文本 LLM 作答（大批量按篇章分批：篇章=原文+译文+题目一体不切分，
    # 边界页跨页文章由字符偏移精准切分——第7页上半归前批下半归后批）
    from .article_split import split_by_article
    batches = split_by_article(marked_text, blanks)
    t1 = time.time()
    answers: dict[str, str] = {}
    if len(batches) <= 1:
        text, nos = batches[0] if batches else (marked_text, [])
        print(f"\n调用文本 LLM 作答（{len(blanks)} 个空位）...")
        answers = answer_homework(text, nos or len(blanks),
                                  prompt_path=run_prompt, model_spec=model_spec,
                                  debug_path=run_dir / "llm_raw_response.txt")
    else:
        print(f"\n调用文本 LLM 作答（{len(blanks)} 个空位，按篇章分 {len(batches)} 批）...")
        for bi, (text, nos) in enumerate(batches, 1):
            print(f"  [批 {bi}/{len(batches)}] 空位 {nos[0] if nos else '-'}~"
                  f"{nos[-1] if nos else '-'}（{len(nos)}个，文本 {len(text)} 字）")
            part = answer_homework(text, nos,
                                   prompt_path=run_prompt, model_spec=model_spec,
                                   debug_path=run_dir / f"llm_raw_response_batch{bi}.txt")
            answers.update(part)
    print(f"作答完成，耗时 {time.time()-t1:.1f}s")
    # 空答案过多的补答（响应偶发被截断/漏答：对缺失项二次询问，最多2轮）
    import re as _re
    import json as _json
    from .text_llm import load_env_config, resolve_backend, _call_openai, _call_dashscope
    cfg = load_env_config()
    for _round in range(2):
        missing = [b.no for b in blanks
                   if not answers.get(str(b.no), "").strip()
                   and b.type in ("bracket", "block", "inline")]
        if not missing or len(missing) >= len(blanks):
            break
        print(f"⚠ {len(missing)} 个空位无答案，第{_round + 1}轮补答...")
        # 按批次分批补答：缺失空位可能散布全卷，若把所有批次文本拼成
        # 一个请求会超模型输入上限（qwen-max 30720 tokens，曾报
        # 400: Range of input length should be [1, 30720]）。
        # 每批单独发（与初次作答等长，初次能成功补答也能成功）。
        miss_set = set(missing)
        filled_this_round = 0
        try:
            backend, api_key, base_url, model = resolve_backend(cfg, model_spec)
        except Exception as e:  # noqa: BLE001
            print(f"  补答失败: {e}")
            break
        for bi, (batch_text, batch_nos) in enumerate(batches, 1):
            batch_missing = miss_set & set(batch_nos)
            if not batch_text or not batch_missing:
                continue
            ctx_lines = "\n".join(
                f"【{b.no}】{'前文:' + b.prev[-14:]}{'(容量' + str(max(1, round(b.w / 10.4))) + '字)' if b.type == 'bracket' else ''}"
                for b in blanks if b.no in batch_missing)
            retry_prompt = (_lp(run_prompt)
                            + f"\n\n【作业文本（含【N】空位标记）】\n{batch_text}\n\n"
                              f"下面这些编号尚未作答（附前文提示），"
                              f"请只输出紧凑单行JSON补答这些编号：\n{ctx_lines}\n"
                              f"句读题输出原句加\"/\"；翻译题输出完整译文；不确定填空字符串。")
            try:
                if backend in ("relay", "zhipu"):
                    text = _call_openai(api_key, base_url, model, retry_prompt,
                                        timeout=120, max_tokens=4000)
                else:
                    text = _call_dashscope(api_key, model, retry_prompt, timeout=120)
                m = _re.search(r"\{.*\}", text, _re.S)
                if m:
                    for k, v in _json.loads(m.group(0)).items():
                        if (str(k) in [str(x) for x in batch_missing]
                                and str(v).strip()
                                and not answers.get(str(k), "").strip()):
                            answers[str(k)] = str(v).strip()
                            filled_this_round += 1
            except Exception as e:  # noqa: BLE001 单批失败不阻断其余批
                print(f"  [批 {bi}] 补答失败: {e}")
        if filled_this_round == 0:
            print("  本轮无进展，停止补答")
            break
    missing = [b.no for b in blanks
               if not answers.get(str(b.no), "").strip()
               and b.type in ("bracket", "block", "inline")]
    if missing:
        print(f"  补答后仍缺 {len(missing)} 个: {missing}")

    # 答案精炼：对超出括号容量的答案做 LLM 压缩（保留语义核心）
    from .text_llm import purity_pass
    refine_items = []
    all_bracket_items = []
    for b in blanks:
        ans = answers.get(str(b.no), "")
        if b.type == "bracket" and ans.strip():
            cap = max(1, round(b.w / 10.4))
            item = {"no": b.no, "k": cap, "ctx": b.prev, "after": b.after, "ans": ans}
            all_bracket_items.append(item)
            if len(ans) > cap:
                refine_items.append(item)
    if refine_items:
        print(f"压缩 {len(refine_items)} 个超容量答案...")
        refined = refine_answers(refine_items, model_spec=model_spec)
        for it in refine_items:
            new = refined.get(str(it["no"]), "")
            # 只接受：装进容量 且 不是从前文摘的字
            if (new and len(new) <= it["k"]
                    and new not in it["ctx"] and new not in it.get("after", "")):
                if new != it["ans"]:
                    print(f"  #{it['no']:3d} '{it['ans']}' → '{new}'")
                    answers[str(it["no"])] = new
                    it["ans"] = new

    # 纯净性终检：程序化初筛"答案混入原句词语"的嫌疑项（答案中含前后文的
    # 2字以上原词），只把嫌疑项交 LLM 修正——全量送检会大面积误伤好答案
    if all_bracket_items:
        suspects = []
        for it in all_bracket_items:
            ans = it["ans"]
            window = it["ctx"][-10:] + it.get("after", "")[:40]
            hits = [ans[i:i + 2] for i in range(len(ans) - 1)
                    if ans[i:i + 2] in window]
            if hits:
                suspects.append(it)
        if suspects:
            print(f"纯净性终检：{len(suspects)} 个嫌疑答案（含原句词）...")
            corrections = purity_pass(suspects, model_spec=model_spec)
            cap_by_no = {it["no"]: it["k"] for it in all_bracket_items}
            for no_s, new in corrections.items():
                old = answers.get(no_s, "")
                if old.strip() in ("A", "B", "C", "D"):
                    continue  # 选择题答案绝不允许改写
                if (new and new != old
                        and len(new) <= cap_by_no.get(int(no_s), 99)
                        and not new.isspace()):
                    answers[no_s] = new
                    print(f"  #{int(no_s):3d} '{old}' → '{new}'（纯净性修正）")

    # 注入答案
    specs = []
    for b in blanks:
        ans = answers.get(str(b.no), "")
        b.answer = ans
        specs.append(b.to_spec())

    # 展示对齐结果
    print(f"\n{'='*66}")
    print(f"{'编号':<6}{'前文':<14}{'答案':<20}")
    print(f"{'='*66}")
    for b in blanks:
        print(f"【{b.no:3d}】  {b.prev[-10:]:<12} → {b.answer[:16]}")

    out = _write_review(str(src), src, run_dir, pages, specs)
    print(f"\n=== 完成（总耗时 {time.time()-t0:.1f}s）===")
    print(f"待审核文件: {out}")
    print(f"标记全文: {run_dir/'marked_text.txt'}（核对空位上下文用）")
    print(f"共 {len(specs)} 个空位；改 answer 后运行: python god.py render")
    return out


# ---------------------------------------------------------------------------
# 照片降级管线（无文本层：CV检测 + VLM）
# ---------------------------------------------------------------------------

def process_images_fallback(src: Path, run_dir: Path,
                            prompt_path: str | None = None,
                            model_spec: str | None = None) -> Path:
    """照片/文件夹：旧方案（归一化 + CV横线检测 + VLM按编号作答）。"""
    from .blank_detector import detect_image_blanks, draw_labels_on_image
    from .convert import A4_H, A4_W, ZOOM, normalize
    from .vlclient import analyze_page

    pages = normalize(src, run_dir / "pages")
    print(f"已归一化 {len(pages)} 页（照片无文本层，降级 CV+VLM 方案）")
    render_src = pages[0][2]

    all_specs: list[dict] = []
    gid = 1
    for page_idx, png, rpdf in pages:
        pb = detect_image_blanks(png, a4_w=A4_W, a4_h=A4_H, zoom=ZOOM)
        if not pb.blanks:
            continue
        pb.page_index = page_idx
        for b in pb.blanks:
            b.page = page_idx
        labeled = run_dir / "pages" / f"page_{page_idx:03d}_labeled.png"
        draw_labels_on_image(pb, png, labeled)
        # VLM 按编号作答（图上红圈编号）
        ctx = "\n".join(f"[{b.id}] {b.context}" for b in pb.blanks)
        raw = analyze_page(labeled, _api_key(), prompt=_img_prompt(ctx, len(pb.blanks)))
        amap = {str(item.get("id")): item.get("answer", "") for item in raw.get("blanks", [])}
        for b in pb.blanks:
            b.answer = amap.get(str(b.id), "")
            spec = b.to_spec()
            spec["id"] = gid
            gid += 1
            all_specs.append(spec)
    return _write_review(render_src, src, run_dir, len(pages), all_specs)


def _api_key() -> str:
    from .vlclient import load_api_key
    return load_api_key()


def _img_prompt(ctx: str, n: int) -> str:
    return (f"图片上红色圆圈数字标注了{n}个待填空位。\n【空位上下文】\n{ctx}\n\n"
            f"按编号给出答案：释义题填1-4字现代汉语；选择题填字母；大题填完整答案；"
            f"不确定填空串。只输出JSON：{{\"blanks\":[{{\"id\":1,\"answer\":\"...\"}},...]}}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main(file: str, interactive: bool = False,
         prompt_path: str | None = None, model_spec: str | None = None):
    src = Path(file)
    if not src.exists():
        sys.exit(f"文件不存在: {src}")

    stem = src.stem if src.is_file() else src.name
    run_dir = ROOT / "output" / "runs" / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    suf = src.suffix.lower()
    if suf == ".pdf":
        from .text_extract import has_text_layer
        if has_text_layer(src):
            return process_pdf_textfirst(src, run_dir, prompt_path, model_spec, interactive)
        print("PDF 无文本层（扫描件），降级 CV+VLM 方案")
        # 无文本层 PDF 转页图走照片管线
        from .convert import pdf_to_page_images
        pdf_to_page_images(src, run_dir / "pages")
        return process_images_fallback(src, run_dir, prompt_path, model_spec)
    if suf in (".jpg", ".jpeg", ".png", ".bmp", ".webp") or src.is_dir():
        # 优先：OCR 重建带文本层 PDF → M9 文本优先管线（对齐精准）
        from .ocr_pdf import rebuild_pdf_from_images
        from .convert import normalize as _norm
        try:
            pages = _norm(src, run_dir / "pages")
            pdf = rebuild_pdf_from_images(src, run_dir, pages=pages)
            return process_pdf_textfirst(pdf, run_dir, prompt_path,
                                         model_spec, interactive)
        except Exception as e:  # noqa: BLE001 OCR 失败（无key/网络）降级
            print(f"OCR 重建失败（{e}），降级 CV+VLM 方案")
        return process_images_fallback(src, run_dir, prompt_path, model_spec)
    if suf in (".doc", ".docx"):
        from .convert import docx_to_pdf
        pdfs = docx_to_pdf(src, run_dir / "pages")
        pdf_path = run_dir / "pages" / (src.stem + ".pdf")
        return process_pdf_textfirst(pdf_path, run_dir, prompt_path, model_spec, interactive)
    raise ValueError(f"不支持的格式: {suf}")


if __name__ == "__main__":
    args = sys.argv[1:]
    _file, _i, _p, _m = None, False, None, None
    for a in args:
        if a in ("-i", "--interactive"):
            _i = True
        elif a.startswith("--prompt="):
            _p = a.split("=", 1)[1]
        elif a.startswith("-m=") or a.startswith("--model="):
            _m = a.split("=", 1)[1]
        elif not a.startswith("-"):
            _file = a
    main(_file or "samples/demo_homework.pdf", _i, _p, _m)
