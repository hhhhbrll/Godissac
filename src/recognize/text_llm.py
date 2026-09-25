"""文本 LLM 客户端（M9 文本优先架构）：纯文本作答，不用视觉模型。

优势：
- 输入是带【N】标记的作业全文 → 答案与空位按编号对齐，永不漂移
- LLM 能读到完整文章 + 题目 + 参考译文（若卷面上有，答案质量极高）
- 文本调用比视觉调用快 ~5倍、便宜 ~10倍

后端（.env 配置，与 vlclient.py 共用）：
- relay  → 中转站 OpenAI 兼容 API（gpt-4o 等，推荐，已实测）
- qwen   → 阿里百炼 DashScope（qwen-max 等文本模型）
- zhipu  → 智谱 bigmodel.cn OpenAI 兼容（glm-4.6 等）
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = ROOT / "config" / "prompt.txt"


# ---------------------------------------------------------------------------
# 配置（与 vlclient.py 的 .env 约定一致）
# ---------------------------------------------------------------------------

def load_env_config() -> dict:
    cfg = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    import os
    for k in ("RELAY_API_KEY", "RELAY_BASE_URL", "DASHSCOPE_API_KEY",
              "ZHIPU_API_KEY", "ZHIPU_BASE_URL", "VENDOR", "MODEL"):
        if os.environ.get(k) and k not in cfg:
            cfg[k] = os.environ[k]
    return cfg


def _backend_key(cfg: dict, backend: str) -> str | None:
    """某后端已配置的 key。"""
    if backend == "relay":
        return cfg.get("RELAY_API_KEY")
    if backend == "qwen":
        return cfg.get("DASHSCOPE_API_KEY")
    if backend == "zhipu":
        return cfg.get("ZHIPU_API_KEY")
    return None


def _make_backend(cfg: dict, backend: str, model: str | None):
    """按后端组装 (backend, key, base_url, model)。gpt-4o 是 relay 专属
    默认名，其他后端遇到自动置空回退本厂默认。"""
    if backend in ("relay", "custom"):
        # custom = 任意 OpenAI 兼容接口（用户自带 base_url/key/模型名）
        return "relay", cfg.get("RELAY_API_KEY"), \
            cfg.get("RELAY_BASE_URL", "https://your-relay.example.com/v1"), \
            model or "gpt-4o"
    if model == "gpt-4o":
        model = None  # 防呆：relay 默认模型名在其他厂不存在
    if backend == "qwen":
        return "qwen", cfg.get("DASHSCOPE_API_KEY"), None, model or "qwen-max"
    if backend == "zhipu":
        return "zhipu", cfg.get("ZHIPU_API_KEY"), \
            cfg.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"), \
            model or "glm-4.6"
    raise ValueError(f"未知后端: {backend}（可选 relay / qwen / zhipu）")


def resolve_backend(cfg: dict, model_spec: str | None = None):
    """返回 (backend, api_key, base_url, model)。

    自动回退：用户可能只配置了任意一种 API（.env 里只有其中一个 key）。
    显式 -m 指定时尊重用户选择（失败直接报错）；否则按
    VENDOR → relay → qwen → zhipu 的顺序找第一个有 key 的后端。
    （曾因 VENDOR=zhipu + MODEL=gpt-4o 或只配单一 key 导致全量调用失败）
    """
    if model_spec and ":" in model_spec:
        backend, model = model_spec.split(":", 1)
        return _make_backend(cfg, backend, model)  # 显式指定不回退

    want = cfg.get("VENDOR", "relay")
    model = model_spec or cfg.get("MODEL")
    order = [want] + [b for b in ("relay", "qwen", "zhipu") if b != want]
    for b in order:
        if _backend_key(cfg, b):
            if b != want:
                print(f"  [提示] 后端 {want} 未配置 API Key，自动改用 {b}")
                # 跨厂回退：MODEL 大概率是原厂模型名（如 glm-4.6 带到
                # relay 会 model_not_found），置空用目标厂默认
                model = None
            return _make_backend(cfg, b, model)
    raise RuntimeError(
        "未找到任何 API Key：请在 .env 配置 RELAY_API_KEY / "
        "DASHSCOPE_API_KEY / ZHIPU_API_KEY 任意一个"
        "（或用软件菜单「API 设置」自动写入）")


# ---------------------------------------------------------------------------
# 调用
# ---------------------------------------------------------------------------

def answer_homework(marked_text: str, blanks_spec: int | list[int],
                    prompt: str | None = None,
                    prompt_path: Path | None = None,
                    model_spec: str | None = None,
                    debug_path: Path | None = None) -> dict[str, str]:
    """让文本 LLM 作答。返回 {编号字符串: 答案}。自动重试3次。

    blanks_spec:
        int     → 编号 1..N 连续（单篇作业）
        list[int] → 本批篇章的空位编号集合（大批量按篇章分批，见 article_split）
    分批策略：>45 个空位自动分批（中转站对单次响应长度有隐性上限），
    每批只答指定编号。
    """
    if prompt is None:
        prompt = load_prompt(prompt_path)

    if isinstance(blanks_spec, int):
        numbers = list(range(1, blanks_spec + 1))
    else:
        numbers = sorted(set(int(x) for x in blanks_spec))

    BATCH = 45
    all_answers: dict[str, str] = {}
    for i in range(0, len(numbers), BATCH):
        chunk = numbers[i:i + BATCH]
        part = _answer_batch(marked_text, chunk, prompt, model_spec,
                             debug_path if i == 0 else None)
        all_answers.update(part)
        if i + BATCH < len(numbers):
            print(f"  批 [{chunk[0]}~{chunk[-1]}] 完成")
    for no in numbers:
        all_answers.setdefault(str(no), "")
    return all_answers


def _answer_batch(marked_text: str, numbers: list[int], prompt: str,
                  model_spec: str | None, debug_path: Path | None) -> dict[str, str]:
    """单批作答：只答 numbers 中的编号。"""
    cfg = load_env_config()
    backend, api_key, base_url, model = resolve_backend(cfg, model_spec)
    if not api_key:
        raise RuntimeError(f"后端 {backend} 未配置 API Key，请检查 .env")

    n = len(numbers)
    lo, hi = numbers[0], numbers[-1]
    contiguous = numbers == list(range(lo, hi + 1))
    nos_desc = (f"编号【{lo}】到【{hi}】" if contiguous
                else "编号 " + "、".join(f"【{x}】" for x in numbers))
    full_prompt = (f"{prompt}\n\n【作业全文（含【N】空位标记）】\n{marked_text}\n\n"
                   f"【注意】本批只作答{nos_desc}共 {n} 个空位标记，"
                   f"其他编号一律不要输出。"
                   f"只输出**紧凑单行**JSON：{{\"{numbers[0]}\":\"答案\",\"{numbers[1] if n > 1 else numbers[0]}\":\"答案\",...}}"
                   f"——禁止换行、禁止缩进（防止超长截断）。"
                   f"{n} 个编号全部必须有答案（不确定填空字符串）。")

    last_err = None
    for attempt in range(3):
        try:
            if backend in ("relay", "zhipu"):
                text = _call_openai(api_key, base_url, model, full_prompt,
                                    max_tokens=6000)
            else:
                text = _call_dashscope(api_key, model, full_prompt)
            if debug_path is not None:
                try:
                    debug_path.write_text(text, encoding="utf-8")
                except Exception:
                    pass
            parsed = _parse_answers(text, n)
            # 只保留本批编号范围的键（_parse_answers 会补默认空键，
            # 不滤掉会用空值覆盖前一批的答案）
            numset = {str(x) for x in numbers}
            return {k: v for k, v in parsed.items() if k in numset}
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"  ⚠ {backend} 调用失败(第{attempt+1}次): {e} —— {wait}秒后重试")
            time.sleep(wait)
    raise RuntimeError(f"LLM 连续3次调用失败: {last_err}")


def refine_answers(items: list[dict], model_spec: str | None = None) -> dict[str, str]:
    """精炼括号答案：①剔除从原句带入的无关词语（纯净性）②压缩到K字。

    items: [{"no": 编号, "k": 目标字数, "ctx": 前文, "after": 后文, "ans": 原答案}, ...]
    返回 {编号字符串: 精炼后答案}。失败时返回空 dict（渲染端有兜底截断）。
    """
    if not items:
        return {}
    cfg = load_env_config()
    backend, api_key, base_url, model = resolve_backend(cfg, model_spec)
    if not api_key:
        return {}
    lines = "\n".join(
        f"{it['no']}|{it['k']}|{it['ctx'][-10:]}|{it['ans']}"
        for it in items)
    prompt = (
        "你是文言文释义精炼助手。每行格式：编号|目标字数|空位前文|原答案。\n"
        "【背景】空位前的1-2个字是待解释的文言词（如前文末尾\"泊如\"→待释词=泊如），"
        "原答案是它的释义但超长了。\n"
        "压缩规则：\n"
        "①字数必须≤目标字数（硬性要求）\n"
        "②压缩后的答案必须仍然是\"待释词的现代汉语释义\"——删的是释义里的次要字，"
        "不是换成本文原句的字（如前文\"泊如\"原答案\"淡泊如水\"→✓\"淡泊\" ✗\"如水\"；"
        "前文\"上以为然\"原答案\"认为对\"→✓\"对\" ✗\"以然\"）\n"
        "③若原答案混入了前文/后文的原句词语，剔除它们\n"
        "④单字母选项原样返回\n"
        f"{lines}\n"
        "只输出JSON：{\"编号\": \"压缩后答案\", ...}，编号与输入一致。")
    try:
        if backend in ("relay", "zhipu"):
            text = _call_openai(api_key, base_url, model, prompt, timeout=60)
        else:
            text = _call_dashscope(api_key, model, prompt, timeout=60)
        m = re.search(r"\{.*\}", text, re.S)
        return {str(k): str(v) for k, v in json.loads(m.group(0)).items()} if m else {}
    except Exception:  # noqa: BLE001 精炼失败不致命
        return {}


def purity_pass(items: list[dict], model_spec: str | None = None) -> dict[str, str]:
    """答案纯净性终检：只返回需要修正的项（避免"全部原样返回"陷阱）。

    items: [{"no": 编号, "k": 容量, "ctx": 前文, "ans": 答案}, ...]
    返回 {编号: 修正后答案}（仅含需修正项；空dict=全部纯净）。
    """
    if not items:
        return {}
    cfg = load_env_config()
    backend, api_key, base_url, model = resolve_backend(cfg, model_spec)
    if not api_key:
        return {}
    lines = "\n".join(
        f"{it['no']}|{it['k']}|{it['ctx'][-10:]}|{it['ans']}" for it in items)
    prompt = (
        "你是文言文答案质检员。每行格式：编号|容量字数|空位前文|答案。\n"
        "空位前文末尾的1-2个字是待释的文言词。逐条检查答案是否违规：\n"
        "A.混入了原句（前文）中的词语（如前文\"造．竹所者\"答案\"到竹林\"——\"竹\"来自原句，应改\"到\"）\n"
        "B.解释的是短语而非带．的加点字（如前文\"蚤．世\"答案\"早逝\"解释了\"蚤世\"，应只释\"蚤\"→\"早\"）\n"
        "C.字数超过容量\n"
        "D.不是现代汉语释义（是从原文摘的字）\n"
        "【重要】以下情况绝对不要修正、不要输出：\n"
        "- 单字母选项（A/B/C/D）——选择题答案永远正确，直接跳过\n"
        "- 看起来已经纯净合规的答案\n"
        "只输出确定违规项的修正：{\"编号\": \"修正后答案\", ...}，全部合规输出{}。\n"
        f"{lines}")
    try:
        if backend in ("relay", "zhipu"):
            text = _call_openai(api_key, base_url, model, prompt, timeout=90)
        else:
            text = _call_dashscope(api_key, model, prompt, timeout=90)
        m = re.search(r"\{.*\}", text, re.S)
        return {str(k): str(v) for k, v in json.loads(m.group(0)).items()} if m else {}
    except Exception:  # noqa: BLE001 质检失败不致命
        return {}


def _call_openai(api_key: str, base_url: str, model: str, prompt: str,
                 timeout: int = 180, max_tokens: int | None = None) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    kwargs = {}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        **kwargs,
    )
    return resp.choices[0].message.content or ""


def _call_dashscope(api_key: str, model: str, prompt: str,
                    timeout: int = 180) -> str:
    import dashscope
    resp = dashscope.Generation.call(
        model=model, api_key=api_key,
        messages=[{"role": "user", "content": prompt}],
        result_format="message", timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API 返回 {resp.status_code}: {resp.message}")
    return resp.output.choices[0]["message"]["content"]


def _parse_answers(text: str, blank_count: int) -> dict[str, str]:
    """容错解析 JSON 答案表。缺失编号补空串。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            text = text[s:e + 1]
    data = json.loads(text)
    result: dict[str, str] = {}
    if isinstance(data, dict):
        items = data.get("answers", data)
        if isinstance(items, dict):
            for k, v in items.items():
                result[str(k)] = str(v).strip() if v is not None else ""
    # 兜底补齐
    for i in range(1, blank_count + 1):
        result.setdefault(str(i), "")
    return result


def load_prompt(prompt_path: Path | None = None) -> str:
    paths = [p for p in (prompt_path, DEFAULT_PROMPT_PATH) if p]
    for p in paths:
        if p.exists():
            t = p.read_text(encoding="utf-8").strip()
            if t:
                return t
    return _FALLBACK_PROMPT


_FALLBACK_PROMPT = """你是一名严谨的高中老师，正在为学生完成作业。

下面是一份作业的完整文本（按阅读顺序提取，可能跨页）。文中【1】【2】【3】…方括号数字标记处是学生需要填写的空位；原文中的①②③小圆圈数字是注释编号，与空位无关。

请阅读全文后，给出每个【N】标记处的正确答案：
- 括号紧跟词语后的（如"环植【1】"）：填该词在文中的释义（1-4字现代汉语）
- 选择题括号：填选项字母（A/B/C/D）
- 翻译/简答大题：填完整答案
- 答案长度需与空位宽度匹配：窄括号1字，宽括号2-4字
- 如果卷面附有"参考译文"，请用它校准答案
- 只输出JSON：{"1": "答案", ...}，不加任何解释"""
