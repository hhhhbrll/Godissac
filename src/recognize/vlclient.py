"""多后端 VLM 客户端：支持阿里百炼(DashScope) 和中转站(Relay)。

VENDOR 配置（优先级）：
1. .env 中的 VENDOR / API_KEY / BASE_URL
2. 命令行 -m/--model 参数指定 provider:model

provider 选项：
  qwen     → 阿里百炼 qwen-vl（原生 DashScope SDK）
  relay    → 中转站 OpenAI 兼容 API（任意模型）
  auto     → 先试 relay，没有 relay key 就用 qwen（默认）
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Literal

from dashscope import MultiModalConversation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = ROOT / "config" / "prompt.txt"

# 默认模型
DEFAULT_PROMPT = """你是严谨的作业答题专家。任务：在这张作业图片上找出所有需要学生填写的空位，给出精准bbox坐标和正确答案。

【空位类型说明】请仔细识别所有类型的待填区域：
1. 下划线空位：题目文字后的横线"____"，bbox从横线起点到终点
2. 方框空位：文中的"□"符号（文言文字词填空常见），bbox就是方框本身的边界
3. 括号空位："（  ）"、"( )"、"【  】"中间的空白，bbox是两括号之间的区域
4. 选择题括号：A/B/C/D 选项前的括号
5. 大题答题区：末尾多条横线组成的答题空白区域（翻译题、简答题、作文题），bbox覆盖整个空白区（从第一条横线上方到最后一条横线下方，左右到边界）
6. 田字格/方格：默写题的格子区域

【bbox坐标规则】（像素坐标，左上原点，极其重要，误差必须<5像素）：
- bbox格式：[x1, y1, x2, y2]
  · x1 = 空位左边缘
  · y1 = 空位上边缘（可写区域的最顶部，单下划线就是文字底部，大题就是第一行上方）
  · x2 = 空位右边缘
  · y2 = 空位下边缘（单下划线就是横线本身，大题就是最后一条横线）
- 对于"两根横线之间"的空位（大题多横线）：y1是上一条横线下方，y2是当前横线，即两条横线之间的区域（写字的地方）
- 对于"单横线到上方文字之间"的空位：y1是上方文字底部+2px，y2是横线，即文字与横线之间的空白（靠下区域写字）
- 方框□空位：x1,y1贴方框左上，x2,y2贴方框右下，不要超出方框
- 括号空位：x1=左括号右边，x2=右括号左边，y1=括号顶部，y2=括号底部
- 连续多条横线的大题：合并为一个空位，bbox覆盖整个答题区（从首线上到末线下），不要逐条拆分

【答案规则】（极其重要，严格执行）
- answer字段必须**只包含要填入的内容本身**，纯文本：
  · 禁止解释（不要"因为…所以…"）
  · 禁止前缀（不要"答："、"答案："、"第X题："）
  · 禁止重复题目文字
  · 不会/不确定的空位填 ""（空字符串），严禁猜测乱填
- 语文默写题必须用原文原字，不得增删字
- 选择题填选项字母（A/B/C/D）
- 方框填空（文言文字词解释）填字词或释义
- 姓名/班级/日期等个人信息栏填 ""
- 大题（翻译/简答）：answer填完整答案，渲染时自动换行

【输出格式】只输出一个JSON对象，不要输出任何其他文字、不要markdown、不要注释：
{
  "blanks": [
    {
      "id": 序号(整数，从1开始，按从上到下、从左到右阅读顺序),
      "bbox": [x1, y1, x2, y2],
      "type": "underline/box/bracket/block",
      "answer": "纯答案文本，不填则空字符串",
      "context": "空位前后10个字的题目文字（帮助人工核对，不要超过20字）"
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# 环境配置加载
# ---------------------------------------------------------------------------

def load_vendor_config(root: Path | None = None) -> dict:
    """从 .env 加载 VENDOR / API_KEY / BASE_URL / MODEL 配置。

    优先级：环境变量 > .env 文件中的同名 key。
    .env 文件支持：
      DASHSCOPE_API_KEY / API_KEY / RELAY_API_KEY
      RELAY_BASE_URL / BASE_URL
      VENDOR / MODEL
    """
    import os
    root = root or ROOT
    cfg = {"vendor": "auto", "api_key": None, "base_url": None, "model": None}

    def _get(key: str) -> str | None:
        """环境变量优先，其次 .env。"""
        v = os.environ.get(key)
        if v:
            return v
        return _read_env_line(key, root)

    for k, env_key in [
        ("vendor", "VENDOR"),
        ("api_key", "API_KEY"),
        ("base_url", "BASE_URL"),
        ("model", "MODEL"),
    ]:
        v = _get(env_key)
        if v:
            cfg[k] = v

    # 兼容：RELAY_* 覆盖通用字段
    relay_key = _get("RELAY_API_KEY")
    if relay_key:
        cfg["api_key"] = relay_key
    relay_url = _get("RELAY_BASE_URL")
    if relay_url:
        cfg["base_url"] = relay_url
    dash_key = _get("DASHSCOPE_API_KEY")
    # dashscope 是 qwen 后端的备选（relay key 不存在时用）
    if dash_key and not cfg["api_key"]:
        cfg["api_key"] = dash_key

    return cfg


def resolve_vendor_key(cfg: dict, explicit: str | None = None) -> tuple[Literal["qwen", "relay", "zhipu"], str, str | None]:
    """解析出实际使用的 vendor + api_key + base_url。

    参数 explicit: 命令行指定的 "provider:model" 或 "model" 字符串
    返回 (vendor, api_key, base_url)

    自动回退：配置的 vendor 缺 key 时，按 relay → qwen → zhipu
    找第一个有 key 的（用户可能只配了任意一种 API）。
    """
    # 显式指定 provider:model
    if explicit:
        if ":" in explicit:
            prov, model = explicit.split(":", 1)
            cfg["model"] = model
            cfg["vendor"] = prov
        else:
            cfg["model"] = explicit

    vendor = cfg["vendor"]

    # auto: 优先 relay
    if vendor == "auto":
        if cfg.get("api_key") and cfg.get("base_url"):
            vendor = "relay"
        else:
            vendor = "qwen"

    # 自动回退：缺 key 的 vendor 换成有 key 的
    def _key_of(v: str) -> str | None:
        if v == "qwen":
            return _read_env_line("DASHSCOPE_API_KEY", ROOT) or cfg.get("api_key")
        if v == "zhipu":
            return _read_env_line("ZHIPU_API_KEY", ROOT) or cfg.get("api_key")
        return cfg.get("api_key") or _read_env_line("RELAY_API_KEY", ROOT)

    if not _key_of(vendor):
        for alt in ("relay", "qwen", "zhipu"):
            if _key_of(alt):
                print(f"  [提示] 视觉后端 {vendor} 未配置 API Key，自动改用 {alt}")
                vendor = alt
                break

    if vendor == "qwen":
        # 优先用 DASHSCOPE_API_KEY（避免被 RELAY_API_KEY 覆盖）
        key = _read_env_line("DASHSCOPE_API_KEY", ROOT)
        if not key:
            key = cfg.get("api_key")
        if not key:
            raise RuntimeError("未找到任何 API Key：请在 .env 配置 DASHSCOPE_API_KEY / RELAY_API_KEY / ZHIPU_API_KEY 任意一个")
        return "qwen", key, None  # 百炼不需要 base_url

    elif vendor == "zhipu":
        key = _read_env_line("ZHIPU_API_KEY", ROOT) or cfg.get("api_key")
        if not key:
            raise RuntimeError("未找到任何 API Key：请在 .env 配置 ZHIPU_API_KEY / DASHSCOPE_API_KEY / RELAY_API_KEY 任意一个")
        url = (
            _read_env_line("ZHIPU_BASE_URL", ROOT)
            or cfg.get("base_url")
            or "https://open.bigmodel.cn/api/paas/v4/"
        )
        return "zhipu", key, url

    elif vendor == "relay":
        key = cfg.get("api_key") or _read_env_line("RELAY_API_KEY", ROOT)
        if not key:
            raise RuntimeError("未找到任何 API Key：请在 .env 配置 RELAY_API_KEY / DASHSCOPE_API_KEY / ZHIPU_API_KEY 任意一个")
        url = cfg.get("base_url") or _read_env_line("RELAY_BASE_URL", ROOT) or "https://your-relay.example.com/v1"
        return "relay", key, url

    raise ValueError(f"未知 vendor: {vendor}")


def _read_env_line(key: str, root: Path | None = None) -> str | None:
    root = root or ROOT
    env_file = root / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return None


# ---------------------------------------------------------------------------
# Prompt 加载
# ---------------------------------------------------------------------------

def load_prompt(prompt_path: Path | None = None) -> str:
    """加载提示词：优先指定路径，否则 config/prompt.txt，最后内置默认。"""
    paths = []
    if prompt_path is not None:
        paths.append(prompt_path)
    paths.append(DEFAULT_PROMPT_PATH)
    for p in paths:
        if p.exists():
            text = p.read_text(encoding="utf-8").strip()
            if text:
                return text
    return DEFAULT_PROMPT


# ---------------------------------------------------------------------------
# 通用 JSON 解析
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """容错解析：VLM 有时会把 JSON 包在 ```json ``` 里或夹杂说明文字。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            text = text[s : e + 1]
    return json.loads(text)


# ---------------------------------------------------------------------------
# 后端 A: 阿里百炼（DashScope）
# ---------------------------------------------------------------------------

DASHSCOPE_MODEL = "qwen3.7-plus"


def _call_dashscope(api_key: str, model: str, messages: list,
                    prompt: str, url: str, timeout: int = 180) -> dict:
    """调用阿里百炼，返回 {"answers": {"1": "...", "2": "...", ...}}"""
    last_err = None
    for attempt in range(3):
        try:
            resp = MultiModalConversation.call(
                model=model,
                api_key=api_key,
                messages=messages,
                timeout=timeout,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"API 返回 {resp.status_code}: {getattr(resp, 'message', '')}")
            text = "".join(seg.get("text", "") for seg in
                           resp.output.choices[0].message.content)
            return _parse_answers(text)
        except Exception as e:
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"  ⚠ DashScope 调用失败(第{attempt + 1}次): {e} —— {wait}秒后重试")
            time.sleep(wait)
    raise RuntimeError(f"DashScope 连续 3 次调用失败: {last_err}")


# ---------------------------------------------------------------------------
# 后端 B: 中转站（OpenAI 兼容）
# ---------------------------------------------------------------------------

def _call_relay(api_key: str, base_url: str, model: str,
                prompt: str, image_path: Path, timeout: int = 180) -> dict:
    """调用中转站 OpenAI 兼容 API，返回 {"answers": {...}}。"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    # 图片转 base64
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
                        {"type": "text", "text": prompt},
                    ],
                }],
                timeout=timeout,
            )
            text = resp.choices[0].message.content
            return _parse_answers(text)
        except Exception as e:
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"  ⚠ Relay 调用失败(第{attempt + 1}次): {e} —— {wait}秒后重试")
            time.sleep(wait)
    raise RuntimeError(f"Relay 连续 3 次调用失败: {last_err}")


# ---------------------------------------------------------------------------
# 答案解析（通用）
# ---------------------------------------------------------------------------

def _parse_answers(text: str) -> dict:
    """从 VLM 响应文本中提取答案字典 {"1": "...", "2": "...", ...}。"""
    data = _extract_json(text)
    # M7 格式：{"answers": {"1": "...", ...}}
    if "answers" in data:
        raw = data["answers"]
        if isinstance(raw, dict):
            return raw
        return {}
    # 旧格式兜底：{"blanks": [{"id": 1, "answer": "..."}]}
    if "blanks" in data:
        return {str(b["id"]): b.get("answer", "") for b in data["blanks"] if "id" in b}
    # 直接是 dict
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------

def answer_by_ids(
    labeled_image_path: str | Path,
    blank_ids: list[int],
    contexts: list[str],
    api_key: str,
    prompt: str | None = None,
    prompt_path: str | Path | None = None,
    *,
    vendor: Literal["qwen", "relay"] = "qwen",
    relay_base_url: str | None = None,
    relay_model: str | None = None,
    dashscope_model: str | None = None,
    timeout: int = 180,
) -> dict[int, str]:
    """VLM 按编号返回每空答案。

    参数：
        labeled_image_path : 已编号的 PNG 图（红圈白字）
        blank_ids         : 空位编号列表（与图上编号对应）
        contexts          : 每个空位的题目上下文（与 blank_ids 一一对应）
        api_key           : DashScope 或 Relay API Key
        prompt            : 提示词（优先），否则从 prompt_path 加载
        prompt_path       : prompt.txt 路径
        vendor            : "qwen"（百炼）或 "relay"（中转站）
        relay_base_url    : relay 专属 base_url（如 https://your-relay.example.com/v1）
        relay_model       : relay 专用模型名（如 qwen3-vl-235b-a22b-thinking）
        dashscope_model   : 百炼模型名（如 qwen3-vl-plus）
        timeout           : 超时秒数

    返回：{编号(int): 答案文字}
    """
    labeled_png = Path(labeled_image_path)
    if prompt is None:
        prompt = load_prompt(Path(prompt_path) if prompt_path else None)

    n = len(blank_ids)
    # 构建题目上下文：明确标注 blank_id → 题目
    ctx_lines = "\n".join(
        f"[blank#{bid}] {ctx}" for bid, ctx in zip(blank_ids, contexts)
    )

    instruction = f"""这是一张高中语文文言文作业图片，上面已用红色圆圈标注了需要作答的空位，圈内数字是空位编号(blank#)。

【重要】本题型说明：
这是高中语文文言文阅读题。图片中红色圆圈标注的是"加点字词解释"空位——括号/方框内需要填入加点字的**现代汉语释义**。

【空位与题目上下文】
{ctx_lines}

【任务】
看图，识别每个编号(blank#)空位上应该填写的正确答案（加点字的现代汉语释义）。

【严格限制】
- 字词解释题：答案必须简短，1-3个字（如"种植""吟咏""看待"），绝对不能超过3个字
- 选择题：填选项字母（如"A"）
- 大题：填完整答案（可较长）
- 只填答案本身，不要前缀、不要解释

【输出格式】（严格只输出 JSON，key 必须是 blank# 编号）：
{{"answers": {{"blank#编号": "该空的答案", ...}}}}"""

    if vendor in ("relay", "zhipu"):
        # relay 和 zhipu 都是 OpenAI 兼容 API（智谱 bigmodel.cn 也是）
        # 注意模型名按厂区分：zhipu 无 qwen 系列，视觉用 glm-4v-plus
        model = relay_model or ("glm-4v-plus" if vendor == "zhipu"
                                else "qwen3.7-plus")
        base_url = relay_base_url or "https://your-relay.example.com/v1"
        raw = _call_relay(api_key, base_url, model, instruction, labeled_png, timeout)
    else:
        model = dashscope_model or DASHSCOPE_MODEL
        url = f"file://{labeled_png.as_posix()}"
        messages = [{"role": "user", "content": [{"image": url}, {"text": instruction}]}]
        raw = _call_dashscope(api_key, model, messages, instruction, url, timeout)

    # 统一转 {int: str}
    out: dict[int, str] = {}
    for i, bid in enumerate(blank_ids):
        # 优先级：找 "blank#N" 格式的 key，再找纯数字 N，再找位置 i+1
        v = ""
        key_blank = f"blank#{bid}"
        if key_blank in raw:
            v = str(raw[key_blank])
        elif bid in raw:
            v = str(raw[bid])
        elif str(bid) in raw:
            v = str(raw[str(bid)])
        elif i + 1 in raw:
            v = str(raw[i + 1])
        elif str(i + 1) in raw:
            v = str(raw[str(i + 1)])
        out[bid] = v.strip()
    return out


# ---------------------------------------------------------------------------
# 兼容旧接口（analyze_page）
# ---------------------------------------------------------------------------

def analyze_page(image_path: str | Path, api_key: str,
                 prompt: str | None = None,
                 prompt_path: Path | None = None,
                 **kwargs) -> dict:
    """【已废弃】一阶段：识别空位 + 给坐标 + 给答案。
    保留此函数仅为兼容旧调用路径（CV+VLM 降级方案）。

    后端自动选择：DASHSCOPE 缺失时自动用已配置的其他后端
    （用户可能只配了任意一种 API），视觉模型按厂区分。
    """
    if prompt is None:
        prompt = load_prompt(prompt_path)
    url = f"file://{Path(image_path).as_posix()}"

    # 后端选择：优先 DASHSCOPE（dashscope SDK），否则 OpenAI 兼容
    cfg = load_vendor_config()
    vendor, vkey, vurl = resolve_vendor_key(cfg)
    if vendor == "qwen":
        messages = [{"role": "user", "content": [{"image": url}, {"text": prompt}]}]
        return _analyze_via_dashscope(api_key or vkey, messages, prompt, url)
    model = kwargs.get("relay_model") or (
        "glm-4v-plus" if vendor == "zhipu" else "qwen3.7-plus")
    raw_text = _call_relay_text(vkey, vurl, model, prompt,
                                Path(image_path), 180)
    blanks = _extract_json(raw_text).get("blanks", [])
    for b in blanks:
        if "id" not in b:
            b["id"] = 0
        if "bbox" not in b or len(b["bbox"]) != 4:
            continue
        if "answer" not in b:
            b["answer"] = ""
        if "type" not in b:
            b["type"] = "underline"
        if "context" not in b:
            b["context"] = ""
        x1, y1, x2, y2 = b["bbox"]
        b["bbox"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return {"blanks": blanks}


def _analyze_via_dashscope(api_key: str, messages: list,
                           prompt: str, url: str) -> dict:
    last_err = None
    for attempt in range(3):
        try:
            resp = MultiModalConversation.call(
                model=DASHSCOPE_MODEL,
                api_key=api_key,
                messages=messages,
                timeout=180,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"API 返回 {resp.status_code}: {getattr(resp, 'message', '')}")
            text = "".join(seg.get("text", "") for seg in
                           resp.output.choices[0].message.content)
            data = _extract_json(text)
            blanks = data.get("blanks", [])
            for b in blanks:
                if "id" not in b:
                    b["id"] = 0
                if "bbox" not in b or len(b["bbox"]) != 4:
                    continue
                if "answer" not in b:
                    b["answer"] = ""
                if "type" not in b:
                    b["type"] = "underline"
                if "context" not in b:
                    b["context"] = ""
                x1, y1, x2, y2 = b["bbox"]
                b["bbox"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            return {"blanks": blanks}
        except Exception as e:
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"  ⚠ API 调用失败(第{attempt + 1}次): {e} —— {wait}秒后重试")
            time.sleep(wait)
    raise RuntimeError(
        f"API 连续 3 次调用失败: {last_err}\n"
        "请检查：① .env 里的 API Key 是否有效 "
        "② 网络/代理 ③ 控制台余额与限流")


def _call_relay_text(api_key: str, base_url: str, model: str,
                     prompt: str, image_path: Path, timeout: int) -> str:
    """OpenAI 兼容接口的原始文本返回（不带答案解析）。"""
    from openai import OpenAI
    import base64 as _b64
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    b64_img = _b64.b64encode(image_path.read_bytes()).decode("utf-8")
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
                        {"type": "text", "text": prompt},
                    ],
                }],
                timeout=timeout,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"  ⚠ API 调用失败(第{attempt + 1}次): {e} —— {wait}秒后重试")
            time.sleep(wait)
    raise RuntimeError(
        f"API 连续 3 次调用失败: {last_err}\n"
        "请检查：① .env 里的 API Key 是否有效 "
        "② 网络/代理 ③ 控制台余额与限流")


# ---------------------------------------------------------------------------
# 环境变量兼容
# ---------------------------------------------------------------------------

import os
def load_api_key(root: Path | None = None) -> str:
    """从 .env 读取 DASHSCOPE_API_KEY（兼容旧代码）。"""
    env = os.environ.get("DASHSCOPE_API_KEY")
    if env:
        return env
    key = _read_env_line("DASHSCOPE_API_KEY", root)
    if key:
        return key
    raise RuntimeError(
        "未找到 API Key：请在 .env 中设置 DASHSCOPE_API_KEY=sk-xxx 或 RELAY_API_KEY=sk-xxx")
