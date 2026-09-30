"""GodApp —— 高中作业手写代笔助手（可分发应用）

首次使用：
    python god_app.py            进入交互式向导

交互约定（全程序统一）：
    · 菜单选择 → 输入数字
    · 确认/继续 → 回车
    · 返回上级 → 0

依赖安装：pip install -r requirements.txt
API Key：项目根目录 .env 文件写入一行  DASHSCOPE_API_KEY=sk-你的key
（阿里云百炼 https://bailian.console.aliyun.com 免费申领）
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

USER_FONT_DIR = ROOT / "fonts_user"
USER_FONT = USER_FONT_DIR / "my_font.ttf"
TRAIN_FONT = ROOT / "output" / "myhand_full.ttf"
FALLBACK = ROOT / "fonts" / "LXGWWenKai-Regular.ttf"


def _latest_review() -> Path | None:
    """最新一次作业的审核文件（output/runs/<作业名>_<时间戳>/m2_review.json）。"""
    runs = ROOT / "output" / "runs"
    if not runs.exists():
        return None
    cands = sorted(runs.glob("*/m2_review.json"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _utf8_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _pause():
    try:
        input("\n回车继续...")
    except EOFError:
        pass


def _run(*args: str, cwd: Path | None = None) -> bool:
    """运行子进程命令，返回是否成功。"""
    r = subprocess.run([sys.executable, *args], cwd=str(cwd or ROOT))
    return r.returncode == 0


def active_font() -> Path | None:
    """当前启用的字体：用户导入 > 本机训练 > 无。"""
    if USER_FONT.exists():
        return USER_FONT
    if TRAIN_FONT.exists():
        return TRAIN_FONT
    return None


def _font_chars(font: Path) -> int:
    from fontTools.ttLib import TTFont
    tt = TTFont(str(font))
    cps = set()
    for t in tt["cmap"].tables:
        if t.isUnicode():
            cps.update(t.cmap.keys())
    return len(cps)


# ---------- 1. 做作业 ----------

def do_homework():
    print("\n【开始做作业】")
    font = active_font()
    if font is None:
        print("  [!] 尚无个人手写字体，答案将以文楷字体誊写")
        print("      建议：先到「字体管理」训练或导入")
    else:
        print(f"  当前字体: {font.name}")
    print("  支持格式：PDF / JPG / PNG / Word(.docx) / 图片文件夹")
    print("  注意：PDF 与 Word 效果最佳；图片为 OCR 重建排版，")
    print("        拍摄照片识别效果不佳——请自行扫描文件后再上传")
    f = input("\n作业文件路径（拖入即可）: ").strip().strip('"').strip("'")
    src = Path(f)
    if not src.exists():
        print(f"  [!] 文件不存在: {f}")
        return
    if not _run("god.py", "homework", str(src)):
        print("  [!] 识别失败，请检查网络/API Key/文件后重试")
        return
    review = _latest_review()
    print(f"\n  答案已生成，审核文件: {review}")
    print("  （核对请修改其中的 answer 字段后保存）")
    mode = input("回车=直接渲染（信任AI）  m=我先核对再渲染: ").strip().lower()
    if mode == "m":
        print("  核对后运行: python god.py render")
        return
    _render()


def _render():
    review = _latest_review()
    if review is None:
        print("  [!] 未找到识别结果，请先做作业")
        return
    print(f"\n【渲染】审核文件: {review.parent.name}")
    if _run("god.py", "render", str(review)):
        print(f"\n  完成！产物在 {review.parent} 目录：")
        print("    *_完成版.pdf   ← 题目+手写答案（打印这份）")
        print("    *_答案层.pdf   ← 仅答案透明层")


# ---------- 2. 字体管理 ----------

TIERS = {
    "1": "极速版 · 300字（约30分钟书写，先试用）",
    "2": "完整版 · 1000字+标点（含标点/数字/字母99字符，最佳效果）",
}


def _train():
    print("\n【训练我的手写体】")
    print("  流程：生成采集表 → 打印书写 → 逐页拍照 → 合并 → 云端补全")
    print("  （云端补全需 AutoDL 租 GPU 约1小时、约2-5元）")
    print("\n  选择训练档位：")
    for k, desc in TIERS.items():
        print(f"    {k}. {desc}")
    print("    0. 返回")
    tier = input("\n  选择: ").strip()
    if tier not in TIERS:
        return
    print(f"\n  —— {TIERS[tier]} ——")
    print("\n  [1/5] 生成采集表...")
    _run("tools/font_cli.py", "sheet", "first")
    if tier == "2":
        _run("tools/font_cli.py", "sheet", "expansion")
        _run("tools/font_cli.py", "sheet", "punct")
        print("  已生成扩展采集表 + 标点采集表 samples/punct_sheet.pdf")
    print("\n  [2/5] 打印采集表并书写")
    print("    · 黑色签字笔，对照每格右上角灰色小字")
    print("    · 写大些、居中、不出格（同批次同一支笔）")
    print("    · 写错直接划掉，补在页尾空格")
    input("\n  写完并拍照后回车继续（照片重命名为 p1.jpg p2.jpg... 放到本文件夹）...")
    print("\n  [3/5] 合并手写照片...")
    photos = sorted(ROOT.glob("p*.jpg")) + sorted(ROOT.glob("p*.png"))
    if not photos:
        print("  [!] 未找到 p*.jpg 照片")
        return
    if not _run("tools/font_cli.py", "base", *(str(p) for p in photos)):
        print("  [!] 合并失败，请检查照片（四角锚点需完整）")
        return
    print(f"  已合并 {len(photos)} 张照片")
    print("\n  [4/5] 云端 AI 补全剩余汉字（可选，推荐）...")
    _run("tools/font_cli.py", "export")
    print("""    上传包已生成: output/m4/upload.zip
    云端操作（AutoDL 租卡约1小时、2-5元）：
      1. autodl.com 租 RTX 4090（PyTorch 镜像）
      2. 上传 upload.zip 到 /root/autodl-tmp 并解压
      3. git clone https://github.com/yeungchenwa/FontDiffuser.git
      4. 下载官方权重 unet/style_encoder/content_encoder 到 FontDiffuser/ckpt/
      5. 执行（跑完自动关机防扣费）:
         cd /root/autodl-tmp/upload
         nohup bash -c 'python batch_generate.py --ckpt_dir ../FontDiffuser/ckpt --style_refs_dir style_refs --target_chars_file target_chars.txt --ttf_path LXGWWenKai-Regular.ttf --save_dir ../results && cd /root/autodl-tmp && zip -r results.zip results && shutdown' > run.log 2>&1 &
      6. 下载 results.zip 放回本文件夹""")
    print("\n  [5/5] 当前已可用 300 字真迹字体（其余字暂用文楷）")
    print("  拿到 results.zip 后：字体管理 → 3 导入云端结果")


def _import_font():
    print("\n【导入字体包（TTF）】")
    print(f"  方式一：把 TTF 放入 {USER_FONT_DIR} 并命名为 my_font.ttf")
    print("  方式二：直接输入 TTF 路径，程序复制")
    f = input("\n  TTF 路径（回车=已手动放入）: ").strip().strip('"').strip("'")
    if f:
        src = Path(f)
        if not src.exists():
            print(f"  [!] 文件不存在: {f}")
            return
        USER_FONT_DIR.mkdir(exist_ok=True)
        shutil.copy2(src, USER_FONT)
    if USER_FONT.exists():
        try:
            n = _font_chars(USER_FONT)
            print(f"\n  启用成功：{USER_FONT.name}（{n} 字）")
            print("  之后所有作业誊写都使用这份字体")
        except Exception as e:
            print(f"  [!] 字体文件无效: {e}")
    else:
        print("  [!] fonts_user/my_font.ttf 不存在")


def _import_cloud():
    print("\n【导入云端生成结果（results.zip）】")
    zips = sorted(ROOT.glob("results*.zip"))
    if not zips:
        print("  [!] 未找到 results*.zip，请把云端下载的压缩包放到本文件夹")
        return
    for i, z in enumerate(zips, 1):
        print(f"    {i}. {z.name}  ({z.stat().st_size/1e6:.1f}MB)")
    pick = input("  选择序号（回车=最新）: ").strip()
    zp = zips[-1] if not pick else zips[int(pick) - 1]
    print(f"\n  应用 {zp.name} ...")
    if _run("tools/font_cli.py", "apply", str(zp)):
        _run("tools/font_cli.py", "rebuild")
        print("  完成！个人字库已启用（字体管理 → 4 查看状态）")


def _status():
    print("\n【字体状态】")
    font = active_font()
    if font is None:
        print("  尚未启用任何个人字体（作业将用文楷誊写）")
        print("  → 字体管理 → 1 训练 / 2 导入")
        return
    kind = "用户导入" if font == USER_FONT else "本机训练"
    try:
        n = _font_chars(font)
        print(f"  当前字体: {font.name}  [{kind}]  覆盖 {n} 字")
        freq = ROOT / "samples" / "charfreq_modern.txt"
        if freq.exists():
            top = [l.split("\t")[1].strip() for l in
                   freq.read_text(encoding="gbk", errors="ignore").splitlines()
                   if l.strip() and not l.startswith("/*") and "\t" in l][:1000]
            cps = _font_cps(font)
            cov = sum(1 for c in top if c and ord(c) in cps)
            print(f"  高频1000字覆盖: {cov}/1000（{cov/10:.0f}%）")
    except Exception as e:
        print(f"  [!] 读取失败: {e}")
    print(f"  回退字体: {FALLBACK.name}（个人字库缺字时顶替）")


def _font_cps(font: Path) -> set:
    from fontTools.ttLib import TTFont
    tt = TTFont(str(font))
    cps = set()
    for t in tt["cmap"].tables:
        if t.isUnicode():
            cps.update(t.cmap.keys())
    return cps


def font_menu():
    while True:
        print("""
———— 字体管理 ————
  1. 训练我的手写体
  2. 导入字体包（TTF）
  3. 导入云端结果（results.zip）
  4. 查看字体状态
  0. 返回""")
        c = input("  选择: ").strip()
        if c == "1":
            _train()
            _pause()
        elif c == "2":
            _import_font()
            _pause()
        elif c == "3":
            _import_cloud()
            _pause()
        elif c == "4":
            _status()
            _pause()
        elif c == "0":
            return
        else:
            print("  无效选择")


# ---------- 3. API 设置 ----------

VENDORS = {
    "1": ("relay", "中转站（OpenAI 兼容）",
          "需填 API Key 与接口地址，模型名可自定义（如 gpt-4o / claude / deepseek）"),
    "2": ("qwen", "阿里百炼 DashScope",
          "需填 API Key（bailian.console.aliyun.com 免费申领），模型如 qwen-max"),
    "3": ("zhipu", "智谱 bigmodel",
          "需填 API Key，模型如 glm-4.6"),
    "4": ("custom", "自定义 OpenAI 兼容接口",
          "填任意 base_url + key + 模型名（走 OpenAI 协议）"),
}


def _read_env() -> dict:
    env = ROOT / ".env"
    cfg = {}
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip().lstrip("\ufeff")  # 防御记事本BOM
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def _write_env(cfg: dict) -> None:
    """写回 .env（保留注释与未知键）。"""
    env = ROOT / ".env"
    old = {}
    lines = []
    if env.exists():
        lines = env.read_text(encoding="utf-8").splitlines()
        for line in lines:
            s = line.strip()
            if "=" in s and not s.startswith("#"):
                old[s.split("=", 1)[0].strip()] = line
    out = []
    written = set()
    # 按原顺序更新已有键
    for line in lines:
        s = line.strip()
        k = s.split("=", 1)[0].strip() if "=" in s and not s.startswith("#") else None
        if k in cfg:
            out.append(f"{k}={cfg[k]}")
            written.add(k)
        else:
            out.append(line)
    # 新键追加
    for k, v in cfg.items():
        if k not in written:
            out.append(f"{k}={v}")
    env.write_text("\n".join(out) + "\n", encoding="utf-8")


def do_api_config():
    print("\n【API 设置】（识别与作答走文本 LLM，配置保存在 .env）")
    cfg = _read_env()
    cur_vendor = cfg.get("VENDOR", "未设置")
    cur_model = cfg.get("MODEL", "未设置")
    cur_key = cfg.get("RELAY_API_KEY") or cfg.get("DASHSCOPE_API_KEY") or ""
    print(f"  当前: vendor={cur_vendor}  模型={cur_model}  "
          f"key={'已配置 ' + cur_key[:6] + '...' if cur_key else '未配置'}")
    print("\n  选择服务商：")
    for k, (vid, name, desc) in VENDORS.items():
        print(f"    {k}. {name}   {desc}")
    print("    0. 返回")
    c = input("\n  选择: ").strip()
    if c not in VENDORS:
        return
    vid, name, _ = VENDORS[c]

    api_key = input("  API Key（sk-...）: ").strip()
    model = input("  模型名（回车=默认）: ").strip() or None
    base_url = None
    if vid in ("relay", "custom"):
        base_url = input("  接口地址 base_url（如 https://xxx/v1）: ").strip()

    new = {"VENDOR": vid}
    if model:
        new["MODEL"] = model
    if api_key:
        if vid in ("relay", "custom"):
            new["RELAY_API_KEY"] = api_key
            if vid == "custom":
                new["RELAY_BASE_URL"] = base_url or "https://api.openai.com/v1"
        elif vid == "qwen":
            new["DASHSCOPE_API_KEY"] = api_key
        elif vid == "zhipu":
            new["ZHIPU_API_KEY"] = api_key
    if base_url and vid == "relay":
        new["RELAY_BASE_URL"] = base_url
    _write_env(new)
    print("\n  已保存到 .env")
    print(f"  下次做作业将使用: {new.get('MODEL', '默认模型')}（{name}）")


# ---------- 5. 渲染风格 ----------

STYLE_PATH = ROOT / "config" / "render_style.json"


def _read_style() -> dict:
    import json
    style = {"print_fade": 0.1, "ink_darken": 0.016}
    if STYLE_PATH.exists():
        try:
            style.update(json.loads(STYLE_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return style


def _fmt_fade(v: float) -> str:
    levels = ["不变", "微淡", "轻度", "中度", "明显", "重度"]
    return levels[min(int(round(v / 0.1)), 5)]


def _fmt_level(v: float, step: float) -> int:
    return int(round(v / step))


def do_render_style():
    import json
    print("\n【渲染风格】（效果在渲染完成版上体现，可反复调）")
    st = _read_style()
    fade_lv = _fmt_level(st["print_fade"], 0.1)
    dark_lv = _fmt_level(st["ink_darken"], 0.016)
    print(f"  当前: 题目变淡={fade_lv} 档({_fmt_fade(st['print_fade'])})  "
          f"手写加深={dark_lv} 档")
    print("""
  1. 题目变淡  0~5 档（减轻打印痕迹，让手写更突出）
  2. 手写加深  0~5 档（墨色更浓，0=不加深）
  3. 恢复默认（两项均为 1 档）
  0. 返回""")
    c = input("  选择: ").strip()
    if c == "1":
        v = input("  变淡档位 0-5（0=不变，1=推荐）: ").strip()
        if v.isdigit() and 0 <= int(v) <= 5:
            st["print_fade"] = int(v) * 0.1
    elif c == "2":
        v = input("  加深档位 0-5（0=不加深，1=推荐）: ").strip()
        if v.isdigit() and 0 <= int(v) <= 5:
            st["ink_darken"] = int(v) * 0.016
    elif c == "3":
        st = {"print_fade": 0.1, "ink_darken": 0.016}
    else:
        return
    STYLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STYLE_PATH.write_text(json.dumps(st, indent=2), encoding="utf-8")
    print(f"\n  已保存：题目变淡 {_fmt_level(st['print_fade'], 0.1)} 档，"
          f"手写加深 {_fmt_level(st['ink_darken'], 0.016)} 档")
    print("  下次渲染（做作业/渲染上次）生效")


# ---------- 主循环 ----------

def _check_deps() -> None:
    """启动依赖自检：缺包时给一键安装命令，而非深处报 ModuleNotFoundError。"""
    import importlib.util
    # 做作业主线必需（字体工具链的 potracer 等按需提示）
    critical = [("fitz", "pymupdf"), ("cv2", "opencv-python"),
                ("numpy", "numpy"), ("dashscope", "dashscope"),
                ("PIL", "Pillow"), ("openai", "openai")]
    missing = [pkg for mod, pkg in critical
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print("=" * 52)
    print("[!] 缺少运行依赖，请先安装（二选一）：")
    print("    方式一：在本文件夹打开命令行（地址栏输入 cmd 回车），执行：")
    print("      pip install -r requirements.txt "
          "-i https://mirrors.aliyun.com/pypi/simple/")
    print(f"    方式二：只装缺的 {len(missing)} 个：")
    print(f"      pip install {' '.join(missing)} "
          f"-i https://mirrors.aliyun.com/pypi/simple/")
    print("    （若提示 pip 不是命令，说明 Python 未加入 PATH，重装时勾选")
    print("      'Add Python to PATH'）")
    print("=" * 52)
    try:
        input("安装完成后重新启动本程序，回车退出...")
    except EOFError:
        pass
    sys.exit(1)


def main():
    _utf8_console()
    _check_deps()
    env = ROOT / ".env"
    if not env.exists() and not os.environ.get("DASHSCOPE_API_KEY"):
        print("[首次使用] 请在 .env 文件写入 API Key（一行）：")
        print("    DASHSCOPE_API_KEY=sk-你的key")
        print("（阿里云百炼控制台免费申领，识别作业需要它）\n")
    while True:
        print("""
══════════════════════════════════════════
  GodApp · 高中作业手写代笔助手
══════════════════════════════════════════
  1. 开始做作业    （AI识别作答→手写誊写）
  2. 渲染上次作业  （核对答案后重新誊写）
  3. 字体管理      （训练/导入/状态）
  4. API 设置      （服务商/Key/模型名）
  5. 渲染风格      （题目变淡/手写加深）
  0. 退出""")
        try:
            c = input("  选择: ").strip()
        except EOFError:
            return
        if c == "1":
            do_homework()
            _pause()
        elif c == "2":
            _render()
            _pause()
        elif c == "3":
            font_menu()
        elif c == "4":
            do_api_config()
            _pause()
        elif c == "5":
            do_render_style()
            _pause()
        elif c == "0":
            print("再见！")
            return
        else:
            print("  无效选择")


if __name__ == "__main__":
    main()
