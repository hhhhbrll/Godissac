"""代笔软件统一入口（M9 文本优先架构）：作业文件 → 手写答案 PDF。

用法：
    python god.py homework <作业文件>    分步模式：可修改prompt → AI作答 → 人工审核 → 渲染
    python god.py auto <作业文件>        全自动：识别作答后直接渲染（信任AI）
    python god.py render [审核文件]      审核修改后重新渲染（默认最新一次作业）

架构（M9）：
    PDF 有文本层 → 空位坐标从字符bbox提取（100%精准）→ 重建全文嵌入【N】标记
    → 文本LLM一次作答（对齐率100%，LLM能看到完整文章+参考译文）→ 手写字体渲染
    照片/扫描件（无文本层）→ 降级 CV横线检测 + VLM 方案

支持格式：PDF / JPG / PNG / BMP / WebP / Word(.doc/.docx)
注意：请优先使用含文本层的PDF（电子版/扫描版OCR）；纯照片效果有限。

提示词基础模板在 config/prompt.txt，每次作业复制到输出目录可修改。
模型后端在 .env 配置（VENDOR=relay/qwen/zhipu + MODEL），默认中转站 gpt-4o。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "output" / "runs"


def _latest_review() -> Path | None:
    """最新一次作业的审核文件。"""
    if not RUNS.exists():
        return None
    cands = sorted(RUNS.glob("*/m2_review.json"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _utf8_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _run(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(f"\n[失败] {module} 退出码 {r.returncode}，请检查上方输出")


def do_homework(file: str) -> None:
    """识别作答（交互模式：暂停让用户改prompt）→ 停下等人工审核答案。"""
    src = Path(file)
    if not src.exists():
        sys.exit(f"文件不存在: {src}")
    print(f"[1/2] VLM识别空位并作答（交互模式，作答前可修改prompt）: {src.name}")
    _run("src.recognize.pipeline", str(src), "-i")
    review = _latest_review()
    print("\n[2/2] 请人工审核答案")
    print(f"    打开 {review}")
    print("    核对/修改每项的 answer 字段")
    print("    改完运行: python god.py render")


def do_render(review_path: str | None = None) -> None:
    if review_path:
        rp = Path(review_path)
    else:
        rp = _latest_review()
        if rp is None:
            sys.exit("未找到任何识别结果（先运行 python god.py homework <文件>）")
    if not rp.exists():
        sys.exit(f"审核文件不存在: {rp}")
    _run("src.render.render_answered", str(rp))


def do_auto(file: str) -> None:
    """全自动：不暂停改prompt，识别后直接渲染。"""
    src = Path(file)
    if not src.exists():
        sys.exit(f"文件不存在: {src}")
    print(f"[1/2] 全自动识别作答: {src.name}")
    _run("src.recognize.pipeline", str(src))
    print(f"[2/2] 直接渲染...")
    do_render()


def main() -> None:
    _utf8_console()
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = args[0], args[1:]

    if cmd == "homework" and len(rest) >= 1:
        do_homework(rest[0])
    elif cmd == "auto" and len(rest) == 1:
        do_auto(rest[0])
    elif cmd == "render" and len(rest) <= 1:
        do_render(rest[0] if rest else None)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
