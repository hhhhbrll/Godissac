"""个人字库全流程 CLI：采集 → 手写合并 → 云端生成 → 重建 → 验收，一条命令链。

用法：
    python tools/font_cli.py <子命令> [参数]

子命令：
    status                    字库概览（字数/真迹数/最近备份）
    sheet first  [--n N]      生成首批高频字采集表（默认300字）
    sheet expansion [--extra 字串]
                              生成扩展采集表（top1000 − 已采集，追加点名缺字）
    sheet punct               生成标点/数字/字母采集表（99字符）
    sheet list @清单.json [--extra 字串]
                              自定义清单采集表（如离群补录清单）
    base <照片...>            首批手写 → myhand.ttf（M3 基础字库）
    hand <照片...> [--chars xxx_chars.json]
                              手写批次并入 myhand_full.ttf（就地替换/新增，
                              自动维护真迹清单）
    export                    打包云端上传包 upload.zip（参考图=当前全部真迹）
    regen                     打包补齐包 upload_regen.zip（丢弃字+字频表生僻字）
    apply <results.zip>       应用云端生成结果到字库（就地替换，断点续跑）
    rebuild                   全量重建 myhand_full.ttf（合并管线，断点续跑）
    fix-punct                 标点排版规范修复（位置+笔宽，需 punct 批次）
    verify                    闭环验证渲染（demo 作业 → 完成版 PDF）
    guide                     交互式向导：从零建库的分步指引

典型全流程（新用户）：
    sheet first → 打印书写拍照 → base 照片 → export → 云端生成 →
    apply results.zip → rebuild → sheet punct → 书写拍照 →
    hand 照片 --chars samples/punct_chars.json → fix-punct → verify
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT / "output" / "myhand_full.ttf"
MANIFEST = ROOT / "output" / "m4" / "handwritten_chars.json"
PUNCT_LIST = "，。、；：？！“”‘’（）《》〈〉【】—…～·+-×÷=≈≠≤≥±%‰℃°"


def _utf8_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _run(*args: str) -> None:
    """运行项目脚本（统一 UTF-8 环境）。"""
    cmd = [sys.executable, *args]
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(f"\n[失败] {' '.join(args)} 退出码 {r.returncode}，详见对应日志")


def _write_list(path: Path, chars: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chars, ensure_ascii=False, indent=0),
                    encoding="utf-8")


# ---------- 子命令实现 ----------

def cmd_status() -> None:
    if not FONT.exists():
        print("字库不存在（output/myhand_full.ttf）。新用户从 sheet first 开始，或运行 guide。")
        return
    from fontTools.ttLib import TTFont
    tt = TTFont(str(FONT))
    cps = set()
    for t in tt["cmap"].tables:
        if t.isUnicode():
            cps.update(t.cmap.keys())
    hand = 0
    if MANIFEST.exists():
        hand = len(json.loads(MANIFEST.read_text(encoding="utf-8"))["chars"])
    size_mb = FONT.stat().st_size / 1e6
    print(f"字库: {FONT}")
    print(f"  总字数: {len(cps)}  真迹: {hand}  生成: {len(cps) - hand}")
    print(f"  文件: {size_mb:.1f} MB  字形: {tt['maxp'].numGlyphs}")
    backups = sorted((ROOT / "output" / "m4").glob("myhand_full_备份*.ttf"))
    if backups:
        print(f"  可回滚备份: {len(backups)} 份（最近: {backups[-1].name}）")


def cmd_sheet(kind: str, args: list[str]) -> None:
    if kind == "first":
        n = "300"
        if "--n" in args:
            n = args[args.index("--n") + 1]
        _run("tools/make_collect_sheet.py", n, "free")
    elif kind == "expansion":
        # top1000 − 已采集首批 → expansion_list.json
        sys.path.insert(0, str(ROOT))
        from src.fontforge_lib.charlist import load_freq_chars
        collected = set(json.loads(
            (ROOT / "samples" / "collect_chars.json").read_text(
                encoding="utf-8"))["chars"])
        exp = [c for c in load_freq_chars(1000) if c not in collected]
        lst = ROOT / "output" / "m4" / "expansion_list.json"
        _write_list(lst, exp)
        extra = args[args.index("--extra") + 1] if "--extra" in args else ""
        _run("tools/make_collect_sheet.py", f"@{lst}", "free", extra)
    elif kind == "punct":
        lst = ROOT / "output" / "m4" / "punct_list.json"
        _write_list(lst, list("0123456789"
                              "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                              "abcdefghijklmnopqrstuvwxyz" + PUNCT_LIST))
        _run("tools/make_collect_sheet.py", f"@{lst}", "free")
    elif kind == "list":
        src = args[0] if args else ""
        if not src.startswith("@"):
            sys.exit("用法: sheet list @清单.json [--extra 字串]")
        extra = args[args.index("--extra") + 1] if "--extra" in args else ""
        _run("tools/make_collect_sheet.py", src, "free", extra)
    else:
        sys.exit(f"未知采集类型: {kind}（first/expansion/punct/list）")
    # 为现有采集表统一补对照清单（打印后对着写，小字看不清的问题）
    for prefix in ("collect", "expansion", "punct"):
        cj = ROOT / "samples" / f"{prefix}_chars.json"
        if cj.exists():
            _run("tools/make_ref_list.py", str(cj))


def cmd_base(photos: list[str]) -> None:
    if not photos:
        sys.exit("用法: base <照片1> <照片2> ...（首批采集表照片，按页序）")
    _run("-m", "src.fontforge_lib.pipeline", *photos)
    print("\n基础字库完成: output/myhand.ttf（300字真迹）")
    print("下一步: python tools/font_cli.py export 打包云端上传包")


def cmd_hand(args: list[str]) -> None:
    photos = [a for a in args if not a.startswith("--")]
    chars = "samples/recollect_chars.json"
    if "--chars" in args:
        chars = args[args.index("--chars") + 1]
    if not photos:
        sys.exit("用法: hand <照片...> [--chars samples/xxx_chars.json]")
    _run("tools/merge_recollect.py", *photos, "--chars", chars)


def cmd_export() -> None:
    _run("tools/export_m4.py")
    print("\n上传 output/m4/upload.zip 到 AutoDL → 生成后下载 results.zip →")
    print("      python tools/font_cli.py apply results.zip")


def cmd_regen() -> None:
    _run("tools/export_regen.py")
    print("\n上传 output/m4/upload_regen.zip → 生成后下载 results_regen.zip →")
    print("      python tools/font_cli.py apply results_regen.zip")


def cmd_apply(zip_path: str) -> None:
    if not Path(zip_path).exists():
        sys.exit(f"文件不存在: {zip_path}")
    _run("tools/merge_regen.py", zip_path)


def cmd_rebuild() -> None:
    _run("tools/merge_m4.py")


def cmd_fix_punct() -> None:
    _run("tools/fix_punct_layout.py")


def cmd_verify() -> None:
    _run("tools/closed_loop_test.py")


def cmd_guide() -> None:
    print("""=== 个人字库建库向导 ===

[阶段1 首批采集]（一次性，约30分钟书写）
  1. python tools/font_cli.py sheet first          # 打印 collect_sheet.pdf
  2. 黑签字笔书写（对照右上角灰字，写大居中不出格），拍照为 samples/page1.jpg...
  3. python tools/font_cli.py base samples/page1.jpg samples/page2.jpg

[阶段2 云端生成]（AutoDL 租卡，约1小时，数元）
  4. python tools/font_cli.py export               # 产出 upload.zip
  5. 上传云端跑 batch_generate.py（云端命令见 output/m4/upload 内说明）
  6. 下载 results.zip 放到项目根目录
  7. python tools/font_cli.py apply results.zip    # 应用生成字
  8. python tools/font_cli.py rebuild              # 全量重建字库

[阶段3 扩容到1000真迹]（推荐，高频字100%真迹）
  9. python tools/font_cli.py sheet expansion      # 打印扩展表（~700字）
 10. 书写拍照为 samples/1.jpg 2.jpg ...
 11. python tools/font_cli.py hand samples/1.jpg samples/2.jpg ... \
        --chars samples/expansion_chars.json
 12. 重复 4-8（export 时参考图自动升级为1000张真迹）

[阶段4 标点数字]（一次性，10分钟）
 13. python tools/font_cli.py sheet punct          # 打印标点表（99字符）
 14. 书写拍照为 samples/punct_p1.jpg
 15. python tools/font_cli.py hand samples/punct_p1.jpg \
        --chars samples/punct_chars.json
 16. python tools/font_cli.py fix-punct            # 标点排版规范修复

[阶段5 生僻字扩容]（可选，文言文场景）
 17. python tools/font_cli.py regen                # 丢弃字+字频表生僻字
 18. 云端生成 → 下载 results_regen.zip
 19. python tools/font_cli.py apply results_regen.zip

[验收]
 20. python tools/font_cli.py verify               # 闭环渲染 demo 作业
 21. python tools/font_cli.py status               # 字库概览

日常维护：发现某字不像 → sheet list @清单.json 生成补录表 → 书写拍照
          → hand 照片 --chars samples/清单_chars.json 一键替换""")


def main() -> None:
    _utf8_console()
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = args[0], args[1:]
    if cmd == "status":
        cmd_status()
    elif cmd == "sheet" and rest:
        cmd_sheet(rest[0], rest[1:])
    elif cmd == "base" and rest:
        cmd_base(rest)
    elif cmd == "hand" and rest:
        cmd_hand(rest)
    elif cmd == "export":
        cmd_export()
    elif cmd == "regen":
        cmd_regen()
    elif cmd == "apply" and rest:
        cmd_apply(rest[0])
    elif cmd == "rebuild":
        cmd_rebuild()
    elif cmd == "fix-punct":
        cmd_fix_punct()
    elif cmd == "verify":
        cmd_verify()
    elif cmd == "guide":
        cmd_guide()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
