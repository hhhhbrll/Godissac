"""M4-7 打包补齐上传包：丢弃字重生成 + 字频表生僻字扩容 → 云端一次生成。

前置：merge_m4.py 刚跑完（merge_report.json 含最新丢弃清单）。

目标字表 =
  A. 结构校验丢弃的 CJK 字（当前渲染回退文楷，需重新生成补齐）
  B. 字频表（9933字）中字库尚未覆盖的生僻字（文言文/生僻字扩容，
     覆盖后文楷回退仅留给表外极生僻字）

产出 output/m4/upload_regen.zip：
- style_refs/*.png   复用主上传包的 1001 张真迹参考图
- target_chars.txt   A + B 合并（仅 CJK；标点/字母由手写采集表覆盖）
- LXGWWenKai-Regular.ttf + batch_generate.py（含 --ckpt_dir 补丁与 mean 风格模式）

云端流程与主生成相同；跑完 results_regen.zip → 本地 python tools/merge_regen.py。
"""

import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fontforge_lib.charlist import load_freq_chars

REPORT = ROOT / "output" / "m4" / "merge_report.json"
MANIFEST = ROOT / "output" / "m4" / "handwritten_chars.json"
FONT = ROOT / "output" / "myhand_full.ttf"
SRC_UPLOAD = ROOT / "output" / "m4" / "upload"          # 主上传包（复用参考图）
OUT = ROOT / "output" / "m4" / "upload_regen"
BATCH_SCRIPT = ROOT / "tools" / "m4_batch_generate.py"
LXGW = ROOT / "fonts" / "LXGWWenKai-Regular.ttf"


def main():
    if not REPORT.exists():
        sys.exit("缺少 merge_report.json，请先跑 merge_m4.py")
    if not FONT.exists():
        sys.exit(f"缺少字库 {FONT}")
    if not (SRC_UPLOAD / "style_refs").exists():
        sys.exit(f"缺少 {SRC_UPLOAD / 'style_refs'}，请先跑 export_m4.py")

    # A. 丢弃字（当前回退文楷）
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    dropped = [c for c in report.get("gen_dropped_bad", [])
               if "\u4e00" <= c <= "\u9fff"]
    print(f"A. 结构校验丢弃字: {len(dropped)}")

    # B. 字频表生僻字（字库未覆盖）
    from fontTools.ttLib import TTFont
    tt = TTFont(str(FONT))
    have = set()
    for t in tt["cmap"].tables:
        have.update(chr(cp) for cp in t.cmap.keys())
    hand = set(json.loads(MANIFEST.read_text(encoding="utf-8"))["chars"]) \
        if MANIFEST.exists() else set()
    freq_all = load_freq_chars(99999)                    # 全表 9901 字
    rare = [c for c in freq_all if c not in have and c not in hand]
    print(f"B. 字频表未覆盖生僻字: {len(rare)}（字频表共 {len(freq_all)}，"
          f"已覆盖 {len(freq_all) - len(rare)}）")

    # 合并去重（丢弃字可能也在 B 中）
    targets = list(dict.fromkeys(dropped + rare))
    print(f"合计待生成: {len(targets)} 字"
          f"（云端约 {len(targets) / 60 / 60 * 1.1:.0f}~{len(targets) / 60 / 60 * 1.8:.0f} 小时）")
    if not targets:
        sys.exit("无待生成字，无需上传包")

    # 组装目录
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(SRC_UPLOAD / "style_refs", OUT / "style_refs")
    (OUT / "target_chars.txt").write_text("\n".join(targets), encoding="utf-8")
    shutil.copy(LXGW, OUT / "LXGWWenKai-Regular.ttf")
    shutil.copy(BATCH_SCRIPT, OUT / "batch_generate.py")

    # 打 zip（顶层 upload_regen/，云端解压到 /root/autodl-tmp/）
    zip_path = OUT.parent / "upload_regen.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                zf.write(p, Path("upload_regen") / p.relative_to(OUT))
    print(f"上传包: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print("\n云端: unzip upload_regen.zip → cd upload_regen → 同主流程命令"
          "（save_dir ../results_regen）")


if __name__ == "__main__":
    main()
