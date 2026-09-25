"""M4-6 路线A阶段3：打包云端重新生成所需的一切（千字参考版）。

前置：700字扩展采集已通过 merge_recollect.py 合并进 myhand_full.ttf，
且 output/m4/handwritten_chars.json 记录了真迹字清单（merge_recollect 维护）。

产出 output/m4/upload/（含 upload.zip）：
- style_refs/*.png   ~1000 张真迹参考图（从 myhand_full.ttf 渲染，黑字白底，
                      与 FontDiffuser 的 ttf2im 内容图渲染约定一致）
- target_chars.txt   待生成字表（top3500 去除真迹 + 演示作业缺字 +
                      常用标点/数字/字母）
- LXGWWenKai-Regular.ttf  内容图渲染用的全字库源字体
- batch_generate.py  云端批量生成脚本（模型只加载一次）

云端流程：AutoDL 租卡 → 上传 zip → clone FontDiffuser → 下载权重 → 跑批量脚本。
"""

import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fontforge_lib.charlist import load_freq_chars

MYFONT = ROOT / "output" / "myhand_full.ttf"     # 含 ~1000 真迹字形的完整字库
MANIFEST = ROOT / "output" / "m4" / "handwritten_chars.json"
FALLBACK_MANIFEST = ROOT / "samples" / "collect_chars.json"  # 无清单时退回首批300
LXGW = ROOT / "fonts" / "LXGWWenKai-Regular.ttf"
BATCH_SCRIPT = ROOT / "tools" / "m4_batch_generate.py"
OUT = ROOT / "output" / "m4" / "upload"

REF_SIZE = 128          # 参考图边长（模型端会缩到 96）
FONT_RENDER_SIZE = 100  # PIL 渲染字号

# 常用全角标点（答案里高频出现，手写感重要）
PUNCT = "，。、；：？！“”‘’（）《》〈〉【】—…～·"
# 数字与字母（质量待 QC，不好可弃用）
ASCII = "0123456789" + "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_authentic() -> list[str]:
    """真迹字清单：优先 manifest（merge_recollect 维护），退回首批采集表。"""
    src = MANIFEST if MANIFEST.exists() else FALLBACK_MANIFEST
    data = json.loads(src.read_text(encoding="utf-8"))
    chars = data["chars"] if isinstance(data, dict) else data
    chars = [c for c in chars if isinstance(c, str) and len(c) == 1]
    print(f"真迹清单: {len(chars)} 字（来源: {src.name}）")
    return chars


def render_style_refs(chars: list[str], out_dir: Path) -> tuple[int, list[str]]:
    """从 myhand_full.ttf 渲染参考图：黑字白底，em 盒居中。返回 (成功数, 空字形字)。"""
    if out_dir.exists():
        shutil.rmtree(out_dir)   # 清掉上一版参考图，避免残留旧字形
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(MYFONT), FONT_RENDER_SIZE)
    n, empty = 0, []
    for ch in chars:
        img = Image.new("L", (REF_SIZE, REF_SIZE), 255)
        draw = ImageDraw.Draw(img)
        draw.text((REF_SIZE / 2, REF_SIZE / 2), ch, font=font, fill=0, anchor="mm")
        if np.array(img).min() >= 255:   # 渲染为空 → 字库缺该真迹字形
            empty.append(ch)
            continue
        img.convert("RGB").save(out_dir / f"{ch}.png")
        n += 1
    return n, empty


def build_target_list(authentic: list[str]) -> tuple[list[str], list[str]]:
    """top3500 去除真迹，加上演示作业缺字/标点/ASCII。"""
    top3500 = load_freq_chars(3500)
    have = set(authentic)
    targets: list[str] = [c for c in top3500 if c not in have]

    # 演示作业答案中的缺字（m2_review.json 若存在）
    demo_extra: set[str] = set()
    review = ROOT / "output" / "m2_review.json"
    if review.exists():
        data = json.loads(review.read_text(encoding="utf-8"))
        for b in data.get("blanks", []):
            for ch in b.get("answer", ""):
                if ch.strip():
                    demo_extra.add(ch)
    demo_extra -= have
    demo_extra -= set(targets)

    extras = [c for c in demo_extra if c not in PUNCT and c not in ASCII]
    for ch in PUNCT + ASCII:
        if ch not in have:
            extras.append(ch)
    extras = [c for c in extras if c not in targets]

    return targets + extras, sorted(demo_extra)


def main():
    if not MYFONT.exists():
        sys.exit("未找到 output/myhand_full.ttf，请先完成 M4 合并 + 扩展采集合并")

    authentic = load_authentic()

    # 1) 风格参考图（仅真迹字形）
    n_refs, empty = render_style_refs(authentic, OUT / "style_refs")
    print(f"风格参考图: {n_refs} 张 → {OUT / 'style_refs'}")
    if empty:
        print(f"  [!] {len(empty)} 字在字库中无真迹字形（将被列入重新生成目标）: "
              f"{''.join(empty[:40])}")

    # 2) 目标字表（真迹渲染失败的字留在目标里，用新生成覆盖旧字形）
    effective = [c for c in authentic if c not in empty]
    targets, demo_missing = build_target_list(effective)
    (OUT / "target_chars.txt").write_text("\n".join(targets), encoding="utf-8")
    print(f"目标字表: {len(targets)} 字（top3500 缺口 + 演示缺字 {len(demo_missing)} "
          f"+ 标点/ASCII）")

    # 3) 源字体 + 云端脚本
    shutil.copy(LXGW, OUT / "LXGWWenKai-Regular.ttf")
    shutil.copy(BATCH_SCRIPT, OUT / "batch_generate.py")

    # 4) 打 zip（带顶层 upload/ 目录，云端解压到 /root/autodl-tmp/upload/）
    zip_path = OUT.parent / "upload.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                zf.write(p, Path("upload") / p.relative_to(OUT))
    print(f"上传包: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print("\n下一步: AutoDL 租卡 → JupyterLab 上传 upload.zip 到 /root/autodl-tmp")


if __name__ == "__main__":
    main()
