"""闭环验收：用个人手写字体渲染 M2 识别的作业答案（M3 字体 → M1 渲染 → M2 坐标）。"""

import json
import sys
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.render.render_answered as ra
from src.render.perturb import PerturbParams
from src.render.renderer import BlankSpec, HandwriteRenderer

# 个人字体：优先完整字库（M4），否则 300 字版（M3）
MYFONT = (ROOT / "output" / "myhand_full.ttf"
          if (ROOT / "output" / "myhand_full.ttf").exists()
          else ROOT / "output" / "myhand.ttf")
ra.FONT = MYFONT

data = json.loads((ROOT / "output" / "m2_review.json").read_text(encoding="utf-8"))
blanks = [b for b in data["blanks"] if b["answer"].strip()]  # 跳过个人信息空位

have = set()
from fontTools.ttLib import TTFont

tt = TTFont(str(MYFONT))
for t in tt["cmap"].tables:
    have.update(t.cmap.keys())

all_chars = {ch for b in blanks for ch in b["answer"]}
missing = {ch for ch in all_chars if ord(ch) not in have}
covered = len(all_chars) - len(missing)
print(f"答案所需字数（去重）: {len(all_chars)}，个人字库覆盖 {covered} 个"
      + (f"，缺 {len(missing)} 字（暂用文楷回退）: {''.join(sorted(missing))}" if missing else "，全覆盖"))

specs = [BlankSpec(**{k: b.get(k, 0) for k in
                      ("id", "page", "x", "y", "width", "answer", "height")})
         for b in blanks]

src = fitz.open(ROOT / "samples" / "demo_homework.pdf")
out = ROOT / "output" / "我的字_完成版.pdf"
renderer = HandwriteRenderer(
    MYFONT, PerturbParams(), base_size=16,  # 用户反馈：整体字号提升（原14）
    fallback_font=ROOT / "fonts" / "LXGWWenKai-Regular.ttf",
)
for b in specs:
    renderer.render(src, b)
if renderer.fallback_used:
    print(f"回退文楷的字: {''.join(sorted(renderer.fallback_used))}")
src.subset_fonts()
src.save(out, garbage=4, deflate=True)
doc = fitz.open(out)
doc[0].get_pixmap(dpi=150).save(ROOT / "output" / "我的字_完成版_预览.png")
print(f"已生成 {out}（渲染 {len(specs)} 个答案）")
