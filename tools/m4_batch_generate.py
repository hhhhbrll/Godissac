"""FontDiffuser 批量生成脚本（云 GPU 端运行，位于 upload/ 目录）。

用法（在 FontDiffuser 仓库同级目录结构下）:
    python batch_generate.py \
        --ckpt_dir ckpt/ \
        --style_refs_dir style_refs/ \
        --target_chars_file target_chars.txt \
        --ttf_path LXGWWenKai-Regular.ttf \
        --save_dir gen/ \
        --device cuda:0

特性: 模型只加载一次; 断点续跑(跳过已生成); 每字独立种子(可复现);
      覆盖度预检; 结束输出 report。

风格条件策略（--style_mode）:
  mean（默认）: 全部参考图的风格特征图取平均（一次性预计算），作为所有字
      的统一风格条件；内容编码器侧的 style 通路用"最接近平均特征"的典型
      参考字。消除单参考随机抽取导致的字间风格抖动（粗细/笔锋不一致），
      且风格编码每步不再重算，生成更快。
  random: 每字随机抽 1 张参考图（旧行为，可用于 A/B 对比）。
"""

import json
import os
import random
import sys
import time

# 无头服务器: pygame 渲染不需要显示器
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(HERE), "FontDiffuser")
sys.path.insert(0, REPO)

import torch
from PIL import Image
from accelerate.utils import set_seed
import torchvision.transforms as transforms

from src import (
    FontDiffuserDPMPipeline,
    FontDiffuserModelDPM,
    build_ddpm_scheduler,
    build_unet,
    build_content_encoder,
    build_style_encoder,
)
from utils import load_ttf, ttf2im
from configs.fontdiffuser import get_parser


def load_ckpt(model, path: str) -> None:
    """容错加载权重：键完全匹配才静默通过；部分匹配则告警；
    大面积不匹配（镜像权重与官方结构不符）则报错退出并输出证据。"""
    sd = torch.load(path, map_location="cpu")
    result = model.load_state_dict(sd, strict=False)
    n_missing, n_unexpected = len(result.missing_keys), len(result.unexpected_keys)
    if n_missing == 0 and n_unexpected == 0:
        print(f"[OK] {os.path.basename(path)} 键完全匹配")
        return
    print(f"[告警] {os.path.basename(path)}: 缺 {n_missing} 键 / 多 {n_unexpected} 键")
    if n_missing > len(sd) // 2:
        print("  缺失键示例:", result.missing_keys[:5])
        sys.exit("权重与模型结构不匹配（疑似错误镜像），请把以上输出发回")


class MeanStyleEncoder(torch.nn.Module):
    """多参考风格特征平均器。

    预计算 N 张参考图经原 style_encoder 的风格特征图，取平均作为统一风格
    条件。forward 时无条件分支（CFG 的全白图）仍走原编码器保持 CFG 语义；
    有条件分支直接返回平均特征（批维对齐）。DPM forward 中 style 残差特征
    未被使用，返回空表即可。
    """

    def __init__(self, encoder, ref_tensors, chunk=16):
        super().__init__()
        self.encoder = encoder
        feats = []
        with torch.no_grad():
            for i in range(0, len(ref_tensors), chunk):
                f, _, _ = encoder(ref_tensors[i:i + chunk])
                feats.append(f)
        allf = torch.cat(feats)                       # (N, C, H, W)
        self.register_buffer("mean_feat", allf.mean(0, keepdim=True))
        # 典型参考图：特征离平均最近的那张（供内容编码器 style 通路用）
        dist = (allf - self.mean_feat).flatten(1).norm(dim=1)
        self.typical_idx = int(dist.argmin())

    def forward(self, x):
        if bool((x > 0.999).all()):                   # CFG 无条件分支：全白图
            return self.encoder(x)
        b = x.shape[0]
        return self.mean_feat.expand(b, -1, -1, -1), None, []


def main():
    parser = get_parser()
    # 官方 get_parser() 不含 --ckpt_dir/--device（2024-03 版起），显式补齐
    # （conflict-safe：若仓库版本已提供则跳过）
    for _n, _t, _d in [("--ckpt_dir", str, "../FontDiffuser/ckpt"),
                       ("--device", str, "cuda:0")]:
        try:
            parser.add_argument(_n, type=_t, default=_d)
        except Exception:
            pass
    parser.add_argument("--style_refs_dir", type=str, required=True)
    parser.add_argument("--target_chars_file", type=str, required=True)
    parser.add_argument("--ttf_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--style_mode", choices=["mean", "random"], default="mean")
    args = parser.parse_args()
    args.style_image_size = (args.style_image_size, args.style_image_size)
    args.content_image_size = (args.content_image_size, args.content_image_size)

    # ---- 字表与断点续跑 ----
    with open(args.target_chars_file, encoding="utf-8") as f:
        chars = [ln.strip() for ln in f if ln.strip()]
    seen: set = set()
    chars = [c for c in chars if not (c in seen or seen.add(c))]

    os.makedirs(args.save_dir, exist_ok=True)
    todo = [c for c in chars
            if not os.path.exists(os.path.join(args.save_dir, f"{c}.png"))]
    print(f"目标 {len(chars)} 字 | 已生成 {len(chars) - len(todo)} | 待生成 {len(todo)}")

    # ---- 源字体覆盖度预检（一次性） ----
    from fontTools.ttLib import TTFont
    tf = TTFont(args.ttf_path)
    covered = set()
    for t in tf["cmap"].tables:
        covered.update(t.cmap.keys())
    not_covered = [c for c in todo if ord(c) not in covered]
    todo = [c for c in todo if ord(c) in covered]
    if not_covered:
        print(f"源字体缺 {len(not_covered)} 字(跳过): {''.join(not_covered[:50])}")

    # ---- 风格参考图 ----
    refs = sorted(
        os.path.join(args.style_refs_dir, f)
        for f in os.listdir(args.style_refs_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    print(f"风格参考图 {len(refs)} 张")
    if not refs:
        sys.exit("style_refs 目录为空")

    font = load_ttf(ttf_path=args.ttf_path)

    # ---- 模型（只加载一次） ----
    unet = build_unet(args=args)
    load_ckpt(unet, f"{args.ckpt_dir}/unet.pth")
    style_encoder = build_style_encoder(args=args)
    load_ckpt(style_encoder, f"{args.ckpt_dir}/style_encoder.pth")
    content_encoder = build_content_encoder(args=args)
    load_ckpt(content_encoder, f"{args.ckpt_dir}/content_encoder.pth")
    model = FontDiffuserModelDPM(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder)
    model.to(args.device)
    train_scheduler = build_ddpm_scheduler(args=args)
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("模型加载完成")

    content_tf = transforms.Compose([
        transforms.Resize(args.content_image_size,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    style_tf = transforms.Compose([
        transforms.Resize(args.style_image_size,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # ---- 风格策略：mean = 全参考特征平均（统一风格条件） ----
    typical_pil = Image.open(refs[0]).convert("RGB")
    if args.style_mode == "mean":
        with torch.no_grad():
            ref_tensors = torch.cat([
                style_tf(Image.open(p).convert("RGB"))[None, :]
                for p in refs
            ]).to(args.device)
        mean_enc = MeanStyleEncoder(model.style_encoder, ref_tensors)
        mean_enc.to(args.device)
        model.style_encoder = mean_enc
        typical_pil = Image.open(refs[mean_enc.typical_idx]).convert("RGB")
        print(f"风格条件: {len(refs)} 张参考图特征平均 | "
              f"典型参考字: {os.path.basename(refs[mean_enc.typical_idx])}")
    else:
        print("风格条件: 每字随机单参考（旧策略）")
    print("开始生成")

    t0 = time.time()
    done = 0
    failed: list[str] = []
    with torch.no_grad():
        for i, ch in enumerate(todo):
            set_seed(args.base_seed + i)
            try:
                content_pil = ttf2im(font=font, char=ch)
                if content_pil is None:
                    failed.append(ch)
                    continue
                if args.style_mode == "mean":
                    style_pil = typical_pil
                else:
                    style_pil = Image.open(random.choice(refs)).convert("RGB")
                ci = content_tf(content_pil)[None, :].to(args.device)
                si = style_tf(style_pil)[None, :].to(args.device)
                images = pipe.generate(
                    content_images=ci,
                    style_images=si,
                    batch_size=1,
                    order=args.order,
                    num_inference_step=args.num_inference_steps,
                    content_encoder_downsample_size=(
                        args.content_encoder_downsample_size),
                    t_start=args.t_start,
                    t_end=args.t_end,
                    dm_size=args.content_image_size,
                    algorithm_type=args.algorithm_type,
                    skip_type=args.skip_type,
                    method=args.method,
                    correcting_x0_fn=args.correcting_x0_fn,
                )
                images[0].save(os.path.join(args.save_dir, f"{ch}.png"))
            except Exception as e:  # noqa: BLE001 - 单字失败不中断整批
                print(f"  [失败] {ch}: {e}")
                failed.append(ch)
                continue
            done += 1
            if done % 50 == 0 or done == len(todo):
                rate = done / (time.time() - t0)
                remain = (len(todo) - done) / max(rate, 1e-9) / 60
                print(f"[{done}/{len(todo)}] {rate:.2f} 字/秒 | "
                      f"剩余约 {remain:.1f} 分钟", flush=True)

    report = {
        "total_target": len(chars),
        "generated_this_run": done,
        "generated_total": len([f for f in os.listdir(args.save_dir)
                                if f.endswith(".png")]),
        "not_in_source_font": not_covered,
        "failed": failed,
        "style_mode": args.style_mode,
        "n_style_refs": len(refs),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    with open(os.path.join(args.save_dir, "report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
