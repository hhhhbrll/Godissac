"""逐字扰动引擎：让字体渲染摆脱"印刷感"，模拟真人书写的随机变化。

真实手写的不完美主要体现在五个维度：
- 字号：每个字大小略有不同
- 旋转：每个字有微小倾斜，方向随机
- 基线：字不会严格落在同一条直线上
- 字距：疏密不均
- 墨色：用力不同导致的深浅差异

未来扩展位：transform() 已接收字符本身，M3 字库建成后在此处
实现"同字多变体随机选用"。
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class PerturbParams:
    """扰动强度参数。jitter 均为相对比例或绝对 pt 值，设为 0 即关闭该维度。"""

    size_jitter: float = 0.04      # 字号随机 ±4%
    rotation_jitter: float = 1.5   # 旋转随机 ±1.5°
    baseline_jitter: float = 0.5   # 基线纵向随机 ±0.5pt（用户反馈：上下浮动轻微减小）
    advance_jitter: float = 0.08   # 字距随机 ±8%
    ink_min: float = 0.01          # 墨色灰度下限（越小越深）
    ink_max: float = 0.08          # 墨色灰度上限（打印实测：>0.1 灰度打印偏浅不像墨迹）
    seed: int | None = None        # 固定种子便于复现；None = 每次渲染都不同


@dataclass
class CharTransform:
    """单个字符的一次"书写"实例。"""

    size: float      # 实际字号
    rotation: float  # 旋转角（度）
    dy: float        # 基线纵向偏移（pt，向下为正）
    ink: float       # 墨色灰度 0~1（越小越黑）
    advance: float   # 该字符占据的水平宽度（pt，已含字距扰动）


class PerturbEngine:
    def __init__(self, params: PerturbParams | None = None):
        self.params = params or PerturbParams()
        self.rng = random.Random(self.params.seed)

    def transform(self, char: str, base_size: float, base_advance: float) -> CharTransform:
        """为一个字符生成一次随机书写变换。"""
        p = self.params
        return CharTransform(
            size=base_size * (1 + self.rng.uniform(-p.size_jitter, p.size_jitter)),
            rotation=self.rng.uniform(-p.rotation_jitter, p.rotation_jitter),
            dy=self.rng.uniform(-p.baseline_jitter, p.baseline_jitter),
            ink=self.rng.uniform(p.ink_min, p.ink_max),
            advance=base_advance * (1 + self.rng.uniform(-p.advance_jitter, p.advance_jitter)),
        )
