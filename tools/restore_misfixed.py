"""恢复被误伤的字形 + 验证（v2.1 底带专用版）。

背景：v2 顶带检测误删 34 字的合法部首（历丢厂头、安丢宝盖、府丢广头…）。
污迹真实来源只有照片采集底栏（字底），顶带检测有害无益，已从
fix_glyph_noise.py 移除。

本脚本：
1. 对 v2 手术清单里的字，逐一重放 find_noise（仅底带版）于**备份字形**
2. 旧手术删过顶部轮廓的字 → 从备份整体恢复该字形
3. 打印恢复清单供人工确认

用法：
    python tools/restore_misfixed.py --dry-run
    python tools/restore_misfixed.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import fix_glyph_noise as FN  # noqa: E402

FONT = ROOT / "output" / "myhand_full.ttf"
BAK = ROOT / "output" / "m4" / "myhand_full_备份_底噪手术前.ttf"

# v2 手术清单（63字）
V2_FIXED = list('临之乖亏互亓充减匝匡匦危厂历厌古叵号呕咏唁唾妒娓嫉字守安容'
               '尽履岖嵊庑府悉户攻方更永沤洎玉画痈祕禿私秃良苎訐訟讶费辰'
               '透逦靠额高麻')


def clone_glyph(src_font, dst_font, ch):
    """把 src 字体中 ch 的字形（含 hmtx）复制到 dst 字体。"""
    sc, dc = src_font.getBestCmap(), dst_font.getBestCmap()
    sg = src_font['glyf'][sc[ord(ch)]]
    dg = dst_font['glyf'][dc[ord(ch)]]
    # 字形对象整体替换（fontTools glyf 表支持直接赋值）
    dst_font['glyf'][dc[ord(ch)]] = sg
    adv = dst_font['hmtx'].metrics.get(dc[ord(ch)], (1000, 0))[0]
    dst_font['hmtx'].metrics[dc[ord(ch)]] = (adv, int(sg.xMin))
    return dg is not sg


def main(dry_run: bool = False) -> None:
    cur = TTFont(str(FONT), lazy=False)
    bak = TTFont(str(BAK), lazy=False)

    restore, keep, untouched = [], [], []
    for ch in V2_FIXED:
        try:
            g_bak = bak['glyf'][bak.getBestCmap()[ord(ch)]]
        except KeyError:
            untouched.append(ch)
            continue
        # 用 v2.1（仅底带）重放判定于备份字形
        noise = FN.find_noise(g_bak)
        if noise:
            # v2.1 仍会动刀 = 底部真污迹 → 当前手术结果正确，保留
            keep.append(ch)
        else:
            # v2.1 不动 = 旧手术删的是顶部部首（误伤）→ 恢复备份
            restore.append(ch)

    print(f"误伤恢复: {len(restore)} 字")
    print("  " + "".join(restore))
    print(f"底部正确手术（保留现状）: {len(keep)} 字")
    print("  " + "".join(keep))

    if dry_run:
        print("[dry-run] 未修改字体")
        return

    if restore:
        BACKUP2 = ROOT / "output" / "m4" / "myhand_full_备份_恢复误伤前.ttf"
        if not BACKUP2.exists():
            shutil.copy2(FONT, BACKUP2)
            print(f"备份: {BACKUP2.name}")
        for ch in restore:
            clone_glyph(bak, cur, ch)
        cur.save(str(FONT))
        print(f"已保存: {FONT}")

        # 恢复的字中若底部仍有真污迹，v2.1 会再手术；此处保持原样，
        # 污迹字覆盖率本就不差（0.4-0.55），不追求极致


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
