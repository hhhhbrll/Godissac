"""轮廓命令 → TTF 字体文件（fontTools FontBuilder）。

TrueType 只支持二次曲线，potrace 输出三次贝塞尔，经 Cu2QuPen 自适应转换。
填充方向（外圈/内孔绕向）由管线 IoU 质检自动择优（reverse_direction）。
"""

from __future__ import annotations

from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen

from .template import ASCENT, DESCENT, EM


def _draw_commands(pen, cmds: list[tuple]):
    for cmd in cmds:
        if cmd[0] == "M":
            pen.moveTo((cmd[1], cmd[2]))
        elif cmd[0] == "L":
            pen.lineTo((cmd[1], cmd[2]))
        elif cmd[0] == "C":
            pen.curveTo((cmd[1], cmd[2]), (cmd[3], cmd[4]), (cmd[5], cmd[6]))
        elif cmd[0] == "Z":
            pen.closePath()


def build_ttf(glyph_contours: dict[str, list[list[tuple]]], out_path: str | Path,
              reverse: bool = False, family: str = "MyHandwriting"):
    """glyph_contours: {汉字: [轮廓命令列表]} → 保存 TTF。"""
    chars = sorted(glyph_contours)
    glyph_names = {c: f"uni{ord(c):04X}" for c in chars}
    order = [".notdef"] + [glyph_names[c] for c in chars]

    fb = FontBuilder(EM, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({ord(c): glyph_names[c] for c in chars})

    glyf: dict = {".notdef": TTGlyphPen(None).glyph()}
    for c in chars:
        tt = TTGlyphPen(None)
        cu = Cu2QuPen(tt, max_err=1.5, reverse_direction=reverse)
        for contour in glyph_contours[c]:
            _draw_commands(cu, contour)
        glyf[glyph_names[c]] = tt.glyph()

    # TrueType 规范：hmtx.lsb 应等于字形 xMin，否则渲染器会平移字形
    metrics = {".notdef": (EM, 0)}
    for c in chars:
        xs = [pt[1] for contour in glyph_contours[c] for pt in
              [(cmd[0], cmd[1]) for cmd in contour if cmd[0] in ("M", "L")]]
        xs += [cmd[5] for contour in glyph_contours[c] for cmd in contour if cmd[0] == "C"]
        lsb = int(round(min(xs))) if xs else 0
        metrics[glyph_names[c]] = (EM, lsb)

    fb.setupGlyf(glyf)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
    fb.setupOS2(sTypoAscender=ASCENT, sTypoDescender=DESCENT,
                usWinAscent=ASCENT, usWinDescent=-DESCENT)
    fb.setupNameTable({
        "familyName": family, "styleName": "Regular",
        "uniqueFontIdentifier": f"{family}.Regular",
        "fullName": f"{family}-Regular", "psName": f"{family}-Regular",
    })
    fb.setupPost()
    fb.save(str(out_path))
    return out_path
