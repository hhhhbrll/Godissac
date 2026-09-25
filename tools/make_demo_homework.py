"""生成演示用作业 PDF + 空位坐标 JSON。

空位坐标 JSON 是整个系统的"接口契约"：
模块二（VLM 识别）未来负责产出它，模块三（渲染器）负责消费它。
演示阶段坐标由本脚本排版时直接记录，格式与未来 VLM 输出完全一致。
"""

import json
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
FONT = ROOT / "fonts" / "LXGWWenKai-Regular.ttf"

A4_W, A4_H = 595, 842
MARGIN = 72
Q_TOP = 150        # 第一题基线 y
Q_GAP = 42         # 题目行距
BLANK_W = 150.0

# (前缀, 答案, 后缀)
QUESTIONS = [
    ("1. 《红楼梦》的作者是", "曹雪芹", "。"),
    ("2. “落霞与孤鹜齐飞”的下一句是", "秋水共长天一色", "。"),
    ("3. 唐宋八大家中，“文起八代之衰”说的是", "韩愈", "。"),
    ("4. 《背影》的作者是现代散文家", "朱自清(1898—1948)", "。"),
    ("5. “学而不思则罔”的下半句是", "思而不学则殆", "。"),
    ("6. 我国第一部诗歌总集是", "《诗经》", "。"),
    ("7. 屈原《离骚》中表达求索精神的名句是", "路漫漫其修远兮，吾将上下而求索", "。"),
]

TITLE = "语文常识填空练习"


def main():
    SAMPLES.mkdir(exist_ok=True)
    font = fitz.Font(fontfile=str(FONT))
    doc = fitz.open()
    page = doc.new_page(width=A4_W, height=A4_H)

    def put(x, y, text, size):
        page.insert_text((x, y), text, fontsize=size,
                         fontname="HW", fontfile=str(FONT), color=(0, 0, 0))

    def text_w(text, size):
        return font.text_length(text, fontsize=size)

    def blank(x, y, width, answer, blanks, bid):
        """画下划线并登记空位。y 为该行文字基线。"""
        y_line = y + 4
        page.draw_line(fitz.Point(x, y_line), fitz.Point(x + width, y_line),
                       color=(0, 0, 0), width=0.7)
        blanks.append({"id": bid, "page": 0, "x": round(x, 1), "y": round(y_line, 1),
                       "width": width, "answer": answer})

    blanks = []
    bid = 0

    # 标题（居中）
    put((A4_W - text_w(TITLE, 18)) / 2, MARGIN, TITLE, 18)

    # 页眉：姓名 / 班级 / 日期
    hy = MARGIN + 44
    put(MARGIN, hy, "姓名：", 11)
    x = MARGIN + text_w("姓名：", 11) + 4
    blank(x, hy, 90, "张三", blanks, bid := bid + 1)
    x += 90 + 18
    put(x, hy, "班级：", 11)
    x += text_w("班级：", 11) + 4
    blank(x, hy, 110, "高二(3)班", blanks, bid := bid + 1)
    x += 110 + 18
    put(x, hy, "日期：", 11)
    x += text_w("日期：", 11) + 4
    blank(x, hy, 90, "9月12日", blanks, bid := bid + 1)

    # 题目行
    for i, (prefix, answer, suffix) in enumerate(QUESTIONS):
        y = Q_TOP + i * Q_GAP
        put(MARGIN, y, prefix, 12)
        bx = MARGIN + text_w(prefix, 12) + 6
        blank(bx, y, BLANK_W, answer, blanks, bid := bid + 1)
        put(bx + BLANK_W + 6, y, suffix, 12)

    doc.save(SAMPLES / "demo_homework.pdf", garbage=4, deflate=True)
    (SAMPLES / "demo_homework.blanks.json").write_text(
        json.dumps({"source": "demo", "pages": 1, "blanks": blanks},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已生成: samples/demo_homework.pdf（空位数: {len(blanks)}）")


if __name__ == "__main__":
    main()
