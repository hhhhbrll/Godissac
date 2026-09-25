"""常用汉字表：从 MTSU 现代汉语字频表加载（samples/charfreq_modern.txt）。

数据源: lingua.mtsu.edu/chinese-computing/statistics（现代汉语语料，Tab 分隔，
前几行为 /* */ 注释）。列: 序号 汉字 拼音 词频累计百分比 等。

M3 首批采集前 300 高频字（3 页模板），第二批扩至 500。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FREQ_FILE = ROOT / "samples" / "charfreq_modern.txt"


def load_freq_chars(limit: int = 500) -> list[str]:
    """按频率降序返回前 limit 个汉字（跳过非 BMP 汉字与重复）。"""
    chars: list[str] = []
    seen: set[str] = set()
    for line in FREQ_FILE.read_text(encoding="gbk", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("/*"):
            continue
        cols = line.split("\t")
        if len(cols) < 2:
            continue
        ch = cols[1].strip()
        # 仅取中日韩统一表意文字基本区
        if len(ch) == 1 and "\u4e00" <= ch <= "\u9fff" and ch not in seen:
            seen.add(ch)
            chars.append(ch)
            if len(chars) >= limit:
                break
    return chars


def batch(n: int) -> list[str]:
    """前 n 个高频字。"""
    return load_freq_chars(n)


if __name__ == "__main__":
    top = load_freq_chars(500)
    print(f"加载 {len(top)} 字")
    print("前 30:", "".join(top[:30]))
