"""Learning example migrated from 2_1.py. Run from the project root with python -m."""

from project_paths import DATA_DIR

import re


if __name__ == "__main__":
    with open(DATA_DIR / "the-verdict.txt","r",encoding="utf-8") as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(len(preprocessed))

    allow_words = sorted(set(preprocessed))
    vocab_size = len(allow_words)
    print(vocab_size)

    # 创建词汇表
    vacab = {token: integer for integer, token in enumerate(allow_words)}
    for i, item in enumerate(vacab.items()):
        print(item)
        if i >= 50:
            break
