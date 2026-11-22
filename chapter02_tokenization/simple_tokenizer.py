"""Learning example migrated from 2_3.py. Run from the project root with python -m."""

from project_paths import DATA_DIR


import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

if __name__ == "__main__":
    with open(DATA_DIR / "the-verdict.txt","r",encoding="utf-8") as f:
        raw_text = f.read()
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    # all_tokens = sorted(list(set(preprocessed)))
    # all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    # vocab = {token: integer for integer, token in enumerate(all_tokens)}
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    print(len(vocab.items()))
    for i, item in enumerate( list(vocab.items())[-5:] ):
        print(item)

    tokenizer = SimpleTokenizerV1(vocab)

    text = """It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride."""

    ids = tokenizer.encode(text)
    print(ids)

    print(tokenizer.decode(ids))

    text = "Hello, do you like tea?"
    # V1 intentionally has no unknown-token fallback; V2 adds <|unk|>.
    try:
        print(tokenizer.encode(text))
    except KeyError as error:
        print(f"Unknown token (expected in V1): {error}")
