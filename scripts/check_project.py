"""离线 CPU 冒烟检查：不下载权重，不执行训练周期或调用 Ollama。"""

from contextlib import ExitStack, redirect_stdout
import importlib
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from project_paths import PROJECT_ROOT


def check_imports():
    """导入所有学习模块，遇到下载、保存或绘图副作用立即失败。"""
    files = list(PROJECT_ROOT.glob("*.py"))
    for folder in PROJECT_ROOT.iterdir():
        if folder.is_dir() and folder.name.startswith(("chapter", "appendix", "archive")):
            files.extend(folder.rglob("*.py"))
    output = io.StringIO()
    with ExitStack() as stack:
        for target in (
            "urllib.request.urlopen", "requests.sessions.Session.request",
            "tiktoken.get_encoding", "torch.save", "matplotlib.pyplot.show",
        ):
            stack.enter_context(patch(target, side_effect=AssertionError(f"Import side effect: {target}")))
        stack.enter_context(redirect_stdout(output))
        for path in sorted(files):
            module = ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
            if module.endswith(".__init__"):
                module = module.removesuffix(".__init__")
            importlib.import_module(module)
    assert not output.getvalue(), output.getvalue()
    assert "tensorflow" not in sys.modules, "TensorFlow must remain optional at import time"
    return len(files)


class ToyTokenizer:
    """固定的小词表，检查时无需联网获取 GPT-2 分词资源。"""

    def encode(self, text, **kwargs):
        return [ord(char) % 16 for char in text]


def main():
    torch.set_num_threads(1)
    count = check_imports()
    from chapter02 import GPTDatasetV1
    from chapter04 import GPTModel, generate_text_simple
    from chapter05 import calc_loss_batch, generate
    from chapter04_gpt_model.gpt_model import GPTModel as CanonicalGPT
    from chapter05_pretraining.training import generate as canonical_generate
    from chapter06_classification.classifier import SpamDataset
    from chapter07_instruction_finetuning.finetuning import custom_collate_fn

    assert GPTModel is CanonicalGPT and generate is canonical_generate
    dataset = GPTDatasetV1("0123456789", ToyTokenizer(), max_length=4, stride=2)
    assert len(dataset) == 3
    x, y = dataset[0]
    assert torch.equal(x[1:], y[:-1])

    torch.manual_seed(123)
    cfg = dict(vocab_size=32, context_length=8, emb_dim=16,
               n_heads=2, n_layers=1, drop_rate=0.0, qkv_bias=True)
    model = GPTModel(cfg).cpu().eval()
    ids = torch.randint(0, cfg["vocab_size"], (2, 5))
    logits = model(ids)
    assert logits.shape == (2, 5, 32)
    changed = ids.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 32
    torch.testing.assert_close(model(changed)[:, :-1], logits[:, :-1])

    loss = calc_loss_batch(ids, torch.roll(ids, -1, 1), model, torch.device("cpu"))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.tok_emb.weight.grad is not None
    assert torch.isfinite(model.tok_emb.weight.grad).all()
    assert generate_text_simple(model, ids, 2, 8).shape == (2, 7)
    assert generate(model, ids[:1], 2, 8, temperature=0.7, top_k=3).shape == (1, 7)

    inputs, targets = custom_collate_fn([[1, 2, 3], [4]], pad_token_id=31)
    assert inputs.tolist() == [[1, 2, 3], [4, 31, 31]]
    assert targets.tolist() == [[2, 3, 31], [31, -100, -100]]

    with TemporaryDirectory() as folder:
        csv = Path(folder) / "toy.csv"
        csv.write_text("text,Label\nhello,0\nhi,1\n", encoding="utf-8")
        spam = SpamDataset(csv, ToyTokenizer(), max_length=4, pad_token_id=31)
        assert spam[0][0].shape == (4,) and spam[1][1].item() == 1
        assert spam[1][0].tolist()[-2:] == [31, 31]

    print(f"PASS: {count} modules imported; compatibility, dataset, tiny GPT, causal mask,")
    print("loss/backward, sampling, classification data and instruction padding checked on CPU.")
    print("No training epochs, tokenizer/weight downloads, GPU or Ollama calls were run.")


if __name__ == "__main__":
    main()
