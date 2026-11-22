"""chapter05_pretraining: save and load checkpoints.

Original learning file: 5.4.py. Run as a module from the repository root.
"""

from project_paths import OUTPUT_DIR
import torch
from chapter04 import GPTModel
from chapter04 import generate_text_simple


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
if __name__ == "__main__":
    model = GPTModel(GPT_CONFIG_124M)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "gpt_model.pth")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()},
        OUTPUT_DIR / "model_and_optimizer.pth"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(OUTPUT_DIR / "model_and_optimizer.pth", map_location=device)
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()
