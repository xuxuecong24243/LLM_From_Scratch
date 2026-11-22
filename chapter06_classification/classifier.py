"""《从零构建大模型》第 6 章：GPT-2 垃圾短信分类微调（整理版）。

依赖：chapter04.py、chapter05.py、gpt_download.py 兼容入口。
CSV 位于 data/sms_spam_collection/，输出保存在 outputs/。
CSV 应包含 text（短信文本）和 Label（整数 0/1）两列。
"""
from project_paths import DATA_DIR, MODELS_DIR, OUTPUT_DIR

import time

import matplotlib.pyplot as plt
import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from chapter04 import GPTModel, generate_text_simple
from chapter05 import load_weights_into_gpt, text_to_token_ids, token_ids_to_text
from gpt_download import download_and_load_gpt2


# 1. 配置 ---------------------------------------------------------------------
CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}
MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(MODEL_CONFIGS[CHOOSE_MODEL])

BATCH_SIZE = 8
NUM_WORKERS = 0
NUM_EPOCHS = 5
EVAL_FREQ = 50
EVAL_ITER = 5
SEED = 123


# 2. 数据集 -------------------------------------------------------------------
class SpamDataset(Dataset):
    """编码、截断/填充短信，按索引返回 (token_ids, label)。"""

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            max_length = max(max_length, len(encoded_text))
        return max_length


def create_data_loaders(tokenizer):
    train_dataset = SpamDataset(DATA_DIR / "sms_spam_collection" / "train.csv", tokenizer, max_length=None)
    val_dataset = SpamDataset(
        DATA_DIR / "sms_spam_collection" / "validation.csv", tokenizer, max_length=train_dataset.max_length
    )
    test_dataset = SpamDataset(
        DATA_DIR / "sms_spam_collection" / "test.csv", tokenizer, max_length=train_dataset.max_length
    )
    print("Train max length:", train_dataset.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, drop_last=False,
    )
    return train_loader, val_loader, test_loader


# 3. 评估函数 -----------------------------------------------------------------
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]
        predicted_labels = torch.argmax(logits, dim=-1)
        num_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


# 4. 训练函数 -----------------------------------------------------------------
def train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs, eval_freq, eval_iter,
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        print(
            f"Training accuracy: {train_accuracy * 100:.2f}% | "
            f"Validation accuracy: {val_accuracy * 100:.2f}%"
        )
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


# 5. 绘图函数 -----------------------------------------------------------------
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(
        epochs_seen, val_values, linestyle="-.", label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / f"{label}-plot.pdf")
    plt.show()


def classify_review(
    text, model, tokenizer, device, max_length=None,
    pad_token_id=50256,
):
    """Classify one SMS; pass the training dataset's max_length, as in chapter 6."""
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    effective_length = min(max_length, supported_context_length)
    input_ids = input_ids[:effective_length]
    input_ids += [pad_token_id] * (effective_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        # Classification uses the final token's two class logits.
        logits = model(input_tensor)[:, -1, :]
        predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"


# 6. 主程序：模型加载 -> 分类层 -> 数据 -> 微调 -> 评估 ---------------------------
def main():
    torch.manual_seed(SEED)
    tokenizer = tiktoken.get_encoding("gpt2")
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir=MODELS_DIR
    )
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    # 微调前：查看预训练模型的文本生成能力
    for text, max_new_tokens in [
        ("Every effort moves you", 15),
        (
            "Is the following text 'spam'? Answer with 'yes' or 'no':"
            " 'You are a winner you have been specially'"
            " ' selected to receive $1000 cash or a $2000 award.'",
            23,
        ),
    ]:
        token_ids = generate_text_simple(
            model=model,
            idx=text_to_token_ids(text, tokenizer),
            max_new_tokens=max_new_tokens,
            context_size=BASE_CONFIG["context_length"],
        )
        print(token_ids_to_text(token_ids, tokenizer))

    # 冻结原模型，替换分类头，再解冻最后一个 Transformer 块与 LayerNorm
    for param in model.parameters():
        param.requires_grad = False
    torch.manual_seed(SEED)
    model.out_head = torch.nn.Linear(BASE_CONFIG["emb_dim"], 2)
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    # 单个样本演示：输出 (batch, tokens, classes)，分类使用最后一个 token
    inputs = torch.tensor(tokenizer.encode("Do you have time")).unsqueeze(0)
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape)
    with torch.no_grad():
        outputs = model(inputs)
    print("Outputs:\n", outputs)
    print("Outputs dimensions:", outputs.shape)
    print("Last output token:", outputs[:, -1, :])
    probas = torch.softmax(outputs[:, -1, :], dim=-1)
    print("Class label (softmax):", torch.argmax(probas).item())
    logits = outputs[:, -1, :]
    print("Class label (logits):", torch.argmax(logits).item())

    train_loader, val_loader, test_loader = create_data_loaders(tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 微调前：准确率和损失
    torch.manual_seed(SEED)
    for name, loader in [
        ("Training", train_loader), ("Validation", val_loader), ("Test", test_loader)
    ]:
        accuracy = calc_accuracy_loader(loader, model, device, num_batches=10)
        print(f"Initial {name} accuracy: {accuracy * 100:.2f}%")
    with torch.no_grad():
        for name, loader in [
            ("Training", train_loader), ("Validation", val_loader), ("Test", test_loader)
        ]:
            loss = calc_loss_loader(loader, model, device, num_batches=5)
            print(f"Initial {name} loss: {loss:.3f}")

    # 微调训练
    torch.manual_seed(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    start_time = time.time()
    train_losses, val_losses, train_accs, val_accs, examples_seen = (
        train_classifier_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=NUM_EPOCHS, eval_freq=EVAL_FREQ, eval_iter=EVAL_ITER,
        )
    )
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"Training completed in {elapsed_minutes:.2f} minutes.")

    # 曲线：损失按 eval_freq 记录，准确率每轮记录
    epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
    examples_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_tensor, train_losses, val_losses)

    epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_accs))
    examples_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(
        epochs_tensor, examples_tensor, train_accs, val_accs, label="accuracy"
    )

    # 微调后：在全部批次上计算最终准确率
    for name, loader in [
        ("Training", train_loader), ("Validation", val_loader), ("Test", test_loader)
    ]:
        accuracy = calc_accuracy_loader(loader, model, device)
        print(f"{name} accuracy: {accuracy * 100:.2f}%")

    # 保留原 6-6.py 中的单条短信推理与权重保存示例。
    text = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )
    print(classify_review(
        text, model, tokenizer, device,
        max_length=train_loader.dataset.max_length,
    ))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "review_classifier.pth"
    torch.save(model.state_dict(), checkpoint_path)
    model_state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_state_dict)


if __name__ == "__main__":
    main()
