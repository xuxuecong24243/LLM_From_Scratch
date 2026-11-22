"""Learning example migrated from gpt_input_embeddings.py. Run from the project root with python -m."""

from project_paths import DATA_DIR

import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 定义 GPT 数据集
# ============================================================

class GPTDatasetV1(Dataset):
    """将原始文本转换为用于下一词元预测的输入-目标序列对。"""

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 将文本编码为 Token ID
        token_ids = tokenizer.encode(txt)

        # 使用滑动窗口构造训练样本
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# ============================================================
# 2. 创建数据加载器
# ============================================================

def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
):
    """创建用于批量加载 GPT 训练样本的 DataLoader。"""

    # 使用 GPT-2 分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集
    dataset = GPTDatasetV1(
        txt=txt,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride
    )

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# ============================================================
# 3. 读取文本并设置模型参数
# ============================================================

if __name__ == "__main__":
    with open(DATA_DIR / "the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab_size = 50257     # GPT-2 词表大小
    output_dim = 256       # 词元嵌入维度
    max_length = 4         # 每个样本包含的 Token 数量
    batch_size = 8         # 每个批次包含的样本数量
    context_length = max_length


    # ============================================================
    # 4. 创建 DataLoader 并获取一个批次
    # ============================================================

    dataloader = create_dataloader_v1(
        txt=raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        shuffle=False
    )

    # 获取第一个批次
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("Token IDs:\n", inputs)
    print("Inputs shape:", inputs.shape)   # [8, 4]


    # ============================================================
    # 5. 词元嵌入（Token Embedding）
    # ============================================================

    # 创建词元嵌入层
    # 权重矩阵形状：[50257, 256]
    token_embedding_layer = torch.nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=output_dim
    )

    # 将 Token ID 转换为 256 维向量
    token_embeddings = token_embedding_layer(inputs)

    print("Token embeddings shape:", token_embeddings.shape)
    # [8, 4, 256]


    # ============================================================
    # 6. 位置嵌入（Positional Embedding）
    # ============================================================

    # 创建位置嵌入层
    # 权重矩阵形状：[4, 256]
    pos_embedding_layer = torch.nn.Embedding(
        num_embeddings=context_length,
        embedding_dim=output_dim
    )

    # 生成位置索引：[0, 1, 2, 3]
    positions = torch.arange(context_length)

    # 将位置索引转换为 256 维向量
    pos_embeddings = pos_embedding_layer(positions)

    print("Position embeddings shape:", pos_embeddings.shape)
    # [4, 256]


    # ============================================================
    # 7. 合并词元嵌入与位置嵌入
    # ============================================================

    # 利用广播机制，将位置嵌入添加到每个批次样本中
    # [8, 4, 256] + [4, 256] → [8, 4, 256]
    input_embeddings = token_embeddings + pos_embeddings

    print("Input embeddings shape:", input_embeddings.shape)
    # [8, 4, 256]
