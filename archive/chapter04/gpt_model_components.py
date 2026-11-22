"""Learning example migrated from 4-7.py. Run from the project root with python -m."""


import torch
import torch.nn as nn

from chapter03_attention.multi_head_attention import MultiHeadAttention


# ============================================================
# 1. GPT-2 Small 模型配置
# ============================================================

GPT_CONFIG_124M = {
    "vocab_size": 50257,       # 词表大小
    "context_length": 1024,    # 最大上下文长度
    "emb_dim": 768,             # Token 嵌入维度
    "n_heads": 12,              # 多头注意力头数
    "n_layers": 12,             # TransformerBlock 层数
    "drop_rate": 0.1,           # Dropout 概率
    "qkv_bias": False,          # Q、K、V 是否使用偏置
}


# ============================================================
# 2. GELU 激活函数
# ============================================================

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) *
                (x + 0.044715 * torch.pow(x, 3))
            )
        )


# ============================================================
# 3. 前馈神经网络 FeedForward
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# ============================================================
# 4. 层归一化 LayerNorm
# ============================================================

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift


# ============================================================
# 5. TransformerBlock
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 多头注意力
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )

        # 前馈神经网络
        self.ff = FeedForward(cfg)

        # 两个独立的 LayerNorm
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # 残差分支上的 Dropout
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        # 第一部分：多头注意力 + 残差连接
        shortcut = x

        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)

        x = x + shortcut

        # 第二部分：前馈神经网络 + 残差连接
        shortcut = x

        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)

        x = x + shortcut

        return x


# ============================================================
# 6. GPT 模型
# ============================================================

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Token Embedding
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"],
            cfg["emb_dim"],
        )

        # Position Embedding
        self.pos_emb = nn.Embedding(
            cfg["context_length"],
            cfg["emb_dim"],
        )

        # 嵌入层 Dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 堆叠多个 TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[
                TransformerBlock(cfg)
                for _ in range(cfg["n_layers"])
            ]
        )

        # 最终 LayerNorm
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # 输出层：映射到词表大小
        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False,
        )

    def forward(self, in_idx):

        # 输入形状：(batch_size, seq_len)
        batch_size, seq_len = in_idx.shape

        # Token Embedding：(B, T) -> (B, T, D)
        tok_embeds = self.tok_emb(in_idx)

        # Position Embedding：(T,) -> (T, D)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        # Token 嵌入和位置嵌入相加：(B, T, D)
        x = tok_embeds + pos_embeds

        # 嵌入层 Dropout
        x = self.drop_emb(x)

        # TransformerBlocks：(B, T, D) -> (B, T, D)
        x = self.trf_blocks(x)

        # 最终层归一化
        x = self.final_norm(x)

        # 输出 logits：(B, T, D) -> (B, T, vocab_size)
        logits = self.out_head(x)

        return logits


# ============================================================
# 7. 模型测试
# ============================================================

if __name__ == "__main__":

    torch.manual_seed(123)

    # 测试单个 TransformerBlock
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)

    block_output = block(x)

    print("TransformerBlock:")
    print("Input shape:", x.shape)
    print("Output shape:", block_output.shape)

    # 测试完整 GPT 模型
    model = GPTModel(GPT_CONFIG_124M)

    # 两个序列，每个序列包含 4 个 Token ID
    in_idx = torch.tensor([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
    ])

    logits = model(in_idx)

    print("\nGPTModel:")
    print("Input shape:", in_idx.shape)
    print("Output shape:", logits.shape)
