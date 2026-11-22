"""Learning example migrated from chapter04.py. Run from the project root with python -m."""


import math

import torch
import torch.nn as nn
import tiktoken


# ============================================================
# 1. 多头注意力机制
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False,
    ):
        super().__init__()

        assert d_out % num_heads == 0, \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Q、K、V 线性投影
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 多头拼接后的输出投影
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # 注册因果注意力掩码
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1,
            ),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 线性投影
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # (B, T, d_out) -> (B, T, H, d_h)
        keys = keys.view(
            b, num_tokens, self.num_heads, self.head_dim
        )
        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )
        values = values.view(
            b, num_tokens, self.num_heads, self.head_dim
        )

        # (B, T, H, d_h) -> (B, H, T, d_h)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 注意力得分：(B, H, T, T)
        attn_scores = queries @ keys.transpose(2, 3)

        # 因果掩码：屏蔽未来位置
        mask_bool = self.mask.bool()[
            :num_tokens, :num_tokens
        ]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 缩放点积注意力
        attn_weights = torch.softmax(
            attn_scores / self.head_dim**0.5,
            dim=-1,
        )
        attn_weights = self.dropout(attn_weights)

        # 加权求和：(B, H, T, d_h)
        context_vec = attn_weights @ values

        # 合并注意力头：(B, T, d_out)
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        # 输出线性投影
        context_vec = self.out_proj(context_vec)

        return context_vec


# ============================================================
# 2. GELU 激活函数
# ============================================================

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )


# ============================================================
# 3. 前馈神经网络
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
# 4. 层归一化
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
# 5. Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = FeedForward(cfg)

        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 多头注意力 + 残差连接
        shortcut = x

        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 前馈神经网络 + 残差连接
        shortcut = x

        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


# ============================================================
# 6. GPT-2 Small 模型配置
# ============================================================

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


# ============================================================
# 7. GPT 模型
# ============================================================

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Token 嵌入
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"],
            cfg["emb_dim"],
        )

        # 位置嵌入
        self.pos_emb = nn.Embedding(
            cfg["context_length"],
            cfg["emb_dim"],
        )

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 堆叠多个 Transformer Block
        self.trf_blocks = nn.Sequential(
            *[
                TransformerBlock(cfg)
                for _ in range(cfg["n_layers"])
            ]
        )

        # 最终归一化与输出层
        self.final_norm = LayerNorm(cfg["emb_dim"])

        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False,
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # Token 嵌入：(B, T, D)
        tok_embeds = self.tok_emb(in_idx)

        # 位置嵌入：(T, D)
        pos_embeds = self.pos_emb(
            torch.arange(
                seq_len,
                device=in_idx.device,
            )
        )

        # 广播相加：(B, T, D)
        x = tok_embeds + pos_embeds

        x = self.drop_emb(x)

        # Transformer Block
        x = self.trf_blocks(x)

        # 最终归一化
        x = self.final_norm(x)

        # 输出 logits：(B, T, V)
        logits = self.out_head(x)

        return logits


# ============================================================
# 8. 模型参数统计
# ============================================================

def print_model_statistics(model):
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Total number of parameters: {total_params:,}")

    print(
        "Token embedding layer shape:",
        model.tok_emb.weight.shape,
    )

    print(
        "Output layer shape:",
        model.out_head.weight.shape,
    )

    # 假设使用权重共享时的参数数量
    output_params = sum(
        p.numel()
        for p in model.out_head.parameters()
    )

    total_params_tied = total_params - output_params

    print(
        "Number of parameters considering weight tying:",
        f"{total_params_tied:,}",
    )

    # FP32：每个参数占 4 字节
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)

    print(
        f"Total size of the model: {total_size_mb:.2f} MB"
    )


# ============================================================
# 9. GPT 自回归文本生成（贪心解码）
# ============================================================

def generate_text_simple(
    model,
    idx,
    max_new_tokens,
    context_size,
):
    for _ in range(max_new_tokens):

        # 只截取最后 context_size 个 Token
        idx_cond = idx[:, -context_size:]

        # 模型前向传播
        with torch.no_grad():
            logits = model(idx_cond)

        # 取每条序列最后一个位置的 logits
        # (B, T, V) -> (B, V)
        logits = logits[:, -1, :]

        # Softmax 概率分布
        probas = torch.softmax(logits, dim=-1)

        # 贪心选择下一个 Token
        # (B, V) -> (B, 1)
        idx_next = torch.argmax(
            probas,
            dim=-1,
            keepdim=True,
        )

        # 拼接到原始序列末尾
        # (B, L) + (B, 1) -> (B, L+1)
        idx = torch.cat(
            (idx, idx_next),
            dim=1,
        )

    return idx


# ============================================================
# 10. 主程序
# ============================================================

def main():
    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")

    # --------------------------------------------------------
    # 10.1 Transformer Block 测试
    # --------------------------------------------------------

    x = torch.rand(2, 4, 768)

    block = TransformerBlock(GPT_CONFIG_124M)
    block.eval()

    with torch.no_grad():
        block_output = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", block_output.shape)

    # --------------------------------------------------------
    # 10.2 创建 GPT 模型
    # --------------------------------------------------------

    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    # --------------------------------------------------------
    # 10.3 GPT 前向传播测试
    # --------------------------------------------------------

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch = [
        torch.tensor(tokenizer.encode(txt1)),
        torch.tensor(tokenizer.encode(txt2)),
    ]

    batch = torch.stack(batch, dim=0)

    with torch.no_grad():
        logits = model(batch)

    print("\nInput batch:\n", batch)
    print("\nOutput shape:", logits.shape)
    print("\nLogits:\n", logits)

    # --------------------------------------------------------
    # 10.4 模型参数量统计
    # --------------------------------------------------------

    print("\nModel statistics:")
    print_model_statistics(model)

    # --------------------------------------------------------
    # 10.5 自回归文本生成
    # --------------------------------------------------------

    start_context = "Hello, I am"

    encoded = tokenizer.encode(start_context)

    encoded_tensor = torch.tensor(
        encoded
    ).unsqueeze(0)

    print("\nEncoded:", encoded)
    print("Encoded tensor shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    print("\nOutput:", out)
    print("Output length:", len(out[0]))

    # Token ID 解码为文本
    decoded_text = tokenizer.decode(
        out.squeeze(0).tolist()
    )

    print("\nGenerated text:")
    print(decoded_text)


if __name__ == "__main__":
    main()
