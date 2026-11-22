"""Learning example migrated from self_attention_from_scratch.py. Run from the project root with python -m."""


import torch

# ============================================================
# 1. 准备输入数据
# ============================================================

if __name__ == "__main__":
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]
    ])

    # inputs.shape = (6, 3)


    # ============================================================
    # 2. 简单注意力机制：计算单个查询向量的注意力
    # ============================================================

    # 2.1 选择第二个输入向量作为查询向量
    query = inputs[1]

    # 2.2 计算注意力分数（点积）
    attn_scores_2 = torch.empty(inputs.shape[0])

    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)

    print("Attention scores:", attn_scores_2)


    # 2.3 方法一：直接归一化
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

    print("Normalized weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())


    # 2.4 方法二：手动实现 Softmax

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


if __name__ == "__main__":
    attn_weights_2_naive = softmax_naive(attn_scores_2)

    print("Naive softmax weights:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())


    # 2.5 方法三：使用 PyTorch Softmax
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

    print("Softmax weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())


    # 2.6 计算上下文向量（加权求和）
    context_vec_2 = torch.zeros_like(query)

    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i] * x_i

    print("Context vector:", context_vec_2)


    # ============================================================
    # 3. 简单自注意力机制：计算所有输入的注意力
    # ============================================================

    # 3.1 方法一：双重循环计算注意力分数矩阵
    attn_scores_loop = torch.empty(6, 6)

    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores_loop[i, j] = torch.dot(x_i, x_j)

    print("Attention scores (loop):")
    print(attn_scores_loop)


    # 3.2 方法二：矩阵乘法计算注意力分数
    attn_scores = inputs @ inputs.T

    print("Attention scores (matrix):")
    print(attn_scores)


    # 3.3 使用 Softmax 计算注意力权重
    attn_weights = torch.softmax(attn_scores, dim=-1)

    print("Attention weights:")
    print(attn_weights)

    # 检查每一行的注意力权重之和
    print("All row sums:", attn_weights.sum(dim=-1))


    # 3.4 计算所有上下文向量
    all_context_vecs = attn_weights @ inputs

    print("All context vectors:")
    print(all_context_vecs)


    # ============================================================
    # 4. 引入 Query、Key、Value 权重矩阵
    # ============================================================

    # 4.1 设置输入和输出维度
    x_2 = inputs[1]

    d_in = inputs.shape[1]  # 输入维度：3
    d_out = 2               # 输出维度：2

    torch.manual_seed(123)


    # 4.2 初始化权重矩阵
    # requires_grad=False：仅用于演示，不参与梯度更新
    W_query = torch.nn.Parameter(
        torch.rand(d_in, d_out), requires_grad=False
    )

    W_key = torch.nn.Parameter(
        torch.rand(d_in, d_out), requires_grad=False
    )

    W_value = torch.nn.Parameter(
        torch.rand(d_in, d_out), requires_grad=False
    )


    # 4.3 计算第二个输入向量的 Query、Key、Value
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value

    print("Query 2:", query_2)
    print("Key 2:", key_2)
    print("Value 2:", value_2)


    # 4.4 计算所有输入向量的 Key 和 Value
    keys = inputs @ W_key
    values = inputs @ W_value

    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)


    # ============================================================
    # 5. 缩放点积注意力（Scaled Dot-Product Attention）
    # ============================================================

    # 5.1 计算第二个输入向量与自身的注意力分数
    attn_score_22 = torch.dot(query_2, keys[1])

    print("Attention score 2-2:", attn_score_22)


    # 5.2 计算第二个查询向量与所有 Key 的注意力分数
    attn_scores_2 = query_2 @ keys.T

    print("Attention scores 2:", attn_scores_2)


    # 5.3 缩放注意力分数并应用 Softmax
    d_k = keys.shape[-1]

    attn_weights_2 = torch.softmax(
        attn_scores_2 / d_k**0.5,
        dim=-1
    )

    print("Scaled attention weights:", attn_weights_2)


    # 5.4 根据注意力权重计算上下文向量
    context_vec_2 = attn_weights_2 @ values

    print("Final context vector:", context_vec_2)
