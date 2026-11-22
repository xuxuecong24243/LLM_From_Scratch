"""Learning example migrated from simplified_self_attention.py. Run from the project root with python -m."""

import torch

# ==================================================
# 1. 定义输入数据
# ==================================================
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
    # 6 个 Token，每个 Token 的嵌入维度为 3


    # ==================================================
    # 2. 计算第 2 个 Token 的注意力分数
    # ==================================================
    query = inputs[1]

    attn_scores_2 = torch.empty(inputs.shape[0])

    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)

    print("第 2 个 Token 的注意力分数：")
    print(attn_scores_2)


    # ==================================================
    # 3. 注意力分数归一化
    # ==================================================

    # 方法一：直接归一化（仅用于对比）
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

    print("\n直接归一化：")
    print(attn_weights_2_tmp)
    print("权重之和：", attn_weights_2_tmp.sum())


    # 方法二：手动实现 Softmax

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


if __name__ == "__main__":
    attn_weights_2_naive = softmax_naive(attn_scores_2)

    print("\n手动 Softmax：")
    print(attn_weights_2_naive)
    print("权重之和：", attn_weights_2_naive.sum())


    # 方法三：PyTorch Softmax（推荐）
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

    print("\nPyTorch Softmax：")
    print(attn_weights_2)
    print("权重之和：", attn_weights_2.sum())


    # ==================================================
    # 4. 计算第 2 个 Token 的上下文向量
    # ==================================================
    context_vec_2 = torch.zeros(query.shape)

    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i] * x_i

    print("\n第 2 个 Token 的上下文向量：")
    print(context_vec_2)


    # ==================================================
    # 5. 计算所有 Token 的注意力分数
    # ==================================================

    # 方法一：双重循环
    attn_scores_loop = torch.empty(6, 6)

    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores_loop[i, j] = torch.dot(x_i, x_j)

    print("\n双重循环计算注意力分数：")
    print(attn_scores_loop)


    # 方法二：矩阵乘法（推荐）
    # (6, 3) @ (3, 6) = (6, 6)
    attn_scores = inputs @ inputs.T

    print("\n矩阵乘法计算注意力分数：")
    print(attn_scores)


    # ==================================================
    # 6. 计算所有 Token 的注意力权重
    # ==================================================
    # dim=-1：沿最后一个维度进行 Softmax
    # 对二维矩阵而言，即每一行分别归一化
    attn_weights = torch.softmax(attn_scores, dim=-1)

    print("\n所有 Token 的注意力权重：")
    print(attn_weights)

    # 检查每一行的权重之和
    print("\n每一行的注意力权重之和：")
    print(attn_weights.sum(dim=-1))


    # ==================================================
    # 7. 计算所有 Token 的上下文向量
    # ==================================================
    # (6, 6) @ (6, 3) = (6, 3)
    all_context_vecs = attn_weights @ inputs

    print("\n所有 Token 的上下文向量：")
    print(all_context_vecs)


    # ==================================================
    # 8. 验证两种计算方式的一致性
    # ==================================================
    print("\n验证第 2 个 Token 的上下文向量：")

    print("循环计算：", context_vec_2)
    print("矩阵计算：", all_context_vecs[1])

    print(
        "计算结果是否一致：",
        torch.allclose(context_vec_2, all_context_vecs[1])
    )
