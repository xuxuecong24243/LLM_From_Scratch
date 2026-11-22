"""Learning example migrated from A11.py. Run from the project root with python -m."""


# ==========================================
# 1. 导入所需库
# ==========================================
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ==========================================
# 2. 设置随机种子
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(123)
    torch.set_printoptions(sci_mode=False)


    # ==========================================
    # 3. 创建训练集和测试集
    # ==========================================
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])

    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6]
    ])

    y_test = torch.tensor([0, 1])


    # ==========================================
    # 4. 定义 Dataset 类
    # ==========================================

class ToyDataset(Dataset):

    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


# ==========================================
# 5. 创建 Dataset 和 DataLoader
# ==========================================
if __name__ == "__main__":
    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )


    # ==========================================
    # 6. 定义神经网络模型
    # ==========================================

class NeuralNetwork(torch.nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 第一个隐藏层
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 第二个隐藏层
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # 输出层
            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        return self.layers(x)


# ==========================================
# 7. 配置 GPU 并创建模型
# ==========================================
if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"当前使用设备: {device}")

    model = NeuralNetwork(
        num_inputs=2,
        num_outputs=2
    )

    # 将模型移动到 GPU（或 CPU）
    model = model.to(device)


    # ==========================================
    # 8. 创建 SGD 优化器
    # ==========================================
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.5
    )


    # ==========================================
    # 9. 训练神经网络
    # ==========================================
    num_epochs = 3

    for epoch in range(num_epochs):

        # 切换到训练模式
        model.train()

        for batch_idx, (features, labels) in enumerate(train_loader):

            # ① 将数据移动到 GPU（或 CPU）
            features = features.to(device)
            labels = labels.to(device)

            # ② 前向传播
            logits = model(features)

            # ③ 计算交叉熵损失
            loss = F.cross_entropy(logits, labels)

            # ④ 清空旧梯度
            optimizer.zero_grad()

            # ⑤ 反向传播
            loss.backward()

            # ⑥ 使用 SGD 更新模型参数
            optimizer.step()

            # ⑦ 打印训练信息
            print(
                f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                f" | Batch {batch_idx+1:03d}/{len(train_loader):03d}"
                f" | Train Loss: {loss:.2f}"
            )


    # ==========================================
    # 10. 切换到评估模式
    # ==========================================
    model.eval()

    # 后续可以添加模型评估代码
