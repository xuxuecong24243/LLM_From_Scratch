"""Learning example migrated from A9_bufengzhuang.py. Run from the project root with python -m."""

from project_paths import PROJECT_ROOT

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ==========================================
# 1. 设置随机种子
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(123)
    torch.set_printoptions(sci_mode=False)


    # ==========================================
    # 2. 创建训练集和测试集
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
    # 3. 定义 Dataset 类
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
# 4. 创建 Dataset 和 DataLoader
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
    # 5. 定义神经网络
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
# 6. 创建模型和优化器
# ==========================================
if __name__ == "__main__":
    model = NeuralNetwork(num_inputs=2, num_outputs=2)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.5
    )


    # ==========================================
    # 7. 训练模型
    # ==========================================
    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (features, labels) in enumerate(train_loader):
            # ① 前向传播
            logits = model(features)

            # ② 计算损失
            loss = F.cross_entropy(logits, labels)

            # ③ 清空梯度
            optimizer.zero_grad()

            # ④ 反向传播
            loss.backward()

            # ⑤ 更新参数
            optimizer.step()

            # ⑥ 打印训练信息
            print(
                f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                f" | Batch: {batch_idx + 1:03d}/{len(train_loader):03d}"
                f" | Loss: {loss.item():.4f}"
            )


    # ==========================================
    # 8. 查看训练集预测结果
    # ==========================================
    model.eval()

    with torch.no_grad():
        logits = model(X_train)
        probas = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

    print("\n原始输出：")
    print(logits)

    print("\n各类别概率：")
    print(probas)

    print("\n预测标签：")
    print(predictions)

    print("\n真实标签：")
    print(y_train)

    print("\n预测是否正确：")
    print(predictions == y_train)

    print("\n预测正确的数量：")
    print((predictions == y_train).sum().item())


    # ==========================================
    # 9. 计算训练集准确率
    # ==========================================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in train_loader:
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    train_accuracy = correct / total
    print(f"\n训练集准确率：{train_accuracy:.2%}")


    # ==========================================
    # 10. 计算测试集准确率
    # ==========================================
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in test_loader:
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total
    print(f"测试集准确率：{test_accuracy:.2%}")


    # ==========================================
    # 11. 保存模型参数
    # ==========================================
    torch.save(model.state_dict(), PROJECT_ROOT / "model.pth")


    # ==========================================
    # 12. 加载模型参数
    # ==========================================
    loaded_model = NeuralNetwork(num_inputs=2, num_outputs=2)

    loaded_model.load_state_dict(
        torch.load(PROJECT_ROOT / "model.pth", weights_only=True)
    )
    loaded_model.eval()


    # ==========================================
    # 13. 使用加载后的模型预测
    # ==========================================
    with torch.no_grad():
        logits = loaded_model(X_test)
        predictions = torch.argmax(logits, dim=1)

    print("\n加载模型后的测试集预测：")
    print(predictions)
