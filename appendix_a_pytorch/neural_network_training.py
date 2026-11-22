"""Learning example migrated from A9.py. Run from the project root with python -m."""

from project_paths import PROJECT_ROOT

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ==========================================
# 1. 定义数据集
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
# 2. 定义神经网络
# ==========================================
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        return self.layers(x)


# ==========================================
# 3. 定义训练函数
# ==========================================
def train_model(model, dataloader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (features, labels) in enumerate(dataloader):
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

            print(
                f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                f" | Batch: {batch_idx + 1:03d}/{len(dataloader):03d}"
                f" | Loss: {loss.item():.4f}"
            )


# ==========================================
# 4. 定义准确率计算函数
# ==========================================
def compute_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in dataloader:
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ==========================================
# 5. 主程序
# ==========================================
def main():
    torch.manual_seed(123)
    torch.set_printoptions(sci_mode=False)

    # ---------- 准备数据 ----------
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

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    # ---------- 创建模型和优化器 ----------
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # ---------- 训练 ----------
    train_model(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        num_epochs=3
    )

    # ---------- 查看训练集预测结果 ----------
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

    # ---------- 计算准确率 ----------
    train_accuracy = compute_accuracy(model, train_loader)
    test_accuracy = compute_accuracy(model, test_loader)

    print(f"\n训练集准确率：{train_accuracy:.2%}")
    print(f"测试集准确率：{test_accuracy:.2%}")

    # ---------- 保存模型参数 ----------
    torch.save(model.state_dict(), PROJECT_ROOT / "model.pth")

    # ---------- 加载模型参数 ----------
    loaded_model = NeuralNetwork(num_inputs=2, num_outputs=2)
    loaded_model.load_state_dict(
        torch.load(PROJECT_ROOT / "model.pth", weights_only=True)
    )
    loaded_model.eval()


if __name__ == "__main__":
    main()
