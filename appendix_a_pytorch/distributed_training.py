"""Learning example migrated from train_ddp.py. Run from the project root with python -m."""

import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
import torch


def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


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


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


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

# 原先这里另有一个 train_loader，与 prepare_dataset() 里的定义重复，
# 且 DDP 训练需要 DistributedSampler，因此仅删除这一处重复的 train_loader。
test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0
)


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

    torch.cuda.set_device(rank)


def prepare_dataset():
    # 使用上方准备好的 train_ds
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(train_ds)
    )
    return train_loader, test_loader


def main(rank, world_size, num_epochs):
    ddp_setup(rank, world_size)
    try:
        train_loader, test_loader = prepare_dataset()

        model = NeuralNetwork(num_inputs=2, num_outputs=2)
        model.to(rank)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        loss_fn = torch.nn.CrossEntropyLoss()  # 补充损失函数

        model = DDP(model, device_ids=[rank])
        for epoch in range(num_epochs):
            model.train()
            train_loader.sampler.set_epoch(epoch)  # 分布式采样每轮重新打乱
            for features, labels in train_loader:
                features, labels = features.to(rank), labels.to(rank)

                # 补充模型预测与反向传播代码
                logits = model(features)
                loss = loss_fn(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(
                    f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                    f" | Batchsize {labels.shape[0]:03d}"
                    f" | Train/Val Loss: {loss.item():.2f}"
                )

        model.eval()
        # 原代码用 train_loader 评估。保留该做法，但注意 drop_last=True 会漏掉尾部样本。
        # 多 GPU 场景下，此结果只是当前 rank 对应数据子集的准确率。
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] Training accuracy", train_acc)

        # 评估阶段使用原始模型以避免仅 rank 0 进入 DDP forward 所致的集体通信问题。
        test_acc = compute_accuracy(model.module, test_loader, device=rank)
        print(f"[GPU{rank}] Test accuracy", test_acc)
    finally:
        destroy_process_group()


if __name__ == "__main__":
    print("Number of GPUs available:", torch.cuda.device_count())
    torch.manual_seed(123)
    num_epochs = 3
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("需要至少一张 NVIDIA GPU（Colab 请选择 T4 GPU 运行时）")
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
