import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                update = exp_avg.sign()
                if wd != 0:
                    p.mul_(1 - lr * wd)

                p.add_(update, alpha=-lr)

class Muon(torch.optim.Optimizer):
    """
    Reference implementation of Muon optimizer
    (directional momentum-based optimizer)
    """

    def __init__(self, params, lr=1e-3, beta=0.9, weight_decay=0.0, eps=1e-8):
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)

                m = state["momentum"]
                m.mul_(beta).add_(grad, alpha=1 - beta)

                # direction normalization (Muon 核心)
                norm = torch.norm(m)
                direction = m / (norm + eps)

                if wd != 0:
                    p.mul_(1 - lr * wd)

                p.add_(direction, alpha=-lr)

# ===============================
# 1. 固定随机种子（可复现）
# ===============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===============================
# 2. CIFAR-10 Python Dataset
# ===============================
class CIFAR10Python(Dataset):
    def __init__(self, root, train=True):
        self.data = []
        self.labels = []

        if train:
            files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            files = ["test_batch"]

        for file in files:
            with open(os.path.join(root, file), "rb") as f:
                entry = pickle.load(f, encoding="bytes")
                self.data.append(entry[b"data"])
                self.labels.extend(entry[b"labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.astype(np.float32) / 255.0
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])


# ===============================
# 3. ResNet（CIFAR-10 标准简化版）
# ===============================
class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return torch.relu(out)


class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)

        self.layer1 = BasicBlock(64)
        self.layer2 = BasicBlock(64)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


# ===============================
# 4. 训练与验证
# ===============================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
    acc = correct / len(loader.dataset)
    return total_loss / len(loader), acc


# ===============================
# 5. 主函数（关键！！）
# ===============================
def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 修改为你的 CIFAR-10 路径
    cifar_root = "./cifar-10-batches-py"

    train_set = CIFAR10Python(cifar_root, train=True)
    test_set = CIFAR10Python(cifar_root, train=False)

    train_loader = DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=0,pin_memory=False
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=0,pin_memory=False
    )

    optimizers = {
    "SGD": lambda p: optim.SGD(p, lr=0.1, momentum=0.9),
    "AdamW": lambda p: optim.AdamW(p, lr=1e-3),
    "Lion": lambda p: Lion(p, lr=1e-4),
    "Muon": lambda p: Muon(p, lr=1e-3),
}


    criterion = nn.CrossEntropyLoss()
    results = {}

    for name, opt_fn in optimizers.items():
        print(f"\n=== Training with {name} ===")
        model = MiniResNet().to(device)
        optimizer = opt_fn(model.parameters())

        train_losses, val_losses, val_accs = [], [], []

        for epoch in range(100):
            tl = train_one_epoch(model, train_loader, optimizer, criterion, device)
            vl, acc = validate(model, test_loader, criterion, device)

            train_losses.append(tl)
            val_losses.append(vl)
            val_accs.append(acc)

            print(
                f"Epoch {epoch+1:02d} | "
                f"Train Loss {tl:.4f} | "
                f"Val Loss {vl:.4f} | "
                f"Acc {acc*100:.2f}%"
            )

        results[name] = (train_losses, val_losses, val_accs)

    # ===============================
    # 6. 绘制并保存 Loss 曲线
    # ===============================
    os.makedirs("figures", exist_ok=True)

    # -------- Training Loss --------
    plt.figure(figsize=(6, 4))
    for name in results:
        plt.plot(results[name][0], label=f"{name}")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.title("Training Loss Convergence")
    plt.tight_layout()
    plt.savefig("figures/train_loss.png", dpi=300)
    plt.savefig("figures/train_loss.pdf")
    plt.close()

    # -------- Validation Loss --------
    plt.figure(figsize=(6, 4))
    for name in results:
        plt.plot(results[name][1], label=f"{name}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.title("Validation Loss Convergence")
    plt.tight_layout()
    plt.savefig("figures/val_loss.png", dpi=300)
    plt.savefig("figures/val_loss.pdf")
    plt.close()

if __name__ == "__main__": 
    main()
