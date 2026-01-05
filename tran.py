import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from typing import List
import matplotlib.pyplot as plt

# ===============================
# 0. Lion & Muon Optimizer
# ===============================
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, _ = group["betas"]
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

                if wd != 0:
                    p.mul_(1 - lr * wd)

                p.add_(exp_avg.sign(), alpha=-lr)


class Muon(torch.optim.Optimizer):
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

                direction = m / (torch.norm(m) + eps)

                if wd != 0:
                    p.mul_(1 - lr * wd)

                p.add_(direction, alpha=-lr)


# ===============================
# 1. 固定随机种子
# ===============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===============================
# 2. WikiText 本地数据
# ===============================
def read_wikitext(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def tokenize(lines: List[str]) -> List[str]:
    tokens = []
    for line in lines:
        tokens.extend(line.strip().split())
    return tokens


def build_vocab(tokens: List[str]):
    counter = Counter(tokens)
    vocab = {"<unk>": 0}
    for tok in counter:
        vocab[tok] = len(vocab)
    return vocab


def numericalize(tokens: List[str], vocab):
    return torch.tensor([vocab.get(t, 0) for t in tokens], dtype=torch.long)


def batchify(data, batch_size):
    nbatch = data.size(0) // batch_size
    data = data[: nbatch * batch_size]
    return data.view(batch_size, -1).t().contiguous()


def get_batch(source, i, seq_len):
    seq_len = min(seq_len, source.size(0) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target.reshape(-1)


def build_wikitext_local(root, batch_size=32, seq_len=35):
    train = tokenize(read_wikitext(f"{root}/wiki.train.tokens"))
    valid = tokenize(read_wikitext(f"{root}/wiki.valid.tokens"))

    vocab = build_vocab(train)
    print(f"Vocab size: {len(vocab)}")

    train_data = batchify(numericalize(train, vocab), batch_size)
    valid_data = batchify(numericalize(valid, vocab), batch_size)

    return train_data, valid_data, vocab, seq_len


# ===============================
# 3. Transformer LM
# ===============================
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(512, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

        nn.init.normal_(self.embed.weight, 0, 0.02)
        nn.init.normal_(self.fc.weight, 0, 0.02)

    def forward(self, x):
        seq_len = x.size(0)
        h = self.embed(x) + self.pos[:seq_len].unsqueeze(1)
        h = self.encoder(h)
        return self.fc(h)


# ===============================
# 4. Train / Eval
# ===============================
def run_epoch(model, data, optimizer, criterion, seq_len, device, train=True):
    model.train() if train else model.eval()
    total_loss, steps = 0.0, 0

    with torch.set_grad_enabled(train):
        for i in range(0, data.size(0) - 1, seq_len):
            x, y = get_batch(data, i, seq_len)
            x, y = x.to(device), y.to(device)

            if train:
                optimizer.zero_grad()

            out = model(x)
            loss = criterion(out.view(-1, out.size(-1)), y)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            steps += 1

    return total_loss / steps


# ===============================
# 5. Main + Plot
# ===============================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, valid_data, vocab, seq_len = build_wikitext_local(
        "/data/jl/res/wikitext-2"
    )

    optimizers = {
        "SGD": lambda p: optim.SGD(p, lr=0.1, momentum=0.9),
        "AdamW": lambda p: optim.AdamW(p, lr=3e-4),
        "Lion": lambda p: Lion(p, lr=1e-4),
        "Muon": lambda p: Muon(p, lr=1e-3),
    }

    criterion = nn.CrossEntropyLoss()
    epochs = 100
    history = {}

    for name, opt_fn in optimizers.items():
        print(f"\n=== {name} ===")
        model = TransformerLM(len(vocab)).to(device)
        optimizer = opt_fn(model.parameters())

        train_losses, val_losses = [], []

        for ep in range(epochs):
            tl = run_epoch(model, train_data, optimizer, criterion, seq_len, device, True)
            vl = run_epoch(model, valid_data, optimizer, criterion, seq_len, device, False)

            train_losses.append(tl)
            val_losses.append(vl)

            print(f"Epoch {ep+1:02d} | Train {tl:.4f} | Val {vl:.4f}")

        history[name] = (train_losses, val_losses)

    plot_losses(history, epochs)


def plot_losses(history, epochs):
    x = range(1, epochs + 1)

    # =========================
    # Figure 1: Train Loss
    # =========================
    plt.figure(figsize=(8, 6))
    for name, (train_l, _) in history.items():
        plt.plot(x, train_l, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Mean Cross-Entropy Loss")
    plt.title("Training Loss on WikiText-2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("optimizer_train_loss.png", dpi=300)
    plt.show()

    # =========================
    # Figure 2: Validation Loss
    # =========================
    plt.figure(figsize=(8, 6))
    for name, (_, val_l) in history.items():
        plt.plot(x, val_l, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Mean Cross-Entropy Loss")
    plt.title("Validation Loss on WikiText-2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("optimizer_validation_loss.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
