"""
Restricted Boltzmann Machine (RBM) for unsupervised feature learning on Fashion-MNIST.

Run:
    python 03_fashionmnist_rbm.py --epochs 10

Working:
1. RBM learns a hidden representation from visible input pixels.
2. Training uses Contrastive Divergence (CD-1).
3. After training, you can extract hidden features for downstream tasks.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class RBM(torch.nn.Module):
    def __init__(self, n_visible=784, n_hidden=256):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.h_bias = torch.nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = torch.nn.Parameter(torch.zeros(n_visible))

    def sample_from_p(self, p):
        return torch.bernoulli(p)

    def v_to_h(self, v):
        p_h = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        p_v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def forward(self, v):
        _, h = self.v_to_h(v)
        _, v_recon = self.h_to_v(h)
        return v_recon

    def contrastive_divergence(self, v0):
        ph0, h0 = self.v_to_h(v0)
        pvk, vk = self.h_to_v(h0)
        phk, _ = self.v_to_h(vk)

        positive_grad = torch.matmul(v0.t(), ph0)
        negative_grad = torch.matmul(vk.t(), phk)

        batch_size = v0.size(0)
        loss = torch.mean((v0 - vk) ** 2)

        # Manual gradient-like parameter update values.
        dW = (positive_grad - negative_grad) / batch_size
        dvb = torch.mean(v0 - vk, dim=0)
        dhb = torch.mean(ph0 - phk, dim=0)
        return loss, dW, dvb, dhb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x.view(-1) > 0.5).float())  # binarize for RBM
    ])

    train_ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    rbm = RBM(n_visible=784, n_hidden=args.hidden).to(device)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        batches = 0

        for data, _ in train_loader:
            v0 = data.to(device)
            loss, dW, dvb, dhb = rbm.contrastive_divergence(v0)

            with torch.no_grad():
                rbm.W += args.lr * dW
                rbm.v_bias += args.lr * dvb
                rbm.h_bias += args.lr * dhb

            epoch_loss += loss.item()
            batches += 1

        print(f"Epoch {epoch}/{args.epochs} | Reconstruction Loss: {epoch_loss / batches:.6f}")

    torch.save({
        "W": rbm.W.detach().cpu(),
        "v_bias": rbm.v_bias.detach().cpu(),
        "h_bias": rbm.h_bias.detach().cpu()
    }, "fashionmnist_rbm.pth")
    print("RBM parameters saved as fashionmnist_rbm.pth")


if __name__ == "__main__":
    main()
