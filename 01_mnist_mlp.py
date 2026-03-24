"""
MNIST classification using a simple Multi-Layer Perceptron (MLP)
with selectable activation and loss functions.

Run:
    python 01_mnist_mlp.py --epochs 5 --activation relu --loss cross_entropy

This script automatically downloads MNIST using torchvision.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MLP(nn.Module):
    """
    A simple feed-forward network for image classification.

    Working:
    1. Input image (28x28) is flattened to a 784-dimensional vector.
    2. The vector passes through two hidden layers.
    3. An activation function (ReLU/Tanh/Sigmoid) adds non-linearity.
    4. Final layer outputs class scores for digits 0-9.
    """
    def __init__(self, activation="relu", num_classes=10):
        super().__init__()

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation. Use relu/tanh/sigmoid.")

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            act,
            nn.Linear(256, 128),
            act,
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def get_loss_fn(loss_name: str):
    """
    Returns the selected loss function.

    cross_entropy:
        Best for multi-class classification with class-index targets.
    nll:
        Negative log-likelihood; expects log probabilities as input.
    """
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "nll":
        return nn.NLLLoss()
    else:
        raise ValueError("Unsupported loss. Use cross_entropy or nll.")


def train_one_epoch(model, loader, optimizer, loss_name, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(images)

        # If using NLLLoss, convert logits to log-probabilities first.
        if loss_name == "nll":
            loss = nn.NLLLoss()(torch.log_softmax(logits, dim=1), labels)
            preds = torch.argmax(torch.log_softmax(logits, dim=1), dim=1)
        else:
            loss = nn.CrossEntropyLoss()(logits, labels)
            preds = torch.argmax(logits, dim=1)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy", "nll"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = transforms.ToTensor()

    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MLP(activation=args.activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Call once to validate the argument choice and document intent.
    _ = get_loss_fn(args.loss)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, args.loss, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    torch.save(model.state_dict(), "mnist_mlp.pth")
    print("Model saved as mnist_mlp.pth")


if __name__ == "__main__":
    main()
