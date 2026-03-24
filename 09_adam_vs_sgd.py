"""
Compare Adam vs SGD on MNIST.

Run:
    python 09_adam_vs_sgd.py --epochs 5

Working:
1. Trains the same network twice: once with SGD and once with Adam.
2. Stores loss and accuracy for both optimizers.
3. Helps compare convergence speed and final performance.
"""

import argparse
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return 100.0 * correct / total


def train_model(model, optimizer, train_loader, test_loader, device, epochs):
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

        test_acc = evaluate(model, test_loader, device)
        avg_loss = total_loss / total
        history.append((avg_loss, test_acc))
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%")

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_sgd", type=float, default=0.01)
    parser.add_argument("--lr_adam", type=float, default=0.001)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.ToTensor()

    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    base_model = SimpleNet().to(device)

    print("\nTraining with SGD")
    sgd_model = copy.deepcopy(base_model)
    sgd_optimizer = torch.optim.SGD(sgd_model.parameters(), lr=args.lr_sgd, momentum=0.9)
    sgd_history = train_model(sgd_model, sgd_optimizer, train_loader, test_loader, device, args.epochs)

    print("\nTraining with Adam")
    adam_model = copy.deepcopy(base_model)
    adam_optimizer = torch.optim.Adam(adam_model.parameters(), lr=args.lr_adam)
    adam_history = train_model(adam_model, adam_optimizer, train_loader, test_loader, device, args.epochs)

    print("\nSummary")
    print("Epoch | SGD Loss | SGD Acc | Adam Loss | Adam Acc")
    for i, (sgd_row, adam_row) in enumerate(zip(sgd_history, adam_history), start=1):
        print(f"{i:>5} | {sgd_row[0]:>8.4f} | {sgd_row[1]:>7.2f}% | {adam_row[0]:>9.4f} | {adam_row[1]:>8.2f}%")

    torch.save(sgd_model.state_dict(), "mnist_sgd.pth")
    torch.save(adam_model.state_dict(), "mnist_adam.pth")
    print("Saved mnist_sgd.pth and mnist_adam.pth")


if __name__ == "__main__":
    main()
