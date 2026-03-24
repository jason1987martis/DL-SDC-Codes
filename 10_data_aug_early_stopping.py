"""
Data augmentation + early stopping for better model generalization on CIFAR-10.

Run:
    python 10_data_aug_early_stopping.py --epochs 30 --patience 5

Working:
1. Data augmentation creates varied training images.
2. Validation accuracy is monitored after each epoch.
3. Early stopping stops training when validation accuracy stops improving.
"""

import argparse
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])
    test_transform = transforms.ToTensor()

    full_train = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
    val_base = datasets.CIFAR10(root="data", train=True, download=True, transform=test_transform)
    test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)

    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size

    generator = torch.Generator().manual_seed(42)
    train_subset, _ = random_split(full_train, [train_size, val_size], generator=generator)
    _, val_subset = random_split(val_base, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
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

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {total_loss / total:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("Validation improved. Saving best weights in memory.")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_weights)
    test_acc = evaluate(model, test_loader, device)
    print(f"Best Val Acc: {best_val_acc:.2f}% | Final Test Acc: {test_acc:.2f}%")

    torch.save(model.state_dict(), "cifar10_aug_earlystop.pth")
    print("Model saved as cifar10_aug_earlystop.pth")


if __name__ == "__main__":
    main()
