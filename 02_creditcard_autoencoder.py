"""
Autoencoder for anomaly detection on the credit card fraud dataset.

Dataset expected:
    creditcard.csv

Download separately from Kaggle:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Run:
    python 02_creditcard_autoencoder.py --csv_path creditcard.csv --epochs 20

Working:
1. Uses only NORMAL samples for training.
2. Learns to reconstruct normal transaction patterns.
3. Fraud/anomalous samples usually have higher reconstruction error.
4. Threshold is chosen from normal validation reconstruction error.
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score


class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="creditcard.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    print("Loaded data shape:", df.shape)

    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train only on normal samples.
    X_normal = X[y == 0]
    X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)

    X_test = X
    y_test = y

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoEncoder(input_dim=X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = torch.utils.data.DataLoader(X_train_t, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)
            total += batch.size(0)

        print(f"Epoch {epoch}/{args.epochs} | Train Recon Loss: {epoch_loss / total:.6f}")

    # Validation reconstruction error on normal samples.
    model.eval()
    with torch.no_grad():
        val_recon = model(X_val_t.to(device)).cpu()
        val_errors = torch.mean((X_val_t - val_recon) ** 2, dim=1).numpy()

        test_recon = model(X_test_t.to(device)).cpu()
        test_errors = torch.mean((X_test_t - test_recon) ** 2, dim=1).numpy()

    # 95th percentile of normal validation error as anomaly threshold.
    threshold = np.percentile(val_errors, 95)
    preds = (test_errors > threshold).astype(int)

    print(f"Chosen threshold: {threshold:.6f}")
    print(classification_report(y_test, preds, digits=4))

    try:
        auc = roc_auc_score(y_test, test_errors)
        print(f"ROC-AUC using reconstruction error: {auc:.4f}")
    except Exception:
        print("ROC-AUC could not be computed.")

    torch.save(model.state_dict(), "creditcard_autoencoder.pth")
    print("Model saved as creditcard_autoencoder.pth")


if __name__ == "__main__":
    main()
