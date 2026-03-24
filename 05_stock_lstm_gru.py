"""
LSTM/GRU for stock price prediction (time series).

Dataset expected:
    CSV file with a 'Close' column, e.g. from Yahoo Finance export.

Run:
    python 05_stock_lstm_gru.py --csv_path stock.csv --model lstm --epochs 20
    python 05_stock_lstm_gru.py --csv_path stock.csv --model gru --epochs 20

Working:
1. Builds rolling windows from past closing prices.
2. Recurrent network learns temporal patterns from each sequence.
3. Predicts the next closing price value.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def create_sequences(series, seq_len=30):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X), np.array(y)


class RNNPredictor(nn.Module):
    def __init__(self, model_type="lstm", hidden_size=64):
        super().__init__()
        self.model_type = model_type.lower()
        if self.model_type == "lstm":
            self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        elif self.model_type == "gru":
            self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        else:
            raise ValueError("model_type must be 'lstm' or 'gru'")

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # last time step representation
        out = self.fc(out)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="stock.csv")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    close_prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices).flatten()

    X, y = create_sequences(scaled, seq_len=args.seq_len)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RNNPredictor(model_type=args.model).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            total += xb.size(0)

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {epoch_loss / total:.6f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t.to(device)).cpu().numpy()

    preds_inv = scaler.inverse_transform(preds)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
    print(f"Test RMSE: {rmse:.4f}")

    torch.save(model.state_dict(), f"stock_{args.model}.pth")
    print(f"Model saved as stock_{args.model}.pth")


if __name__ == "__main__":
    main()
