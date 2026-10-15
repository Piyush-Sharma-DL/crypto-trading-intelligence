"""Model zoo: Linear (Ridge), Random Forest, XGBoost, LSTM. All predict next-day log-return."""
from __future__ import annotations

import copy

import numpy as np


def sklearn_models(seed: int = 42, fast: bool = False) -> dict:
    trees = 120 if fast else 300
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=trees, max_depth=8, min_samples_leaf=10,
            max_features=0.6, n_jobs=-1, random_state=seed),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=trees, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=2.0, random_state=seed, n_jobs=-1)
    except ImportError:
        models["Gradient Boosting"] = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed)
    return models


def make_sequences(arr: np.ndarray, positions, window: int) -> np.ndarray:
    """Stack `window` consecutive rows ending at each position -> (N, window, n_features)."""
    return np.stack([arr[p - window + 1: p + 1] for p in positions]).astype(np.float32)


class LSTMForecaster:
    """2-layer LSTM regressor (PyTorch) with early stopping on a chronological validation tail."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2,
                 epochs: int = 40, lr: float = 1e-3, batch: int = 64, patience: int = 8,
                 seed: int = 42):
        self.n_features, self.hidden, self.layers, self.dropout = n_features, hidden, layers, dropout
        self.epochs, self.lr, self.batch, self.patience, self.seed = epochs, lr, batch, patience, seed
        self.history: dict[str, list[float]] = {"train": [], "val": []}
        self.net = None
        self.y_scale = 1.0

    def _build(self):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s, n_in, hidden, layers, dropout):
                super().__init__()
                s.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True,
                                 dropout=dropout if layers > 1 else 0.0)
                s.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

            def forward(s, x):
                out, _ = s.lstm(x)
                return s.head(out[:, -1]).squeeze(-1)

        return Net(self.n_features, self.hidden, self.layers, self.dropout)

    def fit(self, seqs: np.ndarray, y: np.ndarray, val_frac: float = 0.1) -> "LSTMForecaster":
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.y_scale = float(np.std(y)) or 1.0
        ys = (y / self.y_scale).astype(np.float32)
        split = int(len(seqs) * (1 - val_frac))
        xtr, ytr = torch.from_numpy(seqs[:split]), torch.from_numpy(ys[:split])
        xva, yva = torch.from_numpy(seqs[split:]), torch.from_numpy(ys[split:])

        self.net = self._build()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = torch.nn.HuberLoss()
        best, best_state, bad = float("inf"), None, 0

        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(xtr))
            total = 0.0
            for i in range(0, len(perm), self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(xtr[idx]), ytr[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx)
            self.net.eval()
            with torch.no_grad():
                val = loss_fn(self.net(xva), yva).item()
            self.history["train"].append(total / len(xtr))
            self.history["val"].append(val)
            if val < best - 1e-5:
                best, best_state, bad = val, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, seqs: np.ndarray) -> np.ndarray:
        import torch

        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), 256):
                out.append(self.net(torch.from_numpy(seqs[i:i + 256])).numpy())
        return np.concatenate(out) * self.y_scale
