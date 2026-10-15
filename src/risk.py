"""Risk overlays and Monte Carlo robustness testing for the backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_stop(position: pd.Series, close: pd.Series, stop: float = 0.0, take: float = 0.0) -> pd.Series:
    """Exit a long when the close is `stop` below (or `take` above) the entry close.
    Re-entry only after the underlying signal goes flat and fires again."""
    if stop <= 0 and take <= 0:
        return position
    cl = close.reindex(position.index).values
    out = np.zeros(len(position))
    entry, blocked = None, False
    for i, p in enumerate(position.values):
        if p <= 0:
            entry, blocked = None, False
            continue
        if blocked:
            continue
        if entry is None:
            entry = cl[i]
        ret = cl[i] / entry - 1
        if (stop > 0 and ret <= -stop) or (take > 0 and ret >= take):
            blocked = True
            continue
        out[i] = p
    return pd.Series(out, index=position.index)


def vol_target(position: pd.Series, close: pd.Series, target_ann: float = 0.0, lookback: int = 30) -> pd.Series:
    """Scale exposure down when realised volatility exceeds the target (never levers up)."""
    if target_ann <= 0:
        return position
    vol = np.log(close).diff().rolling(lookback).std() * np.sqrt(365)
    scale = (target_ann / vol).clip(upper=1.0).reindex(position.index).fillna(1.0)
    return position * scale


def monte_carlo(ret: pd.Series, n: int = 2000, block: int = 10, seed: int = 0) -> pd.DataFrame:
    """Block-bootstrap the strategy's daily returns -> distribution of outcomes."""
    arr = ret.fillna(0).values
    N = len(arr)
    block = max(1, min(block, N))
    nb = int(np.ceil(N / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, N - block + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :N]
    eq = np.cumprod(1 + arr[idx], axis=1)
    return pd.DataFrame({"total_return": eq[:, -1] - 1,
                         "max_drawdown": (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)})
