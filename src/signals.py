"""Explainable BUY / HOLD / SELL signals: model forecast + trend + momentum + RSI votes."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signals(ind: pd.DataFrame, pred_ret: pd.Series, threshold: float = 0.003,
                     min_score: int = 3) -> pd.DataFrame:
    """Score each day from -5..+5. |score| >= min_score triggers BUY / SELL, else HOLD.

    Votes: model forecast (+/-2 if the predicted move exceeds `threshold`),
    trend (close vs SMA50, +/-1), momentum (MACD histogram sign, +/-1),
    RSI extremes (<30 oversold +1, >70 overbought -1).
    """
    ind = ind.reindex(pred_ret.index)
    p = pred_ret.values
    out = pd.DataFrame(index=ind.index)
    out["pred_ret"] = p
    out["v_model"] = np.where(p > threshold, 2, np.where(p < -threshold, -2, 0))
    out["v_trend"] = np.where(ind["Close"] > ind["SMA_50"], 1, -1)
    out["v_momentum"] = np.where(ind["MACD_hist"] > 0, 1, -1)
    out["v_rsi"] = np.where(ind["RSI_14"] < 30, 1, np.where(ind["RSI_14"] > 70, -1, 0))
    out["score"] = out[["v_model", "v_trend", "v_momentum", "v_rsi"]].sum(axis=1)
    out["signal"] = np.where(out["score"] >= min_score, "BUY",
                             np.where(out["score"] <= -min_score, "SELL", "HOLD"))
    return out


def to_position(signal: pd.Series) -> pd.Series:
    """Long-only spot logic: BUY -> 1, SELL -> 0, HOLD keeps the previous position."""
    pos = signal.map({"BUY": 1.0, "SELL": 0.0})
    return pos.ffill().fillna(0.0)


def explain(row: pd.Series) -> list[str]:
    def arrow(v):
        return "bullish" if v > 0 else "bearish" if v < 0 else "neutral"
    return [
        f"Model forecast {row['pred_ret'] * 100:+.2f}% next day -> {arrow(row['v_model'])} ({row['v_model']:+d})",
        f"Price vs 50-day average -> {arrow(row['v_trend'])} ({row['v_trend']:+d})",
        f"MACD histogram -> {arrow(row['v_momentum'])} ({row['v_momentum']:+d})",
        f"RSI zone -> {arrow(row['v_rsi'])} ({row['v_rsi']:+d})",
    ]


def latest_signals(ind: pd.DataFrame, next_pred: pd.Series, threshold: float, min_score: int) -> pd.DataFrame:
    """Signal for tomorrow from every model. Index = model name."""
    last = ind.index[[-1]]
    rows = {}
    for name, p in next_pred.items():
        r = generate_signals(ind.loc[last], pd.Series([p], index=last), threshold, min_score).iloc[0]
        rows[name] = {"pred_ret": float(p), "signal": r["signal"], "score": int(r["score"])}
    return pd.DataFrame(rows).T
