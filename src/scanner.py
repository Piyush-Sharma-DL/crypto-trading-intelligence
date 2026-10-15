"""Multi-coin scanner: fast ensemble forecast + signal for many assets at once."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import completed_bars, load_prices
from .features import add_indicators, build_features
from .models import sklearn_models
from .signals import generate_signals

COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
         "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD"]
ICON = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}


def quick_signal(ticker: str, years: int, threshold: float, min_score: int, offline: bool = False) -> dict:
    df, _ = load_prices(ticker, years, offline)
    df = completed_bars(df)
    ind_full = add_indicators(df)
    X = build_features(ind_full)
    ind = ind_full.reindex(X.index)
    close = ind["Close"]
    y = np.log(close.shift(-1) / close).values
    n_lab = len(X) - 1
    preds = []
    for model in sklearn_models(fast=True).values():
        model.fit(X.values[:n_lab], y[:n_lab])
        preds.append(float(model.predict(X.values[n_lab:])[0]))
    pred = float(np.mean(preds))
    last = ind.index[[-1]]
    s = generate_signals(ind.loc[last], pd.Series([pred], index=last), threshold, min_score).iloc[0]
    logret = np.log(close).diff()
    return {
        "Asset": ticker, "Signal": ICON[s["signal"]], "Score": int(s["score"]),
        "Forecast (%)": (np.exp(pred) - 1) * 100,
        "Model agreement (%)": float(np.mean(np.array(preds) > 0) * 100),
        "RSI": float(ind["RSI_14"].iloc[-1]),
        "7d (%)": (close.iloc[-1] / close.iloc[-8] - 1) * 100,
        "30d (%)": (close.iloc[-1] / close.iloc[-31] - 1) * 100,
        "Volatility 30d (ann. %)": float(logret.tail(30).std() * np.sqrt(365) * 100),
        "Above SMA50": "yes" if close.iloc[-1] > ind["SMA_50"].iloc[-1] else "no",
        "Last close ($)": float(close.iloc[-1]),
    }


def scan(tickers, years=3, threshold=0.003, min_score=3, offline=False, progress=None) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers):
        if progress:
            progress(i, len(tickers), t)
        try:
            rows.append(quick_signal(t, years, threshold, min_score, offline))
        except Exception as exc:
            rows.append({"Asset": t, "Signal": f"error: {type(exc).__name__}"})
    if progress:
        progress(len(tickers), len(tickers), "done")
    df = pd.DataFrame(rows)
    return df.sort_values("Score", ascending=False, na_position="last").reset_index(drop=True) if "Score" in df else df
