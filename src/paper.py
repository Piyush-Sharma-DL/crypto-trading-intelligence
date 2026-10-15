"""Paper-trading log: every forecast is saved, then scored against what actually happened.
This builds a genuine out-of-sample track record over time (the strongest evidence a model can give)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import data
from .signals import to_position

COLS = ["ticker", "asof", "model", "pred_ret", "signal", "score", "close_asof", "logged_at"]


def _path():
    return data.DATA_DIR / "paper_log.csv"


def load_log(ticker: str | None = None) -> pd.DataFrame:
    p = _path()
    if not p.exists():
        log = pd.DataFrame(columns=COLS)
        log["asof"] = pd.to_datetime(log["asof"])
        return log
    log = pd.read_csv(p, parse_dates=["asof"])
    return log[log["ticker"] == ticker] if ticker else log


def log_forecasts(ticker: str, asof: pd.Timestamp, close_asof: float, latest: pd.DataFrame) -> int:
    """Append one row per model for this (ticker, as-of date). Duplicates are ignored."""
    log = load_log()
    done = set(zip(log["ticker"], log["asof"].dt.normalize(), log["model"]))
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")
    rows = [dict(ticker=ticker, asof=asof.normalize(), model=m, pred_ret=r["pred_ret"], signal=r["signal"],
                 score=r["score"], close_asof=close_asof, logged_at=now)
            for m, r in latest.iterrows() if (ticker, asof.normalize(), m) not in done]
    if rows:
        data.DATA_DIR.mkdir(exist_ok=True)
        new = pd.DataFrame(rows)
        out = new if log.empty else pd.concat([log, new], ignore_index=True)
        out.to_csv(_path(), index=False)
    return len(rows)


def score_log(log: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Attach realised next-day returns (NaN while still pending)."""
    log = log.copy()
    realised = (close.shift(-1) / close - 1).reindex(pd.to_datetime(log["asof"]))
    log["realized_ret"] = realised.values
    log["hit"] = np.where(log["realized_ret"].isna(), np.nan,
                          (np.sign(log["pred_ret"]) == np.sign(log["realized_ret"])).astype(float))
    return log


def summary(scored: pd.DataFrame, fee: float = 0.001) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model hit rate table and equity curves (long on BUY, flat on SELL, HOLD keeps position)."""
    done = scored.dropna(subset=["realized_ret"])
    rows, curves = {}, {}
    for model, g in done.groupby("model"):
        g = g.sort_values("asof")
        pos = to_position(g["signal"].reset_index(drop=True))
        ret = pos.values * g["realized_ret"].values
        turn = np.abs(np.diff(np.r_[0.0, pos.values]))
        eq = np.cumprod(1 + ret - turn * fee)
        k, n = int(g["hit"].sum()), len(g)
        rows[model] = {"Forecasts scored": n, "Hit rate (%)": k / n * 100,
                       "p (> 50%)": float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue),
                       "Paper return (%)": (eq[-1] - 1) * 100}
        curves[model] = pd.Series(eq, index=g["asof"].values)
    return pd.DataFrame(rows).T, pd.DataFrame(curves)
