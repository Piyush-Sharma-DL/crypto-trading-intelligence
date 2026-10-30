"""IMBA ALGO: Fibonacci-channel trend signals.

An independent implementation of the *rules* of the TradingView "[IMBA] ALGO" indicator (not a copy of its code):
  channel  = highest high / lowest low over  int(sensitivity x 10)  candles
  levels   = high - range x (0.236, 0.382, 0.5, 0.618, 0.786);  the 50% level is the trend line
  LONG     when a candle closes at/above the 50% line AND at/above the 23.6% level (top of the channel), while not already long
  SHORT    when a candle closes at/below the 50% line AND at/below the 78.6% level (bottom of the channel), while not already short
  trade    entry = signal close;  stop = 78.6% level (long) / 23.6% level (short) or a fixed % from entry;
           targets = entry +/- (1, 2, 3, 4)%  (the original scales out 40/30/20/10% and moves to break-even)
Signals alternate LONG, SHORT, LONG... and use only closed candles, so they cannot repaint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIBS = (0.236, 0.382, 0.5, 0.618, 0.786)
COLUMNS = ["hi", "lo", "f236", "f382", "f5", "f618", "f786", "trend", "signal", "entry", "sl", "tp1", "tp2", "tp3", "tp4"]


def imba_algo(df: pd.DataFrame, sensitivity: float = 18.0, tps=(1.0, 2.0, 3.0, 4.0), sl_pct: float = 0.0,
              fixed_stop: bool = False) -> pd.DataFrame:
    n = max(2, int(sensitivity * 10 + 1e-9))
    hi, lo = df["High"].rolling(n).max(), df["Low"].rolling(n).min()
    rng = hi - lo
    f = {k: hi - rng * k for k in FIBS}
    c = df["Close"].values
    f236, f5, f786 = f[0.236].values, f[0.5].values, f[0.786].values
    sig, trend = np.zeros(len(df), int), np.zeros(len(df), int)
    long_t = short_t = False
    for i in range(len(df)):
        if np.isnan(f5[i]):
            continue
        can_long = c[i] >= f5[i] and c[i] >= f236[i] and not long_t
        can_short = c[i] <= f5[i] and c[i] <= f786[i] and not short_t
        if can_long:
            long_t, short_t, sig[i] = True, False, 1
        elif can_short:
            short_t, long_t, sig[i] = True, False, -1
        trend[i] = 1 if long_t else -1 if short_t else 0

    out = pd.DataFrame({"hi": hi, "lo": lo, "f236": f[0.236], "f382": f[0.382], "f5": f[0.5], "f618": f[0.618],
                        "f786": f[0.786], "trend": trend, "signal": sig}, index=df.index)
    s = out["signal"].values
    entry = np.where(s != 0, c, np.nan)
    sl_long = c * (1 - sl_pct / 100) if fixed_stop else f786 * (1 - sl_pct / 100)
    sl_short = c * (1 + sl_pct / 100) if fixed_stop else f236 * (1 + sl_pct / 100)
    out["entry"] = entry
    out["sl"] = np.where(s == 1, sl_long, np.where(s == -1, sl_short, np.nan))
    for k, tp in enumerate(tps, start=1):
        out[f"tp{k}"] = np.where(s == 1, c * (1 + tp / 100), np.where(s == -1, c * (1 - tp / 100), np.nan))
    return out[COLUMNS]


def ladder(df: pd.DataFrame, res: pd.DataFrame) -> dict | None:
    """Levels of the newest signal and which of them price has touched since (for drawing the TP/SL ladder)."""
    sigs = res.index[res["signal"] != 0]
    if not len(sigs):
        return None
    t0 = sigs[-1]
    r = res.loc[t0]
    after = df.loc[df.index > t0]
    side = int(r["signal"])
    touch = (lambda p: bool((after["High"] >= p).any())) if side == 1 else (lambda p: bool((after["Low"] <= p).any()))
    stop_hit = bool((after["Low"] <= r["sl"]).any()) if side == 1 else bool((after["High"] >= r["sl"]).any())
    tps = [(k, float(r[f"tp{k}"]), touch(r[f"tp{k}"])) for k in (1, 2, 3, 4)]
    return {"t0": t0, "side": side, "entry": float(r["entry"]), "sl": float(r["sl"]), "sl_hit": stop_hit, "tps": tps}


def status_row(res: pd.DataFrame, price: float) -> pd.DataFrame:
    last = res.iloc[-1]
    if np.isnan(last["f5"]):
        return pd.DataFrame([{"Indicator": "IMBA ALGO", "Value": "n/a", "Reading": "not enough candles yet"}])
    rng = last["hi"] - last["lo"]
    pos = (price - last["lo"]) / rng * 100 if rng > 0 else 50.0
    name = {1: "🟢 long trend", -1: "🔴 short trend", 0: "⚪ no trend yet"}[int(last["trend"])]
    sigs = res.index[res["signal"] != 0]
    when = f", last signal {len(res) - 1 - res.index.get_loc(sigs[-1])} candle(s) ago" if len(sigs) else ""
    return pd.DataFrame([{"Indicator": "IMBA ALGO (fib trend)", "Value": f"{last['f5']:,.2f}",
                          "Reading": f"{name}; price at {pos:.0f}% of the channel{when}"}])
