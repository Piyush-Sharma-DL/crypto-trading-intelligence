"""Technical indicators and a scale-free feature set (no look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV -> OHLCV + SMA/EMA/RSI/MACD/Bollinger/ATR columns (for charts & signals)."""
    d = df.copy()
    c = d["Close"]
    for n in (7, 21, 50):
        d[f"SMA_{n}"] = c.rolling(n).mean()
    d["EMA_12"] = c.ewm(span=12, adjust=False).mean()
    d["EMA_26"] = c.ewm(span=26, adjust=False).mean()
    d["RSI_14"] = rsi(c, 14)
    d["MACD"] = d["EMA_12"] - d["EMA_26"]
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["BB_mid"], d["BB_upper"], d["BB_lower"] = mid, mid + 2 * sd, mid - 2 * sd
    tr = pd.concat([d["High"] - d["Low"],
                    (d["High"] - c.shift()).abs(),
                    (d["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["ATR_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return d


def build_features(d: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Model inputs. Row t only uses information available at the close of day t."""
    c = d["Close"]
    f = pd.DataFrame(index=d.index)
    logret = np.log(c).diff()
    for k in (1, 2, 3, 5, 10):
        f[f"ret_{k}"] = np.log(c / c.shift(k))
    f["vol_10"] = logret.rolling(10).std()
    f["vol_30"] = logret.rolling(30).std()
    f["sma7_gap"] = c / d["SMA_7"] - 1
    f["sma21_gap"] = c / d["SMA_21"] - 1
    f["sma50_gap"] = c / d["SMA_50"] - 1
    f["ema_gap"] = d["EMA_12"] / d["EMA_26"] - 1
    f["rsi"] = d["RSI_14"] / 100
    f["macd"] = d["MACD"] / c
    f["macd_hist"] = d["MACD_hist"] / c
    f["bb_pctb"] = (c - d["BB_lower"]) / (d["BB_upper"] - d["BB_lower"])
    f["bb_width"] = (d["BB_upper"] - d["BB_lower"]) / d["BB_mid"]
    f["atr_pct"] = d["ATR_14"] / c
    f["hl_range"] = (d["High"] - d["Low"]) / c
    f["body"] = (c - d["Open"]) / d["Open"]
    vol = d["Volume"].replace(0, np.nan).ffill()
    f["vol_chg"] = np.log(vol / vol.rolling(20).mean())
    if extra is not None:
        ex = extra.reindex(d.index)
        if "FNG" in ex:
            f["fng"] = ex["FNG"] / 100
            f["fng_chg7"] = ex["FNG"].diff(7) / 100
        for col, tag in (("SPX", "spx"), ("DXY", "dxy")):
            if col in ex:
                f[f"{tag}_ret1"] = np.log(ex[col] / ex[col].shift(1))
                f[f"{tag}_ret5"] = np.log(ex[col] / ex[col].shift(5))
    return f.replace([np.inf, -np.inf], np.nan).dropna()
