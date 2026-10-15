"""Market data loading: Yahoo Finance (live) -> local CSV cache -> synthetic demo data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_yahoo(ticker: str, years: int) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, period=f"{years}y", interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[COLUMNS].copy()
    df = df[df["Close"].notna()]                       # keep every bar that has a close
    df["Volume"] = df["Volume"].fillna(0)
    for col in ("Open", "High", "Low"):
        df[col] = df[col].fillna(df["Close"])
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def completed_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the in-progress daily bar (crypto 'days' close at 00:00 UTC)."""
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    return df[df.index < today]


def synthetic_prices(years: int = 5, seed: int = 7) -> pd.DataFrame:
    """Random-walk OHLCV with volatility regimes. For offline demos/tests only."""
    n = int(years * 365)
    rng = np.random.default_rng(seed)
    vol = 0.022 * np.exp(0.5 * np.sin(np.arange(n) / 60))
    r = np.zeros(n)
    eps = rng.normal(size=n)
    for i in range(1, n):
        r[i] = 0.0004 + 0.05 * r[i - 1] + vol[i] * eps[i]
    close = 20000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    volume = rng.lognormal(20, 0.4, n) * (1 + 5 * np.abs(r))
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    idx = pd.date_range(end=end, periods=n, name="Date")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=idx)


def load_prices(ticker: str, years: int = 5, offline: bool = False) -> tuple[pd.DataFrame, str]:
    """Return (OHLCV dataframe, human-readable source label)."""
    cache = DATA_DIR / f"{ticker.replace('/', '_')}.csv"
    if not offline:
        try:
            df = fetch_yahoo(ticker, years)
            DATA_DIR.mkdir(exist_ok=True)
            df.to_csv(cache)
            return df, f"Yahoo Finance (live, latest bar {df.index[-1].date()})"
        except Exception as exc:  # network down, bad ticker, rate limit...
            if cache.exists():
                df = pd.read_csv(cache, index_col=0, parse_dates=True)
                return df, f"Local cache (live fetch failed: {type(exc).__name__})"
    return synthetic_prices(years, seed=sum(map(ord, ticker))), "Synthetic demo data (random walk, NOT real prices)"


# ------------------------------------------------------------------ extra (non-price) data
def _cached(name: str, fetch):
    path = DATA_DIR / f"{name}.csv"
    try:
        s = fetch()
        DATA_DIR.mkdir(exist_ok=True)
        s.to_csv(path)
        return s
    except Exception:
        if path.exists():
            s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            return s
        raise


def fetch_fng() -> pd.Series:
    """Crypto Fear & Greed Index (0-100) from alternative.me. Free, no key."""
    import requests

    r = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=15)
    r.raise_for_status()
    rows = r.json()["data"]
    s = pd.Series({pd.to_datetime(int(x["timestamp"]), unit="s").normalize(): float(x["value"])
                   for x in rows}).sort_index()
    s.name = "FNG"
    return s


def fetch_close(ticker: str, years: int) -> pd.Series:
    import yfinance as yf

    df = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s.name = ticker
    return s


def load_extras(index: pd.DatetimeIndex, use_fng: bool = True, use_macro: bool = True,
                offline: bool = False) -> tuple[pd.DataFrame | None, list[str]]:
    """Sentiment + macro context aligned to the price index (forward-filled, no look-ahead:
    every value at date t is published before the t daily close)."""
    notes: list[str] = []
    if offline:
        return None, ["Offline mode: sentiment and macro features are disabled."]
    years = max(2, int((index[-1] - index[0]).days / 365) + 2)
    cols = {}
    jobs = []
    if use_fng:
        jobs.append(("FNG", "fng", fetch_fng))
    if use_macro:
        jobs.append(("SPX", "spx", lambda: fetch_close("^GSPC", years)))
        jobs.append(("DXY", "dxy", lambda: fetch_close("DX-Y.NYB", years)))
    for col, name, fn in jobs:
        try:
            cols[col] = _cached(name, fn)
        except Exception as exc:
            notes.append(f"Could not load {col} ({type(exc).__name__}); that feature is skipped.")
    if not cols:
        return None, notes
    df = pd.DataFrame(cols)
    df = df.reindex(df.index.union(index)).sort_index().ffill(limit=5).reindex(index)
    return df, notes
