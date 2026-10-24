"""Live market data: Kraken public API (real-time, every timeframe) with Yahoo Finance as fallback.
No API keys needed."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

KRAKEN = "https://api.kraken.com/0/public"
TIMEFRAMES = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1D": 1440, "1W": 10080}
_KRAKEN_BASE = {"BTC": ["XBT", "BTC"], "DOGE": ["DOGE", "XDG"]}
_KRAKEN_QUOTES = {"USD", "USDT", "EUR", "GBP"}
# (yahoo interval, yahoo period, resample rule)
_YAHOO = {"5m": ("5m", "5d", None), "15m": ("15m", "10d", None), "30m": ("30m", "30d", None),
          "1h": ("1h", "90d", None), "4h": ("1h", "360d", "4h"), "1D": ("1d", "3y", None),
          "1W": ("1d", "10y", "W-MON")}


def _http_get_json(url: str, params: dict | None = None, timeout: int = 10) -> dict:
    import requests

    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "crypto-trading-intelligence"})
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------ helpers
def kraken_pairs(ticker: str) -> list[str]:
    """'BTC-USD' -> ['XBTUSD', 'BTCUSD']. Empty if the quote currency is not on Kraken."""
    if "-" not in ticker:
        return []
    base, quote = ticker.upper().split("-", 1)
    if quote not in _KRAKEN_QUOTES:
        return []
    return [b + quote for b in _KRAKEN_BASE.get(base, [base])]


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate to a coarser timeframe. Weekly ('W-MON') bars start on Monday 00:00 UTC."""
    g = df.resample(rule, label="left", closed="left")
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                        "Close": g["Close"].last(), "Volume": g["Volume"].sum()})
    out.index.name = "Date"
    return out.dropna(subset=["Close"])


# ------------------------------------------------------------------ Kraken
def parse_kraken_ohlc(payload: dict) -> pd.DataFrame:
    if payload.get("error"):
        raise RuntimeError("; ".join(payload["error"]))
    result = payload["result"]
    key = next(k for k in result if k != "last")
    df = pd.DataFrame(result[key], columns=["time", "Open", "High", "Low", "Close", "vwap", "Volume", "count"])
    idx = pd.DatetimeIndex(pd.to_datetime(df["time"].astype("int64"), unit="s"), name="Date")
    out = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    out.index = idx
    return out


def fetch_kraken_ohlc(pair: str, interval: int, n_bars: int) -> pd.DataFrame:
    """Last `n_bars` candles (the final one is the live, still-forming candle). Pages if > 720 are needed."""
    since = int(time.time()) - (n_bars + 2) * interval * 60
    frames = []
    for _ in range(8):
        d = parse_kraken_ohlc(_http_get_json(f"{KRAKEN}/OHLC", {"pair": pair, "interval": interval, "since": since}))
        if d.empty:
            break
        frames.append(d)
        if len(d) < 700:
            break
        since = int(d.index[-1].value // 10 ** 9)
    if not frames:
        raise RuntimeError("Kraken returned no candles")
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index().tail(n_bars)


def parse_kraken_ticker(payload: dict) -> dict:
    if payload.get("error"):
        raise RuntimeError("; ".join(payload["error"]))
    t = next(iter(payload["result"].values()))
    return {"price": float(t["c"][0]), "open": float(t["o"]), "high": float(t["h"][1]), "low": float(t["l"][1]),
            "volume": float(t["v"][1]), "bid": float(t["b"][0]), "ask": float(t["a"][0]),
            "open_label": "today's open (UTC)", "window": "24h"}


# ------------------------------------------------------------------ Yahoo fallback
def fetch_yahoo_candles(ticker: str, tf: str, period: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    interval, default_period, rule = _YAHOO[tf]
    period = period or default_period
    df = yf.download(ticker, interval=interval, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df[df["Close"].notna()]
    df["Volume"] = df["Volume"].fillna(0)
    idx = pd.to_datetime(df.index)
    df.index = (idx.tz_convert("UTC") if idx.tz is not None else idx).tz_localize(None)
    df.index.name = "Date"
    return resample_ohlc(df, rule) if rule else df


# ------------------------------------------------------------------ public API
def get_candles(ticker: str, tf: str, n_bars: int) -> tuple[pd.DataFrame, str]:
    """Return (candles oldest->newest incl. the forming candle, source label)."""
    minutes = TIMEFRAMES[tf]
    errors = []
    for pair in kraken_pairs(ticker):
        try:
            if tf == "1W":
                df = resample_ohlc(fetch_kraken_ohlc(pair, 1440, n_bars * 7 + 7), "W-MON")
            else:
                df = fetch_kraken_ohlc(pair, minutes, n_bars)
            if len(df) >= 20:
                return df.tail(n_bars), f"Kraken {pair} (live)"
        except Exception as exc:
            errors.append(f"Kraken {pair}: {exc}")
    try:
        df = fetch_yahoo_candles(ticker, tf)
        if len(df) >= 20:
            return df.tail(n_bars), "Yahoo Finance (near-live fallback)"
        errors.append("Yahoo returned too few candles")
    except Exception as exc:
        errors.append(f"Yahoo: {exc}")
    raise RuntimeError("No live data available. " + " | ".join(errors))


def quote_from_candles(df: pd.DataFrame, tf: str) -> dict:
    last = df.index[-1]
    window = df[df.index > last - pd.Timedelta(hours=24)]
    day = df[df.index.normalize() == last.normalize()]
    if TIMEFRAMES[tf] < 1440 and len(day):
        open_, label = float(day["Open"].iloc[0]), "today's open (UTC)"
    else:
        open_, label = float(df["Open"].iloc[-1]), "this candle's open"
    return {"price": float(df["Close"].iloc[-1]), "open": open_, "high": float(window["High"].max()),
            "low": float(window["Low"].min()), "volume": float(window["Volume"].sum()), "bid": np.nan, "ask": np.nan,
            "open_label": label, "window": "window"}


def get_quote(ticker: str, tf: str, df: pd.DataFrame) -> tuple[dict, str]:
    """Latest trade price and day stats. Uses Kraken's ticker; falls back to the candle data."""
    for pair in kraken_pairs(ticker):
        try:
            return parse_kraken_ticker(_http_get_json(f"{KRAKEN}/Ticker", {"pair": pair})), "Kraken"
        except Exception:
            continue
    return quote_from_candles(df, tf), "candles"


def candle_seconds_left(last_open: pd.Timestamp, tf: str, now: pd.Timestamp | None = None) -> int:
    now = now or pd.Timestamp.now(tz="UTC").tz_localize(None)
    return max(0, int((last_open + pd.Timedelta(minutes=TIMEFRAMES[tf]) - now).total_seconds()))


_HISTORY = {"5m": "58d", "15m": "58d", "30m": "58d", "1h": "700d", "4h": "700d", "1D": "5y", "1W": "10y"}


def fetch_history(ticker: str, tf: str) -> tuple[pd.DataFrame, str]:
    """As much history as Yahoo allows (for backtests). Kraken only serves its latest 720 candles."""
    try:
        df = fetch_yahoo_candles(ticker, tf, period=_HISTORY[tf])
        if len(df) >= 300:
            return df, f"Yahoo Finance ({len(df)} candles)"
    except Exception:
        pass
    df, src = get_candles(ticker, tf, 720)
    return df, f"{src} ({len(df)} candles, short history)"
