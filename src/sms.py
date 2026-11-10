"""Irfan Smart Market Structure v7.1: independent implementation of the rules of the TradingView indicator.

Layers: EMA 9/21 cloud + crossovers, MA 50/200 (golden / death cross), Supertrend, Bollinger Bands, trend-aware Fibonacci,
liquidity levels + sweeps, trendlines, displacement order blocks, last-signal tracker and a confluence dashboard.
`sms_lines` are pure indicator series (fine to compute on the forming candle). `sms_events` (crosses, sweeps, trendlines, order blocks)
must be computed on CLOSED candles only, so nothing repaints.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as I


@dataclass(frozen=True)
class SMSParams:
    ema_fast: int = 9
    ema_slow: int = 21
    ma_mid: int = 50
    ma_long: int = 200
    st_atr: int = 10
    st_factor: float = 3.0
    bb_len: int = 20
    bb_mult: float = 2.0
    fib_lookback: int = 100
    use_trend_fib: bool = True
    liq_pivot: int = 5
    liq_vol_len: int = 20
    liq_vol_mult: float = 1.5
    tl_pivot: int = 5
    max_tl: int = 6
    ob_atr_mult: float = 1.0
    ob_lookback: int = 5
    ob_extend: int = 20
    max_ob: int = 10


FIB_LEVELS = (("fib0", None), ("fib236", 0.236), ("fib382", 0.382), ("fib500", 0.5), ("fib618", 0.618), ("fib786", 0.786), ("fib1000", None))


def sms_lines(df: pd.DataFrame, p: SMSParams | None = None) -> pd.DataFrame:
    p = p or SMSParams()
    c = df["Close"]
    ema_f, ema_s = I.ema(c, p.ema_fast), I.ema(c, p.ema_slow)
    ma_mid, ma_long = I.ema(c, p.ma_mid), I.ema(c, p.ma_long)
    st = I.supertrend(df, p.st_atr, p.st_factor)
    bb = I.bollinger(c, p.bb_len, p.bb_mult)
    hi, lo = df["High"].rolling(p.fib_lookback).max(), df["Low"].rolling(p.fib_lookback).min()
    ema_bull = ema_f >= ema_s
    up, dn = ema_bull & (c > ma_mid), (~ema_bull) & (c < ma_mid)
    lo_n, hi_n = np.minimum(hi, lo), np.maximum(hi, lo)
    if p.use_trend_fib:
        start = pd.Series(np.where(up, lo, np.where(dn, hi, lo_n)), index=df.index)
        end = pd.Series(np.where(up, hi, np.where(dn, lo, hi_n)), index=df.index)
    else:
        start, end = lo_n, hi_n
    rng = end - start
    out = pd.DataFrame({"ema_fast": ema_f, "ema_slow": ema_s, "ema_bull": ema_bull, "ma_mid": ma_mid, "ma_long": ma_long,
                        "st_line": st["line"], "st_bull": st["dir"] == 1, "bb_up": bb["upper"], "bb_mid": bb["mid"], "bb_low": bb["lower"]},
                       index=df.index)
    out["fib0"], out["fib1000"] = start, end
    for name, k in FIB_LEVELS:
        if k is not None:
            out[name] = start + rng * k
    return out


def _cross_up(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        prev_a, prev_b = np.r_[np.nan, a[:-1]], np.r_[np.nan, b[:-1]]
        return (a > b) & (prev_a <= prev_b)


def _cross_down(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        prev_a, prev_b = np.r_[np.nan, a[:-1]], np.r_[np.nan, b[:-1]]
        return (a < b) & (prev_a >= prev_b)


def _pivots(h: np.ndarray, l: np.ndarray, length: int):
    """Confirmed pivots: a pivot candle p is reported at bar p+length. High: >= previous `length` highs and > next `length` highs."""
    n = len(h)
    ph, pl = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        p = i - length
        if p >= length:
            if h[p] >= h[p - length:p].max() and h[p] > h[p + 1:i + 1].max():
                ph[i] = h[p]
            if l[p] <= l[p - length:p].min() and l[p] < l[p + 1:i + 1].min():
                pl[i] = l[p]
    return ph, pl


def sms_events(df: pd.DataFrame, p: SMSParams | None = None) -> dict:
    """Everything that happens on candle CLOSES. Strictly causal (the value at candle i uses only candles <= i), except the
    final trendline / order-block lists, which describe the state as of the newest candle."""
    p = p or SMSParams()
    n = len(df)
    lines = sms_lines(df, p)
    o, h, l, c, v = (df[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
    idx = df.index

    ev = pd.DataFrame(index=idx)
    ev["ema_up"] = _cross_up(lines["ema_fast"].values, lines["ema_slow"].values)
    ev["ema_dn"] = _cross_down(lines["ema_fast"].values, lines["ema_slow"].values)
    ev["golden"] = _cross_up(lines["ma_mid"].values, lines["ma_long"].values)
    ev["death"] = _cross_down(lines["ma_mid"].values, lines["ma_long"].values)

    # ---- liquidity: the latest confirmed swing high / low, and sweeps through them
    ph, pl = _pivots(h, l, p.liq_pivot)
    lsh, lsl = np.full(n, np.nan), np.full(n, np.nan)
    lsh_bar, lsl_bar = np.full(n, -1), np.full(n, -1)
    cur_h = cur_l = np.nan
    bar_h = bar_l = -1
    for i in range(n):
        if not np.isnan(ph[i]):
            cur_h, bar_h = ph[i], i - p.liq_pivot
        if not np.isnan(pl[i]):
            cur_l, bar_l = pl[i], i - p.liq_pivot
        lsh[i], lsl[i], lsh_bar[i], lsl_bar[i] = cur_h, cur_l, bar_h, bar_l
    with np.errstate(invalid="ignore"):
        ev["bsl"] = (h > lsh) & (c < lsh)            # buy-side liquidity sweep: wick above the swing high, close back below
        ev["ssl"] = (l < lsl) & (c > lsl)            # sell-side liquidity sweep
    avg_vol = pd.Series(v, index=idx).rolling(p.liq_vol_len).mean().values
    with np.errstate(invalid="ignore", divide="ignore"):
        ev["vol_ratio"] = np.where(avg_vol > 0, v / avg_vol, np.nan)
        ev["vol_spike"] = v > avg_vol * p.liq_vol_mult
    ev["swing_high"], ev["swing_low"] = lsh, lsl

    # ---- last signal (later events in the same candle override earlier ones, as in the original)
    label, bar = np.array([""] * n, dtype=object), np.full(n, -1)
    cur_label, cur_bar = "None", -1
    order = (("ema_up", "EMA UP"), ("ema_dn", "EMA DN"), ("golden", "GOLDEN"), ("death", "DEATH"), ("bsl", "BSL SWEEP"), ("ssl", "SSL SWEEP"))
    flags = {k: ev[k].values for k, _ in order}
    for i in range(n):
        for k, name in order:
            if flags[k][i]:
                cur_label, cur_bar = name, i
        label[i], bar[i] = cur_label, cur_bar
    ev["last_signal"], ev["last_signal_bar"] = label, bar

    # ---- trendlines through the last two pivot highs / lows (a new line at every new pivot, newest `max_tl` kept)
    tph, tpl = _pivots(h, l, p.tl_pivot)
    ph1 = ph2 = pl1 = pl2 = np.nan
    b_ph1 = b_ph2 = b_pl1 = b_pl2 = -1
    res_lines, sup_lines = [], []
    for i in range(n):
        if not np.isnan(tph[i]):
            ph2, b_ph2, ph1, b_ph1 = ph1, b_ph1, tph[i], i - p.tl_pivot
            if not np.isnan(ph2):
                slope = (ph1 - ph2) / max(b_ph1 - b_ph2, 1)
                res_lines.append({"kind": "resistance", "x1": b_ph2, "y1": ph2, "x2": b_ph1, "y2": ph1, "slope": slope, "conf": i,
                                  "strong": bool(abs(slope) / c[i] * 100 > 0.05)})
        if not np.isnan(tpl[i]):
            pl2, b_pl2, pl1, b_pl1 = pl1, b_pl1, tpl[i], i - p.tl_pivot
            if not np.isnan(pl2):
                slope = (pl1 - pl2) / max(b_pl1 - b_pl2, 1)
                sup_lines.append({"kind": "support", "x1": b_pl2, "y1": pl2, "x2": b_pl1, "y2": pl1, "slope": slope, "conf": i,
                                  "strong": bool(abs(slope) / c[i] * 100 > 0.05)})
    trendlines = res_lines[-p.max_tl:] + sup_lines[-p.max_tl:]

    # ---- displacement order blocks
    atr = I.atr(df).values
    active = {"bull": [], "bear": []}
    obs: list[dict] = []
    for i in range(1, n):
        bull = c[i] > h[i - 1] and c[i] > o[i] and (c[i] - o[i]) > atr[i] * p.ob_atr_mult
        bear = c[i] < l[i - 1] and c[i] < o[i] and (o[i] - c[i]) > atr[i] * p.ob_atr_mult
        for kind, fired, wanted in (("bull", bull, lambda j: c[j] < o[j]), ("bear", bear, lambda j: c[j] > o[j])):
            if fired:
                for j in range(1, p.ob_lookback + 1):
                    if i - j >= 0 and wanted(i - j):
                        z = {"kind": kind, "i": i - j, "conf": i, "t0": idx[i - j], "top": float(h[i - j]), "bottom": float(l[i - j]),
                             "i_right": i + p.ob_extend, "i_end": None, "touched": False, "dropped": False}
                        obs.append(z)
                        active[kind].append(z)
                        break
        for kind in ("bull", "bear"):
            while len(active[kind]) > p.max_ob:                       # newest `max_ob` kept
                old = active[kind].pop(0)
                old["dropped"], old["i_end"] = True, i
        for z in list(active["bull"]):
            if c[i] < z["bottom"]:
                z["i_end"] = i
                active["bull"].remove(z)
            elif l[i] <= z["bottom"]:
                z["touched"] = True
        for z in list(active["bear"]):
            if c[i] > z["top"]:
                z["i_end"] = i
                active["bear"].remove(z)
            elif h[i] >= z["top"]:
                z["touched"] = True
    return {"events": ev, "trendlines": trendlines, "order_blocks": obs, "lines": lines,
            "swing_high_bar": int(lsh_bar[-1]) if n else -1, "swing_low_bar": int(lsl_bar[-1]) if n else -1}


def bias_text(score: int) -> str:
    return ("STRONG BULLISH" if score >= 4 else "BULLISH" if score >= 2 else "STRONG BEARISH" if score <= -4
            else "BEARISH" if score <= -2 else "NEUTRAL")


def sms_dashboard(df: pd.DataFrame, lines: pd.DataFrame, ev: pd.DataFrame, price: float, p: SMSParams | None = None) -> tuple[pd.DataFrame, int, str]:
    """The indicator's dashboard. `lines` may include the forming candle; `ev` comes from closed candles."""
    p = p or SMSParams()
    L, last_open = lines.iloc[-1], float(df["Open"].iloc[-1])
    e = ev.iloc[-1] if len(ev) else None
    atr = float(I.atr(df).iloc[-1])
    bsl, ssl = (bool(e["bsl"]), bool(e["ssl"])) if e is not None else (False, False)
    spike = bool(e["vol_spike"]) if e is not None else False
    score = (1 if L["ema_bull"] else -1) + (1 if L["st_bull"] else -1) + (1 if L["ma_mid"] > L["ma_long"] else -1) \
        + (1 if price > L["bb_mid"] else -1) + ((1 if price > last_open else -1) if spike else 0) + (-1 if bsl else 0) + (1 if ssl else 0) \
        + (1 if abs(price - L["fib618"]) <= atr * 0.5 else 0)
    bb = ("Above Upper" if price > L["bb_up"] else "Below Lower" if price < L["bb_low"] else
          "Upper Half" if price >= L["bb_mid"] else "Lower Half")
    fib = ("78.6-100%" if price >= L["fib786"] else "61.8-78.6%" if price >= L["fib618"] else "50-61.8%" if price >= L["fib500"]
           else "38.2-50%" if price >= L["fib382"] else "23.6-38.2%" if price >= L["fib236"] else "0-23.6%")
    ratio = float(e["vol_ratio"]) if e is not None and not np.isnan(e["vol_ratio"]) else np.nan
    last = ""
    if e is not None:
        last = e["last_signal"] if e["last_signal_bar"] < 0 else f"{e['last_signal']} ({len(ev) - 1 - int(e['last_signal_bar'])} candles ago)"
    rows = [("Confluence", f"{bias_text(score)} ({score:+d})", "score of EMA, Supertrend, MA, Bollinger, volume, sweeps and Fib"),
            ("EMA 9/21", "BULLISH" if L["ema_bull"] else "BEARISH", ""),
            ("Supertrend", f"{'UP' if L['st_bull'] else 'DOWN'} @ {L['st_line']:,.2f}", ""),
            ("MA 50/200", "BULL (50>200)" if L["ma_mid"] > L["ma_long"] else "BEAR (50<200)", ""),
            ("Bollinger", bb, ""), ("Fib zone", fib, ""),
            ("Volume", "n/a" if np.isnan(ratio) else f"{ratio:.2f}x avg" + (" (spike)" if spike else ""), ""),
            ("Last signal", last or "None", "")]
    return pd.DataFrame(rows, columns=["Item", "Value", "Note"]), score, bias_text(score)
