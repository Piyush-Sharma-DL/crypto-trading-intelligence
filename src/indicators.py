"""Chart indicators for the live view: TOTT, IMBA (imbalance / fair value gaps), order blocks,
plus SMA, EMA, RSI, MACD, Bollinger, VWAP, ATR and Supertrend. Pure pandas / numpy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import rsi as _rsi

DEFAULTS = dict(tott_length=40, tott_percent=1.0, tott_coeff=0.001,
                fvg_min_atr=0.25, ob_swing=10, ob_disp=1.5, ob_back=10, ob_body=True, max_zones=3)


# ------------------------------------------------------------------ classic indicators
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    c = df["Close"]
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c.shift()).abs(), (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid, sd = close.rolling(n).mean(), close.rolling(n).std(ddof=0)
    up, lo = mid + k * sd, mid - k * sd
    return pd.DataFrame({"mid": mid, "upper": up, "lower": lo, "pctb": (close - lo) / (up - lo)})


def vwap(df: pd.DataFrame, tf_minutes: int) -> pd.Series:
    """Intraday: session VWAP that resets at 00:00 UTC. Daily and weekly: rolling 20-bar VWAP."""
    pv = (df["High"] + df["Low"] + df["Close"]) / 3 * df["Volume"]
    vol = df["Volume"]
    if tf_minutes < 1440:
        day = df.index.normalize()
        return pv.groupby(day).cumsum() / vol.groupby(day).cumsum()
    return pv.rolling(20, min_periods=1).sum() / vol.rolling(20, min_periods=1).sum()


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    a = atr(df, n).values
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fub, flb, d = ub.copy(), lb.copy(), np.ones(len(c), dtype=int)
    for i in range(1, len(c)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if d[i - 1] == -1 and c[i] > fub[i - 1]:
            d[i] = 1
        elif d[i - 1] == 1 and c[i] < flb[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
    return pd.DataFrame({"line": np.where(d == 1, flb, fub), "dir": d}, index=df.index)


# ------------------------------------------------------------------ TOTT (Twin Optimized Trend Tracker)
def var_ma(src, length: int) -> np.ndarray:
    """VAR / VIDYA moving average (Chande) as used by Anil Ozeksi's OTT. Seeded with the first price."""
    src = np.asarray(src, dtype=float)
    n = len(src)
    d = np.diff(src, prepend=src[0])
    s_up = pd.Series(np.maximum(d, 0)).rolling(9, min_periods=1).sum().values
    s_dn = pd.Series(np.maximum(-d, 0)).rolling(9, min_periods=1).sum().values
    den = s_up + s_dn
    cmo = np.divide(s_up - s_dn, den, out=np.zeros(n), where=den > 0)
    alpha = 2.0 / (length + 1)
    out = np.empty(n)
    out[0] = src[0]
    for i in range(1, n):
        k = alpha * abs(cmo[i])
        out[i] = k * src[i] + (1 - k) * out[i - 1]
    return out


def tott(df: pd.DataFrame, length: int = 40, percent: float = 1.0, coeff: float = 0.001) -> pd.DataFrame:
    """Twin OTT. Support line = VAR moving average. OTT = trailing stop of the support line.
    Twin lines: OTTup = OTT*(1+coeff), OTTdn = OTT*(1-coeff), both plotted 2 bars late (as in the original).
    BUY when the support line crosses above OTTup, SELL when it crosses below OTTdn.
    Signals alternate (a BUY can only follow a SELL, and vice versa)."""
    ma = var_ma(df["Close"].values, length)
    n = len(ma)
    fark = ma * percent * 0.01
    long_raw, short_raw = ma - fark, ma + fark
    ls, ss, mt = np.empty(n), np.empty(n), np.empty(n)
    d = 1
    for i in range(n):
        ls_prev, ss_prev = (ls[i - 1], ss[i - 1]) if i else (long_raw[0], short_raw[0])
        l, s = long_raw[i], short_raw[i]
        if ma[i] > ls_prev:
            l = max(l, ls_prev)
        if ma[i] < ss_prev:
            s = min(s, ss_prev)
        ls[i], ss[i] = l, s
        if d == -1 and ma[i] > ss_prev:
            d = 1
        elif d == 1 and ma[i] < ls_prev:
            d = -1
        mt[i] = l if d == 1 else s
    ott = np.where(ma > mt, mt * (200 + percent) / 200, mt * (200 - percent) / 200)
    up = pd.Series(ott * (1 + coeff)).shift(2).values
    dn = pd.Series(ott * (1 - coeff)).shift(2).values

    prev_ma = np.r_[np.nan, ma[:-1]]
    prev_up, prev_dn = np.r_[np.nan, up[:-1]], np.r_[np.nan, dn[:-1]]
    with np.errstate(invalid="ignore"):
        raw_buy = (ma > up) & (prev_ma <= prev_up)
        raw_sell = (ma < dn) & (prev_ma >= prev_dn)
    buy, sell, trend = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, int)
    state = 0
    for i in range(n):
        if raw_buy[i] and state != 1:
            buy[i], state = True, 1
        elif raw_sell[i] and state != -1:
            sell[i], state = True, -1
        trend[i] = state
    return pd.DataFrame({"support": ma, "ott_up": up, "ott_dn": dn, "buy": buy, "sell": sell, "trend": trend},
                        index=df.index)


# ------------------------------------------------------------------ IMBA: imbalance / fair value gaps
def fvg_zones(df: pd.DataFrame, min_atr: float = 0.25) -> list[dict]:
    """Three-candle imbalance. Bullish: low[i] > high[i-2]. Bearish: high[i] < low[i-2].
    Gaps smaller than min_atr x ATR are ignored. A zone is mitigated once price trades fully through it."""
    h, l = df["High"].values, df["Low"].values
    a = np.nan_to_num(atr(df).values)
    idx, zones = df.index, []
    for i in range(2, len(df)):
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= min_atr * a[i - 1]:
            kind, bottom, top = "bull_fvg", h[i - 2], l[i]
            hit = np.where(l[i + 1:] <= bottom)[0]
        elif h[i] < l[i - 2] and (l[i - 2] - h[i]) >= min_atr * a[i - 1]:
            kind, bottom, top = "bear_fvg", h[i], l[i - 2]
            hit = np.where(h[i + 1:] >= top)[0]
        else:
            continue
        zones.append(dict(kind=kind, i=i - 1, conf=i, t0=idx[i - 1], top=float(top), bottom=float(bottom),
                          mitigated=bool(len(hit)), t_mit=idx[i + 1 + hit[0]] if len(hit) else None,
                          i_mit=int(i + 1 + hit[0]) if len(hit) else None))
    return zones


# ------------------------------------------------------------------ order blocks
def order_blocks(df: pd.DataFrame, swing: int = 10, disp: float = 1.5, back: int = 10,
                 body_only: bool = True) -> list[dict]:
    """Bullish OB: the last down-close candle before a strong up-move that breaks the recent swing high.
    Bearish OB: the last up-close candle before a strong down-move that breaks the recent swing low.
    'Strong' = candle body larger than `disp` x ATR. The zone is the candle body (or the full range with wicks if
    body_only=False). Mitigated when a later candle closes through the zone."""
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    a = np.nan_to_num(atr(df).values)
    idx, zones, used = df.index, [], set()
    for i in range(swing, len(df)):
        body = c[i] - o[i]
        if body > disp * a[i - 1] and c[i] > h[i - swing:i].max():
            kind, cand = "bull_ob", [j for j in range(i - 1, max(i - back - 1, -1), -1) if c[j] < o[j]]
        elif -body > disp * a[i - 1] and c[i] < l[i - swing:i].min():
            kind, cand = "bear_ob", [j for j in range(i - 1, max(i - back - 1, -1), -1) if c[j] > o[j]]
        else:
            continue
        if not cand or (kind, cand[0]) in used:
            continue
        j = cand[0]
        used.add((kind, j))
        top, bottom = (float(max(o[j], c[j])), float(min(o[j], c[j]))) if body_only else (float(h[j]), float(l[j]))
        hit = np.where(c[i + 1:] < bottom)[0] if kind == "bull_ob" else np.where(c[i + 1:] > top)[0]
        zones.append(dict(kind=kind, i=j, conf=i, t0=idx[j], top=top, bottom=bottom,
                          mitigated=bool(len(hit)), t_mit=idx[i + 1 + hit[0]] if len(hit) else None,
                          i_mit=int(i + 1 + hit[0]) if len(hit) else None))
    return zones


def keep_recent(zones: list[dict], max_per_kind: int, show_mitigated: bool = False) -> list[dict]:
    out = []
    for kind in ("bull_fvg", "bear_fvg", "bull_ob", "bear_ob"):
        z = [x for x in zones if x["kind"] == kind and (show_mitigated or not x["mitigated"])]
        out += z[-max_per_kind:]
    return out


# ------------------------------------------------------------------ everything at once
def compute_all(df: pd.DataFrame, tf_minutes: int, cfg: dict | None = None) -> dict:
    cfg = {**DEFAULTS, **(cfg or {})}
    c = df["Close"]
    return dict(
        sma20=sma(c, 20), sma50=sma(c, 50), ema21=ema(c, 21), ema200=ema(c, 200),
        rsi=_rsi(c, 14), macd=macd(c), bb=bollinger(c), vwap=vwap(df, tf_minutes), atr=atr(df),
        st=supertrend(df),
        tott=tott(df, cfg["tott_length"], cfg["tott_percent"], cfg["tott_coeff"]),
        fvg=fvg_zones(df, cfg["fvg_min_atr"]),
        ob=order_blocks(df, cfg["ob_swing"], cfg["ob_disp"], cfg["ob_back"], cfg["ob_body"]),
    )


ZONE_LABEL = {"bull_fvg": "Bullish IMBA", "bear_fvg": "Bearish IMBA", "bull_ob": "Bullish OB", "bear_ob": "Bearish OB"}


def snapshot(df: pd.DataFrame, ind: dict) -> tuple[pd.DataFrame, dict]:
    """Live readout of every indicator on the forming candle. bias: +1 bullish, -1 bearish, 0 neutral."""
    price = float(df["Close"].iloc[-1])
    rows = []

    def add(name, value, reading, bias):
        rows.append({"Indicator": name, "Value": value, "Reading": reading, "bias": bias})

    for key, label in (("sma20", "SMA 20"), ("sma50", "SMA 50"), ("ema21", "EMA 21"), ("ema200", "EMA 200")):
        v = ind[key].iloc[-1]
        if np.isnan(v):
            add(label, "n/a", "not enough candles", 0)
        else:
            add(label, f"{v:,.2f}", "price above" if price > v else "price below", 1 if price > v else -1)
    r = float(ind["rsi"].iloc[-1])
    add("RSI (14)", f"{r:.1f}", "overbought" if r > 70 else "oversold" if r < 30 else "bullish momentum" if r > 55
        else "bearish momentum" if r < 45 else "neutral", 0 if (r > 70 or r < 30) else 1 if r > 55 else -1 if r < 45 else 0)
    m = ind["macd"].iloc[-1]
    add("MACD (12,26,9)", f"{m['macd']:+.2f} / {m['signal']:+.2f}", "histogram up" if m["hist"] > 0 else "histogram down",
        1 if m["hist"] > 0 else -1)
    b = ind["bb"].iloc[-1]
    if not np.isnan(b["pctb"]):
        add("Bollinger %B", f"{b['pctb']:.2f}", "above upper band" if b["pctb"] > 1 else "below lower band" if b["pctb"] < 0 else "inside bands", 0)
    vw = ind["vwap"].iloc[-1]
    if not np.isnan(vw):
        add("VWAP", f"{vw:,.2f}", "price above" if price > vw else "price below", 1 if price > vw else -1)
    add("ATR %", f"{ind['atr'].iloc[-1] / price * 100:.2f}%", "volatility", 0)
    st_dir = int(ind["st"]["dir"].iloc[-1])
    add("Supertrend", f"{ind['st']['line'].iloc[-1]:,.2f}", "uptrend" if st_dir == 1 else "downtrend", st_dir)
    t = ind["tott"]
    last = t.iloc[-1]
    sig = t[t["buy"] | t["sell"]]
    when = f", last signal {'BUY' if sig.iloc[-1]['buy'] else 'SELL'} {sig.index[-1]:%d %b %H:%M} UTC" if len(sig) else ""
    state = "bullish (support above OTT up)" if last["support"] > last["ott_up"] else \
        "bearish (support below OTT down)" if last["support"] < last["ott_dn"] else "flat zone (between twin lines)"
    add("TOTT", f"{last['support']:,.2f}", state + when, 1 if last["trend"] == 1 else -1 if last["trend"] == -1 else 0)
    df_rows = pd.DataFrame(rows)
    icon = {1: "🟢", -1: "🔴", 0: "⚪"}
    df_rows["Reading"] = [f"{icon[b]} {r}" for b, r in zip(df_rows["bias"], df_rows["Reading"])]
    summary = {"bull": int((df_rows["bias"] == 1).sum()), "bear": int((df_rows["bias"] == -1).sum()),
               "neutral": int((df_rows["bias"] == 0).sum())}
    return df_rows.drop(columns="bias"), summary


def active_zones(zones: list[dict], price: float, n_bars: int) -> pd.DataFrame:
    """Unmitigated imbalance / order-block zones nearest to the current price."""
    rows = []
    for z in zones:
        if z["mitigated"]:
            continue
        dist = 0.0 if z["bottom"] <= price <= z["top"] else (z["bottom"] - price if price < z["bottom"] else price - z["top"])
        rows.append({"Zone": ZONE_LABEL[z["kind"]], "From": f"{z['bottom']:,.2f}", "To": f"{z['top']:,.2f}",
                     "Distance %": dist / price * 100, "Age (candles)": n_bars - 1 - z["i"]})
    out = pd.DataFrame(rows, columns=["Zone", "From", "To", "Distance %", "Age (candles)"])
    return out.sort_values("Distance %").reset_index(drop=True)
