"""Confluence indicator: TOTT trend + Support/Resistance retest = ONE signal with a stop and a target.

LONG  : TOTT trend is bullish AND price dips into an active SUPPORT zone and closes back above its floor (the pivot low).
SHORT : TOTT trend is bearish AND price pokes into an active RESISTANCE zone and closes back below its ceiling (the pivot high).
Stop  : just beyond the zone (pivot +/- stop_atr x ATR).   Target: rr x the risk.
Confidence 4..6: base 4 (trend + retest), +1 if TOTT turned within the last `fresh_bars` candles,
                 +1 if the retest candle itself rejected the zone (closed in the trade direction).
Each zone fires at most once. Strictly causal: the signal on candle i uses only candles <= i (tests/confluence_test.py proves it).
Support/Resistance rules: see src/sr.py (after the TradingView script by Trend-My-Friend, MPL 2.0).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import indicators as I
from .sr import sr_zones


@dataclass(frozen=True)
class ConfParams:
    tott_length: int = 40
    tott_percent: float = 1.0
    tott_coeff: float = 0.001
    sr_left: int = 20
    sr_right: int = 5
    sr_max: int = 10
    stop_atr: float = 0.5
    rr: float = 2.0
    fresh_bars: int = 20
    min_conf: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


COLUMNS = ["signal", "conf", "entry", "stop", "target", "zone_top", "zone_bottom", "zone_kind", "fresh", "rejection",
           "trend", "tott_buy", "tott_sell", "wz_top", "wz_bottom", "wz_kind"]


def confluence(df: pd.DataFrame, p: ConfParams | None = None) -> pd.DataFrame:
    p = p or ConfParams()
    n = len(df)
    tt = I.tott(df, p.tott_length, p.tott_percent, p.tott_coeff)
    atr = np.nan_to_num(I.atr(df).values)
    by_conf: dict[int, list[dict]] = {}
    for z in sr_zones(df, p.sr_left, p.sr_right, p.sr_max):
        by_conf.setdefault(z["conf"], []).append({**z, "used": False})

    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    trend, buy, sell = tt["trend"].values, tt["buy"].values, tt["sell"].values
    num = {k: np.full(n, np.nan) for k in ("entry", "stop", "target", "zone_top", "zone_bottom", "wz_top", "wz_bottom")}
    sig, conf = np.zeros(n, int), np.zeros(n, int)
    zkind, wkind = np.array([""] * n, dtype=object), np.array([""] * n, dtype=object)
    fresh_arr, rej_arr = np.zeros(n, bool), np.zeros(n, bool)
    active = {"bull": [], "bear": []}                                  # bull trades support zones, bear trades resistance zones
    last_cross = -10 ** 9

    for i in range(n):
        if buy[i] or sell[i]:
            last_cross = i
        side = "bull" if trend[i] == 1 else "bear" if trend[i] == -1 else None
        if side:
            pool = [z for z in active[side] if not z["used"] and (z["i_end"] is None or z["i_end"] > i)]
            if side == "bull":
                hits = [z for z in pool if l[i] <= z["top"] and c[i] > z["bottom"]]
            else:
                hits = [z for z in pool if h[i] >= z["bottom"] and c[i] < z["top"]]
            if hits:
                z = min(hits, key=lambda q: abs(c[i] - (q["top"] + q["bottom"]) / 2))
                if side == "bull":
                    stop = min(z["bottom"], l[i]) - p.stop_atr * atr[i]
                    ok, target, rejected = stop < c[i], c[i] + p.rr * (c[i] - stop), c[i] > o[i]
                else:
                    stop = max(z["top"], h[i]) + p.stop_atr * atr[i]
                    ok, target, rejected = stop > c[i], c[i] - p.rr * (stop - c[i]), c[i] < o[i]
                fresh = (i - last_cross) <= p.fresh_bars
                score = 4 + int(fresh) + int(rejected)
                if ok and score >= p.min_conf:
                    sig[i], conf[i] = (1 if side == "bull" else -1), score
                    num["entry"][i], num["stop"][i], num["target"][i] = c[i], stop, target
                    num["zone_top"][i], num["zone_bottom"][i] = z["top"], z["bottom"]
                    zkind[i], fresh_arr[i], rej_arr[i] = z["kind"], fresh, rejected
                    z["used"] = True
            waiting = [z for z in active[side] if not z["used"] and (z["i_end"] is None or z["i_end"] > i)]
            if waiting:                                                # what the indicator is waiting for right now
                gap = lambda q: 0.0 if q["bottom"] <= c[i] <= q["top"] else min(abs(c[i] - q["top"]), abs(c[i] - q["bottom"]))
                w = min(waiting, key=gap)
                num["wz_top"][i], num["wz_bottom"][i], wkind[i] = w["top"], w["bottom"], w["kind"]
        for z in by_conf.get(i, []):                                   # zones confirmed at this close become tradable next candle
            active["bull" if z["kind"] == "support" else "bear"].append(z)

    res = pd.DataFrame({"signal": sig, "conf": conf, **{k: num[k] for k in ("entry", "stop", "target", "zone_top", "zone_bottom")},
                        "zone_kind": zkind, "fresh": fresh_arr, "rejection": rej_arr, "trend": trend, "tott_buy": buy,
                        "tott_sell": sell, "wz_top": num["wz_top"], "wz_bottom": num["wz_bottom"], "wz_kind": wkind}, index=df.index)
    return res[COLUMNS]


def waiting_text(res: pd.DataFrame, price: float, allow_short: bool = True) -> str:
    """One plain-English line: what the indicator is waiting for on the newest closed candle."""
    last = res.iloc[-1]
    t = int(last["trend"])
    if t == 0:
        return "Waiting for TOTT to choose a trend (no TOTT signal yet)."
    if t == -1 and not allow_short:
        return "TOTT trend is bearish and this bot is long-only, so it is waiting (switch on shorts to trade the downtrend)."
    word, kind = ("bullish", "support") if t == 1 else ("bearish", "resistance")
    if np.isnan(last["wz_top"]):
        return f"TOTT trend is {word}, but there is no active {kind} zone to trade yet."
    mid = (last["wz_top"] + last["wz_bottom"]) / 2
    dist = abs(price - mid) / price * 100
    where = ("price is inside the zone" if last["wz_bottom"] <= price <= last["wz_top"]
             else f"the zone is {dist:.1f}% {'below' if price > mid else 'above'} the current price")
    return (f"TOTT trend is {word}. Waiting for a retest of {kind} {last['wz_bottom']:,.2f}-{last['wz_top']:,.2f} "
            f"({where}).")


def status_rows(df: pd.DataFrame, res: pd.DataFrame, p: ConfParams | None = None) -> pd.DataFrame:
    """One tracker row describing the confluence state on the last *closed* candle."""
    last = res.iloc[-1]
    trend = {1: "🟢 bullish trend", -1: "🔴 bearish trend", 0: "⚪ no trend yet"}[int(last["trend"])]
    sigs = res[res["signal"] != 0]
    if len(sigs):
        s = sigs.iloc[-1]
        bars = len(res) - 1 - res.index.get_loc(sigs.index[-1])
        txt = (f"last signal {'LONG' if s['signal'] == 1 else 'SHORT'} {bars} candle(s) ago, confidence {int(s['conf'])}, "
               f"stop {s['stop']:,.2f}, target {s['target']:,.2f}")
    else:
        txt = "no signal in view yet"
    return pd.DataFrame([{"Indicator": "Confluence (TOTT + S/R)", "Value": trend, "Reading": txt}])
