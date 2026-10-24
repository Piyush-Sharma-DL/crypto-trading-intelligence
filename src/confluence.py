"""Confluence indicator: TOTT trend + order-block / IMBA (fair value gap) retest = ONE signal with a stop and a target.

LONG  : TOTT trend is bullish AND price retests an unmitigated bullish zone (order block or imbalance) and holds above it.
SHORT : the mirror image.
Stop  : just beyond the zone (zone edge +/- stop_atr x ATR).  Target: rr x the risk.
Confidence 4..6: base 4 (trend + retest), +1 if an order block and an imbalance overlap ("stacked"),
                 +1 if TOTT turned within the last `fresh_bars` candles.

Strictly causal: the signal on candle i uses only candles <= i. (tests/confluence_test.py proves it.)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import indicators as I


@dataclass(frozen=True)
class ConfParams:
    tott_length: int = 40
    tott_percent: float = 1.0
    tott_coeff: float = 0.001
    fvg_min_atr: float = 0.25
    ob_swing: int = 10
    ob_disp: float = 1.5
    ob_back: int = 10
    ob_body: bool = True
    stop_atr: float = 0.5
    rr: float = 2.0
    fresh_bars: int = 20
    min_conf: int = 4
    max_active: int = 12

    def to_dict(self) -> dict:
        return asdict(self)


COLUMNS = ["signal", "conf", "entry", "stop", "target", "zone_top", "zone_bottom", "zone_kind", "stacked", "fresh",
           "trend", "tott_buy", "tott_sell"]


def _family(kind: str) -> str:
    return kind.split("_")[1]            # 'ob' or 'fvg'


def _overlap(a: dict, b: dict) -> bool:
    inter = min(a["top"], b["top"]) - max(a["bottom"], b["bottom"])
    return inter > 0 and inter >= 0.3 * min(a["top"] - a["bottom"], b["top"] - b["bottom"])


def confluence(df: pd.DataFrame, p: ConfParams | None = None) -> pd.DataFrame:
    p = p or ConfParams()
    n = len(df)
    tt = I.tott(df, p.tott_length, p.tott_percent, p.tott_coeff)
    atr = np.nan_to_num(I.atr(df).values)
    zones = I.fvg_zones(df, p.fvg_min_atr) + I.order_blocks(df, p.ob_swing, p.ob_disp, p.ob_back, p.ob_body)
    by_conf: dict[int, list[dict]] = {}
    for z in zones:
        by_conf.setdefault(z["conf"], []).append({**z, "used": False})

    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    trend, buy, sell = tt["trend"].values, tt["buy"].values, tt["sell"].values
    out = {k: np.full(n, np.nan) for k in ("entry", "stop", "target", "zone_top", "zone_bottom")}
    sig, conf = np.zeros(n, int), np.zeros(n, int)
    kind_arr = np.array([""] * n, dtype=object)
    stacked_arr, fresh_arr = np.zeros(n, bool), np.zeros(n, bool)
    active = {"bull": [], "bear": []}
    last_cross = -10 ** 9

    for i in range(n):
        if buy[i] or sell[i]:
            last_cross = i
        side = "bull" if trend[i] == 1 else "bear" if trend[i] == -1 else None
        if side:
            pool = [z for z in active[side] if not z["used"] and (z["i_mit"] is None or z["i_mit"] > i)]
            if side == "bull":
                hits = [z for z in pool if l[i] <= z["top"] and c[i] > z["bottom"]]
            else:
                hits = [z for z in pool if h[i] >= z["bottom"] and c[i] < z["top"]]
            if hits:
                z = min(hits, key=lambda q: abs(c[i] - (q["top"] + q["bottom"]) / 2))
                if side == "bull":
                    stop = min(z["bottom"], l[i]) - p.stop_atr * atr[i]
                    ok = stop < c[i]
                    target = c[i] + p.rr * (c[i] - stop)
                else:
                    stop = max(z["top"], h[i]) + p.stop_atr * atr[i]
                    ok = stop > c[i]
                    target = c[i] - p.rr * (stop - c[i])
                stacked = any(o is not z and _family(o["kind"]) != _family(z["kind"]) and _overlap(o, z) for o in pool)
                fresh = (i - last_cross) <= p.fresh_bars
                score = 4 + int(stacked) + int(fresh)
                if ok and score >= p.min_conf:
                    sig[i], conf[i] = (1 if side == "bull" else -1), score
                    out["entry"][i], out["stop"][i], out["target"][i] = c[i], stop, target
                    out["zone_top"][i], out["zone_bottom"][i] = z["top"], z["bottom"]
                    kind_arr[i], stacked_arr[i], fresh_arr[i] = z["kind"], stacked, fresh
                    z["used"] = True
        for z in by_conf.get(i, []):                              # zones confirmed at this close become tradable next candle
            active["bull" if z["kind"].startswith("bull") else "bear"].append(z)
        for k in active:
            active[k] = active[k][-p.max_active:]

    res = pd.DataFrame({"signal": sig, "conf": conf, **out, "zone_kind": kind_arr, "stacked": stacked_arr,
                        "fresh": fresh_arr, "trend": trend, "tott_buy": buy, "tott_sell": sell}, index=df.index)
    return res[COLUMNS]


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
    return pd.DataFrame([{"Indicator": "Confluence (TOTT+IMBA+OB)", "Value": trend, "Reading": txt}])
