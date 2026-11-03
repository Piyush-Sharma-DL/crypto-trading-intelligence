"""Support / Resistance zones.

An independent implementation of the S/R rules of the TradingView script "Support Resistance with Order Blocks [TMF]"
by Trend-My-Friend (published under the Mozilla Public License 2.0; its order-block part is not used here).

  pivot high  : a candle whose high is >= the previous `left` highs and > the next `right` highs  -> RESISTANCE
  pivot low   : a candle whose low  is <= the previous `left` lows  and < the next `right` lows   -> SUPPORT
  zone        : the outer quarter of that candle's range (resistance: top quarter, support: bottom quarter)
  created     : `right` candles after the pivot (when it is confirmed); ignored if the pivot candle overlaps an existing active zone
  broken      : resistance when a later high reaches the pivot high; support when a later low reaches the pivot low
  max active  : the oldest zone of each kind is dropped when more than `max_active` are alive
Everything is decided from past candles only, so zones never repaint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sr_zones(df: pd.DataFrame, left: int = 20, right: int = 5, max_active: int = 10) -> list[dict]:
    h, l = df["High"].values, df["Low"].values
    idx = df.index
    zones: list[dict] = []
    active: dict[str, list[dict]] = {"resistance": [], "support": []}

    def make(kind, p, i, top, bottom):
        z = {"kind": kind, "i": p, "conf": i, "t0": idx[p], "top": float(top), "bottom": float(bottom),
             "mitigated": False, "t_mit": None, "i_mit": None, "dropped": False, "i_drop": None, "i_end": None}
        zones.append(z)
        active[kind].append(z)
        if len(active[kind]) > max_active:                            # the oldest one disappears (as in the original)
            old = active[kind].pop(0)
            old["dropped"], old["i_drop"], old["i_end"] = True, i, i
        return z

    def overlaps(kind, lo, hi):
        return any(not (hi < z["bottom"] or lo > z["top"]) for z in active[kind])

    for i in range(len(df)):
        p = i - right
        if p >= left:
            if h[p] >= h[p - left:p].max() and h[p] > h[p + 1:i + 1].max() and not overlaps("resistance", l[p], h[p]):
                make("resistance", p, i, h[p], h[p] - (h[p] - l[p]) / 4)
            if l[p] <= l[p - left:p].min() and l[p] < l[p + 1:i + 1].min() and not overlaps("support", l[p], h[p]):
                make("support", p, i, l[p] + (h[p] - l[p]) / 4, l[p])
        for z in list(active["resistance"]):                         # a later high reaching the pivot high breaks resistance
            if h[i] >= z["top"]:
                z.update(mitigated=True, t_mit=idx[i], i_mit=i, i_end=i)
                active["resistance"].remove(z)
        for z in list(active["support"]):                            # a later low reaching the pivot low breaks support
            if l[i] <= z["bottom"]:
                z.update(mitigated=True, t_mit=idx[i], i_mit=i, i_end=i)
                active["support"].remove(z)
    return zones
