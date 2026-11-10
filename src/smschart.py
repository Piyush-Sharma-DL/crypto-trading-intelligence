"""Plotly drawing of the Irfan Smart Market Structure layers (each layer can be switched on or off)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SMS_LAYERS = ["EMA 9/21 cloud", "MA 50 / 200", "Supertrend", "Bollinger Bands", "Fibonacci", "Liquidity levels", "Liquidity sweeps",
              "Trendlines", "Order blocks", "Cross signals"]
UP, DN, PURPLE = "#26a69a", "#ef5350", "rgba(180,80,255,.85)"


def add_sms(fig, v: pd.DataFrame, x, step, loc, idx: pd.DatetimeIndex, bundle: dict, layers, y_lo: float, y_hi: float) -> None:
    ln = bundle["lines"].reindex(v.index)
    ev = bundle["events"].reindex(v.index)
    for col in ("ema_up", "ema_dn", "golden", "death", "bsl", "ssl"):
        ev[col] = ev[col].fillna(False).astype(bool)
    n, vl = len(idx), len(idx) - len(v)                                   # vl = position of the first visible candle

    def tpos(k: int):                                                     # candle number -> time (also beyond the last candle)
        return loc(idx[k]) if k < n else loc(idx[-1] + step * (k - (n - 1)))

    def line(s, name, color, width=1.2, dash=None):
        fig.add_trace(go.Scatter(x=x, y=s.values, name=name, mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color=color, width=width, dash=dash)), 1, 1)

    if "EMA 9/21 cloud" in layers:
        bull = ln["ema_bull"].astype(bool)
        for state, col, fillc, nm in ((True, UP, "rgba(38,166,154,.18)", "up"), (False, DN, "rgba(239,83,80,.18)", "down")):
            m = bull == state
            fig.add_trace(go.Scatter(x=x, y=ln["ema_slow"].where(m).values, name=f"EMA 21 {nm}", mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"), 1, 1)
            fig.add_trace(go.Scatter(x=x, y=ln["ema_fast"].where(m).values, name=f"EMA 9 {nm}", mode="lines", line=dict(color=col, width=2),
                                     fill="tonexty", fillcolor=fillc, showlegend=False, hoverinfo="skip"), 1, 1)
    if "MA 50 / 200" in layers:
        line(ln["ma_mid"], "MA 50", "#f59e0b", 2)
        line(ln["ma_long"], "MA 200", "#3b82f6", 2)
    if "Supertrend" in layers:
        bull = ln["st_bull"].astype(bool)
        for state, col, nm in ((True, UP, "SMS Supertrend up"), (False, DN, "SMS Supertrend down")):
            line(ln["st_line"].where(bull == state), nm, col, 2)
    if "Bollinger Bands" in layers:
        line(ln["bb_up"], "BB upper", "rgba(34,211,238,.6)", 1)
        line(ln["bb_low"], "BB lower", "rgba(34,211,238,.6)", 1)
        line(ln["bb_mid"], "BB basis", "#facc15", 1)
    if "Fibonacci" in layers:
        for key, nm, a, w in (("fib0", "0%", .5, 1), ("fib236", "23.6%", .6, 1), ("fib382", "38.2%", .7, 1), ("fib500", "50%", .8, 1),
                              ("fib618", "61.8%", 1.0, 2), ("fib786", "78.6%", .7, 1), ("fib1000", "100%", .5, 1)):
            line(ln[key], f"Fib {nm}", f"rgba(80,150,255,{a})", w)

    if "Liquidity levels" in layers:
        evc = bundle.get("events_closed", bundle["events"])               # swing levels come from CLOSED candles
        for col, bar, text in (("swing_high", bundle["swing_high_bar"], "Sell Liquidity"), ("swing_low", bundle["swing_low_bar"], "Buy Liquidity")):
            val = evc[col].iloc[-1] if len(evc) else np.nan
            if np.isfinite(val) and bar >= 0 and y_lo <= val <= y_hi:
                x0, x1 = tpos(max(bar, vl)), tpos(n - 1 + 8)
                fig.add_shape(type="line", x0=x0, x1=x1, y0=val, y1=val, line=dict(color=PURPLE, width=2, dash="dash"), row=1, col=1)
                fig.add_annotation(x=x1, y=val, text=f"{text} {val:,.2f}", showarrow=False, xanchor="left",
                                   font=dict(size=10, color=PURPLE), row=1, col=1)
    if "Liquidity sweeps" in layers:
        for col, label, ref, mult, pos in (("bsl", "BSL SWEEP", "High", 1.002, "top center"), ("ssl", "SSL SWEEP", "Low", 0.998, "bottom center")):
            m = ev[col].values
            if m.any():
                fig.add_trace(go.Scatter(x=x[m], y=(v[ref].values * mult)[m], mode="markers+text", name=label, text=[label] * int(m.sum()),
                                         textposition=pos, textfont=dict(size=9, color=PURPLE), showlegend=False,
                                         marker=dict(symbol="diamond", size=9, color=PURPLE)), 1, 1)
    if "Trendlines" in layers:
        x_end = n - 1 + 8
        for t in bundle["trendlines"]:
            y_left, y_right = t["y2"] + t["slope"] * (vl - t["x2"]), t["y2"] + t["slope"] * (x_end - t["x2"])
            if max(y_left, y_right) < y_lo or min(y_left, y_right) > y_hi:
                continue
            start = max(t["x1"], vl)
            y0 = t["y1"] + t["slope"] * (start - t["x1"])
            col = "239,83,80" if t["kind"] == "resistance" else "38,166,154"
            fig.add_shape(type="line", x0=tpos(start), x1=tpos(x_end), y0=y0, y1=y_right,
                          line=dict(color=f"rgba({col},{.95 if t['strong'] else .35})", width=3 if t["strong"] else 1), row=1, col=1)
    if "Order blocks" in layers:
        for z in bundle["order_blocks"]:
            if z["dropped"]:
                continue
            end = z["i_right"] if z["i_end"] is None else min(z["i_end"], z["i_right"])
            if end < vl or z["top"] < y_lo or z["bottom"] > y_hi:
                continue
            x0 = tpos(max(z["i"], vl))
            x1 = max(tpos(end), x0)
            bull = z["kind"] == "bull"
            col = "38,166,154" if bull else "239,83,80"
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=z["bottom"], y1=z["top"], line=dict(color=f"rgba({col},{.35 if z['touched'] else .8})", width=1),
                          fillcolor=f"rgba({col},{.10 if z['touched'] else .22})", layer="below", row=1, col=1)
            fig.add_annotation(x=x0, y=z["top"], text="Bullish OB" if bull else "Bearish OB", showarrow=False, xanchor="left", yanchor="bottom",
                               font=dict(size=9, color=f"rgb({col})"), row=1, col=1)
    if "Cross signals" in layers:
        for col, label, color, sym, ref, mult, pos in (("ema_up", "EMA UP", UP, "triangle-up", "Low", 0.998, "bottom center"),
                                                       ("ema_dn", "EMA DN", DN, "triangle-down", "High", 1.002, "top center"),
                                                       ("golden", "GOLDEN", "#22c55e", "star", "Low", 0.996, "bottom center"),
                                                       ("death", "DEATH", "#ef4444", "star", "High", 1.004, "top center")):
            m = ev[col].values
            if m.any():
                fig.add_trace(go.Scatter(x=x[m], y=(v[ref].values * mult)[m], mode="markers+text", name=label, text=[label] * int(m.sum()),
                                         textposition=pos, textfont=dict(size=9, color=color), showlegend=False,
                                         marker=dict(symbol=sym, size=8 if sym != "star" else 12, color=color)), 1, 1)
