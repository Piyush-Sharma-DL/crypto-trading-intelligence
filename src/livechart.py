"""Plotly figure for the live chart: candles + user-selected overlays and panels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GREEN, RED, BLUE, AMBER, VIOLET = "#22c55e", "#ef4444", "#38bdf8", "#f59e0b", "#a78bfa"
ZONE_STYLE = {  # fill, border, label
    "bull_fvg": ("rgba(34,197,94,.16)", "rgba(34,197,94,.55)", "IMBA ↑"),
    "bear_fvg": ("rgba(239,68,68,.16)", "rgba(239,68,68,.55)", "IMBA ↓"),
    "bull_ob": ("rgba(56,189,248,.22)", "rgba(56,189,248,.9)", "OB ↑"),
    "bear_ob": ("rgba(245,158,11,.22)", "rgba(245,158,11,.9)", "OB ↓"),
}


def _loc(x, tz: str):
    """UTC (naive) timestamps -> wall-clock time in the viewer's time zone."""
    if isinstance(x, pd.DatetimeIndex):
        return x.tz_localize("UTC").tz_convert(tz).tz_localize(None)
    return pd.Timestamp(x).tz_localize("UTC").tz_convert(tz).tz_localize(None)


def live_chart(df: pd.DataFrame, tf_minutes: int, overlays: list[str], panels: list[str], ind: dict,
               zones: list[dict], price: float, tz: str = "UTC", n: int = 150, uirevision: str = "",
               show_old_zones: bool = False) -> go.Figure:
    v = df.iloc[-n:]
    x = _loc(v.index, tz)
    step = pd.Timedelta(minutes=tf_minutes)
    n_pan = len(panels)
    fig = make_subplots(rows=1 + n_pan, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.62] + [0.38 / n_pan] * n_pan if n_pan else [1.0])

    def line(s, name, color, width=1.3, dash=None, row=1):
        fig.add_trace(go.Scatter(x=x, y=s.reindex(v.index).values, name=name, mode="lines",
                                 line=dict(color=color, width=width, dash=dash)), row, 1)

    fig.add_trace(go.Candlestick(x=x, open=v.Open, high=v.High, low=v.Low, close=v.Close, name="Price",
                                 increasing_line_color=GREEN, decreasing_line_color=RED), 1, 1)

    # ---- zones first so they sit behind the lines
    shown = []
    end = _loc(v.index[-1] + step * max(4, n // 12), tz)
    lo, hi = float(v["Low"].min()), float(v["High"].max())
    margin = (hi - lo) * 0.06
    y_lo, y_hi = lo - margin, hi + margin                         # axis fits the candles, not far-away zones
    for z in zones:
        t0 = _loc(z["t0"], tz)
        t_end = _loc(z["t_mit"], tz) if z["mitigated"] else end
        if t_end < x[0] or z["top"] < y_lo or z["bottom"] > y_hi:
            continue
        if t0 < x[0] and not show_old_zones:                      # formed before the visible candles
            continue
        fill, border, label = ZONE_STYLE[z["kind"]]
        fig.add_shape(type="rect", x0=max(t0, x[0]), x1=t_end, y0=z["bottom"], y1=z["top"],
                      fillcolor=fill, line=dict(color=border, width=1), layer="below", row=1, col=1)
        shown.append((z, label))
    for z, label in shown:
        fig.add_annotation(x=max(_loc(z["t0"], tz), x[0]), y=z["top"], text=label, showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(size=10, color=ZONE_STYLE[z["kind"]][1]), row=1, col=1)

    # ---- overlays
    if "TOTT" in overlays:
        t = ind["tott"].reindex(v.index)
        line(t["support"], "TOTT support", BLUE, 2)
        line(t["ott_up"], "OTT up", GREEN, 1.6)
        fig.add_trace(go.Scatter(x=x, y=t["ott_dn"].values, name="OTT down", mode="lines",
                                 line=dict(color=RED, width=1.6), fill="tonexty", fillcolor="rgba(56,189,248,.10)"), 1, 1)
        for col, label, color, sym, ref, mult in (("buy", "BUY", GREEN, "triangle-up", "Low", 0.9985),
                                                  ("sell", "SELL", RED, "triangle-down", "High", 1.0015)):
            m = t[col].values
            if m.any():
                fig.add_trace(go.Scatter(x=x[m], y=(v[ref].values * mult)[m], mode="markers+text", name=f"TOTT {label}",
                                         text=[label] * int(m.sum()), textfont=dict(size=10, color=color),
                                         textposition="bottom center" if col == "buy" else "top center",
                                         marker=dict(symbol=sym, size=11, color=color)), 1, 1)
    for name, key, color in (("SMA 20", "sma20", BLUE), ("SMA 50", "sma50", AMBER), ("EMA 21", "ema21", "#f472b6"),
                             ("EMA 200", "ema200", "#e2e8f0")):
        if name in overlays:
            line(ind[key], name, color)
    if "Bollinger Bands" in overlays:
        for col in ("upper", "lower"):
            line(ind["bb"][col], "Bollinger", "rgba(148,163,184,.8)", 1, "dot")
        fig.data[-2].showlegend = True
        fig.data[-1].showlegend = False
    if "VWAP" in overlays:
        line(ind["vwap"], "VWAP", "#facc15", 1.4, "dash")
    if "Supertrend" in overlays:
        st = ind["st"].reindex(v.index)
        for d_, color, name in ((1, GREEN, "Supertrend up"), (-1, RED, "Supertrend down")):
            fig.add_trace(go.Scatter(x=x, y=st["line"].where(st["dir"] == d_).values, name=name, mode="lines",
                                     line=dict(color=color, width=1.5)), 1, 1)

    fig.add_hline(y=price, line=dict(color=AMBER, width=1, dash="dot"), annotation_text=f"{price:,.2f}",
                  annotation_position="right", annotation_font=dict(color=AMBER), row=1, col=1)

    # ---- panels
    for k, name in enumerate(panels, start=2):
        if name == "Volume":
            fig.add_trace(go.Bar(x=x, y=v["Volume"].values, name="Volume",
                                 marker_color=np.where(v["Close"] >= v["Open"], "rgba(34,197,94,.6)", "rgba(239,68,68,.6)")), k, 1)
        elif name == "RSI":
            line(ind["rsi"], "RSI (14)", VIOLET, 1.4, row=k)
            for lvl in (30, 70):
                fig.add_hline(y=lvl, line=dict(color="gray", dash="dot", width=1), row=k, col=1)
            fig.update_yaxes(range=[0, 100], row=k, col=1)
        elif name == "MACD":
            m = ind["macd"].reindex(v.index)
            fig.add_trace(go.Bar(x=x, y=m["hist"].values, name="MACD hist",
                                 marker_color=np.where(m["hist"] >= 0, "rgba(34,197,94,.6)", "rgba(239,68,68,.6)")), k, 1)
            line(m["macd"], "MACD", BLUE, 1.2, row=k)
            line(m["signal"], "Signal", AMBER, 1.2, row=k)

    fig.update_layout(template="plotly_dark", height=560 + 150 * n_pan, margin=dict(l=10, r=70, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.04), dragmode="pan", uirevision=uirevision, hovermode="x")
    fig.update_yaxes(side="right", range=[y_lo, y_hi], row=1, col=1)
    return fig
