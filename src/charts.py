"""All Plotly figures in one place so the app stays readable."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GREEN, RED, AMBER, BLUE, VIOLET = "#22c55e", "#ef4444", "#f59e0b", "#38bdf8", "#a78bfa"
PALETTE = {"Linear Regression": "#94a3b8", "Random Forest": "#34d399", "XGBoost": "#f472b6",
           "Gradient Boosting": "#f472b6", "LSTM": VIOLET, "Ensemble": "#fbbf24"}
SIG_COLOR = {"BUY": GREEN, "SELL": RED, "HOLD": AMBER}


def _style(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    kw = dict(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=45 if title else 15, b=10),
              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              legend=dict(orientation="h", y=1.1))
    if title:
        kw["title"] = dict(text=title)
    fig.update_layout(**kw)
    return fig


def color_for(name: str) -> str:
    return PALETTE.get(name, BLUE)


def price_chart(ind: pd.DataFrame, sig: pd.DataFrame | None, days: int) -> go.Figure:
    v = ind.iloc[-days:]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.52, 0.12, 0.18, 0.18])
    fig.add_trace(go.Candlestick(x=v.index, open=v.Open, high=v.High, low=v.Low, close=v.Close, name="Price",
                                 increasing_line_color=GREEN, decreasing_line_color=RED), 1, 1)
    for col, color in (("SMA_21", BLUE), ("SMA_50", AMBER)):
        fig.add_trace(go.Scatter(x=v.index, y=v[col], name=col, line=dict(color=color, width=1.2)), 1, 1)
    for col in ("BB_upper", "BB_lower"):
        fig.add_trace(go.Scatter(x=v.index, y=v[col], name="Bollinger", showlegend=(col == "BB_upper"),
                                 line=dict(color="rgba(148,163,184,.6)", width=1, dash="dot")), 1, 1)
    if sig is not None:
        sv = sig[sig.index.isin(v.index)]
        for kind, symbol, color, ycol, mult in (("BUY", "triangle-up", GREEN, "Low", 0.97),
                                                ("SELL", "triangle-down", RED, "High", 1.03)):
            pts = sv[sv["signal"] == kind].index
            if len(pts):
                fig.add_trace(go.Scatter(x=pts, y=v.loc[pts, ycol] * mult, mode="markers", name=kind,
                                         marker=dict(symbol=symbol, size=10, color=color)), 1, 1)
    fig.add_trace(go.Bar(x=v.index, y=v.Volume, name="Volume", marker_color="rgba(56,189,248,.5)"), 2, 1)
    fig.add_trace(go.Scatter(x=v.index, y=v.RSI_14, name="RSI", line=dict(color=VIOLET)), 3, 1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line=dict(color="gray", dash="dot", width=1), row=3, col=1)
    fig.add_trace(go.Bar(x=v.index, y=v.MACD_hist, name="MACD hist",
                         marker_color=np.where(v.MACD_hist >= 0, GREEN, RED)), 4, 1)
    fig.add_trace(go.Scatter(x=v.index, y=v.MACD, name="MACD", line=dict(color=BLUE, width=1)), 4, 1)
    fig.add_trace(go.Scatter(x=v.index, y=v.MACD_signal, name="Signal", line=dict(color=AMBER, width=1)), 4, 1)
    _style(fig, 820)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


def votes_chart(row: pd.Series) -> go.Figure:
    labels = ["Model forecast", "Trend (SMA50)", "Momentum (MACD)", "RSI zone"]
    vals = [int(row["v_model"]), int(row["v_trend"]), int(row["v_momentum"]), int(row["v_rsi"])]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h",
                           marker_color=[GREEN if v > 0 else RED if v < 0 else "#64748b" for v in vals],
                           text=[f"{v:+d}" for v in vals], textposition="outside"))
    fig.update_xaxes(range=[-2.6, 2.6], zeroline=True)
    fig.update_yaxes(autorange="reversed")
    return _style(fig, 240, "Signal votes")


def contrib_chart(contrib: pd.Series, label: str, top: int = 10) -> go.Figure:
    c = contrib.reindex(contrib.abs().sort_values().tail(top).index) * 100
    fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h",
                           marker_color=[GREEN if v > 0 else RED for v in c.values]))
    return _style(fig, 340, f"What is pushing tomorrow's forecast? ({label}, % points)")


def metric_bar(metrics: pd.DataFrame, col: str, title: str, ref: float | None = None, ref_label: str = "",
               yrange=None) -> go.Figure:
    m = metrics.dropna(subset=[col])
    fig = go.Figure(go.Bar(x=m.index, y=m[col], marker_color=[color_for(i) for i in m.index]))
    if ref is not None:
        fig.add_hline(y=ref, line=dict(dash="dash", color="white"), annotation_text=ref_label)
    if yrange:
        fig.update_yaxes(range=yrange)
    return _style(fig, 340, title)


def pred_vs_actual(close: pd.Series, preds: pd.DataFrame, names: list[str], win: int) -> go.Figure:
    ct = close.reindex(preds.index)
    actual = close.shift(-1).reindex(preds.index)
    fig = go.Figure(go.Scatter(x=actual.index[-win:], y=actual.values[-win:], name="Actual next-day close",
                               line=dict(color="white", width=2)))
    for n in names:
        fig.add_trace(go.Scatter(x=ct.index[-win:], y=(ct * np.exp(preds[n])).values[-win:], name=n,
                                 line=dict(color=color_for(n), width=1.4)))
    return _style(fig, 420, "Predicted vs actual (each forecast was made the day before)")


def importance_chart(imp: pd.Series, top: int = 12) -> go.Figure:
    t = imp.tail(top)
    return _style(go.Figure(go.Bar(x=t.values, y=t.index, orientation="h", marker_color="#34d399")), 380,
                  "Random Forest feature importance")


def lstm_curve(hist: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=hist["train"], name="train", line=dict(color=VIOLET)))
    fig.add_trace(go.Scatter(y=hist["val"], name="validation", line=dict(color=AMBER)))
    return _style(fig, 380, "LSTM training curve (Huber loss)")


def equity_chart(equity: dict[str, pd.Series], selected: str, bh: pd.Series) -> go.Figure:
    fig = go.Figure()
    for n, s in equity.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=n, opacity=1 if n == selected else .5,
                                 line=dict(color=color_for(n), width=3 if n == selected else 1)))
    fig.add_trace(go.Scatter(x=bh.index, y=bh.values, name="Buy & Hold", line=dict(color="white", dash="dash", width=2)))
    return _style(fig, 430, "Equity curves (start = 1.0)")


def drawdown_chart(dd: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd["Strategy"] * 100, fill="tozeroy", name="Strategy", line=dict(color=RED)))
    fig.add_trace(go.Scatter(x=dd.index, y=dd["Buy & Hold"] * 100, name="Buy & Hold", line=dict(color="white", dash="dash")))
    return _style(fig, 340, title)


def signal_mix(counts: pd.Series, title: str) -> go.Figure:
    return _style(go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[SIG_COLOR[i] for i in counts.index])), 340, title)


def mc_hist(mc: pd.DataFrame, actual_ret: float, bh_ret: float) -> go.Figure:
    fig = go.Figure(go.Histogram(x=mc["total_return"] * 100, nbinsx=45, marker_color=BLUE, opacity=.85))
    fig.add_vline(x=actual_ret * 100, line=dict(color=GREEN, width=3), annotation_text="actual")
    fig.add_vline(x=bh_ret * 100, line=dict(color="white", dash="dash"), annotation_text="buy & hold")
    fig.add_vline(x=0, line=dict(color="gray", dash="dot"))
    fig.update_xaxes(title_text="Total return over the test period (%)")
    return _style(fig, 340, "Monte Carlo: 2,000 re-shuffled versions of the strategy's returns")


def paper_equity(curves: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for n in curves.columns:
        s = curves[n].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=n, line=dict(color=color_for(n))))
    return _style(fig, 360, "Paper-trading equity (live forecasts only)")


def scanner_bar(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["Forecast (%)"]).sort_values("Forecast (%)")
    return _style(go.Figure(go.Bar(x=d["Forecast (%)"], y=d["Asset"], orientation="h",
                                   marker_color=[GREEN if v > 0 else RED for v in d["Forecast (%)"]])), 380,
                  "Next-day forecast by asset (%)")
