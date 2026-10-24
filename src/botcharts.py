"""Figures for the owner-only Bot Desk."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

GREEN, BLUE = "#22c55e", "#38bdf8"


def _style(fig: go.Figure, title: str, h: int = 340) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=h, title=dict(text=title), margin=dict(l=10, r=10, t=45, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.12))
    return fig


def equity_chart(eq: pd.DataFrame, start: float) -> go.Figure:
    fig = go.Figure(go.Scatter(x=eq.index, y=eq["equity"].values, name="Equity", line=dict(color=BLUE, width=2)))
    fig.add_hline(y=start, line=dict(color="gray", dash="dot", width=1))
    return _style(fig, "Paper account equity (simulated money)")


def backtest_chart(eq: pd.Series, price: pd.Series, start: float) -> go.Figure:
    px = price.reindex(eq.index)
    bh = start * px / float(px.iloc[0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy", line=dict(color=GREEN, width=2)))
    fig.add_trace(go.Scatter(x=bh.index, y=bh.values, name="Buy & hold", line=dict(color="white", dash="dash", width=1.5)))
    return _style(fig, "Backtest equity vs buy & hold", 380)
