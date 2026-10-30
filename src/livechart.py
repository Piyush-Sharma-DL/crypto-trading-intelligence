"""Plotly figure for the live chart: candles + user-selected overlays and panels."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CONF = "Confluence (TOTT + IMBA + OB)"
NEWS = "News markers"
IMBA_ALGO = "IMBA ALGO (Fib trend)"
GREEN, RED, BLUE, AMBER, VIOLET = "#22c55e", "#ef4444", "#38bdf8", "#f59e0b", "#a78bfa"
# TradingView-style palette
BG, GRID, TXT, UP, DN, CROSS = "#131722", "#2a2e39", "#b2b5be", "#26a69a", "#ef5350", "#9598a1"
FONT = "Trebuchet MS, Roboto, Ubuntu, sans-serif"
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
               show_old_zones: bool = False, conf: pd.DataFrame | None = None,
               news: pd.DataFrame | None = None, bot: dict | None = None, imba: pd.DataFrame | None = None,
               imba_ladder: dict | None = None, title: str = "", source: str = "", show_volume: bool = True) -> go.Figure:
    v = df.iloc[-n:]
    x = _loc(v.index, tz)
    step = pd.Timedelta(minutes=tf_minutes)
    n_pan = len(panels)
    fig = make_subplots(rows=1 + n_pan, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.62] + [0.38 / n_pan] * n_pan if n_pan else [1.0],
                        specs=[[{"secondary_y": True}]] + [[{}] for _ in range(n_pan)])

    def line(s, name, color, width=1.3, dash=None, row=1):
        fig.add_trace(go.Scatter(x=x, y=s.reindex(v.index).values, name=name, mode="lines",
                                 line=dict(color=color, width=width, dash=dash)), row, 1)

    fig.add_trace(go.Candlestick(x=x, open=v.Open, high=v.High, low=v.Low, close=v.Close, name="Price",
                                 increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
                                 decreasing=dict(line=dict(color=DN, width=1), fillcolor=DN)), 1, 1)
    if show_volume:                                                       # TradingView-style: faint volume bars along the bottom
        fig.add_trace(go.Bar(x=x, y=v["Volume"].values, name="Volume", showlegend=False, hoverinfo="skip",
                             marker_color=np.where(v["Close"] >= v["Open"], "rgba(38,166,154,.28)", "rgba(239,83,80,.28)")),
                      1, 1, secondary_y=True)
        fig.update_yaxes(range=[0, float(v["Volume"].max() or 1) * 4.5], showgrid=False, visible=False, secondary_y=True, row=1, col=1)

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
    if "TOTT" in overlays or CONF in overlays:
        t = ind["tott"].reindex(v.index)
        line(t["support"], "TOTT support", BLUE, 2)
        line(t["ott_up"], "OTT up", GREEN, 1.6)
        fig.add_trace(go.Scatter(x=x, y=t["ott_dn"].values, name="OTT down", mode="lines",
                                 line=dict(color=RED, width=1.6), fill="tonexty", fillcolor="rgba(56,189,248,.10)"), 1, 1)
        for col, label, color, sym, ref, mult in (("buy", "BUY", GREEN, "triangle-up", "Low", 0.9985),
                                                  ("sell", "SELL", RED, "triangle-down", "High", 1.0015)):
            m = t[col].values
            if "TOTT" in overlays and m.any():
                fig.add_trace(go.Scatter(x=x[m], y=(v[ref].values * mult)[m], mode="markers+text", name=f"TOTT {label}",
                                         text=[label] * int(m.sum()), textfont=dict(size=10, color=color),
                                         textposition="bottom center" if col == "buy" else "top center",
                                         marker=dict(symbol=sym, size=11, color=color)), 1, 1)
    if conf is not None and CONF in overlays:
        cv = conf.reindex(v.index)
        for side, name, color, sym, ref, mult, pos in ((1, "LONG", GREEN, "triangle-up", "Low", 0.997, "bottom center"),
                                                       (-1, "SHORT", RED, "triangle-down", "High", 1.003, "top center")):
            m = (cv["signal"] == side).fillna(False).values
            if m.any():
                fig.add_trace(go.Scatter(
                    x=x[m], y=(v[ref].values * mult)[m], mode="markers+text", name=f"Confluence {name}",
                    text=[f"{name} {int(k)}" for k in cv["conf"].values[m]], textposition=pos, textfont=dict(size=10, color=color),
                    marker=dict(symbol=sym, size=13, color=color, line=dict(width=1, color="white")),
                    hovertext=[f"entry {e:,.2f}<br>stop {s:,.2f}<br>target {tg:,.2f}" for e, s, tg in
                               zip(cv["entry"].values[m], cv["stop"].values[m], cv["target"].values[m])], hoverinfo="text"), 1, 1)
        sigs = cv[cv["signal"].fillna(0) != 0]
        if len(sigs):                                                    # stop / target ladder of the newest signal
            r = sigs.iloc[-1]
            t0, t1 = _loc(sigs.index[-1], tz), _loc(v.index[-1] + step * 8, tz)
            for y, col_, lab in ((r["stop"], RED, "SL"), (r["target"], GREEN, "TP")):
                fig.add_shape(type="line", x0=t0, x1=t1, y0=y, y1=y, line=dict(color=col_, width=1.5, dash="dash"), row=1, col=1)
                fig.add_annotation(x=t1, y=y, text=f"{lab} {y:,.2f}", showarrow=False, xanchor="left",
                                   font=dict(size=10, color=col_), row=1, col=1)
    if news is not None and NEWS in overlays and len(news):               # headlines on the candle they were published in
        nv = news[(news["time"] >= v.index[0]) & (news["time"] < v.index[-1] + step)].head(25)
        if len(nv):
            at = np.clip(v.index.searchsorted(nv["time"], side="right") - 1, 0, len(v) - 1)
            hi = nv["impact"].values == 2
            fig.add_trace(go.Scatter(
                x=x[at], y=v["High"].values[at] * 1.0015, mode="markers", name="News",
                marker=dict(symbol=["star" if h else "diamond" for h in hi], size=[14 if h else 9 for h in hi],
                            color=[GREEN if s > .08 else RED if s < -.08 else "#94a3b8" for s in nv["sentiment"]],
                            line=dict(width=1, color="white")),
                hovertext=[f"<b>{html.escape(str(t))}</b><br>{html.escape(str(s))} · mood {m:+.2f}"
                           for t, s, m in zip(nv["title"], nv["source"], nv["sentiment"])], hoverinfo="text"), 1, 1)
    if bot:                                                               # the owner's paper bot, drawn on the chart
        tr = bot.get("trades")
        if tr is not None and len(tr):
            t_in, t_out = pd.to_datetime(tr["entry_time"]), pd.to_datetime(tr["exit_time"])
            m_in = (t_in >= v.index[0]).values
            if m_in.any():
                long_ = (tr["side"] == "LONG").values
                fig.add_trace(go.Scatter(
                    x=_loc(pd.DatetimeIndex(t_in[m_in]), tz), y=tr["entry"].values[m_in], mode="markers", name="Bot entry",
                    marker=dict(symbol=["triangle-up" if s else "triangle-down" for s in long_[m_in]], size=12, color=BLUE,
                                line=dict(width=1.5, color="white"))), 1, 1)
            m_out = (t_out >= v.index[0]).values
            if m_out.any():
                fig.add_trace(go.Scatter(
                    x=_loc(pd.DatetimeIndex(t_out[m_out]), tz), y=tr["exit"].values[m_out], mode="markers+text", name="Bot exit",
                    text=[f"{r:+.1f}R" for r in tr["r_multiple"].values[m_out]], textposition="top center",
                    marker=dict(symbol="x", size=11, line=dict(width=2),
                                color=[GREEN if p > 0 else RED for p in tr["pnl"].values[m_out]])), 1, 1)
        p = bot.get("position")
        if p:
            t0, t1 = _loc(pd.Timestamp(p["opened"]), tz), _loc(v.index[-1] + step * 8, tz)
            for y, col_, lab in ((p["entry"], BLUE, "BOT ENTRY"), (p["stop"], RED, "BOT SL"), (p["target"], GREEN, "BOT TP")):
                fig.add_shape(type="line", x0=min(t0, t1), x1=t1, y0=y, y1=y, line=dict(color=col_, width=1.5, dash="dot"), row=1, col=1)
                fig.add_annotation(x=t1, y=y, text=f"{lab} {y:,.2f}", showarrow=False, xanchor="left",
                                   font=dict(size=10, color=col_), row=1, col=1)
    if imba is not None and IMBA_ALGO in overlays:
        r = imba.reindex(v.index)
        for d_, col_, nm in ((1, UP, "IMBA trend (long)"), (-1, DN, "IMBA trend (short)")):
            fig.add_trace(go.Scatter(x=x, y=r["f5"].where(r["trend"] == d_).values, name=nm, mode="lines",
                                     line=dict(color=col_, width=3)), 1, 1)
        for col_, nm in (("f236", "IMBA 23.6%"), ("f786", "IMBA 78.6%")):
            fig.add_trace(go.Scatter(x=x, y=r[col_].values, name=nm, mode="lines",
                                     line=dict(color="rgba(148,163,184,.55)", width=1, dash="dot")), 1, 1)
        for side, nm, col_, sym, ref, mult, pos in ((1, "IMBA LONG", UP, "triangle-up", "Low", 0.9975, "bottom center"),
                                                    (-1, "IMBA SHORT", DN, "triangle-down", "High", 1.0025, "top center")):
            m = (r["signal"] == side).values
            if m.any():
                fig.add_trace(go.Scatter(x=x[m], y=(v[ref].values * mult)[m], mode="markers+text", name=nm,
                                         text=[nm] * int(m.sum()), textposition=pos, textfont=dict(size=10, color=col_),
                                         marker=dict(symbol=sym, size=13, color=col_, line=dict(width=1, color="white"))), 1, 1)
        if imba_ladder:                                                   # entry / stop / TP1-4 of the newest IMBA signal
            L = imba_ladder
            t0, t1 = max(_loc(L["t0"], tz), x[0]), _loc(v.index[-1] + step * 8, tz)
            rows = [(L["entry"], "#9598a1", "solid", "ENTRY", "")] + \
                   [(L["sl"], RED, "dot", "SL", " ✖" if L["sl_hit"] else "")] + \
                   [(p, UP, "dot", f"TP{k}", " ✔" if hit else "") for k, p, hit in L["tps"]]
            for y, col_, dash, lab, mark in rows:
                fig.add_shape(type="line", x0=t0, x1=t1, y0=y, y1=y, line=dict(color=col_, width=1.3, dash=dash), row=1, col=1)
                fig.add_annotation(x=t1, y=y, text=f"{lab} {y:,.2f}{mark}", showarrow=False, xanchor="left",
                                   font=dict(size=10, color=col_), row=1, col=1)
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

    last_up = float(v["Close"].iloc[-1]) >= float(v["Open"].iloc[-1])
    tag = UP if last_up else DN                                           # coloured last-price tag, like TradingView
    fig.add_hline(y=price, line=dict(color=tag, width=1, dash="dot"), annotation_text=f"{price:,.2f}",
                  annotation_position="right", annotation_font=dict(color="white", size=11), annotation_bgcolor=tag,
                  row=1, col=1)

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

    for k, name in enumerate(panels, start=2):                              # panel titles, top-left inside each panel
        fig.add_annotation(text=f"<b>{name}</b>", xref="x domain", yref="y domain", x=0.004, y=0.96, showarrow=False,
                           xanchor="left", yanchor="top", font=dict(size=11, color=CROSS), row=k, col=1)

    o, h_, l_, c_ = (float(v[k].iloc[-1]) for k in ("Open", "High", "Low", "Close"))
    chg = (c_ / float(v["Close"].iloc[-2]) - 1) * 100 if len(v) > 1 else 0.0
    colr = UP if chg >= 0 else DN
    names = " · ".join(html.escape(str(o_)) for o_ in overlays) if overlays else "no indicators"
    head = (f"<b>{html.escape(title)}</b>  <span style='color:{CROSS}'>{html.escape(source)}</span>   "
            f"O <span style='color:{colr}'>{o:,.2f}</span>  H <span style='color:{colr}'>{h_:,.2f}</span>  "
            f"L <span style='color:{colr}'>{l_:,.2f}</span>  C <span style='color:{colr}'>{c_:,.2f}</span>  "
            f"<span style='color:{colr}'>{chg:+.2f}%</span><br><span style='font-size:10px;color:{CROSS}'>{names}</span>")
    fig.add_annotation(text=head, xref="paper", yref="paper", x=0.003, y=1.0, xanchor="left", yanchor="bottom",
                       showarrow=False, align="left", font=dict(size=12, color=TXT))

    fig.update_layout(template="plotly_dark", height=580 + 150 * n_pan, margin=dict(l=8, r=64, t=58, b=8),
                      paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family=FONT, size=12, color=TXT), showlegend=False,
                      xaxis_rangeslider_visible=False, dragmode="pan", uirevision=uirevision, hovermode="x",
                      hoverlabel=dict(bgcolor="#1e222d", bordercolor=GRID, font=dict(family=FONT, size=12)),
                      newshape=dict(line=dict(color="#2962ff", width=2), fillcolor="rgba(41,98,255,.15)"),
                      modebar=dict(bgcolor="rgba(0,0,0,0)", color=CROSS, activecolor="#2962ff"))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False, showspikes=True,
                     spikemode="across", spikesnap="cursor", spikecolor=CROSS, spikethickness=1, spikedash="dash",
                     tickfont=dict(size=11, color=TXT))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False, side="right",
                     separatethousands=True, showspikes=True, spikemode="across", spikecolor=CROSS, spikethickness=1,
                     spikedash="dash", tickfont=dict(size=11, color=TXT), secondary_y=False)
    fig.update_yaxes(range=[y_lo, y_hi], secondary_y=False, row=1, col=1)
    return fig
