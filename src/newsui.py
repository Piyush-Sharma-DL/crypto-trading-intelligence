"""Streamlit pieces for showing headlines safely (used by the News, Live Chart and Bot Desk pages)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .news import md_escape


def ago(ts: pd.Timestamp, now: pd.Timestamp | None = None) -> str:
    now = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    m = max(0, int((now - ts).total_seconds() // 60))
    return f"{m}m ago" if m < 60 else f"{m // 60}h ago" if m < 1440 else f"{m // 1440}d ago"


def mood_dot(s: float) -> str:
    return "🟢" if s > 0.08 else "🔴" if s < -0.08 else "⚪"


def render_headlines(df: pd.DataFrame, n: int = 15, now: pd.Timestamp | None = None) -> None:
    """One line per headline. Titles are untrusted text, so they are escaped; links are pre-validated http(s) only."""
    if df is None or df.empty:
        st.caption("No headlines to show right now.")
        return
    for _, r in df.head(n).iterrows():
        title = md_escape(r["title"])
        link = f"[{title}]({r['link']})" if r["link"] else title
        flag = "⚠️ " if r["impact"] == 2 else ""
        tags = f" · `{', '.join(r['tags'])}`" if r["tags"] else ""
        st.markdown(f"{mood_dot(r['sentiment'])} {flag}**{md_escape(r['source'])}** · {ago(r['time'], now)} · {link}{tags}")
