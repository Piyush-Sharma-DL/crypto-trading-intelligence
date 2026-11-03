"""Streamlit pieces for showing headlines safely (used by the News, Live Chart and Bot Desk pages)."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from .news import md_escape, safe_url


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


IMPACT = {2: ("HIGH", "#ef4444"), 1: ("MEDIUM", "#f59e0b"), 0: ("LOW", "#facc15")}     # red = danger, orange = minor, yellow = normal


def render_news_strip(df: pd.DataFrame, tz: str = "UTC", n: int = 12, now: pd.Timestamp | None = None) -> None:
    """Forex-Factory-style list under the chart: time, colour-coded impact, headline, source. Every text is escaped."""
    if df is None or df.empty:
        st.caption("No headlines right now.")
        return
    now = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    rows = []
    for _, r in df.head(n).iterrows():
        label, color = IMPACT.get(int(r["impact"]), IMPACT[0])
        t = pd.Timestamp(r["time"]).tz_localize("UTC").tz_convert(tz)
        today = (now.tz_localize("UTC").tz_convert(tz)).date() == t.date()
        when = f"{t:%H:%M}" if today else f"{t:%d %b %H:%M}"
        url = safe_url(r["link"])
        title = html.escape(str(r["title"]))
        head = (f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>' if url else title)
        mood = "▲" if r["sentiment"] > 0.08 else "▼" if r["sentiment"] < -0.08 else "•"
        mcol = "#22c55e" if r["sentiment"] > 0.08 else "#ef4444" if r["sentiment"] < -0.08 else "#94a3b8"
        rows.append(f'<tr><td class="ffT">{html.escape(when)}<br><span class="ffA">{html.escape(ago(r["time"], now))}</span></td>'
                    f'<td><span class="ffChip" style="background:{color}">{label}</span></td>'
                    f'<td>{head}</td><td class="ffS">{html.escape(str(r["source"]))}</td>'
                    f'<td style="color:{mcol};text-align:center">{mood}</td></tr>')
    st.markdown('<table class="ff"><thead><tr><th>Time (' + html.escape(tz) + ')</th><th>Impact</th><th>Headline</th><th>Source</th>'
                '<th>Mood</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>"
                '<div class="ffLegend"><span class="ffChip" style="background:#ef4444">HIGH</span> danger: shocks, bans, hacks, big macro '
                '&nbsp; <span class="ffChip" style="background:#f59e0b">MEDIUM</span> minor impact &nbsp; '
                '<span class="ffChip" style="background:#facc15;color:#222">LOW</span> normal news. Mood ▲ positive ▼ negative wording.</div>',
                unsafe_allow_html=True)
