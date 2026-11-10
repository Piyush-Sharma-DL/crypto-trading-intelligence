"""Dashboard for the Irfan Smart Market Structure indicator."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

COLORS = {"STRONG BULLISH": "#16a34a", "BULLISH": "#22c55e", "NEUTRAL": "#eab308", "BEARISH": "#ef4444", "STRONG BEARISH": "#b91c1c"}


def render_sms_dashboard(rows: pd.DataFrame, score: int, text: str) -> None:
    """Bias badge + the indicator's table. All text is escaped."""
    color = COLORS.get(text, "#94a3b8")
    st.markdown(f"**IRFAN SMS v7.1 dashboard** &nbsp; <span style='background:{color};color:#fff;padding:.12rem .6rem;border-radius:4px;"
                f"font-weight:800'>{html.escape(text)} ({score:+d})</span>", unsafe_allow_html=True)
    st.dataframe(rows, hide_index=True, use_container_width=True)
