"""Cryptocurrency Price Prediction & Trading Intelligence System.
Run:  streamlit run app.py
Pages: views/live.py (live chart) and views/forecast.py (ML forecast lab).
"""
import streamlit as st

from src import auth

st.set_page_config(page_title="Crypto Trading Intelligence", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1400px;}
[data-testid="stMetric"] {background: rgba(148,163,184,.08); border: 1px solid rgba(148,163,184,.18);
    padding: .7rem 1rem; border-radius: 12px;}
.badge {display:inline-block; padding:.35rem 1.2rem; border-radius:999px; font-weight:800;
        font-size:1.6rem; letter-spacing:.07em; color:#0b1220;}
.subtle {color:#94a3b8; font-size:.85rem;}
.hero {font-size:2.1rem; font-weight:800; margin-bottom:0;}
/* metric cards shrink with the window instead of being cut off */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {font-size: clamp(1.05rem, 2.1vw, 2.2rem) !important;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {font-size: clamp(.62rem, 1vw, .9rem) !important; white-space: normal !important;}
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] div {font-size: clamp(.6rem, .9vw, .85rem) !important;}
</style>""", unsafe_allow_html=True)

pages = [
    st.Page("views/live.py", title="Live Chart", icon="📡", default=True),
    st.Page("views/news.py", title="News & Market", icon="📰"),
    st.Page("views/forecast.py", title="Forecast Lab", icon="🧪"),
]
# the private page is only registered for the signed-in owner; everyone else gets a sign-in page instead
if auth.is_owner():
    pages.append(st.Page("views/bot.py", title="Bot Desk", icon="🤖"))
else:
    pages.append(st.Page("views/login.py", title="Owner sign-in", icon="🔐"))
st.navigation(pages).run()
