"""Cryptocurrency Price Prediction & Trading Intelligence System.
Run:  streamlit run app.py
Pages: views/live.py (live chart) and views/forecast.py (ML forecast lab).
"""
import streamlit as st

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
</style>""", unsafe_allow_html=True)

pages = [
    st.Page("views/live.py", title="Live Chart", icon="📡", default=True),
    st.Page("views/forecast.py", title="Forecast Lab", icon="🧪"),
]
st.navigation(pages).run()
