"""Small UI helpers shared by the pages (asset choice is remembered across pages)."""
from __future__ import annotations

from .security import safe_ticker


def asset_picker(coins: list[str], default_custom: str = "LTC-USD") -> str | None:
    """Asset selector that remembers the choice when you move between Live Chart, News and Bot Desk. Returns a validated ticker."""
    import streamlit as st

    cur = st.session_state.get("shared_asset", coins[0])
    options = list(coins) + ["Custom..."]
    preset = st.selectbox("Asset", options, index=coins.index(cur) if cur in coins else len(coins))
    custom = st.text_input("Custom ticker (Yahoo format, e.g. LTC-USD)", cur if cur not in coins else default_custom)
    ticker = safe_ticker(custom if preset == "Custom..." else preset)
    if ticker:
        st.session_state["shared_asset"] = ticker
    return ticker
