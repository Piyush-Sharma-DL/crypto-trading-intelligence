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


# ---------------------------------------------------------------- remembered settings
def seed_settings(schema: dict, prefix: str, name: str, store=None) -> None:
    """Call BEFORE creating the widgets. Priority: this session > page URL > owner's saved profile > defaults.
    Widget values are put into session_state under `prefix + key`."""
    import streamlit as st
    from . import settings as S

    ss, pkey = st.session_state, f"_profile_{name}"
    if pkey not in ss:
        raw = None
        try:
            raw = S.decode(st.query_params.get("s"))
        except Exception:
            pass
        if raw is None and store is not None:
            try:
                raw = store.get_setting(name)
            except Exception:
                pass
        ss[pkey] = S.clean(raw, schema)
    prof = ss[pkey]
    for k, v in prof.items():
        if k != "asset" and f"{prefix}{k}" not in ss:
            ss[f"{prefix}{k}"] = v
    if "shared_asset" not in ss and prof.get("asset"):
        ss["shared_asset"] = prof["asset"]


def save_settings(schema: dict, prefix: str, name: str, store=None) -> None:
    """Call AFTER the widgets. Stores the current values in the session, the page URL and (owner only) the private database."""
    import streamlit as st
    from . import settings as S

    ss, pkey = st.session_state, f"_profile_{name}"
    cur = {k: ss.get(f"{prefix}{k}") for k in schema if k != "asset"}
    cur["asset"] = ss.get("shared_asset")
    cur = S.clean(cur, schema)
    if cur != ss.get(pkey):
        ss[pkey] = cur
        if store is not None:
            try:
                store.set_setting(name, cur)
            except Exception:
                pass
    enc = S.encode(cur)
    try:
        if st.query_params.get("s") != enc:
            st.query_params["s"] = enc
    except Exception:
        pass


def reset_settings(schema: dict, prefix: str, name: str, store=None) -> None:
    import streamlit as st

    for k in schema:
        st.session_state.pop(f"{prefix}{k}", None)
    st.session_state.pop(f"_profile_{name}", None)
    st.session_state.pop("shared_asset", None)
    if store is not None:
        try:
            store.delete_setting(name)
        except Exception:
            pass
    try:
        st.query_params.clear()
    except Exception:
        pass


def worker_status(rec: dict, now=None) -> tuple[bool, str]:
    """Is the always-on background worker looking after this bot? (page checks only count while a page is open.)"""
    import pandas as pd

    now = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    last = rec.get("last_worker")
    if not last:
        return False, "Background worker: never seen. The bot only runs while a page is open."
    age = (now - pd.Timestamp(last)).total_seconds()
    if age <= 180:
        return True, f"Background worker: running (last check {age:.0f}s ago)."
    return False, f"Background worker: last seen {age / 60:.0f} min ago, so it is not running now."
