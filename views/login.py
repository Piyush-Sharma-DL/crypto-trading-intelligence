"""Owner sign-in page (visible to everyone, reveals nothing about the owner)."""
import streamlit as st

from src import auth

st.title("🔐 Owner area")
if auth.is_owner():
    st.success("You are signed in. Open **Bot Desk** in the sidebar.")
else:
    auth.login_ui()
