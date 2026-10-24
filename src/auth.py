"""Owner-only access. The site is public; everything behind `require_owner()` is private to you.

Two ways to prove you are the owner (both read from Streamlit *secrets*, never from the code or GitHub):
  1. Google sign-in (recommended): needs [auth] settings + OWNER_EMAIL.  Streamlit verifies the identity with Google.
  2. Password fallback: needs OWNER_PASSWORD_HASH (a salted PBKDF2 hash made with `python -m src.auth hash`).
If neither is configured, owner pages stay LOCKED (fail closed).
"""
from __future__ import annotations

import hashlib
import hmac
import secrets as _secrets
import threading
import time
from collections import deque

ITERATIONS = 600_000
IDLE_SECONDS, ABSOLUTE_SECONDS = 30 * 60, 8 * 3600


# ------------------------------------------------------------------ password hashing (pure, testable)
def hash_password(password: str, iterations: int = ITERATIONS, salt: bytes | None = None) -> str:
    salt = salt or _secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt, digest = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(dk, bytes.fromhex(digest))
    except Exception:                                   # malformed hash -> deny, never crash open
        return False


class Throttle:
    """Process-wide brute-force brake: too many failures in a window locks password sign-in for everyone for a while."""
    def __init__(self, max_fails=8, window=600, lock_for=900, clock=time.time):
        self.max_fails, self.window, self.lock_for, self.clock = max_fails, window, lock_for, clock
        self.fails, self.locked_until, self._mu = deque(), 0.0, threading.Lock()

    def locked(self) -> bool:
        with self._mu:
            return self.clock() < self.locked_until

    def fail(self) -> None:
        with self._mu:
            now = self.clock()
            self.fails.append(now)
            while self.fails and now - self.fails[0] > self.window:
                self.fails.popleft()
            if len(self.fails) >= self.max_fails:
                self.locked_until = now + self.lock_for
                self.fails.clear()

    def ok(self) -> None:
        with self._mu:
            self.fails.clear()


THROTTLE = Throttle()


# ------------------------------------------------------------------ Streamlit side (lazy import keeps the helpers testable)
def _st():
    import streamlit as st
    return st


def _secret(key: str):
    try:
        return _st().secrets.get(key)
    except Exception:                                   # no secrets file at all
        return None


def auth_mode() -> str:
    st = _st()
    try:
        has_oidc = "auth" in st.secrets and hasattr(st, "login")
    except Exception:
        has_oidc = False
    if has_oidc and _secret("OWNER_EMAIL"):
        return "google"
    if _secret("OWNER_PASSWORD_HASH"):
        return "password"
    return "none"


def is_owner() -> bool:
    st, mode = _st(), auth_mode()
    if mode == "google":
        try:
            user = getattr(st, "user", None) or getattr(st, "experimental_user", None)
            if user is None or not user.is_logged_in:
                return False
            email = str(getattr(user, "email", "") or "").strip().lower()
            verified = getattr(user, "email_verified", None)
            return bool(email) and verified is not False and email == str(_secret("OWNER_EMAIL")).strip().lower()
        except Exception:
            return False
    if mode == "password":
        now = time.time()
        s = st.session_state
        if s.get("_owner_idle_until", 0) > now and s.get("_owner_abs_until", 0) > now:
            s["_owner_idle_until"] = now + IDLE_SECONDS          # sliding idle timeout
            return True
        for k in ("_owner_idle_until", "_owner_abs_until"):
            s.pop(k, None)
    return False


def require_owner() -> None:
    """Call at the top of every private page. Stops the page for anyone who is not the owner."""
    st = _st()
    if not is_owner():
        st.error("🔒 Owner access only.")
        st.stop()


def logout() -> None:
    st = _st()
    for k in ("_owner_idle_until", "_owner_abs_until"):
        st.session_state.pop(k, None)
    if auth_mode() == "google":
        st.logout()


def login_ui() -> None:
    """Sign-in screen shown to visitors. Reveals nothing about the owner."""
    st, mode = _st(), auth_mode()
    st.subheader("🔐 Owner sign-in")
    if mode == "google":
        user = getattr(st, "user", None) or getattr(st, "experimental_user", None)
        if user is not None and getattr(user, "is_logged_in", False):
            st.error("This Google account is not authorised for the owner area.")
            if st.button("Sign out"):
                st.logout()
        else:
            st.caption("Private area: paper-trading bot and settings. Only the owner can enter.")
            if st.button("Sign in with Google", type="primary"):
                st.login()
    elif mode == "password":
        st.caption("Private area. Only the owner can enter.")
        with st.form("owner_login"):
            pw = st.text_input("Password", type="password")
            go = st.form_submit_button("Sign in")
        if go:
            if THROTTLE.locked():
                st.error("Too many failed attempts. Please try again later.")
            elif pw and verify_password(pw, str(_secret("OWNER_PASSWORD_HASH"))):
                THROTTLE.ok()
                now = time.time()
                st.session_state["_owner_idle_until"] = now + IDLE_SECONDS
                st.session_state["_owner_abs_until"] = now + ABSOLUTE_SECONDS
                st.rerun()
            else:
                THROTTLE.fail()
                time.sleep(1.0)                                  # slow down guessing
                st.error("Incorrect password.")
    else:
        st.warning("Owner sign-in is not set up yet, so the private area is locked. "
                   "See OWNER_SETUP.md in the project for the 10-minute setup.")


if __name__ == "__main__":                              # python -m src.auth hash
    import getpass
    import sys
    if sys.argv[1:] == ["hash"]:
        a, b = getpass.getpass("New owner password (12+ characters): "), getpass.getpass("Repeat: ")
        if a != b or len(a) < 12:
            raise SystemExit("Passwords differ or are shorter than 12 characters.")
        print("\nPaste this line into your Streamlit secrets:\n")
        print(f'OWNER_PASSWORD_HASH = "{hash_password(a)}"')
    else:
        print("usage: python -m src.auth hash")
