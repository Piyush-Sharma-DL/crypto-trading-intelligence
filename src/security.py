"""Small security helpers: validate anything a visitor can type before it reaches an API or the page."""
from __future__ import annotations

import html
import re

_TICKER = re.compile(r"^(?:[A-Z0-9]{1,12}-[A-Z0-9]{2,6}|\^[A-Z0-9]{1,10})$")


def safe_ticker(text: str | None, default: str | None = None) -> str | None:
    """'btc-usd' -> 'BTC-USD'. Anything that is not a plain ticker (HTML, URLs, spaces, ...) -> default."""
    s = (text or "").strip().upper()
    return s if _TICKER.match(s) else default


def esc(text) -> str:
    """HTML-escape before putting text inside unsafe_allow_html markup."""
    return html.escape(str(text))
