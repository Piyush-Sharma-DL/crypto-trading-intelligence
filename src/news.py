"""Crypto news and market context from FREE public sources (RSS feeds, CoinGecko, alternative.me, Yahoo). No API keys.

Everything fetched here is untrusted text from the internet, so: DTD/entity tricks are rejected, HTML is stripped,
only http(s) links are kept, and display helpers escape markdown. Sentiment and impact tags are simple keyword rules:
a rough *headline mood*, not a forecast.
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

SOURCES = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/",
    "CryptoSlate": "https://cryptoslate.com/feed/",
    "The Block": "https://www.theblock.co/rss.xml",
}
MAX_BYTES = 2_000_000
COLUMNS = ["time", "source", "title", "link", "summary", "assets", "tags", "impact", "sentiment"]

ASSET_WORDS = {
    "BTC": r"bitcoin|btc", "ETH": r"ethereum|ether|eth", "BNB": r"bnb|binance coin", "SOL": r"solana|sol",
    "XRP": r"xrp|ripple", "ADA": r"cardano|ada", "DOGE": r"dogecoin|doge", "AVAX": r"avalanche|avax",
    "LINK": r"chainlink|link", "DOT": r"polkadot|dot", "LTC": r"litecoin|ltc",
}
_ASSET_RE = {k: re.compile(rf"\b(?:{v})\b", re.I) for k, v in ASSET_WORDS.items()}

# impact 2 = shock-type or scheduled big-macro events; impact 1 = notable
_HIGH = re.compile(r"\b(hack\w*|exploit\w*|stolen|theft|bankrupt\w*|insolven\w*|bans?|banned|lawsuits?|sues?|sued|sec|cftc|"
                   r"crash\w*|plunge\w*|halts?|halted|delist\w*|depeg\w*|outage|seiz\w+|arrest\w*|fraud\w*|fomc|cpi|"
                   r"rate (?:cut|hike)s?|emergency|shut ?down)\b", re.I)
_MED = re.compile(r"\b(fed|interest rates?|inflation|regulat\w+|etfs?|approv\w+|adoption|partnership|upgrade|launch\w*|"
                  r"liquidat\w+|all-time high|record|treasury|tariffs?|stablecoin\w*)\b", re.I)
_TAGS = {
    "Security": r"hack\w*|exploit\w*|stolen|theft|outage|breach|depeg\w*",
    "Regulation": r"sec|cftc|ban\w*|lawsuit\w*|regulat\w+|compliance|court|sues?|sued",
    "Macro": r"fed|fomc|cpi|inflation|interest rates?|rate (?:cut|hike)s?|treasury|tariffs?|dollar|jobs report",
    "Institutional": r"etfs?|blackrock|fidelity|institution\w*|treasury reserve|grayscale",
    "Exchange": r"exchange|binance|coinbase|kraken|bybit|okx|delist\w*|listing",
    "Market": r"liquidat\w+|crash\w*|plunge\w*|rally|surge\w*|all-time high|sell-?off",
}
_TAG_RE = {k: re.compile(rf"\b(?:{v})\b", re.I) for k, v in _TAGS.items()}
_POS = re.compile(r"\b(surg\w*|soar\w*|jump\w*|rall(?:y|ies|ied)|gain\w*|climb\w*|record|bullish|approv\w*|adopt\w*|inflows?|"
                  r"upgrad\w*|breakout|recover\w*|rebound\w*|partner\w*|launch\w*|boost\w*|rises?|rose|wins?)\b", re.I)
_NEG = re.compile(r"\b(plung\w*|crash\w*|drops?|dropped|falls?|falling|fell|slump\w*|bearish|hack\w*|exploit\w*|stolen|bans?|"
                  r"banned|lawsuits?|sues?|sued|outflows?|liquidat\w*|fear|sell-?off|dump\w*|delay\w*|delist\w*|fraud\w*|"
                  r"declin\w*|loss(?:es)?|warn\w*|tumbl\w*|sink\w*|slide\w*|bankrupt\w*|insolven\w*|probe\w*|charged?)\b", re.I)


# ------------------------------------------------------------------ text safety
def clean_text(s: str | None, limit: int = 300) -> str:
    s = html.unescape(re.sub(r"<[^>]+>", " ", s or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def safe_url(url: str | None) -> str | None:
    """Keep only plain http(s) links; neutralise characters that could break a markdown link."""
    u = (url or "").strip()
    if not re.match(r"^https?://[^\s<>\"']+$", u, re.I):
        return None
    return u.replace("(", "%28").replace(")", "%29")


def md_escape(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~<$])", r"\\\1", str(text))


# ------------------------------------------------------------------ parsing
def _local(tag) -> str:
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def _parse_time(text: str | None) -> pd.Timestamp:
    if not text:
        return pd.NaT
    try:
        dt = pd.Timestamp(parsedate_to_datetime(text.strip()))
    except Exception:
        dt = pd.to_datetime(text.strip(), utc=True, errors="coerce")
    if pd.isna(dt):
        return pd.NaT
    return dt.tz_convert("UTC").tz_localize(None) if dt.tzinfo is not None else dt


def parse_feed(data: bytes | str, source: str) -> list[dict]:
    """RSS 2.0 or Atom -> list of raw items. Rejects DTDs/entities (XML bombs) outright."""
    text = data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else data
    if re.search(r"<!\s*(DOCTYPE|ENTITY)", text, re.I):
        raise ValueError("feed rejected: DTD/entity declarations are not allowed")
    root = ET.fromstring(text)
    items = []
    for el in root.iter():
        if _local(el.tag) not in ("item", "entry"):
            continue
        f = {}
        for child in el:
            name = _local(child.tag)
            if name == "link":
                f.setdefault("link", child.get("href") or (child.text or ""))
            elif name in ("title", "description", "summary", "pubDate", "published", "updated", "date"):
                f.setdefault(name, child.text or "")
        ts = _parse_time(f.get("pubDate") or f.get("published") or f.get("updated") or f.get("date"))
        title = clean_text(f.get("title"), 220)
        if pd.isna(ts) or not title:
            continue
        items.append({"time": ts, "source": source, "title": title, "link": safe_url(f.get("link")),
                      "summary": clean_text(f.get("description") or f.get("summary"), 300)})
    return items


# ------------------------------------------------------------------ enrichment
def score_sentiment(text: str) -> float:
    p, n = len(_POS.findall(text)), len(_NEG.findall(text))
    return 0.0 if p + n == 0 else (p - n) / (p + n)


def enrich(item: dict) -> dict:
    text = f"{item['title']} {item.get('summary', '')}"
    title = item["title"]
    assets = [a for a, rx in _ASSET_RE.items() if rx.search(title)] or [a for a, rx in _ASSET_RE.items() if rx.search(text)]
    impact = 2 if _HIGH.search(title) else 1 if _MED.search(title) else 0
    return {**item, "assets": assets, "tags": [t for t, rx in _TAG_RE.items() if rx.search(title)], "impact": impact,
            "sentiment": round(score_sentiment(title), 3)}


def _http_get_bytes(url: str, timeout: int = 10) -> bytes:
    import requests

    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (crypto-trading-intelligence news reader)"})
    r.raise_for_status()
    if len(r.content) > MAX_BYTES:
        raise ValueError("feed too large")
    return r.content


def fetch_news(sources: dict | None = None, max_age_hours: float = 72, now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """All feeds in parallel. A broken feed never breaks the others. Returns (headlines newest-first, per-source status)."""
    sources = sources or SOURCES
    now = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)

    def load(name_url):
        name, url = name_url
        try:
            return name, parse_feed(_http_get_bytes(url), name), None
        except Exception as exc:
            return name, [], f"{type(exc).__name__}: {str(exc)[:80]}"

    with ThreadPoolExecutor(max_workers=min(8, len(sources))) as pool:
        results = list(pool.map(load, sources.items()))
    rows, status = [], {}
    for name, items, err in results:
        status[name] = f"failed ({err})" if err else f"ok ({len(items)})"
        rows += [enrich(i) for i in items]
    if not rows:
        return pd.DataFrame(columns=COLUMNS), status
    df = pd.DataFrame(rows)[COLUMNS]
    df = df[(df["time"] >= now - pd.Timedelta(hours=max_age_hours)) & (df["time"] <= now + pd.Timedelta(minutes=5))]
    df = df.assign(_k=df["title"].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip())
    df = df.sort_values("time", ascending=False).drop_duplicates("_k").drop(columns="_k")
    return df.reset_index(drop=True), status


# ------------------------------------------------------------------ filtering / summaries
def for_asset(df: pd.DataFrame, base: str, include_market: bool = True) -> pd.DataFrame:
    """Headlines about `base`, plus (optionally) market-wide ones that are notable (impact >= 1, no specific coin)."""
    if df.empty:
        return df
    about = df["assets"].apply(lambda a: base in a)
    wide = df["assets"].apply(lambda a: len(a) == 0) & (df["impact"] >= 1)
    return df[about | (wide if include_market else False)]


def mood(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"score": 0.0, "n": 0, "high": 0, "label": "no headlines"}
    s = float(df["sentiment"].mean())
    label = "positive" if s > 0.25 else "mildly positive" if s > 0.08 else "negative" if s < -0.25 else \
        "mildly negative" if s < -0.08 else "mixed / neutral"
    return {"score": s, "n": len(df), "high": int((df["impact"] == 2).sum()), "label": label}


def news_risk(df: pd.DataFrame, symbol: str, now: pd.Timestamp, minutes: int) -> tuple[bool, str]:
    """Is there a recent high-impact headline about this coin (or the whole market)? Used by the bot's optional news guard."""
    if df is None or df.empty:
        return False, ""
    base = symbol.split("-")[0].upper()
    recent = df[(df["time"] >= now - pd.Timedelta(minutes=minutes)) & (df["time"] <= now + pd.Timedelta(minutes=5)) & (df["impact"] == 2)]
    recent = recent[recent["assets"].apply(lambda a: base in a or len(a) == 0)]
    if recent.empty:
        return False, ""
    r = recent.iloc[0]
    return True, f"{r['source']}: {r['title'][:90]}"


# ------------------------------------------------------------------ market context
def fng_label(v: float) -> str:
    return "Extreme fear" if v < 25 else "Fear" if v < 45 else "Neutral" if v <= 55 else "Greed" if v <= 75 else "Extreme greed"


def parse_global(payload: dict) -> dict:
    d = payload["data"]
    return {"market_cap": float(d["total_market_cap"]["usd"]), "volume": float(d["total_volume"]["usd"]),
            "btc_dom": float(d["market_cap_percentage"]["btc"]), "eth_dom": float(d["market_cap_percentage"].get("eth", np.nan)),
            "cap_change_24h": float(d["market_cap_change_percentage_24h_usd"])}


def parse_trending(payload: dict, n: int = 7) -> list[str]:
    out = []
    for c in payload.get("coins", [])[:n]:
        it = c.get("item", {})
        out.append(f"{clean_text(it.get('name'), 30)} ({clean_text(it.get('symbol'), 10).upper()})")
    return out


def _get_json(url: str, timeout: int = 10) -> dict:
    import requests

    r = requests.get(url, timeout=timeout, headers={"User-Agent": "crypto-trading-intelligence"})
    r.raise_for_status()
    return r.json()


def fetch_market_context() -> dict:
    """Fear & Greed, global market stats, trending coins, S&P 500 and dollar index. Each part is optional."""
    out, errors = {}, []

    def part(name, fn):
        try:
            out[name] = fn()
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}")

    from .data import fetch_close, fetch_fng

    def fng():
        s = fetch_fng()
        return {"value": float(s.iloc[-1]), "label": fng_label(float(s.iloc[-1])), "prev": float(s.iloc[-2]) if len(s) > 1 else None}

    def idx(ticker):
        s = fetch_close(ticker, 1).dropna()
        return {"price": float(s.iloc[-1]), "change": float((s.iloc[-1] / s.iloc[-2] - 1) * 100)}

    part("fng", fng)
    part("global", lambda: parse_global(_get_json("https://api.coingecko.com/api/v3/global")))
    part("trending", lambda: parse_trending(_get_json("https://api.coingecko.com/api/v3/search/trending")))
    part("sp500", lambda: idx("^GSPC"))
    part("dxy", lambda: idx("DX-Y.NYB"))
    out["errors"] = errors
    return out
