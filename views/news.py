"""News & Market page (public): headlines, market mood and context, synced to the Live Chart's asset."""
import pandas as pd
import streamlit as st

from src import news as N
from src.live import get_candles
from src.livechart import NEWS, live_chart
from src.newsui import mood_dot, render_headlines
from src.security import esc
from src.ui import asset_picker

COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD"]


@st.cache_data(ttl=300, show_spinner="Reading the latest headlines...")
def news_data():
    return N.fetch_news(max_age_hours=72)


@st.cache_data(ttl=300, show_spinner=False)
def context():
    return N.fetch_market_context()


@st.cache_data(ttl=120, show_spinner=False)
def hourly(ticker):
    return get_candles(ticker, "1h", 120)


with st.sidebar:
    st.title("📰 News controls")
    ticker = asset_picker(COINS)
    hours = st.slider("Headlines from the last (hours)", 6, 72, 24, 6)
    only_asset = st.toggle("Only this asset (plus notable market-wide news)", True)
    chosen = st.multiselect("Sources", list(N.SOURCES), default=list(N.SOURCES))
if ticker is None:
    st.error("That is not a valid ticker. Use Yahoo format, for example LTC-USD.")
    st.stop()
base = ticker.split("-")[0]
st.markdown(f"<p class='hero'>📰 {esc(base)} · News & Market</p>", unsafe_allow_html=True)


def pct(x):
    return f"{x:+.2f}%"


@st.fragment(run_every="300s")
def panel():
    ndf, nstat = news_data()
    ctx = context()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)

    # ---- market context
    k = st.columns(5)
    f = ctx.get("fng")
    k[0].metric("Fear & Greed", f"{f['value']:.0f} · {f['label']}" if f else "n/a",
                f"{f['value'] - f['prev']:+.0f} vs yesterday" if f and f.get("prev") is not None else None)
    g = ctx.get("global")
    k[1].metric("BTC dominance", f"{g['btc_dom']:.1f}%" if g else "n/a")
    k[2].metric("Total crypto market cap", f"${g['market_cap'] / 1e12:.2f}T" if g else "n/a", pct(g["cap_change_24h"]) if g else None)
    sp, dx = ctx.get("sp500"), ctx.get("dxy")
    k[3].metric("S&P 500", f"{sp['price']:,.0f}" if sp else "n/a", pct(sp["change"]) if sp else None)
    k[4].metric("US dollar index", f"{dx['price']:.2f}" if dx else "n/a", pct(dx["change"]) if dx else None)
    if ctx.get("errors"):
        st.caption("Some market data could not be loaded: " + ", ".join(ctx["errors"]))

    # ---- headlines for this asset
    df = ndf[(ndf["time"] >= now - pd.Timedelta(hours=hours)) & ndf["source"].isin(chosen)] if len(ndf) else ndf
    shown = N.for_asset(df, base) if only_asset else df
    mm = N.mood(shown)
    m = st.columns(4)
    m[0].metric(f"{base} headline mood", f"{mood_dot(mm['score'])} {mm['score']:+.2f}", mm["label"])
    m[1].metric("Headlines", f"{mm['n']}")
    m[2].metric("High-impact", f"{mm['high']}", "⚠️ expect sudden moves" if mm["high"] else None, delta_color="off")
    ok = sum(v.startswith("ok") for v in nstat.values())
    m[3].metric("Sources working", f"{ok} of {len(nstat)}")

    # ---- the same asset's price with headlines pinned on it
    try:
        cdl, src = hourly(ticker)
        if len(shown):
            fig = live_chart(cdl, 60, [NEWS], [], {}, [], float(cdl["Close"].iloc[-1]), "UTC", min(len(cdl), 96),
                             uirevision=f"news-{ticker}", news=shown)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            st.caption(f"{ticker} hourly candles ({src}). Diamonds = headlines, stars = high-impact. "
                       "Green/red = positive/negative wording, grey = neutral. Hover for the headline.")
    except Exception:
        st.caption("Price chart unavailable right now.")

    left, right = st.columns([3, 1])
    with left:
        st.markdown(f"**Headlines** ({'about ' + base + ' and notable market news' if only_asset else 'all crypto news'}, last {hours}h)")
        render_headlines(shown, 40, now)
    with right:
        st.markdown("**Trending on CoinGecko**")
        for t in ctx.get("trending", []):
            st.write("• " + N.md_escape(t))
        if not ctx.get("trending"):
            st.caption("n/a")
        with st.expander("Source status"):
            for name, s in nstat.items():
                st.write(f"{'✅' if s.startswith('ok') else '❌'} {N.md_escape(name)}: {N.md_escape(s)}")
    st.caption("Headline mood uses simple keyword rules on the title. It describes the wording of the news, it is not a price forecast, "
               "and keyword rules can misread sarcasm or context. Headlines come from public RSS feeds; open the source to read the full story. "
               "Nothing here is financial advice.")


panel()
