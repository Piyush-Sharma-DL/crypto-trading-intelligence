"""Live Chart page: real-time price, multi-timeframe candles and user-selectable indicators.
Default chart layers: the combined Confluence indicator and news markers. Owners also get their paper bot on the chart."""
import numpy as np
import pandas as pd
import streamlit as st

from src import indicators as I
from src.live import TIMEFRAMES, candle_seconds_left, get_candles, get_quote
from src.confluence import ConfParams, confluence, status_rows
from src import auth
from src import news as N
from src.bot import kill_switch, run_cycle
from src import settings as S
from src.imba import imba_algo, status_row as imba_status
from src.imba import ladder as imba_ladder_for
from src.livechart import CONF, IMBA_ALGO, NEWS, live_chart
from src.newsui import mood_dot, render_headlines
from src.paper_engine import BotConfig, new_state
from src.paperstore import Store
from src.security import esc, safe_ticker
from src.ui import asset_picker, reset_settings, save_settings, seed_settings, worker_status

COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD"]
FVG, OB = "Imbalance zones (FVG)", "Order Blocks"
OVERLAYS = [CONF, NEWS, IMBA_ALGO, "TOTT", FVG, OB, "SMA 20", "SMA 50", "EMA 21", "EMA 200", "Bollinger Bands", "VWAP", "Supertrend"]
PANELS = ["Volume", "RSI", "MACD"]
BOT_TFS = ["5m", "15m", "30m", "1h", "4h"]
ZONES = ["UTC", "Asia/Kolkata", "America/New_York", "Europe/London", "Asia/Dubai", "Asia/Tokyo", "Australia/Sydney"]
OB_ZONES = ["Candle body", "Full candle (with wicks)"]
CHART_CONFIG = {"scrollZoom": True, "displaylogo": False,                     # TradingView-style drawing tools in the toolbar
                "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "drawcircle", "eraseshape"],
                "toImageButtonOptions": {"format": "png", "scale": 2}}

# every setting the user can change, with its default and what counts as a valid value (also used to restore saved settings)
SCHEMA = {
    "asset": ("BTC-USD", lambda v: isinstance(v, str) and safe_ticker(v) == v),
    "tf": ("15m", S.choice(TIMEFRAMES)), "bars": (150, S.choice(range(50, 401, 10))), "tz": ("UTC", S.choice(ZONES)),
    "auto": (True, S.flag()), "every": (15, S.choice([5, 10, 15, 30, 60])),
    "overlays": ([CONF, NEWS], S.multi(OVERLAYS)), "panels": ([], S.multi(PANELS)),
    "tott_len": (40, S.num(2, 200)), "tott_pct": (1.0, S.num(0.1, 10.0)), "tott_coeff": (0.001, S.num(0.0, 0.05)),
    "c_stop": (0.5, S.num(0.0, 3.0)), "c_rr": (2.0, S.num(0.5, 10.0)), "c_min": (4, S.num(4, 6)),
    "imba_sens": (18.0, S.num(1.0, 60.0)), "imba_sl": (0.0, S.num(0.0, 30.0)), "imba_fixed": (False, S.flag()),
    "fvg_min": (0.25, S.num(0.0, 3.0)), "ob_swing": (10, S.num(3, 50)), "ob_disp": (1.5, S.num(0.5, 5.0)),
    "ob_zone": ("Candle body", S.choice(OB_ZONES)), "max_zones": (3, S.num(1, 15)),
    "show_mit": (False, S.flag()), "show_old": (False, S.flag()), "show_vol": (True, S.flag()),
}


# ---------------------------------------------------------------- short-lived caches (shared by visitors)
@st.cache_data(ttl=4, show_spinner=False)
def candles(ticker, tf, n):
    return get_candles(ticker, tf, n)


@st.cache_data(ttl=3, show_spinner=False)
def quote(ticker, tf, df):
    return get_quote(ticker, tf, df)


@st.cache_data(ttl=300, show_spinner=False)
def news_data():
    return N.fetch_news(max_age_hours=72)


def bot_panel(store: Store, bot_id: str, ticker: str, tf: str, price: float) -> None:
    """Owner-only control card for the paper bot that trades this chart (fake money)."""
    st.markdown(f"#### 🤖 Paper bot on this chart · `{bot_id}` · owner only, fake money")
    if tf not in BOT_TFS:
        st.caption("Bots run on 5m, 15m, 30m, 1h or 4h candles. Pick one of those on the left.")
        return
    rec = store.get(bot_id)
    if rec is None:
        st.caption("No bot on this chart yet. It starts with 10,000 fake dollars, trades the Confluence signal on closed candles only, "
                   "and has a daily-loss limit and a kill switch.")
        if st.button("➕ Create paper bot for this chart", key=f"mk_{bot_id}"):
            cfg0 = BotConfig(symbol=ticker, timeframe=tf)
            store.create(bot_id, cfg0.to_dict(), new_state(cfg0))
            rec = store.get(bot_id)
        else:
            return
    cfg, s = BotConfig.from_dict(rec["cfg"]), rec["state"]
    c1, c2, c3 = st.columns(3)
    if c1.button("⏸ Pause bot" if s["enabled"] else "▶ Resume bot", key=f"tg_{bot_id}", use_container_width=True):
        s["enabled"] = not s["enabled"]
        if s["enabled"]:
            s["last_bar"] = None                                       # re-arm: never trade candles missed while paused
            if s["halt"] == "manual kill switch":
                s["halt"] = None
        store.save_state(bot_id, s)
        store.log(bot_id, "info", "Resumed" if s["enabled"] else "Paused")
    if c2.button("🛑 Kill switch", key=f"ks_{bot_id}", use_container_width=True, help="Close any open position and stop the bot"):
        kill_switch(store, bot_id, price)
    c3.page_link("views/bot.py", label="Open Bot Desk →")
    rec = store.get(bot_id)
    cfg, s = BotConfig.from_dict(rec["cfg"]), rec["state"]
    pos = s["position"]
    ok, why = worker_status(rec)
    if s["enabled"]:
        (st.caption if ok else st.warning)(("✅ " if ok else "⚠️ ") + why + ("" if ok else
              " To keep it running with the website closed, start **Start-Bot-Worker.bat** on your computer."))
    k = st.columns(4)
    k[0].metric("Equity (fake)", f"${s['equity']:,.2f}", f"{(s['equity'] / cfg.start_equity - 1) * 100:+.2f}% since start")
    k[1].metric("Status", f"🛑 {s['halt']}" if s["halt"] else ("▶ Running" if s["enabled"] else "⏸ Paused"))
    k[2].metric("Position", ("LONG" if pos["side"] == 1 else "SHORT") if pos else "none")
    k[3].metric("Unrealised", f"{pos['side'] * (price - pos['entry']) * pos['qty']:+,.2f}" if pos else "n/a")
    st.dataframe(store.events(bot_id, 5), hide_index=True, use_container_width=True)


def countdown(sec: int) -> str:
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return (f"{d}d " if d else "") + f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------- sidebar (settings are remembered)
owner_store = Store() if auth.is_owner() else None          # visitors never touch the database
seed_settings(SCHEMA, "lv_", "live_profile", owner_store)
with st.sidebar:
    st.title("📡 Live controls")
    ticker = asset_picker(COINS)
    tf = st.radio("Candle size", list(TIMEFRAMES), horizontal=True, key="lv_tf")
    bars = st.slider("Candles on chart", 50, 400, step=10, key="lv_bars")
    tz = st.selectbox("Display time zone", ZONES, key="lv_tz")

    st.subheader("Live updates")
    auto = st.toggle("Auto-refresh", key="lv_auto")
    every = st.select_slider("Refresh every (seconds)", [5, 10, 15, 30, 60], key="lv_every")

    st.subheader("Indicators")
    overlays = st.multiselect("On the chart", OVERLAYS, key="lv_overlays",
                              help="Default: the combined Confluence indicator and news markers. Add the real IMBA ALGO, "
                                   "TOTT, order blocks or any classic indicator.")
    panels = st.multiselect("Extra panels below", PANELS, key="lv_panels")
    show_vol = st.checkbox("Volume bars on the price chart", key="lv_show_vol")
    with st.expander("Indicator settings"):
        st.markdown("**IMBA ALGO** (fib-channel trend)")
        imba_sens = st.number_input("Sensitivity (channel = sensitivity x 10 candles)", 1.0, 60.0, step=0.5, key="lv_imba_sens")
        imba_fixed = st.checkbox("Fixed stop % from entry (instead of the 78.6% / 23.6% level)", key="lv_imba_fixed")
        imba_sl = st.number_input("Stop buffer / fixed stop %", 0.0, 30.0, step=0.1, key="lv_imba_sl")
        st.markdown("**TOTT** (Twin Optimized Trend Tracker)")
        tott_len = st.number_input("OTT period", 2, 200, key="lv_tott_len")
        tott_pct = st.number_input("Optimization constant (%)", 0.1, 10.0, step=0.1, key="lv_tott_pct")
        tott_coeff = st.number_input("Twin coefficient", 0.0, 0.05, step=0.0005, format="%.4f", key="lv_tott_coeff")
        st.markdown("**Confluence signal**")
        c_stop = st.number_input("Stop buffer beyond the zone (x ATR)", 0.0, 3.0, step=0.1, key="lv_c_stop")
        c_rr = st.number_input("Reward : risk target", 0.5, 10.0, step=0.5, key="lv_c_rr")
        c_min = st.number_input("Minimum confidence (4-6)", 4, 6, key="lv_c_min")
        st.markdown("**Imbalance zones** (fair value gaps)")
        fvg_min = st.number_input("Minimum gap size (x ATR)", 0.0, 3.0, step=0.05, key="lv_fvg_min")
        st.markdown("**Order blocks**")
        ob_swing = st.number_input("Structure lookback (candles)", 3, 50, key="lv_ob_swing")
        ob_disp = st.number_input("Impulse size (x ATR)", 0.5, 5.0, step=0.1, key="lv_ob_disp")
        ob_zone = st.selectbox("Order block zone covers", OB_ZONES, key="lv_ob_zone")
        max_zones = st.number_input("Max zones shown per type", 1, 15, key="lv_max_zones")
        show_mit = st.checkbox("Also show mitigated zones", key="lv_show_mit")
        show_old = st.checkbox("Show zones that formed before the visible candles", key="lv_show_old")
    st.caption("💾 Your settings are remembered automatically" + (" (saved to your private profile)." if owner_store else
               " (kept in this page's web address: bookmark it to keep them)."))
    if st.button("↺ Reset chart settings"):
        reset_settings(SCHEMA, "lv_", "live_profile", owner_store)
        st.rerun()
save_settings(SCHEMA, "lv_", "live_profile", owner_store)

if ticker is None:
    st.error("That is not a valid ticker. Use Yahoo format, for example LTC-USD.")
    st.stop()
cfg = dict(tott_length=int(tott_len), tott_percent=float(tott_pct), tott_coeff=float(tott_coeff),
           fvg_min_atr=float(fvg_min), ob_swing=int(ob_swing), ob_disp=float(ob_disp), ob_body=(ob_zone == "Candle body"),
           max_zones=int(max_zones))

st.markdown(f"<p class='hero'>📡 {esc(ticker)} · Live</p>", unsafe_allow_html=True)


# ---------------------------------------------------------------- the live panel (re-runs on its own timer)
@st.fragment(run_every=f"{every}s" if auto else None)
def live_panel():
    need = min(bars + 250, 260) if tf == "1W" else bars + 250        # extra candles warm the indicators up
    try:
        df, source = candles(ticker, tf, need)
    except Exception as exc:
        st.error(f"Could not load live data for {ticker}: {exc}")
        return
    q, qsrc = quote(ticker, tf, df)
    price = q["price"]
    if qsrc == "Kraken":                                              # make the forming candle match the latest trade
        df = df.copy()
        df.iloc[-1, df.columns.get_loc("Close")] = price
        df.iloc[-1, df.columns.get_loc("High")] = max(df["High"].iloc[-1], price)
        df.iloc[-1, df.columns.get_loc("Low")] = min(df["Low"].iloc[-1], price)

    chg = (price / q["open"] - 1) * 100
    m = st.columns([1.5, 1, 1, 1, 1.2])
    m[0].metric(f"{ticker} live price", f"${price:,.2f}", f"{chg:+.2f}% vs {q['open_label']}")
    m[1].metric(f"{q['window']} high", f"${q['high']:,.2f}")
    m[2].metric(f"{q['window']} low", f"${q['low']:,.2f}")
    m[3].metric(f"{q['window']} volume", f"{q['volume']:,.0f}")
    m[4].metric(f"{tf} candle closes in", countdown(candle_seconds_left(df.index[-1], tf)))
    now = pd.Timestamp.now(tz="UTC")
    st.caption(f"Candles: **{source}** · price: {qsrc} · updated {now:%H:%M:%S} UTC · "
               + (f"auto-refresh every {every}s" if auto else "auto-refresh off")
               + f" · chart times in {tz}")

    # ---- news (public) and the owner's bot (private): both are optional layers on the chart
    ndf, nstat = news_data()
    base = ticker.split("-")[0]
    asset_news = N.for_asset(ndf, base)
    utc_now = now.tz_localize(None)
    hot = asset_news[(asset_news["time"] >= utc_now - pd.Timedelta(hours=2)) & (asset_news["impact"] == 2)]
    if len(hot):
        st.warning(f"⚠️ High-impact headline in the last 2 hours: {N.md_escape(hot.iloc[0]['title'])} "
                   f"({N.md_escape(hot.iloc[0]['source'])}). Expect sudden moves.")

    def news_for_guard():
        if nstat and all(v.startswith("failed") for v in nstat.values()):
            raise RuntimeError("all news sources unreachable")
        return ndf

    owner = auth.is_owner()
    store, bot_overlay = None, None
    if owner and tf in BOT_TFS:
        store = Store()
        rec = store.get(f"{ticker}@{tf}")
        if rec and rec["state"]["enabled"]:
            try:
                run_cycle(store, f"{ticker}@{tf}", fetch=lambda s_, t_, n_: (df, source), news_fetch=news_for_guard, runner="page")
            except Exception as exc:
                st.warning(f"The bot could not be checked this time ({type(exc).__name__}). It will try again on the next refresh.")
            rec = store.get(f"{ticker}@{tf}")
        if rec:
            bot_overlay = {"trades": store.trades(f"{ticker}@{tf}"), "position": rec["state"]["position"]}

    ind = I.compute_all(df, TIMEFRAMES[tf], cfg)
    closed = df.iloc[:-1]                                             # signals use closed candles only (no repainting)
    cp = ConfParams(tott_length=cfg["tott_length"], tott_percent=cfg["tott_percent"], tott_coeff=cfg["tott_coeff"],
                    fvg_min_atr=cfg["fvg_min_atr"], ob_swing=cfg["ob_swing"], ob_disp=cfg["ob_disp"], ob_body=cfg["ob_body"],
                    stop_atr=float(c_stop), rr=float(c_rr), min_conf=int(c_min))
    imba_res = imba_algo(closed, sensitivity=float(imba_sens), sl_pct=float(imba_sl), fixed_stop=bool(imba_fixed))
    imba_full = imba_res.reindex(df.index)
    imba_full["signal"] = imba_full["signal"].fillna(0)
    conf_res = confluence(closed, cp)
    conf_full = conf_res.reindex(df.index)
    conf_full["signal"] = conf_full["signal"].fillna(0)
    zones = []
    if FVG in overlays or CONF in overlays:
        zones += I.keep_recent(ind["fvg"], cfg["max_zones"], show_mit)
    if OB in overlays or CONF in overlays:
        zones += I.keep_recent(ind["ob"], cfg["max_zones"], show_mit)
    fig = live_chart(df, TIMEFRAMES[tf], overlays, panels, ind, zones, price, tz, bars, uirevision=f"{ticker}-{tf}",
                     show_old_zones=show_old, conf=conf_full, news=asset_news, bot=bot_overlay,
                     imba=imba_full, imba_ladder=imba_ladder_for(closed, imba_res), title=f"{ticker} · {tf}",
                     source=source.split(" (")[0], show_volume=bool(show_vol))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    if owner:
        bot_panel(store or Store(), f"{ticker}@{tf}", ticker, tf, price)

    left, right = st.columns([3, 2])
    with left:
        snap, summ = I.snapshot(df, ind)
        day_news = asset_news[asset_news["time"] >= utc_now - pd.Timedelta(hours=24)]
        mm = N.mood(day_news)
        news_row = pd.DataFrame([{"Indicator": f"News mood ({base}, 24h)", "Value": f"{mm['score']:+.2f}",
                                  "Reading": f"{mood_dot(mm['score'])} {mm['label']} · {mm['n']} headlines · {mm['high']} high-impact"}])
        snap = pd.concat([status_rows(closed, conf_res, cp), imba_status(imba_res, price), news_row, snap], ignore_index=True)
        st.markdown(f"**Live indicator tracker** · 🟢 {summ['bull']} bullish · 🔴 {summ['bear']} bearish · ⚪ {summ['neutral']} neutral")
        st.dataframe(snap, hide_index=True, use_container_width=True)
    with right:
        st.markdown("**Nearest unmitigated zones**")
        zt = I.active_zones(ind["fvg"] + ind["ob"], price, len(df)).head(8)
        zt["Distance %"] = zt["Distance %"].round(2)
        st.dataframe(zt, hide_index=True, use_container_width=True)
    st.caption("Indicator values include the candle that is still forming, so they can change until it closes.")
    with st.expander(f"📰 Latest {base} headlines (synced with this chart)"):
        render_headlines(asset_news, 6, utc_now)
        st.page_link("views/news.py", label="Open the full News dashboard →")


live_panel()

with st.expander("What do these indicators mean?"):
    st.markdown("""
**TOTT, Twin Optimized Trend Tracker** (by Anil Ozeksi). A trend follower built from a smoothed support line (VAR moving average)
and a trailing stop (OTT). The two OTT lines sit slightly above and below it. **BUY** when the support line crosses above the upper
line, **SELL** when it crosses below the lower line; between the lines is a flat zone. Defaults (period 40, 1%) suit slower charts, so tune them
for 5-minute candles.

**IMBA ALGO (fib-channel trend).** Takes the highest high and lowest low of the last sensitivity x 10 candles and draws Fibonacci levels
between them. The 50% line is the trend line. **LONG** when a candle closes above the 50% line and inside the top 23.6% of the channel;
**SHORT** when it closes below the 50% line and inside the bottom 21.4%. The suggested stop sits at the 78.6% (long) or 23.6% (short) level,
which is usually far away, and targets are +1/2/3/4% from entry. Signals use closed candles only.

**Imbalance zones (FVG).** Three candles where the 1st candle's high and the 3rd candle's low do not overlap (or the reverse),
leaving a price gap. Price often revisits it. A green zone is a bullish gap, red is bearish, and it disappears once price trades fully through it.

**Order blocks.** The last opposite-colour candle before a strong impulsive move that breaks recent structure. Blue = bullish block (possible support),
amber = bearish block (possible resistance). A block is mitigated once a candle closes through it.

*These are analysis tools, not predictions. Nothing here is financial advice.*
""")
