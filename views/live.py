"""Live Chart page. The chart is drawn once and STAYS PUT (so your zoom survives); the live price, bot, news and tracker update
around it on their own timers. The chart redraws only when a new candle opens, when you press Update, or never (Hold chart)."""
import numpy as np
import pandas as pd
import streamlit as st

from src import auth
from src import indicators as I
from src import news as N
from src import settings as S
from src.bot import kill_switch, run_cycle
from src.confluence import ConfParams, confluence, status_rows, waiting_text
from src.imba import imba_algo, status_row as imba_status
from src.imba import ladder as imba_ladder_for
from src.live import TIMEFRAMES, candle_seconds_left, get_candles, get_quote
from src.livechart import CONF, IMBA_ALGO, live_chart
from src.newsui import mood_dot, render_headlines, render_news_strip
from src.paper_engine import BotConfig, new_state
from src.paperstore import Store
from src.security import esc, safe_ticker
from src.sr import sr_zones
from src.ui import asset_picker, reset_settings, save_settings, seed_settings, worker_status

COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD"]
SR, FVG, OB = "Support / Resistance zones", "Imbalance zones (FVG)", "Order Blocks"
OVERLAYS = [CONF, IMBA_ALGO, "TOTT", SR, FVG, OB, "SMA 20", "SMA 50", "EMA 21", "EMA 200", "Bollinger Bands", "VWAP", "Supertrend"]
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
    "auto": (True, S.flag()), "every": (15, S.choice([5, 10, 15, 30, 60])), "hold": (False, S.flag()),
    "overlays": ([CONF], S.multi(OVERLAYS)), "panels": ([], S.multi(PANELS)), "show_vol": (True, S.flag()),
    "tott_len": (40, S.num(2, 200)), "tott_pct": (1.0, S.num(0.1, 10.0)), "tott_coeff": (0.001, S.num(0.0, 0.05)),
    "c_stop": (0.5, S.num(0.0, 3.0)), "c_rr": (2.0, S.num(0.5, 10.0)), "c_min": (4, S.num(4, 6)),
    "sr_left": (20, S.num(4, 60)), "sr_right": (5, S.num(2, 20)), "sr_max": (10, S.num(2, 30)),
    "imba_sens": (18.0, S.num(1.0, 60.0)), "imba_sl": (0.0, S.num(0.0, 30.0)), "imba_fixed": (False, S.flag()),
    "fvg_min": (0.25, S.num(0.0, 3.0)), "ob_swing": (10, S.num(3, 50)), "ob_disp": (1.5, S.num(0.5, 5.0)),
    "ob_zone": ("Candle body", S.choice(OB_ZONES)), "max_zones": (4, S.num(1, 15)),
    "show_mit": (False, S.flag()), "show_old": (False, S.flag()),
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
    auto = st.toggle("Auto-refresh prices, bot and news", key="lv_auto")
    every = st.select_slider("Refresh every (seconds)", [5, 10, 15, 30, 60], key="lv_every")
    hold = st.toggle("🔒 Hold chart (never redraw by itself)", key="lv_hold",
                     help="The chart normally redraws only when a new candle opens. Hold keeps it exactly as it is, zoom included, "
                          "until you press Update chart now.")

    st.subheader("Indicators")
    overlays = st.multiselect("On the chart", OVERLAYS, key="lv_overlays",
                              help="Default: the combined Confluence indicator (TOTT trend + Support/Resistance). "
                                   "Add the real IMBA ALGO, TOTT, zones or any classic indicator.")
    panels = st.multiselect("Extra panels below", PANELS, key="lv_panels")
    show_vol = st.checkbox("Volume bars on the price chart", key="lv_show_vol")
    with st.expander("Indicator settings"):
        st.markdown("**Support / Resistance** (used by Confluence)")
        sr_left = st.number_input("Pivot lookback (candles; doubled on 1D / 1W)", 4, 60, key="lv_sr_left")
        sr_right = st.number_input("Confirm after (candles)", 2, 20, key="lv_sr_right")
        sr_max = st.number_input("Max active zones per side", 2, 30, key="lv_sr_max")
        st.markdown("**Confluence signal**")
        c_stop = st.number_input("Stop buffer beyond the zone (x ATR)", 0.0, 3.0, step=0.1, key="lv_c_stop")
        c_rr = st.number_input("Reward : risk target", 0.5, 10.0, step=0.5, key="lv_c_rr")
        c_min = st.number_input("Minimum confidence (4-6)", 4, 6, key="lv_c_min")
        st.markdown("**TOTT** (Twin Optimized Trend Tracker)")
        tott_len = st.number_input("OTT period", 2, 200, key="lv_tott_len")
        tott_pct = st.number_input("Optimization constant (%)", 0.1, 10.0, step=0.1, key="lv_tott_pct")
        tott_coeff = st.number_input("Twin coefficient", 0.0, 0.05, step=0.0005, format="%.4f", key="lv_tott_coeff")
        st.markdown("**IMBA ALGO** (fib-channel trend)")
        imba_sens = st.number_input("Sensitivity (channel = sensitivity x 10 candles)", 1.0, 60.0, step=0.5, key="lv_imba_sens")
        imba_fixed = st.checkbox("Fixed stop % from entry (instead of the 78.6% / 23.6% level)", key="lv_imba_fixed")
        imba_sl = st.number_input("Stop buffer / fixed stop %", 0.0, 30.0, step=0.1, key="lv_imba_sl")
        st.markdown("**Imbalance zones** (fair value gaps, optional layer)")
        fvg_min = st.number_input("Minimum gap size (x ATR)", 0.0, 3.0, step=0.05, key="lv_fvg_min")
        st.markdown("**Order blocks** (optional layer)")
        ob_swing = st.number_input("Structure lookback (candles)", 3, 50, key="lv_ob_swing")
        ob_disp = st.number_input("Impulse size (x ATR)", 0.5, 5.0, step=0.1, key="lv_ob_disp")
        ob_zone = st.selectbox("Order block zone covers", OB_ZONES, key="lv_ob_zone")
        st.markdown("**Zones**")
        max_zones = st.number_input("Max zones shown per type", 1, 15, key="lv_max_zones")
        show_mit = st.checkbox("Also show broken zones", key="lv_show_mit")
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
need = min(bars + 250, 260) if tf == "1W" else bars + 250                     # extra candles warm the indicators up
base = ticker.split("-")[0]
st.markdown(f"<p class='hero'>📡 {esc(ticker)} · Live</p>", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers used by the chart and by the live parts
def load():
    """Candles (forming candle patched to the latest trade) + quote."""
    df, source = candles(ticker, tf, need)
    q, qsrc = quote(ticker, tf, df)
    if qsrc == "Kraken":
        df = df.copy()
        price = q["price"]
        df.iloc[-1, df.columns.get_loc("Close")] = price
        df.iloc[-1, df.columns.get_loc("High")] = max(df["High"].iloc[-1], price)
        df.iloc[-1, df.columns.get_loc("Low")] = min(df["Low"].iloc[-1], price)
    return df, source, q, qsrc


def analyse(df):
    """Every indicator, on closed candles only where a signal is involved (no repainting)."""
    closed = df.iloc[:-1]
    left = int(sr_left) * (2 if tf in ("1D", "1W") else 1)
    cp = ConfParams(tott_length=int(tott_len), tott_percent=float(tott_pct), tott_coeff=float(tott_coeff), sr_left=left,
                    sr_right=int(sr_right), sr_max=int(sr_max), stop_atr=float(c_stop), rr=float(c_rr), min_conf=int(c_min))
    ind = I.compute_all(df, TIMEFRAMES[tf], dict(tott_length=int(tott_len), tott_percent=float(tott_pct), tott_coeff=float(tott_coeff),
                                                 fvg_min_atr=float(fvg_min), ob_swing=int(ob_swing), ob_disp=float(ob_disp),
                                                 ob_body=(ob_zone == "Candle body")))
    return dict(closed=closed, cp=cp, ind=ind, conf=confluence(closed, cp), sr=sr_zones(closed, left, int(sr_right), int(sr_max)),
                imba=imba_algo(closed, sensitivity=float(imba_sens), sl_pct=float(imba_sl), fixed_stop=bool(imba_fixed)))


def news_for_guard():
    ndf, nstat = news_data()
    if nstat and all(v.startswith("failed") for v in nstat.values()):
        raise RuntimeError("all news sources unreachable")
    return ndf


def run_owner_bot(df, source):
    """Owner only: let the bot look at the newest closed candle. Returns the store and the bot id (or None)."""
    if not (auth.is_owner() and tf in BOT_TFS):
        return None, None
    store, bot_id = Store(), f"{ticker}@{tf}"
    rec = store.get(bot_id)
    if rec and rec["state"]["enabled"]:
        try:
            run_cycle(store, bot_id, fetch=lambda s_, t_, n_: (df, source), news_fetch=news_for_guard, runner="page")
        except Exception as exc:
            st.warning(f"The bot could not be checked this time ({type(exc).__name__}). It will try again on the next refresh.")
    return store, bot_id


def bot_panel(store: Store, bot_id: str, price: float) -> None:
    """Owner-only control card for the paper bot that trades this chart (fake money)."""
    st.markdown(f"#### 🤖 Paper bot on this chart · `{bot_id}` · owner only, fake money")
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
    if s.get("detail"):
        st.info("🧠 " + s["detail"])
    with st.expander("⚙ Bot settings (shorts, risk, news guard)"):
        with st.form(f"qs_{bot_id}"):
            a, b, c = st.columns(3)
            short = a.checkbox("Trade both directions (shorts)", cfg.allow_short)
            risk = b.number_input("Risk per trade (%)", 0.1, 5.0, cfg.risk_pct, 0.1)
            rr = c.number_input("Reward : risk", 0.5, 10.0, cfg.rr, 0.5)
            mc = a.number_input("Minimum confidence (4-6)", 4, 6, cfg.min_conf)
            ng = b.checkbox("News guard", cfg.news_guard)
            if st.form_submit_button("Save bot settings"):
                new = {**cfg.to_dict(), "allow_short": short, "risk_pct": risk, "rr": rr, "min_conf": int(mc), "news_guard": ng}
                store.save_cfg(bot_id, BotConfig.from_dict(new).to_dict())
                store.log(bot_id, "info", "Settings changed")
    st.dataframe(store.events(bot_id, 5), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------- A) live strip (own timer; never touches the chart)
@st.fragment(run_every=f"{every}s" if auto else None)
def live_strip():
    try:
        df, source, q, qsrc = load()
    except Exception as exc:
        st.error(f"Could not load live data for {ticker}: {exc}")
        return
    price = q["price"]
    chg = (price / q["open"] - 1) * 100
    m = st.columns([1.5, 1, 1, 1, 1.2])
    m[0].metric(f"{ticker} live price", f"${price:,.2f}", f"{chg:+.2f}% vs {q['open_label']}")
    m[1].metric(f"{q['window']} high", f"${q['high']:,.2f}")
    m[2].metric(f"{q['window']} low", f"${q['low']:,.2f}")
    m[3].metric(f"{q['window']} volume", f"{q['volume']:,.0f}")
    m[4].metric(f"{tf} candle closes in", countdown(candle_seconds_left(df.index[-1], tf)))
    now = pd.Timestamp.now(tz="UTC")
    st.caption(f"Candles: **{source}** · price: {qsrc} · updated {now:%H:%M:%S} UTC · "
               + (f"auto-refresh every {every}s" if auto else "auto-refresh off") + f" · chart times in {tz}")
    ndf, _ = news_data()
    hot = N.for_asset(ndf, base)
    hot = hot[(hot["time"] >= now.tz_localize(None) - pd.Timedelta(hours=2)) & (hot["impact"] == 2)]
    if len(hot):
        st.warning(f"⚠️ High-impact headline in the last 2 hours: {N.md_escape(hot.iloc[0]['title'])} "
                   f"({N.md_escape(hot.iloc[0]['source'])}). Expect sudden moves.")
    drawn = st.session_state.get("_chart_candle")                          # a new candle opened: time to redraw (unless held)
    if drawn and drawn[:2] == (ticker, tf) and drawn[2] != df.index[-1] and not hold:
        st.rerun()


live_strip()

bc1, bc2 = st.columns([1, 4])
bc1.button("🔄 Update chart now", key="lv_update", use_container_width=True)
bc2.caption("🔒 Chart on hold: it will not redraw by itself, so your zoom stays. Press the button to update it." if hold else
            "The chart redraws when a new candle opens, so your zoom stays between candles. Prices, news and the bot keep updating live. "
            "Turn on **Hold chart** in the sidebar to stop all automatic redraws.")

# ---------------------------------------------------------------- the chart (drawn once per run, not on a timer)
try:
    df, source, q, qsrc = load()
except Exception as exc:
    st.error(f"Could not load live data for {ticker}: {exc}")
    st.stop()
price = q["price"]
store, bot_id = run_owner_bot(df, source)
bot_overlay = None
if store is not None and store.get(bot_id):
    bot_overlay = {"trades": store.trades(bot_id), "position": store.get(bot_id)["state"]["position"]}
A = analyse(df)
conf_full = A["conf"].reindex(df.index)
conf_full["signal"] = conf_full["signal"].fillna(0)
imba_full = A["imba"].reindex(df.index)
imba_full["signal"] = imba_full["signal"].fillna(0)
zones = []
if SR in overlays or CONF in overlays:
    zones += I.keep_recent(A["sr"], int(max_zones), show_mit)
if FVG in overlays:
    zones += I.keep_recent(A["ind"]["fvg"], int(max_zones), show_mit)
if OB in overlays:
    zones += I.keep_recent(A["ind"]["ob"], int(max_zones), show_mit)
fig = live_chart(df, TIMEFRAMES[tf], overlays, panels, A["ind"], zones, price, tz, bars, uirevision=f"{ticker}-{tf}",
                 show_old_zones=show_old, conf=conf_full, bot=bot_overlay, imba=imba_full,
                 imba_ladder=imba_ladder_for(A["closed"], A["imba"]), title=f"{ticker} · {tf}",
                 source=source.split(" (")[0], show_volume=bool(show_vol))
st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
st.session_state["_chart_candle"] = (ticker, tf, df.index[-1])


# ---------------------------------------------------------------- B) bot, news and tracker (own timer; never touches the chart)
@st.fragment(run_every=f"{every}s" if auto else None)
def live_details():
    try:
        df, source, q, qsrc = load()
    except Exception as exc:
        st.error(f"Could not load live data for {ticker}: {exc}")
        return
    price = q["price"]
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    ndf, nstat = news_data()
    asset_news = N.for_asset(ndf, base)

    if auth.is_owner() and tf in BOT_TFS:
        store, bot_id = run_owner_bot(df, source)
        bot_panel(store, bot_id, price)
    elif auth.is_owner():
        st.caption("🤖 Paper bots run on 5m, 15m, 30m, 1h or 4h candles. Pick one of those to see your bot here.")

    st.markdown("#### 📰 Market news")
    render_news_strip(asset_news, tz, 12, now)
    if nstat and all(v.startswith("failed") for v in nstat.values()):
        st.caption("News feeds are not reachable right now.")

    A = analyse(df)
    left, right = st.columns([3, 2])
    with left:
        snap, summ = I.snapshot(df, A["ind"])
        day_news = asset_news[asset_news["time"] >= now - pd.Timedelta(hours=24)]
        mm = N.mood(day_news)
        news_row = pd.DataFrame([{"Indicator": f"News mood ({base}, 24h)", "Value": f"{mm['score']:+.2f}",
                                  "Reading": f"{mood_dot(mm['score'])} {mm['label']} · {mm['n']} headlines · {mm['high']} high-impact"}])
        snap = pd.concat([status_rows(A["closed"], A["conf"], A["cp"]), imba_status(A["imba"], price), news_row, snap], ignore_index=True)
        st.markdown(f"**Live indicator tracker** · 🟢 {summ['bull']} bullish · 🔴 {summ['bear']} bearish · ⚪ {summ['neutral']} neutral")
        st.caption("🧭 " + waiting_text(A["conf"], price))
        st.dataframe(snap, hide_index=True, use_container_width=True)
    with right:
        st.markdown("**Nearest unmitigated zones**")
        zt = I.active_zones([z for z in A["sr"] if not z["dropped"]] + A["ind"]["fvg"] + A["ind"]["ob"], price, len(df)).head(8)
        zt["Distance %"] = zt["Distance %"].round(2)
        st.dataframe(zt, hide_index=True, use_container_width=True)
    st.caption("Indicator values include the candle that is still forming, so they can change until it closes.")
    with st.expander(f"📰 More {base} headlines"):
        render_headlines(asset_news, 15, now)
        st.page_link("views/news.py", label="Open the full News dashboard →")


live_details()

with st.expander("What do these indicators mean?"):
    st.markdown("""
**Confluence (TOTT + Support/Resistance).** One signal built from two parts. **TOTT** (Twin Optimized Trend Tracker, by Anil Ozeksi) sets the
trend direction. **Support/Resistance** zones come from swing highs and lows: a resistance zone is the top quarter of a swing-high candle, a support
zone the bottom quarter of a swing-low candle; a zone is broken when price reaches the swing high/low itself. **LONG** when the trend is up and price
dips into support and closes back above it; **SHORT** is the mirror image. Each signal comes with a stop just beyond the zone and a target.
(S/R rules after the TradingView script by Trend-My-Friend, MPL 2.0.)

**IMBA ALGO (fib-channel trend).** Takes the highest high and lowest low of the last sensitivity x 10 candles and draws Fibonacci levels
between them. The 50% line is the trend line. **LONG** when a candle closes above the 50% line and inside the top 23.6% of the channel;
**SHORT** when it closes below the 50% line and inside the bottom 21.4%. The suggested stop sits at the 78.6% (long) or 23.6% (short) level,
which is usually far away, and targets are +1/2/3/4% from entry. Signals use closed candles only.

**News colours (like Forex Factory).** 🔴 HIGH = danger (shocks, bans, hacks, big macro events), 🟠 MEDIUM = minor impact, 🟡 LOW = normal news.

*These are analysis tools, not predictions. Nothing here is financial advice.*
""")
