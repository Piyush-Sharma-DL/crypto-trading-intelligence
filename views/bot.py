"""Bot Desk: OWNER ONLY. Paper-trading bots with simulated money, driven by the Confluence indicator."""
import pandas as pd
import streamlit as st

from src import auth
from src import botcharts as bc
from src import news as N
from src.bot import kill_switch, params_for, run_cycle
from src.confluence import confluence
from src.live import fetch_history, get_candles
from src.paper_engine import BotConfig, new_state, run_backtest
from src.paperstore import Store
from src.newsui import mood_dot, render_headlines
from src.security import esc, safe_ticker

auth.require_owner()                                  # nothing below runs for anyone else
store = Store()
COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD"]
TFS = ["5m", "15m", "30m", "1h", "4h"]


@st.cache_data(ttl=8, show_spinner=False)
def candles(symbol, tf, n):
    return get_candles(symbol, tf, n)


@st.cache_data(ttl=300, show_spinner=False)
def news_data():
    return N.fetch_news(max_age_hours=72)


def guard_news():
    """News for the bot's guard. Raises if every source is down, so the bot fails safe instead of assuming 'no news'."""
    df, status = news_data()
    if status and all(v.startswith("failed") for v in status.values()):
        raise RuntimeError("all news sources unreachable")
    return df


with st.sidebar:
    st.success("Signed in as owner")
    if st.button("Sign out"):
        auth.logout()
        st.rerun()

st.markdown("<p class='hero'>🤖 Bot Desk</p>", unsafe_allow_html=True)
st.warning("**PAPER TRADING.** Simulated money only. No exchange or trading account is connected to this site.")

tab_run, tab_new, tab_bt = st.tabs(["📊 My bots", "➕ New bot", "🧪 Strategy backtest"])


def config_widgets(key: str, d: BotConfig):
    c1, c2, c3 = st.columns(3)
    equity = c1.number_input("Starting money (fake)", 100.0, 10_000_000.0, d.start_equity, 100.0, key=f"{key}_eq")
    risk = c1.number_input("Risk per trade (% of equity)", 0.1, 5.0, d.risk_pct, 0.1, key=f"{key}_risk")
    rr = c1.number_input("Reward : risk target", 0.5, 10.0, d.rr, 0.5, key=f"{key}_rr")
    stop_atr = c2.number_input("Stop buffer (x ATR)", 0.0, 3.0, d.stop_atr, 0.1, key=f"{key}_atr")
    min_conf = c2.number_input("Minimum confidence (4-6)", 4, 6, d.min_conf, key=f"{key}_conf")
    short = c2.checkbox("Allow shorts (not possible on spot)", d.allow_short, key=f"{key}_short")
    fee = c3.number_input("Fee per side (%)", 0.0, 1.0, d.fee_pct, 0.01, key=f"{key}_fee")
    slip = c3.number_input("Slippage per fill (%)", 0.0, 1.0, d.slippage_pct, 0.01, key=f"{key}_slip")
    dloss = c3.number_input("Daily loss limit (%)", 0.5, 20.0, d.daily_loss_pct, 0.5, key=f"{key}_dl")
    mdd = c3.number_input("Max drawdown kill switch (%)", 2.0, 50.0, d.max_drawdown_pct, 1.0, key=f"{key}_dd")
    mtd = c3.number_input("Max trades per day", 1, 50, d.max_trades_per_day, key=f"{key}_mt")
    ng = c3.checkbox("News guard: skip NEW trades after a high-impact headline", d.news_guard, key=f"{key}_ng",
                     help="If the news cannot be checked, the bot does not open new trades (fails safe).")
    ngm = c3.number_input("News guard window (minutes)", 15, 480, d.news_guard_minutes, 15, key=f"{key}_ngm")
    return dict(news_guard=ng, news_guard_minutes=int(ngm), start_equity=equity, risk_pct=risk, rr=rr, stop_atr=stop_atr, min_conf=int(min_conf), allow_short=short,
                fee_pct=fee, slippage_pct=slip, daily_loss_pct=dloss, max_drawdown_pct=mdd, max_trades_per_day=int(mtd))


# ---------------------------------------------------------------- new bot
with tab_new:
    with st.form("new_bot"):
        a, b, c = st.columns(3)
        preset = a.selectbox("Market", COINS + ["Custom..."])
        custom = b.text_input("Custom ticker", "LTC-USD")
        tf = c.selectbox("Candle size", TFS, index=1)
        vals = config_widgets("new", BotConfig())
        create = st.form_submit_button("Create paper bot", type="primary")
    if create:
        symbol = safe_ticker(custom if preset == "Custom..." else preset)
        if symbol is None:
            st.error("Invalid ticker. Use Yahoo format, for example LTC-USD.")
        else:
            bot_id = f"{symbol}@{tf}"
            cfg_new = BotConfig(symbol=symbol, timeframe=tf, **vals)
            try:
                store.create(bot_id, cfg_new.to_dict(), new_state(cfg_new))
                st.success(f"Created {bot_id}. It arms on the next closed candle and then trades the Confluence signal with fake money.")
            except Exception:
                st.error("A bot for that market and candle size already exists.")

# ---------------------------------------------------------------- my bots
with tab_run:
    bots = store.list_bots()
    if not bots:
        st.info("No bots yet. Create one in the **New bot** tab.")
    else:
        top = st.columns([2, 1, 1])
        sel = top[0].selectbox("Bot", [b["bot_id"] for b in bots])
        auto = top[1].toggle("Run bots while this page is open", True)
        every = top[2].select_slider("Check every (seconds)", [15, 30, 60], value=30)
        st.caption("Free hosting cannot run a bot in the background: it acts while this page is open. For a bot that "
                   "keeps running, start `python -m src.bot --loop` on your own computer. The bot only acts on candles that "
                   "have fully closed.")
        rec = store.get(sel)
        cfg, state = BotConfig.from_dict(rec["cfg"]), rec["state"]
        st.session_state["shared_asset"] = cfg.symbol                  # the Live Chart and News pages follow the selected bot
        lc = st.columns(2)
        lc[0].page_link("views/live.py", label=f"📡 See {sel} on the Live Chart →")
        lc[1].page_link("views/news.py", label=f"📰 {cfg.symbol.split('-')[0]} news & market →")
        with st.expander(f"📰 Market pulse for {cfg.symbol}"):
            ndf, nstat = news_data()
            base = cfg.symbol.split("-")[0]
            utc_now = pd.Timestamp.now(tz="UTC").tz_localize(None)
            an = N.for_asset(ndf, base)
            mm = N.mood(an[an["time"] >= utc_now - pd.Timedelta(hours=24)]) if len(an) else N.mood(an)
            st.markdown(f"**{base} headline mood (24h):** {mood_dot(mm['score'])} {mm['score']:+.2f} · {mm['label']} · "
                        f"{mm['n']} headlines · {mm['high']} high-impact")
            if cfg.news_guard:
                try:
                    blocked, why = N.news_risk(guard_news(), cfg.symbol, utc_now, cfg.news_guard_minutes)
                    st.markdown("🛡 **News guard:** " + (f"BLOCKING new trades now. {N.md_escape(why)}" if blocked else "clear, new trades allowed."))
                except Exception as exc:
                    st.markdown(f"🛡 **News guard:** cannot check the news ({N.md_escape(str(exc))}), so new trades are blocked.")
            else:
                st.caption("News guard is off for this bot (turn it on in Edit risk settings).")
            render_headlines(an, 6, utc_now)

        b1, b2, b3, b4 = st.columns(4)
        if b1.button("▶ Resume" if not state["enabled"] else "⏸ Pause", use_container_width=True):
            state["enabled"] = not state["enabled"]
            if state["enabled"]:
                if state["halt"] == "manual kill switch":
                    state["halt"] = None
                state["last_bar"] = None                         # re-arm: never trade candles missed while paused
            store.save_state(sel, state)
            store.log(sel, "info", "Resumed" if state["enabled"] else "Paused")
            st.rerun()
        if b2.button("🛑 Kill switch", use_container_width=True, help="Close any open position and stop the bot"):
            try:
                df_, _ = candles(cfg.symbol, cfg.timeframe, 500)
                kill_switch(store, sel, float(df_["Close"].iloc[-1]))
            except Exception as exc:
                st.error(f"Could not get a price: {exc}")
            st.rerun()
        with b3.popover("♻ Reset account", use_container_width=True):
            ok = st.checkbox("Erase this bot's history and restart with fresh fake money", key="rst")
            if st.button("Reset now", disabled=not ok):
                store.clear_history(sel)
                store.save_state(sel, new_state(cfg))
                store.log(sel, "info", "Account reset")
                st.rerun()
        with b4.popover("🗑 Delete bot", use_container_width=True):
            ok = st.checkbox("Delete this bot and its history", key="del")
            if st.button("Delete now", disabled=not ok):
                store.delete(sel)
                st.rerun()

        with st.expander("Edit risk settings (applies to future trades)"):
            with st.form("edit_cfg"):
                new = config_widgets("edit", cfg)
                if st.form_submit_button("Save settings"):
                    store.save_cfg(sel, BotConfig(symbol=cfg.symbol, timeframe=cfg.timeframe, **new).to_dict())
                    store.log(sel, "info", "Settings changed")
                    st.rerun()

        @st.fragment(run_every=f"{every}s" if auto else None)
        def desk(bot_id: str):
            if auto:
                for b in store.list_bots():
                    if b["state"]["enabled"]:
                        run_cycle(store, b["bot_id"], fetch=candles, news_fetch=guard_news)
            r = store.get(bot_id)
            if r is None:
                return
            c_, s = BotConfig.from_dict(r["cfg"]), r["state"]
            eq, trades = store.equity(bot_id), store.trades(bot_id)
            try:
                price = float(candles(c_.symbol, c_.timeframe, 500)[0]["Close"].iloc[-1])
            except Exception:
                price = None
            ret = (s["equity"] / c_.start_equity - 1) * 100
            today = (s["equity"] / s["day_start_equity"] - 1) * 100
            dd = float((eq["equity"] / eq["equity"].cummax() - 1).min() * 100) if len(eq) else 0.0
            status = f"🛑 {s['halt']}" if s["halt"] else ("▶ Running" if s["enabled"] else "⏸ Paused")
            k = st.columns(5)
            k[0].metric("Equity (fake)", f"${s['equity']:,.2f}", f"{ret:+.2f}% since start")
            k[1].metric("Today", f"{today:+.2f}%")
            k[2].metric("Max drawdown", f"{dd:.2f}%")
            k[3].metric("Trades closed", f"{len(trades)}")
            k[4].metric("Status", status)

            pos = s["position"]
            if pos and price:
                unreal = pos["side"] * (price - pos["entry"]) * pos["qty"]
                st.markdown(f"**Open {'LONG' if pos['side'] == 1 else 'SHORT'}** · {pos['qty']:.4f} @ {pos['entry']:,.2f} · "
                            f"stop {pos['stop']:,.2f} · target {pos['target']:,.2f} · confidence {pos['conf']} · "
                            f"unrealised **{unreal:+,.2f}**")
            else:
                st.markdown("**No open position.** Waiting for a Confluence signal on a closed candle.")
            if len(eq) > 1:
                st.plotly_chart(bc.equity_chart(eq, c_.start_equity), use_container_width=True)
            if len(trades):
                st.markdown("**Closed trades**")
                st.dataframe(trades.iloc[::-1][["entry_time", "side", "entry", "exit", "pnl", "r_multiple", "reason", "conf"]]
                             .round({"entry": 2, "exit": 2, "pnl": 2, "r_multiple": 2}), hide_index=True, use_container_width=True)
            st.markdown("**Activity log**")
            st.dataframe(store.events(bot_id, 40), hide_index=True, use_container_width=True)

        desk(sel)

# ---------------------------------------------------------------- backtest
with tab_bt:
    st.caption("Replays the Confluence signals through the same engine the live bot uses, with fees and slippage. "
               "Signals use only data available at each candle's close (no look-ahead).")
    with st.form("bt_form"):
        a, b = st.columns(2)
        sym = a.selectbox("Market", COINS, key="bt_sym")
        tfb = b.selectbox("Candle size", TFS, index=1, key="bt_tf")
        a2, b2 = st.columns(2)
        riskb = a2.number_input("Risk per trade (%)", 0.1, 5.0, 0.5, 0.1, key="bt_risk")
        rrb = b2.number_input("Reward : risk", 0.5, 10.0, 2.0, 0.5, key="bt_rr")
        run = st.form_submit_button("Run backtest", type="primary")
    if run:
        with st.spinner("Downloading history and replaying signals..."):
            try:
                hist, src = fetch_history(sym, tfb)
                cfg_b = BotConfig(symbol=sym, timeframe=tfb, risk_pct=riskb, rr=rrb)
                res = confluence(hist, params_for(cfg_b))
                st.session_state["bt"] = (run_backtest(hist, res, cfg_b), hist, src, sym, tfb)
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
    if "bt" in st.session_state:
        out, hist, src, sym_, tf_ = st.session_state["bt"]
        stt = out["stats"]
        st.markdown(f"**{esc(sym_)} {esc(tf_)}** · {esc(src)} · {hist.index[0]:%d %b %Y} to {hist.index[-1]:%d %b %Y}")
        m = st.columns(5)
        m[0].metric("Strategy return", f"{stt['Return (%)']:+.2f}%")
        m[1].metric("Buy & hold", f"{stt['Buy & hold (%)']:+.2f}%")
        m[2].metric("Trades", f"{stt['Trades']}")
        m[3].metric("Win rate", "n/a" if pd.isna(stt["Win rate (%)"]) else f"{stt['Win rate (%)']:.0f}%")
        m[4].metric("Max drawdown", f"{stt['Max drawdown (%)']:.1f}%")
        if stt["Trades"] < 30:
            st.warning(f"Only {stt['Trades']} trades: far too few to conclude anything. A lucky streak looks identical to skill at this sample size.")
        else:
            st.info("A single backtest can flatter any strategy. Prefer many markets, several periods, and then weeks of paper trading before believing a number.")
        if len(out["equity"]) > 1:
            st.plotly_chart(bc.backtest_chart(out["equity"], hist["Close"], 10_000.0), use_container_width=True)
        if len(out["trades"]):
            st.dataframe(out["trades"][["entry_time", "side", "entry", "exit", "pnl", "r_multiple", "reason", "conf"]]
                         .round({"entry": 2, "exit": 2, "pnl": 2, "r_multiple": 2}), hide_index=True, use_container_width=True)
