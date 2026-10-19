"""Forecast Lab page: walk-forward ML forecasts, signals, backtest, track record."""
import numpy as np
import pandas as pd
import streamlit as st

from src import charts as ch
from src import paper
from src.backtest import backtest
from src.data import completed_bars, load_extras, load_prices
from src.features import add_indicators, tail_diagnostics
from src.pipeline import NAIVE, run_experiment, summarize
from src.report import build_report
from src.risk import apply_stop, monte_carlo, vol_target
from src.scanner import COINS, ICON, scan
from src.signals import explain, generate_signals, latest_signals, to_position

# ---------------------------------------------------------------- cached steps
@st.cache_data(ttl=900, show_spinner="Fetching latest market data...")
def get_data(ticker, years, offline):
    return load_prices(ticker, years, offline)


@st.cache_data(ttl=3600, show_spinner="Fetching sentiment and macro data...")
def get_extras(start, end, fng, macro, offline):
    return load_extras(pd.date_range(start, end), fng, macro, offline)


@st.cache_data(show_spinner="Training models with walk-forward validation (the LSTM can take a minute)...")
def get_experiment(df, extras, test_frac, use_lstm, epochs, retrain_every):
    return run_experiment(df, test_frac=test_frac, use_lstm=use_lstm, epochs=epochs,
                          retrain_every=retrain_every, extras=extras)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("⚙️ Controls")
    with st.form("cfg"):
        st.markdown("**Data & models** (press Run to apply)")
        preset = st.selectbox("Asset", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "Custom..."])
        custom = st.text_input("Custom Yahoo ticker", "DOGE-USD")
        years = st.slider("History (years)", 2, 10, 5)
        test_pct = st.slider("Test period (% of data)", 10, 40, 20)
        retrain_every = st.slider("Walk-forward: retrain every N days (0 = never)", 0, 90, 30, 10)
        use_fng = st.checkbox("Fear & Greed sentiment feature", True)
        use_macro = st.checkbox("S&P 500 + dollar index features", True)
        use_lstm = st.checkbox("Include LSTM (needs PyTorch)", True)
        epochs = st.slider("LSTM max epochs", 10, 100, 40, 5)
        offline = st.checkbox("Offline demo mode (synthetic data)", False)
        st.form_submit_button("🚀 Run / apply", use_container_width=True)
    if st.button("🔄 Refresh live data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.subheader("Signal rules")
    threshold = st.slider("Model threshold (% predicted move)", 0.0, 2.0, 0.30, 0.05) / 100
    min_score = st.slider("Signal strictness (min |score|)", 1, 5, 3,
                          help="Votes: model +/-2, trend +/-1, MACD +/-1, RSI extremes +/-1.")
    st.subheader("Trading & risk")
    fee = st.slider("Fee per side (%)", 0.0, 0.5, 0.10, 0.01) / 100
    stop = st.slider("Stop-loss (%) - 0 = off", 0, 30, 0) / 100
    take = st.slider("Take-profit (%) - 0 = off", 0, 100, 0) / 100
    vt = st.slider("Volatility target (% annual) - 0 = off", 0, 150, 0, 10) / 100

ticker = custom.strip().upper() if preset == "Custom..." else preset

# ---------------------------------------------------------------- run
try:
    raw, source = get_data(ticker, years, offline)
    df = completed_bars(raw)
    live_price, live_date = float(raw["Close"].iloc[-1]), raw.index[-1]
    extras, extra_notes = get_extras(str(df.index[0].date()), str(df.index[-1].date()), use_fng, use_macro, offline)
    exp = get_experiment(df, extras, test_pct / 100, use_lstm, epochs, retrain_every)
except Exception as exc:
    st.error(f"Could not complete the analysis: {exc}")
    st.stop()

models = list(exp.preds.columns)
ind, close = exp.indicators, exp.close
last_close, prev_close = float(close.iloc[-1]), float(close.iloc[-2])
next_date = exp.last_date + pd.Timedelta(days=1)
synthetic = "Synthetic" in source

st.markdown(f"<p class='hero'>📈 {ticker} · Trading Intelligence</p>", unsafe_allow_html=True)
st.caption(f"Data: **{source}** · Train {exp.n_train} d / Test {exp.n_test} d · {exp.n_folds} walk-forward fold(s) · "
           f"{len(exp.features_used)} features · Research tool, not financial advice.")
for note in exp.notes + extra_notes:
    st.warning(note)
if synthetic:
    st.warning("Synthetic random-walk data: for demonstrating the pipeline only, not real prices.")

# data-health check: the newest completed daily bar should be yesterday (UTC)
expected_last = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
if not synthetic and exp.last_date < expected_last:
    if df.index[-1] < expected_last:
        gaps = pd.date_range(df.index[-8], df.index[-1]).difference(df.index)
        st.warning(f"Yahoo Finance has no completed bar for {expected_last:%b %d} (newest completed bar: "
                   f"{df.index[-1]:%b %d}"
                   + (f"; missing days: {', '.join(f'{g:%b %d}' for g in gaps)}" if len(gaps) else "")
                   + f"). Forecasts start from {exp.last_date:%b %d}. Try 'Refresh live data' in a few minutes.")
    else:
        cols = tail_diagnostics(add_indicators(df), extras)
        st.warning(f"The {df.index[-1]:%b %d} bar was unusable because these features were empty: "
                   f"{', '.join(cols) or 'unknown'}. Forecasts start from {exp.last_date:%b %d}.")

selected = st.selectbox("Model driving the trading signal", models, index=models.index("Ensemble"))
latest_idx = ind.index[[-1]]
latest = generate_signals(ind.loc[latest_idx], pd.Series([exp.next_pred[selected]], index=latest_idx),
                          threshold, min_score).iloc[0]
latest_tbl = latest_signals(ind, exp.next_pred, threshold, min_score)
lo, hi, p_up = exp.forecast_range(selected)
pred_price = last_close * float(np.exp(exp.next_pred[selected]))

# log today's forecasts for the live track record (real data only)
if not synthetic:
    try:
        if paper.log_forecasts(ticker, exp.last_date, last_close, latest_tbl):
            st.toast("Forecast saved to your live track record.", icon="📒")
    except OSError:
        pass

# backtests for every model (cheap, vectorised)
results, sigs = {}, {}
for name in models:
    s = generate_signals(ind.loc[exp.preds.index], exp.preds[name], threshold, min_score)
    pos = vol_target(apply_stop(to_position(s["signal"]), close, stop, take), close, vt)
    sigs[name], results[name] = s, backtest(close, pos, fee)
res = results[selected]

tabs = st.tabs(["🏠 Overview", "🔭 Market Scanner", "🧪 Model Lab", "💹 Backtest & Risk",
                "📒 Live Track Record", "ℹ️ Method & Limits"])

# ================================================================ OVERVIEW
with tabs[0]:
    k = st.columns([1.15, 1.15, 1.35, 0.9, 1.2])
    k[0].metric(f"Live price ({live_date:%b %d}, bar in progress)", f"${live_price:,.2f}",
                f"{(live_price / last_close - 1) * 100:+.2f}% vs last close")
    k[1].metric(f"Last completed close ({exp.last_date:%b %d})", f"${last_close:,.2f}",
                f"{(last_close / prev_close - 1) * 100:+.2f}%")
    k[2].metric(f"Forecast for {next_date:%b %d} close ({selected})", f"${pred_price:,.2f}",
                f"{(pred_price / last_close - 1) * 100:+.2f}%")
    k[2].caption(f"90% range: \\${lo:,.0f} to \\${hi:,.0f}")
    k[3].metric("P(up day)", f"{p_up * 100:.0f}%")
    k[3].caption("from out-of-sample errors")
    k[4].markdown("**Signal**")
    k[4].markdown(f"<span class='badge' style='background:{ch.SIG_COLOR[latest['signal']]}'>{latest['signal']}</span> "
                  f"<span class='subtle'>score {int(latest['score']):+d}</span>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.2])
    with left:
        st.plotly_chart(ch.votes_chart(latest), use_container_width=True)
        for line in explain(latest):
            st.write("• " + line)
    with right:
        st.plotly_chart(ch.contrib_chart(exp.contrib, exp.contrib_label), use_container_width=True)

    rows = {}
    for m in models:
        l, h, pu = exp.forecast_range(m)
        rows[m] = {"Move (%)": (np.exp(exp.next_pred[m]) - 1) * 100, "Close ($)": last_close * np.exp(exp.next_pred[m]),
                   "Low ($)": l, "High ($)": h, "P(up) (%)": pu * 100, "Signal": ICON[latest_tbl.loc[m, "signal"]]}
    st.markdown("**Forecast by model**")
    st.dataframe(pd.DataFrame(rows).T, use_container_width=True)

    d1, d2, _ = st.columns([1, 1, 3])
    html_report = build_report(ticker, source, exp.last_date.date(), selected, latest["signal"], int(latest["score"]),
                               pred_price, lo, hi, p_up, exp.metrics, summarize(exp), res.stats, explain(latest))
    d1.download_button("⬇️ HTML report", html_report, f"{ticker}_report.html", "text/html")
    d2.download_button("⬇️ Signals CSV", sigs[selected].to_csv(), f"{ticker}_signals.csv", "text/csv")

    days = st.slider("Chart window (days)", 60, min(900, len(ind)), 200, 10)
    st.plotly_chart(ch.price_chart(ind, sigs[selected], days), use_container_width=True)
    st.caption("Signal markers are shown for the out-of-sample test period only.")

# ================================================================ SCANNER
with tabs[1]:
    st.subheader("Multi-asset signal scanner")
    st.caption("Trains a fast Linear + Random Forest + XGBoost ensemble on each asset's last 3 years of completed "
               "daily bars, then applies the same signal rules. It is a screening tool: open an asset above for the full analysis.")
    picks = st.multiselect("Assets", COINS, default=COINS)
    if st.button("▶️ Run scan", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        st.session_state["scan"] = scan(picks, 3, threshold, min_score, offline,
                                        lambda i, n, t: bar.progress(i / n, text=f"Scanning {t}..."))
        bar.empty()
    scan_df = st.session_state.get("scan")
    if scan_df is not None and len(scan_df):
        st.dataframe(scan_df.round(2), use_container_width=True, hide_index=True)
        if "Forecast (%)" in scan_df:
            st.plotly_chart(ch.scanner_bar(scan_df), use_container_width=True)
    else:
        st.info("Press **Run scan** to analyse all selected assets (about a minute).")

# ================================================================ MODEL LAB
with tabs[2]:
    st.subheader("Out-of-sample model comparison")
    st.caption(f"Walk-forward validation: classical models were retrained {exp.n_folds} time(s) on all data available "
               f"before each test block. {exp.n_test} test days. Nothing the models predicted was seen in training.")
    for line in summarize(exp):
        st.info(line)
    st.dataframe(exp.metrics.round(3), use_container_width=True)
    st.caption("p (dir > 50%): chance of this direction accuracy arising by luck. p (beats naive): chance of this error "
               "reduction arising by luck. Below 0.05 = statistically significant. IC = correlation of forecast and actual return.")

    a, b = st.columns(2)
    m = exp.metrics.drop(index=NAIVE)
    a.plotly_chart(ch.metric_bar(m, "MAE ($)", "MAE (lower is better)", exp.metrics.loc[NAIVE, "MAE ($)"], "Naive baseline"),
                   use_container_width=True)
    b.plotly_chart(ch.metric_bar(m, "Directional Acc. (%)", "Directional accuracy (%)", 50, "Coin flip", [40, 65]),
                   use_container_width=True)

    chosen = st.multiselect("Models to plot", models, default=[x for x in ("LSTM", "Ensemble") if x in models] or models[:2])
    win = st.slider("Last N test days", 30, exp.n_test, min(120, exp.n_test), 5)
    st.plotly_chart(ch.pred_vs_actual(close, exp.preds, chosen, win), use_container_width=True)

    c, d = st.columns(2)
    c.plotly_chart(ch.importance_chart(exp.importance), use_container_width=True)
    if exp.lstm_history:
        d.plotly_chart(ch.lstm_curve(exp.lstm_history), use_container_width=True)

# ================================================================ BACKTEST
with tabs[3]:
    st.subheader("Walk-forward backtest on the unseen test period")
    st.caption("Long-only spot strategy: BUY = hold the coin, SELL = cash, HOLD = keep position. A signal at today's close "
               "earns tomorrow's return. Fees on every position change. Stop-loss, take-profit and volatility targeting "
               "(sidebar) are applied when set.")
    st.plotly_chart(ch.equity_chart({n: results[n].equity["Strategy"] for n in models}, selected,
                                    res.equity["Buy & Hold"]), use_container_width=True)
    summary = pd.DataFrame({n: results[n].stats["Strategy"] for n in models})
    summary["Buy & Hold"] = res.stats["Buy & Hold"]
    st.dataframe(summary.T.round(2), use_container_width=True)

    l, r = st.columns(2)
    l.plotly_chart(ch.drawdown_chart(res.drawdown, f"Drawdown (%) · {selected}"), use_container_width=True)
    counts = sigs[selected]["signal"].value_counts().reindex(["BUY", "HOLD", "SELL"]).fillna(0)
    r.plotly_chart(ch.signal_mix(counts, f"Signal mix · {selected}"), use_container_width=True)

    st.markdown("#### Is the result skill or luck? Monte Carlo test")
    mc = monte_carlo(res.returns["Strategy"])
    actual_ret = float(res.equity["Strategy"].iloc[-1] - 1)
    bh_ret = float(res.equity["Buy & Hold"].iloc[-1] - 1)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Actual strategy return", f"{actual_ret * 100:+.1f}%")
    m2.metric("Median simulated return", f"{mc['total_return'].median() * 100:+.1f}%")
    m3.metric("Chance of losing money", f"{(mc['total_return'] < 0).mean() * 100:.0f}%")
    m4.metric("Chance of beating Buy & Hold", f"{(mc['total_return'] > bh_ret).mean() * 100:.0f}%")
    st.plotly_chart(ch.mc_hist(mc, actual_ret, bh_ret), use_container_width=True)
    st.caption("Block-bootstrap of the strategy's own daily returns. A wide spread means the single backtest number "
               "is not reliable on its own.")

    st.markdown("**Recent signals**")
    recent = sigs[selected].tail(15).copy()
    recent["pred_ret"] = (recent["pred_ret"] * 100).round(2)
    st.dataframe(recent.rename(columns={"pred_ret": "pred move (%)"})[["pred move (%)", "score", "signal"]].iloc[::-1],
                 use_container_width=True)
    with st.expander("Trade log"):
        st.dataframe(res.trades.round({"Return (%)": 2}), use_container_width=True)

# ================================================================ TRACK RECORD
with tabs[4]:
    st.subheader("Live paper-trading track record")
    st.caption("Every time you run the dashboard on real data, each model's forecast for the next day is saved here "
               "(once per day). Once that day closes, the forecast is scored against reality. Unlike a backtest, "
               "nothing here can be overfitted: it is a genuine out-of-sample record.")
    log = paper.load_log(ticker)
    if log.empty:
        st.info("No forecasts logged yet for this asset. Open the dashboard on real data and come back tomorrow.")
    else:
        scored = paper.score_log(log, close)
        pending = int(scored["realized_ret"].isna().sum())
        tbl, curves = paper.summary(scored, fee)
        s1, s2, s3 = st.columns(3)
        s1.metric("Days logged", int(log["asof"].nunique()))
        s2.metric("Forecasts scored", int(scored["realized_ret"].notna().sum()))
        s3.metric("Pending (awaiting close)", pending)
        if tbl.empty:
            st.info("Forecasts are saved. They will be scored after the next daily close.")
        else:
            st.dataframe(tbl.round(3), use_container_width=True)
            st.plotly_chart(ch.paper_equity(curves), use_container_width=True)
            st.caption("Small samples are noisy: judge hit rates only after 30+ scored days. The paper curve assumes "
                       "you run the app every day.")
        st.download_button("⬇️ Download full log", scored.to_csv(index=False), f"{ticker}_paper_log.csv", "text/csv")

# ================================================================ METHOD
with tabs[5]:
    st.markdown("""
### Method
1. **Data**: daily OHLCV from Yahoo Finance (cached 15 min; falls back to a local cache, then synthetic data). The in-progress daily bar is excluded from training and forecasting, so every forecast starts from a completed close.
2. **Features** (scale-free): multi-horizon returns, volatility, SMA/EMA gaps, RSI, MACD, Bollinger %B and width, ATR, candle shape, volume change, plus optional Fear & Greed sentiment and S&P 500 / dollar-index context. Each value is published before the daily close it is used with.
3. **Target**: next-day log-return. Price forecast = last close x exp(predicted return).
4. **Validation**: chronological split plus **walk-forward retraining** every N days. Scalers are fit on training data only.
5. **Models**: Ridge linear regression, Random Forest, XGBoost, a 2-layer LSTM (30-day window, early stopping), an equal-weight Ensemble and a naive baseline.
6. **Statistics**: binomial test for direction, Diebold-Mariano style test against the naive forecast, information coefficient.
7. **Uncertainty**: 90% ranges and P(up) come from the out-of-sample residuals of each model.
8. **Explainability**: TreeSHAP contributions (XGBoost) or coefficient x z-score (linear) for tomorrow's forecast; Random Forest global importance.
9. **Signals**: vote score (model, trend, MACD, RSI) gives BUY / HOLD / SELL, with the reasons shown.
10. **Backtest**: long-only, fees, optional stop-loss, take-profit and volatility targeting; no look-ahead; Monte Carlo robustness test.
11. **Live track record**: forecasts are logged and scored against real outcomes.

### Limits
- Daily crypto is close to a random walk and driven by news the models never see. If nothing beats the naive row, that is a real finding.
- One test window can flatter or punish a strategy; use the Monte Carlo and live track record before trusting any number.
- The LSTM is not retrained inside the test window (the classical models are).
- **Not financial advice.** This is an analytical research tool.
""")
