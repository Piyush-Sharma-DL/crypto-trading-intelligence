# 📈 Cryptocurrency Price Prediction & Trading Intelligence System

A two-part crypto platform: a **live multi-timeframe chart** with TOTT, IMBA and order-block indicators, and a research-grade **forecasting lab**: walk-forward validated ML + LSTM,
statistical significance testing, explainable BUY / HOLD / SELL signals, risk-aware backtesting,
a multi-asset scanner and a live paper-trading track record, all in one Streamlit dashboard.

## Run it
```bash
python -m venv .venv && .venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements-lstm.txt                    # includes PyTorch for the LSTM
streamlit run app.py                                    # or double-click Start-Dashboard.bat (Windows)
```
Tests (no internet needed): `python -m tests.imba_settings_test`, `python -m tests.bot_reliability_test`, `python -m tests.news_test`, `python -m tests.news_guard_test`, `python -m tests.confluence_test`, `python -m tests.engine_test`, `python -m tests.bot_auth_test`, `python -m tests.smoke_test`, `python -m tests.live_test`, `python -m tests.ui_smoke` (the last also runs the indicator tests).

## Owner-only Bot Desk (new)
The site is public, but the **Bot Desk** appears only after you sign in (Google account or password). See **OWNER_SETUP.md** and **SECURITY.md**.
It runs **paper-trading bots (simulated money only)** on the combined **Confluence** indicator (TOTT trend + order-block / IMBA retest, one signal with stop and target), with fees, slippage, a daily-loss limit and a max-drawdown kill switch.
`python -m src.bot --loop` keeps the bots running on your own computer. No exchange is connected.

## News, and your bot on the chart (new)
- Headlines appear as markers on the Live Chart (star = high-impact, green/red = positive/negative wording) and a banner warns about fresh high-impact news.
- The tracker shows a **News mood** row for the selected coin. Choosing a coin on one page selects it on the others, and the Bot Desk follows the selected bot.
- **Owners** see their paper bot on the Live Chart: entry/exit markers, the open trade's stop and target lines, and a card to create, pause or kill the bot.
- Optional per-bot **news guard**: skip *new* trades for N minutes after a high-impact headline about that coin or the whole market. If the news cannot be checked, it blocks (fails safe).
- Headline mood is a simple keyword rule on titles: a description of the wording, not a forecast.

## Chart upgrades (new)
- **TradingView-style chart:** dark palette, crosshair, OHLC header line, coloured last-price tag, faint volume bars, and drawing tools (line, path, rectangle, circle, eraser) in the chart toolbar.
- **IMBA ALGO** layer: Fibonacci-channel trend with LONG/SHORT signals and an entry / stop / TP1-4 ladder (an independent implementation of the rules of the TradingView script; closed candles only).
- **Remembered settings:** your chart settings come back automatically. The page address carries them (bookmark it), and signed-in owners also keep a private profile.
- **Reliable bots:** only one runner can process a bot at a time, every check leaves a heartbeat, errors are logged instead of silently stopping the bot, and `Start-Bot-Worker.bat` keeps bots running with the website closed (see `ALWAYS_ON.md`).

## Pages
| Page | What it does |
|---|---|
| 📡 **Live Chart** | Real-time price, candles at 5m / 15m / 30m / 1h / 4h / 1D / 1W, auto-refresh, user-selectable indicators, live indicator tracker, nearest imbalance / order-block zones |
| 📰 **News & Market** | Crypto headlines from public RSS feeds, headline mood, high-impact flags, Fear & Greed, BTC dominance, S&P 500 and dollar index. Synced with the Live Chart's asset |
| 🤖 **Bot Desk** (owner only) | Paper bots, trades, equity curve, kill switch, strategy backtest |
| 🧪 **Forecast Lab** | Walk-forward ML forecasts, significance tests, signals, backtest, scanner, live track record |

### Live Chart indicators
Default on the chart: **TOTT**, **IMBA**, **Order Blocks**. Optional: SMA 20/50, EMA 21/200, Bollinger Bands, VWAP, Supertrend, and Volume / RSI / MACD panels.
The tracker table always shows live SMA, EMA, RSI, MACD, Bollinger %B, VWAP, ATR, Supertrend and TOTT readings.
- **TOTT** = Twin Optimized Trend Tracker (Anil Ozeksi): VAR moving average + OTT trailing stop, with twin lines OTT x (1 +/- coeff) plotted 2 bars late. BUY when the support line crosses above the upper line, SELL below the lower line. Defaults 40 / 1% / 0.001 are from the original; tune for short timeframes. (Differences from the Pine original: VAR is seeded with the first price instead of 0, and signals strictly alternate.)
- **IMBA** = imbalance / fair value gap: 3-candle gap (bullish: low[i] > high[i-2]; bearish: high[i] < low[i-2]) larger than a minimum ATR fraction, removed once price trades fully through it.
- **Order blocks**: last opposite-colour candle before an impulse (body > 1.5 x ATR) that breaks the recent swing high/low; mitigated when a candle closes through it. Order-block rules vary between traders; this is one common, rule-based version.
- Data: Kraken public API (real-time, no key) with Yahoo Finance fallback. Weekly candles start Monday 00:00 UTC.

## What makes the Forecast Lab more than a basic LSTM project
| Capability | Where | Why it matters |
|---|---|---|
| **Walk-forward validation** | `pipeline.py` | Models are retrained every N days, like real deployment, instead of one lucky split |
| **Significance tests** | `pipeline.py` | Binomial test (direction vs coin flip) and Diebold-Mariano style test (vs naive): is it skill or noise? |
| **Prediction intervals + P(up)** | `Experiment.forecast_range` | Forecasts come with honest uncertainty, not a single number |
| **Explainability** | `_contributions` | TreeSHAP (XGBoost) shows what pushes tomorrow's forecast |
| **Sentiment & macro features** | `data.py`, `features.py` | Fear & Greed index, S&P 500 and dollar index (all published before the close they are used with) |
| **Completed-bar discipline** | `data.completed_bars` | The in-progress candle is never used for training or forecasting |
| **Explainable signals** | `signals.py` | Vote score from model, trend, MACD, RSI, with reasons shown |
| **Risk-aware backtest** | `backtest.py`, `risk.py` | Fees, stop-loss, take-profit, volatility targeting, no look-ahead |
| **Monte Carlo robustness** | `risk.monte_carlo` | Shows how much of a backtest result could be luck |
| **Multi-asset scanner** | `scanner.py` | Signals for 10 coins in one table |
| **Live track record** | `paper.py` | Forecasts are logged and scored against reality: a genuine out-of-sample record |
| **Exports** | `report.py` | One-click HTML report (print to PDF) and CSVs |

## Design decisions (good viva answers)
- **Target = next-day log-return**, not price: returns are stationary; price = close x exp(return).
- **No leakage**: chronological split, walk-forward retraining, scalers fit on past data only, signal at close *t* earns the *t to t+1* return.
- **Naive baseline**: "tomorrow = today" is very hard to beat, so every model is judged against it with a statistical test.
- **LSTM** (30-day window, Huber loss, early stopping) is trained once on the training period; the live forecast uses a refit on all data.

## Honest expectations
Daily crypto is close to a random walk. Expect direction accuracy near 50-56% and errors close to the naive baseline.
The dashboard is built to *measure that honestly* rather than hide it. Use the live track record: it needs about 30+ days before it means anything.

## Hosting note
On Streamlit Community Cloud the PyTorch LSTM is skipped (use `requirements.txt`) and the paper-trading log is not permanent.
Run locally for the full feature set.

*Not financial advice. Analytical research tool only.*
