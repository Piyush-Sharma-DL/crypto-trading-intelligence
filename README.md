# 📈 Cryptocurrency Price Prediction & Trading Intelligence System

A research-grade crypto forecasting and trading-signal platform: walk-forward validated ML + LSTM,
statistical significance testing, explainable BUY / HOLD / SELL signals, risk-aware backtesting,
a multi-asset scanner and a live paper-trading track record, all in one Streamlit dashboard.

## Run it
```bash
python -m venv .venv && .venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements-lstm.txt                    # includes PyTorch for the LSTM
streamlit run app.py                                    # or double-click Start-Dashboard.bat (Windows)
```
Tests (no internet needed): `python -m tests.smoke_test` and `python -m tests.ui_smoke`.

## What makes it more than a basic LSTM project
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
