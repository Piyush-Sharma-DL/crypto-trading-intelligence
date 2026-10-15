"""Vectorised backtest with fees and no look-ahead (signal at close t earns the t -> t+1 return)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERIODS = 365  # crypto trades every day


@dataclass
class BacktestResult:
    equity: pd.DataFrame      # Strategy, Buy & Hold (start = 1.0)
    drawdown: pd.DataFrame
    stats: pd.DataFrame       # one column per curve
    trades: pd.DataFrame
    returns: pd.DataFrame     # daily net returns: Strategy, Buy & Hold


def _stats(ret: pd.Series, equity: pd.Series) -> dict:
    n = len(ret)
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (PERIODS / n) - 1 if n > 0 and equity.iloc[-1] > 0 else np.nan
    vol = ret.std() * np.sqrt(PERIODS)
    sharpe = ret.mean() / ret.std() * np.sqrt(PERIODS) if ret.std() > 0 else np.nan
    mdd = (equity / equity.cummax() - 1).min()
    return {"Total return (%)": total * 100, "CAGR (%)": cagr * 100, "Volatility (%)": vol * 100,
            "Sharpe": sharpe, "Max drawdown (%)": mdd * 100}


def backtest(close: pd.Series, position: pd.Series, fee: float = 0.001) -> BacktestResult:
    """`close` must contain one extra trailing day beyond `position` so the last return is known."""
    asset_ret = (close.shift(-1) / close - 1).reindex(position.index)
    turnover = position.diff().abs().fillna(position.abs())
    strat = position * asset_ret - turnover * fee
    bh = asset_ret.copy()
    bh.iloc[0] -= fee                                   # buy-and-hold pays one entry fee

    equity = pd.DataFrame({"Strategy": (1 + strat).cumprod(), "Buy & Hold": (1 + bh).cumprod()})
    drawdown = equity / equity.cummax() - 1

    stats = pd.DataFrame({"Strategy": _stats(strat, equity["Strategy"]),
                          "Buy & Hold": _stats(bh, equity["Buy & Hold"])})

    # per-trade results
    entry = (position == 1) & (position.shift(1, fill_value=0) == 0)
    trade_id = entry.cumsum().where(position == 1)
    trades = []
    for tid, g in strat.groupby(trade_id):
        trades.append({"Entry": g.index[0], "Exit": g.index[-1],
                       "Days": len(g), "Return (%)": ((1 + g).prod() - 1) * 100})
    trades = pd.DataFrame(trades, columns=["Entry", "Exit", "Days", "Return (%)"])
    stats.loc["Trades", "Strategy"] = len(trades)
    stats.loc["Win rate (%)", "Strategy"] = (trades["Return (%)"] > 0).mean() * 100 if len(trades) else np.nan
    stats.loc["Time in market (%)", "Strategy"] = position.mean() * 100
    return BacktestResult(equity, drawdown, stats, trades, pd.DataFrame({"Strategy": strat, "Buy & Hold": bh}))
