"""Paper-trading engine (FAKE MONEY). One candle at a time, closed candles only.

Rules: one position at a time, fixed-fraction risk sizing, stop and target checked on every later candle's high/low
(if both are hit in the same candle the STOP is assumed first), fees and slippage on every fill, and hard risk limits:
daily-loss halt, max-drawdown kill switch, max trades per day.
The same `step` function drives both the live paper bot and the backtest, so they behave identically.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class BotConfig:
    symbol: str = "ETH-USD"
    timeframe: str = "15m"
    start_equity: float = 10_000.0
    risk_pct: float = 0.5               # % of equity risked per trade
    rr: float = 2.0                     # target = rr x risk
    stop_atr: float = 0.5               # stop buffer beyond the zone, in ATR
    min_conf: int = 4                   # minimum confluence confidence (4-6)
    max_position_pct: float = 100.0     # cap on position value, % of equity (100 = no leverage)
    fee_pct: float = 0.10               # per side
    slippage_pct: float = 0.05          # per fill, against you
    allow_short: bool = False           # spot accounts cannot short
    exit_on_trend_flip: bool = True
    daily_loss_pct: float = 3.0         # halt new trades (and flatten) after losing this much in a UTC day
    max_drawdown_pct: float = 15.0      # kill switch: flatten and stop the bot
    max_trades_per_day: int = 6
    news_guard: bool = False           # skip NEW entries shortly after a high-impact headline (fails safe)
    news_guard_minutes: int = 90

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BotConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def new_state(cfg: BotConfig) -> dict:
    return {"cash": cfg.start_equity, "equity": cfg.start_equity, "peak": cfg.start_equity, "position": None,
            "day": None, "day_start_equity": cfg.start_equity, "trades_today": 0, "halt": None,
            "last_bar": None, "enabled": True}


def _close_position(s: dict, cfg: BotConfig, ts, price: float, reason: str, limit_fill: bool = False) -> dict:
    pos = s["position"]
    side = pos["side"]
    px = price if limit_fill else price * (1 - side * cfg.slippage_pct / 100)       # market exit: slippage against you
    pnl = side * (px - pos["entry"]) * pos["qty"]
    exit_fee = px * pos["qty"] * cfg.fee_pct / 100
    s["cash"] += pnl - exit_fee
    net = pnl - pos["entry_fee"] - exit_fee
    s["position"] = None
    return {"side": "LONG" if side == 1 else "SHORT", "qty": pos["qty"], "entry": pos["entry"], "exit": px,
            "entry_time": pos["opened"], "exit_time": str(ts), "pnl": net,
            "r_multiple": net / pos["risk"] if pos["risk"] else 0.0, "reason": reason, "conf": pos["conf"]}


def step(state: dict, cfg: BotConfig, ts, o: float, h: float, l: float, c: float, signal: int = 0, stop: float = np.nan,
         target: float = np.nan, conf: int = 0, trend: int = 0, allow_entry: bool = True):
    """Process ONE closed candle. Returns (new_state, events, closed_trades). Never mutates the input state."""
    s, events, trades = copy.deepcopy(state), [], []
    day = pd.Timestamp(ts).strftime("%Y-%m-%d")
    if s["day"] != day:
        s["day"], s["day_start_equity"], s["trades_today"] = day, s["equity"], 0
        if s["halt"] == "daily loss limit":
            s["halt"] = None
            events.append(("info", "New UTC day: daily-loss halt lifted"))

    pos = s["position"]
    if pos:                                                       # 1) manage the open position with this candle's range
        side, exit_px, reason, limit = pos["side"], None, None, False
        if side == 1:
            if o <= pos["stop"]:
                exit_px, reason = o, "stop (gap)"
            elif l <= pos["stop"]:
                exit_px, reason = pos["stop"], "stop"
            elif h >= pos["target"]:
                exit_px, reason, limit = pos["target"], "target", True
        else:
            if o >= pos["stop"]:
                exit_px, reason = o, "stop (gap)"
            elif h >= pos["stop"]:
                exit_px, reason = pos["stop"], "stop"
            elif l <= pos["target"]:
                exit_px, reason, limit = pos["target"], "target", True
        if exit_px is None and cfg.exit_on_trend_flip and trend == -side:
            exit_px, reason = c, "trend flip"
        if exit_px is not None:
            t = _close_position(s, cfg, ts, exit_px, reason, limit_fill=limit)
            trades.append(t)
            events.append(("trade", f"CLOSE {t['side']} @ {t['exit']:,.2f} ({reason}) net {t['pnl']:+,.2f} ({t['r_multiple']:+.2f}R)"))

    if (s["position"] is None and signal != 0 and allow_entry and s["enabled"] and s["halt"] is None   # 2) maybe enter
            and s["trades_today"] < cfg.max_trades_per_day and conf >= cfg.min_conf
            and (signal == 1 or cfg.allow_short) and np.isfinite(stop) and np.isfinite(target)):
        entry = c * (1 + signal * cfg.slippage_pct / 100)
        per_unit = abs(entry - stop)
        valid = per_unit > 0 and ((signal == 1 and stop < entry < target) or (signal == -1 and target < entry < stop))
        if valid:
            equity = s["cash"]
            qty = equity * cfg.risk_pct / 100 / per_unit
            qty = min(qty, equity * cfg.max_position_pct / 100 / entry)
            if qty > 0:
                fee = entry * qty * cfg.fee_pct / 100
                s["cash"] -= fee
                s["position"] = {"side": int(signal), "qty": qty, "entry": entry, "stop": float(stop), "target": float(target),
                                 "opened": str(ts), "entry_fee": fee, "risk": qty * per_unit, "conf": int(conf)}
                s["trades_today"] += 1
                events.append(("trade", f"OPEN {'LONG' if signal == 1 else 'SHORT'} {qty:.4f} @ {entry:,.2f}  stop {stop:,.2f}  "
                                        f"target {target:,.2f}  confidence {conf}"))

    def mark() -> float:                                           # 3) mark to market
        p = s["position"]
        return s["cash"] + (p["side"] * (c - p["entry"]) * p["qty"] if p else 0.0)

    s["equity"] = mark()
    s["peak"] = max(s["peak"], s["equity"])
    dd = (s["equity"] / s["peak"] - 1) * 100
    day_pnl = (s["equity"] / s["day_start_equity"] - 1) * 100

    halt = None                                                    # 4) hard risk limits
    if dd <= -cfg.max_drawdown_pct:
        halt = "max drawdown kill switch"
    elif day_pnl <= -cfg.daily_loss_pct:
        halt = "daily loss limit"
    if halt and s["halt"] != halt:
        if s["position"]:
            t = _close_position(s, cfg, ts, c, "risk halt")
            trades.append(t)
            events.append(("trade", f"FLATTEN {t['side']} @ {t['exit']:,.2f} (risk halt) net {t['pnl']:+,.2f}"))
        s["halt"] = halt
        if halt == "max drawdown kill switch":
            s["enabled"] = False
        events.append(("warn", f"HALT: {halt} (drawdown {dd:.1f}%, today {day_pnl:+.1f}%)"))
        s["equity"] = mark()
    s["last_bar"] = str(ts)
    return s, events, trades


def stats(equity: pd.Series, trades: pd.DataFrame, start: float, buy_hold: float | None = None) -> dict:
    ret = (equity.iloc[-1] / start - 1) * 100 if len(equity) else 0.0
    dd = ((equity / equity.cummax() - 1).min() * 100) if len(equity) else 0.0
    n = len(trades)
    wins = trades[trades["pnl"] > 0] if n else trades
    losses = trades[trades["pnl"] <= 0] if n else trades
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if n and losses["pnl"].sum() < 0 else np.nan
    return {"Return (%)": ret, "Buy & hold (%)": buy_hold if buy_hold is not None else np.nan, "Trades": n,
            "Win rate (%)": (len(wins) / n * 100) if n else np.nan, "Avg R": trades["r_multiple"].mean() if n else np.nan,
            "Profit factor": pf, "Max drawdown (%)": dd}


def run_backtest(df: pd.DataFrame, res: pd.DataFrame, cfg: BotConfig, warmup: int = 100) -> dict:
    """Replay the confluence signals through the same engine the live bot uses."""
    state = new_state(cfg)
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    eq, trades, events = [], [], []
    for i in range(warmup, len(df)):
        r = res.iloc[i]
        state, ev, tr = step(state, cfg, df.index[i], o[i], h[i], l[i], c[i], int(r["signal"]), r["stop"], r["target"],
                             int(r["conf"]), int(r["trend"]))
        eq.append(state["equity"])
        trades += tr
        events += ev
        if not state["enabled"]:
            break
    eq = pd.Series(eq, index=df.index[warmup:warmup + len(eq)])
    tdf = pd.DataFrame(trades)
    bh = (c[warmup + len(eq) - 1] / c[warmup] - 1) * 100 if len(eq) else 0.0
    return {"equity": eq, "trades": tdf, "events": events, "stats": stats(eq, tdf, cfg.start_equity, bh), "state": state}
