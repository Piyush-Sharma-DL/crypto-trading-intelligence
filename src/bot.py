"""Paper-trading bot runner. Acts ONLY on newly closed candles, so it never repaints or trades a forming candle.

    python -m src.bot --once            run one cycle for every enabled bot
    python -m src.bot --loop --every 30 keep running on your own computer (Ctrl+C to stop)
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from .confluence import ConfParams, confluence
from .live import TIMEFRAMES, get_candles
from .news import fetch_news, news_risk
from .paper_engine import BotConfig, _close_position, step
from .paperstore import Store


def params_for(cfg: BotConfig) -> ConfParams:
    return ConfParams(stop_atr=cfg.stop_atr, rr=cfg.rr, min_conf=cfg.min_conf)


def closed_candles(df: pd.DataFrame, tf: str, now: pd.Timestamp) -> pd.DataFrame:
    return df[(df.index + pd.Timedelta(minutes=TIMEFRAMES[tf])) <= now]


def _news_guard(cfg: BotConfig, news_fetch, now: pd.Timestamp) -> tuple[bool, str]:
    """True = block new entries. If the news cannot be checked, block (fail safe)."""
    try:
        if news_fetch is not None:
            df = news_fetch()
        else:
            df, status = fetch_news(max_age_hours=6, now=now)
            if status and all(v.startswith("failed") for v in status.values()):
                raise RuntimeError("all news sources unreachable")
    except Exception as exc:
        return True, f"news check failed ({str(exc)[:60]}), failing safe"
    return news_risk(df, cfg.symbol, now, cfg.news_guard_minutes)


def run_cycle(store: Store, bot_id: str, fetch=None, now: pd.Timestamp | None = None, min_history: int = 150,
              news_fetch=None) -> dict:
    rec = store.get(bot_id)
    if rec is None:
        return {"bot": bot_id, "status": "missing", "new_bars": 0}
    cfg, state = BotConfig.from_dict(rec["cfg"]), rec["state"]
    if not state["enabled"]:
        return {"bot": bot_id, "status": "paused", "new_bars": 0}
    now = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    try:
        df, source = (fetch or get_candles)(cfg.symbol, cfg.timeframe, 500)
    except Exception as exc:
        store.log(bot_id, "warn", f"Data unavailable: {exc}")
        return {"bot": bot_id, "status": "no data", "new_bars": 0}
    closed = closed_candles(df, cfg.timeframe, now)
    if len(closed) < min_history:
        return {"bot": bot_id, "status": "not enough candles", "new_bars": 0}

    if state["last_bar"] is None:                                   # first run: arm on the latest closed candle, never backfill
        state["last_bar"] = str(closed.index[-1])
        store.save_state(bot_id, state)
        store.log(bot_id, "info", f"Armed on {cfg.symbol} {cfg.timeframe}. Waiting for the next candle to close.")
        return {"bot": bot_id, "status": "armed", "new_bars": 0}

    prev = pd.Timestamp(state["last_bar"])
    new = [i for i, ts in enumerate(closed.index) if ts > prev]
    if not new:
        return {"bot": bot_id, "status": "idle", "new_bars": 0}

    res = confluence(closed, params_for(cfg))
    if len(new) > 3:
        store.log(bot_id, "warn", f"{len(new)} candles were missed while the bot was offline: managing the open trade, "
                                  f"but only the newest candle can open a new one.")
    for i in new:
        ts, r = closed.index[i], res.iloc[i]
        allow = (i == new[-1])
        if allow and cfg.news_guard and int(r["signal"]) != 0 and state["position"] is None and state["enabled"]:
            blocked, why = _news_guard(cfg, news_fetch, now)
            if blocked:
                allow = False
                store.log(bot_id, "warn", f"[{ts:%Y-%m-%d %H:%M}] News guard blocked a "
                                          f"{'LONG' if int(r['signal']) == 1 else 'SHORT'} signal: {why}")
        state, events, trades = step(state, cfg, ts, *closed.iloc[i][["Open", "High", "Low", "Close"]].tolist(),
                                     signal=int(r["signal"]), stop=r["stop"], target=r["target"], conf=int(r["conf"]),
                                     trend=int(r["trend"]), allow_entry=allow)
        for level, msg in events:
            store.log(bot_id, level, f"[{ts:%Y-%m-%d %H:%M}] {msg}")
        for t in trades:
            store.add_trade(bot_id, t)
        store.add_equity(bot_id, ts, state["equity"], closed["Close"].iloc[i])
    store.save_state(bot_id, state)
    return {"bot": bot_id, "status": "processed", "new_bars": len(new), "equity": state["equity"], "halt": state["halt"]}


def kill_switch(store: Store, bot_id: str, price: float) -> None:
    """Flatten any open position at `price` and stop the bot until you resume it."""
    rec = store.get(bot_id)
    cfg, state = BotConfig.from_dict(rec["cfg"]), rec["state"]
    if state["position"]:
        t = _close_position(state, cfg, pd.Timestamp.now(tz="UTC").tz_localize(None), price, "manual kill switch")
        store.add_trade(bot_id, t)
        state["equity"] = state["cash"]
        store.log(bot_id, "warn", f"KILL SWITCH: flattened {t['side']} at {price:,.2f}, net {t['pnl']:+,.2f}")
    state["enabled"], state["halt"] = False, "manual kill switch"
    store.save_state(bot_id, state)
    store.log(bot_id, "warn", "Bot stopped by kill switch")


def main():
    ap = argparse.ArgumentParser(description="Run paper bots (simulated money only).")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--every", type=int, default=30, help="seconds between cycles")
    a = ap.parse_args()
    store = Store()
    while True:
        for b in store.list_bots():
            try:
                print(time.strftime("%H:%M:%S"), run_cycle(store, b["bot_id"]))
            except Exception as exc:
                print("error:", exc)
        if not a.loop:
            break
        time.sleep(max(5, a.every))


if __name__ == "__main__":
    main()
