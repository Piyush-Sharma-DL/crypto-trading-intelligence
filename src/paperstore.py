"""Private storage for paper bots (SQLite file, standard library only).
The file lives in data/private/ which is git-ignored: trading data never goes to GitHub.
Note: on free Streamlit hosting the disk is wiped on restart, so run the bot on your own computer for a lasting record."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS bots (bot_id TEXT PRIMARY KEY, cfg TEXT NOT NULL, state TEXT NOT NULL, created TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, bot_id TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS equity (bot_id TEXT NOT NULL, ts TEXT NOT NULL, equity REAL NOT NULL, price REAL, PRIMARY KEY (bot_id, ts));
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, bot_id TEXT NOT NULL, ts TEXT NOT NULL, level TEXT, msg TEXT);
"""


def default_path() -> Path:
    return Path(os.environ.get("BOT_DB") or Path(__file__).resolve().parent.parent / "data" / "private" / "bot.db")


def _now() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")


class _Conn:
    """sqlite3 connection that commits on success, rolls back on error, and always closes."""
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, *a):
        (self.conn.rollback if exc_type else self.conn.commit)()
        self.conn.close()
        return False


class Store:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return _Conn(conn)

    # ---- bots
    @staticmethod
    def _bot(r) -> dict:
        return {"bot_id": r["bot_id"], "cfg": json.loads(r["cfg"]), "state": json.loads(r["state"])}

    def list_bots(self) -> list[dict]:
        with self._conn() as c:
            return [self._bot(r) for r in c.execute("SELECT * FROM bots ORDER BY bot_id").fetchall()]

    def get(self, bot_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM bots WHERE bot_id=?", (bot_id,)).fetchone()
        return self._bot(r) if r else None

    def create(self, bot_id: str, cfg: dict, state: dict) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO bots VALUES (?,?,?,?,?)", (bot_id, json.dumps(cfg), json.dumps(state), _now(), _now()))
        self.log(bot_id, "info", "Bot created (paper trading, simulated money)")

    def save_state(self, bot_id: str, state: dict) -> None:
        with self._conn() as c:
            c.execute("UPDATE bots SET state=?, updated=? WHERE bot_id=?", (json.dumps(state), _now(), bot_id))

    def save_cfg(self, bot_id: str, cfg: dict) -> None:
        with self._conn() as c:
            c.execute("UPDATE bots SET cfg=?, updated=? WHERE bot_id=?", (json.dumps(cfg), _now(), bot_id))

    def clear_history(self, bot_id: str) -> None:
        with self._conn() as c:
            for t in ("trades", "equity", "events"):
                c.execute(f"DELETE FROM {t} WHERE bot_id=?", (bot_id,))

    def delete(self, bot_id: str) -> None:
        self.clear_history(bot_id)
        with self._conn() as c:
            c.execute("DELETE FROM bots WHERE bot_id=?", (bot_id,))

    # ---- history
    def add_trade(self, bot_id: str, trade: dict) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO trades (bot_id, data) VALUES (?,?)", (bot_id, json.dumps(trade)))

    def trades(self, bot_id: str) -> pd.DataFrame:
        with self._conn() as c:
            rows = c.execute("SELECT data FROM trades WHERE bot_id=? ORDER BY id", (bot_id,)).fetchall()
        return pd.DataFrame([json.loads(r["data"]) for r in rows])

    def add_equity(self, bot_id: str, ts, equity: float, price: float) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?)", (bot_id, str(ts), float(equity), float(price)))

    def equity(self, bot_id: str) -> pd.DataFrame:
        with self._conn() as c:
            rows = c.execute("SELECT ts, equity, price FROM equity WHERE bot_id=? ORDER BY ts", (bot_id,)).fetchall()
        df = pd.DataFrame([dict(r) for r in rows], columns=["ts", "equity", "price"])
        df["ts"] = pd.to_datetime(df["ts"])
        return df.set_index("ts")

    def log(self, bot_id: str, level: str, msg: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO events (bot_id, ts, level, msg) VALUES (?,?,?,?)", (bot_id, _now(), level, msg))

    def events(self, bot_id: str, n: int = 100) -> pd.DataFrame:
        with self._conn() as c:
            rows = c.execute("SELECT ts, level, msg FROM events WHERE bot_id=? ORDER BY id DESC LIMIT ?", (bot_id, n)).fetchall()
        return pd.DataFrame([dict(r) for r in rows], columns=["ts", "level", "msg"])
