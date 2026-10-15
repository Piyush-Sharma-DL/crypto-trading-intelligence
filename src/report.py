"""Self-contained HTML report (open in a browser, print to PDF)."""
from __future__ import annotations

import html

import pandas as pd

CSS = """body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:2rem auto;color:#111;padding:0 1rem}
h1{margin-bottom:0}h2{border-bottom:2px solid #38bdf8;padding-bottom:.2rem;margin-top:2rem}
table{border-collapse:collapse;width:100%;font-size:.85rem}th,td{border:1px solid #ddd;padding:.35rem .6rem;text-align:right}
th{background:#f1f5f9}td:first-child,th:first-child{text-align:left}.sig{display:inline-block;padding:.3rem 1rem;
border-radius:999px;font-weight:700;color:#fff}.small{color:#555;font-size:.8rem}"""
COLORS = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#d97706"}


def _tbl(df: pd.DataFrame, digits: int = 2) -> str:
    return df.round(digits).to_html(border=0, na_rep="-")


def build_report(ticker, source, as_of, model, signal, score, forecast_price, lo, hi, prob_up,
                 metrics, verdicts, bt_stats, votes: list[str]) -> str:
    e = html.escape
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{e(ticker)} report</title><style>{CSS}</style></head><body>
<h1>{e(ticker)} Trading Intelligence Report</h1>
<p class="small">Data: {e(source)}. Last completed bar: {e(str(as_of))}. Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC.</p>
<h2>Current signal ({e(model)})</h2>
<p><span class="sig" style="background:{COLORS[signal]}">{signal}</span> &nbsp; score {score:+d}</p>
<ul>{''.join(f'<li>{e(v)}</li>' for v in votes)}</ul>
<p>Next-close forecast: <b>${forecast_price:,.2f}</b> (90% range ${lo:,.2f} to ${hi:,.2f}); estimated probability of an up day: <b>{prob_up * 100:.0f}%</b>.</p>
<h2>Out-of-sample model comparison</h2>{_tbl(metrics)}
<ul>{''.join(f'<li>{e(v)}</li>' for v in verdicts)}</ul>
<h2>Backtest ({e(model)})</h2>{_tbl(bt_stats)}
<p class="small">Research tool, not financial advice. Past performance does not predict future results.</p></body></html>"""
