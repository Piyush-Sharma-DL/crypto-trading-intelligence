# Security and privacy notes

## What is public and what is private
| Public (anyone with the link) | Private (owner only, behind sign-in) |
|---|---|
| Live Chart, Forecast Lab | Bot Desk: paper bots, trades, equity, settings |

- The private page is **not even registered** for visitors, and it checks the owner again when it loads (two independent checks).
- If sign-in is not configured, or anything looks wrong, the owner area stays **locked** (it fails closed).
- Google sign-in: only your verified email is accepted. Password mode: salted PBKDF2 hash, brute-force lockout, idle and absolute session timeouts.

## Your data
- Secrets (Google client secret, cookie secret, password hash) live **only** in `.streamlit/secrets.toml` locally and in Streamlit Cloud *Secrets*. Never in code or GitHub.
- Bot history is stored in `data/private/bot.db`, which is git-ignored. Nothing about visitors is stored.
- Everything a visitor can type (tickers) is validated before use and escaped before display.

## News content is untrusted
Headlines come from public RSS feeds, so everything in them is treated as hostile text: feeds with DTD/entity tricks are rejected, HTML is stripped,
only http(s) links are kept, titles are escaped before display (page and chart tooltips), and each source is fetched with a timeout and a size cap.
The bot's optional news guard can only make it *more* cautious (block new entries), never place trades.

## Remembered settings are untrusted input
Chart settings are restored from the page address or the owner's profile. Every value is checked against a strict list of allowed options and number ranges;
anything unexpected (including oversized or compressed-bomb data) is ignored and replaced by the default. Only owners have a saved profile on the server.

## Honest limits
- This is a free hosting tier: no firewall, no audit log, and the disk is wiped on restart (bot history on the website does not last; run the bot on your own computer for a lasting record).
- The code repository is **public**, so anyone can read the strategy rules. If that matters, make the GitHub repository **private** (Streamlit Cloud can deploy private repositories).
- Turn on **two-factor authentication** for GitHub, Google and any exchange account. These accounts are the real keys.

## Before any real money (not built, on purpose)
Keep paper trading until you have weeks of results. If you later connect an exchange: create an API key with trading permission only (no withdrawals), restrict it to your server's IP, store it only as a secret on a private always-on server (not the free website), add position limits, a daily loss limit and a kill switch, and start with an amount you can afford to lose.
