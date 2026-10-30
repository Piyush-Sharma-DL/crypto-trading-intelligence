# Keeping your paper bots running when the website is closed

**Why bots stop:** a Streamlit page only runs while someone has it open in a browser. Closing the tab, a browser putting a
background tab to sleep, or the free host going to sleep all stop the checking. The bot only acts on candles that have fully
closed, so a gap means it simply misses signals (it never trades old candles later).

The Bot Desk and the Live Chart bot card now show whether the **background worker** has been seen recently, and the bot is
protected against two runners (page + worker) ever processing the same candle.

## Option 1: your own computer (free, works today)
1. Double-click **`Start-Bot-Worker.bat`** in the project folder. A black window stays open and prints one line every 30 seconds.
2. Leave it open. Close the window to stop it.
3. Your computer must stay **on, awake and online**. In Windows: Settings > System > Power > set *Sleep* to *Never* while plugged in.
4. The bot's data lives in `data/private/bot.db` on that computer, so your **laptop copy** of the Bot Desk (localhost:8501) shows it.
   The public website has its own, separate, temporary storage.

## Option 2: an always-on server (about $5/month, or a free cloud VM)
Run the same worker (`python -m src.bot --loop`) as a background service on a small Linux server and run the dashboard there too,
so both see the same database. This is the right home for a bot that must run day and night.

## Option 3: GitHub's free timer (not recommended yet)
A scheduled GitHub job starts from a blank machine every time, so the bot would lose its memory. It would first need an online
database that the website and the job both use, and GitHub's timer can run many minutes late. Possible later, not now.

## Honest advice
For the first weeks of paper trading, Option 1 is enough. Move to a server only if the results justify it.
