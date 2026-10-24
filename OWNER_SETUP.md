# Owner sign-in setup (about 10 minutes)

The website stays public. The **Bot Desk** only appears after YOU sign in. Until this is set up, the private area stays locked.

## Option A (recommended): sign in with your Google account

### 1. Create the Google login (free)
1. Go to <https://console.cloud.google.com/> and sign in with the Google account you want as owner.
2. Top bar > project picker > **New project** > name it `crypto-dashboard` > **Create**.
3. Menu > **APIs & Services** > **OAuth consent screen** (or *Google Auth Platform*) > **Get started**.
   App name `Crypto Dashboard`, your email for support and contact. Audience: **External**. Finish.
4. Under **Audience**, keep the app in **Testing** and add **your own Gmail address under "Test users"**.
   (In Testing mode only listed test users can sign in. That is exactly what you want.)
5. **Clients** (or *Credentials*) > **Create client / Create credentials > OAuth client ID** > type **Web application**.
6. **Authorized redirect URIs** > add BOTH:
   - `http://localhost:8501/oauth2callback`
   - `https://crypto-trading-intelligence.streamlit.app/oauth2callback`
7. **Create**. Copy the **Client ID** and **Client secret**.

### 2. Make a secret file on your laptop
1. In your project folder open the `.streamlit` folder. Copy `secrets.toml.example` to a new file named `secrets.toml`.
2. Fill it in: your Gmail as `OWNER_EMAIL`, the Client ID and secret, and a cookie secret made with
   `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
3. **Never upload `secrets.toml` to GitHub.** It is listed in `.gitignore`, but the GitHub website upload does not read that file, so just do not select it.
4. Install the login library once: `pip install -r requirements-lstm.txt`, then run `streamlit run app.py`.

### 3. Add the same secrets to the website
1. share.streamlit.io > your app > **Manage app > Settings > Secrets**.
2. Paste the same text, but change `redirect_uri` to `https://crypto-trading-intelligence.streamlit.app/oauth2callback`.
3. **Save**. The app restarts.

### 4. Test it
- Open the site in a **private/incognito window**: the sidebar shows **Owner sign-in** and no Bot Desk.
- Click **Sign in with Google** with your Gmail: **Bot Desk** appears.
- Try a different Google account: it says *not authorised*.

## Option B: a password (simpler, less secure)
1. Run `python -m src.auth hash` and type a long password (12+ characters, ideally a passphrase).
2. Paste the printed `OWNER_PASSWORD_HASH = "..."` line into your secrets (local file and Streamlit Cloud). Remove the `[auth]` section.
3. The site locks sign-in for everyone for 15 minutes after 8 wrong attempts, and sessions expire after 30 minutes idle.

## If you forget or lose access
Change the secrets in Streamlit Cloud (Manage app > Settings > Secrets). Nothing sensitive is stored in the code.
