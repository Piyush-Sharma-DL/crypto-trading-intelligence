"""End-to-end experiment: features -> chronological split -> walk-forward training ->
evaluation with significance tests, prediction intervals and explainability."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .features import add_indicators, build_features
from .models import LSTMForecaster, make_sequences, sklearn_models

NAIVE = "Naive (tomorrow = today)"


@dataclass
class Experiment:
    indicators: pd.DataFrame            # OHLCV + indicators aligned to the feature index
    preds: pd.DataFrame                 # out-of-sample predicted next-day log-returns (test period)
    next_pred: pd.Series                # forecast log-return for the next, unseen day
    metrics: pd.DataFrame
    importance: pd.Series               # Random Forest global importance
    resid: pd.DataFrame                 # out-of-sample residual std / 5% / 95% quantiles per model
    contrib: pd.Series                  # per-feature contribution to the latest forecast
    contrib_label: str
    n_train: int
    n_test: int
    n_folds: int
    up_share: float                     # share of up days in the test period
    features_used: list
    lstm_history: dict | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def close(self) -> pd.Series:
        return self.indicators["Close"]

    @property
    def last_date(self) -> pd.Timestamp:
        return self.indicators.index[-1]

    def forecast_range(self, model: str, level: float = 0.9) -> tuple[float, float, float]:
        """(low, high, P(up)) for the next close, from out-of-sample residuals."""
        r, last, p = self.resid.loc[model], float(self.close.iloc[-1]), float(self.next_pred[model])
        lo, hi = last * np.exp(p + r["q05"]), last * np.exp(p + r["q95"])
        prob_up = float(stats.norm.cdf(p / r["std"])) if r["std"] > 0 else 0.5
        return float(lo), float(hi), prob_up


# ------------------------------------------------------------------ statistics
def _dir_pvalue(pred: np.ndarray, actual: np.ndarray) -> float:
    k, n = int(np.sum(np.sign(pred) == np.sign(actual))), len(pred)
    return float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue)


def _dm_pvalue(pred: np.ndarray, actual: np.ndarray) -> float:
    """One-sided Diebold-Mariano style test: is the model's squared error below the naive forecast's?"""
    d = (actual - pred) ** 2 - actual ** 2
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(stats.norm.cdf(d.mean() / se)) if se > 0 else np.nan


def evaluate(close_t: pd.Series, close_next: pd.Series, pred_ret: np.ndarray, naive: bool = False) -> dict:
    pred_price = close_t.values * np.exp(pred_ret)
    err = pred_price - close_next.values
    actual = np.log(close_next.values / close_t.values)
    out = {
        "MAE ($)": float(np.mean(np.abs(err))),
        "RMSE ($)": float(np.sqrt(np.mean(err ** 2))),
        "MAPE (%)": float(np.mean(np.abs(err) / close_next.values) * 100),
        "Directional Acc. (%)": np.nan, "IC": np.nan, "p (dir > 50%)": np.nan, "p (beats naive)": np.nan,
    }
    if not naive:
        out["Directional Acc. (%)"] = float(np.mean(np.sign(pred_ret) == np.sign(actual)) * 100)
        out["IC"] = float(np.corrcoef(pred_ret, actual)[0, 1]) if np.std(pred_ret) > 0 else np.nan
        out["p (dir > 50%)"] = _dir_pvalue(pred_ret, actual)
        out["p (beats naive)"] = _dm_pvalue(pred_ret, actual)
    return out


# ------------------------------------------------------------------ explainability
def _contributions(models: dict, x_next: np.ndarray, cols: list) -> tuple[pd.Series, str]:
    if "XGBoost" in models:
        try:
            import xgboost as xgb
            c = models["XGBoost"].get_booster().predict(xgb.DMatrix(x_next), pred_contribs=True)[0][:-1]
            return pd.Series(c, index=cols), "XGBoost TreeSHAP"
        except Exception:
            pass
    lin = models["Linear Regression"]
    z = lin[0].transform(x_next)[0]
    return pd.Series(lin[-1].coef_ * z, index=cols), "Ridge (coefficient x z-score)"


def summarize(exp: Experiment) -> list[str]:
    """Plain-English, honest reading of the results."""
    m = exp.metrics.drop(index=NAIVE)
    out = []
    best = m["MAE ($)"].idxmin()
    imp, p_dm = m.loc[best, "MAE vs Naive (%)"], m.loc[best, "p (beats naive)"]
    verdict = "statistically significant" if p_dm < 0.05 else "not statistically significant"
    out.append(f"Lowest price error: {best}, {imp:+.2f}% vs the naive baseline ({verdict}, p = {p_dm:.2f}).")
    bd = m["Directional Acc. (%)"].idxmax()
    pd_ = m.loc[bd, "p (dir > 50%)"]
    out.append(f"Best direction call: {bd} at {m.loc[bd, 'Directional Acc. (%)']:.1f}% "
               f"({'significantly above' if pd_ < 0.05 else 'not distinguishable from'} a coin flip, p = {pd_:.2f}). "
               f"The test period had {exp.up_share * 100:.0f}% up days.")
    if (m["p (beats naive)"] >= 0.05).all():
        out.append("No model beats the naive forecast with statistical confidence. This is a normal result for daily crypto.")
    return out


# ------------------------------------------------------------------ main
def run_experiment(df: pd.DataFrame, test_frac: float = 0.2, use_lstm: bool = True, window: int = 30,
                   epochs: int = 40, seed: int = 42, retrain_every: int = 30,
                   extras: pd.DataFrame | None = None) -> Experiment:
    ind_full = add_indicators(df)
    X = build_features(ind_full, extras)
    ind = ind_full.reindex(X.index)
    close = ind["Close"]
    close_next = close.shift(-1)
    y = np.log(close_next / close)                      # NaN on the final (unlabeled) row

    n_lab = len(X) - 1                                  # last row = the "tomorrow" query
    n_test = max(int(n_lab * test_frac), 30)
    n_train = n_lab - n_test
    if n_train < 200:
        raise ValueError("Not enough history: need at least ~200 training days. Increase the history length.")

    cols = list(X.columns)
    Xv, yv = X.values, y.values
    test_idx = X.index[n_train:n_lab]
    x_next = Xv[n_lab:n_lab + 1]

    # ---- walk-forward: retrain on all data up to each fold start, predict the next `step` days
    names = list(sklearn_models(seed))
    preds = {n: np.empty(n_test) for n in names}
    step = n_test if retrain_every <= 0 else retrain_every
    n_folds = 0
    for start in range(n_train, n_lab, step):
        end = min(start + step, n_lab)
        for name, model in sklearn_models(seed).items():
            model.fit(Xv[:start], yv[:start])
            preds[name][start - n_train:end - n_train] = model.predict(Xv[start:end])
        n_folds += 1

    # ---- final models on ALL labeled data -> live forecast + explainability
    final, nxt = sklearn_models(seed), {}
    for name, model in final.items():
        model.fit(Xv[:n_lab], yv[:n_lab])
        nxt[name] = float(model.predict(x_next)[0])
    importance = pd.Series(final["Random Forest"].feature_importances_, index=cols).sort_values()
    contrib, contrib_label = _contributions(final, x_next, cols)

    notes, lstm_history = [], None
    if use_lstm:
        try:
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(Xv[:n_train])                  # evaluation model: train data only
            Xs = sc.transform(Xv)
            lstm = LSTMForecaster(X.shape[1], epochs=epochs, seed=seed)
            lstm.fit(make_sequences(Xs, range(window - 1, n_train), window), yv[window - 1:n_train])
            preds["LSTM"] = lstm.predict(make_sequences(Xs, range(n_train, n_lab), window))
            lstm_history = lstm.history
            sc2 = StandardScaler().fit(Xv[:n_lab])                   # live model: refit on everything
            Xs2 = sc2.transform(Xv)
            live = LSTMForecaster(X.shape[1], epochs=epochs, seed=seed)
            live.fit(make_sequences(Xs2, range(window - 1, n_lab), window), yv[window - 1:n_lab])
            nxt["LSTM"] = float(live.predict(make_sequences(Xs2, [n_lab], window))[0])
            notes.append("LSTM is trained once on the training period (it is not retrained inside the test window).")
        except ImportError:
            notes.append("PyTorch is not installed, so the LSTM was skipped. Run `pip install torch`.")

    preds_df = pd.DataFrame(preds, index=test_idx)
    preds_df["Ensemble"] = preds_df.mean(axis=1)
    nxt["Ensemble"] = float(np.mean(list(nxt.values())))

    ct, cn = close.iloc[n_train:n_lab], close_next.iloc[n_train:n_lab]
    actual = np.log(cn.values / ct.values)
    rows = {name: evaluate(ct, cn, preds_df[name].values) for name in preds_df.columns}
    rows[NAIVE] = evaluate(ct, cn, np.zeros(len(ct)), naive=True)
    metrics = pd.DataFrame(rows).T
    naive_mae = metrics.loc[NAIVE, "MAE ($)"]
    metrics["MAE vs Naive (%)"] = (naive_mae - metrics["MAE ($)"]) / naive_mae * 100

    res = {}
    for name in preds_df.columns:
        r = actual - preds_df[name].values
        res[name] = {"std": float(np.std(r)), "q05": float(np.quantile(r, 0.05)), "q95": float(np.quantile(r, 0.95))}

    return Experiment(ind, preds_df, pd.Series(nxt), metrics, importance, pd.DataFrame(res).T,
                      contrib, contrib_label, n_train, n_test, n_folds, float(np.mean(actual > 0)),
                      cols, lstm_history, notes)
