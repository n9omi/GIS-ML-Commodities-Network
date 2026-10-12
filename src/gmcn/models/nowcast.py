"""Point-in-time supply nowcasting with rolling-origin evaluation, quantile gradient boosting,
split-conformal intervals and Diebold-Mariano tests.

Timing convention (monthly): a nowcast for month t is made at the end of month t. Official
production for month s is known only if s <= t - publication_lag. Prices (and satellite
activity indices) for month t are known at the end of month t.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_features(production: pd.Series, price: pd.Series, pub_lag: int = 2, price_lags: int = 12,
                  extra: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Target = monthly log growth of production. Features respect the publication lag.

    - own growth at lags pub_lag .. pub_lag + 2 (the latest official prints)
    - price log returns at lags 0 .. price_lags (drilling responds to price with a delay)
    - seasonal harmonics (winter freeze-offs, hurricane season)
    - optional `extra` (e.g. satellite flaring index), assumed observed in month t
    """
    g = np.log(production).diff().rename("y")
    r = np.log(price).diff()
    X = pd.DataFrame(index=g.index)
    for l in range(pub_lag, pub_lag + 3):
        X[f"y_l{l}"] = g.shift(l)
    for l in range(0, price_lags + 1):
        X[f"ret_l{l}"] = r.reindex(g.index).shift(l)
    m = X.index.month
    X["sin12"] = np.sin(2 * np.pi * m / 12)
    X["cos12"] = np.cos(2 * np.pi * m / 12)
    if extra is not None:
        X = X.join(extra, how="left")
    d = pd.concat([g, X], axis=1).dropna()
    return d.drop(columns="y"), d["y"]


def _gbm(alpha: float, seed: int):
    import lightgbm as lgb

    return lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=300, learning_rate=0.03,
                             num_leaves=8, min_child_samples=20, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=seed, verbose=-1)


def rolling_origin(X: pd.DataFrame, y: pd.Series, test_start: str, pub_lag: int, quantiles: list[float],
                   conformal_alpha: float, seed: int = 0, calib_window: int = 36) -> pd.DataFrame:
    """Expanding-window nowcasts. Training for month t uses targets dated <= t - pub_lag only."""
    own = [c for c in X.columns if c.startswith("y_l")]
    rows = []
    test_idx = X.index[X.index >= pd.Timestamp(test_start)]
    history: list[tuple[pd.Timestamp, float]] = []  # (month, ridge error) once that month is published
    for t in test_idx:
        cutoff = t - pd.DateOffset(months=pub_lag)
        tr = X.index <= cutoff
        Xtr, ytr = X.loc[tr], y.loc[tr]
        xt = X.loc[[t]]
        row = {"date": t, "y": float(y.loc[t]), "naive": float(xt[f"y_l{pub_lag}"].iloc[0])}
        ar = LinearRegression().fit(Xtr[own], ytr)
        row["ar"] = float(ar.predict(xt[own])[0])
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))).fit(Xtr, ytr)
        row["ridge"] = float(ridge.predict(xt)[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for q in quantiles:
                row[f"gbm_q{int(q * 100)}"] = float(_gbm(q, seed).fit(Xtr, ytr).predict(xt)[0])
        # split-conformal band: |errors| of past nowcasts whose outcomes are published by month t
        errs = [e for (d, e) in history if d <= cutoff][-calib_window:]
        if len(errs) >= 12:
            n = len(errs)
            level = min(1.0, np.ceil((n + 1) * (1 - conformal_alpha)) / n)
            qhat = float(np.quantile(np.abs(errs), level))
            row["ridge_lo"], row["ridge_hi"] = row["ridge"] - qhat, row["ridge"] + qhat
        rows.append(row)
        history.append((t, row["y"] - row["ridge"]))
    return pd.DataFrame(rows).set_index("date")


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """DM test of equal squared-error loss, with the Harvey-Leybourne-Newbold small-sample fix.

    Negative statistic = model 1 has lower loss than model 2.
    """
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    T = len(d)
    dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    lrv = gamma0 + 2 * sum(np.cov(d[k:], d[:-k], ddof=0)[0, 1] for k in range(1, h))
    dm = dbar / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    return {"stat": float(stat), "p_value": float(2 * tdist.cdf(-abs(stat), df=T - 1)), "T": T}


def evaluate(pred: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    """RMSE, MAE, pinball loss, interval coverage and DM tests vs the AR benchmark."""
    y = pred["y"].values
    rows = []
    for m in ["naive", "ar", "ridge", "gbm_q50"]:
        e = y - pred[m].values
        r = {"model": m, "rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e))),
             "pinball_q50": pinball(y, pred[m].values, 0.5)}
        if m != "ar":
            dm = diebold_mariano(e, y - pred["ar"].values)
            r["dm_vs_ar_stat"], r["dm_vs_ar_p"] = dm["stat"], dm["p_value"]
        rows.append(r)
    out = pd.DataFrame(rows)
    lo, hi = f"gbm_q{int(quantiles[0] * 100)}", f"gbm_q{int(quantiles[-1] * 100)}"
    out.attrs["gbm_interval_coverage"] = float(np.mean((y >= pred[lo]) & (y <= pred[hi])))
    ok = pred[["ridge_lo", "ridge_hi"]].notna().all(axis=1) if "ridge_lo" in pred else pd.Series(False, index=pred.index)
    if ok.any():
        p = pred[ok]
        out.attrs["conformal_coverage"] = float(np.mean((p["y"] >= p["ridge_lo"]) & (p["y"] <= p["ridge_hi"])))
    return out


def standardized_surprise(pred: pd.DataFrame, model: str = "ridge", window: int = 36, pub_lag: int = 2) -> pd.Series:
    """s_t = (y_t - yhat_t) / sigma_t, sigma_t from errors already known at t (lagged by pub_lag)."""
    err = pred["y"] - pred[model]
    sigma = err.shift(pub_lag).rolling(window, min_periods=12).std()
    s = (err / sigma).rename("supply_surprise")
    return s.dropna()
