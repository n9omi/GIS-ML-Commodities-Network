"""Local projections (Jorda 2005) of price on a supply surprise, linear and state-dependent
(Ramey and Zubairy 2018 style), with Newey-West HAC standard errors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


def _design(y: pd.Series, shock: pd.Series, h: int, n_lags: int) -> pd.DataFrame:
    dy = y.diff()
    d = pd.DataFrame({"dep": y.shift(-h) - y.shift(1), "shock": shock})
    for l in range(1, n_lags + 1):
        d[f"dy_l{l}"] = dy.shift(l)
        d[f"shock_l{l}"] = shock.shift(l)
    return d


def lp_linear(y: pd.Series, shock: pd.Series, horizons: int, n_lags: int = 2, ci: float = 0.9) -> pd.DataFrame:
    """beta_h from  y_{t+h} - y_{t-1} = a_h + beta_h s_t + lags + e_{t+h}."""
    z = norm.ppf(0.5 + ci / 2)
    rows = []
    for h in range(horizons + 1):
        d = _design(y, shock, h, n_lags).dropna()
        X = sm.add_constant(d.drop(columns="dep"))
        res = sm.OLS(d["dep"], X).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        b, se = res.params["shock"], res.bse["shock"]
        rows.append({"h": h, "state": "all", "beta": b, "se": se, "lo": b - z * se, "hi": b + z * se,
                     "n": int(res.nobs)})
    return pd.DataFrame(rows)


def lp_state(y: pd.Series, shock: pd.Series, state: pd.Series, horizons: int, n_lags: int = 2,
             ci: float = 0.9) -> pd.DataFrame:
    """State-dependent LP: every coefficient (intercept, shock, lags) interacts with 1[S_{t-1} = g].

    `state` should already be lagged/filtered so it is known before the shock arrives.
    """
    z = norm.ppf(0.5 + ci / 2)
    levels = sorted(state.dropna().unique())
    rows = []
    for h in range(horizons + 1):
        d = _design(y, shock, h, n_lags)
        d["S"] = state
        d = d.dropna()
        cols = [c for c in d.columns if c not in ("dep", "S")]
        parts = {}
        for g in levels:
            ind = (d["S"] == g).astype(float)
            parts[f"const_{g}"] = ind
            for c in cols:
                parts[f"{c}_{g}"] = d[c] * ind
        X = pd.DataFrame(parts, index=d.index)
        X = X.loc[:, X.abs().sum() > 0]
        res = sm.OLS(d["dep"], X).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        for g in levels:
            key = f"shock_{g}"
            if key not in res.params:
                continue
            b, se = res.params[key], res.bse[key]
            rows.append({"h": h, "state": g, "beta": b, "se": se, "lo": b - z * se, "hi": b + z * se,
                         "n": int((d["S"] == g).sum())})
        # Wald test that the shock response is equal across states at this horizon
        keys = [f"shock_{g}" for g in levels if f"shock_{g}" in res.params]
        if len(keys) >= 2:
            R = np.zeros((len(keys) - 1, len(res.params)))
            names = list(res.params.index)
            for i in range(len(keys) - 1):
                R[i, names.index(keys[i])] = 1
                R[i, names.index(keys[i + 1])] = -1
            w = res.wald_test(R, scalar=True)
            rows.append({"h": h, "state": "equality_test", "beta": float(w.statistic),
                         "se": np.nan, "lo": np.nan, "hi": float(w.pvalue), "n": int(res.nobs)})
    return pd.DataFrame(rows)
