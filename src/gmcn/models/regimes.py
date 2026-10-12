"""Macro regimes: Gaussian hidden Markov model with BIC model selection, filtered (real-time)
state probabilities, and a dynamic factor model cross-check."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def n_params(k: int, d: int) -> int:
    """Free parameters of a k-state full-covariance Gaussian HMM in d dimensions."""
    return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2


def _fit_k(X: np.ndarray, k: int, restarts: int, seed: int):
    """Best-of-restarts Gaussian HMM with k states (full covariance)."""
    from hmmlearn.hmm import GaussianHMM

    best_m, best_ll = None, -np.inf
    for r in range(restarts):
        m = GaussianHMM(n_components=k, covariance_type="full", n_iter=1000, tol=1e-5,
                        random_state=seed + 97 * r + k, min_covar=1e-3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m.fit(X)
                ll = m.score(X)
            except Exception:
                continue
        if np.isfinite(ll) and ll > best_ll:
            best_m, best_ll = m, ll
    return best_m, best_ll


def fit_hmm(X: np.ndarray, candidate_states: list[int], restarts: int, seed: int,
            min_state_obs: int = 0):
    """Fit HMMs for each k; return the lowest-BIC model among those whose every state holds at
    least `min_state_obs` observations (decoded), plus the full comparison table.

    The occupancy floor guards against BIC choosing states too short to estimate anything
    conditional on them (a frequent outcome with full covariances on a few hundred months).
    """
    import logging

    logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # silence tolerance-level EM chatter
    T, d = X.shape
    rows, models = [], {}
    for k in candidate_states:
        m, ll = _fit_k(X, k, restarts, seed)
        occ = np.bincount(m.predict(X), minlength=k).min() if m is not None else 0
        rows.append({"k": k, "loglik": ll, "n_params": n_params(k, d),
                     "bic": -2 * ll + n_params(k, d) * np.log(T), "min_state_obs": int(occ),
                     "eligible": bool(occ >= min_state_obs)})
        models[k] = m
    table = pd.DataFrame(rows)
    ok = table[table.eligible] if table.eligible.any() else table
    k_best = int(ok.sort_values("bic").k.iloc[0])
    table["selected"] = table.k == k_best
    return models[k_best], table


def filtered_probabilities(model, X: np.ndarray) -> np.ndarray:
    """P(S_t | X_1..X_t): the forward recursion only, so no future data leaks into the state."""
    k = model.n_components
    B = np.column_stack([multivariate_normal(model.means_[j], model.covars_[j], allow_singular=True).pdf(X)
                         for j in range(k)])
    alpha = np.zeros_like(B)
    a = model.startprob_ * B[0]
    alpha[0] = a / a.sum()
    for t in range(1, len(X)):
        a = (alpha[t - 1] @ model.transmat_) * B[t]
        s = a.sum()
        alpha[t] = a / s if s > 0 else np.full(k, 1.0 / k)
    return alpha


def relabel_by(model, col: int) -> np.ndarray:
    """Permutation that orders states by the mean of column `col` (state 0 = lowest)."""
    return np.argsort(model.means_[:, col])


def common_factor(panel: pd.DataFrame, seed: int = 0) -> pd.Series:
    """One-factor dynamic factor model (state space, EM + Kalman smoother) on a standardized panel."""
    import statsmodels.api as sm

    Z = (panel - panel.mean()) / panel.std()
    Z = Z.asfreq(pd.infer_freq(Z.index) or "MS")
    mod = sm.tsa.DynamicFactorMQ(Z, factors=1, factor_orders=1, idiosyncratic_ar1=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(disp=False, maxiter=500)
    f = res.factors.smoothed.iloc[:, 0]
    # sign convention: factor moves with the first column
    if np.corrcoef(f.values, Z.iloc[:, 0].fillna(0).values)[0, 1] < 0:
        f = -f
    f.name = "common_factor"
    return f
