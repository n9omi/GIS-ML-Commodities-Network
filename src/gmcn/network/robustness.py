"""Robustness of the supply network to supplier removal, scenario stress tests, and Monte Carlo
shock propagation with correlated (one-factor Gaussian copula) supplier failures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from gmcn.network.metrics import shares


# --------------------------------------------------------------------------- deterministic removal
def removal_curve(M: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Supply retained as suppliers are removed in `order`.

    retained_equal: mean over commodities of the share of supply left (each commodity counts once)
    retained_value: share of total import value left (large flows dominate)
    """
    S = shares(M)
    v = M.sum(axis=0).values
    removed = np.zeros(len(M.columns))
    rows = [{"n_removed": 0, "frac_removed": 0.0, "retained_equal": 1.0, "retained_value": 1.0}]
    for i, c in enumerate(order, start=1):
        removed += S.loc[c].values
        left = np.clip(1.0 - removed, 0.0, 1.0)
        rows.append({"n_removed": i, "frac_removed": i / len(order),
                     "retained_equal": float(left.mean()),
                     "retained_value": float((left * v).sum() / v.sum())})
    return pd.DataFrame(rows)


def robustness_index(curve: pd.DataFrame, col: str = "retained_equal") -> float:
    """R = mean retained supply over all removal steps (Schneider et al. 2011); 0.5 = linear decay."""
    return float(curve[col].iloc[1:].mean())


def random_curves(M: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Mean and 5-95 % band of retained supply under random removal orders."""
    rng = np.random.default_rng(seed)
    countries = list(M.index)
    mats = []
    for _ in range(n):
        order = list(rng.permutation(countries))
        mats.append(removal_curve(M, order)["retained_equal"].values)
    A = np.vstack(mats)
    base = removal_curve(M, countries)[["n_removed", "frac_removed"]]
    base["mean"] = A.mean(axis=0)
    base["p05"] = np.quantile(A, 0.05, axis=0)
    base["p95"] = np.quantile(A, 0.95, axis=0)
    base["R_mean"] = float(A[:, 1:].mean())
    return base


def single_supplier_loss(M: pd.DataFrame) -> pd.DataFrame:
    """Share of each commodity's supply lost if one country stops exporting (= the share matrix)."""
    return shares(M)


def scenario_loss(M: pd.DataFrame, countries: list[str], labels: dict) -> pd.DataFrame:
    """Supply lost per commodity if every country in `countries` stops exporting."""
    S = shares(M)
    hit = [c for c in countries if c in S.index]
    loss = S.loc[hit].sum(axis=0) if hit else S.iloc[0] * 0
    out = pd.DataFrame({"hts6": loss.index, "loss_share": loss.values,
                        "value_at_risk_usd": loss.values * M.sum(axis=0).values})
    out["commodity"] = out.hts6.map(lambda k: labels.get(k, {}).get("label", k))
    return out.sort_values("loss_share", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- stochastic propagation
def simulate_losses(M: pd.DataFrame, p: float, rho: float, n_sims: int, seed: int,
                    severity: float = 1.0) -> np.ndarray:
    """Monte Carlo supply losses (n_sims x commodities) under correlated supplier failures.

    Each supplier c fails when Z_c = sqrt(rho) F + sqrt(1 - rho) e_c < Phi^-1(p). The common
    factor F stands for a systemic geopolitical shock; rho = 0 means independent failures.
    Loss_k = severity * sum_c s_ck * 1[c fails].
    """
    rng = np.random.default_rng(seed)
    S = shares(M).values
    n_c = S.shape[0]
    thr = norm.ppf(p)
    F = rng.standard_normal((n_sims, 1))
    E = rng.standard_normal((n_sims, n_c))
    Z = np.sqrt(rho) * F + np.sqrt(1.0 - rho) * E
    fail = (Z < thr).astype(float)
    return severity * fail @ S


def supply_at_risk(losses: np.ndarray, columns: list[str], q: float = 0.95) -> pd.DataFrame:
    """Expected loss, VaR_q, expected shortfall_q and P(loss > 25 %) per commodity."""
    var = np.quantile(losses, q, axis=0)
    es = np.array([losses[losses[:, k] >= var[k], k].mean() for k in range(losses.shape[1])])
    return pd.DataFrame({
        "hts6": columns,
        "expected_loss": losses.mean(axis=0),
        f"var_{int(q * 100)}": var,
        f"es_{int(q * 100)}": es,
        "p_loss_gt_25pct": (losses > 0.25).mean(axis=0),
    })
