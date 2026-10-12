"""Weekly market-model event studies around dated supply disruptions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as tdist


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna()


def event_week(index: pd.DatetimeIndex, event_date: str) -> pd.Timestamp:
    """First weekly bar (week-ending date) on or after the event date."""
    ts = pd.Timestamp(event_date)
    after = index[index >= ts]
    return after[0]


def market_model_car(r: pd.Series, r_m: pd.Series, event_date: str, est_weeks: int = 104,
                     gap_weeks: int = 4, car_weeks: int = 3) -> dict:
    """Abnormal returns vs a market model estimated on [-gap-est, -gap) weeks before the event.

    CAR covers weeks 0..car_weeks-1. The t-statistic uses the estimation-window residual variance.
    """
    df = pd.concat({"r": r, "m": r_m}, axis=1).dropna()
    e = event_week(df.index, event_date)
    i = df.index.get_loc(e)
    est = df.iloc[max(0, i - gap_weeks - est_weeks): i - gap_weeks]
    res = sm.OLS(est["r"], sm.add_constant(est["m"])).fit()
    a, b = res.params["const"], res.params["m"]
    sigma = np.sqrt(res.scale)
    win = df.iloc[i: i + car_weeks]
    ar = win["r"] - (a + b * win["m"])
    car = float(ar.sum())
    t_stat = car / (sigma * np.sqrt(len(ar)))
    return {
        "event_week": e.date().isoformat(),
        "alpha": float(a), "beta": float(b), "sigma_weekly": float(sigma),
        "ar_week0": float(ar.iloc[0]),
        "car": car, "t_stat": float(t_stat),
        "p_value": float(2 * (1 - tdist.cdf(abs(t_stat), df=len(est) - 2))),
        "weeks": int(len(ar)),
        "ar_path": ar,
    }
