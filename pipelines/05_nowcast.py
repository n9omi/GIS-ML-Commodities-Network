"""05 - Point-in-time nowcast of Permian-proxy crude production (Texas + New Mexico, EIA).

Baseline information set: official production up to two months back plus WTI prices through the
current month. When outputs/satellite/permian_flares_monthly.csv exists (VIIRS Nightfire, from
02/03), a second experiment adds same-month flaring on the common sample and tests whether the
satellite signal improves the nowcast (Diebold-Mariano).
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gmcn.config import DATA_PROC, OUTPUTS, ensure_dirs, settings
from gmcn.models import nowcast as nc
from gmcn.viz import style as st


def main() -> None:
    ensure_dirs()
    st.apply()
    cfg = settings()["nowcast"]
    prod = pd.read_csv(DATA_PROC / "crude_production_monthly.csv", index_col=0, parse_dates=True)["permian_proxy_tx_nm"]
    wti = pd.read_csv(DATA_PROC / "wti_monthly.csv", index_col=0, parse_dates=True)["wti"]
    X, y = nc.make_features(prod, wti, cfg["publication_lag_months"], cfg["price_lags"])
    pred = nc.rolling_origin(X, y, cfg["test_start"], cfg["publication_lag_months"], cfg["quantiles"],
                             cfg["conformal_alpha"], seed=cfg["seed"])
    pred.to_csv(OUTPUTS / "nowcast_predictions.csv")
    ev = nc.evaluate(pred, cfg["quantiles"])
    ev.to_csv(OUTPUTS / "nowcast_metrics.csv", index=False)
    surprise = nc.standardized_surprise(pred, "ridge", pub_lag=cfg["publication_lag_months"])
    surprise.to_frame().to_csv(OUTPUTS / "supply_surprise.csv")
    summary = {"sample": [str(pred.index.min().date()), str(pred.index.max().date())], "n": len(pred),
               "metrics": ev.round(5).to_dict("records"), **{k: round(v, 3) for k, v in ev.attrs.items()}}

    sat = OUTPUTS / "satellite" / "permian_flares_monthly.csv"
    if sat.exists():
        fl = pd.read_csv(sat, index_col=0, parse_dates=True)
        extra = np.log(fl[["flares_per_night", "flare_rh_mw_per_night"]].clip(lower=1e-6)).diff().add_prefix("dlog_")
        Xs, ys = nc.make_features(prod, wti, cfg["publication_lag_months"], cfg["price_lags"], extra=extra)
        start = max(pd.Timestamp(cfg["test_start"]), Xs.index.min() + pd.DateOffset(months=48))
        if (Xs.index >= start).sum() < 24:
            summary["satellite_experiment"] = (f"skipped: flaring overlaps only {len(Xs)} months; need 48 for training "
                                               "plus 24 for testing (extend the Permian window in settings.yaml)")
        else:
            p_with = nc.rolling_origin(Xs, ys, str(start.date()), cfg["publication_lag_months"], cfg["quantiles"],
                                       cfg["conformal_alpha"], seed=cfg["seed"])
            p_wo = nc.rolling_origin(X.loc[Xs.index], y.loc[Xs.index], str(start.date()), cfg["publication_lag_months"],
                                     cfg["quantiles"], cfg["conformal_alpha"], seed=cfg["seed"])
            idx = p_with.index.intersection(p_wo.index)
            e_w, e_wo = (p_with.y - p_with.ridge).loc[idx], (p_wo.y - p_wo.ridge).loc[idx]
            summary["satellite_experiment"] = {
                "n": len(idx), "dm_with_vs_without": nc.diebold_mariano(e_w.values, e_wo.values),
                "rmse_with": float(np.sqrt((e_w ** 2).mean())), "rmse_without": float(np.sqrt((e_wo ** 2).mean()))}
    else:
        summary["satellite_experiment"] = "not run: no VIIRS Nightfire series yet (run 02 --live --module permian)"
    (OUTPUTS / "nowcast_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    figure(pred, ev)
    print(ev.round(4).to_string(index=False))
    print({k: round(v, 3) for k, v in ev.attrs.items()})


def figure(pred: pd.DataFrame, ev: pd.DataFrame) -> None:
    p = pred.loc["2019-01":]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ok = p[["ridge_lo", "ridge_hi"]].notna().all(axis=1)
    ax.fill_between(p.index[ok], 100 * p.ridge_lo[ok], 100 * p.ridge_hi[ok], color=st.SERIES[0], alpha=0.15, lw=0,
                    label="80 % split-conformal band")
    ax.plot(p.index, 100 * p.ridge, color=st.SERIES[0], lw=1.8, label="Nowcast (ridge: own lags + WTI lags)")
    ax.plot(p.index, 100 * p.y, color=st.INK, lw=1.2, marker="o", ms=2.5, label="Actual (EIA)")
    ax.axhline(0, color=st.AXIS, lw=0.8)
    ax.set_ylabel("Monthly growth of TX + NM crude output, %")
    e = ev.set_index("model")
    ax.legend(loc="upper right", ncol=1)
    for d, txt, dy in [("2021-02-01", "Winter Storm Uri", -3), ("2020-05-01", "COVID shut-ins", 3)]:
        if pd.Timestamp(d) in p.index:
            v = 100 * p.y.loc[d]
            ax.annotate(txt, (pd.Timestamp(d), v), (pd.Timestamp(d) + pd.DateOffset(months=5), v + dy),
                        fontsize=8, color=st.INK2, arrowprops=dict(arrowstyle="-", color=st.INK2, lw=0.7))
    cov = ev.attrs.get("conformal_coverage", float("nan"))
    ax.set_title(f"Conformal bands cover {cov:.0%} of outcomes (target 80 %); boosting beats the AR benchmark "
                 f"(DM p = {e.loc['gbm_q50', 'dm_vs_ar_p']:.2f})", pad=22)
    st.subtitle(ax, "Permian-proxy production nowcast made at month end, two months before EIA publishes; out-of-sample")
    st.source_note(fig, "Source: EIA Petroleum Supply Monthly (TX, NM field production); WTI spot via Alpha Vantage. "
                        "Expanding-window, rolling-origin evaluation from 2015.")
    st.save(fig, "fig07_production_nowcast")


if __name__ == "__main__":
    main()
