"""07 - Price response to supply surprises, linear and conditional on the macro regime.

Shock: standardized Permian-proxy production surprise from 05 (actual minus point-in-time nowcast).
Outcome: 100 x log WTI (monthly average). State: real-time (filtered) macro regime at t-1 from 06,
collapsed to two pre-specified groups so each has enough months:
  restrictive   = inflation-and-tightening or high-real-rates regimes
  accommodative = low-rate-expansion or recession-and-easing regimes
When satellite indices exist (corn NDVI, copper hotspots, Permian flares), the same code runs on them.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gmcn.config import DATA_PROC, OUTPUTS, ensure_dirs, settings
from gmcn.models import local_projections as lp
from gmcn.viz import style as st

RESTRICTIVE = {"Inflation and tightening", "High real rates"}


def main() -> None:
    ensure_dirs()
    st.apply()
    cfg = settings()["local_projections"]
    wti = pd.read_csv(DATA_PROC / "wti_monthly.csv", index_col=0, parse_dates=True)["wti"]
    y = 100 * np.log(wti)
    s = pd.read_csv(OUTPUTS / "supply_surprise.csv", index_col=0, parse_dates=True)["supply_surprise"]
    reg = pd.read_csv(OUTPUTS / "regimes_monthly.csv", index_col=0, parse_dates=True)
    prof = pd.read_csv(OUTPUTS / "regime_profiles.csv", index_col=0)
    label = prof["label"].to_dict()
    group = reg["state_filtered"].map(lambda r: "restrictive" if label.get(r) in RESTRICTIVE else "accommodative")
    state = group.shift(1)

    idx = s.index
    y_, s_ = y.reindex(pd.date_range(idx.min() - pd.DateOffset(months=6), y.index.max(), freq="MS")), s
    s_full = s_.reindex(y_.index)
    lin = lp.lp_linear(y_, s_full, cfg["horizons"], cfg["control_lags"], cfg["ci"])
    sd = lp.lp_state(y_, s_full, state.reindex(y_.index), cfg["horizons"], cfg["control_lags"], cfg["ci"])
    out = pd.concat([lin, sd])
    out.to_csv(OUTPUTS / "lp_wti_supply_surprise.csv", index=False)

    eq = sd[sd.state == "equality_test"].set_index("h")["hi"]
    summary = {"shock_sample": [str(s.index.min().date()), str(s.index.max().date())], "n_shock_months": int(s.notna().sum()),
               "state_months": state.reindex(s.index).value_counts().to_dict(),
               "linear_beta_h0_h6_h12": lin.set_index("h").loc[[0, 6, 12], ["beta", "lo", "hi"]].round(2).to_dict("index"),
               "min_equality_p": float(eq.min()), "h_of_min_equality_p": int(eq.idxmin())}
    (OUTPUTS / "lp_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    figure(lin, sd, summary)
    print(json.dumps(summary, indent=1, default=str))


def figure(lin: pd.DataFrame, sd: pd.DataFrame, summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    ax = axes[0]
    ax.fill_between(lin.h, lin.lo, lin.hi, color=st.SERIES[0], alpha=0.15, lw=0)
    ax.plot(lin.h, lin.beta, color=st.SERIES[0], marker="o", ms=3.5)
    ax.axhline(0, color=st.AXIS, lw=0.8)
    ax.set_xlabel("Months after the surprise")
    ax.set_ylabel("WTI response, % per 1-sd production surprise")
    ax.set_title("All months", fontsize=10, pad=4)
    ax2 = axes[1]
    for j, g in enumerate(["accommodative", "restrictive"]):
        d = sd[sd.state == g]
        off = (j - 0.5) * 0.18
        ax2.fill_between(d.h + off, d.lo, d.hi, color=st.SERIES[j + 1], alpha=0.14, lw=0)
        ax2.plot(d.h + off, d.beta, color=st.SERIES[j + 1], marker="o", ms=3.5,
                 label=f"{g.capitalize()} regime ({summary['state_months'].get(g, 0)} months)")
    ax2.axhline(0, color=st.AXIS, lw=0.8)
    ax2.set_xlabel("Months after the surprise")
    ax2.set_title("By real-time macro regime", fontsize=10, pad=4)
    ax2.legend(loc="lower left", fontsize=8)
    b0 = lin.set_index("h").loc[0]
    fig.suptitle(f"A 1-sd upside Permian supply surprise moves WTI {b0.beta:+.1f} % on impact "
                 f"(90 % CI {b0.lo:+.1f} to {b0.hi:+.1f})", x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.06)
    fig.text(0.01, 0.955, f"Local projections, 90 % HAC bands; shock = standardized nowcast error, "
             f"{summary['shock_sample'][0][:7]} to {summary['shock_sample'][1][:7]}.\nRegime gap is significant only at "
             f"h = 1-2 (p = {summary['min_equality_p']:.2f}) and the accommodative group is mostly 2016 and 2020-21.",
             fontsize=9, color=st.INK2)
    st.source_note(fig, "Source: EIA (TX + NM crude output), WTI spot via Alpha Vantage, FRED macro series. Controls: two lags of "
                        "the price change and of the shock.")
    st.save(fig, "fig10_lp_supply_surprise")


if __name__ == "__main__":
    main()
