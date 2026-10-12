"""06 - Macro-financial regimes (Gaussian HMM, BIC-selected) and commodity returns by regime.

Panel (monthly, standardized): 10-yr real yield (TIPS), 12-month change in fed funds, CPI
inflation (y/y), 12-month change in unemployment, broad-dollar change (y/y). October 2025 CPI and
unemployment were never published (federal shutdown); those two points are linearly interpolated.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gmcn.config import DATA_PROC, OUTPUTS, ensure_dirs, settings
from gmcn.models import regimes as rg
from gmcn.viz import style as st

VARS = {"real10": "10-yr real yield", "d12_fedfunds": "Fed funds, 12-m change", "cpi_yoy": "CPI inflation, y/y",
        "d12_unrate": "Unemployment, 12-m change", "usd_yoy": "Broad dollar, y/y"}


def describe(r: pd.Series) -> str:
    """Plain-language name for a regime from its mean profile (rule-based, documented)."""
    if r["d12_unrate"] > 1.0:
        return "Recession and easing"
    if r["cpi_yoy"] > 4.0:
        return "Inflation and tightening"
    if r["real10"] > 1.0:
        return "High real rates"
    return "Low-rate expansion"


def panel(start: str) -> pd.DataFrame:
    m = pd.read_csv(DATA_PROC / "macro_monthly.csv", index_col=0, parse_dates=True)
    m[["cpi", "unrate"]] = m[["cpi", "unrate"]].interpolate(limit=1, limit_area="inside")
    m["cpi_yoy"] = 100 * (np.log(m["cpi"]) - np.log(m["cpi"].shift(12)))
    m["d12_fedfunds"] = m["fedfunds"].diff(12)
    m["d12_unrate"] = m["unrate"].diff(12)
    return m.loc[start:, list(VARS)].dropna()


def main() -> None:
    ensure_dirs()
    st.apply()
    cfg = settings()["regimes"]
    P = panel(cfg["start"])
    Z = (P - P.mean()) / P.std()
    model, bic = rg.fit_hmm(Z.values, cfg["candidate_states"], cfg["restarts"], cfg["seed"],
                            min_state_obs=cfg["min_state_months"])
    bic.to_csv(OUTPUTS / "regime_bic.csv", index=False)
    order = rg.relabel_by(model, list(VARS).index("real10"))
    filt = rg.filtered_probabilities(model, Z.values)[:, order]
    smooth = model.predict_proba(Z.values)[:, order]
    k = model.n_components
    names = [f"R{i + 1}" for i in range(k)]
    reg = pd.DataFrame(filt, index=P.index, columns=[f"p_{n}" for n in names])
    reg["state_filtered"] = np.array(names)[filt.argmax(axis=1)]
    reg["state_smoothed"] = np.array(names)[smooth.argmax(axis=1)]
    reg["dfm_factor"] = rg.common_factor(P)
    reg.index.name = "month"
    reg.to_csv(OUTPUTS / "regimes_monthly.csv")

    prof = P.join(reg["state_smoothed"]).groupby("state_smoothed").mean().round(2)
    prof["label"] = [describe(r) for _, r in prof.iterrows()]
    prof["months"] = reg["state_smoothed"].value_counts()
    prof["persistence"] = np.diag(model.transmat_[np.ix_(order, order)]).round(3)
    prof.to_csv(OUTPUTS / "regime_profiles.csv")

    # commodity returns in the month after each filtered state (tradable conditioning)
    wti = pd.read_csv(DATA_PROC / "wti_monthly.csv", index_col=0, parse_dates=True)["wti"]
    wk = pd.read_csv(DATA_PROC / "prices_weekly.csv", index_col=0, parse_dates=True)
    px = pd.concat({"WTI crude": wti,
                    "Copper (CPER)": wk["CPER"].resample("MS").last(),
                    "Corn (CORN)": wk["CORN"].resample("MS").last()}, axis=1)
    ret = 100 * np.log(px).diff()
    st_prev = reg["state_filtered"].shift(1).reindex(ret.index)
    rows = []
    for c in ret.columns:
        for s in names:
            r = ret[c][st_prev == s].dropna()
            if len(r) < 6:
                continue
            rows.append({"asset": c, "state": s, "n": len(r), "mean_pct": r.mean(), "vol_pct": r.std(),
                         "se": r.std() / np.sqrt(len(r)), "ann_mean_pct": 12 * r.mean()})
    rbr = pd.DataFrame(rows)
    rbr.to_csv(OUTPUTS / "returns_by_regime.csv", index=False)

    summary = {"k": k, "bic": bic.round(1).to_dict("records"), "profiles": prof.reset_index().to_dict("records"),
               "current_state_filtered": reg["state_filtered"].iloc[-1],
               "current_probs": reg[[f"p_{n}" for n in names]].iloc[-1].round(3).to_dict(),
               "last_month": str(reg.index[-1].date()),
               "agreement_filtered_vs_smoothed": float((reg.state_filtered == reg.state_smoothed).mean())}
    (OUTPUTS / "regime_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    fig_regimes(P, reg, names, prof)
    fig_returns(rbr, names, prof)
    print(bic.round(1).to_string(index=False))
    print(prof.to_string())
    print(rbr.round(2).to_string(index=False))


def _shade(ax, reg: pd.DataFrame, names: list[str]) -> None:
    s = reg["state_smoothed"]
    starts = s.index[s != s.shift()]
    for i, t0 in enumerate(starts):
        t1 = starts[i + 1] if i + 1 < len(starts) else s.index[-1] + pd.DateOffset(months=1)
        ax.axvspan(t0, t1, color=st.SERIES[names.index(s.loc[t0])], alpha=0.14, lw=0)


def fig_regimes(P: pd.DataFrame, reg: pd.DataFrame, names: list[str], prof: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.6), sharex=True, gridspec_kw={"height_ratios": [1.2, 1, 1]})
    ax = axes[0]
    _shade(ax, reg, names)
    ax.plot(P.index, P["real10"], color=st.INK, lw=1.5)
    ax.set_ylabel("Real yield, %")
    for i, n in enumerate(names):
        ax.fill_between([], [], color=st.SERIES[i], alpha=0.35,
                        label=f"{n} {prof.loc[n, 'label']}: real {prof.loc[n, 'real10']:+.1f} %, "
                              f"CPI {prof.loc[n, 'cpi_yoy']:.1f} %, fed funds {prof.loc[n, 'd12_fedfunds']:+.1f} pp")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=2, fontsize=8)
    ax.set_title("Macro regimes: the HMM separates easing, tightening and inflation episodes", pad=22)
    st.subtitle(ax, f"{len(names)}-state Gaussian HMM (lowest BIC with every state >= 20 months) on five standardized "
                    f"macro series; shading = smoothed state")
    ax2 = axes[1]
    _shade(ax2, reg, names)
    ax2.plot(P.index, P["cpi_yoy"], color=st.SERIES[1], lw=1.5, label="CPI inflation, y/y %")
    ax2.plot(P.index, P["d12_fedfunds"], color=st.SERIES[0], lw=1.5, label="Fed funds, 12-m change, pp")
    ax2.axhline(0, color=st.AXIS, lw=0.8)
    ax2.legend(loc="upper left", fontsize=8)
    ax3 = axes[2]
    for i, n in enumerate(names):
        ax3.plot(reg.index, reg[f"p_{n}"], color=st.SERIES[i], lw=1.2, label=f"P({n}) filtered")
    ax3.set_ylim(0, 1.02)
    ax3.set_ylabel("Real-time P(state)")
    st.source_note(fig, "Source: FRED (DFII10, FEDFUNDS, UNRATE, DTWEXBGS), BLS CPI via Alpha Vantage. "
                        "Filtered = forward recursion only (no look-ahead in the state).")
    st.save(fig, "fig08_macro_regimes")


def fig_returns(rbr: pd.DataFrame, names: list[str], prof: pd.DataFrame) -> None:
    assets = list(dict.fromkeys(rbr.asset))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    w = 0.8 / len(names)
    x = np.arange(len(assets))
    for i, n in enumerate(names):
        d = rbr[rbr.state == n].set_index("asset").reindex(assets)
        ax.bar(x + (i - (len(names) - 1) / 2) * w, d.ann_mean_pct, width=w * 0.92, color=st.SERIES[i],
               label=f"{n} {prof.loc[n, 'label']}")
        ax.errorbar(x + (i - (len(names) - 1) / 2) * w, d.ann_mean_pct, yerr=1.645 * 12 * d.se, fmt="none",
                    ecolor=st.INK2, lw=1, capsize=0)
    ax.axhline(0, color=st.AXIS, lw=0.8)
    ax.set_xticks(x, assets)
    ax.set_ylabel("Annualized mean log return, %")
    ax.grid(axis="x", visible=False)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=8)
    sig = rbr[(rbr.ann_mean_pct - 1.645 * 12 * rbr.se > 0) | (rbr.ann_mean_pct + 1.645 * 12 * rbr.se < 0)]
    if len(sig):
        s0 = sig.iloc[0]
        lab = prof.loc[s0.state, "label"].lower()
        title = (f"Next-month returns separate by regime only after {lab} states, and clearly only for "
                 f"{', '.join(sig.asset.str.split(' ').str[0].str.lower().unique())}")
    else:
        title = "Next-month commodity returns do not separate by real-time macro regime"
    ax.set_title(title, pad=22)
    st.subtitle(ax, "State known at month end (filtered); bars = mean x 12; whiskers = 90 % CI. Wide intervals: few months per state")
    st.source_note(fig, "Source: WTI spot (EIA via Alpha Vantage); CPER and CORN fund prices (weekly closes, not dividend-adjusted).")
    st.save(fig, "fig09_returns_by_regime")


if __name__ == "__main__":
    main()
