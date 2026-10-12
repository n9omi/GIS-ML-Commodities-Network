"""04 - Supply-network theory on official US trade flows (USITC DataWeb, HTS6, 2023 and 2025).

Outputs: concentration tables, country centrality, commodity co-dependency and communities,
nestedness vs a fixed-degree null, robustness curves, scenario stress tests, Monte Carlo
supply-at-risk, and the export-side (buyer) concentration layer.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from gmcn.config import DATA_PROC, OUTPUTS, commodities, ensure_dirs, settings
from gmcn.network import metrics as nm
from gmcn.network import robustness as rb
from gmcn.viz import style as st

SRC = "Source: USITC DataWeb (general imports, customs value; total exports, FAS value), HTS 6-digit."


def load(flow: str) -> pd.DataFrame:
    f = "trade_imports_hts6.csv" if flow == "import" else "trade_exports_hts6.csv"
    return pd.read_csv(DATA_PROC / f, dtype={"hts6": str, "year": str})


def short(code: str, labels: dict) -> str:
    return labels.get(code, {}).get("label", code)


def main() -> None:
    ensure_dirs()
    st.apply()
    cfg = settings()["network"]
    labels = commodities()["imports"]
    exp_labels = commodities()["exports"]
    imp = load("import")
    fy, py = cfg["focal_year"], [y for y in cfg["years"] if y != cfg["focal_year"]][0]
    summary: dict = {"focal_year": fy, "prior_year": py}

    # ------------------------------------------------------------------ concentration
    M = {y: nm.trade_matrix(imp, y, labels) for y in cfg["years"]}
    conc = {y: nm.concentration_table(M[y], labels, cfg["incidence_min_share"]) for y in cfg["years"]}
    for y in cfg["years"]:
        conc[y].to_csv(OUTPUTS / f"network_concentration_{y}.csv", index=False)
    comp = conc[fy].merge(conc[py][["hts6", "hhi", "n_eff_hhi", "top1_country", "top1_share", "value_usd"]],
                          on="hts6", suffixes=("", f"_{py}"))
    comp["hhi_change"] = comp["hhi"] - comp[f"hhi_{py}"]
    comp["top1_changed"] = comp["top1_country"] != comp[f"top1_country_{py}"]
    comp.to_csv(OUTPUTS / "network_concentration_change.csv", index=False)
    summary["n_lines"] = int(len(comp))
    summary["n_lines_neff_below_2"] = int((comp.n_eff_hhi < 2).sum())
    summary["top1_changes"] = comp.loc[comp.top1_changed, ["commodity", f"top1_country_{py}", "top1_country", "value_usd"]] \
        .rename(columns={f"top1_country_{py}": "from", "top1_country": "to"}).to_dict("records")
    fig01(comp, fy, py, summary)

    # ------------------------------------------------------------------ centrality, communities, nestedness
    Mf = M[fy]
    ct = nm.country_table(Mf, cfg["incidence_min_share"])
    ct.to_csv(OUTPUTS / f"network_countries_{fy}.csv", index=False)
    C = nm.commodity_projection(Mf)
    comms = nm.commodity_communities(C, threshold=0.3)
    C.to_csv(OUTPUTS / f"network_codependency_{fy}.csv")
    summary["communities"] = [sorted(short(k, labels) for k in c) for c in comms if len(c) > 1]
    B = (nm.shares(Mf) >= cfg["incidence_min_share"]).astype(int).values
    nest = nm.nodf_null(B, n=cfg["null_model_samples"], seed=3)
    summary["nestedness"] = nest
    summary["top_betweenness"] = ct.sort_values("betweenness", ascending=False).head(5)[["country", "betweenness"]] \
        .round(3).to_dict("records")
    summary["top_eigenvector"] = ct.sort_values("eigenvector", ascending=False).head(5)[["country", "eigenvector"]] \
        .round(3).to_dict("records")
    fig02(C, comms, labels, fy)
    fig_network(Mf, labels, fy)

    # ------------------------------------------------------------------ robustness
    order_strength = list(ct.sort_values("strength_usd", ascending=False).country)
    order_top = list(ct.sort_values(["commodities_top1", "max_share"], ascending=False).country)
    cur_s = rb.removal_curve(Mf, order_strength)
    cur_t = rb.removal_curve(Mf, order_top)
    rnd = rb.random_curves(Mf, cfg["robustness_random_permutations"], seed=5)
    pd.concat({"targeted_strength": cur_s, "targeted_dominance": cur_t}).to_csv(OUTPUTS / f"network_robustness_{fy}.csv")
    rnd.to_csv(OUTPUTS / f"network_robustness_random_{fy}.csv", index=False)
    summary["robustness_R"] = {"targeted_strength": rb.robustness_index(cur_s),
                               "targeted_dominance": rb.robustness_index(cur_t),
                               "random_mean": float(rnd.R_mean.iloc[0])}
    summary["retained_after_top3_dominance"] = float(cur_t.retained_equal.iloc[3])
    summary["retained_after_top3_strength"] = float(cur_s.retained_equal.iloc[3])
    summary["top3_dominance_order"] = order_top[:3]
    summary["top3_strength_order"] = order_strength[:3]
    fig03(cur_s, cur_t, rnd, fy, summary)

    # ------------------------------------------------------------------ scenarios
    scen_rows = []
    for sc in cfg["scenarios"]:
        for y in cfg["years"]:
            s = rb.scenario_loss(M[y], sc["countries"], labels)
            s["scenario"], s["year"] = sc["name"], y
            scen_rows.append(s)
    scen = pd.concat(scen_rows)
    scen.to_csv(OUTPUTS / "network_scenarios.csv", index=False)
    fig05(scen, cfg["scenarios"][0]["name"], fy, py, summary)

    # ------------------------------------------------------------------ Monte Carlo supply-at-risk
    mc = cfg["monte_carlo"]
    sar = []
    for rho in mc["correlations"]:
        L = rb.simulate_losses(Mf, mc["annual_failure_prob"], rho, mc["n_sims"], mc["seed"])
        t = rb.supply_at_risk(L, list(Mf.columns))
        t["rho"] = rho
        sar.append(t)
    sar = pd.concat(sar).merge(conc[fy][["hts6", "commodity", "hhi"]], on="hts6")
    sar.to_csv(OUTPUTS / f"network_supply_at_risk_{fy}.csv", index=False)
    summary["mc"] = {"p": mc["annual_failure_prob"], "n_sims": mc["n_sims"],
                     "corr_es95_vs_hhi_rho0": float(np.corrcoef(sar[sar.rho == 0].es_95, sar[sar.rho == 0].hhi)[0, 1]),
                     "expected_loss_range": [float(sar.expected_loss.min()), float(sar.expected_loss.max())]}
    fig04(sar, mc, fy)

    # ------------------------------------------------------------------ export side: buyer concentration
    ex = load("export")
    Mx = {y: nm.trade_matrix(ex, y, exp_labels) for y in cfg["years"]}
    cx = {y: nm.concentration_table(Mx[y], exp_labels) for y in cfg["years"]}
    cxx = cx[fy].merge(cx[py][["hts6", "n_eff_hhi", "top1_country", "top1_share"]], on="hts6", suffixes=("", f"_{py}"))
    cxx.to_csv(OUTPUTS / "network_export_buyers.csv", index=False)
    china = pd.concat([rb.scenario_loss(Mx[y], ["China"], exp_labels).assign(year=y) for y in cfg["years"]])
    china.to_csv(OUTPUTS / "network_export_china_share.csv", index=False)
    fig06(cxx, china, fy, py)

    (OUTPUTS / "network_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("n_lines_neff_below_2", "robustness_R", "nestedness")}, indent=1))


# ====================================================================== figures
def fig01(comp: pd.DataFrame, fy: str, py: str, summary: dict) -> None:
    d = comp.sort_values("n_eff_hhi", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 7.2))
    y = np.arange(len(d))
    ax.barh(y, d.n_eff_hhi, height=0.62, color=st.SERIES[0], label=f"{fy}")
    ax.scatter(d[f"n_eff_hhi"].values * 0 + d[f"n_eff_hhi_{py}"].values, y, s=26, facecolor=st.SURFACE,
               edgecolor=st.INK, linewidth=1.2, zorder=3, label=f"{py}")
    ax.axvline(2, color=st.AXIS, lw=1)
    ax.set_ylim(-0.7, len(d) + 0.3)
    ax.text(2.05, len(d) - 0.15, "two equal suppliers", color=st.INK2, fontsize=8, va="bottom")
    ax.set_yticks(y, d.commodity)
    for i, r in d.iterrows():
        ax.text(max(r.n_eff_hhi, r[f"n_eff_hhi_{py}"]) + 0.15, i, f"{r.top1_country} {r.top1_share:.0%}",
                va="center", fontsize=8, color=st.INK2)
    ax.set_xlim(0, d[["n_eff_hhi", f"n_eff_hhi_{py}"]].max().max() + 2.2)
    ax.set_xlabel("Effective number of suppliers, 1 / HHI")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title(f"{summary['n_lines_neff_below_2']} of {summary['n_lines']} US import lines behave like fewer than two suppliers",
                 pad=22)
    st.subtitle(ax, f"Supplier concentration of US imports, {fy} (bars) vs {py} (rings); label = top supplier and share in {fy}")
    st.source_note(fig, SRC)
    st.save(fig, "fig01_import_concentration")


def fig02(C: pd.DataFrame, comms: list[set], labels: dict, fy: str) -> None:
    order = [k for c in sorted(comms, key=len, reverse=True) for k in sorted(c, key=lambda k: -C.loc[k].sum())]
    D = C.loc[order, order]
    cmap = LinearSegmentedColormap.from_list("seq", [st.SURFACE] + st.SEQ[2:])
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(D.values, cmap=cmap, vmin=0, vmax=1)
    names = [short(k, labels) for k in order]
    ax.set_xticks(range(len(order)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)), names, fontsize=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    edge = 0
    for c in sorted(comms, key=len, reverse=True):
        n = len(c)
        if n > 1:
            ax.add_patch(plt.Rectangle((edge - 0.5, edge - 0.5), n, n, fill=False, ec=st.INK, lw=1.2))
        edge += n
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Supplier-profile similarity (cosine)", color=st.INK2)
    cb.outline.set_visible(False)
    ax.set_title("Commodities that share suppliers would fail together", pad=22)
    st.subtitle(ax, f"Co-dependency of US import lines, {fy}; boxes = modularity communities (edges at similarity >= 0.3)")
    st.source_note(fig, SRC)
    st.save(fig, "fig02_codependency")


def fig_network(M: pd.DataFrame, labels: dict, fy: str, min_share: float = 0.10) -> None:
    S = nm.shares(M)
    hubs = {"China": st.SERIES[1], "Canada": st.SERIES[0], "Indonesia": st.SERIES[2]}
    links = [(c, k, S.at[c, k]) for c in S.index for k in S.columns if S.at[c, k] >= min_share]
    countries = sorted({c for c, _, _ in links}, key=lambda c: -sum(s for cc, _, s in links if cc == c))
    groups = commodities()["imports"]
    coms = sorted(S.columns, key=lambda k: (groups[k]["group"], groups[k]["label"]))
    yk = {k: i for i, k in enumerate(coms)}
    yc = {c: i * (len(coms) - 1) / max(1, len(countries) - 1) for i, c in enumerate(countries)}
    fig, ax = plt.subplots(figsize=(10, 8.5))
    for c, k, s in sorted(links, key=lambda x: x[0] in hubs):
        col = hubs.get(c, st.AXIS)
        ax.plot([0, 1], [yk[k], yc[c]], color=col, lw=0.6 + 4.5 * s, alpha=0.85 if c in hubs else 0.6,
                solid_capstyle="round")
    for k, yy in yk.items():
        ax.text(-0.02, yy, short(k, labels), ha="right", va="center", fontsize=8, color=st.INK)
    for c, yy in yc.items():
        ax.text(1.02, yy, c, ha="left", va="center", fontsize=8,
                color=st.INK if c in hubs else st.INK2, fontweight="bold" if c in hubs else "normal")
    ax.scatter([0] * len(yk), list(yk.values()), s=14, color=st.INK, zorder=3)
    ax.scatter([1] * len(yc), list(yc.values()), s=14, color=st.INK, zorder=3)
    for name, col in hubs.items():
        ax.plot([], [], color=col, lw=3, label=name)
    ax.plot([], [], color=st.AXIS, lw=3, label="Other suppliers")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4)
    ax.set_xlim(-0.45, 1.3)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("Three supplier hubs anchor the concentrated links", pad=22)
    st.subtitle(ax, f"Bipartite supply network, US imports {fy}: links carrying >= {min_share:.0%} of a commodity's imports; width = share")
    st.source_note(fig, SRC)
    st.save(fig, "fig00_supply_network")


def fig03(cur_s, cur_t, rnd, fy: str, summary: dict) -> None:
    n = 15
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.fill_between(rnd.n_removed[: n + 1], rnd.p05[: n + 1], rnd.p95[: n + 1], color=st.AXIS, alpha=0.45, lw=0,
                    label="Random removal, 5-95 %")
    ax.plot(rnd.n_removed[: n + 1], rnd["mean"][: n + 1], color=st.INK2, lw=1.5, label="Random removal, mean")
    ax.plot(cur_s.n_removed[: n + 1], cur_s.retained_equal[: n + 1], color=st.SERIES[0], marker="o", ms=4,
            label="Targeted: largest trade value first")
    ax.plot(cur_t.n_removed[: n + 1], cur_t.retained_equal[: n + 1], color=st.SERIES[1], marker="o", ms=4,
            label="Targeted: most dominant supplier first")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, n)
    ax.set_xlabel("Supplier countries removed")
    ax.set_ylabel("Share of supply retained (mean over commodities)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    r3 = summary["retained_after_top3_dominance"]
    ax.annotate(f"{', '.join(summary['top3_dominance_order'])} out: {r3:.0%} left", (3, r3), (5.2, r3 - 0.12),
                fontsize=8.5, color=st.INK, arrowprops=dict(arrowstyle="-", color=st.INK2, lw=0.8))
    ax.legend(loc="upper right")
    rnd3 = float(rnd["mean"].iloc[3])
    ax.set_title(f"Losing the three dominant suppliers removes {1 - r3:.0%} of supply; "
                 f"three random losses remove {1 - rnd3:.0%}", pad=22)
    st.subtitle(ax, f"Robustness of the {fy} US import network to supplier removal; each commodity weighted equally")
    st.source_note(fig, SRC)
    st.save(fig, "fig03_robustness")


def fig04(sar: pd.DataFrame, mc: dict, fy: str) -> None:
    order = sar[sar.rho == 0].sort_values("hhi").commodity.tolist()
    y = {c: i for i, c in enumerate(order)}
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for j, rho in enumerate(mc["correlations"]):
        d = sar[sar.rho == rho]
        ax.scatter(d.es_95, d.commodity.map(y) + (j - 1) * 0.22, s=30, color=st.SERIES[j], zorder=3,
                   label=f"Correlation rho = {rho}")
    ax.axvline(mc["annual_failure_prob"], color=st.INK, lw=1)
    ax.text(mc["annual_failure_prob"] + 0.01, len(order) - 0.3, "expected loss = 5 % for every line",
            fontsize=8, color=st.INK, va="top")
    ax.set_yticks(range(len(order)), order, fontsize=8.5)
    ax.set_xlabel("Expected shortfall of supply lost, worst 5 % of years")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0, 1.0)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Same expected loss, very different tails", pad=22)
    st.subtitle(ax, f"Monte Carlo supply-at-risk, {fy} shares; each supplier fails with p = {mc['annual_failure_prob']:.0%} a year; "
                f"rows ordered by HHI (top = most concentrated)")
    st.source_note(fig, SRC + f" Simulation: {mc['n_sims']:,} draws, one-factor Gaussian copula.")
    st.save(fig, "fig04_supply_at_risk")


def fig05(scen: pd.DataFrame, name: str, fy: str, py: str, summary: dict) -> None:
    d = scen[scen.scenario == name].pivot_table(index="commodity", columns="year", values="loss_share").fillna(0)
    d = d[d[fy].add(d[py]) > 0.02].sort_values(fy)
    summary["china_scenario"] = d[fy].sort_values(ascending=False).round(3).to_dict()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(d))
    ax.barh(y, d[fy], height=0.6, color=st.SERIES[1], label=fy)
    ax.scatter(d[py], y, s=26, facecolor=st.SURFACE, edgecolor=st.INK, lw=1.2, zorder=3, label=py)
    for i, (k, r) in enumerate(d.iterrows()):
        x = r[fy] + 0.012
        if r[fy] <= r[py] <= r[fy] + 0.07:
            x = r[py] + 0.022
        ax.text(x, i, f"{r[fy]:.0%}", va="center", fontsize=8, color=st.INK2)
    ax.set_yticks(y, d.index)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of US imports lost")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    big = d[d[fy] > 0.5]
    ax.set_title(f"A Chinese export stop would remove {big[fy].min():.0%}-{big[fy].max():.0%} of US imports "
                 f"in {len(big)} lines", pad=22)
    st.subtitle(ax, f"Scenario: China stops exporting to the US; bars {fy}, rings {py}; lines with > 2 % exposure")
    st.source_note(fig, SRC)
    st.save(fig, "fig05_china_scenario")


def fig06(cxx: pd.DataFrame, china: pd.DataFrame, fy: str, py: str) -> None:
    d = cxx.sort_values("n_eff_hhi", ascending=False).reset_index(drop=True)
    ch = china[china.year == fy].set_index("hts6").loss_share
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(d))
    ax.barh(y, d.n_eff_hhi, height=0.6, color=st.SERIES[0], label=fy)
    ax.scatter(d[f"n_eff_hhi_{py}"], y, s=26, facecolor=st.SURFACE, edgecolor=st.INK, lw=1.2, zorder=3, label=py)
    for i, r in d.iterrows():
        txt = f"top buyer {r.top1_country} {r.top1_share:.0%}"
        if r.top1_country != "China":
            txt += f"; China {ch.get(r.hts6, 0):.0%}"
        ax.text(max(r.n_eff_hhi, r[f"n_eff_hhi_{py}"]) + 0.2, i, txt, va="center", fontsize=8, color=st.INK2)
    ax.set_yticks(y, d.commodity)
    ax.set_xlim(0, d[["n_eff_hhi", f"n_eff_hhi_{py}"]].max().max() + 9)
    ax.set_xlabel("Effective number of buyers, 1 / HHI")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    soy = china[(china.hts6 == "120190")].set_index("year").loss_share
    ax.set_title(f"China's share of US soybean exports fell from {soy[py]:.0%} to {soy[fy]:.0%}, "
                 f"spreading buyers", pad=22)
    st.subtitle(ax, f"Buyer concentration of US exports, {fy} (bars) vs {py} (rings)")
    st.source_note(fig, SRC)
    st.save(fig, "fig06_export_buyers")


if __name__ == "__main__":
    main()
