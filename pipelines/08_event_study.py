"""08 - Market reaction to the 2025 copper mine disruptions (weekly market-model event study).

The Grasberg case is the motivating test for the satellite layer: the physical incident (Sept 8)
and the force-majeure disclosure (Sept 24) are two weeks apart. If the market priced the loss only
on disclosure, an independent physical signal had an information window.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gmcn.config import DATA_PROC, OUTPUTS, ensure_dirs, events, settings
from gmcn.models import event_study as es
from gmcn.viz import style as st


def main() -> None:
    ensure_dirs()
    st.apply()
    cfg = settings()["event_study"]
    px = pd.read_csv(DATA_PROC / "prices_weekly.csv", index_col=0, parse_dates=True)
    r = np.log(px).diff()
    rows, paths = [], {}
    for ev in events():
        for tk in ev["tickers"]:
            res = es.market_model_car(r[tk], r["SPY"], ev["date"], cfg["estimation_weeks"], cfg["gap_weeks"],
                                      cfg["car_weeks"])
            one = es.market_model_car(r[tk], r["SPY"], ev["date"], cfg["estimation_weeks"], cfg["gap_weeks"], 1)
            rows.append({"event": ev["id"], "name": ev["name"], "date": ev["date"], "ticker": tk,
                         "event_week": res["event_week"], "beta": res["beta"], "ar_week0": one["car"],
                         "t_week0": one["t_stat"], f"car_0_{cfg['car_weeks'] - 1}": res["car"],
                         "t_car": res["t_stat"], "p_car": res["p_value"], "source": ev["source"],
                         "confounded_by": ev.get("confounded_by", "")})
            # cumulative abnormal return path from 4 weeks before to 8 weeks after
            df = pd.concat({"r": r[tk], "m": r["SPY"]}, axis=1).dropna()
            e = es.event_week(df.index, ev["date"])
            i = df.index.get_loc(e)
            win = df.iloc[i - 4: i + 9]
            ar = win["r"] - (res["alpha"] + res["beta"] * win["m"])
            paths[(ev["id"], tk)] = pd.Series(ar.cumsum().values, index=range(-4, -4 + len(ar)))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTPUTS / "event_study_copper_2025.csv", index=False)
    tab = tab.drop(columns="confounded_by")
    pd.DataFrame({f"{k[0]}|{k[1]}": v for k, v in paths.items()}).to_csv(OUTPUTS / "event_study_car_paths.csv")
    (OUTPUTS / "event_study_summary.json").write_text(json.dumps(tab.drop(columns="source").round(4).to_dict("records"),
                                                                  indent=2))
    figure(paths, tab)
    print(tab.drop(columns=["source", "name"]).round(3).to_string(index=False))


def figure(paths: dict, tab: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    ax = axes[0]
    inc = paths[("grasberg_incident_2025", "FCX")]
    ax.plot(inc.index, 100 * inc.values, color=st.SERIES[0], marker="o", ms=3.5, label="Freeport-McMoRan (FCX)")
    cp = paths[("grasberg_incident_2025", "CPER")]
    ax.plot(cp.index, 100 * cp.values, color=st.SERIES[1], marker="o", ms=3.5, label="Copper fund (CPER)")
    ax.axvline(0, color=st.INK2, lw=1)
    ax.axvline(2, color=st.INK, lw=1)
    ax.set_ylim(-20, 13)
    ax.text(-0.15, 11.3, "mud rush, Sept 8", fontsize=8, color=st.INK2, ha="right", va="center")
    ax.text(2.15, 11.3, "force majeure, Sept 24", fontsize=8, color=st.INK, ha="left", va="center")
    ax.axhline(0, color=st.AXIS, lw=0.8)
    ax.set_xlabel("Weeks relative to the incident week")
    ax.set_ylabel("Cumulative abnormal return, %")
    ax.set_title("Grasberg, 2025", fontsize=10, pad=4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=2)
    ax2 = axes[1]
    for eid, lab, col in [("kakula_2025", "Kamoa-Kakula, May 18", st.SERIES[2]),
                          ("el_teniente_2025", "El Teniente, Jul 31 (same week: US tariff exempts refined copper)", st.SERIES[3]),
                          ("grasberg_incident_2025", "Grasberg, Sept 8", st.SERIES[1])]:
        p = paths[(eid, "CPER")]
        ax2.plot(p.index, 100 * p.values, color=col, marker="o", ms=3.5, label=lab)
    ax2.axvline(0, color=st.INK2, lw=1)
    ax2.axhline(0, color=st.AXIS, lw=0.8)
    ax2.set_xlabel("Weeks relative to the incident week")
    ax2.set_title("Copper fund (CPER) around each 2025 disruption", fontsize=10, pad=4)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=1)
    g = tab[(tab.event == "grasberg_incident_2025") & (tab.ticker == "FCX")].iloc[0]
    f = tab[(tab.event == "grasberg_force_majeure_2025") & (tab.ticker == "FCX")].iloc[0]
    fig.suptitle(f"Freeport's abnormal return was {100 * g.ar_week0:+.0f} % in the incident week and "
                 f"{100 * f.ar_week0:+.0f} % in the disclosure week", x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.06)
    fig.text(0.01, 0.955, "Weekly market model vs SPY (104-week estimation window ending 4 weeks before each event). The two-week "
             "gap is the information window\na free physical signal could have exploited; whether Sentinel-2 saw it is the "
             "satellite layer's first test.", fontsize=9, color=st.INK2)
    st.source_note(fig, "Source: weekly closes via Alpha Vantage (not dividend-adjusted); event dates from public reports "
                        "listed in config/events.yaml.")
    st.save(fig, "fig11_grasberg_event_study")


if __name__ == "__main__":
    main()
