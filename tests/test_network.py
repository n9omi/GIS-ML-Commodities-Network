import numpy as np
import pandas as pd
import pytest

from gmcn.network import metrics as nm
from gmcn.network import robustness as rb

LABELS = {"A": {"label": "Alpha", "group": "g"}, "B": {"label": "Beta", "group": "g"}}


def toy():
    # commodity A: one dominant supplier; commodity B: two equal suppliers
    return pd.DataFrame({"A": [90.0, 10.0, 0.0], "B": [0.0, 50.0, 50.0]}, index=["X", "Y", "Z"])


def test_hhi_and_effective_suppliers():
    assert nm.hhi([1.0]) == 1.0
    assert nm.hhi([0.5, 0.5]) == 0.5
    t = nm.concentration_table(toy(), LABELS).set_index("hts6")
    assert t.loc["A", "hhi"] == pytest.approx(0.82)
    assert t.loc["B", "n_eff_hhi"] == pytest.approx(2.0)
    assert t.loc["B", "n_eff_shannon"] == pytest.approx(2.0)
    assert t.loc["A", "top1_country"] == "X"


def test_nodf_extremes():
    nested = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
    assert nm.nodf(nested) == pytest.approx(100.0)
    checker = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    assert nm.nodf(checker) == pytest.approx(0.0)


def test_curveball_preserves_margins():
    rng = np.random.default_rng(1)
    B = (rng.random((15, 8)) < 0.35).astype(int)
    R = nm.curveball(B, rng, n_swaps=500)
    assert (R.sum(axis=1) == B.sum(axis=1)).all()
    assert (R.sum(axis=0) == B.sum(axis=0)).all()


def test_country_table_and_projection():
    ct = nm.country_table(toy()).set_index("country")
    assert ct.loc["Y", "n_commodities"] == 2
    # Y supplies both commodities, so it bridges them in the bipartite graph
    assert ct.loc["Y", "betweenness"] == ct.betweenness.max()
    C = nm.commodity_projection(toy())
    assert C.loc["A", "A"] == pytest.approx(1.0)
    assert 0 < C.loc["A", "B"] < 1


def test_removal_curve_and_scenarios():
    curve = rb.removal_curve(toy(), ["X", "Y", "Z"])
    # removing X takes 90 % of A and none of B -> equal-weighted retained = (0.1 + 1.0) / 2
    assert curve.retained_equal.iloc[1] == pytest.approx(0.55)
    assert curve.retained_equal.iloc[-1] == pytest.approx(0.0)
    sc = rb.scenario_loss(toy(), ["X"], LABELS).set_index("hts6")
    assert sc.loc["A", "loss_share"] == pytest.approx(0.9)
    assert sc.loc["B", "loss_share"] == pytest.approx(0.0)


def test_monte_carlo_concentration_moves_the_tail_not_the_mean():
    M = pd.DataFrame({"conc": [100.0] + [0.0] * 9, "div": [10.0] * 10}, index=[f"c{i}" for i in range(10)])
    L = rb.simulate_losses(M, p=0.05, rho=0.0, n_sims=40000, seed=0)
    sar = rb.supply_at_risk(L, list(M.columns)).set_index("hts6")
    assert sar.loc["conc", "expected_loss"] == pytest.approx(0.05, abs=0.005)
    assert sar.loc["div", "expected_loss"] == pytest.approx(0.05, abs=0.005)
    assert sar.loc["conc", "es_95"] > 3 * sar.loc["div", "es_95"]


def test_correlation_fattens_the_tail_of_a_diversified_commodity():
    M = pd.DataFrame({"div": [10.0] * 10}, index=[f"c{i}" for i in range(10)])
    lo = rb.supply_at_risk(rb.simulate_losses(M, 0.05, 0.0, 40000, 1), ["div"])
    hi = rb.supply_at_risk(rb.simulate_losses(M, 0.05, 0.6, 40000, 1), ["div"])
    assert hi["es_95"].iloc[0] > 1.5 * lo["es_95"].iloc[0]
