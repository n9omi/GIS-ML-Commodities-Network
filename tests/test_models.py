import numpy as np
import pandas as pd
import pytest

from gmcn.models import event_study as es
from gmcn.models import local_projections as lp
from gmcn.models import nowcast as nc
from gmcn.models import regimes as rg


def test_state_dependent_lp_recovers_true_betas():
    rng = np.random.default_rng(0)
    T = 1200
    idx = pd.date_range("1920-01-01", periods=T, freq="MS")
    state = pd.Series((np.arange(T) // 100) % 2, index=idx)       # alternating regimes
    s = pd.Series(rng.standard_normal(T), index=idx)
    beta = np.where(state.values == 0, -1.0, -3.0)
    dy = beta * s.values + rng.standard_normal(T) * 0.5
    y = pd.Series(np.cumsum(dy), index=idx)
    out = lp.lp_state(y, s, state.shift(1), horizons=2, n_lags=2)
    h0 = out[(out.h == 0) & (out.state != "equality_test")].set_index("state")
    assert h0.loc[0.0, "beta"] == pytest.approx(-1.0, abs=0.1)
    assert h0.loc[1.0, "beta"] == pytest.approx(-3.0, abs=0.1)
    eq = out[(out.h == 0) & (out.state == "equality_test")]
    assert eq["hi"].iloc[0] < 0.01      # p-value of equal responses


def test_linear_lp_on_permanent_shock():
    rng = np.random.default_rng(1)
    T = 800
    idx = pd.date_range("1950-01-01", periods=T, freq="MS")
    s = pd.Series(rng.standard_normal(T), index=idx)
    y = pd.Series(np.cumsum(-2.0 * s.values + 0.3 * rng.standard_normal(T)), index=idx)
    out = lp.lp_linear(y, s, horizons=3)
    assert out.beta.iloc[3] == pytest.approx(-2.0, abs=0.15)   # permanent effect persists


def test_hmm_selects_two_states_and_filters():
    rng = np.random.default_rng(2)
    z = np.r_[np.zeros(150), np.ones(150), np.zeros(150)].astype(int)
    means = np.array([[0, 0], [3, -2]])
    X = means[z] + rng.standard_normal((450, 2)) * 0.5
    model, table = rg.fit_hmm(X, [2, 3], restarts=3, seed=0)
    assert table.sort_values("bic").k.iloc[0] == 2
    P = rg.filtered_probabilities(model, X)
    assert np.allclose(P.sum(axis=1), 1.0)
    order = rg.relabel_by(model, 0)
    hard = np.argmax(P[:, order], axis=1)
    assert (hard == z).mean() > 0.95


def test_dynamic_factor_tracks_true_factor():
    rng = np.random.default_rng(3)
    T = 240
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = 0.8 * f[t - 1] + rng.standard_normal()
    lam = np.array([1.0, 0.8, -0.6, 0.5])
    X = np.outer(f, lam) + rng.standard_normal((T, 4)) * 0.6
    panel = pd.DataFrame(X, columns=list("abcd"), index=pd.date_range("2000-01-01", periods=T, freq="MS"))
    fac = rg.common_factor(panel)
    assert abs(np.corrcoef(fac.values, f)[0, 1]) > 0.9


def test_nowcast_features_respect_publication_lag():
    idx = pd.date_range("2000-01-01", periods=60, freq="MS")
    prod = pd.Series(np.exp(np.linspace(0, 1, 60)), index=idx)
    price = pd.Series(np.exp(np.sin(np.arange(60) / 5)), index=idx)
    X, y = nc.make_features(prod, price, pub_lag=2, price_lags=3)
    own = [c for c in X.columns if c.startswith("y_l")]
    assert own == ["y_l2", "y_l3", "y_l4"]
    assert "ret_l0" in X.columns


def test_rolling_origin_and_metrics_run():
    rng = np.random.default_rng(4)
    idx = pd.date_range("1995-01-01", periods=200, freq="MS")
    price = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.05, 200))), index=idx)
    r = np.log(price).diff().fillna(0)
    g = 0.5 * r.shift(6).fillna(0) + rng.normal(0, 0.01, 200)
    prod = pd.Series(np.exp(np.cumsum(g)), index=idx)
    X, y = nc.make_features(prod, price, pub_lag=2, price_lags=8)
    pred = nc.rolling_origin(X, y, "2008-01", pub_lag=2, quantiles=[0.1, 0.5, 0.9], conformal_alpha=0.2)
    ev = nc.evaluate(pred, [0.1, 0.5, 0.9]).set_index("model")
    assert ev.loc["ridge", "rmse"] < ev.loc["ar", "rmse"]      # prices carry real signal here
    assert 0.6 < ev.attrs["conformal_coverage"] <= 1.0
    s = nc.standardized_surprise(pred)
    assert abs(s.mean()) < 0.5


def test_diebold_mariano_sign():
    rng = np.random.default_rng(5)
    e_small, e_big = rng.normal(0, 1, 300), rng.normal(0, 2, 300)
    dm = nc.diebold_mariano(e_small, e_big)
    assert dm["stat"] < 0 and dm["p_value"] < 0.01


def test_event_study_detects_injected_shock():
    rng = np.random.default_rng(6)
    idx = pd.date_range("2022-01-07", periods=200, freq="W-FRI")
    m = pd.Series(rng.normal(0.001, 0.02, 200), index=idx)
    r = 0.002 + 1.2 * m + rng.normal(0, 0.02, 200)
    ev_date = idx[150] - pd.Timedelta(days=2)
    r.iloc[150] -= 0.20
    res = es.market_model_car(r, m, ev_date.date().isoformat(), est_weeks=104, gap_weeks=4, car_weeks=1)
    assert res["event_week"] == idx[150].date().isoformat()
    assert res["car"] == pytest.approx(-0.20, abs=0.06)
    assert res["t_stat"] < -4
