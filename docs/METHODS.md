# Methods

Each layer states its estimand, the method, and how it is validated. Code references point to `src/gmcn/`.

## 1. Satellite feature extraction (`ingest/sentinel2.py`, `ingest/nightfire.py`, `ingest/earthengine.py`, `features/`)

**Access.** Sentinel-2 L2A scenes are found through a STAC API (Microsoft Planetary Computer by default, Element 84
Earth Search as a fallback) and read as cloud-optimized GeoTIFFs. Only the window covering each area of interest (AOI)
is read; bands at 20 m (B8A, B11, B12, SCL) are resampled onto the 10 m reference grid, and rasters in other projections
(the 30 m Albers Cropland Data Layer) are warped lazily onto that grid.

**Radiometric harmonization.** From processing baseline 04.00 (January 2022) ESA adds 1000 to every L2A digital number.
Reflectance is therefore `(DN + BOA_ADD_OFFSET) / 10000` with `BOA_ADD_OFFSET = -1000` for baseline >= 04.00 and 0 before.
Skipping this step biases post-2022 NDVI downward and creates a spurious break in every time series; a unit test
(`test_without_harmonization_ndvi_would_be_biased`) checks the size of that bias.

**Cloud masking.** The Scene Classification Layer removes no-data (0), defective/saturated (1), cloud shadow (3), medium
and high cloud probability (8, 9), cirrus (10) and snow (11) for vegetation. For hotspots, class 1 is *kept*: the hottest
smelter pixels often saturate the SWIR bands. An AOI-scene result is kept only if at least 20 % of its pixels are clear;
the clear fraction is stored, so missingness is modelled rather than interpolated.

**Indices.**

```
NDVI      = (B08 - B04) / (B08 + B04)
EVI       = 2.5 (B08 - B04) / (B08 + 6 B04 - 7.5 B02 + 1)
NHI_SWIR  = (B12 - B11) / (B12 + B11)
NHI_SWNIR = (B11 - B8A) / (B11 + B8A)
hot pixel = (NHI_SWIR > 0 or NHI_SWNIR > 0) and B12 >= 0.15
```

The normalized hotspot indices follow Marchese et al. (2019), who define them on top-of-atmosphere radiance; here they are
applied to L2A surface reflectance, with a B12 floor that screens dark pixels where the ratios are unstable.

**Flaring.** VIIRS Nightfire nightly detections inside the Permian polygon with a Planck-fit temperature at or above
1,400 K are counted as gas flares (biomass fires burn cooler); radiant heat is summed per night and averaged per month.

**Emissions and land change (optional, Earth Engine).** Sentinel-5P L3 SO2 and tropospheric NO2 are averaged monthly
inside a 5 km facility buffer minus a 20-40 km background ring (the enhancement). AlphaEarth annual 64-dimensional
embeddings are averaged over an AOI, re-normalized, and compared year over year by cosine distance.

**Composites, anomalies, breaks.** Scenes become weekly medians per AOI (`weekly_composite`). Crop anomalies are
standardized against the same week of year in *other* years (leave-one-year-out), so a drought year cannot dampen its
own anomaly. Monthly facility activity has two parts: hot pixels per clear look (intensity) and the share of clear looks
with any hot pixel (operating status). Mean shifts are dated with PELT (Killick et al., 2012) under a penalty of
`3 sigma^2 log n`, with sigma from the MAD of first differences. In 200 simulations (n = 70, one 2-sigma shift) this
penalty detected 95 % of shifts with a 0.5 % false-break rate; the plain BIC multiplier of 2 false-alarmed 15.5 % of the time.

## 2. Nowcasting (`models/nowcast.py`)

**Timing.** A nowcast for month t is made at the end of month t. Official production for month s is known only if
`s <= t - 2` (the EIA publication lag). Prices and satellite indices for month t are known at the end of month t.
Training for month t uses only targets dated `<= t - 2`, so no future information enters the fit.

**Target and features.** Target: monthly log growth of Texas + New Mexico crude output (a Permian proxy).
Features: own growth at lags 2-4, WTI log returns at lags 0-12, seasonal harmonics, and, when present, same-month flaring.

**Models.** Last-print naive; AR on the available own lags (the benchmark); ridge with cross-validated penalty; LightGBM
with quantile loss at 0.1, 0.5 and 0.9. A one-factor dynamic factor model (Bańbura and Modugno, 2014, as implemented in
`statsmodels.DynamicFactorMQ`) extracts a common factor from noisy, partly missing panels.

**Uncertainty.** Split-conformal intervals around the ridge nowcast use the absolute errors of the last 36 nowcasts whose
outcomes were already published; the band is the `ceil((n+1)(1-alpha))/n` empirical quantile, giving finite-sample
marginal coverage under exchangeability (Lei et al., 2018).

**Evaluation.** Expanding-window rolling origin from 2015. RMSE, MAE, pinball loss, interval coverage, and
Diebold-Mariano tests (Diebold and Mariano, 1995) with the Harvey-Leybourne-Newbold (1997) small-sample correction.

**Supply surprise.** `s_t = (y_t - yhat_t) / sigma_t`, with `sigma_t` the 36-month rolling standard deviation of nowcast
errors already known at t. The ridge model is the pre-specified expectation model (chosen before seeing test results).

## 3. Network theory (`network/metrics.py`, `network/robustness.py`)

**Graph.** For one year and one flow, M is the countries x commodities matrix of US trade values (USITC DataWeb, HTS 6-digit;
general imports at customs value, total exports at FAS value). Shares `S = M / column sums`; every column sums to one.
The bipartite graph links country c to commodity k with weight `s_ck` and distance `1 / s_ck`.

| Metric | Definition | Reading |
| --- | --- | --- |
| HHI | `HHI_k = sum_c s_ck^2` | 1 = single supplier |
| Effective suppliers | `N_eff = 1 / HHI_k`; also `exp(H_k)`, the order-1 Hill number | supply behaves like N_eff equal suppliers |
| Strength | `sum_k M_ck` | total dependence on country c |
| Country projection | `W = S S^T`, diagonal removed | countries playing the same roles |
| Eigenvector centrality, PageRank | on W | systemic suppliers |
| Betweenness | shortest paths on the bipartite graph with distance `1 / s_ck` | countries bridging separate supply chains |
| Co-dependency | cosine similarity of commodity supplier-share vectors | commodities that one export ban would hit together |
| Communities | greedy modularity on co-dependency edges >= 0.3 | clusters of jointly exposed commodities |
| Nestedness | NODF (Almeida-Neto et al., 2008) on the binary matrix `s_ck >= 1 %`, tested against a fixed-degree null drawn with the curveball algorithm (Strona et al., 2014) | hidden common exposure beyond the degree sequence |
| k-core | core number on the bipartite graph | the irreducible core |
| Robustness | share of supply retained as suppliers are removed (targeted by dominance, targeted by value, random); `R` = mean retained over all steps (Schneider et al., 2011) | how fast supply collapses under export bans |

**Scenarios.** Loss for commodity k when the set C of countries stops exporting: `sum_{c in C} s_ck`.

**Monte Carlo supply-at-risk.** Supplier c fails when `Z_c = sqrt(rho) F + sqrt(1 - rho) e_c < Phi^-1(p)`, with F a
common (geopolitical) factor. Loss: `L_k = sum_c s_ck 1[c fails]`. Because the shares sum to one,
`E[L_k] = p` for every commodity whatever its concentration: concentration changes only the distribution of losses, not
their mean. The pipeline reports expected loss, VaR95, expected shortfall ES95 and `P(L_k > 25 %)` for `rho` in
{0, 0.3, 0.6}. The failure probability `p = 5 %` is an illustrative prior, identical across suppliers; supplying
country-specific `p_c` (for example from the Caldara-Iacoviello country GPR indices) is a one-line change.

## 4. Regimes and price response (`models/regimes.py`, `models/local_projections.py`, `models/event_study.py`)

**Hidden Markov model.** Gaussian emissions with full covariances on five standardized monthly series: 10-year TIPS yield,
12-month change in fed funds, CPI inflation (y/y), 12-month change in unemployment, broad dollar (y/y). Twenty random
restarts per k. Selection: the lowest BIC among models whose every decoded state covers at least 20 months; without the
floor, BIC keeps adding states of 16-17 months that cannot support conditional estimates. Regimes are labelled by rule from
their mean profiles. The state used downstream is the **filtered** state (forward recursion only), so no future macro data
enters it; parameters are estimated on the full sample, which the results flag.

**Local projections** (Jordà, 2005), linear and state-dependent (Ramey and Zubairy, 2018):

```
p_{t+h} - p_{t-1} = sum_S 1[S_{t-1} = S] (a_{S,h} + b_{S,h} s_t + g_{S,h}' x_t) + e_{t+h}
```

with p = 100 x log WTI, s the standardized supply surprise, x two lags of the price change and of the shock, Newey-West
standard errors with h + 1 lags, and a Wald test of `b_A,h = b_B,h` at each horizon.

**Event study.** Weekly log returns; market model against SPY estimated on 104 weeks ending 4 weeks before the event;
abnormal return in the event week and cumulative abnormal returns over weeks 0-2, with t-statistics from the
estimation-window residual variance. Events and their sources live in `config/events.yaml`; confounders are recorded there.

## References

- Almeida-Neto, M. et al. (2008). A consistent metric for nestedness analysis in ecological systems. *Oikos* 117, 1227-1239.
- Bańbura, M. and Modugno, M. (2014). Maximum likelihood estimation of factor models on datasets with arbitrary pattern of missing data. *Journal of Applied Econometrics* 29, 133-160.
- Caldara, D. and Iacoviello, M. (2022). Measuring geopolitical risk. *American Economic Review* 112(4), 1194-1225.
- Diebold, F. and Mariano, R. (1995). Comparing predictive accuracy. *Journal of Business and Economic Statistics* 13, 253-263.
- Elvidge, C. et al. (2013). VIIRS nightfire: satellite pyrometry at night. *Remote Sensing* 5, 4423-4449.
- Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting* 13, 281-291.
- Jordà, Ò. (2005). Estimation and inference of impulse responses by local projections. *American Economic Review* 95(1), 161-182.
- Killick, R., Fearnhead, P. and Eckley, I. (2012). Optimal detection of changepoints with a linear computational cost. *JASA* 107, 1590-1598.
- Lei, J. et al. (2018). Distribution-free predictive inference for regression. *JASA* 113, 1094-1111.
- Marchese, F. et al. (2019). A multi-channel algorithm for mapping volcanic thermal anomalies by means of Sentinel-2 MSI and Landsat-8 OLI data. *Remote Sensing* 11(23), 2876.
- Maus, V. et al. (2020). A global-scale data set of mining areas. *Scientific Data* 7, 289.
- Ramey, V. and Zubairy, S. (2018). Government spending multipliers in good times and in bad. *Journal of Political Economy* 126(2), 850-901.
- Schneider, C. et al. (2011). Mitigation of malicious attacks on networks. *PNAS* 108(10), 3838-3841.
- Strona, G. et al. (2014). A fast and unbiased procedure to randomize ecological binary matrices with fixed row and column totals. *Nature Communications* 5, 4114.
