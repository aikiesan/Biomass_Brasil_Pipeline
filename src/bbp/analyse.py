"""H2: is the divergence between observed and declared area systematic and explainable?

Response: the municipal log-ratio ln(A_obs / A_dec) of each crop (reconcile stage), for the
reference years, over the municipalities where both sources report the crop.

Predictors (predictors.parquet, plus the crop-specific border share when available):
log municipal area, crop share, double-crop intensity, irrigated share, share of the crop
within 1 km of a municipal border, clear Landsat observations, dominant biome, year,
new-municipality flag.

Three complementary views, all with fixed seeds:

* **Mixed model** (statsmodels MixedLM): fixed effects for the predictors, biome and year;
  random intercept for the intermediate region (municipalities nested in regions).
  Marginal and conditional R² after Nakagawa & Schielzeth (2013).
* **Random forest** (scikit-learn): out-of-sample R² with folds grouped by intermediate
  region (no spatial leakage between train and test) and permutation importance.
* **Moran's I** of the mixed-model residuals (mean per municipality), queen contiguity,
  permutation p-value: is there spatial structure the predictors leave unexplained?
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("bbp")

NUMERIC = ["log_mun_area_ha", "crop_share", "log_double_crop", "irrigated_share", "border_share_1km",
           "log_landsat_valid_obs"]
SEED = 20261015
MIN_LEVEL = 30  # biome levels with fewer municipality-years are pooled as "Other"


def model_frame(units: pd.DataFrame, pred: pd.DataFrame, border: pd.DataFrame | None, years: list[int],
                crop: str) -> pd.DataFrame:
    """One row per municipality x year for `crop`: response + predictors, complete cases only."""
    u = units[(units["scale"] == "municipality") & (units["crop"] == crop) & units["year"].isin(years)].copy()
    u = u[np.isfinite(u["log_ratio"])]
    u["geocode"] = u["unit"].astype("int64")
    df = u[["geocode", "year", "log_ratio"]].merge(pred, on=["geocode", "year"], how="left")
    if border is not None and "border_share_1km" in border:
        b = border[border["crop"] == crop][["geocode", "year", "border_share_1km"]]
        df = df.drop(columns=["border_share_1km", "border_share_2km"], errors="ignore").merge(
            b, on=["geocode", "year"], how="left")
    if "double_crop_intensity" in df:
        df["log_double_crop"] = np.log1p(df["double_crop_intensity"].clip(lower=0, upper=10).fillna(0))
    if "landsat_valid_obs" in df:
        df["log_landsat_valid_obs"] = np.log1p(df["landsat_valid_obs"])
    numeric = [c for c in NUMERIC if c in df and df[c].notna().any()]
    df = df.dropna(subset=["log_ratio", "dominant_biome", "cd_rgint", *numeric]).copy()
    df.attrs["numeric"] = numeric
    return df


def _standardise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    z = df.copy()
    for c in cols:
        sd = z[c].std()
        z[c] = (z[c] - z[c].mean()) / sd if sd and np.isfinite(sd) else 0.0
    return z


def mixed_model(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.Series]:
    """Standardised fixed effects (one SD change -> change in log-ratio), fit stats, residuals."""
    import statsmodels.formula.api as smf

    numeric = df.attrs["numeric"]
    z = _standardise(df, numeric)
    z["dominant_biome"] = z["dominant_biome"].astype(str)
    counts = z["dominant_biome"].value_counts()
    z.loc[z["dominant_biome"].isin(counts[counts < MIN_LEVEL].index), "dominant_biome"] = "Other"  # rare levels pooled
    terms = [*numeric, "C(year)"] + (["C(dominant_biome)"] if z["dominant_biome"].nunique() > 1 else [])
    if z["created_after_start"].astype(bool).any():
        z["new_municipality"] = z["created_after_start"].astype(int)
        terms.append("new_municipality")
    formula = f"log_ratio ~ {' + '.join(terms)}"
    base = {"n": int(len(z)), "municipalities": int(z["geocode"].nunique()), "regions": int(z["cd_rgint"].nunique())}

    fit = None
    for method in ("lbfgs", "powell"):
        try:
            fit = smf.mixedlm(formula, z, groups=z["cd_rgint"]).fit(method=method, reml=True)
            if float(fit.cov_re.iloc[0, 0]) > 0 and np.all(np.isfinite(fit.bse_fe)):
                break
        except (np.linalg.LinAlgError, ValueError):
            fit = None
        fit = None if fit is None or not np.all(np.isfinite(fit.bse_fe)) else fit
    if fit is not None:
        params, se, pv, ci = fit.fe_params, fit.bse_fe, fit.pvalues[fit.fe_params.index], fit.conf_int()
        var_fixed = float(np.var(fit.predict(z)))            # fixed-effects prediction (no random part)
        var_re, var_res = float(fit.cov_re.iloc[0, 0]), float(fit.scale)
        total = var_fixed + var_re + var_res
        stats = {**base, "model": "mixed (random intercept: intermediate region)", "r2_marginal": var_fixed / total,
                 "r2_conditional": (var_fixed + var_re) / total, "region_variance_share": var_re / total,
                 "converged": bool(fit.converged)}
        resid = fit.resid
    else:
        # the region variance cannot be estimated (e.g. a crop present in few regions): pooled OLS with
        # standard errors clustered by region, so the inference still respects the nesting
        ols = smf.ols(formula, z).fit(cov_type="cluster", cov_kwds={"groups": z["cd_rgint"]})
        params, se, pv, ci = ols.params, ols.bse, ols.pvalues, ols.conf_int()
        stats = {**base, "model": "OLS, SE clustered by intermediate region", "r2_marginal": float(ols.rsquared),
                 "r2_conditional": np.nan, "region_variance_share": np.nan, "converged": True}
        resid = ols.resid
    coefs = pd.DataFrame({"term": params.index, "estimate": params.values, "std_error": se[params.index].values,
                          "p_value": pv[params.index].values, "ci_low": ci.loc[params.index, 0].values,
                          "ci_high": ci.loc[params.index, 1].values})
    return coefs, stats, pd.Series(resid, index=z.index)


def random_forest(df: pd.DataFrame, n_splits: int = 5) -> tuple[pd.DataFrame, dict]:
    """Grouped (by intermediate region) cross-validated R² and permutation importance."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import GroupKFold

    numeric = df.attrs["numeric"]
    x = pd.get_dummies(df[[*numeric, "dominant_biome", "year"]].astype({"dominant_biome": str}),
                       columns=["dominant_biome"], dtype=float)
    y, groups = df["log_ratio"].to_numpy(), df["cd_rgint"].to_numpy()
    r2, imps = [], []
    for train, test in GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(x, y, groups):
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
        rf.fit(x.iloc[train], y[train])
        r2.append(rf.score(x.iloc[test], y[test]))
        pi = permutation_importance(rf, x.iloc[test], y[test], n_repeats=5, random_state=SEED, n_jobs=-1)
        imps.append(pi.importances_mean)
    imp = pd.DataFrame(imps, columns=x.columns)
    # the biome dummies are one predictor
    biome_cols = [c for c in imp if c.startswith("dominant_biome_")]
    imp["dominant_biome"] = imp[biome_cols].sum(axis=1)
    imp = imp.drop(columns=biome_cols)
    out = pd.DataFrame({"predictor": imp.columns, "importance_mean": imp.mean().values, "importance_sd": imp.std().values})
    return out.sort_values("importance_mean", ascending=False), {"rf_r2_grouped_cv": float(np.mean(r2)),
                                                                 "rf_r2_folds": [float(v) for v in r2]}


def queen_neighbours(geom: pd.Series) -> dict[int, list[int]]:
    """{geocode: [neighbouring geocodes]} — polygons that share at least a point (queen contiguity)."""
    import shapely

    codes = geom.index.to_numpy()
    tree = shapely.STRtree(geom.values)
    left, right = tree.query(geom.values, predicate="intersects")
    nb: dict[int, list[int]] = {int(c): [] for c in codes}
    for i, j in zip(left, right, strict=True):
        if i != j:
            nb[int(codes[i])].append(int(codes[j]))
    return nb


def morans_i(values: pd.Series, neighbours: dict[int, list[int]], permutations: int = 999) -> dict:
    """Moran's I with row-standardised weights over the units in `values`; permutation p-value."""
    v = values.dropna()
    idx = {c: k for k, c in enumerate(v.index)}
    rows, cols = [], []
    for c in v.index:
        nbs = [idx[n] for n in neighbours.get(int(c), []) if n in idx]
        rows += [idx[c]] * len(nbs)
        cols += nbs
    if not rows:
        return {"morans_i": np.nan, "p_value": np.nan, "n": int(len(v))}
    rows, cols = np.asarray(rows), np.asarray(cols)
    deg = np.bincount(rows, minlength=len(v)).astype(float)
    w = 1.0 / deg[rows]
    x = v.to_numpy() - v.mean()
    has = deg > 0

    def stat(arr: np.ndarray) -> float:
        lag = np.bincount(rows, weights=w * arr[cols], minlength=len(arr))
        return float((arr[has] * lag[has]).sum() / (arr[has] ** 2).sum())

    i_obs = stat(x)
    rng = np.random.default_rng(SEED)
    sims = np.array([stat(rng.permutation(x)) for _ in range(permutations)])
    p = (np.sum(np.abs(sims - sims.mean()) >= abs(i_obs - sims.mean())) + 1) / (permutations + 1)
    return {"morans_i": i_obs, "expected": -1 / (len(v) - 1), "p_value": float(p), "n": int(has.sum())}
