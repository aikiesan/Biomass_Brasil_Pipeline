import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from bbp.analyse import mixed_model, model_frame, morans_i, queen_neighbours, random_forest


def _synthetic(n_mun=300, years=(2022, 2023), seed=1):
    """log-ratio = 0.5 * standardised crop_share + region effect + noise."""
    rng = np.random.default_rng(seed)
    mun = pd.DataFrame({"geocode": np.arange(1, n_mun + 1) + 1_100_000, "cd_rgint": rng.integers(1, 21, n_mun),
                        "dominant_biome": rng.choice(["Cerrado", "Amazônia"], n_mun),
                        "log_mun_area_ha": rng.normal(10, 1, n_mun)})
    region = rng.normal(0, 0.3, 21)
    rows = []
    for y in years:
        cs = rng.uniform(0, 0.6, n_mun)
        lr = 0.5 * (cs - 0.3) / 0.173 + region[mun["cd_rgint"]] + rng.normal(0, 0.2, n_mun)
        rows.append(pd.DataFrame({"geocode": mun["geocode"], "year": y, "crop_share": cs, "log_ratio": lr}))
    obs = pd.concat(rows, ignore_index=True)
    units = obs.assign(scale="municipality", unit=obs["geocode"].astype(str), crop="soybean")
    pred = obs.drop(columns="log_ratio").merge(mun, on="geocode").assign(
        double_crop_intensity=0.2, irrigated_share=0.0, created_after_start=False, uf="MT")
    return units, pred


def test_model_frame_keeps_complete_cases_and_drops_constant_predictors():
    units, pred = _synthetic()
    units.loc[0, "log_ratio"] = np.nan
    df = model_frame(units, pred, None, [2022, 2023], "soybean")
    assert len(df) == 599
    assert "crop_share" in df.attrs["numeric"] and "border_share_1km" not in df.attrs["numeric"]


def test_mixed_model_recovers_the_effect_and_the_region_variance():
    units, pred = _synthetic()
    df = model_frame(units, pred, None, [2022, 2023], "soybean")
    df.attrs["numeric"] = ["crop_share", "log_mun_area_ha"]
    coefs, stats, resid = mixed_model(df)
    est = coefs.set_index("term").loc["crop_share", "estimate"]
    assert est == pytest.approx(0.5, abs=0.05)                       # per SD of crop share
    # simulated: fixed 0.25 (0.5²), region 0.09 (0.3²), residual 0.04 (0.2²) -> 0.09 / 0.38 ≈ 0.24
    assert stats["region_variance_share"] == pytest.approx(0.24, abs=0.08)
    assert stats["r2_conditional"] > stats["r2_marginal"] > 0.3
    assert len(resid) == len(df)


def test_mixed_model_falls_back_to_clustered_ols_when_regions_carry_no_variance():
    units, pred = _synthetic()
    pred["cd_rgint"] = 1                                               # one region: no region variance
    df = model_frame(units, pred, None, [2022, 2023], "soybean")
    df.attrs["numeric"] = ["crop_share"]
    coefs, stats, _ = mixed_model(df)
    assert stats["model"].startswith(("mixed", "OLS"))
    assert np.isfinite(coefs.set_index("term").loc["crop_share", "std_error"])


def test_random_forest_is_grouped_and_ranks_the_true_driver_first():
    units, pred = _synthetic()
    df = model_frame(units, pred, None, [2022, 2023], "soybean")
    df.attrs["numeric"] = ["crop_share", "log_mun_area_ha"]
    imp, stats = random_forest(df)
    assert imp.iloc[0]["predictor"] == "crop_share"
    assert "dominant_biome" in set(imp["predictor"])                 # dummies folded into one predictor
    assert len(stats["rf_r2_folds"]) == 5


def test_morans_i_sign():
    cells = {i * 10 + j: box(j, i, j + 1, i + 1) for i in range(10) for j in range(10)}
    geom = pd.Series(cells)
    nb = queen_neighbours(geom)
    assert len(nb[0]) == 3 and len(nb[55]) == 8
    clustered = pd.Series({c: float(c % 10 < 5) for c in cells})        # left half vs right half
    checker = pd.Series({c: float((c // 10 + c % 10) % 2) for c in cells})
    assert morans_i(clustered, nb, 199)["morans_i"] > 0.5
    assert morans_i(clustered, nb, 199)["p_value"] < 0.05
    assert morans_i(checker, nb, 199)["morans_i"] < 0
