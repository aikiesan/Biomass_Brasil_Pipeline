"""bbp.demo: compositions, CLR, clustering, the mandate targets and the stock within reach, on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from bbp import demo


def test_zero_replacement_closes_and_keeps_ratios():
    s = pd.DataFrame({"a": [0.6, 0.5], "b": [0.4, 0.4995], "c": [0.0, 0.0005]})
    r, d = demo.replace_zeros(s, detection=1e-3)
    assert d == pytest.approx(6.5e-4)
    assert np.allclose(r.sum(axis=1), 1.0)
    assert r.loc[0, "c"] == d and r.loc[1, "c"] == d                    # 0 and 0.05 % are both below 0.1 %
    assert r.loc[0, "a"] / r.loc[0, "b"] == pytest.approx(1.5)           # ratios of the other parts kept
    z = demo.clr(r)
    assert np.allclose(z.sum(axis=1), 0.0)


def test_region_composition_counts_the_second_season_apart():
    cov = pd.DataFrame({"geocode": [1, 1, 2, 2], "year": 2024, "class_id": [39, 20, 39, 41],
                        "area_ha": [100.0, 50.0, 80.0, 20.0]})
    second = pd.DataFrame({"geocode": [1, 2], "year": 2024, "class": ["corn_2nd", "cotton_2nd"],
                           "area_ha": [60.0, 10.0]})
    mun = pd.DataFrame({"geocode": [1, 2], "cd_rgint": [10, 10]})
    c = demo.region_composition(cov, second, mun, 2024)
    assert list(c.columns) == demo.ORDER
    row = c.loc[10]
    assert row["soybean"] == 180 and row["corn_2nd"] == 60 and row["sugarcane"] == 50
    assert row["cotton"] == 10 and row["other"] == 20


def test_cluster_finds_separated_groups_and_picks_a_stable_k():
    rng = np.random.default_rng(1)
    centres = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=float)
    pts = np.vstack([c + 0.05 * rng.standard_normal((15, 3)) for c in centres])
    pts -= pts.mean(axis=1, keepdims=True)                              # CLR-like rows
    z = pd.DataFrame(pts, columns=["a", "b", "c"])
    labels, diag = demo.cluster(z, (2, 6), seed=0, select="stability", min_cluster=10)
    assert int(diag.loc[diag["chosen"], "k"].iloc[0]) == 3
    assert labels.nunique() == 3 and set(labels.value_counts()) == {15}
    assert diag.loc[diag["k"] == 3, "ari_kmeans"].iloc[0] == pytest.approx(1.0)


def test_mandate_targets_keep_official_volumes_and_derive_the_rest():
    t = demo.mandate_targets([
        {"year": 2026, "ghg_reduction_pct": 0.5, "volume_m3d": 504800},
        {"year": 2027, "ghg_reduction_pct": 1.0, "volume_m3d": 995200, "base_reference": True},
        {"year": 2028, "ghg_reduction_pct": 3.0}]).set_index("year")
    assert t.loc[2026, "volume_m3d"] == 504800 and t.loc[2026, "volume_source"] == "official"
    assert t.loc[2026, "implied_base_m3d"] == pytest.approx(100.96e6)
    assert t.loc[2028, "volume_source"] == "derived"
    assert t.loc[2028, "volume_m3d"] == pytest.approx(3 * 995200)


def test_stock_within_reach_is_n1_times_the_share_within():
    hist = pd.DataFrame({"pair": "all|bm", "geocode": [1, 1, 2], "bin_km": [5.0, 60.0, 200.0],
                         "area_ha": [30.0, 70.0, 100.0]})
    n1 = pd.DataFrame({"geocode": [1, 2], "n1_ref_nm3": [365e6, 365e6], "n1_min_nm3": [182.5e6, 182.5e6],
                       "n1_max_nm3": [730e6, 730e6]})
    s = demo.stock_within_reach(hist, n1, "all|bm", [30, 100]).set_index("radius_km")
    assert s.loc[30.0, "n1_ref_nm3_yr"] == pytest.approx(0.3 * 365e6)
    assert s.loc[100.0, "n1_ref_nm3_yr"] == pytest.approx(365e6)          # municipality 2 is beyond 100 km
    assert s.loc[30.0, "n1_ref_m3d"] == pytest.approx(0.3 * 1e6)
    assert s.loc[30.0, "n1_min_m3d"] <= s.loc[30.0, "n1_ref_m3d"] <= s.loc[30.0, "n1_max_m3d"]
    assert s.loc[30.0, "n1_total_ref_nm3_yr"] == pytest.approx(730e6)
