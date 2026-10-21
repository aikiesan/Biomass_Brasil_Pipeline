"""bbp.reconcile: trend agreement between observed and declared area, on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from bbp import reconcile as rc

W = [[2008, 2009], [2020, 2021]]


def _units(rows):
    """rows: (unit, crop, obs_early, dec_early, obs_late, dec_late), constant within each window."""
    out = []
    for u, c, o0, d0, o1, d1 in rows:
        for y, o, d in ((2008, o0, d0), (2009, o0, d0), (2020, o1, d1), (2021, o1, d1)):
            out.append({"scale": "municipality", "unit": u, "year": y, "crop": c, "a_obs_ha": o, "a_dec_ha": d})
    return pd.DataFrame(out)


def test_direction_classes_and_log_change():
    u = _units([("a", "soy", 100, 100, 200, 300),     # both up
                ("b", "soy", 400, 400, 200, 100),     # both down
                ("c", "soy", 100, 400, 300, 200),     # map up, statistics down
                ("d", "soy", 400, 100, 200, 300)])    # map down, statistics up
    t = rc.trend_units(u, W, 50).set_index("unit")
    assert list(t["direction"]) == ["both_up", "both_down", "obs_up_dec_down", "obs_down_dec_up"]
    assert t.loc["a", "dlog_obs"] == pytest.approx(np.log(2)) and t.loc["a", "dlog_dec"] == pytest.approx(np.log(3))
    s = rc.trend_agreement(t.reset_index()).iloc[0]
    assert s["n_units"] == 4 and s["share_same_sign"] == 0.5
    assert s["share_obs_up_dec_down"] == 0.25 and s["share_obs_down_dec_up"] == 0.25
    # the total change is that of the summed area: map 1000 -> 900, statistics 1000 -> 900
    assert s["dlog_obs_total"] == pytest.approx(np.log(900 / 1000))
    assert s["dlog_dec_total"] == pytest.approx(np.log(900 / 1000))


def test_small_or_incomplete_units_are_not_scored():
    u = _units([("a", "soy", 100, 100, 200, 300),
                ("small", "soy", 100, 100, 20, 300),      # one window mean below the floor
                ("gap", "soy", 100, 100, 200, 300)])
    u = u[~((u["unit"] == "gap") & (u["year"] == 2021))]  # a year missing (e.g. mesh mismatch)
    t = rc.trend_units(u, W, 50)
    assert list(t["unit"]) == ["a"]


def test_identical_sources_agree_perfectly():
    rng = np.random.default_rng(3)
    rows = [(str(i), "cane", a0, a0, a1, a1) for i, (a0, a1) in enumerate(rng.uniform(100, 1000, (30, 2)))]
    s = rc.trend_agreement(rc.trend_units(_units(rows), W, 50)).iloc[0]
    assert s["share_same_sign"] == 1.0 and s["spearman_rho"] == pytest.approx(1.0) and s["lin_ccc"] == pytest.approx(1.0)
    assert s["median_dlog_obs"] == pytest.approx(s["median_dlog_dec"])


def test_magnitude_bins_split_units_and_area():
    u = _units([("small", "soy", 100, 100, 105, 95),      # ~5 % changes, opposite signs
                ("big", "soy", 100, 100, 300, 250)])      # large change, same sign
    m = rc.trend_by_magnitude(rc.trend_units(u, W, 50), [0.10, 0.50]).set_index("bin")
    assert list(m.index) == [0, 2]
    assert m.loc[0, "share_same_sign"] == 0.0 and m.loc[2, "share_same_sign"] == 1.0
    assert m.loc[0, "area_share"] == pytest.approx(95 / 345) and m["area_share"].sum() == pytest.approx(1.0)


def test_landcover_change_in_percentage_points():
    rows = []
    for y in (2008, 2009, 2020, 2021):
        late = y >= 2020
        rows += [{"geocode": 1, "year": y, "class_id": 15, "area_ha": 30.0 if late else 60.0},   # pasture 60 % -> 30 %
                 {"geocode": 1, "year": y, "class_id": 3, "area_ha": 20.0},                       # forest stays 20 %
                 {"geocode": 1, "year": y, "class_id": 39, "area_ha": 50.0 if late else 20.0}]    # soybean, not a cover
    c = rc.landcover_change(pd.DataFrame(rows), W, {"natural": (3,), "pasture": (15,)}).set_index("geocode")
    assert c.loc[1, "dpp_pasture"] == pytest.approx(-30.0) and c.loc[1, "dpp_natural"] == pytest.approx(0.0)


def test_trend_landcover_reads_the_gap_against_the_cover():
    rng = np.random.default_rng(5)
    n = 40
    dpp = np.linspace(-20, 0, n)
    # map growth exceeds the statistics where pasture was lost: gap falls as dpp rises
    rows = [(str(i), "cane", 100, 100, 100 * np.exp(0.5 - 0.02 * d + rng.normal(0, 0.01)), 150) for i, d in enumerate(dpp)]
    tu = rc.trend_units(_units(rows), W, 50)
    ch = pd.DataFrame({"geocode": [str(i) for i in range(n)], "dpp_pasture": dpp})
    t = rc.trend_landcover(tu, ch)
    assert set(t["quantile"]) == {1, 2, 3, 4} and t["n_units"].sum() == n
    assert t["spearman_rho"].iloc[0] < -0.9
    q = t.set_index("quantile")["median_gap"]
    assert q[1] > q[4]
