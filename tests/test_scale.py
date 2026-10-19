"""bbp.scale: the same land within reach at the pixel, the municipality and the region, on synthetic data."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from bbp import scale
from bbp.checks import CheckLog
from bbp.reach import haversine_km, unit_xyz

R = [10.0, 30.0, math.inf]


def _tree(lon, lat):
    return cKDTree(unit_xyz(np.atleast_1d(lon), np.atleast_1d(lat)))


def test_rules_area_fraction_centroid_and_point():
    # unit A: half its land within 10 km, all within 30 km, centroid 5 km from the plant; unit B: far away
    plant = (-47.0, -21.0)
    km = 1 / 111.2
    crop = pd.Series({1: 100.0, 2: 50.0})
    land = pd.Series({1: 10.0, 2: 10.0})
    land_in = pd.DataFrame({10.0: [5.0, 0.0], 30.0: [10.0, 0.0], math.inf: [10.0, 10.0]}, index=[1, 2])
    pts = pd.DataFrame({"centroid_lon": [plant[0], plant[0]], "centroid_lat": [plant[1] - 5 * km, plant[1] - 50 * km],
                        "representative_point_lon": [plant[0], plant[0]],
                        "representative_point_lat": [plant[1] - 12 * km, plant[1] - 50 * km]}, index=[1, 2])
    t = scale.unit_table(crop, land, land_in, pts, _tree(*plant), R).set_index(["unit", "rule", "radius_km"])
    w = t["within_ha"]
    assert w[(1, "area_fraction", 10.0)] == pytest.approx(50.0) and w[(1, "area_fraction", 30.0)] == 100.0
    assert w[(2, "area_fraction", 30.0)] == 0.0 and w[(2, "area_fraction", math.inf)] == 50.0
    assert w[(1, "centroid", 10.0)] == 100.0 and w[(1, "representative_point", 10.0)] == 0.0   # 12 km away
    assert w[(1, "representative_point", 30.0)] == 100.0
    assert all(w[(u, r, math.inf)] == crop[u] for u in (1, 2) for r in scale.RULES)


def test_unit_without_land_cell_uses_its_centroid():
    crop = pd.Series({1: 10.0})
    land = pd.Series({}, dtype=float)
    land_in = pd.DataFrame(columns=R, dtype=float)
    pts = pd.DataFrame({"centroid_lon": [-47.0], "centroid_lat": [-21.0], "representative_point_lon": [-47.0],
                        "representative_point_lat": [-21.0]}, index=[1])
    t = scale.unit_table(crop, land, land_in, pts, _tree(-47.0, -21.05), R)
    af = t[t["rule"] == "area_fraction"].set_index("radius_km")["within_ha"]
    assert af[10.0] == 10.0 and t.attrs["no_land_units"] == 1


def test_land_within_holds_every_cell():
    from rasterio.transform import from_origin

    res = 0.01
    grid = np.array([[1, 1, 2], [1, 2, 2], [0, 2, 2]], dtype=np.int32)
    tr = from_origin(-47.0, -21.0, res, res)
    tree = _tree(-47.0 + 0.5 * res, -21.0 - 0.5 * res)          # centre of the top-left cell
    land, within = scale.land_within(grid, tr, {"f": tree}, R, rows=2)
    from bbp.ingest.second_crop import row_pixel_area_m2

    a = row_pixel_area_m2(tr, 3) / 1e4
    assert land[1] == pytest.approx(2 * a[0] + a[1]) and land[2] == pytest.approx(a[0] + 2 * a[1] + 2 * a[2])
    w = within["f"]
    assert np.allclose(w[math.inf], land)
    d = haversine_km(-47.0 + 0.5 * res, -21.0 - 0.5 * res, -47.0 + 2.5 * res, -21.0 - 2.5 * res)
    assert d < 10                                                  # the whole 3 x 3 block is within 10 km
    assert np.allclose(w[10.0], land)


def test_pixel_shares_use_municipal_bins_then_the_national_curve():
    hist = pd.DataFrame({"pair": "c|f", "bin_km": [0.0, 5.0, 40.0, 150.0], "area_ha": [10.0, 20.0, 30.0, 40.0]})
    dist = pd.DataFrame({"pair": "c|f", "bin_km": np.arange(2000.0), "area_ha": 0.0})
    dist.loc[[0, 5, 40, 400], "area_ha"] = [10.0, 20.0, 30.0, 40.0]
    total, s = scale.pixel_shares(hist, dist, "c|f", [10.0, 50.0, 300.0, 500.0, math.inf], max_km=150)
    assert total == 100.0
    assert s[10.0] == 30.0 and s[50.0] == 60.0 and s[300.0] == 60.0 and s[500.0] == 100.0 and s[math.inf] == 100.0


def _nat(share_inf=1.0):
    rows = []
    for res, rule in (("pixel", "pixel"), *[(r, x) for r in ("municipality", "rgint") for x in scale.RULES]):
        for r, s in ((10.0, 0.2), (30.0, 0.5), (math.inf, share_inf if res == "rgint" else 1.0)):
            rows.append({"pair": "c|f", "resolution": res, "rule": rule, "radius_km": r, "within_ha": 100 * s,
                         "crop_area_ha": 100.0, "share": s})
    return pd.DataFrame(rows)


def test_checks_convergence_and_reference():
    ck = CheckLog("scale")
    scale.check(ck, _nat(), pd.DataFrame(), {"c|f": {"pixel": 0.48, "rgint": 0.40}})
    assert not ck.failures
    ref = next(c for c in ck.checks if c.name == "scale_c|f_vs_exploratory_x05")
    assert ref.value == {"pixel": 2.0, "rgint": 10.0}
    ck = CheckLog("scale")
    scale.check(ck, _nat(share_inf=0.9), pd.DataFrame())
    assert any(c.name == "scale_converges_to_100" for c in ck.failures)
