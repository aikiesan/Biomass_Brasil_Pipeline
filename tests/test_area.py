import numpy as np
import pandas as pd

from bbp.area import crop_area_table

CROPS = {
    "cotton": {"mapbiomas_class": [62], "mapbiomas_second_crop": "cotton_2nd", "pam_table": 1612,
               "pam_product": "Algodão herbáceo (em caroço)"},
    "soybean": {"mapbiomas_class": [39], "pam_table": 5457, "pam_product": "Soja (em grão)"},
}


def _inputs():
    mb = pd.DataFrame({"geocode": [1, 1, 1, 1], "year": [2024, 2025, 2024, 2025], "class_id": [62, 62, 39, 39],
                       "area_ha": [100.0, 110.0, 500.0, 520.0]})
    sc = pd.DataFrame({"geocode": [1], "year": [2024], "class": ["cotton_2nd"], "area_ha": [40.0]})  # rasters end 2024
    pam = pd.DataFrame({"table": [1612] * 2, "geocode": [1, 1], "year": [2024, 2025], "product": ["Algodão herbáceo (em caroço)"] * 2,
                        "variable": ["area_planted_ha"] * 2, "value": [150.0, 160.0], "flag": ["ok"] * 2,
                        "preliminary": [False, True]})
    return mb, sc, pam


def test_cotton_sums_first_and_second_season_and_needs_both_sources():
    mb, sc, pam = _inputs()
    a = crop_area_table(pd.Series([1]), [2024, 2025], CROPS, mb, pam, sc).set_index(["crop", "year"])
    assert a.loc[("cotton", 2024), "a_obs_ha"] == 140.0           # class 62 + second-season cotton
    assert np.isnan(a.loc[("cotton", 2025), "a_obs_ha"])           # no second-crop raster for 2025: not observed
    assert a.loc[("cotton", 2024), "obs_source"] == "mapbiomas_coverage+mapbiomas_second_crop"
    assert a.loc[("soybean", 2025), "a_obs_ha"] == 520.0           # coverage-only crops are unaffected
    assert a.loc[("cotton", 2025), "dec_preliminary"] and not a.loc[("cotton", 2024), "dec_preliminary"]
