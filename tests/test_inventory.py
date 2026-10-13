import numpy as np
import pandas as pd
import pytest

from bbp.inventory import factor_envelope, n1_table, yields_with_fallback

SPEC = {"g": {"min": 1.0, "ref": 1.1, "max": 1.2}, "ts": {"min": 0.7, "ref": 0.8, "max": 0.9},
        "vs_ts": {"min": 0.8, "ref": 0.85, "max": 0.9}, "bmp": {"min": 150, "ref": 230, "max": 300}}


def test_units_nm3_per_t():
    # 1 t grain -> 1.1 t stover -> 0.88 t DM -> 0.748 t VS -> x 230 Nm3/t VS
    env = factor_envelope(SPEC)
    assert env["ref"] == pytest.approx(1.1 * 0.8 * 0.85 * 230)
    assert env["min"] < env["ref"] < env["max"]
    assert env["oat_bmp_min"] == pytest.approx(1.1 * 0.8 * 0.85 * 150)


def _area():
    return pd.DataFrame({
        "geocode": [1, 2, 3], "year": [2020] * 3, "crop": ["corn_2nd"] * 3,
        "a_obs_ha": [100.0, 50.0, 10.0], "a_dec_ha": [100.0, 40.0, 0.0],
        "prod_t": [500.0, 200.0, 0.0], "obs_source": ["mapbiomas_second_crop"] * 3,
    })


ATTRS = pd.DataFrame({"geocode": [1, 2, 3], "cd_rgint": [10, 10, 10], "uf": ["MT"] * 3})


def test_yield_fallback_to_region():
    y = yields_with_fallback(_area(), ATTRS, ["rgint", "uf", "nation"]).set_index("geocode")
    assert y.loc[1, "yield_t_ha"] == 5.0 and y.loc[1, "yield_source"] == "municipality"
    # municipality 3 has no declared area -> regional yield (700 t / 140 ha)
    assert y.loc[3, "yield_t_ha"] == pytest.approx(5.0) and y.loc[3, "yield_source"] == "rgint"


def test_n1_variants_differ_only_through_area():
    crops = {"corn_2nd": {"feedstock": "PALHA_MILHO"}}
    factors = {"feedstocks": {"PALHA_MILHO": SPEC}}
    n1 = n1_table(_area(), ATTRS, crops, factors, ["rgint", "uf", "nation"])
    w = n1.pivot_table(index="geocode", columns="variant", values="n1_ref_nm3", observed=True)
    ref = factor_envelope(SPEC)["ref"]
    assert w.loc[1, "dec"] == pytest.approx(w.loc[1, "obs"])          # same area -> same N1
    assert w.loc[2, "obs"] / w.loc[2, "dec"] == pytest.approx(50 / 40)  # ratio of areas
    assert w.loc[3, "dec"] == 0 and w.loc[3, "obs"] == pytest.approx(10 * 5.0 * ref)
    assert not np.isnan(w.values).any()
