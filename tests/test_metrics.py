import numpy as np
import pandas as pd
import pytest

from bbp.metrics import agreement, lin_ccc, top_decile_turnover


def test_ccc_identity_and_shift():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert lin_ccc(x, x) == pytest.approx(1.0)
    # perfectly correlated but biased -> CCC < 1
    assert lin_ccc(x, x + 1) < 1.0


def test_agreement_counts_and_tolerance():
    obs = np.array([100.0, 109.0, 150.0, 0.0, 30.0, 10.0])
    dec = np.array([100.0, 100.0, 100.0, 80.0, 0.0, 5.0])
    m = agreement(obs, dec, tolerances=(0.10,), min_area=20)
    # last unit: both below 20 ha -> not scored
    assert m["n_units"] == 5
    assert m["n_both"] == 3
    assert m["n_obs_only"] == 1 and m["n_dec_only"] == 1
    # within 10 %: units 1 and 2 of 5
    assert m["share_within_10pct"] == pytest.approx(2 / 5)
    assert m["ratio_total"] == pytest.approx((100 + 109 + 150 + 0 + 30) / (100 + 100 + 100 + 80 + 0))


def test_agreement_ignores_nan():
    m = agreement(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, np.nan]))
    assert m["n_units"] == 1


def test_top_decile_turnover():
    a = pd.Series(np.arange(100.0))
    assert top_decile_turnover(a, a) == 0.0
    assert top_decile_turnover(a, -a) == 1.0
