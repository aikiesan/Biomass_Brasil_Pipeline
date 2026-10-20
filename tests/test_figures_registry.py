"""The paper's figure registry, its captions, the composite canvas and the hot-spot statistics (no source data)."""

import numpy as np
import pandas as pd
import pytest

from bbp import figstyle, figures
from bbp.analyse import getis_ord_gi_star, local_moran
from bbp.captions import captions
from bbp.checks import CheckLog
from bbp.config import load_config


def _lattice(n=6):
    """Rook-and-diagonal (queen) neighbours of an n × n grid; codes r * n + c."""
    nb = {}
    for r in range(n):
        for c in range(n):
            nb[r * n + c] = [rr * n + cc for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)
                             if 0 <= rr < n and 0 <= cc < n and (rr, cc) != (r, c)]
    return nb


def test_every_figure_has_a_caption_and_a_unique_file():
    cfg = load_config(profile="paper")
    caps = captions(cfg, {}, {})
    keys = [*figures.FIGURES, *figures.SUPPLEMENT]
    names = [(figures.FIGURES.get(k) or figures.SUPPLEMENT[k])[0] for k in keys]
    assert len(set(names)) == len(names)
    assert not [n for n in names if n not in caps], "figures without a caption"
    assert list(figures.FIGURES) == list(range(1, 10))                     # nine figures in the paper
    assert len({figures.file_name(k) for k in keys}) == len(keys)


def test_gi_star_matches_the_formula():
    nb = _lattice(4)
    rng = np.random.default_rng(1)
    v = pd.Series(rng.uniform(0, 10, 16), index=range(16))
    z = getis_ord_gi_star(v, nb)
    x, n = v.to_numpy(), 16
    i = 5
    js = [i, *nb[i]]
    w = len(js)
    s = np.sqrt((x ** 2).mean() - x.mean() ** 2)
    expect = (x[js].sum() - x.mean() * w) / (s * np.sqrt((n * w - w ** 2) / (n - 1)))
    assert z[i] == pytest.approx(expect)


def test_hot_spots_find_a_high_block():
    nb = _lattice(8)
    v = pd.Series(1.0, index=range(64))
    high = [r * 8 + c for r in range(3) for c in range(3)]                 # 3 × 3 block in a corner
    v[high] = 10.0
    v += np.random.default_rng(0).normal(0, 0.1, 64)
    z = getis_ord_gi_star(v, nb)
    assert z[9] > 2.576 and z[63] < 1.0                                    # centre of the block is hot
    lisa = local_moran(v, nb, permutations=199, seed=3)
    assert lisa.loc[9, "quadrant"] == "HH" and lisa.loc[9, "p_value"] < 0.05
    assert lisa.loc[63, "quadrant"] == "LL"
    assert lisa["p_value"].between(0, 1).all()


def test_local_moran_leaves_islands_out():
    v = pd.Series([1.0, 2.0, 3.0, 4.0], index=[1, 2, 3, 4])
    lisa = local_moran(v, {1: [2], 2: [1, 3], 3: [2], 4: []}, permutations=99)
    assert np.isnan(lisa.loc[4, "local_i"]) and lisa.loc[4, "quadrant"] == ""


def test_stack_draws_parts_in_place_and_drops_the_measuring_checks():
    import matplotlib.pyplot as plt

    figstyle.setup()
    ck = CheckLog("figures")

    def part(h, name):
        def draw(host):
            cv = figstyle.canvas(100.0, h, host)
            cv.panel("box", 1, 1, 99, h - 1)
            ck.add(f"{name}_drawn", True)
            return cv
        return draw

    st = figstyle.Stack(gap=2.0, ck=ck)
    st.add("a", part(30.0, "a"), head="(a) First")
    st.add("b", part(20.0, "b"))
    st.add("c", lambda host: None)                                        # inputs missing: left out
    cv = st.draw()
    assert (cv.fw, cv.fh) == (100.0, pytest.approx(30 + figstyle.HEAD_H + 2 + 20))
    assert [c.name for c in ck.checks] == ["a_drawn", "b_drawn"]           # one check per part, not two
    assert set(cv.panels) >= {"a_box", "b_box"}
    with pytest.raises(RuntimeError):
        figstyle.SubCanvas(cv, 0, 0, 10, 10).save()
    plt.close(cv.fig)


def test_checks_caught_reads_the_logs():
    cfg = load_config(profile="paper")
    counts = {"plants": {"checks": [
        {"name": "plants_biomethane_production_notation", "value": 2, "detail": ""},
        {"name": "plants_point_in_declared_municipality_anp_biodiesel_plants", "value": 0.16,
         "detail": "ANP biodiesel plants: 12 of 73 points outside"},
        {"name": "plants_possible_duplicates", "value": {"biogas": 400, "biodiesel": 7}, "detail": ""}]},
        "reach": {"checks": [{"name": "reach_soybean_area_matches_statistics", "value": -2e-6, "detail": ""}]}}
    items = figures.checks_caught(cfg, counts)
    assert any("12 of 73" in s for s in items) and any("407 pairs" in s for s in items)
    assert any("0.0002%" in s for s in items)
    assert not any("{" in s for s in items)                                # nothing left unfilled
    empty = figures.checks_caught(cfg, {})                                 # no logs: items from logs left out
    assert empty and not any("{" in s for s in empty) and len(empty) < len(items)
