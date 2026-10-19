"""Fig 1 (study area): land-cover display rasters, cities, connectors, labels, the 2025 variant."""

import json

import numpy as np
import pytest
from test_paper import _write_run

from bbp import figures
from bbp import landcover as lc
from bbp.checks import CheckLog
from bbp.config import load_config

RES = 0.00026949458523585647


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("BBP_DATA_DIR", str(tmp_path))
    cfg = load_config(profile="paper")
    _write_run(tmp_path, cfg)
    return cfg


def _display(path, codes, x0, y0, res):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", height=codes.shape[0], width=codes.shape[1], count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(x0, y0, res, res), nodata=0) as dst:
        dst.write(codes, 1)


def _landcover_cache(data_dir, year, second_crop: bool):
    """Display rasters as `bbp landcover` leaves them, over the synthetic municipalities (-50..-46, -24..-21)."""
    out = data_dir / "interim" / "landcover"
    out.mkdir(parents=True, exist_ok=True)
    for f in figures.LC_FACTORS:
        res = RES * f
        n_x, n_y = int(4.6 / res), int(3.6 / res)
        codes = np.full((n_y, n_x), lc.CODE["pasture"], dtype=np.uint8)
        codes[: n_y // 2, : n_x // 2] = lc.CODE["sugarcane"]
        if second_crop:
            codes[n_y // 2:, n_x // 2:] = lc.CODE["soy_corn2"]
        _display(out / f"col11_{year}_f{f}.tif", codes, -50.3, -20.7, res)
    area = [0.0] * lc.NBINS
    area[lc.CODE["sugarcane"]], area[lc.CODE["pasture"]] = 5e6, 9e6
    if second_crop:
        area[lc.CODE["soy_corn2"]] = 2e6
    meta = {"params": {"second_crop": "2024_second_crop.tif" if second_crop else None}, "complete": True,
            "bands_done": 1, "area_ha": area, "pixels": [0] * lc.NBINS, "elapsed_s": 1.0}
    (out / f"col11_{year}.json").write_text(json.dumps(meta), encoding="utf-8")


def test_fig01_draws_the_land_cover_and_its_variant(run, tmp_path):
    cfg = run
    _landcover_cache(tmp_path, 2024, second_crop=True)
    _landcover_cache(tmp_path, 2025, second_crop=False)
    cfg.raw["paper"]["figure_formats"] = ["png"]
    cfg.raw["paper"]["dpi"] = 60
    ck = CheckLog("figures")
    written = figures.build(cfg, ck, only=[1])
    assert not ck.failures, [(c.name, c.detail) for c in ck.failures]
    names = sorted(p.name for p in written)
    assert names == ["fig01_study_area.png", "fig01_study_area_2025.png"]
    passed = {c.name for c in ck.checks if c.passed}
    assert {"fig01_landcover_2024", "fig01_landcover_2025"} <= passed
    caption = (tmp_path / "paper" / "captions.md").read_text(encoding="utf-8")
    assert "Study area at three resolutions" in caption and "2nd-season corn (2.0 of the 2.0 Mha" in caption


def test_fig01_without_rasters_still_draws_the_base_map(run):
    cfg = run
    cfg.raw["inputs"]["coverage_tif_dir"] = "no_such_folder"
    ck = CheckLog("figures")
    data = figures._Data(cfg, ck)
    cv = figures.fig01_study_area(cfg, ck, data)
    assert cv is not None and cv.fw == 132.0
    assert any(c.name == "fig01_landcover_2024" and not c.passed and c.severity == "warn" for c in ck.checks)
    assert figures.fig01_study_area(cfg, ck, data, year=2025) is None      # a variant needs its own raster


def test_city_points_sit_on_the_urban_cells(tmp_path):
    import pandas as pd
    from shapely.geometry import box

    codes = np.full((40, 40), lc.CODE["pasture"], dtype=np.uint8)
    codes[2:6, 30:34] = lc.CODE["built"]                         # town in the north-east corner
    _display(tmp_path / "lc.tif", codes, -48.0, -20.0, 0.01)
    g = pd.Series({1: box(-48.0, -20.4, -47.6, -20.0), 2: box(-47.6, -20.4, -47.2, -20.0)})
    pts = figures.city_points(g, [1, 2], tmp_path / "lc.tif")
    assert pts[1] == pytest.approx((-47.68, -20.04), abs=0.006)
    assert pts[2] == pytest.approx((g[2].representative_point().x, g[2].representative_point().y))   # no town


def test_connectors_join_the_sides_that_face_each_other():
    from bbp.figstyle import Canvas

    cv = Canvas(100, 100)
    figures._connect_boxes(cv, (50, 50, 60, 60), (5, 40, 30, 70))        # panel to the left
    figures._connect_boxes(cv, (50, 50, 60, 60), (40, 5, 90, 30))        # panel below
    ends = [tuple(np.round(ln.get_xydata(), 3).ravel()) for ln in cv.ov.lines]
    assert (50, 60, 30, 70) in ends and (50, 50, 30, 40) in ends
    assert (50, 50, 40, 30) in ends and (60, 50, 90, 30) in ends


def test_city_labels_do_not_overlap():
    from bbp.figstyle import Canvas

    cv = Canvas(60, 40)
    pts = [("Ribeirão Preto", 30, 20), ("Araraquara", 30.5, 18.5), ("Sertãozinho", 29, 21.5)]
    boxes = figures._place_cities(cv, pts, (0, 0, 60, 40), 5.0)
    assert len(boxes) == 3
    assert not any(figures._overlap(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:])


def test_ghost_colours_are_light_greys():
    from bbp import figstyle as fs
    from bbp.cartography import ghost_lut, landcover_rgba

    lut = landcover_rgba(fs.LANDCOVER_COLORS)
    g = ghost_lut(lut)
    for key in ("soybean", "sugarcane", "coffee"):
        r, gg, b, a = g[lc.CODE[key]]
        assert r == gg == b and r >= 170 and a == 255
    assert (g[0] == lut[0]).all()                                    # no data stays transparent
