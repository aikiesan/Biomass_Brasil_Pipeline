import json

import numpy as np
import pytest

from bbp import landcover as lc

RES = 0.00026949458523585647  # MapBiomas 30 m grid, degrees


def _write(path, arr, x0, y0):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(x0, y0, RES, RES), nodata=0) as dst:
        dst.write(arr, 1)


def _read(path):
    import rasterio

    with rasterio.open(path) as ds:
        return ds.read(1), ds.transform


def test_every_class_of_the_legend_has_one_group():
    lut = lc.lookup_table()
    for c in (3, 4, 9, 12, 15, 20, 21, 24, 33, 39, 40, 41, 46, 47, 62, 75, 91):
        assert lut[c] not in (0, lc.UNMAPPED), c
    assert lut[0] == lut[27] == 0
    assert lut[200] == lc.UNMAPPED


def test_a_cell_takes_its_most_frequent_group_and_crops_win_ties():
    c = np.zeros((1, 3, lc.NBINS), dtype=np.int64)
    c[0, 0, lc.CODE["sugarcane"]], c[0, 0, lc.CODE["pasture"]] = 3, 5        # pasture covers more
    c[0, 1, lc.CODE["sugarcane"]], c[0, 1, lc.CODE["natural"]] = 4, 4        # tie: the crop
    c[0, 2, lc.UNMAPPED] = 9                                                 # nothing mapped: no data
    assert lc.decide(c).tolist() == [[lc.CODE["pasture"], lc.CODE["sugarcane"], 0]]


def test_soybean_groups_count_together_then_split():
    c = np.zeros((1, 2, lc.NBINS), dtype=np.int64)
    # 3 + 3 soybean pixels beat 5 pasture; corn after soybean on more of them -> soybean + corn
    c[0, 0, lc.CODE["soybean"]], c[0, 0, lc.CODE["soy_corn2"]], c[0, 0, lc.CODE["soy_cotton2"]] = 2, 3, 1
    c[0, 0, lc.CODE["pasture"]] = 5
    c[0, 1, lc.CODE["soybean"]], c[0, 1, lc.CODE["soy_corn2"]] = 2, 2           # tie inside: plain soybean
    assert lc.decide(c).tolist() == [[lc.CODE["soy_corn2"], lc.CODE["soybean"]]]


def test_coarsen_sums_blocks_and_pads_edges():
    c = np.ones((3, 5, lc.NBINS), dtype=np.int64)
    out = lc.coarsen(c, 2)
    assert out.shape == (2, 3, lc.NBINS)
    assert out[0, 0, 0] == 4 and out[1, 2, 0] == 1 and out.sum() == c.sum()


def test_aggregate_two_tiles_second_crop_and_resume(tmp_path):
    # a 32 x 64 mosaic in two 32 x 32 tiles; factor 4 and 8
    left = np.full((32, 32), 15, dtype=np.uint8)            # pasture
    left[:16, :16] = 20                                     # sugarcane block
    right = np.full((32, 32), 39, dtype=np.uint8)           # soybean
    right[:, 24:] = 3                                       # forest at the east edge
    x0, y0 = -50.0, -10.0
    _write(tmp_path / "cov-0000000000-0000000000.tif", left, x0, y0)
    _write(tmp_path / "cov-0000000000-0000000032.tif", right, x0 + 32 * RES, y0)
    sc = np.zeros((40, 80), dtype=np.uint8)                 # larger second-crop grid, offset by (2, 3) px
    sc[3:3 + 32, 2 + 32:2 + 44] = lc.SECOND_CORN            # corn after soybean on mosaic columns 32–43
    _write(tmp_path / "second.tif", sc, x0 - 2 * RES, y0 + 3 * RES)
    tiles = sorted(tmp_path.glob("cov-*.tif"))

    meta = lc.aggregate(tiles, tmp_path / "out", "t", factors=(4, 8), second_crop=tmp_path / "second.tif",
                        band_rows=8, chunk_cols=16, time_budget=0.0)
    assert not meta["complete"] and meta["bands_done"] == 0            # budget spent before the first band
    meta = lc.aggregate(tiles, tmp_path / "out", "t", factors=(4, 8), second_crop=tmp_path / "second.tif",
                        band_rows=8, chunk_cols=16)
    assert meta["complete"] and meta["bands_done"] == 4
    g4, tr4 = _read(tmp_path / "out" / "t_f4.tif")
    g8, _ = _read(tmp_path / "out" / "t_f8.tif")
    assert g4.shape == (8, 16) and g8.shape == (4, 8)
    assert tr4.a == pytest.approx(4 * RES) and tr4.c == pytest.approx(x0)
    assert (g4[:4, :4] == lc.CODE["sugarcane"]).all() and (g4[4:, :8] == lc.CODE["pasture"]).all()
    assert (g4[:, 8:11] == lc.CODE["soy_corn2"]).all()
    assert (g4[:, 11:14] == lc.CODE["soybean"]).all() and (g4[:, 14:] == lc.CODE["natural"]).all()
    # factor 8 from the same counts; columns 40–47 hold 32 soybean + corn and 32 plain soybean pixels: a tie
    assert (g8[:, 4] == lc.CODE["soy_corn2"]).all() and (g8[:, 5] == lc.CODE["soybean"]).all()
    assert g8[0, 0] == lc.CODE["sugarcane"]
    # areas: every pixel accounted for, soybean class = its three groups
    px = np.asarray(meta["pixels"])
    assert px[1:].sum() == 32 * 64
    assert lc.class_area(meta, 39) == pytest.approx(lc.area_by_group(meta)["soybean"]
                                                    + lc.area_by_group(meta)["soy_corn2"])
    px_ha = (RES * 111_320) ** 2 * np.cos(np.radians(10.0)) / 1e4
    assert lc.area_by_group(meta)["sugarcane"] == pytest.approx(256 * px_ha, rel=0.01)
    # a finished run is not redone; changed parameters start again
    again = lc.aggregate(tiles, tmp_path / "out", "t", factors=(4, 8), second_crop=tmp_path / "second.tif",
                         band_rows=8, chunk_cols=16)
    assert again == json.loads((tmp_path / "out" / "t.json").read_text(encoding="utf-8"))
    no_sc = lc.aggregate(tiles, tmp_path / "out", "t", factors=(4, 8), band_rows=8, chunk_cols=16)
    assert no_sc["complete"] and no_sc["params"]["second_crop"] is None
    assert (_read(tmp_path / "out" / "t_f4.tif")[0][:, 8:14] == lc.CODE["soybean"]).all()


def test_aggregate_a_window(tmp_path):
    cov = np.full((64, 64), 15, dtype=np.uint8)
    cov[20:40, 20:40] = 46                                   # coffee in the middle
    _write(tmp_path / "cov.tif", cov, -45.0, -20.0)
    b = (-45.0 + 18 * RES, -20.0 - 42 * RES, -45.0 + 41 * RES, -20.0 - 19 * RES)   # lon0, lat0, lon1, lat1
    meta = lc.aggregate(tmp_path / "cov.tif", tmp_path / "out", "w", factors=(4,), bounds=b, band_rows=8,
                        chunk_cols=8)
    c0, r0, c1, r1 = meta["params"]["window_px"]
    assert (c0, r0) == (16, 16) and (c1, r1) == (44, 44)              # snapped outward to multiples of 4
    g, tr = _read(tmp_path / "out" / "w_f4.tif")
    assert g.shape == (7, 7) and tr.c == pytest.approx(-45.0 + 16 * RES)
    assert g[3, 3] == lc.CODE["coffee"] and g[0, 0] == lc.CODE["pasture"]


def test_tiles_off_the_grid_are_refused(tmp_path):
    a = np.full((8, 8), 15, dtype=np.uint8)
    _write(tmp_path / "a.tif", a, -50.0, -10.0)
    _write(tmp_path / "b.tif", a, -50.0 + 8.5 * RES, -10.0)
    with pytest.raises(ValueError, match="pixel-aligned"):
        lc.Mosaic.open([tmp_path / "a.tif", tmp_path / "b.tif"])


def test_counts_raster_keeps_every_pixel(tmp_path):
    import rasterio

    cov = np.full((32, 32), 15, dtype=np.uint8)
    cov[:10, :7] = 20                                        # 70 sugarcane pixels across cell borders
    _write(tmp_path / "cov.tif", cov, -45.0, -20.0)
    meta = lc.aggregate(tmp_path / "cov.tif", tmp_path / "out", "c", factors=(4, 8), band_rows=8, chunk_cols=8,
                        counts=True)
    assert meta["complete"] and meta["params"]["counts"]
    arr, tr = lc.read_counts(lc.counts_path(tmp_path / "out", "c", 4), ["sugarcane", "pasture"])
    assert arr.shape == (2, 8, 8) and tr.a == pytest.approx(4 * RES)
    assert arr[0].sum() == 70 and arr[1].sum() == 32 * 32 - 70 and arr[0, 0, 0] == 16
    with rasterio.open(lc.counts_path(tmp_path / "out", "c", 4)) as ds:
        assert ds.count == lc.NG and ds.descriptions[lc.CODE["sugarcane"] - 1] == "sugarcane"
