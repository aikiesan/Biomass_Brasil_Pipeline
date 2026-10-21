import numpy as np
import pandas as pd
import pytest

from bbp.ingest.border_share import (
    ALL,
    border_area,
    boundary_mask,
    collection_agreement,
    distance_to_boundary_m,
)
from bbp.ingest.mapbiomas import parse_irrigation_rows

RES = 0.00026949458523585647  # MapBiomas 30 m grid, degrees


def test_boundary_only_between_two_municipalities():
    lab = np.array([[1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [0, 0, 0, 0]], dtype="uint16")       # row of "no municipality" (sea): not a border
    b = boundary_mask(lab)
    assert b[:2, 1].all() and b[:2, 2].all()          # the 1|2 line, both sides
    assert not b[:, 0].any() and not b[:, 3].any()
    assert not b[2].any()                              # coast is not a municipal border


def test_distance_in_metres_grows_away_from_the_border():
    lab = np.zeros((5, 200), dtype="uint16")
    lab[:, :100], lab[:, 100:] = 1, 2
    d = distance_to_boundary_m(lab, lat_deg=0.0, res_deg=RES, cap_m=2000)
    assert d[2, 99] == 0 and d[2, 100] == 0
    assert d[2, 99 - 10] == pytest.approx(10 * RES * 111_320, rel=1e-3)   # ~300 m, 10 px west of the line
    assert distance_to_boundary_m(np.ones((3, 3), "uint16"), 0, RES, 2000).min() == 2001


def _write(path, arr, transform):
    import rasterio

    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype, crs="EPSG:4326", transform=transform, nodata=0) as dst:
        dst.write(arr, 1)


def test_border_area_bins_and_all_crops(tmp_path):
    from rasterio.transform import from_origin

    tr = from_origin(-50.0, 0.0, RES, RES)
    lab = np.zeros((40, 300), dtype="uint16")
    lab[:, :150], lab[:, 150:] = 1, 2
    _write(tmp_path / "grid.tif", lab, tr)
    pd.DataFrame({"label": [1, 2], "geocode": [1100015, 1100023]}).to_csv(tmp_path / "grid.lookup.csv", index=False)
    cov = np.full((40, 300), 3, dtype="uint8")        # forest
    cov[:, 145:150] = 39                              # soybean touching the border (within 1 km)
    cov[:, 60:70] = 39                                # soybean 80–90 px (~2.4–2.7 km) from it
    cov[:, 120:125] = 20                              # sugarcane ~25–30 px (~0.8 km) from it
    _write(tmp_path / "cov2024.tif", cov, tr)

    t = border_area(tmp_path / "grid.tif", {2024: tmp_path / "cov2024.tif"}, {39: "soybean", 20: "sugarcane"},
                    [1000, 2000], window=64)
    soy = t[(t["geocode"] == 1100015) & (t["crop"] == "soybean")].iloc[0]
    assert soy["border_share_1km"] == pytest.approx(5 / 15, rel=1e-6)
    assert soy["border_share_2km"] == pytest.approx(5 / 15, rel=1e-6)   # the far block is > 2 km away
    cane = t[(t["geocode"] == 1100015) & (t["crop"] == "sugarcane")].iloc[0]
    assert cane["border_share_1km"] == pytest.approx(1.0)
    every = t[(t["geocode"] == 1100015) & (t["crop"] == ALL)].iloc[0]
    assert every["crop_area_ha"] == pytest.approx(soy["crop_area_ha"] + cane["crop_area_ha"])
    assert not (t["geocode"] == 1100023).any()        # no crops in municipality 2


def test_tiled_year_is_read_as_one_mosaic(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from bbp.ingest.border_share import build_vrt, find_year_sources

    full = np.arange(6 * 8, dtype="uint8").reshape(6, 8)
    for r0 in (0, 3):
        for c0 in (0, 4):
            tr = from_origin(-50.0 + c0 * RES, 0.0 - r0 * RES, RES, RES)
            _write(tmp_path / f"cov_2024-{r0:010d}-{c0:010d}.tif", full[r0:r0 + 3, c0:c0 + 4], tr)
    _write(tmp_path / "cov_2025.tif", full, from_origin(-50.0, 0.0, RES, RES))
    found = find_year_sources(tmp_path, "cov_{year}.tif", [2023, 2024, 2025])
    assert set(found) == {2024, 2025} and len(found[2024]) == 4
    meta = build_vrt(found[2024], tmp_path / "vrt" / "cov_2024.vrt")
    assert (meta["width"], meta["height"], meta["tiles"]) == (8, 6, 4)
    with rasterio.open(tmp_path / "vrt" / "cov_2024.vrt") as ds:
        assert (ds.read(1) == full).all()
        assert ds.transform.c == pytest.approx(-50.0)
    with pytest.raises(ValueError, match="gap or overlap"):
        build_vrt(found[2024][:3], tmp_path / "vrt" / "bad.vrt")


def test_collection_guard_compares_with_the_statistics():
    table = pd.DataFrame({"geocode": [1, 1], "year": [2024, 2024], "crop": ["soybean", ALL],
                          "crop_area_ha": [10_100.0, 10_100.0]})
    stats = pd.DataFrame({"geocode": [1], "year": [2024], "class_id": [39], "area_ha": [10_000.0]})
    rel = collection_agreement(table, stats, {39: "soybean"})["rel"]
    assert rel.tolist() == pytest.approx([0.01])


def test_irrigation_rows():
    header = ["COUNTRY", "BIOME", "MUNICIPALITY_GEOCODE", "IRRIGATION_CLASS_NAME", "AGRICULTURE_CLASS_ID",
              "AGRICULTURE_CLASS_NAME", "COLLECTION", "y2008", "y2009"]
    rows = [header,
            ["Brasil", "Cerrado", 5104542, "Center Pivot Irrigation", 39, "Soybean", 11, 0, 50.5],
            ["Brasil", "Cerrado", 5104542, "Other Irrigation Systems", 41, "Other", 11, 1.0, None],
            [None] * 9]
    df = parse_irrigation_rows(rows, [2008, 2009, 2010])
    assert len(df) == 3                                # one None cell dropped, the blank row skipped
    assert df.loc[df["year"] == 2009, "area_ha"].sum() == 50.5
    assert set(df["collection"]) == {11}
    with pytest.raises(ValueError):
        parse_irrigation_rows([["geocode", "y2008"]], [2008])
