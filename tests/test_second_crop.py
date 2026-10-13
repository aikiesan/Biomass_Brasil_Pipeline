"""End-to-end test of the raster stage on a tiny synthetic grid."""

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import box  # noqa: E402

from bbp.ingest.second_crop import build_label_grid, row_pixel_area_m2, zonal_area  # noqa: E402

RES = 0.01  # degrees
T = from_origin(-50.0, -10.0, RES, RES)  # 40 x 40 pixels, lon -50..-49.6, lat -10..-10.4


def _write(path, arr):
    prof = dict(driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype="uint8",
                crs="EPSG:4326", transform=T, nodata=0)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)


def test_pixel_area_matches_equator_order_of_magnitude():
    a = row_pixel_area_m2(T, 2)
    # 0.01 deg at ~10 S ~ 1.1 km x 1.09 km
    assert 1.15e6 < a[0] < 1.25e6
    assert a[0] == pytest.approx(a[1], rel=1e-3)


def test_zonal_area(tmp_path):
    mesh = gpd.GeoDataFrame(
        {"geocode": [1111111, 2222222]},
        geometry=[box(-50.0, -10.4, -49.8, -10.0), box(-49.8, -10.4, -49.6, -10.0)], crs="EPSG:4326",
    )
    data = np.zeros((40, 40), dtype="uint8")
    data[:10, :20] = 1      # corn, west municipality: 200 px
    data[:10, 20:25] = 62   # cotton, east: 50 px
    data[30:, 30:] = 7      # other, east: 100 px
    tif = tmp_path / "2020_second_crop.tif"
    _write(tif, data)

    grid = tmp_path / "grid.tif"
    lookup = build_label_grid(mesh, tif, grid, window=16)
    assert len(lookup) == 2

    table, diag = zonal_area(grid, {2020: tif}, {1: "corn_2nd", 62: "cotton_2nd"}, window=16)
    t = table.set_index(["geocode", "class"])["area_ha"]
    px_ha = row_pixel_area_m2(T, 40) / 1e4
    assert t[(1111111, "corn_2nd")] == pytest.approx(px_ha[:10].sum() * 20)
    assert t[(2222222, "cotton_2nd")] == pytest.approx(px_ha[:10].sum() * 5)
    assert t[(2222222, "other_2nd")] == pytest.approx(px_ha[30:].sum() * 10)
    assert diag["pixel_values_seen"][2020] == [1, 7, 62]
    assert sum(diag["unassigned_ha"][2020].values()) == 0
