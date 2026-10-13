"""MapBiomas second-crop rasters -> municipal area per class and year.

Two steps, both windowed so that the 131 072 x 131 072 national grid never
has to fit in memory:

1. `build_label_grid` rasterises the IBGE municipal mesh onto the exact grid of
   the second-crop rasters (pixel-centre rule, one municipality per pixel) and
   writes it once to data/interim/municipality_grid.tif (uint16 label + lookup).
2. `zonal_area` reads, window by window, the label grid and every year's
   raster, and accumulates the geodesic area (WGS84 ellipsoid) of each pixel
   value per municipality. Pixel area depends only on the row (latitude), so it
   is precomputed once per row.

Progress is checkpointed so a long run can be resumed after an interruption.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("bbp")

OTHER = "other_2nd"


def find_rasters(tif_dir: Path, pattern: str, years: list[int]) -> dict[int, Path]:
    found = {}
    for y in years:
        hits = sorted(tif_dir.glob(pattern.format(year=y)))
        if len(hits) > 1:
            raise ValueError(f"More than one raster for {y}: {[h.name for h in hits]}")
        if hits:
            found[y] = hits[0]
    return found


def iter_windows(width: int, height: int, size: int):
    from rasterio.windows import Window

    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width - col), min(size, height - row))


def row_pixel_area_m2(transform, height: int) -> np.ndarray:
    """Geodesic area (m²) of one pixel in each row of a north-up lat/lon grid (WGS84)."""
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    dx, dy = transform.a, transform.e  # dy < 0
    lon0 = transform.c
    top = transform.f + dy * np.arange(height)
    bottom = top + dy
    # The area of a lat/lon cell does not depend on longitude: compute one cell per row.
    areas = np.empty(height, dtype="float64")
    for i in range(height):
        lons = [lon0, lon0 + dx, lon0 + dx, lon0]
        lats = [bottom[i], bottom[i], top[i], top[i]]
        area, _ = geod.polygon_area_perimeter(lons, lats)
        areas[i] = abs(area)
    return areas


def build_label_grid(mesh_gdf, template: Path, out_tif: Path, window: int = 2048) -> pd.DataFrame:
    """Rasterise municipalities onto the template grid. Returns the label -> geocode lookup."""
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import bounds as window_bounds
    from rasterio.windows import transform as window_transform
    from shapely.geometry import box

    with rasterio.open(template) as src:
        profile = src.profile.copy()
        crs, transform, width, height = src.crs, src.transform, src.width, src.height

    gdf = mesh_gdf.to_crs(crs).reset_index(drop=True)
    if len(gdf) >= np.iinfo("uint16").max:
        raise ValueError("Too many polygons for a uint16 label grid")
    gdf["label"] = np.arange(1, len(gdf) + 1, dtype="uint16")
    lookup = pd.DataFrame({"label": gdf["label"].astype("int32"), "geocode": gdf["geocode"].astype("int64")})
    sindex = gdf.sindex

    profile.update(
        driver="GTiff", dtype="uint16", nodata=0, count=1, compress="deflate", predictor=2,
        tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES", SPARSE_OK="TRUE",
    )
    tmp = out_tif.with_suffix(".partial.tif")
    t0 = time.time()
    n_done = 0
    windows = list(iter_windows(width, height, window))
    with rasterio.open(tmp, "w", **profile) as dst:
        for k, win in enumerate(windows):
            hits = sindex.query(box(*window_bounds(win, transform)))
            if len(hits) == 0:
                continue
            sub = gdf.iloc[hits]
            arr = rasterize(
                zip(sub.geometry, sub["label"].astype("int32"), strict=True),
                out_shape=(int(win.height), int(win.width)),
                transform=window_transform(win, transform),
                fill=0, dtype="uint16", all_touched=False,
            )
            dst.write(arr, 1, window=win)
            n_done += 1
            if n_done % 100 == 0:
                log.info("  label grid: window %d/%d (%.0f s)", k + 1, len(windows), time.time() - t0)
    tmp.replace(out_tif)
    lookup.to_csv(out_tif.with_suffix(".lookup.csv"), index=False)
    log.info("Label grid written: %s (%d windows with municipalities, %.0f s)", out_tif.name, n_done, time.time() - t0)
    return lookup


def zonal_area(
    label_tif: Path,
    rasters: dict[int, Path],
    classes: dict[int, str],
    window: int = 2048,
    checkpoint: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Area (ha) per municipality label, year and class. Returns (table, diagnostics)."""
    import rasterio

    lookup = pd.read_csv(label_tif.with_suffix(".lookup.csv"))
    n_labels = int(lookup["label"].max()) + 1  # label 0 = outside every municipality

    class_names = list(dict.fromkeys([*classes.values(), OTHER]))
    lut = np.full(256, class_names.index(OTHER), dtype="int64")
    for value, name in classes.items():
        lut[int(value)] = class_names.index(name)
    n_cls = len(class_names)
    years = sorted(rasters)

    with rasterio.open(label_tif) as lab_ds:
        transform, width, height = lab_ds.transform, lab_ds.width, lab_ds.height
    for y, p in rasters.items():
        with rasterio.open(p) as ds:
            if ds.transform != transform or (ds.width, ds.height) != (width, height):
                raise ValueError(f"Raster {p.name} ({y}) is not on the label grid")

    row_area_ha = row_pixel_area_m2(transform, height) / 1e4
    windows = list(iter_windows(width, height, window))

    acc = np.zeros((len(years), n_labels * n_cls), dtype="float64")
    values_seen: dict[int, set] = {y: set() for y in years}
    start = 0
    if checkpoint is not None and checkpoint.exists():
        state = np.load(checkpoint, allow_pickle=False)
        if list(state["years"]) == years and int(state["window"]) == window:
            acc = state["acc"]
            start = int(state["next"])
            log.info("Resuming second-crop zonal statistics at window %d/%d", start, len(windows))

    t0 = time.time()
    lab_ds = rasterio.open(label_tif)
    year_ds = {y: rasterio.open(rasters[y]) for y in years}
    try:
        for k in range(start, len(windows)):
            win = windows[k]
            lab = lab_ds.read(1, window=win)
            if lab.any():
                r0 = int(win.row_off)
                w_area = np.broadcast_to(row_area_ha[r0:r0 + int(win.height), None], lab.shape)
                for yi, y in enumerate(years):
                    data = year_ds[y].read(1, window=win)
                    nz = data != 0
                    if not nz.any():
                        continue
                    vals = data[nz]
                    values_seen[y].update(np.unique(vals).tolist())
                    index = lab[nz].astype("int64") * n_cls + lut[vals]
                    acc[yi] += np.bincount(index, weights=w_area[nz], minlength=n_labels * n_cls)
            if checkpoint is not None and (k + 1) % 200 == 0:
                np.savez(checkpoint, acc=acc, next=k + 1, years=np.array(years), window=window)
                elapsed = time.time() - t0
                rate = (k + 1 - start) / max(elapsed, 1e-9)
                log.info("  second crop: window %d/%d, %.0f s elapsed, ~%.0f min left",
                         k + 1, len(windows), elapsed, (len(windows) - k - 1) / rate / 60)
    finally:
        lab_ds.close()
        for ds in year_ds.values():
            ds.close()

    acc = acc.reshape(len(years), n_labels, n_cls)
    label_to_geocode = dict(zip(lookup["label"], lookup["geocode"], strict=True))
    rows = []
    for yi, y in enumerate(years):
        for ci, cname in enumerate(class_names):
            col = acc[yi, :, ci]
            for lab_id in np.nonzero(col[1:])[0] + 1:
                rows.append((label_to_geocode[int(lab_id)], y, cname, col[lab_id]))
    table = pd.DataFrame(rows, columns=["geocode", "year", "class", "area_ha"])
    table = table.groupby(["geocode", "year", "class"], as_index=False)["area_ha"].sum()

    diagnostics = {
        "unassigned_ha": {int(y): {c: float(acc[yi, 0, ci]) for ci, c in enumerate(class_names)} for yi, y in enumerate(years)},
        "pixel_values_seen": {int(y): sorted(int(v) for v in values_seen[y]) for y in years},
        "window": window,
    }
    if checkpoint is not None and checkpoint.exists():
        checkpoint.unlink()
    return table, diagnostics


def read_national_series(path: Path) -> pd.DataFrame:
    """MapBiomas platform export: rows = class label, columns = years, values in ha."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    label_col = df.columns[0]
    long = df.melt(id_vars=label_col, var_name="year", value_name="area_ha").rename(columns={label_col: "label"})
    long["year"] = long["year"].astype(int)
    return long


def save_diagnostics(diag: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(diag, fh, indent=2)
