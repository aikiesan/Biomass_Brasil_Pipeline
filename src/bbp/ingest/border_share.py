"""Crop area near municipal borders, from the MapBiomas coverage rasters (H2 predictor).

A municipality whose crops lie close to its boundary is more exposed to two
errors that make the sources disagree: MapBiomas pixels assigned by the mesh to
the neighbour, and PAM informants reporting a field on the other side of the
line. For every municipality, year and crop this module measures the share of
the observed crop area that lies within 1 km and 2 km (configurable) of a
boundary with ANOTHER municipality (coastline and the national border do not
count: there is no neighbour to confuse it with).

Method, windowed so that the ~167 000 x 145 000 national grid never has to fit
in memory:

1. The IBGE mesh is rasterised once onto the coverage raster grid (pixel-centre
   rule; `second_crop.build_label_grid`).
2. Each window is read with a halo wider than the largest distance. Boundary
   pixels are those with a 4-neighbour of a different municipality; the
   distance of every pixel to the nearest boundary pixel is the Euclidean
   distance transform in metres (pixel size from the window's central latitude).
3. The geodesic area of each crop pixel of the window interior is added to its
   municipality and distance bin. Pixel area depends only on the row.

Progress is checkpointed, so a run can be resumed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bbp.ingest.second_crop import iter_windows, row_pixel_area_m2

log = logging.getLogger("bbp")

ALL = "all"
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    """Pixels with a 4-neighbour inside a DIFFERENT municipality (label 0 = no municipality, ignored)."""
    b = np.zeros(labels.shape, dtype=bool)
    for a, c in ((labels[1:, :], labels[:-1, :]), (labels[:, 1:], labels[:, :-1])):
        diff = (a != c) & (a > 0) & (c > 0)
        if a.shape[0] != labels.shape[0]:   # vertical neighbours
            b[1:, :] |= diff
            b[:-1, :] |= diff
        else:                               # horizontal neighbours
            b[:, 1:] |= diff
            b[:, :-1] |= diff
    return b


def distance_to_boundary_m(labels: np.ndarray, lat_deg: float, res_deg: float, cap_m: float) -> np.ndarray:
    """Distance (m) of every pixel to the nearest boundary pixel; `cap_m + 1` where there is none."""
    from scipy.ndimage import distance_transform_edt

    b = boundary_mask(labels)
    if not b.any():
        return np.full(labels.shape, cap_m + 1.0, dtype="float32")
    dy = res_deg * M_PER_DEG_LAT
    dx = res_deg * M_PER_DEG_LON_EQ * np.cos(np.deg2rad(lat_deg))
    return distance_transform_edt(~b, sampling=(dy, dx)).astype("float32")


def border_area(
    label_tif: Path,
    rasters: dict[int, Path],
    classes: dict[int, str],
    distances_m: list[float],
    window: int = 2048,
    checkpoint: Path | None = None,
) -> pd.DataFrame:
    """Area (ha) per municipality, year, crop and distance bin.

    Returns geocode, year, crop (the configured crop names and "all"), crop_area_ha and, for every
    distance d, within_<d>m_ha = crop area at most d metres from a boundary with another municipality.
    """
    import rasterio
    from rasterio.windows import Window

    distances_m = sorted(float(d) for d in distances_m)
    lookup = pd.read_csv(label_tif.with_suffix(".lookup.csv"))
    n_labels = int(lookup["label"].max()) + 1
    crops = list(dict.fromkeys(classes.values()))
    groups = [*crops, ALL]
    lut = np.full(256, -1, dtype="int64")
    for value, name in classes.items():
        lut[int(value)] = crops.index(name)
    n_bins = len(distances_m) + 1          # bin k: distance <= distances_m[k]; last bin: farther
    years = sorted(rasters)

    with rasterio.open(label_tif) as lab_ds:
        transform, width, height = lab_ds.transform, lab_ds.width, lab_ds.height
    for y, p in rasters.items():
        with rasterio.open(p) as ds:
            if ds.transform != transform or (ds.width, ds.height) != (width, height):
                raise ValueError(f"Raster {p.name} ({y}) is not on the label grid")
    res = abs(transform.a)
    halo = int(np.ceil(distances_m[-1] / (res * M_PER_DEG_LAT * np.cos(np.deg2rad(35))))) + 2  # worst case ~35° S
    row_area_ha = row_pixel_area_m2(transform, height) / 1e4
    windows = list(iter_windows(width, height, window))

    shape = (len(years), n_labels, len(groups), n_bins)
    acc = np.zeros(shape, dtype="float64")
    start = 0
    if checkpoint is not None and checkpoint.exists():
        state = np.load(checkpoint, allow_pickle=False)
        if list(state["years"]) == years and int(state["window"]) == window and state["acc"].shape == shape:
            acc, start = state["acc"], int(state["next"])
            log.info("Resuming border share at window %d/%d", start, len(windows))

    t0 = time.time()
    lab_ds = rasterio.open(label_tif)
    year_ds = {y: rasterio.open(rasters[y]) for y in years}
    try:
        for k in range(start, len(windows)):
            win = windows[k]
            c0, r0 = max(int(win.col_off) - halo, 0), max(int(win.row_off) - halo, 0)
            c1 = min(int(win.col_off + win.width) + halo, width)
            r1 = min(int(win.row_off + win.height) + halo, height)
            big = Window(c0, r0, c1 - c0, r1 - r0)
            lab_big = lab_ds.read(1, window=big)
            inner = (slice(int(win.row_off) - r0, int(win.row_off) - r0 + int(win.height)),
                     slice(int(win.col_off) - c0, int(win.col_off) - c0 + int(win.width)))
            lab = lab_big[inner]
            if not lab.any():
                continue
            lat_mid = transform.f + transform.e * (int(win.row_off) + int(win.height) / 2)
            dist = distance_to_boundary_m(lab_big, lat_mid, res, distances_m[-1])[inner]
            dbin = np.searchsorted(np.asarray(distances_m, dtype="float32"), dist, side="left")
            w_area = np.broadcast_to(row_area_ha[int(win.row_off):int(win.row_off) + int(win.height), None], lab.shape)
            for yi, y in enumerate(years):
                data = year_ds[y].read(1, window=win)
                crop = lut[data]
                sel = (crop >= 0) & (lab > 0)
                if not sel.any():
                    continue
                labs, bins, w = lab[sel].astype("int64"), dbin[sel], w_area[sel]
                for gi in range(len(crops)):
                    m = crop[sel] == gi
                    if m.any():
                        acc[yi, :, gi, :] += np.bincount(labs[m] * n_bins + bins[m], weights=w[m],
                                                         minlength=n_labels * n_bins).reshape(n_labels, n_bins)
                acc[yi, :, len(crops), :] += np.bincount(labs * n_bins + bins, weights=w,
                                                         minlength=n_labels * n_bins).reshape(n_labels, n_bins)
            if checkpoint is not None and (k + 1) % 200 == 0:
                np.savez(checkpoint, acc=acc, next=k + 1, years=np.array(years), window=window)
                elapsed = time.time() - t0
                rate = (k + 1 - start) / max(elapsed, 1e-9)
                log.info("  border share: window %d/%d, %.0f s elapsed, ~%.0f min left",
                         k + 1, len(windows), elapsed, (len(windows) - k - 1) / rate / 60)
    finally:
        lab_ds.close()
        for ds in year_ds.values():
            ds.close()

    label_to_geocode = dict(zip(lookup["label"], lookup["geocode"], strict=True))
    rows = []
    for yi, y in enumerate(years):
        for gi, g in enumerate(groups):
            block = acc[yi, :, gi, :]
            for lab_id in np.nonzero(block[1:].sum(axis=1))[0] + 1:
                cum = np.cumsum(block[lab_id])
                rows.append((label_to_geocode[int(lab_id)], y, g, cum[-1], *cum[:-1]))
    cols = ["geocode", "year", "crop", "crop_area_ha", *[f"within_{int(d)}m_ha" for d in distances_m]]
    table = pd.DataFrame(rows, columns=cols).groupby(["geocode", "year", "crop"], as_index=False).sum()
    for d in distances_m:
        km = f"{d / 1000:g}km"
        table[f"border_share_{km}"] = table[f"within_{int(d)}m_ha"] / table["crop_area_ha"]
    if checkpoint is not None and checkpoint.exists():
        checkpoint.unlink()
    return table


def find_year_sources(folder: Path, pattern: str, years: list[int]) -> dict[int, Path | list[Path]]:
    """{year: one GeoTIFF | its tiles}. A year is either one file matching `pattern`
    ("brazil_coverage-col11_{year}.tif") or the tiles Earth Engine writes when an export is large
    ("brazil_coverage-col11_{year}-0000065536-0000131072.tif")."""
    out: dict[int, Path | list[Path]] = {}
    for y in years:
        name = pattern.format(year=y)
        single = folder / name
        tiles = sorted(folder.glob(f"{Path(name).stem}-*{Path(name).suffix}"))
        if single.exists() and tiles:
            raise ValueError(f"{y}: both {single.name} and {len(tiles)} tiles in {folder} — keep one")
        if single.exists():
            out[y] = single
        elif tiles:
            out[y] = tiles
    return out


def build_vrt(tiles: list[Path], out_vrt: Path) -> dict:
    """Mosaic Earth Engine tiles into one VRT (a few KB of XML; the tiles stay where they are).

    Every tile must be on the same grid (same CRS, pixel size, data type, pixel-aligned origins),
    and together they must cover the mosaic once, without gaps or overlaps. Returns
    {"width", "height", "tiles"}.
    """
    from xml.sax.saxutils import escape

    import rasterio

    info = []
    for p in tiles:
        with rasterio.open(p) as ds:
            info.append((p, ds.transform, ds.width, ds.height, ds.crs, ds.dtypes[0], ds.block_shapes[0]))
    t0, crs0, dt0 = info[0][1], info[0][4], info[0][5]
    res = t0.a
    x0 = min(t.c for _, t, *_ in info)
    y0 = max(t.f for _, t, *_ in info)
    placed = []
    for p, t, w, h, crs, dt, _ in info:
        if crs != crs0 or dt != dt0 or abs(t.a - res) > 1e-15 or abs(t.e + res) > 1e-15:
            raise ValueError(f"tile {p.name} is not on the grid of {info[0][0].name}")
        col, row = (t.c - x0) / res, (y0 - t.f) / res
        if abs(col - round(col)) > 1e-6 or abs(row - round(row)) > 1e-6:
            raise ValueError(f"tile {p.name} is not pixel-aligned with the others")
        placed.append((p, int(round(col)), int(round(row)), w, h))
    width = max(c + w for _, c, _, w, _ in placed)
    height = max(r + h for _, _, r, _, h in placed)
    covered = sum(w * h for *_, w, h in placed)
    if covered != width * height:
        raise ValueError(f"tiles cover {covered:,} pixels of a {width:,} x {height:,} mosaic: gap or overlap")
    gdal_type = {"uint8": "Byte", "uint16": "UInt16", "int16": "Int16"}[dt0]
    sources = []
    for p, c, r, w, h in placed:
        bx, by = info[[i[0] for i in info].index(p)][6][::-1]
        sources.append(
            f'    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{escape(str(p))}</SourceFilename>\n'
            f"      <SourceBand>1</SourceBand>\n"
            f'      <SourceProperties RasterXSize="{w}" RasterYSize="{h}" DataType="{gdal_type}" '
            f'BlockXSize="{bx}" BlockYSize="{by}"/>\n'
            f'      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>\n'
            f'      <DstRect xOff="{c}" yOff="{r}" xSize="{w}" ySize="{h}"/>\n    </SimpleSource>')
    xml = (f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">\n'
           f"  <SRS>{escape(crs0.to_wkt())}</SRS>\n"
           f"  <GeoTransform>{x0!r}, {res!r}, 0.0, {y0!r}, 0.0, {-res!r}</GeoTransform>\n"
           f'  <VRTRasterBand dataType="{gdal_type}" band="1">\n' + "\n".join(sources) +
           "\n  </VRTRasterBand>\n</VRTDataset>\n")
    out_vrt.parent.mkdir(parents=True, exist_ok=True)
    out_vrt.write_text(xml, encoding="utf-8")
    return {"width": width, "height": height, "tiles": len(placed)}


def grid_key(path: Path) -> str:
    """A short name for a raster's pixel grid: rasters exported in different years are not always
    on the same grid (e.g. 154 470 x 146 501 vs 154 469 x 146 480), so the label grid is built per grid."""
    import hashlib

    import rasterio

    with rasterio.open(path) as ds:
        sig = f"{ds.crs}|{tuple(round(v, 12) for v in ds.transform[:6])}|{ds.width}x{ds.height}"
    return hashlib.sha1(sig.encode()).hexdigest()[:10]


def collection_agreement(table: pd.DataFrame, coverage: pd.DataFrame, classes: dict[int, str]) -> pd.DataFrame:
    """Raster crop area vs the municipal statistics, per municipality, year and crop.

    The MapBiomas statistics are computed from the same rasters, so rasters of the configured
    collection reproduce them to a fraction of a percent; rasters of another collection differ by
    1–3 % per municipality. This is the guard against mixing collections.
    """
    by_crop = {}
    for cls, crop in classes.items():
        by_crop.setdefault(crop, []).append(int(cls))
    stats = []
    for crop, cls in by_crop.items():
        s = coverage[coverage["class_id"].isin(cls)].groupby(["geocode", "year"], as_index=False)["area_ha"].sum()
        stats.append(s.assign(crop=crop))
    stats = pd.concat(stats, ignore_index=True).rename(columns={"area_ha": "stats_area_ha"})
    j = table[table["crop"] != ALL].merge(stats, on=["geocode", "year", "crop"], how="inner")
    return j[j["stats_area_ha"] >= 1000].assign(rel=lambda d: d["crop_area_ha"] / d["stats_area_ha"] - 1)
