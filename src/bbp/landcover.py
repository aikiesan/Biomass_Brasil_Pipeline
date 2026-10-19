"""Land cover for the maps: MapBiomas coverage (30 m) aggregated to display groups.

The coverage rasters hold ~2 x 10^10 pixels for Brazil; a map at print size needs a few million.
`aggregate` reduces them in one windowed pass to a grid `factor` times coarser (16 -> ~480 m),
in which every cell takes the display group (`GROUPS`) that covers most of its 30 m pixels.
Counts are exact (a bincount of the 30 m pixels per cell), so the result depends only on the
rasters and the grouping below.

* The groups merge MapBiomas classes that share one colour on the maps (all natural formations,
  pasture with the mosaic of uses, ...), so a cell is a crop only where that crop covers more of it
  than any merged group: crops are never inflated by the aggregation.
* With the second-crop layer (2008–2024), soybean pixels followed by 2nd-season corn or cotton
  become their own groups; for the choice of a cell, the three soybean groups count as one
  (soybean), and the cell shows the most frequent of the three.
* Coarser grids (e.g. 32) come from the same counts (summed 2 x 2), not from a second pass.
* The area of every group (geodesic, WGS84) is accumulated on the way, so the display raster
  can be checked against the MapBiomas coverage statistics.

The pass runs in bands of source rows and is resumable: a sidecar JSON records the finished bands,
so a run interrupted (or stopped by `time_budget`) continues where it stopped.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("bbp")


@dataclass(frozen=True)
class Group:
    key: str
    label: str
    classes: tuple[int, ...]


# Display groups, in tie-break order (crops first). Code = position + 1; 0 = no data.
GROUPS: tuple[Group, ...] = (
    Group("soybean", "Soybean", (39,)),
    Group("soy_corn2", "Soybean + corn (2nd season)", ()),        # class 39 with corn in the second-crop layer
    Group("soy_cotton2", "Soybean + cotton (2nd season)", ()),    # class 39 with cotton in the second-crop layer
    Group("sugarcane", "Sugarcane", (20,)),
    Group("coffee", "Coffee", (46,)),
    Group("citrus", "Citrus", (47,)),
    Group("cotton", "Cotton", (62,)),
    Group("rice", "Rice", (40,)),
    Group("other_crops", "Other crops", (41, 48, 35)),
    Group("pasture", "Pasture and mosaic of uses", (15, 21)),
    Group("plantation", "Forest plantation", (9,)),
    Group("natural", "Natural vegetation", (3, 4, 5, 6, 7, 11, 12, 13, 23, 29, 32, 49, 50, 77, 84)),
    Group("water", "Water", (31, 33)),
    Group("built", "Urban and other non-vegetated", (24, 25, 30, 75, 91)),
)
CODE = {g.key: i + 1 for i, g in enumerate(GROUPS)}
NG = len(GROUPS)
UNMAPPED = NG + 1                    # counted, never chosen: a class the grouping does not know
NBINS = NG + 2                       # 0 (no data), the groups, unmapped
NO_DATA_CLASSES = (0, 27)            # 27 = not observed
SOYBEAN_CLASS = 39
SECOND_CORN, SECOND_COTTON = 1, 62   # pixel values of the MapBiomas second-crop rasters
SOY_FAMILY = (CODE["soybean"], CODE["soy_corn2"], CODE["soy_cotton2"])
VERSION = 1                          # bump when the grouping or the rule changes (invalidates caches)


def lookup_table() -> np.ndarray:
    """MapBiomas class (0..255) -> display code; classes no group names map to UNMAPPED."""
    lut = np.full(256, UNMAPPED, dtype=np.uint8)
    for c in NO_DATA_CLASSES:
        lut[c] = 0
    for g in GROUPS:
        for c in g.classes:
            if lut[c] not in (0, UNMAPPED):
                raise ValueError(f"MapBiomas class {c} is in two display groups")
            lut[c] = CODE[g.key]
    return lut


def decide(counts: np.ndarray) -> np.ndarray:
    """Display code of every cell from its pixel counts (..., NBINS): the most frequent group, with the
    soybean groups counted together; 0 where the cell has no mapped pixel."""
    c = counts.astype(np.int64, copy=False)
    groups = c[..., 1:NG + 1].copy()                     # columns = codes 1..NG
    soy = [k - 1 for k in SOY_FAMILY]
    family = groups[..., soy].sum(-1)
    groups[..., soy[0]] = family
    groups[..., soy[1:]] = -1
    best = groups.argmax(-1) + 1                         # ties -> lowest code (crops first)
    best[groups.max(-1) <= 0] = 0
    within = c[..., list(SOY_FAMILY)].argmax(-1)         # which soybean group, ties -> plain soybean
    is_soy = best == SOY_FAMILY[0]
    best[is_soy] = np.asarray(SOY_FAMILY)[within[is_soy]]
    return best.astype(np.uint8)


def coarsen(counts: np.ndarray, k: int) -> np.ndarray:
    """Sum (rows, cols, NBINS) counts in k x k blocks (edges padded with empty cells)."""
    if k == 1:
        return counts
    r, c, b = counts.shape
    pr, pc = -r % k, -c % k
    if pr or pc:
        counts = np.pad(counts, ((0, pr), (0, pc), (0, 0)))
    r2, c2 = counts.shape[0] // k, counts.shape[1] // k
    return counts.reshape(r2, k, c2, k, b).sum(axis=(1, 3))


# ---------------------------------------------------------------------------
# the source grid
# ---------------------------------------------------------------------------
@dataclass
class Mosaic:
    """One raster or Earth Engine tiles on one grid: origin, pixel size and each part's place in it."""

    x0: float
    y0: float
    res: float
    width: int
    height: int
    crs: str
    parts: list[tuple[Path, int, int, int, int]]          # (path, col_off, row_off, width, height)

    @classmethod
    def open(cls, sources: Path | list[Path]) -> Mosaic:
        import rasterio

        paths = [Path(p) for p in (sources if isinstance(sources, (list, tuple)) else [sources])]
        info = []
        for p in paths:
            with rasterio.open(p) as ds:
                info.append((p, ds.transform, ds.width, ds.height, ds.crs.to_string()))
        t0 = info[0][1]
        res = t0.a
        x0 = min(t.c for _, t, *_ in info)
        y0 = max(t.f for _, t, *_ in info)
        parts = []
        for p, t, w, h, crs in info:
            if crs != info[0][4] or abs(t.a - res) > 1e-15 or abs(t.e + res) > 1e-15:
                raise ValueError(f"{p.name} is not on the grid of {info[0][0].name}")
            col, row = (t.c - x0) / res, (y0 - t.f) / res
            if abs(col - round(col)) > 1e-6 or abs(row - round(row)) > 1e-6:
                raise ValueError(f"{p.name} is not pixel-aligned with the other parts")
            parts.append((p, int(round(col)), int(round(row)), w, h))
        width = max(c + w for _, c, _, w, _ in parts)
        height = max(r + h for _, _, r, _, h in parts)
        return cls(x0, y0, res, width, height, info[0][4], parts)

    def offset_of(self, path: Path) -> tuple[int, int, int, int]:
        """(col_off, row_off, width, height) of another raster on this grid (e.g. the second-crop layer)."""
        import rasterio

        with rasterio.open(path) as ds:
            t = ds.transform
            if abs(t.a - self.res) > 1e-15 or abs(t.e + self.res) > 1e-15:
                raise ValueError(f"{Path(path).name} does not have the pixel size of the coverage")
            col, row = (t.c - self.x0) / self.res, (self.y0 - t.f) / self.res
            if abs(col - round(col)) > 1e-6 or abs(row - round(row)) > 1e-6:
                raise ValueError(f"{Path(path).name} is not pixel-aligned with the coverage")
            return int(round(col)), int(round(row)), ds.width, ds.height


def window_cells(m: Mosaic, factor: int, bounds: tuple[float, float, float, float] | None):
    """Source pixel window (c0, r0, c1, r1) of the output grid: whole multiples of `factor` from the
    mosaic origin, clipped to the mosaic; bounds = (lon0, lat0, lon1, lat1)."""
    if bounds is None:
        return 0, 0, m.width, m.height
    lon0, lat0, lon1, lat1 = bounds
    c0 = max(0, math.floor((lon0 - m.x0) / m.res / factor) * factor)
    r0 = max(0, math.floor((m.y0 - lat1) / m.res / factor) * factor)
    c1 = min(m.width, math.ceil((lon1 - m.x0) / m.res / factor) * factor)
    r1 = min(m.height, math.ceil((m.y0 - lat0) / m.res / factor) * factor)
    if c1 <= c0 or r1 <= r0:
        raise ValueError(f"bounds {bounds} do not intersect the coverage")
    return c0, r0, c1, r1


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------
def _read(path: Path, col: int, row: int, w: int, h: int, off: tuple[int, int, int, int], ds_cache: dict):
    """Pixels [row:row+h, col:col+w] (mosaic coordinates) of a raster placed at `off`; 0 outside it."""
    from rasterio.windows import Window

    oc, orow, ow, oh = off
    out = np.zeros((h, w), dtype=np.uint8)
    a0, a1 = max(col, oc), min(col + w, oc + ow)
    b0, b1 = max(row, orow), min(row + h, orow + oh)
    if a1 <= a0 or b1 <= b0:
        return out
    ds = ds_cache.get(path)
    if ds is None:
        import rasterio

        ds = ds_cache[path] = rasterio.open(path)
    out[b0 - row:b1 - row, a0 - col:a1 - col] = ds.read(1, window=Window(a0 - oc, b0 - orow, a1 - a0, b1 - b0))
    return out


def chunk_counts(cov: np.ndarray, sc: np.ndarray | None, lut: np.ndarray, row0: int, col0: int, factor: int,
                 r_origin: int, c_origin: int, cache: dict | None = None):
    """Counts per output cell (rows, cols, NBINS) of a source chunk whose top-left pixel is (row0, col0);
    the output grid starts at (r_origin, c_origin). Returns (counts, first cell row, first cell col).
    `cache` keeps the cell-index array between chunks of the same shape and phase."""
    h, w = cov.shape
    codes = lut[cov]
    if sc is not None:
        soy = cov == SOYBEAN_CLASS
        codes[soy & (sc == SECOND_CORN)] = CODE["soy_corn2"]
        codes[soy & (sc == SECOND_COTTON)] = CODE["soy_cotton2"]
    pr, pc = (row0 - r_origin) % factor, (col0 - c_origin) % factor
    cr0, cc0 = (row0 - r_origin) // factor, (col0 - c_origin) // factor
    nr, nc = (pr + h - 1) // factor + 1, (pc + w - 1) // factor + 1
    key = (h, w, pr, pc, factor)
    cell = cache.get(key) if cache is not None else None
    if cell is None:
        rr = (np.arange(h) + pr) // factor
        cc = (np.arange(w) + pc) // factor
        cell = ((rr[:, None] * nc + cc[None, :]) * NBINS).astype(np.int32)
        if cache is not None:
            cache.clear()                       # one shape at a time: chunks repeat, edges are rare
            cache[key] = cell
    idx = cell + codes
    counts = np.bincount(idx.ravel(), minlength=nr * nc * NBINS).reshape(nr, nc, NBINS)
    return counts, cr0, cc0


def _profile(width: int, height: int, transform, crs: str) -> dict:
    return {"driver": "GTiff", "width": width, "height": height, "count": 1, "dtype": "uint8", "crs": crs,
            "transform": transform, "nodata": 0, "compress": "deflate", "predictor": 2, "tiled": True,
            "blockxsize": 256, "blockysize": 256, "SPARSE_OK": "TRUE"}


def _create(path: Path, prof: dict, descriptions: list[str] | None = None):
    """A new raster, or an existing one of the same shape and grid reused in place (its bands are rewritten
    band by band anyway): synced or shared folders may forbid deleting files."""
    import rasterio

    if path.exists():
        try:
            with rasterio.open(path) as ds:
                same = (ds.width, ds.height, ds.count, ds.dtypes[0], ds.transform) == (
                    prof["width"], prof["height"], prof["count"], prof["dtype"], prof["transform"])
        except rasterio.errors.RasterioIOError:
            same = False
        if same:
            return
    with rasterio.open(path, "w", **prof) as d:
        for i, name in enumerate(descriptions or [], start=1):
            d.set_band_description(i, name)


def output_paths(out_dir: Path, name: str, factors: tuple[int, ...]) -> dict[int, Path]:
    return {f: out_dir / f"{name}_f{f}.tif" for f in factors}


def counts_path(out_dir: Path, name: str, factor: int) -> Path:
    """Pixel counts per display group at the base factor: band b = group code b (uint16)."""
    return out_dir / f"{name}_f{factor}_counts.tif"


def read_counts(path: Path, groups: list[str] | None = None, window=None) -> tuple[np.ndarray, object]:
    """(counts [bands, rows, cols], transform) of a counts raster; `groups`: keys to read (default all)."""
    import rasterio

    with rasterio.open(path) as ds:
        idx = [CODE[g] for g in groups] if groups else list(range(1, ds.count + 1))
        arr = ds.read(idx, window=window)
        tr = ds.window_transform(window) if window is not None else ds.transform
    return arr, tr


def aggregate(coverage: Path | list[Path], out_dir: Path, name: str, factors: tuple[int, ...] = (16, 32),
              second_crop: Path | None = None, bounds: tuple[float, float, float, float] | None = None,
              band_rows: int | None = None, chunk_cols: int = 4096, time_budget: float | None = None,
              counts: bool = False) -> dict:
    """Aggregate a coverage raster (or its tiles) to display groups at every factor in `factors`.

    Writes <out_dir>/<name>_f<factor>.tif (uint8 display codes, EPSG of the source) and
    <name>.json (parameters, progress, area per group). Returns the JSON content; `complete` is False
    when `time_budget` (seconds) ran out first — call again to continue. `counts`: also write
    <name>_f<smallest factor>_counts.tif, the pixel count of every group per cell (one uint16 band per
    group), so areas within any polygon or radius stay exact at the cell size.
    """
    import rasterio
    from affine import Affine
    from rasterio.windows import Window

    from bbp.ingest.second_crop import row_pixel_area_m2

    factors = tuple(sorted(int(f) for f in factors))
    base = factors[0]
    if any(f % base for f in factors):
        raise ValueError(f"factors {factors} must be multiples of the smallest")
    top = factors[-1]
    band_rows = band_rows or max(4096, top)
    if band_rows % top:
        raise ValueError(f"band_rows ({band_rows}) must be a multiple of {top}")
    if chunk_cols % top:
        raise ValueError(f"chunk_cols ({chunk_cols}) must be a multiple of {top}")
    m = Mosaic.open(coverage)
    c0, r0, c1, r1 = window_cells(m, top, bounds)
    sc_off = m.offset_of(second_crop) if second_crop else None
    params = {"version": VERSION, "coverage": [p.name for p, *_ in m.parts],
              "second_crop": Path(second_crop).name if second_crop else None, "factors": list(factors),
              "window_px": [c0, r0, c1, r1], "band_rows": band_rows, "chunk_cols": chunk_cols,
              "groups": [g.key for g in GROUPS]}
    if counts:
        params["counts"] = True
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{name}.json"
    paths = output_paths(out_dir, name, factors)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    cpath = counts_path(out_dir, name, base) if counts else None
    fresh = (meta is None or meta.get("params") != params or not all(p.exists() for p in paths.values())
             or (cpath is not None and not cpath.exists()))
    if not fresh and meta.get("complete"):
        return meta
    if fresh:
        meta = {"params": params, "bands_done": 0, "complete": False, "pixels": [0] * NBINS,
                "area_ha": [0.0] * NBINS, "elapsed_s": 0.0}
        for f, p in paths.items():
            ncol, nrow = math.ceil((c1 - c0) / f), math.ceil((r1 - r0) / f)
            tr = Affine(m.res * f, 0, m.x0 + c0 * m.res, 0, -m.res * f, m.y0 - r0 * m.res)
            _create(p, _profile(ncol, nrow, tr, m.crs))
        if cpath is not None:
            ncol, nrow = math.ceil((c1 - c0) / base), math.ceil((r1 - r0) / base)
            tr = Affine(m.res * base, 0, m.x0 + c0 * m.res, 0, -m.res * base, m.y0 - r0 * m.res)
            prof = _profile(ncol, nrow, tr, m.crs) | {"count": NG, "dtype": "uint16", "interleave": "band"}
            _create(cpath, prof, [g.key for g in GROUPS])
    lut = lookup_table()
    n_bands = math.ceil((r1 - r0) / band_rows)
    # pixel area of each output row of the base grid (mean over its source rows; varies < 1e-4 within)
    full_tr = Affine(m.res, 0, m.x0, 0, -m.res, m.y0 - r0 * m.res)
    row_area = row_pixel_area_m2(full_tr, r1 - r0)
    nrow_base = math.ceil((r1 - r0) / base)
    cell_area_ha = np.array([row_area[i * base:(i + 1) * base].mean() for i in range(nrow_base)]) / 1e4
    ncol_base = math.ceil((c1 - c0) / base)
    t_start = time.time()
    ds_cache: dict = {}
    cell_cache: dict = {}
    dst = {f: rasterio.open(p, "r+") for f, p in paths.items()}
    cdst = rasterio.open(cpath, "r+") if cpath is not None else None
    try:
        for b in range(int(meta["bands_done"]), n_bands):
            if time_budget is not None and time.time() - t_start > time_budget:
                break
            ra = r0 + b * band_rows
            rb = min(ra + band_rows, r1)
            br0 = (ra - r0) // base
            counts = np.zeros((math.ceil((rb - ra) / base), ncol_base, NBINS), dtype=np.int64)
            for path, oc, orow, ow, oh in m.parts:
                if orow >= rb or orow + oh <= ra or oc >= c1 or oc + ow <= c0:
                    continue
                pa, pb = max(c0, oc), min(c1, oc + ow)
                for ca in range(pa - (pa - c0) % chunk_cols, pb, chunk_cols):
                    ca_, cb_ = max(ca, pa), min(ca + chunk_cols, pb)
                    cov = _read(path, ca_, ra, cb_ - ca_, rb - ra, (oc, orow, ow, oh), ds_cache)
                    if not cov.any():
                        continue
                    sc = (_read(second_crop, ca_, ra, cb_ - ca_, rb - ra, sc_off, ds_cache)
                          if second_crop else None)
                    cnt, cr, cc = chunk_counts(cov, sc, lut, ra, ca_, base, r0, c0, cell_cache)
                    counts[cr - br0:cr - br0 + cnt.shape[0], cc:cc + cnt.shape[1]] += cnt
            meta["pixels"] = (np.asarray(meta["pixels"], dtype=np.int64) + counts.sum(axis=(0, 1))).tolist()
            area = (counts.sum(axis=1) * cell_area_ha[br0:br0 + counts.shape[0], None]).sum(axis=0)
            meta["area_ha"] = (np.asarray(meta["area_ha"]) + area).tolist()
            for f in factors:
                grid = decide(coarsen(counts, f // base))
                ncol_f = dst[f].width
                row_f = (ra - r0) // f
                dst[f].write(grid[:, :ncol_f][None], window=Window(0, row_f, ncol_f, grid.shape[0]))
            if cdst is not None:
                cw = cdst.width
                block = np.moveaxis(counts[:, :cw, 1:NG + 1], 2, 0).astype(np.uint16)
                cdst.write(block, window=Window(0, br0, cw, counts.shape[0]))
            meta["bands_done"] = b + 1
            meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
            log.info("  land cover %s: band %d/%d (%.0f s)", name, b + 1, n_bands, time.time() - t_start)
    finally:
        for d in dst.values():
            d.close()
        if cdst is not None:
            cdst.close()
        for d in ds_cache.values():
            d.close()
    meta["elapsed_s"] = float(meta.get("elapsed_s", 0.0)) + (time.time() - t_start)
    meta["complete"] = int(meta["bands_done"]) >= n_bands
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def area_by_group(meta: dict) -> dict[str, float]:
    """{group key: ha} from an `aggregate` sidecar, plus 'unmapped'."""
    a = meta["area_ha"]
    out = {g.key: float(a[CODE[g.key]]) for g in GROUPS}
    out["unmapped"] = float(a[UNMAPPED])
    return out


def class_area(meta: dict, mapbiomas_class: int) -> float:
    """Area (ha) of one MapBiomas class in the aggregated raster: the groups it went to."""
    if mapbiomas_class == SOYBEAN_CLASS:
        return sum(meta["area_ha"][k] for k in SOY_FAMILY)
    for g in GROUPS:
        if mapbiomas_class in g.classes and len(g.classes) == 1:
            return float(meta["area_ha"][CODE[g.key]])
    raise ValueError(f"class {mapbiomas_class} shares its group with other classes")
