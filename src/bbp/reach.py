"""Crop area within reach of existing processing plants, from the 30 m pixels (`bbp reach`).

Inputs: the MapBiomas coverage raster of `plants.reach.year` (Collection 11, 30 m; Earth Engine tiles are read
through the VRT of `bbp border-share`), the second-crop raster of the same year, the municipality label grid on
the coverage grid (pixel-centre rule, built once) and the plant register (`bbp run plants`).

One windowed pass over the label grid. For every crop pixel (classes in `plants.reach.crops`; a pixel in both
the coverage and the second-crop layer counts once) and every pair (crop, plant family) in `plants.reach.pairs`:

* the straight-line distance to the nearest operating plant of the family: great circle on the mean Earth
  sphere (R = 6371.0088 km; within 0.6 % of the WGS84 geodesic, i.e. < 200 m at 30 km, checked; a
  planar distance in SIRGAS 2000 / Brazil Polyconic would be off by up to 5.5 % at the edges of the country) — buffers are geodesic circles, not a
  road network, so the result does not depend on a projection;
* the number of plants of the family within `competition_km`.

Pixel areas are geodesic (WGS84, per row). Everything is accumulated per municipality in 1 km distance bins
(`reach_hist`), so the area within any radius, at any level (municipality, intermediate region, state, nation),
is a sum of whole bins, and every level adds up to the same national total. Around each sugarcane mill the cane
area is accumulated in 1 km rings up to `ring_max_km` (required radius, sufficiency, contested cane).

The same distances are also measured from the centre of each pixel's `check_factor` cell (16 -> ~480 m), to
report how much aggregating the land cover would change the result. Aggregated pixels are written to
reach_cells_f<factor>.tif (crop area and area within `competition_km`, per crop and pair).

The pass is resumable: accumulators are checkpointed every few windows, so an interrupted run (or one stopped
by `--budget`) continues where it stopped.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("bbp")

R_EARTH_KM = 6371.0088
NAT_BINS = 2000                     # national distance curves: 1 km bins to 2000 km
VERSION = 1                         # bump when the accumulation changes (invalidates checkpoints)
LEVELS = ("municipality", "rgint", "uf", "nation")


# ---------------------------------------------------------------------------
# distances on the sphere
# ---------------------------------------------------------------------------
def haversine_km(lon0, lat0, lon, lat):
    p0, p = np.radians(lat0), np.radians(lat)
    a = np.sin((p - p0) / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin(np.radians(np.asarray(lon) - lon0) / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def unit_xyz(lon, lat) -> np.ndarray:
    lo, la = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def chord_to_km(ch):
    return 2 * R_EARTH_KM * np.arcsin(np.clip(np.asarray(ch) / 2, 0, 1))


def km_to_chord(km):
    return 2 * np.sin(np.asarray(km, float) / (2 * R_EARTH_KM))


def nearest_km(tree, lon, lat, k: int = 1, upper_km: float | None = None, chunk: int = 2_000_000):
    """Distances (km) from points to their k nearest tree points (inf where none within upper_km)."""
    out_d = np.empty((len(lon), k), dtype=np.float32)
    ub = float(km_to_chord(upper_km)) if upper_km else np.inf
    for a in range(0, len(lon), chunk):
        d, _ = tree.query(unit_xyz(lon[a:a + chunk], lat[a:a + chunk]), k=k, distance_upper_bound=ub)
        d = np.asarray(d).reshape(-1, k)
        out_d[a:a + chunk] = np.where(np.isfinite(d), chord_to_km(np.where(np.isfinite(d), d, 0)), np.inf)
    return out_d


def required_radius(cum_cane: np.ndarray, need_ha: np.ndarray, ring_km: float) -> np.ndarray:
    """Smallest radius (km) whose cumulative cane area reaches need_ha; inf beyond the rings."""
    ok = cum_cane >= need_ha[:, None]
    first = np.where(ok.any(axis=1), ok.argmax(axis=1), -1)
    return np.where(first >= 0, (first + 1) * ring_km, np.inf)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class Spec:
    crops: dict[str, tuple[tuple[int, ...], tuple[int, ...]]]       # name: (coverage classes, second-crop values)
    pairs: list[tuple[str, str]]                                     # (crop, family)
    radii_km: list[float] = field(default_factory=lambda: [10, 20, 30, 50, 75, 100])
    bin_km: float = 1.0
    max_km: float = 150.0
    ring_max_km: float = 100.0
    competition_km: float = 30.0
    competition_cap: int = 10
    cane_yield_t_ha: float = 75.0
    window: int = 4096
    check_factor: int = 16
    mill_family: str = "sugarcane_mill"
    cane_crop: str = "sugarcane"

    @property
    def n_bins(self) -> int:            # 1 km bins below max_km + one overflow bin (>= max_km)
        return int(round(self.max_km / self.bin_km)) + 1

    @property
    def n_rings(self) -> int:
        return int(round(self.ring_max_km / self.bin_km))

    @property
    def tags(self) -> list[str]:
        return [pair_tag(c, f) for c, f in self.pairs]

    def signature(self) -> dict:
        return {"version": VERSION, "crops": {k: [list(a), list(b)] for k, (a, b) in self.crops.items()},
                "pairs": [list(p) for p in self.pairs], "bin_km": self.bin_km, "max_km": self.max_km,
                "ring_max_km": self.ring_max_km, "competition_km": self.competition_km,
                "competition_cap": self.competition_cap, "window": self.window, "check_factor": self.check_factor}

    @classmethod
    def from_config(cls, cfg) -> Spec:
        r = (cfg.section("plants") or {}).get("reach", {})
        crops = {k: (tuple(int(c) for c in v.get("coverage", [])), tuple(int(c) for c in v.get("second_crop", [])))
                 for k, v in (r.get("crops") or {}).items()}
        pairs = [(p["crop"], p["family"]) for p in (r.get("pairs") or [])]
        kw = {k: r[k] for k in ("radii_km", "bin_km", "max_km", "ring_max_km", "competition_km", "competition_cap",
                                "cane_yield_t_ha", "window", "check_factor") if k in r}
        return cls(crops=crops, pairs=pairs, **kw)


def pair_tag(crop: str, family: str) -> str:
    return f"{crop}|{family}"


def crop_masks(cov: np.ndarray, sc: np.ndarray | None, crops: dict) -> dict[str, np.ndarray]:
    """Boolean mask of every crop: coverage classes OR second-crop values (a pixel counts once)."""
    out = {}
    for k, (cc, ss) in crops.items():
        m = np.isin(cov, cc) if cc else np.zeros(cov.shape, bool)
        if ss and sc is not None:
            m |= np.isin(sc, ss)
        out[k] = m
    return out


# ---------------------------------------------------------------------------
# the accumulator
# ---------------------------------------------------------------------------
class Acc:
    """Everything the pass adds up; saved to and restored from one .npz."""

    def __init__(self, spec: Spec, n_labels: int, n_mills: int):
        p, nb = len(spec.pairs), spec.n_bins
        self.hist = np.zeros((p, n_labels, nb))                   # area by municipality and distance bin
        self.dsum = np.zeros((p, n_labels))                       # sum of area x distance (mean distance)
        self.nat = np.zeros((p, NAT_BINS))                        # national curve, 1 km bins
        self.comp = np.zeros((p, n_labels, spec.competition_cap + 1))   # area by plants within competition_km
        self.check = np.zeros((p, nb))                            # national histogram from cell centres
        self.crop_area = np.zeros((len(spec.crops), n_labels))    # area of each crop per municipality
        self.unassigned = np.zeros(len(spec.crops))               # crop area outside the municipal mesh
        self.rings = np.zeros((n_mills, spec.n_rings))            # cane area in 1 km rings around each mill
        self.contested = np.zeros((n_mills, spec.n_rings))        # ... of it with >= 2 mills within competition_km
        self.windows_done = 0

    def save(self, path: Path, signature: dict) -> None:
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **{k: v for k, v in vars(self).items() if isinstance(v, np.ndarray)},
                 windows_done=self.windows_done, signature=json.dumps(signature, sort_keys=True))
        tmp.replace(path)

    def load(self, path: Path, signature: dict) -> bool:
        if not path.exists():
            return False
        z = np.load(path, allow_pickle=False)
        if str(z["signature"]) != json.dumps(signature, sort_keys=True):
            log.info("reach: checkpoint from another configuration, starting again")
            return False
        for k, v in vars(self).items():
            if isinstance(v, np.ndarray):
                if z[k].shape != v.shape:
                    return False
                setattr(self, k, z[k])
        self.windows_done = int(z["windows_done"])
        return True


@dataclass
class Plants:
    """Per family: the KD tree of the operating plants (unit vectors) and their count."""
    trees: dict
    counts: dict
    mills_xyz: np.ndarray            # every sugarcane mill (rings), operating or not
    mills_op_tree: object | None     # operating mills (contested cane)


def plant_trees(plants: pd.DataFrame, spec: Spec) -> Plants:
    from scipy.spatial import cKDTree

    from bbp.plants import select_plants

    trees, counts = {}, {}
    for fam in dict.fromkeys(f for _, f in spec.pairs):
        sel = select_plants(plants, fam)
        counts[fam] = len(sel)
        trees[fam] = cKDTree(unit_xyz(sel["lon"], sel["lat"])) if len(sel) else None
    mills = plants[plants["family"].astype(str) == spec.mill_family]
    op = select_plants(plants, spec.mill_family)
    return Plants(trees, counts, unit_xyz(mills["lon"], mills["lat"]) if len(mills) else np.zeros((0, 3)),
                  cKDTree(unit_xyz(op["lon"], op["lat"])) if len(op) else None)


# ---------------------------------------------------------------------------
# one window
# ---------------------------------------------------------------------------
def process_window(acc: Acc, spec: Spec, P: Plants, lab: np.ndarray, cov: np.ndarray, sc: np.ndarray | None,
                   row0: int, col0: int, transform, row_area_ha: np.ndarray) -> np.ndarray | None:
    """Add one window (top-left pixel row0, col0 of the label grid) to `acc`. Returns the aggregated cells
    [bands, rows, cols] (crop areas, then area within competition_km per pair) or None if the window is empty."""
    h, w = lab.shape
    masks = crop_masks(cov, sc, spec.crops)
    crop_keys = list(spec.crops)
    union = np.zeros_like(lab, dtype=bool)
    for m in masks.values():
        union |= m
    if not union.any():
        return None
    out_lab = union & (lab == 0)
    for ci, k in enumerate(crop_keys):
        o = masks[k] & out_lab
        if o.any():
            acc.unassigned[ci] += float(row_area_ha[row0 + np.nonzero(o)[0]].sum())
    r, c = np.nonzero(union & (lab > 0))
    if len(r) == 0:
        return None
    labels = lab[r, c].astype(np.int64)
    area = row_area_ha[row0 + r]
    res = transform.a
    lat = transform.f - (row0 + r + 0.5) * res
    lon = transform.c + (col0 + c + 0.5) * res
    xyz = unit_xyz(lon, lat)
    # the cell (check_factor) of every pixel, and its centre
    f = spec.check_factor
    cr, cc = (row0 + r) // f, (col0 + c) // f
    nl = acc.hist.shape[1]
    nb = spec.n_bins
    fr0, fc0 = row0 // f, col0 // f
    ncr, ncc = math.ceil(h / f), math.ceil(w / f)
    cell_idx = (cr - fr0) * ncc + (cc - fc0)
    cells = np.zeros((len(crop_keys) + len(spec.pairs), ncr, ncc), dtype=np.float32)
    in_crop = {}
    for ci, k in enumerate(crop_keys):
        s = masks[k][r, c]
        in_crop[k] = s
        acc.crop_area[ci] += np.bincount(labels[s], weights=area[s], minlength=nl)
        cells[ci] += np.bincount(cell_idx[s], weights=area[s], minlength=ncr * ncc).reshape(ncr, ncc)
    uniq, inv = np.unique(cell_idx, return_inverse=True)
    ucr, ucc = uniq // ncc + fr0, uniq % ncc + fc0
    cxyz = unit_xyz(transform.c + (ucc + 0.5) * f * res, transform.f - (ucr + 0.5) * f * res)
    comp_chord = float(km_to_chord(spec.competition_km))
    cap = spec.competition_cap
    n_mill_px = None
    for pi, (crop, fam) in enumerate(spec.pairs):
        tree = P.trees.get(fam)
        s = in_crop[crop]
        if tree is None or not s.any():
            continue
        x, lb, a = xyz[s], labels[s], area[s]
        d = chord_to_km(tree.query(x, workers=-1)[0])
        b = np.minimum((d / spec.bin_km).astype(np.int64), nb - 1)
        acc.hist[pi] += np.bincount(lb * nb + b, weights=a, minlength=nl * nb).reshape(nl, nb)
        acc.dsum[pi] += np.bincount(lb, weights=a * d, minlength=nl)
        acc.nat[pi] += np.bincount(np.minimum(d.astype(np.int64), NAT_BINS - 1), weights=a, minlength=NAT_BINS)
        n = np.minimum(np.asarray(tree.query_ball_point(x, comp_chord, return_length=True, workers=-1)), cap)
        acc.comp[pi] += np.bincount(lb * (cap + 1) + n, weights=a, minlength=nl * (cap + 1)).reshape(nl, cap + 1)
        if fam == spec.mill_family and crop == spec.cane_crop:
            n_mill_px = n
        # from the cell centres: the aggregation check
        dc_u = chord_to_km(tree.query(cxyz, workers=-1)[0])
        dc = dc_u[inv[s]]
        bc = np.minimum((dc / spec.bin_km).astype(np.int64), nb - 1)
        acc.check[pi] += np.bincount(bc, weights=a, minlength=nb)
        within = d < spec.competition_km
        cells[len(crop_keys) + pi] += np.bincount(cell_idx[s][within], weights=a[within],
                                                  minlength=ncr * ncc).reshape(ncr, ncc)
    if len(P.mills_xyz) and spec.cane_crop in in_crop and in_crop[spec.cane_crop].any():
        _mill_rings(acc, spec, P, xyz[in_crop[spec.cane_crop]], area[in_crop[spec.cane_crop]], n_mill_px,
                    transform, row0, col0, h, w)
    return cells


def _mill_rings(acc: Acc, spec: Spec, P: Plants, x: np.ndarray, a: np.ndarray, n_mills_px, transform,
                row0: int, col0: int, h: int, w: int, chunk: int = 8):
    """Cane area in 1 km rings around every mill whose ring_max_km disc can reach this window."""
    res = transform.a
    lat_c = transform.f - (row0 + h / 2) * res
    lon_c = transform.c + (col0 + w / 2) * res
    half_diag = 0.5 * math.hypot(h * res * 111.32, w * res * 111.32 * max(math.cos(math.radians(lat_c)), 0.05))
    centre = unit_xyz([lon_c], [lat_c])[0]
    reach_ch = float(km_to_chord(spec.ring_max_km + half_diag + 1.0))
    near = np.nonzero(np.linalg.norm(P.mills_xyz - centre, axis=1) <= reach_ch)[0]
    if len(near) == 0:
        return
    nr = spec.n_rings
    contested = (n_mills_px >= 2) if n_mills_px is not None else np.zeros(len(a), bool)
    for k0 in range(0, len(near), chunk):
        idx = near[k0:k0 + chunk]
        dot = x @ P.mills_xyz[idx].T                                  # cos of the central angle
        km = R_EARTH_KM * np.arccos(np.clip(dot, -1.0, 1.0))
        ring = (km / spec.bin_km).astype(np.int64)
        ok = ring < nr
        if not ok.any():
            continue
        px, mk = np.nonzero(ok)
        flat = idx[mk] * nr + ring[px, mk]
        size = acc.rings.size
        acc.rings += np.bincount(flat, weights=a[px], minlength=size).reshape(acc.rings.shape)
        cmask = contested[px]
        if cmask.any():
            acc.contested += np.bincount(flat[cmask], weights=a[px][cmask], minlength=size).reshape(acc.rings.shape)


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------
def _second_crop_offset(label_transform, sc_path: Path) -> tuple[int, int, int, int]:
    import rasterio

    with rasterio.open(sc_path) as ds:
        t = ds.transform
        if abs(t.a - label_transform.a) > 1e-15 or abs(t.e - label_transform.e) > 1e-15:
            raise ValueError(f"{sc_path.name} does not have the pixel size of the coverage")
        col, row = (t.c - label_transform.c) / t.a, (label_transform.f - t.f) / t.a
        if abs(col - round(col)) > 1e-6 or abs(row - round(row)) > 1e-6:
            raise ValueError(f"{sc_path.name} is not pixel-aligned with the coverage")
        return int(round(col)), int(round(row)), ds.width, ds.height


def accumulate(label_tif: Path, coverage: Path, second_crop: Path | None, plants: pd.DataFrame, spec: Spec,
               checkpoint: Path, cells_tif: Path | None = None, budget: float | None = None,
               save_every: int = 10, stop_after: int | None = None) -> tuple[Acc, dict]:
    """The windowed pass. Returns (accumulator, meta) with meta['complete'] False when `budget` (seconds) ran
    out first, or after `stop_after` windows in this call."""
    import rasterio
    from affine import Affine
    from rasterio.windows import Window

    from bbp.ingest.second_crop import iter_windows, row_pixel_area_m2
    from bbp.landcover import _read

    lookup = pd.read_csv(label_tif.with_suffix(".lookup.csv"))
    n_labels = int(lookup["label"].max()) + 1
    with rasterio.open(label_tif) as ds:
        tr, width, height = ds.transform, ds.width, ds.height
    with rasterio.open(coverage) as ds:
        if ds.transform != tr or (ds.width, ds.height) != (width, height):
            raise ValueError(f"{Path(coverage).name} is not on the label grid {label_tif.name}")
    sc_off = _second_crop_offset(tr, second_crop) if second_crop else None
    mills = plants[plants["family"].astype(str) == spec.mill_family]
    P = plant_trees(plants, spec)
    acc = Acc(spec, n_labels, len(mills))
    sig = spec.signature() | {"label": label_tif.name, "coverage": Path(coverage).name,
                              "second_crop": Path(second_crop).name if second_crop else None,
                              "plants": int(len(plants)), "mills": int(len(mills)),
                              "plant_ids": int(pd.util.hash_pandas_object(plants["plant_id"].astype(str),
                                                                          index=False).sum() % (1 << 31))}
    resumed = acc.load(checkpoint, sig)
    if resumed:
        log.info("reach: resuming at window %d", acc.windows_done)
    windows = list(iter_windows(width, height, spec.window))
    f = spec.check_factor
    if spec.window % f:
        raise ValueError(f"window ({spec.window}) must be a multiple of check_factor ({f})")
    row_area_ha = row_pixel_area_m2(tr, height) / 1e4
    bands = [f"crop_ha:{k}" for k in spec.crops] + [f"within_{spec.competition_km:g}km_ha:{t}" for t in spec.tags]
    cdst = None
    if cells_tif is not None:
        prof = {"driver": "GTiff", "width": math.ceil(width / f), "height": math.ceil(height / f),
                "count": len(bands), "dtype": "float32", "crs": "EPSG:4326", "nodata": 0,
                "transform": Affine(tr.a * f, 0, tr.c, 0, tr.e * f, tr.f), "compress": "deflate", "predictor": 3,
                "tiled": True, "blockxsize": 256, "blockysize": 256, "SPARSE_OK": "TRUE", "interleave": "band"}
        if not (resumed and cells_tif.exists()):
            cells_tif.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(cells_tif, "w", **prof) as d:
                for i, b in enumerate(bands, start=1):
                    d.set_band_description(i, b)
        cdst = rasterio.open(cells_tif, "r+")
    t0 = time.time()
    start = acc.windows_done
    lab_ds, cov_ds, cache = rasterio.open(label_tif), rasterio.open(coverage), {}
    try:
        for k in range(start, len(windows)):
            if (budget is not None and time.time() - t0 >= budget) or (stop_after is not None and k - start >= stop_after):
                break
            win = windows[k]
            lab = lab_ds.read(1, window=win)
            if lab.any():
                c0, r0, ww, hh = int(win.col_off), int(win.row_off), int(win.width), int(win.height)
                cov = cov_ds.read(1, window=win)
                sc = _read(second_crop, c0, r0, ww, hh, sc_off, cache) if second_crop else None
                cells = process_window(acc, spec, P, lab, cov, sc, r0, c0, tr, row_area_ha)
                if cells is not None and cdst is not None:
                    cdst.write(cells, window=Window(c0 // f, r0 // f, cells.shape[2], cells.shape[1]))
            acc.windows_done = k + 1
            if (k + 1) % save_every == 0:
                acc.save(checkpoint, sig)
                el = time.time() - t0
                rate = (k + 1 - start) / max(el, 1e-9)
                log.info("  reach: window %d/%d, %.0f s, ~%.0f min left", k + 1, len(windows), el,
                         (len(windows) - k - 1) / rate / 60)
    finally:
        lab_ds.close()
        cov_ds.close()
        for d in cache.values():
            d.close()
        if cdst is not None:
            cdst.close()
    acc.save(checkpoint, sig)
    meta = {"complete": acc.windows_done >= len(windows), "windows": len(windows), "done": acc.windows_done,
            "elapsed_s": time.time() - t0, "label_to_geocode": dict(zip(lookup["label"], lookup["geocode"],
                                                                        strict=True)),
            "plant_counts": P.counts, "bands": bands}
    return acc, meta


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def _median_from_bins(hist: np.ndarray, bin_km: float, overflow: bool = True) -> np.ndarray:
    """Weighted median distance per row of a bin histogram (linear within the bin); nan in the overflow bin."""
    tot = hist.sum(axis=1)
    cum = np.cumsum(hist, axis=1)
    half = tot / 2
    k = (cum < half[:, None]).sum(axis=1)
    k = np.minimum(k, hist.shape[1] - 1)
    prev = np.where(k > 0, cum[np.arange(len(k)), np.maximum(k - 1, 0)], 0.0)
    inbin = hist[np.arange(len(k)), k]
    frac = np.where(inbin > 0, (half - prev) / np.where(inbin > 0, inbin, 1), 0.5)
    med = (k + frac) * bin_km
    if overflow:
        med = np.where(k == hist.shape[1] - 1, np.nan, med)
    return np.where(tot > 0, med, np.nan)


def build_tables(acc: Acc, meta: dict, spec: Spec, plants: pd.DataFrame, mun: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every reach table from the accumulator. `mun`: geocode, cd_rgint, nm_rgint, uf."""
    l2g = meta["label_to_geocode"]
    labels = np.array(sorted(l2g), dtype=np.int64)
    geocodes = np.array([l2g[i] for i in labels], dtype=np.int64)
    nb = spec.n_bins
    edges = np.arange(nb) * spec.bin_km
    attrs = mun.set_index("geocode")
    from bbp.plants import select_plants

    out: dict[str, pd.DataFrame] = {}
    # ---- the base table: area by municipality and distance bin --------------------------------------------
    rows = []
    for pi, (crop, fam) in enumerate(spec.pairs):
        h = acc.hist[pi][labels]
        li, bi = np.nonzero(h)
        rows.append(pd.DataFrame({"pair": pair_tag(crop, fam), "crop": crop, "family": fam,
                                  "geocode": geocodes[li], "bin_km": edges[bi], "area_ha": h[li, bi]}))
    out["hist"] = pd.concat(rows, ignore_index=True)
    out["hist"]["overflow"] = out["hist"]["bin_km"] >= spec.max_km

    # ---- units at every level --------------------------------------------------------------------------
    unit_rows, comp_rows = [], []
    radii = [float(r) for r in spec.radii_km]
    kr = [int(round(r / spec.bin_km)) for r in radii]
    for pi, (crop, fam) in enumerate(spec.pairs):
        tag = pair_tag(crop, fam)
        h = pd.DataFrame(acc.hist[pi][labels], index=geocodes)
        ds = pd.Series(acc.dsum[pi][labels], index=geocodes)
        cp = pd.DataFrame(acc.comp[pi][labels], index=geocodes)
        sel = select_plants(plants, fam)
        for level in LEVELS:
            if level == "municipality":
                key = pd.Series(geocodes, index=geocodes)
                pk = sel["geocode"]
            elif level == "nation":
                key = pd.Series("BR", index=geocodes)
                pk = pd.Series("BR", index=sel.index)
            else:
                col = "cd_rgint" if level == "rgint" else "uf"
                key = attrs[col].reindex(geocodes)
                key.index = geocodes
                pk = sel["geocode"].map(attrs[col])
            gh, gd, gc = h.groupby(key.values).sum(), ds.groupby(key.values).sum(), cp.groupby(key.values).sum()
            tot = gh.sum(axis=1)
            keep = tot > 0
            gh, gd, gc, tot = gh[keep], gd[keep], gc[keep], tot[keep]
            cum = gh.to_numpy().cumsum(axis=1)
            med = _median_from_bins(gh.to_numpy(), spec.bin_km)
            if level == "nation":
                med = _median_from_bins(acc.nat[pi][None, :], 1.0, overflow=False)
            npl = pk.value_counts()
            base = {"level": level, "unit": gh.index.astype(str), "pair": tag, "crop": crop, "family": fam,
                    "crop_area_ha": tot.to_numpy(), "mean_km": (gd / tot).to_numpy(), "median_km": med,
                    "n_plants": npl.reindex(gh.index).fillna(0).astype(int).to_numpy(),
                    "contested_share": (gc.iloc[:, 2:].sum(axis=1) / tot).to_numpy()}
            for r, k in zip(radii, kr, strict=True):
                within = cum[:, k - 1]
                unit_rows.append(pd.DataFrame(base | {"radius_km": r, "within_ha": within,
                                                      "share": within / tot.to_numpy()}))
            if level in ("municipality", "nation"):
                for n in range(gc.shape[1]):
                    comp_rows.append(pd.DataFrame({"level": level, "unit": gc.index.astype(str), "pair": tag,
                                                   "radius_km": spec.competition_km, "n_plants": n,
                                                   "area_ha": gc.iloc[:, n].to_numpy()}))
    units = pd.concat(unit_rows, ignore_index=True)
    out["units"] = units
    comp = pd.concat(comp_rows, ignore_index=True)
    out["competition"] = comp[comp["area_ha"] > 0].reset_index(drop=True)

    # ---- national distance curves (as x04 reads them) ------------------------------------------------------
    out["distance"] = pd.concat([pd.DataFrame({"pair": pair_tag(c, f), "crop": c, "family": f, "subtype": "",
                                               "bin_km": np.arange(NAT_BINS, dtype=float), "area_ha": acc.nat[pi],
                                               "n_plants": meta["plant_counts"].get(f, 0)})
                                 for pi, (c, f) in enumerate(spec.pairs)], ignore_index=True)

    # ---- aggregation check -------------------------------------------------------------------------------
    chk = []
    for pi, (c, f) in enumerate(spec.pairs):
        a30, a_cell = acc.hist[pi].sum(axis=0), acc.check[pi]
        t30, tc = a30.sum(), a_cell.sum()
        for r, k in zip(radii, kr, strict=True):
            s30 = a30[:k].sum() / t30 if t30 else np.nan
            sc_ = a_cell[:k].sum() / tc if tc else np.nan
            chk.append({"pair": pair_tag(c, f), "radius_km": r, "share_30m": s30,
                        f"share_f{spec.check_factor}": sc_, "diff_pp": 100 * (sc_ - s30)})
    out["aggregation_check"] = pd.DataFrame(chk)

    # ---- sugarcane mills -----------------------------------------------------------------------------------
    mills = plants[plants["family"].astype(str) == spec.mill_family].copy()
    if len(mills):
        cum = np.cumsum(acc.rings, axis=1)
        cumc = np.cumsum(acc.contested, axis=1)
        need = mills["capacity"].to_numpy(float) / spec.cane_yield_t_ha
        mills["need_ha"] = need
        rr = required_radius(cum, np.nan_to_num(need, nan=np.inf), spec.bin_km)
        mills["required_radius_km"] = np.where(np.isfinite(need), rr, np.nan)
        for r in radii:
            k = int(round(r / spec.bin_km)) - 1
            if k >= cum.shape[1]:
                continue
            mills[f"cane_ha_{int(r)}km"] = cum[:, k]
            mills[f"supply_ratio_{int(r)}km"] = cum[:, k] / need
        kc = int(round(spec.competition_km / spec.bin_km)) - 1
        mills[f"contested_share_{int(spec.competition_km)}km"] = np.where(cum[:, kc] > 0, cumc[:, kc] / cum[:, kc],
                                                                          np.nan)
        if (P_tree := _op_mill_tree(plants, spec)) is not None:
            n = P_tree.query_ball_point(unit_xyz(mills["lon"], mills["lat"]), float(km_to_chord(spec.competition_km)),
                                        return_length=True)
            mills[f"other_mills_{int(spec.competition_km)}km"] = np.asarray(n) - mills["operating"].astype(int).to_numpy()
        keep = ["plant_id", "site_id", "name", "subtype", "uf", "municipality", "geocode", "cd_rgint", "nm_rgint",
                "status", "operating", "lon", "lat", "capacity", "need_ha", "required_radius_km"] + \
               [c for c in mills if c.startswith(("cane_ha_", "supply_ratio_", "contested_share_", "other_mills_"))]
        out["mills"] = pd.DataFrame(mills[keep])
        out["catchment"] = pd.concat(
            [pd.DataFrame({"plant_id": mills["plant_id"].to_numpy(), "group": spec.cane_crop, "radius_km": r,
                           "area_ha": cum[:, int(round(r / spec.bin_km)) - 1], "family": spec.mill_family,
                           "uf": mills["uf"].to_numpy(), "cd_rgint": mills["cd_rgint"].to_numpy(),
                           "capacity": mills["capacity"].to_numpy(), "capacity_unit": mills["capacity_unit"].to_numpy()})
             for r in radii if int(round(r / spec.bin_km)) - 1 < cum.shape[1]], ignore_index=True)

    # ---- per intermediate region, in the layout the exploratory figures read ------------------------------
    rg = units[units["level"] == "rgint"]
    wide = rg.pivot_table(index=["unit", "pair", "crop", "family"], columns="radius_km", values="within_ha").reset_index()
    wide.columns = [f"within_{int(c)}km_ha" if isinstance(c, float) else c for c in wide.columns]
    first = rg.drop_duplicates(["unit", "pair"]).set_index(["unit", "pair"])
    wide["area_ha"] = first.loc[list(zip(wide["unit"], wide["pair"], strict=True)), "crop_area_ha"].to_numpy()
    wide["median_km"] = first.loc[list(zip(wide["unit"], wide["pair"], strict=True)), "median_km"].to_numpy()
    wide["cd_rgint"] = wide.pop("unit").astype("int64")
    names = mun.drop_duplicates("cd_rgint").set_index("cd_rgint")
    wide["nm_rgint"] = wide["cd_rgint"].map(names["nm_rgint"])
    wide["uf"] = wide["cd_rgint"].map(names["uf"])
    out["rgint"] = wide
    pf = plants.assign(family=plants["family"].astype(str))
    out["rgint_plants"] = pf.groupby(["cd_rgint", "family"]).agg(n=("plant_id", "size"),
                                                                capacity=("capacity", "sum")).reset_index()
    return out


def _op_mill_tree(plants: pd.DataFrame, spec: Spec):
    from scipy.spatial import cKDTree

    from bbp.plants import select_plants

    op = select_plants(plants, spec.mill_family)
    return cKDTree(unit_xyz(op["lon"], op["lat"])) if len(op) else None


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_tables(ck, acc: Acc, meta: dict, spec: Spec, t: dict[str, pd.DataFrame], reference: dict[str, float]):
    """Invariants of the reach tables; `reference`: national area per crop from the statistics (ha)."""
    crop_keys = list(spec.crops)
    # crop areas: assigned + unassigned vs the published statistics
    for ci, k in enumerate(crop_keys):
        assigned, outside = float(acc.crop_area[ci].sum()), float(acc.unassigned[ci])
        total = assigned + outside
        share_out = outside / total if total else 0.0
        ck.add(f"reach_{k}_inside_mesh", share_out <= 0.001,
               f"{outside:,.0f} of {total:,.0f} ha outside the municipal mesh ({share_out:.3%})", share_out,
               severity="warn")
        if k in reference and reference[k]:
            rel = total / reference[k] - 1
            ck.add(f"reach_{k}_area_matches_statistics", abs(rel) <= 0.005,
                   f"30 m pass {total / 1e6:.3f} Mha vs statistics {reference[k] / 1e6:.3f} Mha ({rel:+.3%})",
                   round(rel, 6))
    # every pair holds the whole crop
    for pi, (crop, fam) in enumerate(spec.pairs):
        a, b = float(acc.hist[pi].sum()), float(acc.crop_area[crop_keys.index(crop)].sum())
        ck.add(f"reach_{pair_tag(crop, fam)}_holds_the_crop", abs(a - b) <= 1e-6 * max(b, 1),
               f"histogram {a:,.0f} ha vs crop {b:,.0f} ha", a - b)
    u = t["units"]
    # levels add up to the nation
    for (pair, r), g in u.groupby(["pair", "radius_km"]):
        nat = g[g["level"] == "nation"]
        if nat.empty:
            continue
        for lev in ("municipality", "rgint", "uf"):
            s = g[g["level"] == lev]
            ok = np.isclose(s["within_ha"].sum(), nat["within_ha"].iloc[0], rtol=1e-9) and \
                np.isclose(s["crop_area_ha"].sum(), nat["crop_area_ha"].iloc[0], rtol=1e-9)
            if not ok:
                ck.add(f"reach_{pair}_{lev}_{r:g}km_adds_up", False, f"{lev} does not add up to the nation")
    ck.add("reach_levels_add_up", True, "municipality, intermediate region and state sums equal the nation "
           "(area and area within every radius)")
    bad = u[(u["share"] < -1e-12) | (u["share"] > 1 + 1e-12)]
    ck.add("reach_shares_in_0_1", bad.empty, f"{len(bad)} shares outside [0, 1]", len(bad))
    mono = u.sort_values("radius_km").groupby(["level", "unit", "pair"])["within_ha"].apply(
        lambda s: bool((np.diff(s.to_numpy()) >= -1e-9).all()))
    ck.add("reach_within_grows_with_radius", bool(mono.all()), f"{int((~mono).sum())} units where it does not",
           int((~mono).sum()))
    # the aggregation check and the headline numbers
    ac = t["aggregation_check"]
    worst = ac.loc[ac["diff_pp"].abs().idxmax()] if len(ac) else None
    if worst is not None:
        ck.add("reach_aggregation_effect", abs(worst["diff_pp"]) <= 1.0,
               f"distances from {spec.check_factor}x cells (~{spec.check_factor * 30} m) instead of 30 m pixels "
               f"move the share within reach by at most {worst['diff_pp']:+.2f} pp ({worst['pair']}, "
               f"{worst['radius_km']:g} km)", round(float(worst["diff_pp"]), 3), severity="warn")
    nat = u[(u["level"] == "nation")]
    for pair, g in nat.groupby("pair"):
        s = g.set_index("radius_km")["share"]
        ck.add(f"reach_{pair}", True, f"{g['n_plants'].iloc[0]} plants; {g['crop_area_ha'].iloc[0] / 1e6:.2f} Mha; "
               + ", ".join(f"{r:g} km {v:.1%}" for r, v in s.items()), s.round(4).to_dict())
    if "mills" in t:
        m = t["mills"]
        col = f"supply_ratio_{int(spec.competition_km)}km"
        if col in m:
            op = m[m["operating"].astype(bool)]
            ck.add("reach_mills_supply_below_1", True,
                   f"{(op[col] < 1).mean():.1%} of {len(op)} operating mills ({(m[col] < 1).mean():.1%} of all "
                   f"{len(m)}) have less cane than they need within {spec.competition_km:g} km "
                   f"({spec.cane_yield_t_ha:g} t/ha)", round(float((op[col] < 1).mean()), 4))
    comp = t["competition"]
    for pair, g in comp[comp["level"] == "nation"].groupby("pair"):
        share = g.loc[g["n_plants"] >= 2, "area_ha"].sum() / g["area_ha"].sum()
        ck.add(f"reach_{pair}_competition", True, f"{share:.1%} of the area has >= 2 plants within "
               f"{spec.competition_km:g} km", round(float(share), 4))


def geodesic_check(ck, plants: pd.DataFrame, n: int = 2000, seed: int = 0):
    """Great circle (sphere) vs WGS84 geodesic for random pixel-plant pairs within 150 km."""
    from pyproj import Geod

    if len(plants) == 0:
        return
    rng = np.random.default_rng(seed)
    p = plants.sample(n=min(n, len(plants) * 10), replace=True, random_state=seed)
    az = rng.uniform(0, 360, len(p))
    dist = rng.uniform(5_000, 150_000, len(p))
    g = Geod(ellps="WGS84")
    lon2, lat2, _ = g.fwd(p["lon"].to_numpy(), p["lat"].to_numpy(), az, dist)
    hs = haversine_km(p["lon"].to_numpy(), p["lat"].to_numpy(), lon2, lat2) * 1000
    rel = np.abs(hs / dist - 1)
    ck.add("reach_great_circle_vs_wgs84", float(rel.max()) <= 0.0075,
           f"max |sphere/WGS84 − 1| = {rel.max():.3%} over {len(p)} pairs of 5–150 km", round(float(rel.max()), 6),
           severity="warn")


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------
def _sources(cfg, year: int, ck):
    """(label grid, coverage raster or VRT, second-crop raster) of `year`; builds the VRT and the label grid."""
    import geopandas as gpd

    from bbp.ingest import border_share as bs
    from bbp.ingest import second_crop as sc

    pattern = cfg.section("border_share").get("tif_pattern", "brazil_coverage-col11_{year}.tif")
    src = bs.find_year_sources(cfg.input_path("coverage_tif_dir"), pattern, [year]).get(year)
    if src is None:
        ck.add("reach_coverage_found", False, f"no coverage raster for {year}")
        return None
    if isinstance(src, list):
        # a VRT of its own: the one of `bbp border-share` may hold the paths of another machine (Docker)
        vrt = cfg.out("interim", "reach", f"{Path(pattern.format(year=year)).stem}.vrt")
        bs.build_vrt(src, vrt)
        src = vrt
    ck.add("reach_coverage_found", True, f"{Path(src).name}")
    grid = cfg.out("interim", f"municipality_grid_{bs.grid_key(src)}.tif")
    if not grid.exists():
        mesh = gpd.read_parquet(cfg.out("interim", "municipalities_geom.parquet"))
        sc.build_label_grid(mesh[mesh["is_municipality"]], src, grid,
                            window=cfg.section("border_share").get("window", 2048))
    sc_spec = cfg.section("second_crop")
    scr = sc.find_rasters(cfg.input_path("second_crop_tif_dir"), sc_spec["tif_pattern"], [year]).get(year)
    ck.add("reach_second_crop_found", scr is not None, f"second-crop raster {year}: {scr.name if scr else 'missing'}",
           severity="warn")
    return grid, src, scr


def reference_areas(cfg, spec: Spec, year: int) -> dict[str, float]:
    """National area of each reach crop from the statistics: Col. 11 coverage (classes) and the second-crop
    zonal statistics of `bbp second-crop` (values); only crops that are one or the other."""
    out = {}
    cov_p = cfg.data_dir / "interim" / "mapbiomas_coverage_municipal.parquet"
    sc_p = cfg.data_dir / "interim" / "second_crop_municipal.parquet"
    cov = pd.read_parquet(cov_p) if cov_p.exists() else None
    scd = pd.read_parquet(sc_p) if sc_p.exists() else None
    names = {int(k): v for k, v in (cfg.section("second_crop").get("classes") or {}).items()}
    for k, (cc, ss) in spec.crops.items():
        if cc and not ss and cov is not None:
            out[k] = float(cov[(cov["year"] == year) & cov["class_id"].isin(cc)]["area_ha"].sum())
        elif ss and not cc and scd is not None:
            out[k] = float(scd[(scd["year"] == year) & scd["class"].isin([names[s] for s in ss])]["area_ha"].sum())
    return out


def run(cfg, ck=None, budget: float | None = None) -> dict[str, Path]:
    """`bbp reach`: the 30 m pass (resumable) and, once complete, every reach table."""
    from bbp.checks import CheckLog
    from bbp.plants import load_plants

    ck = ck if ck is not None else CheckLog("reach")
    spec = Spec.from_config(cfg)
    r = (cfg.section("plants") or {}).get("reach", {})
    year = int(r.get("year", cfg.years[-1]))
    plants = load_plants(cfg)
    if not ck.add("reach_plants_found", plants is not None, "data/interim/plants.parquet (run bbp run plants)"):
        return {}
    src = _sources(cfg, year, ck)
    if src is None:
        return {}
    grid, cov, scr = src
    interim = cfg.out("interim", "reach")
    interim.mkdir(parents=True, exist_ok=True)
    acc, meta = accumulate(grid, cov, scr, plants, spec, interim / f"reach_{year}_checkpoint.npz",
                           cells_tif=cfg.out("processed", f"reach_cells_f{spec.check_factor}.tif"), budget=budget)
    ck.add("reach_pass_complete", meta["complete"], f"{meta['done']}/{meta['windows']} windows, "
           f"{meta['elapsed_s']:.0f} s this call", meta["done"], severity="warn")
    if not meta["complete"]:
        return {}
    import geopandas as gpd  # noqa: F401  (plants is a GeoDataFrame)

    mun = pd.read_parquet(cfg.data_dir / "interim" / "municipalities.parquet",
                          columns=["geocode", "cd_rgint", "nm_rgint", "uf"])
    tables = build_tables(acc, meta, spec, pd.DataFrame(plants.drop(columns="geometry")), mun)
    geodesic_check(ck, plants)
    check_tables(ck, acc, meta, spec, tables, reference_areas(cfg, spec, year))
    out = cfg.data_dir / "processed"
    written = {}
    for name, df in tables.items():
        p = out / f"reach_{name}.parquet"
        df.to_parquet(p)
        written[name] = p
    written["cells"] = cfg.out("processed", f"reach_cells_f{spec.check_factor}.tif")
    (out / "reach_meta.json").write_text(json.dumps(
        {"year": year, "coverage": Path(cov).name, "second_crop": scr.name if scr else None,
         "label_grid": grid.name, "plants": meta["plant_counts"], "bands": meta["bands"],
         "spec": spec.signature()}, indent=1, ensure_ascii=False), encoding="utf-8")
    return written


__all__ = ["Spec", "accumulate", "build_tables", "run", "required_radius", "haversine_km", "nearest_km"]
