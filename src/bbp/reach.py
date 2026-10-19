"""What lies around the processing plants: crop area by distance, at the pixel and at the statistical units.

Input: the per-group pixel counts of `bbp landcover` (MapBiomas Col. 11, 30 m pixels counted in ~480 m
cells, so every area below is the 30 m area, located to within one cell) and `bbp.plants`.

* rings: for every plant, the area of each land-cover group in 1 km rings up to `max_km` (great-circle
  distance from the plant to the cell centre) -> catchment areas at any radius (Cervi et al. 2019 for cane);
* required radius of each sugarcane mill: the radius whose cane area, at `cane_yield_t_ha`, supplies its
  crushing class;
* national distance curves: for every crop cell, the distance to the nearest plant of a family (and the
  number of mills within each radius: competition for the same cane);
* the MAUP access test: the share of a crop "within r km" of a plant when distance is measured from the
  pixel, from the municipality's representative point, or from the intermediate region's — the same land,
  three answers;
* per intermediate region: crop areas, plants and capacity per family, share within each radius.

Everything is written to data/processed/reach_*.parquet (and reach_rings.npz).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from bbp import landcover as lc
from bbp.plants import FAMILIES, SOY

log = logging.getLogger("bbp")

R_EARTH_KM = 6371.0088
BASE_RES = 0.00026949458523585647                           # MapBiomas 30 m pixel (degrees)
GROUPS = [g.key for g in lc.GROUPS][:10]                    # crops + pasture (codes 1..10)
CROPS_ONLY = tuple(GROUPS[:9])
PSEUDO = {"soy_all": SOY, "all_crops": CROPS_ONLY}
LABELS = {"soy_all": "Soybean (all)", "all_crops": "All crops", "soy_corn2": "Soybean + 2nd-season corn",
          "sugarcane": "Sugarcane", "pasture": "Pasture", "soybean": "Soybean (single crop)"}


def members(crop: str) -> tuple[str, ...]:
    return PSEUDO.get(crop, (crop,))


def _spec(cfg) -> dict:
    return (cfg.section("plants") or {}).get("reach", {})


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------
class Grid:
    """The counts raster in memory: bands of GROUPS (uint16), cell geometry, and areas per pixel."""

    def __init__(self, path: Path, groups=GROUPS):
        import rasterio

        from bbp.ingest.second_crop import row_pixel_area_m2

        with rasterio.open(path) as ds:
            self.counts = ds.read([lc.CODE[g] for g in groups])
            self.tr = ds.transform
            self.h, self.w = ds.height, ds.width
        self.groups = list(groups)
        self.res = self.tr.a
        self.x0, self.y0 = self.tr.c, self.tr.f
        self.lat = self.y0 - (np.arange(self.h) + 0.5) * self.res
        self.lon = self.x0 + (np.arange(self.w) + 0.5) * self.res
        self.factor = int(round(self.res / BASE_RES))
        self.px_ha = row_pixel_area_m2(self.tr, self.h) / 1e4 / self.factor ** 2      # one 30 m pixel, per row

    def band(self, crop: str) -> np.ndarray:
        """Pixel counts of a group or pseudo-group (uint32)."""
        idx = [self.groups.index(g) for g in members(crop)]
        out = self.counts[idx[0]].astype(np.uint32)
        for i in idx[1:]:
            out += self.counts[i]
        return out

    def area_ha(self, crop: str) -> np.ndarray:
        return self.band(crop) * self.px_ha[:, None]

    def cells(self, crop: str):
        """(rows, cols, area_ha) of the cells holding the crop."""
        b = self.band(crop)
        r, c = np.nonzero(b)
        return r, c, b[r, c] * self.px_ha[r]

    def window(self, lon: float, lat: float, km: float):
        dlat = km / 111.0 + self.res
        dlon = km / (111.0 * max(math.cos(math.radians(lat)), 0.05)) + self.res
        r0 = max(0, int((self.y0 - (lat + dlat)) / self.res))
        r1 = min(self.h, int((self.y0 - (lat - dlat)) / self.res) + 1)
        c0 = max(0, int((lon - dlon - self.x0) / self.res))
        c1 = min(self.w, int((lon + dlon - self.x0) / self.res) + 1)
        return r0, r1, c0, c1


def haversine_km(lon0, lat0, lon, lat):
    p0, p = np.radians(lat0), np.radians(lat)
    a = np.sin((p - p0) / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin(np.radians(lon - lon0) / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def unit_xyz(lon, lat) -> np.ndarray:
    lo, la = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def chord_to_km(ch):
    return 2 * R_EARTH_KM * np.arcsin(np.clip(np.asarray(ch) / 2, 0, 1))


def km_to_chord(km):
    return 2 * np.sin(np.asarray(km) / (2 * R_EARTH_KM))


# ---------------------------------------------------------------------------
# rings around each plant
# ---------------------------------------------------------------------------
def plant_rings(grid: Grid, plants: pd.DataFrame, ring_km: float = 1.0, max_km: float = 150.0) -> np.ndarray:
    """[plant, group, ring] area (ha) of every group in rings [k, k+1) * ring_km around each plant."""
    n_r = int(round(max_km / ring_km))
    out = np.zeros((len(plants), len(grid.groups), n_r), dtype=np.float32)
    nb = len(grid.groups)
    for i, (lon, lat) in enumerate(zip(plants["lon"].to_numpy(), plants["lat"].to_numpy(), strict=True)):
        r0, r1, c0, c1 = grid.window(lon, lat, max_km)
        if r1 <= r0 or c1 <= c0:
            continue
        d = haversine_km(lon, lat, grid.lon[None, c0:c1], grid.lat[r0:r1, None])
        k = (d / ring_km).astype(np.int32)
        ok = k < n_r
        kk = k[ok]
        w = np.broadcast_to(grid.px_ha[r0:r1, None], d.shape)[ok]
        sub = grid.counts[:, r0:r1, c0:c1][:, ok]
        flat = (np.arange(nb)[:, None] * n_r + kk[None, :]).ravel()
        out[i] = np.bincount(flat, weights=(sub * w[None, :]).ravel(), minlength=nb * n_r).reshape(nb, n_r)
    return out


def catchment_table(plants: pd.DataFrame, rings: np.ndarray, groups: list[str], radii, ring_km: float) -> pd.DataFrame:
    cum = np.cumsum(rings, axis=2)
    rows = []
    for r in radii:
        k = int(round(r / ring_km)) - 1
        for gi, g in enumerate(groups):
            rows.append(pd.DataFrame({"plant_id": plants["plant_id"].values, "group": g, "radius_km": r,
                                      "area_ha": cum[:, gi, k]}))
    return pd.concat(rows, ignore_index=True)


def required_radius(cum_cane: np.ndarray, need_ha: np.ndarray, ring_km: float) -> np.ndarray:
    """Smallest radius (km) whose cumulative cane area reaches need_ha; inf beyond the rings."""
    ok = cum_cane >= need_ha[:, None]
    first = np.where(ok.any(axis=1), ok.argmax(axis=1), -1)
    return np.where(first >= 0, (first + 1) * ring_km, np.inf)


# ---------------------------------------------------------------------------
# national distance curves, competition, the MAUP test
# ---------------------------------------------------------------------------
def select_plants(plants: pd.DataFrame, family: str, subtype: str | None = None, operating_only: bool = True):
    sel = plants[plants["family"].astype(str) == family]
    if subtype:
        sel = sel[sel["subtype"] == subtype]
    if operating_only and "operating" in sel:
        sel = sel[sel["operating"].fillna(True).astype(bool)]
    return sel


def nearest_km(tree, lon, lat, k: int = 1, upper_km: float | None = None, chunk: int = 2_000_000):
    """Distances (km) from points to their k nearest tree points (inf where none within upper_km)."""
    out_d = np.empty((len(lon), k), dtype=np.float32)
    ub = float(km_to_chord(upper_km)) if upper_km else np.inf
    for a in range(0, len(lon), chunk):
        d, _ = tree.query(unit_xyz(lon[a:a + chunk], lat[a:a + chunk]), k=k, distance_upper_bound=ub)
        d = np.asarray(d).reshape(-1, k)
        out_d[a:a + chunk] = np.where(np.isfinite(d), chord_to_km(np.where(np.isfinite(d), d, 0)), np.inf)
    return out_d


def municipality_grid(cfg, grid: Grid) -> np.ndarray:
    """Geocode of every cell (int32, 0 outside), rasterised once (cell centre inside) and cached."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    path = cfg.data_dir / "interim" / "landcover" / f"municipality_f{grid.factor}.tif"
    if path.exists():
        with rasterio.open(path) as ds:
            if ds.shape == (grid.h, grid.w):
                return ds.read(1)
    g = gpd.read_parquet(cfg.data_dir / "interim" / "municipalities_geom.parquet", columns=["geocode", "geometry"])
    # SIRGAS 2000 and WGS 84 agree to < 1 m; simplified to a quarter cell (20 M vertices otherwise)
    g["geometry"] = g.geometry.simplify(grid.res / 4, preserve_topology=True)
    arr = rasterize(((geom, int(code)) for geom, code in zip(g.geometry, g["geocode"], strict=True)),
                    out_shape=(grid.h, grid.w), transform=grid.tr, fill=0, dtype="int32")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", driver="GTiff", width=grid.w, height=grid.h, count=1, dtype="int32",
                           crs="EPSG:4326", transform=grid.tr, nodata=0, compress="deflate", tiled=True) as d:
            d.write(arr, 1)
    except Exception as exc:  # a read-only cache folder is no reason to stop
        log.warning("municipality grid not cached: %s", exc)
    return arr


def unit_points(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Representative points of municipalities and of intermediate regions (lon, lat), with names."""
    import geopandas as gpd

    attrs = pd.read_parquet(cfg.data_dir / "interim" / "municipalities.parquet")
    geom = gpd.read_parquet(cfg.data_dir / "interim" / "municipalities_geom.parquet", columns=["geocode", "geometry"])
    geom = geom.merge(attrs[["geocode", "cd_rgint", "nm_rgint", "uf", "name"]], on="geocode")
    pt = geom.geometry.representative_point()
    mun = geom.drop(columns="geometry").assign(lon=pt.x.values, lat=pt.y.values)
    rg = geom.dissolve(by="cd_rgint", as_index=False)[["cd_rgint", "nm_rgint", "uf", "geometry"]]
    rpt = rg.geometry.representative_point()
    rg = rg.drop(columns="geometry").assign(lon=rpt.x.values, lat=rpt.y.values)
    return mun, rg


def run(cfg, ck=None) -> dict[str, Path]:
    """All reach tables; returns the written paths."""
    from scipy.spatial import cKDTree

    from bbp.plants import load_plants

    spec = _spec(cfg)
    radii = [float(r) for r in spec.get("radii_km", [10, 20, 30, 50, 75, 100])]
    ring_km = float(spec.get("ring_km", 1.0))
    max_km = float(spec.get("max_km", 150.0))
    yield_t_ha = float(spec.get("cane_yield_t_ha", 75.0))
    pairs = spec.get("pairs") or []
    year = int((cfg.section("paper").get("fig1") or {}).get("year", cfg.years[-1]))
    name = f"col{cfg.section('mapbiomas').get('collection', 11)}_{year}"
    cpath = lc.counts_path(cfg.data_dir / "interim" / "landcover", name, 16)
    plants = load_plants(cfg)
    ok = cpath.exists() and plants is not None
    if ck is not None:
        ck.add("reach_inputs", ok, f"{cpath.name}: {cpath.exists()}; plants: {plants is not None}",
               severity="warn")
    if not ok:
        return {}
    out = cfg.data_dir / "processed"
    out.mkdir(parents=True, exist_ok=True)
    grid = Grid(cpath)
    total = {g: float(grid.area_ha(g).sum()) for g in GROUPS}
    log.info("  reach: grid %d x %d, sugarcane %.2f Mha, soybean family %.2f Mha", grid.h, grid.w,
             total["sugarcane"] / 1e6, sum(total[g] for g in SOY) / 1e6)
    written = {}

    # ---- rings and catchments --------------------------------------------------------------------
    rings = plant_rings(grid, plants, ring_km, max_km)
    np.savez_compressed(out / "reach_rings.npz", rings=rings, plant_id=plants["plant_id"].to_numpy().astype(str),
                        groups=np.array(grid.groups), ring_km=ring_km)
    cat = catchment_table(plants, rings, grid.groups, radii, ring_km)
    cat = cat.merge(plants[["plant_id", "family", "subtype", "uf", "cd_rgint", "capacity", "capacity_unit"]],
                    on="plant_id")
    cat["family"] = cat["family"].astype(str)
    cat.to_parquet(out / "reach_catchment.parquet")
    written["catchment"] = out / "reach_catchment.parquet"

    # ---- sugarcane mills: required radius, supply ratio ------------------------------------------------
    mills = plants[plants["family"].astype(str) == "sugarcane_mill"].copy()
    mi = mills.index.to_numpy()
    pos = plants.index.get_indexer(mi)
    cum_cane = np.cumsum(rings[pos, grid.groups.index("sugarcane"), :], axis=1)
    need = mills["capacity"].to_numpy() / yield_t_ha
    mills["need_ha"] = need
    mills["required_radius_km"] = np.where(np.isfinite(need), required_radius(cum_cane, np.nan_to_num(need, nan=0),
                                                                              ring_km), np.nan)
    for r in radii:
        k = int(round(r / ring_km)) - 1
        mills[f"cane_ha_{int(r)}km"] = cum_cane[:, k]
        mills[f"supply_ratio_{int(r)}km"] = cum_cane[:, k] / need
    keep = ["plant_id", "name", "subtype", "uf", "municipality", "cd_rgint", "nm_rgint", "status", "lon", "lat",
            "capacity", "need_ha", "required_radius_km"] + [c for c in mills if c.startswith(("cane_ha_", "supply_"))]
    mills[keep].to_parquet(out / "reach_mills.parquet")
    written["mills"] = out / "reach_mills.parquet"

    # ---- national: nearest plant per crop cell, competition, MAUP ----------------------------------
    gmun = municipality_grid(cfg, grid)
    mun, rg = unit_points(cfg)
    mun_rg = mun.set_index("geocode")["cd_rgint"]
    hist_rows, maup_rows, comp_rows, rg_rows = [], [], [], []
    bins = np.arange(0, 2001, ring_km)
    for pr in pairs:
        crop, fam, sub = pr["crop"], pr["family"], pr.get("subtype")
        tag = f"{crop}|{fam}" + (f"|{sub}" if sub else "")
        sel = select_plants(plants, fam, sub)
        if len(sel) == 0:
            continue
        tree = cKDTree(unit_xyz(sel["lon"].to_numpy(), sel["lat"].to_numpy()))
        r, c, a = grid.cells(crop)
        kq = 6 if fam == "sugarcane_mill" else 1
        d = nearest_km(tree, grid.lon[c], grid.lat[r], k=kq, upper_km=max(radii) if kq > 1 else None)
        d1 = d[:, 0] if kq == 1 else nearest_km(tree, grid.lon[c], grid.lat[r], k=1)[:, 0]   # unbounded
        h, _ = np.histogram(np.minimum(d1, bins[-1] - 1e-6), bins=bins, weights=a)
        hist_rows.append(pd.DataFrame({"pair": tag, "crop": crop, "family": fam, "subtype": sub or "",
                                       "bin_km": bins[:-1], "area_ha": h, "n_plants": len(sel)}))
        # MAUP: pixel vs municipality point vs region point
        cell_mun = gmun[r, c]
        df = pd.DataFrame({"geocode": cell_mun, "a": a})
        mun_area = df[df["geocode"] > 0].groupby("geocode")["a"].sum()
        mpts = mun.set_index("geocode").loc[mun_area.index]
        dm = nearest_km(tree, mpts["lon"].to_numpy(), mpts["lat"].to_numpy())[:, 0]
        rg_area = mun_area.groupby(mun_rg.reindex(mun_area.index).to_numpy()).sum()
        rpts = rg.set_index("cd_rgint").loc[rg_area.index]
        dr = nearest_km(tree, rpts["lon"].to_numpy(), rpts["lat"].to_numpy())[:, 0]
        tot = a.sum()
        for rad in radii:
            maup_rows += [{"pair": tag, "crop": crop, "family": fam, "radius_km": rad, "level": lev,
                           "share": float(v / tot)} for lev, v in (
                ("pixel", a[d1 <= rad].sum()), ("municipality", mun_area[dm <= rad].sum()),
                ("intermediate region", rg_area[dr <= rad].sum()))]
        # per intermediate region: crop area and the part within each radius (pixel distances)
        cell_rg = mun_rg.reindex(cell_mun).to_numpy()
        frame = pd.DataFrame({"cd_rgint": cell_rg, "a": a, "d": d1})
        frame = frame[pd.notna(frame["cd_rgint"])]
        agg = frame.groupby("cd_rgint")["a"].sum().rename("area_ha").to_frame()
        for rad in radii:
            agg[f"within_{int(rad)}km_ha"] = frame[frame["d"] <= rad].groupby("cd_rgint")["a"].sum()
        agg["median_km"] = frame.groupby("cd_rgint").apply(
            lambda x: _weighted_median(x["d"].to_numpy(), x["a"].to_numpy()), include_groups=False)
        agg = agg.fillna(0).reset_index().assign(pair=tag, crop=crop, family=fam)
        rg_rows.append(agg)
        # competition: how many plants within each radius of the crop
        if kq > 1:
            for rad in radii:
                n = (d <= rad).sum(axis=1)
                cnt = np.bincount(np.minimum(n, kq), weights=a, minlength=kq + 1)
                comp_rows += [{"pair": tag, "radius_km": rad, "n_plants": int(j), "area_ha": float(v)}
                              for j, v in enumerate(cnt)]
        if ck is not None:
            within = a[d1 <= 30].sum() / tot
            ck.add(f"reach_{tag}", True, f"{len(sel)} plants; {tot / 1e6:.2f} Mha; {within:.1%} within 30 km",
                   round(within, 4))
    pd.concat(hist_rows, ignore_index=True).to_parquet(out / "reach_distance.parquet")
    pd.DataFrame(maup_rows).to_parquet(out / "reach_maup.parquet")
    pd.DataFrame(comp_rows).to_parquet(out / "reach_competition.parquet")
    rgt = pd.concat(rg_rows, ignore_index=True)
    rgt = rgt.merge(rg[["cd_rgint", "nm_rgint", "uf"]], on="cd_rgint", how="left")
    rgt.to_parquet(out / "reach_rgint.parquet")
    # plants per region and family
    pfam = plants.assign(family=plants["family"].astype(str)).groupby(["cd_rgint", "family"]).agg(
        n=("plant_id", "size"), capacity=("capacity", "sum")).reset_index()
    pfam.to_parquet(out / "reach_rgint_plants.parquet")
    for k in ("distance", "maup", "competition", "rgint", "rgint_plants"):
        written[k] = out / f"reach_{k}.parquet"
    return written


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, cw[-1] / 2)])


__all__ = ["FAMILIES", "Grid", "run", "plant_rings", "required_radius", "haversine_km", "nearest_km"]
