"""Scale as a property of the system (`bbp scale`): the same land and the same plants, reach measured at three
resolutions.

* **pixel** — the reference: every 30 m crop pixel at its own distance (`bbp reach`, `reach_hist` and
  `reach_distance`).
* **municipality** and **intermediate region** — what a study working with statistical units can know: the crop
  area of the unit, and the unit's geometry. Two rules say how much of that area is "within r km":
  * `area_fraction` (main rule): the crop is spread over the unit like its land, so the share within reach is the
    share of the unit's land area within r km of a plant. Land area comes from the ~480 m cells of
    `reach_cells_f<factor>.tif` (cell centre in the unit, geodesic cell area);
  * `centroid` (sensitivity): the unit counts whole when its centroid is within r km;
  * `representative_point` (the exploratory x05): the same with a point guaranteed inside the polygon.

Distances are great circles to the nearest operating plant of the pair, as in `bbp reach`. With the radius
growing, every resolution and rule must reach 100 % (checked, with `inf` among the radii).

Output: data/processed/reach_scale.parquet (pair × radius × resolution × rule, national) and
reach_scale_units.parquet (the same per municipality and per intermediate region).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from bbp.reach import NAT_BINS, Spec, chord_to_km, pair_tag, unit_xyz

log = logging.getLogger("bbp")

RULES = ("area_fraction", "centroid", "representative_point")
DEFAULT_RADII = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000]


def radii(cfg) -> list[float]:
    r = ((cfg.section("plants") or {}).get("scale") or {}).get("radii_km") or DEFAULT_RADII
    return sorted(float(x) for x in r) + [math.inf]


# ---------------------------------------------------------------------------
# the land grid (~480 m cells)
# ---------------------------------------------------------------------------
def land_grid(cfg, cells_tif: Path) -> tuple[np.ndarray, object]:
    """Geocode of every cell of the reach cell grid (int32, 0 outside the mesh): cell centre inside the municipality
    (polygons simplified to a quarter cell). Built once and cached in data/interim/scale."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(cells_tif) as ds:
        tr, shape = ds.transform, (ds.height, ds.width)
    path = cfg.out("interim", "scale", f"municipality_{shape[1]}x{shape[0]}.tif")
    if path.exists():
        with rasterio.open(path) as ds:
            if ds.shape == shape and ds.transform == tr:
                return ds.read(1), tr
    g = gpd.read_parquet(cfg.out("interim", "municipalities_geom.parquet"))
    g = g[g["is_municipality"]] if "is_municipality" in g else g
    geom = g.geometry.simplify(abs(tr.a) / 4, preserve_topology=True)
    arr = rasterize(((gm, int(code)) for gm, code in zip(geom, g["geocode"], strict=True)), out_shape=shape,
                    transform=tr, fill=0, dtype="int32")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=shape[1], height=shape[0], count=1, dtype="int32",
                       crs="EPSG:4326", transform=tr, nodata=0, compress="deflate", tiled=True) as d:
        d.write(arr, 1)
    return arr, tr


def land_within(grid: np.ndarray, tr, trees: dict, radii_km: list[float], rows: int = 512) -> tuple[pd.Series, dict]:
    """Land area per municipality (ha) and, per family, the land area within each radius of its nearest plant
    ({family: DataFrame geocode × radius})."""
    from bbp.ingest.second_crop import row_pixel_area_m2

    area_row = row_pixel_area_m2(tr, grid.shape[0]) / 1e4
    codes = np.unique(grid[grid > 0])
    lut = np.zeros(int(grid.max()) + 1, dtype=np.int64)
    lut[codes] = np.arange(len(codes))
    land = np.zeros(len(codes))
    finite = [r for r in radii_km if math.isfinite(r)]
    within = {f: np.zeros((len(codes), len(radii_km))) for f in trees}
    lon = tr.c + (np.arange(grid.shape[1]) + 0.5) * tr.a
    for r0 in range(0, grid.shape[0], rows):
        g = grid[r0:r0 + rows]
        rr, cc = np.nonzero(g)
        if len(rr) == 0:
            continue
        u = lut[g[rr, cc]]
        a = area_row[r0 + rr]
        land += np.bincount(u, weights=a, minlength=len(codes))
        lat = tr.f + (r0 + rr + 0.5) * tr.e
        x = unit_xyz(lon[cc], lat)
        for fam, tree in trees.items():
            if tree is None:
                continue
            d = chord_to_km(tree.query(x, workers=-1)[0])
            for k, r in enumerate(finite):
                m = d < r
                within[fam][:, k] += np.bincount(u[m], weights=a[m], minlength=len(codes))
            within[fam][:, len(finite):] += np.bincount(u, weights=a, minlength=len(codes))[:, None]   # inf
    land = pd.Series(land, index=codes)
    return land, {f: pd.DataFrame(w, index=codes, columns=radii_km) for f, w in within.items()}


def unit_points(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Centroid and representative point (lon, lat) of every municipality and intermediate region; centroids are
    taken in SIRGAS 2000 / Brazil Polyconic and brought back to SIRGAS 2000 geographic coordinates."""
    import geopandas as gpd

    geom = gpd.read_parquet(cfg.out("interim", "municipalities_geom.parquet"), columns=["geocode", "geometry"])
    attrs = pd.read_parquet(cfg.out("interim", "municipalities.parquet"), columns=["geocode", "cd_rgint",
                                                                                  "is_municipality"])
    geom = geom.merge(attrs, on="geocode")
    geom = geom[geom["is_municipality"]]

    def pts(g, key):
        c = g.to_crs("EPSG:5880").geometry.centroid.to_crs("EPSG:4674")
        rp = g.to_crs("EPSG:4674").geometry.representative_point()
        return pd.DataFrame({key: g[key].to_numpy(), "centroid_lon": c.x.to_numpy(), "centroid_lat": c.y.to_numpy(),
                             "representative_point_lon": rp.x.to_numpy(), "representative_point_lat": rp.y.to_numpy()})

    mun = pts(geom, "geocode")
    rg = geom.dissolve(by="cd_rgint", as_index=False)[["cd_rgint", "geometry"]]
    return mun.set_index("geocode"), pts(rg, "cd_rgint").set_index("cd_rgint")


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
def pixel_shares(hist: pd.DataFrame, distance: pd.DataFrame, pair: str, radii_km: list[float], max_km: float):
    """National crop area within each radius at the pixel: municipal 1 km bins below max_km, the national curve
    (1 km bins to 2000 km) above; inf = everything."""
    h = hist[hist["pair"] == pair]
    nat = distance[distance["pair"] == pair].sort_values("bin_km")
    total = float(h["area_ha"].sum())
    out = {}
    for r in radii_km:
        if not math.isfinite(r):
            out[r] = total
        elif r <= max_km:
            out[r] = float(h.loc[h["bin_km"] < r, "area_ha"].sum())
        else:
            out[r] = float(nat.loc[nat["bin_km"] < min(r, NAT_BINS), "area_ha"].sum())
    return total, out


def unit_table(crop_ha: pd.Series, land: pd.Series, land_in: pd.DataFrame, pts: pd.DataFrame, tree,
               radii_km: list[float]) -> pd.DataFrame:
    """Per unit: crop area within each radius under every rule. `crop_ha` and `land` indexed by unit."""
    units = crop_ha.index[crop_ha > 0]
    rows = []
    frac = land_in.reindex(units).div(land.reindex(units), axis=0)
    no_land = frac.isna().any(axis=1)
    for rule in ("centroid", "representative_point"):
        p = pts.reindex(units)
        d = chord_to_km(tree.query(unit_xyz(p[f"{rule}_lon"], p[f"{rule}_lat"]), workers=-1)[0])
        for r in radii_km:
            rows.append(pd.DataFrame({"unit": units, "rule": rule, "radius_km": r,
                                      "within_ha": crop_ha.reindex(units).to_numpy() * (d < r)}))
        if rule == "centroid":
            cen = pd.DataFrame({r: (d < r).astype(float) for r in radii_km}, index=units)
    # area_fraction; a unit with no land cell (smaller than a cell) falls back to its centroid
    frac[no_land] = cen[no_land]
    for r in radii_km:
        rows.append(pd.DataFrame({"unit": units, "rule": "area_fraction", "radius_km": r,
                                  "within_ha": crop_ha.reindex(units).to_numpy() * frac[r].to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    out["crop_area_ha"] = out["unit"].map(crop_ha)
    out.attrs["no_land_units"] = int(no_land.sum())
    return out


def build(cfg, ck) -> dict[str, pd.DataFrame]:
    from scipy.spatial import cKDTree

    from bbp.plants import load_plants, select_plants

    spec = Spec.from_config(cfg)
    rad = radii(cfg)
    proc = cfg.data_dir / "processed"
    need = [proc / f"reach_{n}.parquet" for n in ("hist", "distance", "units")]
    cells = proc / f"reach_cells_f{spec.check_factor}.tif"
    plants = load_plants(cfg)
    ok = all(p.exists() for p in need) and cells.exists() and plants is not None
    if not ck.add("scale_inputs", ok, "reach tables, reach cells and the plant register (run bbp reach)"):
        return {}
    hist, distance = pd.read_parquet(need[0]), pd.read_parquet(need[1])
    units = pd.read_parquet(need[2])
    trees = {}
    for fam in dict.fromkeys(f for _, f in spec.pairs):
        sel = select_plants(plants, fam)
        trees[fam] = cKDTree(unit_xyz(sel["lon"], sel["lat"])) if len(sel) else None
    grid, tr = land_grid(cfg, cells)
    land, land_in = land_within(grid, tr, trees, rad)
    mun_pts, rg_pts = unit_points(cfg)
    attrs = pd.read_parquet(cfg.out("interim", "municipalities.parquet"),
                            columns=["geocode", "cd_rgint", "area_km2_ibge", "is_municipality"])
    attrs = attrs[attrs["is_municipality"]].set_index("geocode")
    # the land grid against the IBGE areas
    rel = (land / 100) / attrs["area_km2_ibge"].reindex(land.index) - 1
    ck.add("scale_land_grid_matches_ibge_area", float(rel.abs().median()) <= 0.02,
           f"median |cell land area / IBGE area − 1| = {rel.abs().median():.2%} over {len(land)} municipalities "
           f"({len(attrs) - len(land)} without a cell)", round(float(rel.abs().median()), 4), severity="warn")
    rg_of = attrs["cd_rgint"]
    land_rg = land.groupby(rg_of.reindex(land.index).to_numpy()).sum()

    nat_rows, unit_rows = [], []
    for crop, fam in spec.pairs:
        tag = pair_tag(crop, fam)
        mu = units[(units["level"] == "municipality") & (units["pair"] == tag)].drop_duplicates("unit")
        crop_mun = pd.Series(mu["crop_area_ha"].to_numpy(), index=mu["unit"].astype("int64"))
        crop_rg = crop_mun.groupby(rg_of.reindex(crop_mun.index).to_numpy()).sum()
        total, pix = pixel_shares(hist, distance, tag, rad, spec.max_km)
        for r, v in pix.items():
            nat_rows.append({"pair": tag, "crop": crop, "family": fam, "radius_km": r, "resolution": "pixel",
                             "rule": "pixel", "within_ha": v, "crop_area_ha": total})
        lin = land_in[fam]
        lin_rg = lin.groupby(rg_of.reindex(lin.index).to_numpy()).sum()
        for res, crop_ha, ld, li, pts in (("municipality", crop_mun, land, lin, mun_pts),
                                          ("rgint", crop_rg, land_rg, lin_rg, rg_pts)):
            t = unit_table(crop_ha, ld, li, pts, trees[fam], rad)
            if t.attrs.get("no_land_units"):
                ck.add(f"scale_{tag}_{res}_units_without_land_cell", True,
                       f"{t.attrs['no_land_units']} units smaller than a cell use their centroid",
                       t.attrs["no_land_units"])
            t = t.assign(pair=tag, crop=crop, family=fam, resolution=res)
            unit_rows.append(t)
            g = t.groupby(["rule", "radius_km"], as_index=False)[["within_ha", "crop_area_ha"]].sum()
            nat_rows += [{"pair": tag, "crop": crop, "family": fam, "radius_km": x.radius_km, "resolution": res,
                          "rule": x.rule, "within_ha": x.within_ha, "crop_area_ha": x.crop_area_ha}
                         for x in g.itertuples()]
    nat = pd.DataFrame(nat_rows)
    nat["share"] = nat["within_ha"] / nat["crop_area_ha"]
    return {"scale": nat, "scale_units": pd.concat(unit_rows, ignore_index=True)}


def check(ck, nat: pd.DataFrame, units: pd.DataFrame, reference: dict | None = None):
    """Convergence, monotonicity, totals; the headline gaps; the exploratory x05 against `reference`
    ({pair: {resolution: share at 30 km}})."""
    inf = nat[np.isinf(nat["radius_km"])]
    bad = inf[~np.isclose(inf["share"], 1.0, atol=1e-9)]
    ck.add("scale_converges_to_100", bad.empty, "every resolution and rule reaches 100 % as the radius grows"
           if bad.empty else f"{len(bad)} combinations do not: {bad[['pair', 'resolution', 'rule']].values.tolist()}")
    # tolerance relative to the crop area: the same sum added in another order differs in the last digits
    mono = pd.Series({k: bool((np.diff(g["within_ha"].to_numpy()) >= -1e-9 * g["crop_area_ha"].max()).all())
                      for k, g in nat.sort_values("radius_km").groupby(["pair", "resolution", "rule"])})
    ck.add("scale_within_grows_with_radius", bool(mono.all()), f"{int((~mono).sum())} series do not", int((~mono).sum()))
    tot = nat.groupby("pair")["crop_area_ha"].agg(["min", "max"])
    ck.add("scale_same_land_at_every_resolution", bool(np.allclose(tot["min"], tot["max"], rtol=1e-9)),
           "the crop area is the same at the pixel, the municipality and the region")
    at30 = nat[nat["radius_km"] == 30].set_index(["pair", "resolution", "rule"])["share"]
    for pair in nat["pair"].unique():
        px = at30[(pair, "pixel", "pixel")]
        parts = [f"pixel {px:.1%}"]
        vals = {}
        for res in ("municipality", "rgint"):
            for rule in RULES:
                v = at30[(pair, res, rule)]
                vals[f"{res}|{rule}"] = round(float(v), 4)
                if rule != "representative_point":
                    parts.append(f"{res} {rule} {v:.1%} ({100 * (v - px):+.1f} pp)")
        ck.add(f"scale_{pair}_30km", True, "; ".join(parts), vals | {"pixel": round(float(px), 4)})
    for pair, ref in (reference or {}).items():
        if pair not in set(nat["pair"]):
            continue
        diffs = {}
        for res, v in ref.items():
            key = (pair, res, "pixel" if res == "pixel" else "representative_point")
            if key in at30.index:
                diffs[res] = round(100 * (float(at30[key]) - v), 2)
        ck.add(f"scale_{pair}_vs_exploratory_x05", True, f"30 km, new − exploratory (pp): {diffs}", diffs)


def run(cfg, ck=None) -> dict[str, Path]:
    """`bbp scale`."""
    from bbp.checks import CheckLog

    ck = ck if ck is not None else CheckLog("scale")
    t = build(cfg, ck)
    if not t:
        return {}
    ref = ((cfg.section("plants") or {}).get("scale") or {}).get("exploratory_x05")
    check(ck, t["scale"], t["scale_units"], ref)
    out = cfg.data_dir / "processed"
    written = {}
    for name, df in t.items():
        p = out / f"reach_{name}.parquet"
        df.to_parquet(p)
        written[name] = p
    return written


__all__ = ["RULES", "build", "check", "run", "unit_table", "pixel_shares", "land_within"]
