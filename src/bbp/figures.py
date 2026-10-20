"""Figures of the paper in the NIPE cartographic style (bbp.figstyle), at Geocarto sizes.

    bbp figures [--only 1 5]   ->  data/<paper.out_dir>/figures/fig<NN>_<name>.{pdf,svg,png,tif}
                                   data/<paper.out_dir>/captions.md

Every figure is a millimetre canvas; maps are drawn at a fixed scale (metres per mm) in EPSG:5880 with
the NIPE strip (north arrow, scale bar, credits) and a structured legend. One figure, one message.

 1  study area at three resolutions: MapBiomas land cover (display groups, bbp.landcover), the 133
    intermediate regions, details with the municipal mesh (NIPE device); variant per year (fig01_..._2025)
 2  workflow: sources by agency -> stages -> products, and what the invariant checks caught
 3  national area per crop, observed vs declared, 2008–2024
 4  agreement vs scale (share within ±25 %, by area and by number of units)                [C1]
 5  municipal log-ratio maps per crop (one map scale), one-source-only as its own hatched category
 6  example municipalities on the Col. 11 coverage raster (rule-based candidates)
 7  H2 variance partition: predictors / intermediate region / residual, Moran's I           [C2]
 8  H2 standardised effects of the predictors
 9  H3: municipal source effect against the conversion-factor envelope                     [C3]
10  C4 (droppable): source effect by decile of municipal N1
S1  detail of Fig 5 for SP and MG (dashed detail box and connectors);  S2 rank shift;  S3 scatters;
S4  Lin's CCC vs scale.
"""

from __future__ import annotations

import json
import logging
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from bbp import STAGES
from bbp import figstyle as fs
from bbp.analyse import NUMERIC
from bbp.cartography import MAP_CRS, base_layers, boundaries, detail_box, extent, state_labels
from bbp.checks import CheckLog
from bbp.config import Config
from bbp.figstyle import Canvas, map_frame
from bbp.paper import (
    PREDICTOR_LABELS,
    SCALE_LABELS,
    load_tables,
    out_dir,
    source_effect_by_decile,
    summary_years,
)

log = logging.getLogger("bbp")

# log-ratio classes, symmetric on the log scale: ±10, 25, 50, 100 %
RATIO_EDGES = [1 / 2, 1 / 1.5, 1 / 1.25, 1 / 1.1, 1.1, 1.25, 1.5, 2.0]
DEC_ONLY, OBS_ONLY = -1, 9          # class codes for "only one source reports the crop"
ONE_SOURCE = {DEC_ONLY: ("white", "//////"), OBS_ONLY: ("white", "......")}
ONE_SOURCE_LABELS = {DEC_ONLY: "PAM only (no MapBiomas area)", OBS_ONLY: "MapBiomas only (no PAM area)"}
SECOND_CROP = "corn_2nd"
FACTOR_TICKS = [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100]
DATA_CREDIT = "MapBiomas Col. 11; IBGE PAM (this study)"


def nice_scale(m_per_mm: float, step: float = 1000.0) -> float:
    """Round a map scale (metres per mm) up to a whole `step`, so the figure states a round 1:N."""
    return math.ceil(m_per_mm / step) * step


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------
class _Data:
    """Processed tables (restricted to the profile) and the base layers, loaded once."""

    def __init__(self, cfg: Config, ck: CheckLog):
        self.cfg, self.ck, self._t, self._base = cfg, ck, {}, None
        self.facts: dict = {}           # numbers a figure computed for its caption

    def get(self, name: str) -> pd.DataFrame | None:
        if name not in self._t:
            self._t[name] = load_tables(self.cfg, self.ck, [name]).get(name)
        return self._t[name]

    def base(self):
        """(municipal polygons, states, country, extent) in EPSG:5880."""
        if self._base is None:
            g = map_geometry(self.cfg, self.ck)
            states, country = base_layers(g)
            self._base = (g, states, country, extent(country))
        return self._base

    def explore(self):
        """The context of bbp explore (plants, reach and demo tables), built on first use."""
        if getattr(self, "_ctx", None) is None:
            from bbp.explore import Ctx

            self._ctx = Ctx(self.cfg, self.ck, data=self)
        return self._ctx


def ratio_classes() -> tuple[np.ndarray, list, list[str]]:
    """log-ratio bin edges, one colour per class (blue: MapBiomas below PAM, red: above, grey: ±10 %), labels."""
    from matplotlib import colormaps

    edges = np.log(RATIO_EDGES)
    cmap = colormaps["RdBu_r"]
    colors = [cmap(x) for x in (0.0, 0.14, 0.27, 0.38)] + ["#d4d4d4"] + [cmap(x) for x in (0.62, 0.73, 0.86, 1.0)]
    pct = ["−50 %", "−33 %", "−20 %", "−9 %", "+10 %", "+25 %", "+50 %", "+100 %"]
    labels = [f"< {pct[0]}"] + [f"{a} to {b}" for a, b in zip(pct[:-1], pct[1:], strict=True)] + [f"> {pct[-1]}"]
    labels[4] = "±10 % (agree)"
    return edges, colors, labels


def classify(values: pd.Series, edges: np.ndarray) -> pd.Series:
    """Class 0..len(edges) for finite log-ratios; DEC_ONLY / OBS_ONLY for −inf / +inf; NaN stays NaN."""
    out = pd.Series(np.nan, index=values.index)
    v = values.to_numpy(float)
    finite = np.isfinite(v)
    out[finite] = np.digitize(v[finite], edges)
    out[v == -np.inf] = DEC_ONLY
    out[v == np.inf] = OBS_ONLY
    return out


def municipal_log_ratio(units: pd.DataFrame, crop: str, years: list[int], min_area: float) -> pd.Series:
    """Median over `years` of ln(A_obs / A_dec) per municipality, ±inf where one source has none.

    Municipality-years where both sources are below `min_area` are not scored (as in the metrics).
    """
    u = units[(units["crop"].astype(str) == crop) & units["year"].isin(years)]
    u = u[np.maximum(u["a_obs_ha"], u["a_dec_ha"]) >= max(min_area, 1e-9)]
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(u["a_obs_ha"].to_numpy(float) / u["a_dec_ha"].to_numpy(float))
    s = pd.Series(lr, index=u["unit"].astype("int64").to_numpy())
    return s.groupby(level=0).median()


def map_geometry(cfg: Config, ck: CheckLog):
    """Municipal polygons, simplified (cached in data/interim) and projected to Brazil Polyconic."""
    import geopandas as gpd
    import shapely

    tol = float(cfg.section("paper").get("simplify_deg", 0.005))
    src = cfg.data_dir / "interim" / "municipalities_geom.parquet"
    cache = cfg.data_dir / "interim" / f"municipalities_geom_simplified_{tol:g}.parquet"
    if not (cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime):
        log.info("  simplifying the municipal mesh (%g°) for the maps...", tol)
        g = gpd.read_parquet(src, columns=["geocode", "uf", "cd_rgint", "is_municipality", "geometry"])
        g = g[g["is_municipality"]].reset_index(drop=True)
        if hasattr(shapely, "coverage_simplify"):  # shared edges stay shared: no slivers between neighbours
            g["geometry"] = shapely.coverage_simplify(g.geometry.values, tol)
        else:
            g["geometry"] = g.geometry.simplify(tol, preserve_topology=True)
        g = g[~g.geometry.is_empty]
        g.to_parquet(cache, index=False)
    g = gpd.read_parquet(cache)
    ck.add("map_geometry_municipalities", 5565 <= len(g) <= 5575, f"{len(g)} polygons", len(g), severity="warn")
    return g.to_crs(MAP_CRS)


def _layer_path(cfg: Config, ck: CheckLog, key: str) -> Path | None:
    """Path of an optional context layer (`paper.layers.<key>`, relative to raw_root); None if absent."""
    rel = (cfg.section("paper").get("layers") or {}).get(key)
    if not rel:
        ck.add(f"layer_{key}", False, "not configured in paper.layers", severity="warn")
        return None
    path = cfg.raw_root / rel
    return path if ck.add(f"layer_{key}", path.exists(), str(path), severity="warn") else None


def _layer(cfg: Config, ck: CheckLog, key: str, simplify_m: float = 2000.0):
    """An optional context layer (`paper.layers.<key>`), projected and simplified; None if not configured."""
    import geopandas as gpd

    path = _layer_path(cfg, ck, key)
    if path is None:
        return None
    g = gpd.read_file(path).to_crs(MAP_CRS)
    g["geometry"] = g.geometry.simplify(simplify_m)
    return g


def _plot_classes(ax, gdf, classes: pd.Series, colors: list, edge: float = 0.0, edgecolor: str = "white"):
    """Diverging classes, then the two one-source-only categories as hatched patterns (one dissolved
    shape per category: thousands of hatched paths are slow to render)."""
    import geopandas as gpd
    import shapely

    cls = gdf["geocode"].map(classes)
    for k, col in enumerate(colors):
        sel = gdf[cls == k]
        if len(sel):
            sel.plot(ax=ax, color=col, linewidth=edge, edgecolor=edgecolor, rasterized=True, zorder=1)
    for code, (fc, hatch) in ONE_SOURCE.items():
        sel = gdf[cls == code]
        if len(sel):
            merged = gpd.GeoSeries([shapely.union_all(sel.geometry.values)], crs=gdf.crs)
            merged.plot(ax=ax, facecolor=fc, edgecolor=fs.MUTED, hatch=hatch, linewidth=0, rasterized=True, zorder=1)


def _ratio_legend(cv: Canvas, x: float, y_top: float, years: list[int], min_area: float):
    """The log-ratio legend in three columns (5 + 4 classes, then the categories that cannot be compared)."""
    _, colors, labels = ratio_classes()
    rows = [("patch", c, lab) for c, lab in zip(colors[::-1], labels[::-1], strict=True)]
    first_row = y_top - 3.9 - fs.ROW
    cv.legend(x, y_top, [(f"A$_{{obs}}$ / A$_{{dec}}$, median {years[0]}–{years[-1]}", rows[:5])])
    cv.legend(x + 40, first_row, [(None, rows[5:])], header=None)
    cv.legend(x + 78, y_top - 3.9, [("Not comparable", [
        ("patch", {"fc": fc, "hatch": h, "ec": fs.MUTED}, ONE_SOURCE_LABELS[k]) for k, (fc, h) in ONE_SOURCE.items()]
        + [("patch", {"fc": "white", "ec": fs.HAIR}, f"Neither source ≥ {min_area:g} ha")])], header=None)


def _factor_axis(ax, axis: str = "x", lo: float = 0.0, hi: float = 3.0, title: str = ""):
    """An |ln| axis labelled as the multiplicative difference between the two sources (×1.5, ×2, ...)."""
    span = hi - lo
    keep = [f for f in FACTOR_TICKS if lo <= np.log(f) <= hi]
    ticks, labels, last = [], [], 0.0
    for f in keep:                     # drop ticks closer than 6 % of the axis to the previous one
        if np.log(f) - last >= 0.06 * span:
            ticks.append(np.log(f))
            labels.append(f"×{f:g}")
            last = np.log(f)
    (ax.set_xticks if axis == "x" else ax.set_yticks)([0, *ticks], ["×1", *labels])
    if title:
        (ax.set_xlabel if axis == "x" else ax.set_ylabel)(title)


def _sig(p: float) -> str:
    return "*" if p < 0.05 else ""


def _casing(crop: str, lw: float):
    import matplotlib.patheffects as pe

    return [pe.Stroke(linewidth=lw + 0.7, foreground=fs.MUTED), pe.Normal()] if crop in fs.LIGHT else None


def _chart_axes(cv: Canvas, x, y, w, h, grid_y: bool = False):
    a = cv.ax(x, y, w, h)
    if grid_y:
        a.grid(axis="y", color=fs.FAINT, lw=0.4)
    return a


# ---------------------------------------------------------------------------
# Fig 1 — study area: one landscape at three resolutions
# ---------------------------------------------------------------------------
# (a) Brazil: MapBiomas land cover (pixel view) with the 133 intermediate regions (regional view);
# (b..) details with the municipal mesh (municipal view), joined to (a) by the NIPE detail device.
# Right of (a): Brazil and Europe on one equal-area scale, then the legend. Biomes: names only, no lines.
BIOME_NUDGE = {"Pampa": (-230e3, -150e3), "Cerrado": (0, 90e3)}   # metres, off the UF labels and connectors
LC_FACTORS = (16, 32)               # display grids: ~480 m (details) and ~960 m (national map)
EU27 = ("AUT BEL BGR HRV CYP CZE DNK EST FIN FRA DEU GRC HUN IRL ITA LVA LTU LUX MLT NLD POL PRT ROU SVK SVN "
        "ESP SWE").split()
PRINT_PX_PER_MM = 600 / 25.4        # raster grids are drawn at about one cell per printed pixel


def _fig1_spec(cfg: Config) -> dict:
    return cfg.section("paper").get("fig1") or {}


def coverage_sources(cfg: Config, year: int):
    """The 30 m coverage of `year`: `paper.fig1.coverage.<year>` (relative to raw_root) or the file / tiles
    in inputs.coverage_tif_dir. None when absent."""
    from bbp.ingest.border_share import find_year_sources

    over = _fig1_spec(cfg).get("coverage") or {}
    rel = over.get(year) or over.get(str(year))
    if rel:
        p = cfg.raw_root / rel
        return p if p.exists() else None
    folder = cfg.input_path("coverage_tif_dir")
    if not folder.is_dir():
        return None
    pattern = cfg.section("border_share").get("tif_pattern", "brazil_coverage-col11_{year}.tif")
    return find_year_sources(folder, pattern, [year]).get(year)


def second_crop_source(cfg: Config, year: int) -> Path | None:
    from bbp.ingest.second_crop import find_rasters

    folder = cfg.input_path("second_crop_tif_dir")
    if not folder.is_dir():
        return None
    return find_rasters(folder, cfg.section("second_crop")["tif_pattern"], [year]).get(year)


def landcover_display(cfg: Config, ck: CheckLog, year: int, required: bool = True) -> dict | None:
    """The land-cover display rasters of `year` ({factor: path}, sidecar) in data/interim/landcover,
    aggregated from the 30 m coverage on first use (bbp.landcover: ~2 min a year, resumable).
    Checked against the Collection 11 statistics when the ingest left them in data/interim."""
    from bbp import landcover as lc

    out = cfg.data_dir / "interim" / "landcover"
    name = f"col{cfg.section('mapbiomas').get('collection', 11)}_{year}"
    paths = lc.output_paths(out, name, LC_FACTORS)
    meta_path = out / f"{name}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    if not (meta and meta.get("complete") and all(p.exists() for p in paths.values())):
        cov = coverage_sources(cfg, year)
        if cov is None:
            ck.add(f"fig01_landcover_{year}", False, "no coverage raster for this year: no land cover drawn",
                   severity="warn" if required else "info")
            return None
        log.info("  aggregating the %s coverage to display groups (once, cached in %s)...", year, out)
        meta = lc.aggregate(cov, out, name, factors=LC_FACTORS, second_crop=second_crop_source(cfg, year))
    ck.add(f"fig01_landcover_{year}", True, f"{name}, second crop: {meta['params']['second_crop'] or 'none'}",
           meta["params"]["second_crop"])
    stats = cfg.data_dir / "interim" / "mapbiomas_coverage_municipal.parquet"
    if stats.exists():
        t = pd.read_parquet(stats, columns=["class_id", "year", "area_ha"])
        nat = t[t["year"] == year].groupby("class_id")["area_ha"].sum()
        dev = {int(c): lc.class_area(meta, int(c)) / nat[c] - 1 for c in (39, 20, 46, 47, 40, 62) if nat.get(c, 0) > 0}
        worst = max(dev.values(), key=abs) if dev else 0.0
        ck.add(f"fig01_landcover_{year}_matches_statistics", abs(worst) < 1e-3,
               f"display groups vs Col. 11 statistics, worst crop class {worst:+.4%}", worst, severity="warn")
    return {"paths": paths, "meta": meta, "year": year}


def build_landcover(cfg: Config, ck: CheckLog, years: list[int] | None = None,
                    budget: float | None = None) -> dict[int, dict]:
    """`bbp landcover`: aggregate the coverage of each year (default: paper.fig1.year and its variants) to the
    display rasters of Fig 1 (and, for paper.fig1.year, the per-group pixel counts); with `budget` (seconds)
    a long pass stops early and resumes on the next call."""
    from bbp import landcover as lc

    spec = _fig1_spec(cfg)
    primary = int(spec.get("year", cfg.years[-1]))
    years = years or [primary] + [int(y) for y in spec.get("variants") or []]
    out_dir = cfg.data_dir / "interim" / "landcover"
    done = {}
    for year in years:
        cov = coverage_sources(cfg, year)
        if not ck.add(f"landcover_{year}_coverage", cov is not None, f"coverage raster of {year}", severity="warn"):
            continue
        name = f"col{cfg.section('mapbiomas').get('collection', 11)}_{year}"
        meta = lc.aggregate(cov, out_dir, name, factors=LC_FACTORS, second_crop=second_crop_source(cfg, year),
                            time_budget=budget, counts=year == primary)   # counts: bbp.plants reach analyses
        ck.add(f"landcover_{year}_complete", meta["complete"],
               f"{meta['bands_done']} bands, {meta['elapsed_s']:.0f} s" + ("" if meta["complete"] else
                                                                         " — run again to continue"),
               severity="warn")
        if meta["complete"]:
            landcover_display(cfg, ck, year)              # the statistics check
        done[year] = meta
    return done


def _to_map(lon_lat_bounds, crs: str = MAP_CRS) -> tuple[float, float, float, float]:
    """(lon0, lat0, lon1, lat1) -> (x0, x1, y0, y1) in `crs` (densified edges)."""
    from rasterio.warp import transform_bounds

    x0, y0, x1, y1 = transform_bounds("EPSG:4674", crs, *lon_lat_bounds, densify_pts=101)
    return x0, x1, y0, y1


def _landcover_layer(a, disp: dict | None, extent_m, scale: float, lut: np.ndarray, z: float = 1.0,
                     sampling: str = "nearest", focus=None) -> bool:
    """Draw the land cover of `disp` on map axes at `scale` (m/mm), about one grid cell per printed pixel.
    sampling "nearest": each printed pixel shows the ~480 m cell under it (a systematic sample, so a crop
    covering 30 % of an area colours ~30 % of its pixels); "mode": the most frequent cell (crisper, but
    minority crops fade at small scales)."""
    from bbp.cartography import draw_landcover, landcover_on_map

    if disp is None:
        return False
    res_m = scale / PRINT_PX_PER_MM
    if sampling == "mode":
        factor = max([f for f in LC_FACTORS if f * 30.0 <= res_m] or [LC_FACTORS[0]])
    else:
        factor = LC_FACTORS[0]
    res_m = max(res_m, factor * 30.0)
    codes, grid = landcover_on_map(disp["paths"][factor], extent_m, res_m,
                                   resampling="mode" if sampling == "mode" and res_m > 1.5 * factor * 30.0
                                   else "nearest")
    if focus is None:
        draw_landcover(a, codes, grid, lut, z=z)
    else:                                   # colour inside `focus`, light grey context around it
        from bbp.cartography import ghost_lut

        draw_landcover(a, codes, grid, ghost_lut(lut), z=z)
        draw_landcover(a, codes, grid, lut, z=z + 0.01, clip=focus)
    return True


def _connect_boxes(cv: Canvas, box, panel):
    """NIPE connectors between a detail box and its panel (both (x0, y0, x1, y1) in mm), on the sides that
    face each other."""
    bx0, by0, bx1, by1 = box
    px0, py0, px1, py1 = panel
    if px1 <= bx0:                                    # panel to the left
        pairs = [((bx0, by1), (px1, py1)), ((bx0, by0), (px1, py0))]
    elif px0 >= bx1:                                  # to the right
        pairs = [((bx1, by1), (px0, py1)), ((bx1, by0), (px0, py0))]
    elif py1 <= by0:                                  # below
        pairs = [((bx0, by0), (px0, py1)), ((bx1, by0), (px1, py1))]
    else:                                             # above
        pairs = [((bx0, by1), (px0, py0)), ((bx1, by1), (px1, py0))]
    for p0, p1 in pairs:
        cv.connect(p0, p1)


def _text_box(x, y, text, size_pt, ha, va, pad=0.25):
    """Approximate extent (x0, y0, x1, y1) in mm of a one-line label (Arial-metric: ~0.53 em per glyph)."""
    w = len(text) * size_pt * 0.3528 * 0.53
    h = size_pt * 0.3528 * 0.9
    x0 = x - w if ha == "right" else x - w / 2 if ha == "center" else x
    y0 = y - h if va == "top" else y - h / 2 if va == "center" else y
    return x0 - pad, y0 - pad, x0 + w + pad, y0 + h + pad


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def city_points(g4674, geocodes, lc_path: Path | None) -> dict[int, tuple[float, float]]:
    """Lon/lat of each city: the centre of the urban cells (display group 'built') inside its municipality,
    or the polygon's representative point where the raster has none (or is absent)."""
    from bbp import landcover as lc

    ds = None
    if lc_path is not None and Path(lc_path).exists():
        import rasterio

        ds = rasterio.open(lc_path)
    out = {}
    try:
        for gc in geocodes:
            geom = g4674.loc[gc]
            p = geom.representative_point()
            xy = (p.x, p.y)
            if ds is not None:
                from rasterio.features import geometry_mask
                from rasterio.transform import xy as cell_xy
                from rasterio.windows import from_bounds

                win = from_bounds(*geom.bounds, transform=ds.transform).round_offsets().round_lengths()
                arr = ds.read(1, window=win, boundless=True, fill_value=0)
                tr = ds.window_transform(win)
                inside = ~geometry_mask([geom], out_shape=arr.shape, transform=tr)
                rows, cols = np.nonzero(inside & (arr == lc.CODE["built"]))
                if len(rows):
                    xs, ys = cell_xy(tr, rows, cols)
                    xy = (float(np.mean(xs)), float(np.mean(ys)))
            out[gc] = xy
    finally:
        if ds is not None:
            ds.close()
    return out


def _place_cities(cv: Canvas, pts_mm: list[tuple[str, float, float]], frame, size_pt: float, taken=None):
    """White dots and halo labels; each label at the first of right / left / above / below that stays in
    `frame` (x0, y0, x1, y1, mm) and clear of the labels and dots already placed."""
    import matplotlib.patheffects as pe

    taken = list(taken or [])
    dots = [(x - 0.6, y - 0.6, x + 0.6, y + 0.6) for _, x, y in pts_mm]
    for name, x, y in sorted(pts_mm, key=lambda t: -len(t[0])):
        cands = [(x + 0.9, y, "left", "center"), (x - 0.9, y, "right", "center"),
                 (x, y + 0.8, "center", "bottom"), (x, y - 0.8, "center", "top"),
                 (x + 0.7, y + 0.6, "left", "bottom"), (x - 0.7, y - 0.6, "right", "top")]
        best = None
        for tx, ty, ha, va in cands:
            bb = _text_box(tx, ty, name, size_pt, ha, va)
            inside = frame[0] <= bb[0] and bb[2] <= frame[2] and frame[1] <= bb[1] and bb[3] <= frame[3]
            clear = not any(_overlap(bb, t) for t in taken) and not any(
                _overlap(bb, d) for d in dots if not (d[0] < x < d[2] and d[1] < y < d[3]))
            if inside and clear:
                best = (tx, ty, ha, va, bb)
                break
        if best is None:                                  # nothing clear: right of the dot anyway
            tx, ty, ha, va = cands[0]
            best = (tx, ty, ha, va, _text_box(tx, ty, name, size_pt, ha, va))
        tx, ty, ha, va, bb = best
        taken.append(bb)
        cv.text(tx, ty, name, fontsize=size_pt, ha=ha, va=va, zorder=21,
                path_effects=[pe.withStroke(linewidth=1.3, foreground="white")])
    for _, x, y in pts_mm:
        cv.ov.scatter([x], [y], s=4.2, c="white", edgecolors=fs.INK, linewidths=0.45, zorder=22)
    return taken


def _detail_specs(cfg: Config, attrs: pd.DataFrame, states) -> list[dict]:
    """Detail windows of Fig 1 (paper.fig1.details): a state (`uf`) or lon/lat `bounds`; the cities are the
    seats of the intermediate regions inside (the municipality named after its region) plus `cities`."""
    out = []
    for d in _fig1_spec(cfg).get("details") or []:
        if d.get("uf"):
            sel = states[states["uf"] == d["uf"]]
            if sel.empty:
                continue
            x0, y0, x1, y1 = sel.total_bounds
            pad = 0.03 * max(x1 - x0, y1 - y0)
            win = (x0 - pad, x1 + pad, y0 - pad, y1 + pad)
            in_win = attrs["uf"] == d["uf"]
        else:
            win = _to_map(d["bounds"])
            in_win = None
        out.append({**d, "window": win, "in_win": in_win})
    return out


def _region_seats(attrs: pd.DataFrame, rgint_codes, extra: list[str], uf: str | None) -> list[int]:
    """Geocodes of the municipality that names each intermediate region in `rgint_codes`, plus `extra` names."""
    mun = attrs[attrs["is_municipality"]] if "is_municipality" in attrs else attrs
    seats = []
    for rg in sorted(set(rgint_codes)):
        r = mun[mun["cd_rgint"] == rg]
        hit = r[r["name"] == r["nm_rgint"]] if "nm_rgint" in r else r.iloc[0:0]
        if len(hit) == 1:
            seats.append(int(hit["geocode"].iloc[0]))
    for name in extra:
        hit = mun[(mun["name"] == name) & ((mun["uf"] == uf) if uf else True)]
        if len(hit) == 1:
            seats.append(int(hit["geocode"].iloc[0]))
    return list(dict.fromkeys(seats))


def _equal_area(gdf, lon0: float, lat0: float):
    return gdf.to_crs(f"+proj=laea +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")


def _geodesic_km2(geom) -> float:
    from pyproj import Geod

    return abs(Geod(ellps="WGS84").geometry_area_perimeter(geom)[0]) / 1e6


def fig01_study_area(cfg: Config, ck: CheckLog, data: _Data, year: int | None = None, host=None):
    import geopandas as gpd
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle
    from shapely.geometry import box

    from bbp import landcover as lc
    from bbp.cartography import cased_lines, landcover_rgba

    spec = _fig1_spec(cfg)
    main_year = int(spec.get("year", cfg.years[-1]))
    year = int(year or main_year)
    variant = year != main_year
    disp = landcover_display(cfg, ck, year, required=not variant)
    if variant and disp is None:
        return None                                       # a variant is drawn only from its own land cover
    g, states, country, ext = data.base()
    attrs = data.get("municipality_attributes")
    lut = landcover_rgba(fs.LANDCOVER_COLORS)
    halo = [pe.withStroke(linewidth=1.2, foreground="white", alpha=0.9)]
    present = {k for k, v in lc.area_by_group(disp["meta"]).items() if v > 0} if disp else set()

    # ---- layout (mm) -----------------------------------------------------------------------------
    FW = fs.FULL_W
    S = nice_scale((ext[1] - ext[0]) / float(spec.get("map_width_mm", 86.0)))
    MW, MH = (ext[1] - ext[0]) / S, (ext[3] - ext[2]) / S
    dets = []
    for d in _detail_specs(cfg, attrs, states):
        w = d["window"]
        if w[1] <= ext[0] or w[0] >= ext[1] or w[3] <= ext[2] or w[2] >= ext[3]:
            ck.add(f"fig01_detail_{d['key']}", False, "window outside the map: detail left out", severity="warn")
            continue
        dets.append(d)
    bottom = [d for d in dets if d.get("place", "bottom") == "bottom"]
    corner = [d for d in dets if d.get("place") == "corner"]
    big_w = FW - 2.0                            # bottom details span the width; the legend sits right of (a)
    TITLE = 4.6
    for d in bottom:
        w = d["window"]
        d["scale"] = nice_scale(max((w[1] - w[0]) / big_w, (w[3] - w[2]) / float(spec.get("detail_max_h_mm", 60))),
                                500.0)
        d["size"] = ((w[1] - w[0]) / d["scale"], (w[3] - w[2]) / d["scale"])
    HB = (max((d["size"][1] for d in bottom), default=0.0) + TITLE + 6.5) if bottom else 0.0
    yA = HB + 1.0 + fs.STRIP                    # bottom of map (a)
    FH = yA + MH + TITLE + 0.5
    cv = fs.canvas(FW, FH, host)
    mx = 1.0
    a = cv.map_ax(mx, yA, ext, S)

    def mm(x, y):                                # map (a) metres -> canvas mm
        return mx + (x - ext[0]) / S, yA + (y - ext[2]) / S

    # ---- (a) Brazil --------------------------------------------------------------------------------
    if not _landcover_layer(a, disp, ext, S, lut, sampling=spec.get("national_sampling", "nearest")):
        country.plot(ax=a, color="#efefef", linewidth=0, zorder=1)
    regions = g.dissolve(by="cd_rgint", as_index=False)
    cased_lines(a, regions, lw=0.16, color="#7d7d7d", casing_lw=0.55, z=3)
    states.boundary.plot(ax=a, color="#4f4f4f", linewidth=0.32, zorder=3.4)
    country.boundary.plot(ax=a, color=fs.COUNTRY_EDGE, linewidth=0.55, zorder=3.5)
    biomes = _layer(cfg, ck, "biomes_shp", simplify_m=35000)
    if biomes is not None:
        name_col = next(c for c in ("Bioma", "NAME", "name") if c in biomes)
        parts = biomes.explode(index_parts=False)                    # enclaves < 10,000 km² tangle the map
        biomes = parts[parts.area > 1e10].dissolve(by=name_col, as_index=False)
        for _, b in biomes.iterrows():
            p = b.geometry.representative_point()
            dx, dy = BIOME_NUDGE.get(b[name_col], (0, 0))
            a.text(p.x + dx, p.y + dy, b[name_col], fontsize=fs.FS_TITLE, style="italic", ha="center",
                   va="center", color=fs.INK, zorder=14, path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])
    state_labels(a, states, halo=True)
    ocean = spec.get("ocean_label", [-41.5, 2.6])
    from pyproj import Transformer

    to_map = Transformer.from_crs("EPSG:4674", MAP_CRS, always_xy=True)
    ox, oy = to_map.transform(*ocean)
    a.text(ox, oy, "Atlantic\nOcean", fontsize=fs.FS_TITLE, style="italic", color="#5a7f99", ha="center",
           va="center", linespacing=1.1, zorder=14)
    map_frame(a, ext)
    region_count = int(regions["cd_rgint"].nunique())
    cv.text(mx, FH - 0.6, f"(a) Brazil, {year}: land cover and the {region_count} intermediate regions",
            fontsize=fs.FS_TITLE + 0.4, fontweight="bold", va="top")

    # ---- detail panels ----------------------------------------------------------------------------------
    g4674 = gpd.read_parquet(cfg.data_dir / "interim" / "municipalities_geom.parquet",
                             columns=["geocode", "geometry"]).set_index("geocode")["geometry"]
    letters = iter("bcdefg")
    drawn = []                                   # letters of the detail panels drawn (legend: municipal mesh)
    lc16 = disp["paths"][LC_FACTORS[0]] if disp else None

    def draw_detail(d, px, py, dsc):
        win = d["window"]
        dw, dh = (win[1] - win[0]) / dsc, (win[3] - win[2]) / dsc
        dax = cv.map_ax(px, py, win, dsc, z=3)
        dax.patch.set_alpha(1)
        dax.patch.set_facecolor("white")
        focus = states[states["uf"] == d["uf"]].union_all() if d.get("uf") else None   # the state in colour
        if not _landcover_layer(dax, disp, win, dsc, lut, focus=focus):
            country.plot(ax=dax, color="#efefef", linewidth=0, zorder=1)
        gi = g.cx[win[0]:win[1], win[2]:win[3]]
        gi.boundary.plot(ax=dax, color="white", linewidth=0.12, zorder=2.5)
        ri = regions.cx[win[0]:win[1], win[2]:win[3]]
        cased_lines(dax, ri, lw=0.3, color="#5c5c5c", casing_lw=0.9, z=3)
        states.boundary.plot(ax=dax, color="#3d3d3d", linewidth=0.45, zorder=3.4)
        country.boundary.plot(ax=dax, color=fs.COUNTRY_EDGE, linewidth=0.6, zorder=3.5)
        map_frame(dax, win)
        fs.detail_border(dax)
        letter = next(letters)
        drawn.append((letter, d))
        # cities: seats of the intermediate regions in the window, plus the named ones
        cen = g[["geocode", "cd_rgint", "geometry"]].copy()
        cen["pt"] = cen.geometry.representative_point()
        inside = cen[cen["pt"].within(box(win[0], win[2], win[1], win[3]))]
        seats = _region_seats(attrs, inside["cd_rgint"].dropna().astype(int), d.get("cities") or [],
                              d.get("city_uf") or d.get("uf"))
        seats = [s for s in seats if s in g4674.index]
        if d.get("uf"):
            seats = [s for s in seats if s in set(attrs.loc[attrs["uf"] == d["uf"], "geocode"])]
        if len(seats) < len(d.get("cities") or []):
            ck.add(f"fig01_detail_{d['key']}_cities", False, "a named city was not found", severity="warn")
        ll = city_points(g4674, seats, lc16)
        names = attrs.set_index("geocode")["name"]
        pts = []
        for gc in seats:
            x, y = to_map.transform(*ll[gc])
            if win[0] < x < win[1] and win[2] < y < win[3]:
                pts.append((str(names.get(gc, gc)), px + (x - win[0]) / dsc, py + (y - win[2]) / dsc))
        _place_cities(cv, pts, (px + 0.8, py + 0.8, px + dw - 0.8, py + dh - 0.8), fs.FS_ITEM - 0.4)
        return letter, (px, py, px + dw, py + dh), dax

    # (b..) small details in the empty south-west corner of (a), next to their box
    land_mm = None
    insets = []
    for d in corner:
        win = d["window"]
        bx0, by0 = mm(win[0], win[2])
        bx1, by1 = mm(win[1], win[3])
        if land_mm is None:
            from shapely.affinity import affine_transform

            land_mm = affine_transform(country.union_all(), [1 / S, 0, 0, 1 / S, mx - ext[0] / S, yA - ext[2] / S])
        side = float(d.get("size_mm", 34.0))
        aspect = (win[3] - win[2]) / (win[1] - win[0])
        placed = None
        for s_ in np.arange(side, 22.0, -1.0):
            pw, ph = s_, s_ * aspect
            for py in np.arange(by0 - 3.0 - ph, yA - 0.01, -0.5):
                r = box(mx + 0.5, py, mx + 0.5 + pw, py + ph)
                if py >= yA + 0.5 and not r.buffer(0.8).intersects(land_mm) and mx + 0.5 + pw < bx0 - 2:
                    placed = (mx + 0.5, py, pw)
                    break
            if placed:
                break
        if placed is None:
            ck.add(f"fig01_detail_{d['key']}_placed", False, "no room in the corner: detail left out", severity="warn")
            continue
        px, py, pw = placed
        dsc = (win[1] - win[0]) / pw
        mark = cv.snapshot()
        letter, panel, _ = draw_detail(d, px, py, dsc)
        detail_box(a, ((win[0], win[1]), (win[2], win[3])), lw=0.5)
        a.text(win[0], win[3], f" {letter}", fontsize=fs.FS_TITLE, fontweight="bold", color=fs.DETAIL, ha="left",
               va="bottom", zorder=15)
        _connect_boxes(cv, (bx0, by0, bx1, by1), panel)
        cv.text(px, panel[3] + 0.6, f"({letter}) {d.get('title', d['key'])}", fontsize=fs.FS_TITLE, fontweight="bold",
                va="bottom", zorder=20, path_effects=halo)
        seg = next(v for v in (10, 20, 25, 50, 100, 200) if v * 1000 / dsc >= 4.0)
        cv.ov.add_patch(Rectangle((px + 0.6, py + 0.6), seg * 1000 / dsc + 4.6, 3.3, fc="white", ec="none", alpha=0.85,
                                  zorder=19))
        cv.scale(px + 1.2, py + 1.8, dsc, seg_km=seg, n=1, bh=0.6, z=20)
        insets += cv.since(mark)
        cv.panel(f"{letter}_{d['key']}", px, py, panel[2], panel[3] + 3.4, hide=[a])

    cv.panel("a_brazil", mx, yA - fs.STRIP + 0.5, mx + MW, FH, hide=insets)

    # (c..) large details in the bottom row, under (a) and the legend
    x_big = 1.0
    for d in bottom:
        win, dsc = d["window"], d["scale"]
        dw, dh = d["size"]
        px, py = x_big + (big_w - dw) / 2, 5.5
        letter, panel, _ = draw_detail(d, px, py, dsc)
        detail_box(a, ((win[0], win[1]), (win[2], win[3])), lw=0.5)
        a.text(win[1], win[3], f"{letter} ", fontsize=fs.FS_TITLE, fontweight="bold", color=fs.DETAIL, ha="right",
               va="bottom", zorder=15)
        bx0, by0 = mm(win[0], win[2])
        bx1, by1 = mm(win[1], win[3])
        _connect_boxes(cv, (bx0, by0, bx1, by1), panel)
        title = d.get("title", d["key"])
        if d.get("uf"):
            st = attrs[(attrs["uf"] == d["uf"]) & (attrs["is_municipality"] if "is_municipality" in attrs else True)]
            title += f": {len(st)} municipalities, {st['cd_rgint'].nunique()} intermediate regions"
        cv.text(px, panel[3] + 0.6, f"({letter}) {title}", fontsize=fs.FS_TITLE + 0.4, fontweight="bold",
                va="bottom")
        seg = next(v for v in (25, 50, 100, 200, 250) if v * 1000 / dsc >= 8.0)
        cv.scale(px, 2.6, dsc, seg_km=seg, n=2, bh=0.8)
        cv.text(px + dw, 1.2, f"1:{dsc * 1000:,.0f}", fontsize=fs.FS_CRED, color=fs.MUTED, ha="right", va="bottom")
        cv.panel(f"{letter}_{d['key']}", px, 0.0, panel[2], panel[3] + 3.4)

    # ---- right of (a): same-scale comparisons ----------------------------------------------------------
    xr = mx + MW + 3.0
    wr = FW - 1.0 - xr
    cpath = _layer_path(cfg, ck, "countries_shp")
    y_top = FH - 0.6
    if cpath is not None:
        nc = gpd.read_file(cpath).to_crs("EPSG:4326")
        iso = next((c for c in ("ADM0_A3", "ISO_A3", "SOV_A3") if c in nc), None)
        cont = "CONTINENT" if "CONTINENT" in nc else None
        sa = nc[(nc[cont] == "South America") & (nc[iso] != "SGS")] if cont else nc     # no South Georgia
        eu_all = nc[((nc[cont] == "Europe") & (nc[iso] != "RUS")) | nc[iso].isin(EU27)] if cont else nc.iloc[0:0]
        eu27_full = nc[nc[iso].isin(EU27)]                         # area incl. the outermost regions
        eu_all = gpd.clip(eu_all, box(-11.5, 34.0, 32.0, 71.5))   # continental Europe, Iceland/Ukraine left out
        sa_m = _equal_area(sa, -58, -15)
        eu_m = _equal_area(eu_all, 10, 52)
        mun = attrs[attrs["is_municipality"]] if "is_municipality" in attrs else attrs
        bra_area = float(mun["area_km2_ibge"].sum()) if "area_km2_ibge" in mun else _geodesic_km2(
            country.to_crs("EPSG:4326").union_all())
        eu_area = _geodesic_km2(eu27_full.union_all()) if len(eu27_full) else float("nan")
        bs, be = sa_m.total_bounds, eu_m.total_bounds
        gap = 2.0
        sl = nice_scale(((bs[2] - bs[0]) + (be[2] - be[0])) / (wr - gap), 10_000.0)
        cv.text(xr, y_top, "Brazil and Europe, same scale", fontsize=fs.FS_TITLE, fontweight="bold", va="top")
        y_top -= 3.6
        hsa, heu = (bs[3] - bs[1]) / sl, (be[3] - be[1]) / sl
        la = cv.map_ax(xr, y_top - hsa, (bs[0], bs[2], bs[1], bs[3]), sl, z=3)
        sa_m.plot(ax=la, facecolor="#f0f0f0", edgecolor="#a8a8a8", linewidth=0.2)
        sa_m[sa_m[iso] == "BRA"].plot(ax=la, facecolor="#9e9e9e", edgecolor=fs.COUNTRY_EDGE, linewidth=0.3)
        map_frame(la, (bs[0], bs[2], bs[1], bs[3]))
        ex = xr + (bs[2] - bs[0]) / sl + gap
        lb = cv.map_ax(ex, y_top - heu, (be[0], be[2], be[1], be[3]), sl, z=3)
        eu_m.plot(ax=lb, facecolor="#f0f0f0", edgecolor="#a8a8a8", linewidth=0.2)
        eu_m[eu_m[iso].isin(EU27)].plot(ax=lb, facecolor="#c4c4c4", edgecolor="#6b6b6b", linewidth=0.25)
        map_frame(lb, (be[0], be[2], be[1], be[3]))
        bp = sa_m[sa_m[iso] == "BRA"].union_all().representative_point()
        cv.text(xr + (bp.x - bs[0]) / sl, y_top - hsa + (bp.y - bs[1]) / sl, f"Brazil\n{bra_area / 1e6:.1f} M km²",
                fontsize=fs.FS_CRED, ha="center", va="center", linespacing=1.2, path_effects=halo, zorder=21)
        cv.text(ex, y_top - heu - 0.7, f"EU-27\n{eu_area / 1e6:.1f} M km²", fontsize=fs.FS_CRED, va="top",
                linespacing=1.2)
        cv.scale(ex, y_top - heu - 6.4, sl, seg_km=1000, n=1, bh=0.6, compact=True)
        cv.text(ex, y_top - heu - 7.9, "Lambert equal-area", fontsize=fs.FS_CRED - 0.3, color=fs.MUTED, va="top")
        cv.panel("locator_europe", xr, y_top - max(hsa, heu + 8.5), FW - 0.5, y_top + 3.6)
        y_top -= max(hsa, heu) + 4.0
        ck.add("fig01_locator", True, f"1:{sl * 1000:,.0f}; Brazil {bra_area:,.0f} km², EU-27 {eu_area:,.0f} km²")

    else:
        ck.add("fig01_locator", False, "no countries layer (paper.layers.countries_shp): comparisons left out",
               severity="warn")

    # ---- legend right of (a), under the comparison; north arrow, scale bar and credits under (a) -------------
    rows = []
    has_second = bool(disp and disp["meta"]["params"]["second_crop"])
    for keys, text in fs.LANDCOVER_LEGEND:
        if disp is None or any(k in present for k in keys):
            if not has_second and "soy_cotton2" in keys:
                text = text.split(" (")[0]                     # no second-crop layer: cotton is 1st season only
            rows.append(("patch", {"fc": fs.LANDCOVER_COLORS[keys[0]], "ec": "#7a7a7a"}, text))
    bnd = [("line", ("#4f4f4f", 0.4), f"State ({states['uf'].nunique()})"),
           ("line", ("#7d7d7d", 0.3), f"Intermediate region ({region_count})")]
    if drawn:
        bnd.append(("lineon", (fs.LANDCOVER_COLORS["pasture"], "white", 0.6),
                    f"Municipality ({', '.join(lt for lt, _ in drawn)})"))
    if drawn:
        towns = [lt for lt, d in drawn if d.get("cities")]
        bnd.append(("dot", (4.2, "white", fs.INK, 0.45),
                    "City (region seats" + (f"; towns in {', '.join(towns)})" if towns else ")")))
    leg_top = y_top - 0.5
    leg_bottom = cv.legend(xr, leg_top, [(f"Land cover {year}, MapBiomas Col. 11", rows), ("Boundaries", bnd)])
    ck.add("fig01_legend_fits", leg_bottom - 1.0 >= HB, f"legend bottom at {leg_bottom:.1f} mm", leg_bottom,
           severity="warn")
    cv.panel("legend", xr - 0.5, leg_bottom - 1.5, FW - 0.5, leg_top + 0.5)
    cv.north(mx + 3.0, yA - fs.STRIP + 3.4)
    cv.scale(mx + 6.5, yA - fs.STRIP + 3.2, S, 500)
    sc_note = " and second crop (beta)" if has_second else ""
    lines = ["SIRGAS 2000 / Brazil Polyconic (EPSG:5880).", f"MapBiomas Col. 11 coverage{sc_note}, {year};",
             "MapBiomas biomes; IBGE mesh 2025; Natural Earth.", fs.CREDIT_CARTO]
    cv.ov.text(mx + MW, yA - fs.STRIP + 1.0, "\n".join(lines), fontsize=fs.FS_CRED, color=fs.MUTED, va="bottom",
               ha="right", linespacing=1.3)
    cv.panel("credits", mx + MW - 50.0, yA - fs.STRIP + 0.5, mx + MW, yA - 0.5)
    ck.add("fig01_scale", True, f"{S:,.0f} m per mm (1:{S * 1000:,.0f})", S)
    area = lc.area_by_group(disp["meta"]) if disp else {}
    data.facts.setdefault("fig01", {})[year] = {
        "year": year, "scale": S, "regions": region_count, "states": int(states["uf"].nunique()),
        "second_crop": has_second, "soybean_mha": (sum(area.get(k, 0) for k in ("soybean", "soy_corn2", "soy_cotton2"))
                                                   / 1e6) if disp else None,
        "soy_corn2_mha": area.get("soy_corn2", 0) / 1e6 if disp else None,
        "max_dev_statistics": next((c.value for c in reversed(ck.checks)
                                    if c.name == f"fig01_landcover_{year}_matches_statistics"), None),
        "details": [{k: d.get(k) for k in ("key", "title", "scale", "uf")} for d in dets]}
    return cv


# ---------------------------------------------------------------------------
# Fig 2 — workflow and what the checks caught
# ---------------------------------------------------------------------------
STAGE_TEXT = {"ingest": "read and tidy\nevery source", "plants": "registers, sites,\ncut-offs",
              "harmonise": "AMC, SIDRA\nflags, gaps", "area": "observed vs\ndeclared area",
              "reconcile": "agreement by\nscale (H1)", "inventory": "N$_1$ and factor\nenvelope (H3)",
              "analyse": "drivers of the\ndivergence (H2)", "publish": "release,\nmanifest, STAC",
              "second_crop": "2nd-season\narea, 30 m", "border_share": "crop near\nborders (H2)",
              "reach": "crop within\nreach, 30 m", "scale": "pixel · mun. ·\nregion (MAUP)",
              "demo": "typology,\nmandate ruler"}
# 30 m passes (bbp commands of their own): the stage each one feeds (up) or reads (down)
PASSES = [("second_crop", "ingest", "up"), ("reach", "plants", "down"), ("scale", "harmonise", None),
          ("demo", "inventory", "down"), ("border_share", "analyse", "up")]


def check_counts(cfg: Config) -> dict[str, dict]:
    out = {}
    for p in sorted((cfg.data_dir / "logs").glob("checks_*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        cs = data["checks"]
        out[data["stage"]] = {"total": len(cs), "checks": cs,
                              "failed": sum(not c["passed"] and c["severity"] == "error" for c in cs),
                              "warnings": sum(not c["passed"] and c["severity"] == "warn" for c in cs)}
    return out


def checks_caught_at(cfg: Config, counts: dict) -> list[tuple[str, str | None]]:
    """(text, station) of each item the checks caught, numbers filled from the check logs; the station
    (`at` in paper.checks_caught) is where Fig 2 marks the item."""
    import re

    def of(stage):
        return {c["name"]: c for c in counts.get(stage, {}).get("checks", [])}

    ingest, border, plants, reach = of("ingest"), of("border_share"), of("plants"), of("reach")
    herds = sum(len(c["value"] or []) for n, c in ingest.items() if n.startswith("sidra_3939_") and "_within_" in n
                and not c["passed"])
    classes = sum(c["passed"] for n, c in ingest.items() if n.startswith("mapbiomas_class_"))
    guard = [c for n, c in border.items() if "_raster_matches_collection_" in n and c["passed"]]
    values = {"herd_subsets": {"n": herds}, "class_guard": {"n": classes},
              "collection_guard": {"value": f"{guard[-1]['value']:.3%}" if guard else "not run"}}
    if (c := plants.get("plants_biomethane_production_notation")) is not None:
        values["anp_notation"] = {"n": c["value"]}
    if (c := plants.get("plants_point_in_declared_municipality_anp_biodiesel_plants")) is not None:
        m = re.search(r"(\d+) of (\d+)", c.get("detail", ""))
        values["biodiesel_outside"] = {"n": m.group(1), "of": m.group(2)} if m else {"n": "?", "of": "?"}
    if (c := plants.get("plants_possible_duplicates")) is not None:
        values["plant_duplicates"] = {"n": sum((c["value"] or {}).values()),
                                      "km": f"{float(cfg.section('plants').get('duplicate_km', 2.0)):g}"}
    stat = [abs(c["value"]) for n, c in reach.items() if n.endswith("_area_matches_statistics") and c["value"] is not None]
    if stat:
        values["reach_statistics"] = {"value": f"{max(stat):.4%}"}
    out = []
    for item in cfg.section("paper").get("checks_caught", []):
        key = item.get("from")
        if key and key not in values:          # its stage has not run: leave the item out
            continue
        out.append((item["text"].format(**values.get(key, {})), item.get("at")))
    return out


def checks_caught(cfg: Config, counts: dict) -> list[str]:
    """The items of the 'what the checks caught' list, numbers filled from the check logs."""
    return [text for text, _ in checks_caught_at(cfg, counts)]


# Fig 2 as a transit map: each data family is a line with its own stations (round: tabular stage; square:
# 30 m pass, the pixel); the lines flow into one trunk where the families are compared, ending at publish.
LINE_COLORS = {"factors": "#8a8a8a", "satellite": "#3f6b8c", "statistics": "#b07a24", "plants": "#7b4a7e",
               "trunk": "#333333"}
LINE_LW = 2.6                       # pt


def fig02_workflow(cfg: Config, ck: CheckLog, data: _Data, host=None):
    import textwrap

    import matplotlib.patheffects as pe
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

    counts = check_counts(cfg)
    stages = [s for s in STAGES if s != "sanitation"]  # sanitation feeds PILAR-2b, not the paper
    passes = [p for p, _, _ in PASSES]
    W, H = fs.FULL_W, 91.0
    cv = fs.canvas(W, H, host)
    ax = cv.ov
    halo = [pe.withStroke(linewidth=1.6, foreground="white")]
    C = LINE_COLORS

    # ---- geometry (mm) -------------------------------------------------------------------------------------------
    y_fac, y_sat, y_sta, y_pla = 78.0, 68.0, 52.0, 36.0
    y_t = y_sta                                                   # the trunk continues the statistics line
    x0, x_ing, x_a, x_reach = 26.5, 31.0, 41.5, 55.0
    xs = {"area": 61.0, "reconcile": 70.0, "scale": 79.0, "inventory": 88.0, "border_share": 97.0,
          "analyse": 106.0, "demo": 115.0, "publish": 124.0}
    bend = 3.0

    def path(pts, color, lw=LINE_LW, z=2):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, solid_capstyle="round",
                solid_joinstyle="round", zorder=z)

    d = y_sat - y_t                                               # the satellite line comes down at 45°
    path([(x0, y_fac), (xs["inventory"] - bend, y_fac), (xs["inventory"], y_fac - bend), (xs["inventory"], y_t)],
         C["factors"])
    path([(x0, y_sat), (xs["area"] - d, y_sat), (xs["area"], y_t)], C["satellite"])
    path([(x0, y_pla), (xs["demo"] - bend, y_pla), (xs["demo"], y_pla + bend), (xs["demo"], y_t)], C["plants"])
    path([(x0, y_sta), (xs["area"], y_t)], C["statistics"])
    path([(xs["area"], y_t), (xs["publish"], y_t)], C["trunk"], lw=LINE_LW * 1.25, z=2.5)

    # ---- stations ------------------------------------------------------------------------------------------------
    R, SQ, PADC = 1.35, 2.7, 1.35

    def stop(x, y, r=R):
        ax.add_patch(Circle((x, y), r, fc="white", ec=fs.INK, lw=0.7, zorder=6))

    def pixel(x, y):                                       # a 30 m pass: a square with a 2 × 2 pixel grid
        ax.add_patch(Rectangle((x - SQ / 2, y - SQ / 2), SQ, SQ, fc="white", ec=fs.INK, lw=0.7, zorder=6))
        ax.plot([x, x], [y - SQ / 2, y + SQ / 2], color="#9a9a9a", lw=0.3, zorder=6.1)
        ax.plot([x - SQ / 2, x + SQ / 2], [y, y], color="#9a9a9a", lw=0.3, zorder=6.1)

    def label(x, y, side, name, dx=0.0):
        """Station name (bold) with its check count, and the two-line description; above or below (x, y),
        shifted `dx` along the line (off an incoming line)."""
        x += dx
        sub = STAGE_TEXT.get(name, "")
        title = name.replace("_", " ")
        c = counts.get(name)
        line_h = fs.FS_CRED * 0.3528 * 1.18
        if side == "above":
            y_sub, y_name = y, y + (sub.count("\n") + 1) * line_h + 0.4
        else:
            y_name, y_sub = y, y - fs.FS_ITEM * 0.3528 * 1.05 - 0.3
        va = "bottom" if side == "above" else "top"
        ax.text(x, y_name, title, fontsize=fs.FS_ITEM, fontweight="bold", ha="center", va=va, zorder=8,
                path_effects=halo)
        if sub:
            ax.text(x, y_sub, sub, fontsize=fs.FS_CRED, color=fs.MUTED, ha="center", linespacing=1.12, va=va,
                    zorder=8, path_effects=halo)
        if c:                                              # the check count: a dark tab right of the name
            w = len(title) * fs.FS_ITEM * 0.3528 * 0.58
            yc = y_name + (1 if side == "above" else -1) * fs.FS_ITEM * 0.3528 * 0.48
            ax.text(x + w / 2 + 1.0, yc, f"{c['total']}", fontsize=fs.FS_CRED - 0.4, color="white",
                    fontweight="bold", ha="left", va="center", zorder=9,
                    bbox={"boxstyle": "round,pad=0.18,rounding_size=0.25", "fc": fs.INK, "ec": "none"})

    up, dn = 2.4, -2.4
    # ingest reads every source but the plant registers (read by plants itself)
    ax.add_patch(FancyBboxPatch((x_ing - PADC, y_sta - PADC), 2 * PADC, (y_fac - y_sta) + 2 * PADC,
                                boxstyle=f"round,pad=0,rounding_size={PADC}", fc="white", ec=fs.INK, lw=0.7,
                                zorder=6))
    label(x_ing, y_fac + PADC + 1.2, "above", "ingest")
    pixel(x_a, y_sat)
    label(x_a, y_sat + up, "above", "second_crop")
    stop(x_a, y_sta)
    label(x_a, y_sta + dn, "below", "harmonise")
    stop(x_a, y_pla)
    label(x_a, y_pla + dn, "below", "plants")
    pixel(x_reach, y_pla)
    label(x_reach, y_pla + dn, "below", "reach")
    side = {"area": "above", "reconcile": "below", "scale": "above", "inventory": "below",
            "border_share": "above", "analyse": "below", "demo": "above", "publish": "below"}
    for k, x in xs.items():
        if k in passes:
            pixel(x, y_t)
        else:
            stop(x, y_t, R * (1.25 if k == "publish" else 1.0))
        label(x, y_t + (up if side[k] == "above" else dn), side[k], k, dx=3.5 if k == "area" else 0.0)

    # ---- sources: the line names, at the start of each line ------------------------------------------------------
    pam = ", ".join(str(t) for t in sorted(int(k) for k, v in (cfg.raw.get("sidra") or {}).items()
                                          if str((v or {}).get("survey", "")).upper() == "PAM"))
    fac = cfg.factors.get("source", {}).get("version", "").split(" / ")[-1]
    sources = [(y_fac, "factors", "CP2b factors", ["g, TS, VS/TS, BMP", fac]),
               (y_sat, "satellite", "MapBiomas Col. 11", ["coverage 30 m, stats,", "2nd crop (beta),", "irrigation"]),
               (y_sta, "statistics", "IBGE", [f"PAM {pam}", "municipal mesh 2025"]),
               (y_pla, "plants", "Plant registers", ["ANP, CIBiogás, ANEEL,", "MAPA SIF; CONAB, EPE", "(location check)"])]
    for y, key, head, lines in sources:
        ax.add_patch(Rectangle((x0 - 0.9, y - 1.6), 1.8, 3.2, fc=C[key], ec="none", zorder=5))    # line terminal
        top = y + (len(lines) * 1.95 + 2.4) / 2
        ax.text(x0 - 2.0, top, head, fontsize=fs.FS_ITEM, fontweight="bold", ha="right", va="top", color=C[key])
        ax.text(x0 - 2.0, top - 2.4, "\n".join(lines), fontsize=fs.FS_CRED, ha="right", va="top", color=fs.INK,
                linespacing=1.12)

    # ---- what the checks caught: numbered marks on the station that caught each item ------------------------------
    corner = {"ingest": (x_ing + PADC, y_sta - PADC - 1.0), "second_crop": (x_a + SQ / 2, y_sat + SQ / 2),
              "harmonise": (x_a + R, y_sta + R), "plants": (x_a + R, y_pla + R),
              "reach": (x_reach + SQ / 2, y_pla + SQ / 2),
              **{k: (x + R, y_t + R) for k, x in xs.items()},
              "border_share": (xs["border_share"] + SQ / 2, y_t - SQ / 2 - 1.2),   # its label is above
              **{k: (x0 + 0.9, y + 1.6) for y, k, _, _ in sources}}
    items = [(t, at) for t, at in checks_caught_at(cfg, counts) if at in corner]
    MR = 1.15

    def mark(x, y, i):
        ax.add_patch(Circle((x, y), MR, fc=fs.DETAIL, ec="white", lw=0.5, zorder=10))
        ax.text(x, y - 0.05, str(i), fontsize=fs.FS_CRED - 0.6, color="white", fontweight="bold", ha="center",
                va="center", zorder=11)

    used: dict[str, int] = {}
    for i, (_, at) in enumerate(items, 1):
        k = used.get(at, 0)
        x, y = corner[at]
        mark(x + 0.4 + k * (2 * MR + 0.3), y + 0.4, i)
        used[at] = k + 1

    # ---- key (top) and the list of findings (bottom) ---------------------------------------------------------------
    ky, kx = H - 1.8, 1.0
    ax.plot([kx, kx + 5], [ky, ky], color=fs.INK, lw=LINE_LW * 0.5)
    stop(kx + 2.5, ky)
    ax.text(kx + 6.2, ky, "stage", fontsize=fs.FS_CRED, va="center")
    kx += 15
    pixel(kx + 1.3, ky)
    ax.text(kx + 3.4, ky, "30 m pass (own command)", fontsize=fs.FS_CRED, va="center")
    kx += 27
    ax.text(kx, ky, "41", fontsize=fs.FS_CRED - 0.4, color="white", fontweight="bold", va="center",
            bbox={"boxstyle": "round,pad=0.18,rounding_size=0.25", "fc": fs.INK, "ec": "none"})
    ax.text(kx + 3.2, ky, "invariant checks", fontsize=fs.FS_CRED, va="center")
    kx += 20
    mark(kx + MR, ky, 1)
    ax.text(kx + 2 * MR + 1.0, ky, "what the checks caught (below)", fontsize=fs.FS_CRED, va="center")

    rel = {k: v for k, v in counts.items() if k in stages or k in passes}
    total, failed = sum(v["total"] for v in rel.values()), sum(v["failed"] for v in rel.values())
    warns = sum(v["warnings"] for v in rel.values())
    fy = 21.0
    ax.plot([1, W - 1], [fy + 1.6, fy + 1.6], color=fs.INK, lw=0.4)
    ax.text(1.0, fy, "What the checks caught", fontsize=fs.FS_ITEM, fontweight="bold", va="top")
    ax.text(W - 1.0, fy, f"{total} checks in this run · {failed} failed · {warns} warnings", fontsize=fs.FS_CRED,
            va="top", ha="right", color=fs.MUTED)
    half = (len(items) + 1) // 2
    line_h = fs.FS_CRED * 0.3528 * 1.15
    for col in range(2):
        y = fy - 3.9
        for i in range(col * half, min(len(items), (col + 1) * half)):
            lines = textwrap.wrap(items[i][0], 80)
            x = 1.0 + col * 66.0
            mark(x + MR, y - line_h / 2, i + 1)
            ax.text(x + 2 * MR + 0.9, y, "\n".join(lines), fontsize=fs.FS_CRED, va="top", linespacing=1.15)
            y -= len(lines) * line_h + 0.9
    missing = [s for s in [*stages, *passes] if s not in counts]
    ck.add("fig02_stage_check_counts", not missing, f"no checks file for: {missing}", severity="warn")
    return cv


# ---------------------------------------------------------------------------
# Fig 3 — national area, observed vs declared
# ---------------------------------------------------------------------------
def fig03_national_area(cfg: Config, ck: CheckLog, data: _Data, host=None):
    nat = data.get("unit_ratios_nation")
    years = summary_years(cfg)
    crops = [c for c in cfg.crops if c in set(nat["crop"])]
    ncol, left, gx, w, h, gy = 4, 8.0, 6.0, 25.0, 24.0, 10.5
    nrow = int(np.ceil((len(crops) + 1) / ncol))
    FW, FH = fs.FULL_W, nrow * (h + gy) + 1
    cv = fs.canvas(FW, FH, host)
    for k, c in enumerate(crops):
        r, col = divmod(k, ncol)
        x, y = left + col * (w + gx), FH - (r + 1) * (h + gy) + 5.5
        a = _chart_axes(cv, x, y, w, h)
        s = nat[nat["crop"] == c].sort_values("year")
        a.axvspan(min(years) - 0.5, max(years) + 0.5, color=fs.FAINT, lw=0, zorder=0)
        a.plot(s["year"], s["a_dec_ha"] / 1e6, color=fs.DECLARED_COLOR, lw=0.7, ls=(0, (2.5, 1.2)))
        fs.crop_line(a, s["year"], s["a_obs_ha"] / 1e6, c, lw=1.0, marker=False)
        a.set_ylim(0, max(s["a_obs_ha"].max(), s["a_dec_ha"].max()) / 1e6 * 1.12)
        a.set_xlim(2007.5, 2024.5)
        a.set_xticks([2008, 2016, 2024])
        cv.text(x, y + h + 1.2, fs.label(c), fontsize=fs.FS_TITLE + 0.4, fontweight="bold", va="bottom")
        if col == 0:
            a.set_ylabel("Mha", labelpad=1.5)
    r, col = divmod(len(crops), ncol)
    kx, ky = left + col * (w + gx), FH - (r + 1) * (h + gy) + 5.5 + h
    cv.legend(kx, ky, [("", [("line", (fs.MUTED, 1.0), "Observed (MapBiomas),"),
                             ("line", ("none", 0), "in the crop's colour"),
                             ("line", (fs.DECLARED_COLOR, 0.7, (0, (2.5, 1.2))), "Declared (IBGE PAM)"),
                             ("patch", fs.FAINT, f"{years[0]}–{years[-1]}")])])
    return cv


# ---------------------------------------------------------------------------
# Fig 4 — agreement vs scale (C1)
# ---------------------------------------------------------------------------
def fig04_agreement_scale(cfg: Config, ck: CheckLog, data: _Data, host=None):
    from matplotlib.ticker import PercentFormatter

    ag = data.get("agreement_by_scale")
    years = summary_years(cfg)
    crops = [c for c in cfg.crops if c in set(ag["crop"])]
    scales = [s for s in ("municipality", "amc", "rgint", "uf") if s in set(ag["scale"])]
    m = ag[ag["year"].isin(years)].groupby(["crop", "scale"])[["area_share_within_25pct",
                                                               "share_within_25pct"]].mean()
    FW, FH = fs.FULL_W, 64.0
    cv = fs.canvas(FW, FH, host)
    xs = np.arange(len(scales))
    pw, ph, y0 = 45.0, 44.0, 11.0
    for i, (key, title) in enumerate([("area_share_within_25pct", "Weighted by declared area"),
                                      ("share_within_25pct", "Share of units")]):
        x0 = 13.0 + i * (pw + 7.0)
        a = _chart_axes(cv, x0, y0, pw, ph, grid_y=True)
        ends = []
        for c in crops:
            y = [m[key].get((c, s), np.nan) for s in scales]
            fs.crop_line(a, xs, y, c)
            ends.append((fs.label(c), xs[-1] + 0.05, y[-1], fs.CROP_COLORS[c]))
        a.set_xticks(xs, [SCALE_LABELS[s].replace("Intermediate region", "Intermediate\nregion") for s in scales])
        a.set_xlim(-0.25, len(scales) - 0.75)
        a.set_ylim(0, 1)
        a.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        if i:
            a.set_yticklabels([])
            fs.direct_labels(a, ends, x_pad=0.3, min_gap=0.062, fontsize=fs.FS_ITEM)
        else:
            a.set_ylabel(f"Within ±25 % of each other\n(mean {years[0]}–{years[-1]})")
        cv.text(x0, y0 + ph + 1.5, title, fontsize=fs.FS_LABEL, fontweight="bold", va="bottom")
    cv.legend(14.0, y0 + ph - 1.0, [("", [("line", (fs.MUTED, 0.9), "annual crop"),
                                          ("line", (fs.MUTED, 0.9, (0, (3.5, 1.4))), "(semi-)perennial")])],
              header=None)
    return cv


# ---------------------------------------------------------------------------
# Fig 5 — municipal log-ratio maps (one map scale)
# ---------------------------------------------------------------------------
def _crop_maps_layout(ext, ncol: int = 4, width: float = fs.FULL_W, gap: float = 1.5):
    S = nice_scale((ext[1] - ext[0]) / ((width - 2 - (ncol - 1) * gap) / ncol))
    return S, (ext[1] - ext[0]) / S, (ext[3] - ext[2]) / S


def fig05_log_ratio_maps(cfg: Config, ck: CheckLog, data: _Data, host=None):
    g, states, country, ext = data.base()
    units = data.get("unit_ratios_municipality")
    years = summary_years(cfg)
    min_area = float(cfg.section("reconcile").get("min_area_ha", 0))
    crops = [c for c in cfg.crops if c in set(units["crop"])]
    edges, colors, _ = ratio_classes()
    ncol, gap, LB, title_h = 4, 1.5, 21.0, 4.0
    S, MW, MH = _crop_maps_layout(ext, ncol, gap=gap)
    nrow = int(np.ceil(len(crops) / ncol))
    row_h = MH + title_h
    FW, FH = fs.FULL_W, LB + nrow * row_h + 1.0
    cv = fs.canvas(FW, FH, host)
    for i, c in enumerate(crops):
        r, col = divmod(i, ncol)
        x, y = 1.0 + col * (MW + gap), LB + (nrow - 1 - r) * row_h
        a = cv.map_ax(x, y, ext, S)
        lr = municipal_log_ratio(units, c, years, min_area)
        _plot_classes(a, g, classify(lr, edges), colors)
        boundaries(a, states, country, 0.25)
        map_frame(a, ext)
        finite = lr[np.isfinite(lr)]
        within = float((np.abs(finite) <= np.log(1.1)).mean()) if len(finite) else np.nan
        cv.text(x + 0.5, y + MH + 0.6, fs.label(c), fontsize=fs.FS_TITLE + 0.4, fontweight="bold", va="bottom")
        cv.text(x + 0.5, y + 0.6, f"±10 %: {within:.0%}", fontsize=fs.FS_CRED, color=fs.MUTED, va="bottom")
        ck.add(f"fig05_{c}_has_municipalities", len(lr) > 0, f"{len(lr)} municipalities scored", len(lr),
               severity="warn")
    if len(crops) % ncol:                                  # furniture in the empty cell
        r, col = divmod(len(crops), ncol)
        x, y = 1.0 + col * (MW + gap), LB + (nrow - 1 - r) * row_h
        cv.north(x + 3.0, y + MH - 6.5)
        cv.scale(x + 7.5, y + MH - 6.7, S, seg_km=1000, n=1)
        cv.credits(x + 1.0, y + MH - 11.0, "MapBiomas Col. 11;\nIBGE PAM (this study)", narrow=True)
    else:
        cv.strip(1.0, S, DATA_CREDIT, y0=LB - fs.STRIP)
    _ratio_legend(cv, 1.0, LB - 0.5, years, min_area)
    ck.add("fig05_scale", True, f"{S:,.0f} m per mm", S)
    return cv


# ---------------------------------------------------------------------------
# Fig 6 — example municipalities on the coverage raster
# ---------------------------------------------------------------------------
RULES = {
    "a": "soybean, MT/GO, high double-crop intensity, large |log-ratio|",
    "b": "sugarcane or coffee, much of the crop near the municipal border, large |log-ratio|",
    "c": "coffee in MG, PAM area destined for harvest ≫ observed",
    "d": "sources agree within ±10 % (reference)",
}
CLASS_NAME_FIX = {"Coffe": "Coffee", "Sugar cane": "Sugarcane", "Forest Formation": "Forest",
                  "Savanna Formation": "Savanna", "Grassland Formation": "Grassland", "Mosaic of Uses": "Mosaic of uses",
                  "Other Temporary Crops": "Other temp. crops", "Other Perennial Crops": "Other perennial",
                  "Floodable Forest": "Floodable forest", "Urban Area": "Urban", "River, Lake and Ocean": "Water",
                  "Other non Forest Formations": "Other non-forest", "Forest Plantation": "Forest plantation"}


def example_candidates(cfg: Config, data: _Data, n: int = 5) -> pd.DataFrame:
    """Candidate municipalities for Fig 6, by rule (see RULES); the first of each rule is drawn by default."""
    units = data.get("unit_ratios_municipality")
    attrs = data.get("municipality_attributes")
    pred = data.get("predictors")
    years = summary_years(cfg)
    year = int(cfg.section("paper").get("fig6", {}).get("year", years[-1]))
    u = units[units["year"].isin(years)].copy()
    u["geocode"] = u["unit"].astype("int64")
    both = u[(u["a_obs_ha"] > 0) & (u["a_dec_ha"] > 0)]
    med = both.groupby(["geocode", "crop"]).agg(median_log_ratio=("log_ratio", "median"),
                                                mean_dec_ha=("a_dec_ha", "mean"), years_both=("year", "size"))
    now = u[u["year"] == year].set_index(["geocode", "crop"])[["a_obs_ha", "a_dec_ha"]]
    t = med.join(now, how="inner").reset_index()
    t = t.merge(attrs[["geocode", "name", "uf"]], on="geocode", how="left")
    if pred is not None and "double_crop_intensity" in pred:
        t = t.merge(pred[pred["year"] == year][["geocode", "double_crop_intensity"]], on="geocode", how="left")
    bp = cfg.data_dir / "interim" / "border_share_municipal.parquet"
    if bp.exists():
        b = pd.read_parquet(bp)
        by = year if year in set(b["year"]) else int(b["year"].max())
        b = b[b["year"] == by][["geocode", "crop", "border_share_1km"]].assign(border_share_year=by)
        t = t.merge(b, on=["geocode", "crop"], how="left")
    t["abs_lr"] = t["median_log_ratio"].abs()
    out = []

    def take(rule, sel, by, ascending=False):
        s = sel.sort_values(by, ascending=ascending).head(n).copy()
        s.insert(0, "rank", range(1, len(s) + 1))
        s.insert(0, "rule", rule)
        out.append(s)

    full = t["years_both"] == len(years)
    soy = t[(t["crop"] == "soybean") & t["uf"].isin(["MT", "GO"]) & full & (t["mean_dec_ha"] >= 10000)]
    if "double_crop_intensity" in soy and soy["double_crop_intensity"].notna().any():
        soy = soy[soy["double_crop_intensity"] >= soy["double_crop_intensity"].quantile(0.75)]
    take("a", soy, "abs_lr")
    sc = t[t["crop"].isin(["sugarcane", "coffee"]) & full & (t["mean_dec_ha"] >= 2000)]
    if "border_share_1km" in sc and sc["border_share_1km"].notna().any():
        sc = sc[sc["border_share_1km"] >= sc["border_share_1km"].quantile(0.75)]
    take("b", sc, "abs_lr")
    cof = t[(t["crop"] == "coffee") & (t["uf"] == "MG") & full & (t["mean_dec_ha"] >= 2000)
            & (t["median_log_ratio"] <= np.log(0.5))].assign(gap_ha=lambda d: d["a_dec_ha"] - d["a_obs_ha"])
    take("c", cof, "gap_ha")
    agree = t[t["crop"].isin(["soybean", "sugarcane"]) & full & (t["abs_lr"] <= np.log(1.1))
              & (t["mean_dec_ha"] >= 20000)]
    take("d", agree, "mean_dec_ha")
    res = pd.concat(out, ignore_index=True)
    res["rule_text"] = res["rule"].map(RULES)
    res["year"] = year
    return res.drop(columns=["abs_lr"])


def _vrt_sources_exist(vrt: Path) -> bool:
    """Every SourceFilename of a VRT is a file on this machine."""
    import xml.etree.ElementTree as ET

    for el in ET.parse(vrt).getroot().iter("SourceFilename"):
        p = Path(el.text or "")
        if el.get("relativeToVRT") == "1":
            p = vrt.parent / p
        if not p.exists():
            return False
    return True


def _coverage_source(cfg: Config, year: int) -> Path | None:
    vrt = cfg.data_dir / "interim" / "coverage_vrt" / f"brazil_coverage-col11_{year}.vrt"
    if vrt.exists() and _vrt_sources_exist(vrt):      # a VRT written elsewhere (Docker) is rebuilt here
        return vrt
    from bbp.ingest.border_share import build_vrt, find_year_sources

    pattern = cfg.section("border_share").get("tif_pattern", "brazil_coverage-col11_{year}.tif")
    src = find_year_sources(cfg.input_path("coverage_tif_dir"), pattern, [year]).get(year)
    if src is None:
        return None
    if isinstance(src, list):
        build_vrt(src, vrt)
        return vrt
    return src


def _chip(a, src: Path, geom_4674, neighbours, max_px: int, lut: np.ndarray, side_km: float):
    """Coverage classes in a square window of `side_km` around one municipality (degrees, local aspect)."""
    import geopandas as gpd
    import rasterio
    from rasterio.windows import from_bounds

    c = geom_4674.centroid
    hx = side_km / 2 / (111.32 * np.cos(np.radians(c.y)))
    hy = side_km / 2 / 111.32
    x0, x1, y0, y1 = c.x - hx, c.x + hx, c.y - hy, c.y + hy
    with rasterio.open(src) as ds:
        win = from_bounds(x0, y0, x1, y1, ds.transform).round_offsets().round_lengths()
        k = max(win.width, win.height) / max_px
        shape = (max(int(win.height / max(k, 1)), 1), max(int(win.width / max(k, 1)), 1))
        arr = ds.read(1, window=win, out_shape=shape, boundless=True, fill_value=0)
        b = ds.window_bounds(win)
    a.imshow(lut[arr], extent=(b[0], b[2], b[1], b[3]), interpolation="nearest", rasterized=True, aspect="auto")
    neighbours.boundary.plot(ax=a, color="white", lw=0.5)
    neighbours.boundary.plot(ax=a, color=fs.STATE_EDGE, lw=0.25)
    gpd.GeoSeries([geom_4674]).boundary.plot(ax=a, color="white", lw=1.6)
    gpd.GeoSeries([geom_4674]).boundary.plot(ax=a, color=fs.INK, lw=0.7)
    map_frame(a, (x0, x1, y0, y1))
    for s in a.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)
        s.set_edgecolor(fs.MUTED)
    return arr


def fig06_examples(cfg: Config, ck: CheckLog, data: _Data, host=None):
    import geopandas as gpd

    spec = cfg.section("paper").get("fig6", {})
    cand = example_candidates(cfg, data)
    cand.to_csv(out_dir(cfg) / "fig06_candidates.csv", index=False)
    year = int(spec.get("year", summary_years(cfg)[-1]))
    chosen = [int(c) for c in spec.get("examples") or []]
    if chosen:
        picks = [cand[cand["geocode"] == gc].iloc[0] for gc in chosen if (cand["geocode"] == gc).any()]
    else:
        picks = [cand[cand["rule"] == r].iloc[0] for r in RULES if (cand["rule"] == r).any()]
    picks = picks[:4]
    ck.add("fig06_examples_from_rules", bool(picks), f"{len(picks)} examples; {len(cand)} candidates written",
           len(picks), severity="warn")
    src = _coverage_source(cfg, year)
    if not ck.add("fig06_coverage_raster", src is not None, f"coverage raster {year}", severity="warn") or not picks:
        return None
    leg = pd.read_csv(cfg.data_dir / "interim" / "mapbiomas_legend.csv").set_index("class_id")
    lut = np.full((256, 4), (0.93, 0.93, 0.93, 1.0))
    lut[0] = (1, 1, 1, 1)
    for cid, row in leg.iterrows():
        if 0 < int(cid) < 256 and isinstance(row.get("hex_code"), str):
            h = row["hex_code"].lstrip("#")
            lut[int(cid)] = (*(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)), 1.0)
    mesh = gpd.read_parquet(cfg.data_dir / "interim" / "municipalities_geom.parquet",
                            columns=["geocode", "name", "uf", "geometry"])
    cw, cx0, cgap, below, title_h = 44.0, 1.0, 6.0, 14.0, 4.0
    FW = fs.FULL_W
    FH = fs.STRIP + 2 * (cw + below + title_h) + 1.0
    cv = fs.canvas(FW, FH, host)
    counts = np.zeros(256, dtype=np.int64)
    letters = "abcd"
    for k, p in enumerate(picks):
        r, c = divmod(k, 2)
        x = cx0 + c * (cw + cgap)
        y = fs.STRIP + (1 - r) * (cw + below + title_h) + below
        a = cv.ax(x, y, cw, cw, z=1)
        gc = int(p["geocode"])
        geom = mesh.loc[mesh["geocode"] == gc, "geometry"].iloc[0]
        x0, y0, x1, y1 = geom.bounds
        lat0 = (y0 + y1) / 2
        side = max((x1 - x0) * 111.32 * np.cos(np.radians(lat0)), (y1 - y0) * 111.32) * 1.12
        nb = mesh.cx[x0 - 1:x1 + 1, y0 - 1:y1 + 1]
        nb = nb[nb["geocode"] != gc]
        arr = _chip(a, src, geom, nb, int(spec.get("max_chip_px", 900)), lut, side)
        counts += np.bincount(arr.ravel(), minlength=256)
        chip_scale = side * 1000 / cw                                        # metres per mm
        seg = next(v for v in (1, 2, 2.5, 5, 10, 20, 25, 50, 100) if v * 1000 / chip_scale >= 5)
        bar = cv.scale(x + 0.5, y - 2.4, chip_scale, seg_km=seg, n=2, bh=0.8)
        ratio = p["a_obs_ha"] / p["a_dec_ha"] if p["a_dec_ha"] else np.nan
        cv.text(x, y + cw + 0.8, f"({letters[k]}) {p['name']} ({p['uf']}) · {fs.label(p['crop'])}",
                fontsize=fs.FS_TITLE + 0.4, fontweight="bold", va="bottom")
        del bar
        cv.text(x + 0.5, y - 5.0, f"{year}: MapBiomas {p['a_obs_ha']:,.0f} ha, PAM {p['a_dec_ha']:,.0f} ha\n"
                f"A$_{{obs}}$ / A$_{{dec}}$ = {ratio:.2f}", fontsize=fs.FS_ITEM, va="top", linespacing=1.35)
    # right column: locator (Brazil, the examples) and the legend of the classes in the chips
    g, states, country, ext = data.base()
    lx = cx0 + 2 * cw + cgap + 3.0
    lw_mm = FW - lx - 1.0
    ls = nice_scale((ext[1] - ext[0]) / lw_mm)
    loc_h = (ext[3] - ext[2]) / ls
    la = cv.map_ax(lx, FH - 1.0 - title_h - loc_h, ext, ls, z=2)
    states.plot(ax=la, facecolor="#f2f2f2", edgecolor=fs.STATE_EDGE, linewidth=0.2)
    country.boundary.plot(ax=la, color=fs.COUNTRY_EDGE, linewidth=0.35)
    placed = []
    for k, p in enumerate(picks):
        pt = g.loc[g["geocode"] == int(p["geocode"]), "geometry"].iloc[0].representative_point()
        la.plot(pt.x, pt.y, "o", ms=2.4, color=fs.INK)
        near = sum(np.hypot(pt.x - q.x, pt.y - q.y) < 250e3 for q in placed)    # alternate sides when close
        off = [(2.2, 1.2), (-6.0, 1.2), (2.2, -6.5), (-6.0, -6.5)][near % 4]
        la.annotate(letters[k], (pt.x, pt.y), xytext=off, textcoords="offset points", fontsize=fs.FS_ITEM,
                    fontweight="bold")
        placed.append(pt)
    map_frame(la, ext)
    top = [int(c) for c in np.argsort(counts)[::-1] if counts[c] > 0.004 * counts.sum() and c != 0][:12]
    rows = [("patch", tuple(lut[c]), CLASS_NAME_FIX.get(str(leg.loc[c, "class_name"]), str(leg.loc[c, "class_name"]))
             if c in leg.index else f"class {c}") for c in top]
    cv.legend(lx, FH - 1.0 - title_h - loc_h - 4.0, [(f"MapBiomas Col. 11, {year}", rows)])
    cv.north(cx0 + 3.0, 3.4)
    cv.credits(cx0 + 7.5, 7.6, f"MapBiomas Col. 11 coverage {year}, 30 m; IBGE PAM (this study)")
    return cv


# ---------------------------------------------------------------------------
# Fig 7 — H2 variance partition (C2)
# ---------------------------------------------------------------------------
def fig07_h2_variance(cfg: Config, ck: CheckLog, data: _Data, host=None):
    from matplotlib.patches import Rectangle

    fit, moran = data.get("h2_model_fit"), data.get("h2_moran")
    if not ck.add("fig07_h2_tables", fit is not None and moran is not None, "H2 tables", severity="warn"):
        return None
    crops = [c for c in cfg.crops if c in set(fit["crop"])]
    f = fit.set_index("crop").reindex(crops)
    mi = moran.set_index("crop").reindex(crops)
    fixed = f["r2_marginal"].fillna(0).to_numpy()
    region = f["region_variance_share"].fillna(0).to_numpy()
    resid = np.clip(1 - fixed - region, 0, None)
    row = 5.0
    FW, FH = fs.FULL_W, len(crops) * row + 22.0
    cv = fs.canvas(FW, FH, host)
    a = _chart_axes(cv, 22.0, 9.0, 94.0, len(crops) * row)
    ys = np.arange(len(crops))[::-1]
    parts = [(fixed, "#3b3b3b", "white", "Predictors (marginal R²)"),
             (region, "#9a9a9a", fs.INK, "Intermediate region"), (resid, "#e3e3e3", fs.INK, "Residual")]
    left = np.zeros(len(crops))
    for vals, col, txt, _ in parts:
        a.barh(ys, vals, left=left, color=col, height=0.68, edgecolor="white", lw=0.5)
        for y, l0, v in zip(ys, left, vals, strict=True):
            if v >= 0.07:
                a.text(l0 + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=fs.FS_ITEM, color=txt)
        left += vals
    for y, c in zip(ys, crops, strict=True):
        i_, p_ = mi.loc[c, "morans_i"], mi.loc[c, "p_value"]
        if np.isfinite(i_):
            a.text(1.03, y, f"{i_:.2f}{_sig(p_)}", va="center", fontsize=fs.FS_ITEM)
    a.text(1.03, len(crops) - 0.35, "Moran's I", fontsize=fs.FS_TITLE, fontweight="bold", va="bottom")
    a.set_yticks(ys, [fs.label(c) for c in crops])
    a.set_ylim(-0.5, len(crops) - 0.5)
    a.set_xlim(0, 1)
    a.xaxis.set_major_formatter("{x:.0%}")
    a.set_xlabel("Share of the variance of ln(A$_{obs}$ / A$_{dec}$), mixed model", labelpad=2)
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    x = 22.0
    for _, col, _, lab in parts:                      # one-row key above the bars
        cv.ov.add_patch(Rectangle((x, FH - 6.0), fs.HW, 1.7, fc=col, ec="#7a7a7a", lw=0.3))
        cv.text(x + fs.HW + fs.TXT, FH - 5.15, lab, va="center")
        x += fs.HW + fs.TXT + 3.0 + len(lab) * 1.05
    return cv


# ---------------------------------------------------------------------------
# Fig 8 — H2 standardised effects
# ---------------------------------------------------------------------------
SHORT_PREDICTOR = {"log_mun_area_ha": "Municipal area\n(log)", "crop_share": "Crop share\nof municipality",
                   "log_double_crop": "Double cropping\n(log)", "irrigated_share": "Irrigated\nshare",
                   "border_share_1km": "Crop within 1 km\nof a border", "log_landsat_valid_obs": "Clear Landsat\nobs. (log)"}


def fig08_h2_effects(cfg: Config, ck: CheckLog, data: _Data, host=None):
    coefs = data.get("h2_mixed_effects")
    if not ck.add("fig08_h2_tables", coefs is not None, "H2 coefficients", severity="warn"):
        return None
    crops = [c for c in cfg.crops if c in set(coefs["crop"])]
    preds = [p for p in NUMERIC if p in set(coefs["term"])]
    row, left, gap = 4.6, 22.0, 2.5
    pw = (fs.FULL_W - left - 1.0 - (len(preds) - 1) * gap) / len(preds)
    FW, FH = fs.FULL_W, len(crops) * row + 21.0
    cv = fs.canvas(FW, FH, host)
    ys = np.arange(len(crops))[::-1]
    lo = coefs.loc[coefs["term"].isin(preds), "ci_low"].min()
    hi = coefs.loc[coefs["term"].isin(preds), "ci_high"].max()
    span = max(abs(lo), abs(hi)) * 1.08
    for i, p in enumerate(preds):
        a = _chart_axes(cv, left + i * (pw + gap), 15.0, pw, len(crops) * row)
        cp = coefs[coefs["term"] == p].set_index("crop")
        a.axvline(0, color=fs.MUTED, lw=0.5)
        a.grid(axis="x", color=fs.FAINT, lw=0.4)
        for y, c in zip(ys, crops, strict=True):
            if c not in cp.index:
                continue
            r = cp.loc[c]
            st = fs.crop_style(c)
            a.plot([r["ci_low"], r["ci_high"]], [y, y], color=st["color"], lw=1.0, solid_capstyle="butt",
                   path_effects=_casing(c, 1.0))
            a.plot(r["estimate"], y, st["marker"], color=st["color"], ms=3.4, mec=fs.INK, mew=0.3)
        a.set_xlim(-span, span)
        a.set_ylim(-0.5, len(crops) - 0.5)
        a.set_yticks(ys, [fs.label(c) for c in crops] if i == 0 else [])
        a.tick_params(axis="y", length=0)
        a.spines["left"].set_visible(False)
        cv.text(left + i * (pw + gap) + pw / 2, 15.0 + len(crops) * row + 1.0,
                SHORT_PREDICTOR.get(p, PREDICTOR_LABELS[p]), fontsize=fs.FS_TITLE, fontweight="bold", ha="center",
                va="bottom", linespacing=1.15)
    cv.text(left + (fs.FULL_W - left) / 2, 1.5, "Change in ln(A$_{obs}$ / A$_{dec}$) per standard deviation of the "
            "predictor (95 % CI)", fontsize=fs.FS_LABEL, ha="center", va="bottom")
    return cv


# ---------------------------------------------------------------------------
# Fig 9 — source effect vs factor envelope (C3)
# ---------------------------------------------------------------------------
def municipal_source_effect(n1: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """|ln(N1_obs / N1_dec)| per municipality-year and crop, where both are positive (observed crops only)."""
    n = n1[n1["year"].isin(years) & ~n1["activity_declared_only"].astype(bool)]
    w = n.pivot_table(index=["geocode", "year", "crop"], columns="variant", values="n1_ref_nm3",
                      observed=True).dropna(subset=["dec", "obs"])
    w = w[(w["dec"] > 0) & (w["obs"] > 0)]
    return pd.DataFrame({"abs_log": np.abs(np.log(w["obs"] / w["dec"]))}).reset_index()


def fig09_source_vs_factors(cfg: Config, ck: CheckLog, data: _Data, host=None):
    import matplotlib.pyplot as plt

    n1 = data.get("n1_municipal")
    sve = data.get("source_vs_factor_effect")
    years = summary_years(cfg)
    eff = municipal_source_effect(n1, years)
    env = sve[sve["year"].isin(years)].groupby("crop")[["factor_envelope_log", "factor_oat_bmp_log"]].mean()
    crops = [c for c in cfg.crops if c in set(eff["crop"]) and c in env.index]
    q = eff.groupby("crop")["abs_log"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    row = 7.5
    FW, FH = fs.FULL_W, len(crops) * row + 22.0
    cv = fs.canvas(FW, FH, host)
    a = _chart_axes(cv, 22.0, 12.0, fs.FULL_W - 23.0, len(crops) * row + 4.0)
    ys = np.arange(len(crops))[::-1]
    for y, c in zip(ys, crops, strict=True):
        e, b = env.loc[c, "factor_envelope_log"], env.loc[c, "factor_oat_bmp_log"]
        a.add_patch(plt.Rectangle((0, y - 0.36), e, 0.72, color="#e6e6e6", lw=0, zorder=0))
        a.add_patch(plt.Rectangle((0, y - 0.36), b, 0.72, color="#cdcdcd", lw=0, zorder=0))
        r = q.loc[c]
        col = fs.CROP_COLORS[c]
        a.plot([r[0.1], r[0.9]], [y, y], color=col, lw=0.8, solid_capstyle="butt", zorder=2,
               path_effects=_casing(c, 0.8))
        a.plot([r[0.25], r[0.75]], [y, y], color=col, lw=3.0, solid_capstyle="butt", zorder=2,
               path_effects=_casing(c, 3.0))
        a.plot(r[0.5], y, "o", ms=4.4, color="white", mec=fs.INK, mew=0.8, zorder=3)
        a.plot(r[0.9], y, "|", ms=6.5, color=fs.INK, mew=1.0, zorder=3)
    top = ys[0]
    r0 = q.loc[crops[0]]
    lab = {"fontsize": fs.FS_ITEM, "color": fs.MUTED, "ha": "center", "va": "bottom"}
    arr = {"arrowstyle": "-", "lw": 0.3, "color": fs.MUTED}
    a.annotate("median", (r0[0.5], top + 0.12), xytext=(r0[0.5], top + 0.65), arrowprops=arr, **lab)
    a.annotate("90th percentile", (r0[0.9], top + 0.12), xytext=(r0[0.9], top + 0.65), arrowprops=arr, **lab)
    cb = crops[-1]
    a.text(env.loc[cb, "factor_oat_bmp_log"] - 0.03, ys[-1] - 0.5, "factors: BMP only", fontsize=fs.FS_ITEM,
           color=fs.MUTED, ha="right", va="top")
    a.text(env.loc[cb, "factor_envelope_log"] - 0.03, ys[-1] - 0.5, "all factors jointly", fontsize=fs.FS_ITEM,
           color=fs.MUTED, ha="right", va="top")
    a.set_yticks(ys, [fs.label(c) for c in crops])
    a.set_ylim(-1.0, len(crops) - 0.05)
    xmax = float(max(q[0.9].max(), env["factor_envelope_log"].max())) * 1.04
    a.set_xlim(0, xmax)
    _factor_axis(a, "x", 0, xmax, "Difference between N$_1$ from observed and from declared area, per "
                 "municipality (×, either direction)")
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    a.grid(axis="x", color=fs.FAINT, lw=0.4, zorder=0)
    ck.add("fig09_median_below_envelope", bool((q[0.5].reindex(crops) < env.loc[crops, "factor_envelope_log"]).all()),
           "median source effect below the factor envelope for every crop", severity="warn")
    return cv


# ---------------------------------------------------------------------------
# Fig 10 — C4: source effect by decile of municipal N1 (droppable)
# ---------------------------------------------------------------------------
def fig10_source_effect_deciles(cfg: Config, ck: CheckLog, data: _Data, host=None):
    n1 = data.get("n1_municipal")
    years = summary_years(cfg)
    d = source_effect_by_decile(n1, years)
    if not ck.add("fig10_deciles", len(d) == 10, f"{len(d)} deciles", severity="warn"):
        return None
    FW, FH = fs.HALF_W, 82.0
    cv = fs.canvas(FW, FH, host)
    a = _chart_axes(cv, 13.0, 40.0, 64.0, 39.0, grid_y=True)
    b = _chart_axes(cv, 13.0, 11.0, 64.0, 25.0)
    x = d["decile"].to_numpy()
    a.plot(x, d["p90_abs_log"], color=fs.MUTED, lw=0.8, ls=(0, (2.5, 1.2)), marker="o", ms=3.0, mfc="white",
           mec=fs.MUTED, mew=0.7)
    a.plot(x, d["median_abs_log"], color=fs.INK, lw=1.0, marker="o", ms=3.0)
    a.text(x[-1] + 0.35, d["p90_abs_log"].iloc[-1], "90th pct.", fontsize=fs.FS_ITEM, va="center", color=fs.MUTED)
    a.text(x[-1] + 0.35, d["median_abs_log"].iloc[-1], "median", fontsize=fs.FS_ITEM, va="center")
    ymax = float(d["p90_abs_log"].max()) * 1.08
    a.set_ylim(0, ymax)
    _factor_axis(a, "y", 0, ymax, "N$_1$ obs vs dec (×)")
    a.set_xlim(0.4, 10.6)
    a.set_xticks(x, [])
    b.bar(x, d["share_of_national_n1"], color="#9a9a9a", width=0.72)
    for xi, v in zip(x, d["share_of_national_n1"], strict=True):
        if v >= 0.04:
            b.text(xi, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=fs.FS_ITEM)
    b.set_ylim(0, float(d["share_of_national_n1"].max()) * 1.22)
    b.set_xlim(0.4, 10.6)
    b.yaxis.set_major_formatter("{x:.0%}")
    b.set_ylabel("Share of national N$_1$")
    b.set_xticks(x)
    b.set_xlabel("Decile of municipal N$_1$ (declared), smallest → largest", labelpad=2)
    return cv


# ---------------------------------------------------------------------------
# Supplement
# ---------------------------------------------------------------------------
def figS1_state_detail(cfg: Config, ck: CheckLog, data: _Data, host=None):
    """Fig 5 for SP and MG: national map with a dashed detail box, connectors, and the detail."""
    g, states, country, ext = data.base()
    units = data.get("unit_ratios_municipality")
    years = summary_years(cfg)
    min_area = float(cfg.section("reconcile").get("min_area_ha", 0))
    insets = [i for i in cfg.section("paper").get("fig4_insets", []) if i["crop"] in set(units["crop"])]
    if not insets:
        return None
    edges, colors, _ = ratio_classes()
    S, MW, MH = _crop_maps_layout(ext)
    det_w_max, det_h, row_gap, LB = fs.FULL_W - MW - 14.0, 44.0, 7.0, 21.0
    rows = []
    for ins in insets:
        x0, y0, x1, y1 = states[states["uf"] == ins["uf"]].total_bounds
        pad = 0.04 * max(x1 - x0, y1 - y0)
        win = (x0 - pad, x1 + pad, y0 - pad, y1 + pad)
        sd = nice_scale(max((win[1] - win[0]) / det_w_max, (win[3] - win[2]) / det_h))
        rows.append((ins, win, sd))
    heights = [max(MH, (w[3] - w[2]) / sd) for _, w, sd in rows]
    FW, FH = fs.FULL_W, LB + fs.STRIP + sum(h + row_gap for h in heights) + 1.0
    cv = fs.canvas(FW, FH, host)
    y = FH - 1.0
    for k, ((ins, win, sd), h) in enumerate(zip(rows, heights, strict=True)):
        y -= h + row_gap
        cls = classify(municipal_log_ratio(units, ins["crop"], years, min_area), edges)
        a = cv.map_ax(1.0, y + h - MH, ext, S)
        _plot_classes(a, g, cls, colors)
        boundaries(a, states, country, 0.25)
        detail_box(a, ((win[0], win[1]), (win[2], win[3])), lw=0.5)
        map_frame(a, ext)
        dw, dh = (win[1] - win[0]) / sd, (win[3] - win[2]) / sd
        dx, dy = 1.0 + MW + 12.0, y + h - dh
        d = cv.map_ax(dx, dy, win, sd, z=3)
        d.patch.set_alpha(1)
        d.patch.set_facecolor("white")
        gi = g.cx[win[0]:win[1], win[2]:win[3]]
        gi.plot(ax=d, color="white", edgecolor=fs.FAINT, linewidth=0.15, zorder=0.5)
        _plot_classes(d, gi, cls, colors, edge=0.08)
        boundaries(d, states, country, 0.35)
        state_labels(d, states, halo=True, window=((win[0], win[1]), (win[2], win[3])), fontsize=fs.FS_ITEM)
        map_frame(d, win)
        fs.detail_border(d)
        xb = 1.0 + (win[1] - ext[0]) / S                  # connectors: detail box -> detail
        for ym, yd in ((win[3], dy + dh), (win[2], dy)):
            cv.connect((xb, y + h - MH + (ym - ext[2]) / S), (dx, yd))
        cv.text(1.0, y + h + 0.8, f"({'abc'[k]}) {ins['uf']} — {fs.label(ins['crop'])}", fontsize=fs.FS_TITLE + 0.4,
                fontweight="bold", va="bottom")
        cv.scale(dx + dw + 2.0, dy + 1.5, sd, seg_km=100 if sd < 20000 else 200, n=1, bh=0.8)
    _ratio_legend(cv, 1.0, LB + fs.STRIP - 0.5, years, min_area)
    cv.strip(1.0, S, DATA_CREDIT, seg_km=1000)
    return cv


def figS2_rank_shift(cfg: Config, ck: CheckLog, data: _Data, host=None):
    rs = data.get("rank_shift")
    FW, FH = fs.FULL_W, 48.0
    cv = fs.canvas(FW, FH, host)
    for i, (col, title) in enumerate([("spearman_rho", "Spearman ρ, municipal N$_1$ obs vs dec"),
                                      ("top_decile_turnover", "Top-decile turnover")]):
        x0 = 11.0 + i * 61.0
        a = _chart_axes(cv, x0, 8.0, 53.0, 34.0, grid_y=True)
        a.plot(rs["year"], rs[col], color=fs.INK, lw=0.9, marker="o", ms=2.6)
        a.set_xticks([2008, 2012, 2016, 2020, 2024])
        if col == "top_decile_turnover":
            a.set_ylim(0, None)
            a.yaxis.set_major_formatter("{x:.0%}")
        cv.text(x0, 43.0, title, fontsize=fs.FS_LABEL, fontweight="bold", va="bottom")
    return cv


def _hexbin(ax, x, y, xlabel, ylabel):
    from matplotlib.colors import LinearSegmentedColormap

    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(x):
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no data", ha="center", va="center", color=fs.MUTED, transform=ax.transAxes)
        return
    lo = max(np.floor(min(np.nanpercentile(x, 0.5), np.nanpercentile(y, 0.5))), 0.0)
    hi = np.ceil(max(np.nanmax(x), np.nanmax(y)))
    cmap = LinearSegmentedColormap.from_list("grey", ["#d9d9d9", "#1a1a1a"])
    ax.hexbin(x, y, gridsize=34, cmap=cmap, mincnt=1, bins="log", linewidths=0, extent=(lo, hi, lo, hi),
              rasterized=True)
    ax.plot([lo, hi], [lo, hi], color=fs.INK, lw=0.5, ls=(0, (2.5, 1.2)))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    t = np.arange(lo, hi + 1, 1 if hi - lo <= 6 else 2)
    ax.set_xticks(t, [f"$10^{{{int(v)}}}$" for v in t])
    ax.set_yticks(t, [f"$10^{{{int(v)}}}$" for v in t])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def figS3_scatters(cfg: Config, ck: CheckLog, data: _Data, host=None):
    units = data.get("unit_ratios_municipality")
    n1 = data.get("n1_municipal")
    year = summary_years(cfg)[-1]
    FW, FH = fs.FULL_W, 64.0
    cv = fs.canvas(FW, FH, host)
    a = _chart_axes(cv, 12.0, 11.0, 47.0, 47.0)
    u = units[(units["crop"] == SECOND_CROP) & (units["year"] == year) & (units["presence"] == "both")]
    _hexbin(a, np.log10(u["a_dec_ha"]), np.log10(u["a_obs_ha"]), "Declared (ha)", "Observed (ha)")
    cv.text(12.0, 59.0, f"Second-season corn by municipality, {year}", fontsize=fs.FS_LABEL, fontweight="bold",
            va="bottom")
    b = _chart_axes(cv, 80.0, 11.0, 47.0, 47.0)
    w = n1[n1["year"] == year].pivot_table(index=["geocode", "crop"], columns="variant", values="n1_ref_nm3",
                                           observed=True).dropna(subset=["dec", "obs"])
    tot = w.groupby(level="geocode").sum()
    tot = tot[(tot["dec"] > 0) & (tot["obs"] > 0)]
    _hexbin(b, np.log10(tot["dec"]), np.log10(tot["obs"]), "N$_{1,dec}$ (Nm³ CH$_4$ yr$^{-1}$)",
            "N$_{1,obs}$ (Nm³ CH$_4$ yr$^{-1}$)")
    cv.text(80.0, 59.0, f"Municipal N$_1$, all crops, {year}", fontsize=fs.FS_LABEL, fontweight="bold", va="bottom")
    return cv


def figS4_ccc_scale(cfg: Config, ck: CheckLog, data: _Data, host=None):
    ag = data.get("agreement_by_scale")
    years = summary_years(cfg)
    crops = [c for c in cfg.crops if c in set(ag["crop"])]
    scales = [s for s in ("municipality", "amc", "rgint", "uf") if s in set(ag["scale"])]
    m = ag[ag["year"].isin(years)].groupby(["crop", "scale"])["lin_ccc"].mean()
    FW, FH = fs.HALF_W, 60.0
    cv = fs.canvas(FW, FH, host)
    a = _chart_axes(cv, 12.0, 10.0, 52.0, 47.0, grid_y=True)
    xs = np.arange(len(scales))
    ends = []
    for c in crops:
        y = [m.get((c, s), np.nan) for s in scales]
        fs.crop_line(a, xs, y, c)
        ends.append((fs.label(c), xs[-1] + 0.05, y[-1], fs.CROP_COLORS[c]))
    a.set_xticks(xs, [SCALE_LABELS[s].replace("Intermediate region", "Intermediate\nregion") for s in scales])
    a.set_xlim(-0.25, len(scales) - 0.75)
    lo = float(np.nanmin(m.values))
    a.set_ylim(np.floor(lo * 10) / 10, 1.005)
    a.set_ylabel(f"Lin's CCC (mean {years[0]}–{years[-1]})")
    fs.direct_labels(a, ends, x_pad=0.3, min_gap=(1 - lo) * 0.06, fontsize=fs.FS_ITEM)
    return cv


# ---------------------------------------------------------------------------
# Supplement: N1 by level (state, intermediate region, municipality) and its hot spots
# ---------------------------------------------------------------------------
N1_SHARE_Q = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]          # class breaks: quantiles of the municipal density
GI_CLASSES = [(2.576, "Hot spot, 99 %", "#b2182b"), (1.960, "Hot spot, 95 %", "#ef8a62"),
              (1.645, "Hot spot, 90 %", "#fddbc7"), (-1.645, "Not significant", "#f0f0f0"),
              (-1.960, "Cold spot, 90 %", "#d1e5f0"), (-2.576, "Cold spot, 95 %", "#67a9cf"),
              (-np.inf, "Cold spot, 99 %", "#2166ac")]
LISA_COLORS = {"HH": ("#c1121f", "High–high"), "LL": ("#2166ac", "Low–low"), "HL": ("#f4a582", "High–low"),
               "LH": ("#92c5de", "Low–high"), "": ("#f0f0f0", "Not significant")}


def _n1_density(cfg: Config, data: _Data):
    """Municipal N1 (observed area, CP2b reference factors) of the demo year, with the IBGE area (km²)."""
    year = int(cfg.section("demo").get("year", cfg.years[-1]))
    n1 = data.get("n1_municipal")
    attrs = data.get("municipality_attributes")
    if n1 is None or attrs is None or "area_km2_ibge" not in attrs or year not in set(n1["year"]):
        return None
    n1 = n1[(n1["year"] == year) & (n1["variant"] == "obs")]
    g = data.base()[0]
    g = g.assign(n1=g["geocode"].map(n1.groupby("geocode")["n1_ref_nm3"].sum()).fillna(0.0),
                 area=g["geocode"].map(attrs.set_index("geocode")["area_km2_ibge"]))
    return year, g


def figS_n1_levels(cfg: Config, ck: CheckLog, data: _Data, host=None):
    """N1 per km² at three levels on the same classes: the MAUP of the potential itself."""
    from matplotlib import colormaps

    got = _n1_density(cfg, data)
    if got is None:
        return None
    year, g = got
    _, states, country, ext = data.base()
    levels = [("uf", "State", g.dissolve(by="uf", aggfunc="sum", as_index=False)[["uf", "n1", "area", "geometry"]]),
              ("cd_rgint", "Intermediate region",
               g.dissolve(by="cd_rgint", aggfunc="sum", as_index=False)[["cd_rgint", "n1", "area", "geometry"]]),
              ("geocode", "Municipality", g)]
    tot = [float(lv["n1"].sum()) for _, _, lv in levels]
    ck.add("figS_n1_levels_add_up", max(abs(t / tot[-1] - 1) for t in tot) < 1e-9 if tot[-1] else False,
           f"N1 {tot[-1] / 1e9:.2f} bn Nm³ at every level", tot[-1])
    dens = (g["n1"] / g["area"] / 1e3).replace([np.inf, -np.inf], np.nan)
    edges = np.quantile(dens[dens > 0].dropna(), N1_SHARE_Q)
    cmap = colormaps["YlOrBr"]
    from matplotlib.colors import to_hex

    colors = ["#f7f7f7"] + [to_hex(cmap(v)) for v in np.linspace(0.12, 0.95, len(edges) + 1)]   # 0, then bins
    FW, gap = fs.FULL_W, 2.0
    w = (FW - 2.0 - 2 * gap) / 3
    S = nice_scale((ext[1] - ext[0]) / w)
    MW, MH = (ext[1] - ext[0]) / S, (ext[3] - ext[2]) / S
    TT, LEG = 4.4, 4 * fs.ROW + 7.0
    FH = MH + TT + LEG
    cv = fs.canvas(FW, FH, host)
    y0 = FH - MH - TT + 0.2
    for i, (key, title, lv) in enumerate(levels):
        x = 1.0 + i * (w + gap)
        a = cv.map_ax(x, y0, ext, S)
        d = (lv["n1"] / lv["area"] / 1e3).fillna(0)
        cls = np.where(d > 0, np.digitize(d, edges) + 1, 0)
        lv.plot(ax=a, color=[colors[k] for k in cls], linewidth=0, zorder=1)
        if key != "geocode":
            lv.boundary.plot(ax=a, color="white", linewidth=0.2, zorder=2)
        states.boundary.plot(ax=a, color="#4d4d4d", linewidth=0.25, zorder=3)
        country.boundary.plot(ax=a, color=fs.COUNTRY_EDGE, linewidth=0.4, zorder=3.2)
        map_frame(a, ext)
        cv.text(x, y0 + MH + 0.4, f"({'abc'[i]}) {title} ({len(lv):,})", fontsize=fs.FS_TITLE + 0.2,
                fontweight="bold", va="bottom")
        cv.panel(f"{'abc'[i]}_{key}", x, y0, x + MW, y0 + MH + TT - 0.2)
    labs = ["0", f"< {edges[0]:.2g}", *[f"{lo:.2g}–{hi:.2g}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)],
            f"≥ {edges[-1]:.2g}"]
    rows = [("patch", {"fc": c, "ec": "#7a7a7a"}, lab) for c, lab in zip(colors, labs, strict=True)]
    half = (len(rows) + 1) // 2
    cv.text(1.0, LEG - 1.0, "N1, thousand Nm³ CH4 per km² per year", fontsize=fs.FS_TITLE, va="top")
    cv.legend(1.0, LEG - 4.4, [("", rows[:half])], header=None)
    cv.legend(30.0, LEG - 4.4, [("", rows[half:])], header=None)
    cv.text(62.0, LEG - 4.4, f"Crop residues, observed area (MapBiomas Col. 11 {year}; 2nd-season corn: second-crop "
            "layer),\nCP2b reference factors. Classes: quantiles 20–95 % of the municipal density, the same\non "
            f"the three maps. IBGE areas and mesh 2025. {fs.CREDIT_CARTO}", fontsize=fs.FS_CRED - 0.2,
            color=fs.MUTED, va="top", linespacing=1.3)
    cv.panel("legend", 0.5, 0.0, FW - 0.5, LEG)
    data.facts["n1_levels"] = {"year": year, "edges": [float(e) for e in edges], "n1_total_nm3": tot[-1]}
    return cv


def figS_hot_spots(cfg: Config, ck: CheckLog, data: _Data, host=None):
    """Where N1 per km² clusters: (a) Getis–Ord Gi*, (b) local Moran (LISA) clusters, queen contiguity."""
    from bbp.analyse import getis_ord_gi_star, local_moran, morans_i, queen_neighbours

    got = _n1_density(cfg, data)
    if got is None:
        return None
    year, g = got
    _, states, country, ext = data.base()
    perms = int(cfg.section("paper").get("lisa_permutations", 999))
    alpha = 0.05
    dens = (g.set_index("geocode")["n1"] / g.set_index("geocode")["area"] / 1e3).replace([np.inf, -np.inf], np.nan)
    nb = queen_neighbours(g.set_index("geocode").geometry)
    islands = sum(not v for v in nb.values())
    gi = getis_ord_gi_star(dens, nb)
    lisa = local_moran(dens, nb, permutations=perms)
    glob = morans_i(dens, nb, permutations=perms)
    ck.add("figS_hot_spots_gi_finite", int(gi.notna().sum()) >= len(dens.dropna()) - islands,
           f"{int(gi.notna().sum())} of {len(dens.dropna())} municipalities with a Gi* z ({islands} without "
           "neighbours)", int(gi.notna().sum()))
    hot = int((gi >= 1.960).sum())
    cold = int((gi <= -1.960).sum())
    sig = lisa[lisa["p_value"] < alpha]
    counts = sig["quadrant"].value_counts().to_dict()
    ck.add("figS_hot_spots_global_moran", glob["p_value"] < alpha, f"Moran's I {glob['morans_i']:.3f} "
           f"(p = {glob['p_value']:.3f}); Gi* 95 %: {hot} hot, {cold} cold; LISA p < {alpha}: {counts}",
           glob["morans_i"], severity="warn")
    FW, gap = fs.FULL_W, 3.0
    w = (FW - 2.0 - gap) / 2
    S = nice_scale((ext[1] - ext[0]) / w)
    MW, MH = (ext[1] - ext[0]) / S, (ext[3] - ext[2]) / S
    TT, LEG = 4.4, 7 * fs.ROW + 4.0
    FH = MH + TT + LEG + 7.0
    cv = fs.canvas(FW, FH, host)
    y0 = FH - MH - TT + 0.2
    gcls = np.full(len(g), len(GI_CLASSES) - 1)
    z = g["geocode"].map(gi).to_numpy()
    for k in range(len(GI_CLASSES) - 1, -1, -1):
        gcls = np.where(z >= GI_CLASSES[k][0], k, gcls)
    gcls = np.where(np.isnan(z), 3, gcls)
    q = g["geocode"].map(sig["quadrant"]).fillna("")
    panels = [("a_gi_star", "(a) Getis–Ord Gi* of N1 per km²", [GI_CLASSES[k][2] for k in gcls]),
              ("b_lisa", "(b) Local Moran clusters (LISA)", [LISA_COLORS[v][0] for v in q])]
    for i, (name, title, cols) in enumerate(panels):
        x = 1.0 + i * (w + gap)
        a = cv.map_ax(x, y0, ext, S)
        g.plot(ax=a, color=cols, linewidth=0, zorder=1)
        states.boundary.plot(ax=a, color="#4d4d4d", linewidth=0.28, zorder=3)
        country.boundary.plot(ax=a, color=fs.COUNTRY_EDGE, linewidth=0.45, zorder=3.2)
        map_frame(a, ext)
        cv.text(x, y0 + MH + 0.4, title, fontsize=fs.FS_TITLE + 0.2, fontweight="bold", va="bottom")
        cv.panel(name, x, y0, x + MW, y0 + MH + TT - 0.2)
    cv.legend(1.0, y0 - 1.5, [("Gi* z-score (a)", [("patch", {"fc": c, "ec": "#7a7a7a"}, lab)
                                                   for _, lab, c in GI_CLASSES])], header=None)
    cv.legend(1.0 + w + gap, y0 - 1.5, [(f"LISA, p < {alpha:g} (b)", [("patch", {"fc": c, "ec": "#7a7a7a"}, lab)
                                                                        for c, lab in LISA_COLORS.values()])],
              header=None)
    cv.text(1.0, 0.6, f"Municipal N1 per km², observed area {year}, CP2b reference factors. Queen contiguity on the "
            f"2025 mesh; Gi*: binary weights with the unit itself,\nnormal z; LISA: row-standardised weights, "
            f"{perms} conditional permutations, no correction for multiple tests. Global Moran's I "
            f"{glob['morans_i']:.2f} (p = {glob['p_value']:.3f}). {fs.CREDIT_CARTO}", fontsize=fs.FS_CRED - 0.2,
            color=fs.MUTED, va="bottom", linespacing=1.3)
    data.facts["n1_hot_spots"] = {"year": year, "morans_i": glob["morans_i"], "p_value": glob["p_value"],
                                  "gi_hot_95": hot, "gi_cold_95": cold, "lisa": counts, "permutations": perms}
    return cv


# ---------------------------------------------------------------------------
# composites and figures drawn by bbp explore
# ---------------------------------------------------------------------------
def _explore(name: str, **kw):
    def draw(cfg: Config, ck: CheckLog, data: _Data, host=None):
        from bbp import explore

        return getattr(explore, name)(data.explore(), host=host, **kw)
    return draw


def fig_h2(cfg: Config, ck: CheckLog, data: _Data, host=None):
    """Fig 5: effects of the predictors (old Fig 8) over the variance shares and Moran's I (old Fig 7)."""
    st = fs.Stack(gap=3.0, ck=ck)
    st.add("a", lambda h: fig08_h2_effects(cfg, ck, data, host=h), head="(a) Standardised effects of the predictors")
    st.add("b", lambda h: fig07_h2_variance(cfg, ck, data, host=h),
           head="(b) Share of the variance explained; Moran's I of the residuals")
    return st.draw()


def fig_reach(cfg: Config, ck: CheckLog, data: _Data, host=None):
    """Fig 7: crop area by distance to the plants (x04) over the plants by family (x01)."""
    from bbp import explore

    ctx = data.explore()
    st = fs.Stack(gap=2.0, ck=ck)
    st.add("ab", lambda h: explore.x04_distance(ctx, host=h))
    st.add("cdefgh", lambda h: explore.x01_plants(ctx, host=h, start=2))
    return st.draw()


def fig_typology_mandate(cfg: Config, ck: CheckLog, data: _Data, host=None):
    from bbp import explore

    return explore.typology_mandate(data.explore(), host=host)


# the paper: one figure per conclusion step (outline v3); the supplement holds the rest
FIGURES = {1: ("study_area", fig01_study_area), 2: ("workflow", fig02_workflow),
           3: ("agreement_scale", fig04_agreement_scale), 4: ("log_ratio_maps", fig05_log_ratio_maps),
           5: ("h2_drivers", fig_h2), 6: ("source_vs_factors", fig09_source_vs_factors),
           7: ("distance_to_plants", fig_reach), 8: ("maup_access", _explore("x05_maup")),
           9: ("typology_mandate", fig_typology_mandate)}
# other renderings of a figure, drawn after it (only when their inputs exist), e.g. Fig 1 for 2025
VARIANTS = {1: lambda cfg: [int(y) for y in _fig1_spec(cfg).get("variants") or []]}
SUPPLEMENT = {"S1": ("national_area", fig03_national_area), "S2": ("example_municipalities", fig06_examples),
              "S3": ("state_detail", figS1_state_detail), "S4": ("rank_shift", figS2_rank_shift),
              "S5": ("scatters", figS3_scatters), "S6": ("ccc_scale", figS4_ccc_scale),
              "S7": ("source_effect_deciles", fig10_source_effect_deciles),
              "S8": ("sp_mills_catchments", _explore("x02_sp_mills")),
              "S9": ("mill_catchments", _explore("x03_mill_catchments")),
              "S10": ("regions_within_reach", _explore("x06_regions")),
              "S11": ("region_typology", _explore("x07_typology")),
              "S12": ("registers_biomethane", _explore("x08_registers")),
              "S13": ("n1_by_level", figS_n1_levels), "S14": ("n1_hot_spots", figS_hot_spots)}


def file_name(key) -> str:
    name = (FIGURES.get(key) or SUPPLEMENT[key])[0]
    return f"fig{key:02d}_{name}" if isinstance(key, int) else f"fig{key}_{name}"


def build(cfg: Config, ck: CheckLog, only: list[int] | None = None) -> list[Path]:
    from bbp.captions import write_captions

    family = fs.setup()
    ck.add("figure_font", family in ("Arial", "Liberation Sans", "Helvetica"), f"text in {family!r}", family,
           severity="warn")
    spec = cfg.section("paper")
    folder = out_dir(cfg) / "figures"
    formats = spec.get("figure_formats", ["pdf", "svg", "png", "tif"])
    dpi = int(spec.get("dpi", 600))
    data = _Data(cfg, ck)
    keys = [k for k in FIGURES if not only or k in only] + ([] if only else list(SUPPLEMENT))
    written = []
    for key in keys:
        name, func = FIGURES.get(key) or SUPPLEMENT[key]
        variants = VARIANTS.get(key, lambda c: [])(cfg)
        for variant in [None, *variants]:          # e.g. Fig 1 for another year: fig01_study_area_2025
            log.info("Figure %s: %s%s", key, name, f" ({variant})" if variant else "")
            tag = f"fig{key}" + (f"_{variant}" if variant else "")
            try:
                cv = func(cfg, ck, data) if variant is None else func(cfg, ck, data, year=variant)
            except Exception as exc:  # one broken figure must not stop the others
                log.error("%s", traceback.format_exc())
                ck.add(f"{tag}_built", False, f"{type(exc).__name__}: {exc}")
                continue
            if cv is None:
                ck.add(f"{tag}_built", False, "inputs missing: figure left out", severity="warn")
                continue
            ck.add(f"{tag}_final_size", cv.fw <= 132.5 and cv.fh <= fs.MAX_H, f"{cv.fw:.1f} × {cv.fh:.1f} mm",
                   [round(cv.fw, 1), round(cv.fh, 1)], severity="warn")
            stem = file_name(key) + (f"_{variant}" if variant else "")
            paths = cv.save(folder, stem, formats, dpi, panels_dir=out_dir(cfg) / "panels" / stem)
            ck.add(f"{tag}_written", all(p.stat().st_size > 0 for p in paths), ", ".join(p.name for p in paths))
            if spec.get("legibility_check", True):
                for p in paths:
                    if p.suffix == ".png":
                        fs.simulate(p, out_dir(cfg) / "figures_check")
            written += paths
    if not only:                                   # a full run: drop figures of an older numbering
        import shutil

        stems = {p.stem for p in written}
        stale = [p for p in folder.glob("fig*") if p.is_file() and p.stem not in stems]
        stale_panels = [p for p in (out_dir(cfg) / "panels").glob("fig*") if p.is_dir() and p.name not in stems]
        for p in stale:
            p.unlink()
        for p in stale_panels:
            shutil.rmtree(p)
        ck.add("figures_stale_removed", True, f"{len(stale)} files, {len(stale_panels)} panel folders of figures "
               "not in this run", len(stale), severity="warn")
    write_captions(cfg, data, out_dir(cfg) / "captions.md")
    return written
