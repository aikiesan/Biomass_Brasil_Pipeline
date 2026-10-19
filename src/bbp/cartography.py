"""Base layers and map helpers in the NIPE style (see bbp.figstyle).

States come from the IBGE municipal mesh: dissolved by UF, generalised with coverage-aware
simplification (shared borders stay shared), islets below 150 km² dropped; the national outline is the
union of the states with hairline gaps closed. Everything is in SIRGAS 2000 / Brazil Polyconic
(EPSG:5880, metres), for display only; bbp computes every area geodesically.
"""

from __future__ import annotations

import numpy as np

from bbp import figstyle as fs

MAP_CRS = "EPSG:5880"


def base_layers(municipalities, tolerance_m: float = 2500.0, min_islet_m2: float = 150e6):
    """(states GeoDataFrame [uf, geometry], country GeoSeries) from projected municipal polygons."""
    import geopandas as gpd
    import shapely
    from shapely.geometry import MultiPolygon, Polygon

    st = municipalities.dissolve(by="uf", as_index=False)[["uf", "geometry"]]
    if hasattr(shapely, "coverage_simplify"):
        st["geometry"] = shapely.coverage_simplify(st.geometry.values, tolerance=tolerance_m)
    parts = st.explode(index_parts=False)
    st = parts[parts.area > min_islet_m2].dissolve(by="uf", as_index=False)[["uf", "geometry"]]
    u = st.buffer(400).union_all().buffer(-400)
    u = MultiPolygon([Polygon(g.exterior) for g in getattr(u, "geoms", [u])])
    country = gpd.GeoSeries([u], crs=st.crs)
    return st, country


def extent(geoms, pad_m: float = 20e3) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = geoms.total_bounds
    return x0 - pad_m, x1 + pad_m, y0 - pad_m, y1 + pad_m


def boundaries(a, states, country, lw: float = 0.3):
    states.boundary.plot(ax=a, color=fs.STATE_EDGE, linewidth=lw, zorder=3)
    country.boundary.plot(ax=a, color=fs.COUNTRY_EDGE, linewidth=lw * 1.7, zorder=3.2)


def state_labels(a, states, halo: bool = True, window=None, fontsize: float = fs.FS_STATE, z: float = 13):
    """UF abbreviations at a representative point (DF nudged off Goiás); skip labels near the window edge."""
    import matplotlib.patheffects as pe

    for _, r in states.iterrows():
        p = r.geometry.representative_point()
        x, y = p.x + (60e3 if r.uf == "DF" else 0), p.y + (45e3 if r.uf == "DF" else 0)
        if window:
            (x0, x1), (y0, y1) = window
            mx, my = 0.06 * (x1 - x0), 0.06 * (y1 - y0)
            if not (x0 + mx < x < x1 - mx and y0 + my < y < y1 - my):
                continue
        a.text(x, y, r.uf, fontsize=fontsize, color=fs.STATE_LABEL, ha="center", va="center", zorder=z,
               path_effects=[pe.withStroke(linewidth=1.2, foreground="white", alpha=0.85)] if halo else None)


def cased_lines(a, geoms, lw: float, color: str = "#8c8c8c", casing: str = "white", casing_lw: float | None = None,
                z: float = 3.0, ls="-"):
    """Boundary lines with a white casing underneath, so they read on dark (crops) and light (neutrals) ground."""
    geoms.boundary.plot(ax=a, color=casing, linewidth=casing_lw if casing_lw is not None else lw * 3, zorder=z)
    geoms.boundary.plot(ax=a, color=color, linewidth=lw, linestyle=ls, zorder=z + 0.05)


# ---------------------------------------------------------------------------
# land cover (bbp.landcover display rasters)
# ---------------------------------------------------------------------------
def landcover_rgba(colors: dict[str, str]) -> np.ndarray:
    """(256, 4) uint8 lookup: display code -> RGBA; 0 and unknown codes transparent."""
    from matplotlib.colors import to_rgba

    from bbp import landcover as lc

    lut = np.zeros((256, 4), dtype=np.uint8)
    for key, code in lc.CODE.items():
        lut[code] = np.round(np.array(to_rgba(colors[key])) * 255).astype(np.uint8)
    return lut


def landcover_on_map(src, extent: tuple[float, float, float, float], res_m: float, crs: str = MAP_CRS,
                     resampling: str = "nearest") -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Display codes of a land-cover raster resampled onto a `res_m` grid covering `extent` (x0, x1, y0, y1)
    in `crs`: nearest, or "mode" when the grid is coarser than the raster. Returns (codes, the grid's own
    extent); row 0 is the north edge."""
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import Window, from_bounds

    x0, x1, y0, y1 = extent
    w, h = int(np.ceil((x1 - x0) / res_m)), int(np.ceil((y1 - y0) / res_m))
    out = np.zeros((h, w), dtype=np.uint8)
    with rasterio.open(src) as ds:
        b = transform_bounds(crs, ds.crs, x0, y0, x1, y1, densify_pts=101)
        win = from_bounds(*b, transform=ds.transform)
        win = Window(int(np.floor(win.col_off)) - 2, int(np.floor(win.row_off)) - 2,
                     int(np.ceil(win.width)) + 4, int(np.ceil(win.height)) + 4)
        win = win.intersection(Window(0, 0, ds.width, ds.height))
        arr = ds.read(1, window=win)
        reproject(arr, out, src_transform=ds.window_transform(win), src_crs=ds.crs,
                  dst_transform=Affine(res_m, 0, x0, 0, -res_m, y1), dst_crs=crs,
                  resampling=getattr(Resampling, resampling), src_nodata=0, dst_nodata=0)
    return out, (x0, x0 + w * res_m, y1 - h * res_m, y1)


def draw_landcover(a, codes: np.ndarray, grid_extent, lut: np.ndarray, z: float = 1.0, clip=None):
    """The codes as an image (nearest, rasterised in vector output) on their grid extent; `clip`: a
    (multi)polygon in map coordinates outside which the image is not drawn."""
    im = a.imshow(lut[codes], extent=grid_extent, origin="upper", interpolation="nearest", zorder=z,
                  rasterized=True, aspect="auto")
    if clip is not None:
        im.set_clip_path(polygon_patch(clip, a.transData))
    return im


def ghost_lut(lut: np.ndarray, strength: float = 0.32) -> np.ndarray:
    """A light greyscale version of an RGBA lookup: context drawn behind the colour of a detail."""
    out = lut.copy()
    rgb = lut[:, :3].astype(float) / 255
    lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    grey = 1 - strength * (1 - lum)                       # pull every tone towards white
    shown = lut[:, 3] > 0
    out[shown, :3] = np.round(np.clip(grey[shown], 0, 1)[:, None] * 255).astype(np.uint8)
    return out


def polygon_patch(geom, transform):
    """A matplotlib PathPatch of a shapely (multi)polygon, holes included (for clip paths)."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    verts, codes = [], []
    for poly in getattr(geom, "geoms", [geom]):
        for ring in [poly.exterior, *poly.interiors]:
            xy = np.asarray(ring.coords)
            verts.append(xy)
            codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY]
    path = MplPath(np.concatenate(verts), codes)
    return PathPatch(path, transform=transform, facecolor="none", edgecolor="none")


def detail_box(a, window, lw: float = 0.6):
    from matplotlib.patches import Rectangle

    (x0, x1), (y0, y1) = window
    a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=fs.DETAIL, ls=fs.DETAIL_LS, lw=lw, zorder=12))


def _transformer(crs: str):
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def _fmt_lon(v: float) -> str:
    return "0°" if v == 0 else f"{abs(v):g}°{'W' if v < 0 else 'E'}"


def _fmt_lat(v: float) -> str:
    return "0°" if v == 0 else f"{abs(v):g}°{'S' if v < 0 else 'N'}"


def graticule(ax, crs: str = MAP_CRS, step: float = 10.0, lines: bool = True,
              sides: tuple[str, ...] = ("bottom", "left"), fontsize: float = fs.FS_SCALE,
              lon_range=(-90, -20), lat_range=(-40, 10)):
    """Optional frame ticks: meridians/parallels every `step` degrees, labelled where they meet the frame.
    Not used by the NIPE style (projection credit instead); kept for figures that need coordinates."""
    tr = _transformer(crs)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lons = np.arange(np.ceil(lon_range[0] / step) * step, lon_range[1] + 1e-9, step)
    lats = np.arange(np.ceil(lat_range[0] / step) * step, lat_range[1] + 1e-9, step)
    dense_lat = np.linspace(lat_range[0] - 5, lat_range[1] + 5, 400)
    dense_lon = np.linspace(lon_range[0] - 5, lon_range[1] + 5, 400)
    ticks = {s: ([], []) for s in ("bottom", "top", "left", "right")}
    for lon in lons:
        x, y = tr.transform(np.full_like(dense_lat, lon), dense_lat)
        if lines:
            ax.plot(x, y, color=fs.HAIR, lw=0.25, zorder=0.5)
        for side, yy in (("bottom", y0), ("top", y1)):
            xs = float(np.interp(yy, y, x)) if y[0] <= yy <= y[-1] else np.nan
            if x0 < xs < x1:
                ticks[side][0].append(xs)
                ticks[side][1].append(_fmt_lon(lon))
    for lat in lats:
        x, y = tr.transform(dense_lon, np.full_like(dense_lon, lat))
        if lines:
            ax.plot(x, y, color=fs.HAIR, lw=0.25, zorder=0.5)
        for side, xx in (("left", x0), ("right", x1)):
            ys = float(np.interp(xx, x, y)) if x[0] <= xx <= x[-1] else np.nan
            if y0 < ys < y1:
                ticks[side][0].append(ys)
                ticks[side][1].append(_fmt_lat(lat))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    kw = {"length": 2, "width": 0.5, "color": fs.INK, "labelsize": fontsize, "pad": 1.5}
    if "bottom" in sides:
        ax.set_xticks(ticks["bottom"][0], ticks["bottom"][1])
        ax.tick_params(axis="x", bottom=True, labelbottom=True, **kw)
    if "left" in sides:
        ax.set_yticks(ticks["left"][0], ticks["left"][1], rotation=90, va="center")
        ax.tick_params(axis="y", left=True, labelleft=True, **kw)
    return ticks
