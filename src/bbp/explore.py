"""Exploratory figures: the processing plants and what lies around them (`bbp explore`).

Drawn in the house style (bbp.figstyle, NIPE cartography) at final size, so any of them can move into the
paper as is; every element is also written on its own (data/paper/explore/panels/<figure>/), with its place
in layout.json, for assembling figures by hand. Inputs: data/interim/plants.parquet (bbp.plants) and the
reach tables (bbp.reach); the land cover of Fig 1 as ground.

  x01  plants by family over the land cover each one draws on (Brazil, six maps)
  x02  São Paulo sugarcane mills: catchments and the radius each mill needs (after Cervi et al. 2019)
  x03  mill catchments: required radius, cane area by radius, supply ratio, competition
  x04  distance from every crop cell to the nearest plant: curves and shares within 10–100 km
  x05  the MAUP access test: the same land "within reach" at pixel, municipality and region level
  x06  intermediate regions: share of crop area within reach (maps)
  x07  intermediate-region typology: crop area vs share within reach, plants
  x08  registers compared (ANP, CONAB, MapBiomas/EPE) and the biomethane plants against the mandate
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bbp import figstyle as fs
from bbp.cartography import MAP_CRS
from bbp.checks import CheckLog
from bbp.config import Config
from bbp.figstyle import Canvas, map_frame

log = logging.getLogger("bbp")

PLANT_INK = "#141414"
CONSTRUCTION = "white"
PAIR_LABEL = {
    "sugarcane|sugarcane_mill": "Sugarcane → sugarcane mill",
    "sugarcane|biomass_power|bagasse": "Sugarcane → bagasse power plant",
    "corn_2nd|corn_ethanol": "2nd-season corn → corn ethanol",
    "soybean|biodiesel": "Soybean → biodiesel plant",
    "soy_all|biogas": "Soybean → biogas plant",
    "pasture|slaughterhouse": "Pasture → slaughterhouse",
    "all_crops|biogas": "All crops → biogas plant",
    "all_crops|biomethane": "All crops → biomethane plant",
}
PAIR_SHORT = {
    "sugarcane|sugarcane_mill": "Cane → mill", "sugarcane|biomass_power|bagasse": "Cane → bagasse power",
    "corn_2nd|corn_ethanol": "2nd corn → corn ethanol", "soybean|biodiesel": "Soybean → biodiesel",
    "soy_all|biogas": "Soybean → biogas", "pasture|slaughterhouse": "Pasture → slaughterhouse",
    "all_crops|biogas": "All crops → biogas", "all_crops|biomethane": "All crops → biomethane",
}
PAIR_COLOR = {
    "sugarcane|sugarcane_mill": fs.CROP_COLORS["sugarcane"], "sugarcane|biomass_power|bagasse": "#7fc9a8",
    "corn_2nd|corn_ethanol": fs.CROP_COLORS["corn_2nd"], "soybean|biodiesel": fs.CROP_COLORS["soybean"],
    "soy_all|biogas": "#7fb3d9", "pasture|slaughterhouse": "#a0784a", "all_crops|biogas": "#555555",
    "all_crops|biomethane": "#b0b0b0",
}
PAIR_LS = {"sugarcane|biomass_power|bagasse": (0, (3, 1.2)), "soy_all|biogas": (0, (3, 1.2)),
           "all_crops|biomethane": (0, (1, 1))}
LEVEL_STYLE = {"pixel": ("#1a1a1a", "-", "o"), "municipality": ("#c1121f", (0, (3, 1.2)), "s"),
               "intermediate region": ("#6a6a6a", (0, (1, 1)), "^")}
EXPLORE = {}                                             # key: (file name, function)


def register(key: str, name: str):
    def deco(func):
        EXPLORE[key] = (name, func)
        return func
    return deco


# ---------------------------------------------------------------------------
# context loaded once
# ---------------------------------------------------------------------------
class Ctx:
    def __init__(self, cfg: Config, ck: CheckLog):
        from bbp.figures import _Data, landcover_display
        from bbp.plants import load_plants

        self.cfg, self.ck = cfg, ck
        self.data = _Data(cfg, ck)
        self.g, self.states, self.country, self.ext = self.data.base()
        self.year = int((cfg.section("paper").get("fig1") or {}).get("year", cfg.years[-1]))
        self.disp = landcover_display(cfg, ck, self.year, required=False)
        p = load_plants(cfg)
        self.plants = None
        if p is not None:
            p = p.to_crs(MAP_CRS)
            p["x"], p["y"] = p.geometry.x, p.geometry.y
            p["family"] = p["family"].astype(str)
            self.plants = p
        self.proc = cfg.data_dir / "processed"
        self._t = {}

    def t(self, name: str) -> pd.DataFrame | None:
        if name not in self._t:
            p = self.proc / f"reach_{name}.parquet"
            self._t[name] = pd.read_parquet(p) if p.exists() else None
        return self._t[name]

    def regions(self):
        if not hasattr(self, "_rg"):
            import shapely

            rg = self.g.dissolve(by="cd_rgint", as_index=False)[["cd_rgint", "geometry"]]
            if hasattr(shapely, "coverage_simplify"):          # shared borders stay shared; ~1/3 mm at 1:80M
                rg["geometry"] = shapely.coverage_simplify(rg.geometry.values, tolerance=3000.0)
            self._rg = rg
        return self._rg


def focus_lut(keys, strength: float = 0.30, override: dict | None = None) -> np.ndarray:
    """Land-cover lookup with `keys` in colour and everything else as light grey context."""
    from matplotlib.colors import to_rgba

    from bbp import landcover as lc
    from bbp.cartography import ghost_lut, landcover_rgba

    lut = landcover_rgba(fs.LANDCOVER_COLORS)
    out = ghost_lut(lut, strength)
    for k in keys:
        col = (override or {}).get(k)
        out[lc.CODE[k]] = np.round(np.array(to_rgba(col)) * 255).astype(np.uint8) if col else lut[lc.CODE[k]]
    return out


def base_map(ctx: Ctx, a, ext, S, lut, state_lw: float = 0.25, regions: bool = False):
    from bbp.figures import _landcover_layer

    if not _landcover_layer(a, ctx.disp, ext, S, lut):
        ctx.country.plot(ax=a, color="#efefef", linewidth=0, zorder=1)
    if regions:
        ctx.regions().boundary.plot(ax=a, color="white", linewidth=state_lw * 0.6, zorder=2.6)
    ctx.states.boundary.plot(ax=a, color="#6b6b6b", linewidth=state_lw, zorder=3)
    ctx.country.boundary.plot(ax=a, color=fs.COUNTRY_EDGE, linewidth=state_lw * 1.6, zorder=3.2)
    map_frame(a, ext)


def size_of(cap, ref: float, s_ref: float = 9.0, s_min: float = 1.6):
    """Marker area (pt²) proportional to capacity; `ref` capacity gets `s_ref`."""
    c = np.asarray(cap, dtype=float)
    return np.where(np.isfinite(c) & (c > 0), np.maximum(s_ref * c / ref, s_min), s_min * 1.4)


def plant_points(a, df, sizes, marker="o", z=6, lw=0.3):
    op = df["operating"].fillna(True).astype(bool).to_numpy()
    for flag, fc in ((True, PLANT_INK), (False, CONSTRUCTION)):
        m = op == flag
        if m.any():
            a.scatter(df["x"].to_numpy()[m], df["y"].to_numpy()[m], s=np.asarray(sizes)[m], marker=marker, c=fc,
                      edgecolors="white" if flag else PLANT_INK, linewidths=lw if flag else 0.45, zorder=z)


def flow_legend(cv: Canvas, x: float, y_top: float, width: float, entries, nrow: int | None = None,
                fontsize: float = fs.FS_ITEM - 0.4):
    """Entries [(patch, colour, label) | (marker, (marker, size pt²), label)] in two columns (or one)."""
    from matplotlib.patches import Rectangle

    nrow = nrow or 4
    cw = width / 2
    for k, (kind, spec, text) in enumerate(entries):
        cx, cy = x + (k // nrow) * cw, y_top - (k % nrow) * fs.ROW
        if kind == "patch":
            cv.ov.add_patch(Rectangle((cx, cy - 0.75), 3.4, 1.5, fc=spec, ec="#7a7a7a", lw=0.35))
        else:
            mk, sz = spec
            cv.ov.scatter([cx + 1.7], [cy], s=sz, marker=mk, c=PLANT_INK, edgecolors="white", linewidths=0.3)
        cv.text(cx + 4.4, cy, text, fontsize=fontsize, va="center")


def panel_title(cv: Canvas, x, y, text, size=None):
    cv.text(x, y, text, fontsize=size or fs.FS_TITLE + 0.2, fontweight="bold", va="bottom")


def letter(i: int) -> str:
    return "abcdefghijkl"[i]


def axis_style(ax, grid_y: bool = True):
    ax.tick_params(length=2)
    if grid_y:
        ax.grid(axis="y", color=fs.FAINT, lw=0.4)


def _pct(v: float) -> str:
    return f"{100 * v:.0f} %"


# ---------------------------------------------------------------------------
# x01 — plants by family, Brazil
# ---------------------------------------------------------------------------
X01_PANELS = [
    # family, title, focus groups, override colours, size reference (capacity, label), subtype markers
    ("sugarcane_mill", "Sugarcane mills (ANP)", ("sugarcane",), None, (3.0e6, "3 Mt cane/season"), None),
    ("corn_ethanol", "Corn ethanol plants (ANP)", ("soy_corn2",), None, (5.0e5, "0.5 Mt corn/yr"), None),
    ("biodiesel", "Biodiesel plants (ANP)", ("soybean", "soy_corn2", "soy_cotton2"),
     {"soy_corn2": fs.CROP_COLORS["soybean"], "soy_cotton2": fs.CROP_COLORS["soybean"]}, (1000, "1,000 m³/day"), None),
    ("biogas", "Biogas plants (CIBiogás)", ("soybean", "soy_corn2", "soy_cotton2", "sugarcane", "coffee", "citrus",
                                             "cotton", "rice", "other_crops"), None, None,
     {"agriculture": "o", "industry": "s", "urban": "^"}),
    ("biomass_power", "Biomass power plants (ANEEL)", ("sugarcane", "plantation"),
     {"plantation": "#7a6a4f"}, (5.0e4, "50 MW"), {"bagasse": "^", "forestry": "s", "biogas": "o",
                                                    "urban": "D", "agro_other": "v"}),
    ("slaughterhouse", "Slaughterhouses (SIF)", ("pasture",), {"pasture": "#c9a66b"},
     None, None),
]
S_REF = {"sugarcane_mill": 5.0, "corn_ethanol": 5.0, "biodiesel": 7.0, "biomass_power": 4.0}
SUBTYPE_LABEL = {"agriculture": "Agro-livestock", "industry": "Industry", "urban": "Urban waste, sewage",
                 "bagasse": "Bagasse", "forestry": "Wood, liquor", "biogas": "Biogas",
                 "agro_other": "Other agro"}


@register("x01", "x01_plants_by_family")
def x01_plants(ctx: Ctx):
    from bbp.figures import nice_scale

    if ctx.plants is None:
        return None
    P = ctx.plants
    ext = ctx.ext
    FW, gap, ncol = fs.FULL_W, 2.0, 3
    w = (FW - 2.0 - gap * (ncol - 1)) / ncol
    S = nice_scale((ext[1] - ext[0]) / w)
    MW, MH = (ext[1] - ext[0]) / S, (ext[3] - ext[2]) / S
    TT, LG = 4.2, 4 * fs.ROW + 1.6
    FH = 2 * (MH + TT + LG) + 9.0
    cv = Canvas(FW, FH)
    for i, (fam, title, focus, override, ref, subs) in enumerate(X01_PANELS):
        col, row = i % ncol, i // ncol
        x = 1.0 + col * (w + gap) + (w - MW) / 2
        y = FH - (row + 1) * (MH + TT) - row * LG + 0.3
        a = cv.map_ax(x, y, ext, S)
        base_map(ctx, a, ext, S, focus_lut(focus, override=override), state_lw=0.18)
        df = P[P["family"] == fam].sort_values("capacity", ascending=False, na_position="last")
        sizes = size_of(df["capacity"], ref[0], S_REF.get(fam, 9.0)) if ref else np.full(len(df), 2.6)
        if subs:
            for st, mk in subs.items():
                m = (df["subtype"] == st).to_numpy()
                plant_points(a, df[m], sizes[m], marker=mk)
        else:
            plant_points(a, df, sizes)
        n_op = int(df["operating"].fillna(True).astype(bool).sum())
        panel_title(cv, x, y + MH + 0.4, f"({letter(i)}) {title}")
        # legend under the map, in two columns
        lab = {"sugarcane": "Sugarcane", "soy_corn2": "Soy + 2nd-season corn", "soybean": "Soybean, all seasons",
               "pasture": "Pasture", "plantation": "Forest plantation"}
        entries = [("patch", (override or {}).get(k, fs.LANDCOVER_COLORS[k]), lab[k]) for k in focus if k in lab]
        if fam == "biodiesel":
            entries = entries[:1]
        if fam == "biogas":
            entries = [("patch", fs.LANDCOVER_COLORS["soybean"], "Crops (Fig 1 colours)")]
        for st, mk in (subs or {}).items():
            k = int((df["subtype"] == st).sum())
            if k:
                entries.append(("marker", (mk, 5.0), f"{SUBTYPE_LABEL[st]} ({k})"))
        if ref:
            entries.append(("marker", ("o", float(size_of([ref[0]], ref[0], S_REF.get(fam, 9.0))[0])), ref[1]))
        flow_legend(cv, x, y - 1.6, MW, entries)
        cv.text(x + MW - 0.5, y + MH - 0.5, f"{len(df)} plants\n{n_op} operating", fontsize=fs.FS_ITEM - 0.3,
                ha="right", va="top", color=fs.MUTED, linespacing=1.2)
        cv.panel(f"{letter(i)}_{fam}", x, y - LG + 0.4, x + MW, y + MH + TT - 0.3)
    # strip: plant symbols, scale, credits
    y0 = 0.4
    cv.north(4.0, y0 + 3.4)
    cv.scale(7.5, y0 + 3.2, S, seg_km=1000, n=1, bh=0.7)
    ox = 7.5 + 1000e3 / S + 8.0
    cv.ov.scatter([ox], [y0 + 6.6], s=6, c=PLANT_INK, edgecolors="white", linewidths=0.3)
    cv.text(ox + 1.8, y0 + 6.6, "Operating (marker area proportional to capacity)", va="center")
    cv.ov.scatter([ox + 62.0], [y0 + 6.6], s=6, c=CONSTRUCTION, edgecolors=PLANT_INK, linewidths=0.45)
    cv.text(ox + 63.8, y0 + 6.6, "Construction, expansion, inactive", va="center")
    cv.text(ox, y0 + 3.6, f"Ground: MapBiomas Col. 11 {ctx.year}, the crops each family draws on in colour. "
            "Data: ANP; CIBiogás, ANEEL (via MapBiomas); MAPA SIF.", va="center",
            fontsize=fs.FS_CRED, color=fs.MUTED)
    cv.text(ox, y0 + 1.2, fs.CREDIT_PROJ + ". " + fs.CREDIT_CARTO, va="center", fontsize=fs.FS_CRED,
            color=fs.MUTED)
    cv.panel("legend_strip", 0.5, 0.0, FW - 0.5, 9.0)
    return cv


# ---------------------------------------------------------------------------
# x02 — São Paulo mills: catchment circles and required radius
# ---------------------------------------------------------------------------
RADIUS_CLASSES = [(0, 10, "≤ 10 km"), (10, 20, "10–20 km"), (20, 30, "20–30 km"), (30, 50, "30–50 km"),
                  (50, np.inf, "> 50 km")]
RADIUS_COLORS = ["#08306b", "#2f7fbf", "#8cc2e0", "#f4a259", "#c1121f"]


@register("x02", "x02_sp_mills_catchments")
def x02_sp_mills(ctx: Ctx):
    import geopandas as gpd

    from bbp.figures import nice_scale

    mills = ctx.t("mills")
    if mills is None or ctx.plants is None:
        return None
    radius = 20.0
    sp = ctx.states[ctx.states["uf"] == "SP"]
    b = sp.total_bounds
    pad = 15e3
    ext = (b[0] - pad, b[2] + pad, b[1] - pad, b[3] + pad)
    FW, gap = fs.FULL_W, 2.0
    w = (FW - 2.0 - gap) / 2
    S = nice_scale((ext[1] - ext[0]) / w, 500.0)
    MW, MH = (ext[1] - ext[0]) / S, (ext[3] - ext[2]) / S
    TT = 4.2
    LEG = 17.0
    FH = MH + TT + LEG
    cv = Canvas(FW, FH)
    m = gpd.GeoDataFrame(mills, geometry=gpd.points_from_xy(mills["lon"], mills["lat"]), crs="EPSG:4674")
    m = m.to_crs(MAP_CRS)
    m["x"], m["y"] = m.geometry.x, m.geometry.y
    m = m[(m["x"] > ext[0]) & (m["x"] < ext[1]) & (m["y"] > ext[2]) & (m["y"] < ext[3])]
    op = ctx.plants.set_index("plant_id")["operating"]
    m["operating"] = m["plant_id"].map(op)
    in_sp = m[m["uf"] == "SP"]
    lut_c = focus_lut(("sugarcane",))
    focus = sp.union_all()
    from bbp.figures import _landcover_layer

    for i in range(2):
        x = 1.0 + i * (w + gap)
        y = LEG
        a = cv.map_ax(x, y, ext, S)
        if not _landcover_layer(a, ctx.disp, ext, S, lut_c, focus=focus):
            ctx.country.plot(ax=a, color="#efefef", lw=0)
        ctx.g.cx[ext[0]:ext[1], ext[2]:ext[3]].boundary.plot(ax=a, color="white", linewidth=0.1, zorder=2.5)
        ctx.states.boundary.plot(ax=a, color="#3d3d3d", linewidth=0.4, zorder=3.4)
        if i == 0:
            buf = gpd.GeoSeries(gpd.points_from_xy(m["x"], m["y"]), crs=MAP_CRS).buffer(radius * 1000, 64)
            buf.boundary.plot(ax=a, color=PLANT_INK, linewidth=0.18, alpha=0.55, zorder=4)
            plant_points(a, m.assign(x=m["x"], y=m["y"]), size_of(m["capacity"], 3.0e6, 7.0), z=6)
            panel_title(cv, x, y + MH + 0.4, f"(a) Sugarcane mills and {radius:.0f} km catchments")
        else:
            rr = m["required_radius_km"].to_numpy()
            for (lo, hi, _lab), col in zip(RADIUS_CLASSES, RADIUS_COLORS, strict=True):
                sel = (rr > lo) & (rr <= hi) if lo > 0 else (rr <= hi)
                a.scatter(m["x"][sel], m["y"][sel], s=size_of(m["capacity"][sel], 3.0e6, 7.0), c=col,
                          edgecolors="white", linewidths=0.3, zorder=6)
            na = ~np.isfinite(rr) & m["capacity"].isna().to_numpy()
            a.scatter(m["x"][na], m["y"][na], s=2.2, c="#9a9a9a", edgecolors="white", linewidths=0.2, zorder=5)
            panel_title(cv, x, y + MH + 0.4, "(b) Radius each mill needs for its crushing class")
        map_frame(a, ext)
        cv.panel(f"{letter(i)}_{'catchments' if i == 0 else 'required_radius'}", x, y, x + MW, y + MH + TT - 0.3)
    # legend row
    ly = LEG - 4.0
    rows_a = [("patch", {"fc": fs.LANDCOVER_COLORS["sugarcane"], "ec": "#7a7a7a"}, f"Sugarcane {ctx.year} (SP)"),
              ("line", (PLANT_INK, 0.3), f"{radius:.0f} km around each mill")]
    cv.legend(1.0, ly, [("", rows_a)], header=None)
    for k, (frac, lab) in enumerate(((1 / 3, "≤ 1.5 Mt"), (1.0, "2.5–3.5 Mt"), (6.5 / 3, "> 5.5 Mt"))):
        cx = 36.0 + k * 9.0
        cv.ov.scatter([cx], [ly], s=size_of([3.0e6 * frac], 3.0e6, 7.0)[0], c=PLANT_INK, edgecolors="white", lw=0.3)
        cv.text(cx, ly - 2.2, lab, ha="center", va="top", fontsize=fs.FS_ITEM - 0.4)
    cv.text(36.0 - 4.0, ly + 1.9, "Crushing class (cane/season)", fontsize=fs.FS_ITEM - 0.2, va="bottom")
    rows_b = [("dot", (6, c, "white", 0.3), lab) for (_, _, lab), c in zip(RADIUS_CLASSES, RADIUS_COLORS, strict=True)]
    x_b = 1.0 + w + gap
    yy = cv.legend(x_b, ly + 1.2, [("Required radius (cane at "
                                    f"{ctx.cfg.section('plants')['reach'].get('cane_yield_t_ha', 75)} t/ha)",
                                    rows_b[:3])], header=None)
    cv.legend(x_b + 24.0, ly + 1.2 - fs.ROW, [("", rows_b[3:])], header=None)
    del yy
    med = float(np.nanmedian(in_sp["required_radius_km"].replace(np.inf, np.nan)))
    cv.text(x_b + 46.0, ly - 0.2, f"SP: {len(in_sp)} mills\nmedian {med:.0f} km", fontsize=fs.FS_ITEM, va="center",
            linespacing=1.25)
    cv.scale(1.0, 1.4, S, seg_km=100, n=1, bh=0.6, compact=True)
    cv.text(22.0, 0.8, f"1:{S * 1000:,.0f}. Great-circle distance from each mill to every 30 m cane pixel "
            "(all states). Data: ANP; MapBiomas Col. 11. NIPE/Unicamp",
            fontsize=fs.FS_CRED, color=fs.MUTED, va="bottom")
    cv.panel("legend", 0.5, 0.0, FW - 0.5, LEG - 0.5)
    ctx.ck.add("x02_sp", True, f"{len(in_sp)} SP mills; median required radius {med:.1f} km", med)
    return cv


# ---------------------------------------------------------------------------
# x03 — mill catchments (after Cervi et al. 2019)
# ---------------------------------------------------------------------------
COMP_COLORS = ["#d9d9d9", "#c7e9c0", "#74c476", "#31a354", "#006d2c", "#00441b", "#002a10"]


@register("x03", "x03_mill_catchments")
def x03_mill_catchments(ctx: Ctx):
    mills, comp = ctx.t("mills"), ctx.t("competition")
    if mills is None:
        return None
    yield_t = ctx.cfg.section("plants")["reach"].get("cane_yield_t_ha", 75)
    FW, FH = fs.FULL_W, 108.0
    cv = Canvas(FW, FH)
    W, H = 52.0, 32.0
    X = (12.0, 12.0 + W + 15.0)
    Y = (FH - 5.5 - H, FH - 5.5 - H - 23.0 - H)
    sp = mills["uf"] == "SP"
    green, light = fs.CROP_COLORS["sugarcane"], "#a8dcc6"

    # (a) required radius
    a = cv.ax(X[0], Y[0], W, H)
    rr = mills["required_radius_km"].replace(np.inf, 99).clip(upper=62.4)
    bins = np.arange(0, 65, 2.5)
    a.hist([rr[sp].dropna(), rr[~sp].dropna()], bins=bins, stacked=True, color=[green, light], edgecolor="white",
           linewidth=0.3, label=["São Paulo", "Other states"])
    for j, (v, c, ls, who) in enumerate((
            (np.nanmedian(mills.loc[sp, "required_radius_km"]), green, "-", "SP"),
            (np.nanmedian(mills.loc[~sp, "required_radius_km"].replace(np.inf, np.nan)), "#5f9e86", "--", "others"))):
        a.axvline(v, color=c, lw=0.7, ls=ls)
        a.text(v + 0.6, a.get_ylim()[1] * (0.97 - 0.1 * j), f"median {who} {v:.0f} km", fontsize=fs.FS_TEXT - 0.6,
               va="top", color=c)
    a.set_xlim(0, 65)
    a.set_xticks([0, 10, 20, 30, 40, 50, 60], ["0", "10", "20", "30", "40", "50", "≥ 60"])
    a.set_xlabel("Radius needed for the crushing class (km)")
    a.set_ylabel("Mills")
    a.legend(loc="center right", fontsize=fs.FS_ITEM, handlelength=1.2, borderaxespad=0.2)
    axis_style(a)
    cv.text(X[0] - 9.5, Y[0] + H + 1.5, f"(a) Radius each mill needs (cane at {yield_t} t/ha)",
            fontsize=fs.FS_LABEL, fontweight="bold", va="bottom")
    cv.panel("a_required_radius", X[0] - 10.5, Y[0] - 8.5, X[0] + W, Y[0] + H + 5.0)

    # (b) cane within r, by radius
    b = cv.ax(X[1], Y[0], W, H)
    radii = [10, 20, 30, 50, 75, 100]
    vals = [mills[f"cane_ha_{r}km"].to_numpy() / 1e3 for r in radii]
    bp = b.boxplot(vals, positions=range(len(radii)), widths=0.55, patch_artist=True, showfliers=False,
                   medianprops={"color": fs.INK, "lw": 0.8}, whiskerprops={"lw": 0.5, "color": fs.MUTED},
                   capprops={"lw": 0.5, "color": fs.MUTED}, boxprops={"lw": 0.5, "edgecolor": fs.MUTED})
    for p_ in bp["boxes"]:
        p_.set_facecolor(light)
    need = np.nanmedian(mills["need_ha"]) / 1e3
    b.axhline(need, color=fs.DETAIL, lw=0.7, ls=(0, (3, 1.2)))
    b.text(len(radii) - 0.6, need * 1.12, f"median need {need:.0f} kha", color=fs.DETAIL, fontsize=fs.FS_TEXT - 0.4,
           ha="right", va="bottom")
    b.set_yscale("log")
    b.set_xticks(range(len(radii)), [str(r) for r in radii])
    b.set_xlabel("Radius around the mill (km)")
    b.set_ylabel("Sugarcane within the radius (kha)")
    axis_style(b)
    cv.text(X[1] - 9.5, Y[0] + H + 1.5, "(b) Cane area around each mill", fontsize=fs.FS_LABEL, fontweight="bold",
            va="bottom")
    cv.panel("b_cane_by_radius", X[1] - 10.5, Y[0] - 8.5, X[1] + W + 1.0, Y[0] + H + 5.0)

    # (c) supply ratio at 30 km by crushing class
    c = cv.ax(X[0], Y[1], W, H)
    classes = sorted(mills["capacity"].dropna().unique())
    rng = np.random.default_rng(7)
    for i, cap in enumerate(classes):
        s = mills[mills["capacity"] == cap]
        v = s["supply_ratio_30km"].clip(lower=0.01, upper=200).to_numpy()
        xs = i + rng.uniform(-0.28, 0.28, len(v))
        c.scatter(xs, v, s=3.5, c=np.where(s["uf"] == "SP", green, "#7a7a7a"), edgecolors="white", linewidths=0.2,
                  zorder=3)
        c.plot([i - 0.32, i + 0.32], [np.median(v)] * 2, color=fs.INK, lw=0.8, zorder=4)
    c.axhline(1, color=fs.DETAIL, lw=0.7, ls=(0, (3, 1.2)))
    c.set_yscale("log")
    c.set_xticks(range(len(classes)), [f"{v / 1e6:g}" for v in classes])
    c.set_xlabel("Crushing class midpoint (Mt cane/season)")
    c.set_ylabel("Cane within 30 km / cane needed")
    short = (mills["supply_ratio_30km"] < 1).mean()
    c.text(len(classes) - 0.5, 0.012, f"{_pct(short)} of mills below 1", ha="right", va="bottom",
           fontsize=fs.FS_TEXT - 0.4, color=fs.DETAIL)
    axis_style(c)
    cv.text(X[0] - 9.5, Y[1] + H + 7.0, "(c) Supply ratio at 30 km (green: SP)", fontsize=fs.FS_LABEL,
            fontweight="bold", va="bottom")
    cv.panel("c_supply_ratio", X[0] - 10.5, Y[1] - 8.5, X[0] + W, Y[1] + H + 10.5)

    # (d) competition
    d = cv.ax(X[1], Y[1], W, H)
    if comp is not None and len(comp):
        cp = comp[comp["pair"] == "sugarcane|sugarcane_mill"]
        tab = cp.pivot_table(index="radius_km", columns="n_plants", values="area_ha", aggfunc="sum").fillna(0)
        tab = tab.div(tab.sum(axis=1), axis=0)
        bottom = np.zeros(len(tab))
        labels = {0: "none", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6+"}
        for j, col in enumerate(tab.columns):
            d.bar(range(len(tab)), tab[col].to_numpy(), bottom=bottom, color=COMP_COLORS[min(j, 6)], width=0.7,
                  edgecolor="white", linewidth=0.3, label=labels.get(int(col), str(col)))
            bottom += tab[col].to_numpy()
        d.set_xticks(range(len(tab)), [f"{r:g}" for r in tab.index])
        d.set_ylim(0, 1)
        d.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
        d.set_xlabel("Radius (km)")
        d.set_ylabel("Share of Brazil's cane area")
        h_, l_ = d.get_legend_handles_labels()
        d.legend(h_, l_, title="Mills within the radius", loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=7,
                 fontsize=fs.FS_ITEM - 0.2, title_fontsize=fs.FS_ITEM, handlelength=0.9, handleheight=0.9,
                 columnspacing=0.8, handletextpad=0.3, borderaxespad=0.2)
    cv.text(X[1] - 9.5, Y[1] + H + 7.0, "(d) How many mills compete for the cane", fontsize=fs.FS_LABEL,
            fontweight="bold", va="bottom")
    cv.panel("d_competition", X[1] - 10.5, Y[1] - 8.5, FW - 0.5, Y[1] + H + 10.5)
    cv.text(1.0, 0.8, f"Sugarcane mills: ANP ({len(mills)}; crushing class midpoints). Cane: MapBiomas Col. 11 "
            f"{ctx.year}, 30 m pixels; great-circle distances. After Cervi et al. (2019).",
            fontsize=fs.FS_CRED, color=fs.MUTED, va="bottom")
    ctx.ck.add("x03_supply_ratio_below_1", True, f"{short:.1%} of mills have < their need within 30 km", short)
    return cv


# ---------------------------------------------------------------------------
# x04 — distance from every crop cell to the nearest plant
# ---------------------------------------------------------------------------
X04_PAIRS = ["sugarcane|sugarcane_mill", "sugarcane|biomass_power|bagasse", "corn_2nd|corn_ethanol",
             "soybean|biodiesel", "soy_all|biogas", "all_crops|biogas", "pasture|slaughterhouse",
             "all_crops|biomethane"]


@register("x04", "x04_distance_to_plants")
def x04_distance(ctx: Ctx):
    dist = ctx.t("distance")
    if dist is None:
        return None
    FW, FH = fs.FULL_W, 62.0
    cv = Canvas(FW, FH)
    a = cv.ax(11.0, 9.5, 50.0, FH - 16.0)
    ends = []
    shares = {}
    radii = [10, 20, 30, 50, 100]
    for pair in X04_PAIRS:
        t = dist[dist["pair"] == pair].sort_values("bin_km")
        if t.empty:
            continue
        cdf = t["area_ha"].cumsum() / t["area_ha"].sum()
        x = t["bin_km"].to_numpy() + 1
        lw = 1.1 if pair in ("sugarcane|sugarcane_mill", "soybean|biodiesel", "corn_2nd|corn_ethanol") else 0.75
        a.plot(x, cdf, color=PAIR_COLOR[pair], lw=lw, ls=PAIR_LS.get(pair, "-"))
        k = np.searchsorted(x, 300)
        ends.append((PAIR_LABEL[pair].split(" → ")[0] + " → " + PAIR_LABEL[pair].split(" → ")[1].replace(" plant", ""),
                     300, float(cdf.iloc[min(k, len(cdf) - 1)]), PAIR_COLOR[pair]))
        shares[pair] = [float(cdf.iloc[min(int(r) - 1, len(cdf) - 1)]) for r in radii]
    for r in (30, 50):
        a.axvline(r, color=fs.HAIR, lw=0.4, zorder=0)
    a.set_xlim(0, 300)
    a.set_ylim(0, 1.0)
    a.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
    a.set_xlabel("Distance to the nearest plant (km)")
    a.set_ylabel("Share of the crop area (cumulative)")
    axis_style(a)
    a.grid(axis="x", color=fs.FAINT, lw=0.4)
    cv.text(1.0, FH - 1.0, "(a) Crop area by distance to the nearest plant", fontsize=fs.FS_LABEL,
            fontweight="bold", va="top")
    # (b) heat table
    from matplotlib import colormaps

    cmap = colormaps["Greens"]
    tx, ty = 98.0, FH - 9.0
    rh, cw = 4.6, 6.6
    for j, r in enumerate(radii):
        cv.text(tx + j * cw + cw / 2, ty + 1.2, f"{r} km", ha="center", va="bottom", fontsize=fs.FS_ITEM)
    for i, pair in enumerate([p for p in X04_PAIRS if p in shares]):
        yy = ty - (i + 1) * rh
        cv.text(tx - 1.0, yy + rh / 2, PAIR_SHORT[pair], ha="right", va="center", fontsize=fs.FS_ITEM - 0.2)
        cv.ov.plot([tx - 31.5, tx - 27.5], [yy + rh / 2] * 2, color=PAIR_COLOR[pair], ls=PAIR_LS.get(pair, "-"),
                   lw=1.0, solid_capstyle="butt")
        for j, v in enumerate(shares[pair]):
            from matplotlib.patches import Rectangle

            cv.ov.add_patch(Rectangle((tx + j * cw, yy), cw - 0.4, rh - 0.4, fc=cmap(0.1 + 0.8 * v), ec="none"))
            cv.text(tx + j * cw + cw / 2 - 0.2, yy + rh / 2 - 0.2, f"{100 * v:.0f}", ha="center", va="center",
                    fontsize=fs.FS_ITEM, color="white" if v > 0.55 else fs.INK)
    cv.text(tx - 31.5, FH - 1.0, "(b) Share within the radius (%)", fontsize=fs.FS_LABEL, fontweight="bold",
            va="top")
    for e in ends:
        a.plot([e[1]], [e[2]], marker="o", ms=2.0, color=e[3], mec="white", mew=0.2, clip_on=False)
    cv.text(1.0, 0.8, f"Crop pixels: MapBiomas Col. 11 {ctx.year}, 30 m (2nd-season corn: second-crop layer). Plants operating "
            "(ANP, CIBiogás, ANEEL, MAPA). Great-circle distance: road distances are longer.", fontsize=fs.FS_CRED,
            color=fs.MUTED, va="bottom")
    cv.panel("a_cdf", 0.5, 7.5, 64.0, FH - 0.3)
    cv.panel("b_table", 65.5, 7.5, FW - 0.5, FH - 0.3)
    for pair, v in shares.items():
        ctx.ck.add(f"x04_{pair}_30km", True, f"{v[2]:.1%} within 30 km, {v[3]:.1%} within 50 km", v[2])
    return cv


# ---------------------------------------------------------------------------
# x05 — MAUP access test
# ---------------------------------------------------------------------------
X05_PAIRS = ["sugarcane|sugarcane_mill", "corn_2nd|corn_ethanol", "soybean|biodiesel", "all_crops|biogas",
             "pasture|slaughterhouse", "all_crops|biomethane"]


@register("x05", "x05_maup_access")
def x05_maup(ctx: Ctx):
    maup = ctx.t("maup")
    if maup is None:
        return None
    FW = fs.FULL_W
    ncol, w, h, gx, gy = 3, 32.0, 26.0, 11.0, 13.0
    FH = 2 * (h + gy) + 11.0
    cv = Canvas(FW, FH)
    x0 = 11.0
    for i, pair in enumerate(X05_PAIRS):
        t = maup[maup["pair"] == pair]
        if t.empty:
            continue
        col, row = i % ncol, i // ncol
        x, y = x0 + col * (w + gx), FH - 6.0 - (row + 1) * h - row * gy
        a = cv.ax(x, y, w, h)
        for lev, (c, ls, mk) in LEVEL_STYLE.items():
            s = t[t["level"] == lev].sort_values("radius_km")
            a.plot(s["radius_km"], s["share"], color=c, ls=ls, marker=mk, ms=2.2, lw=0.8, mec="white", mew=0.2,
                   label=lev.capitalize())
        s30 = t[t["radius_km"] == 30].set_index("level")["share"]
        lo, hi = (s30.min(), s30.max()) if len(s30) == 3 else (0, 0)
        if hi - lo >= 0.03:
            a.annotate("", (30, hi), (30, lo), arrowprops={"arrowstyle": "<->", "lw": 0.5, "color": fs.DETAIL,
                                                             "shrinkA": 0, "shrinkB": 0})
            a.text(32, (lo + hi) / 2, f"{100 * (hi - lo):.0f} pp", color=fs.DETAIL, fontsize=fs.FS_TEXT - 0.5,
                   va="center")
        elif len(s30):
            a.text(0.97, 0.94, "at 30 km all levels agree within 3 pp", transform=a.transAxes, ha="right", va="top",
                   fontsize=fs.FS_TEXT - 1.0, color=fs.MUTED)
        a.set_xlim(0, 105)
        a.set_ylim(0, 1)
        a.set_xticks([10, 30, 50, 75, 100])
        a.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}")
        axis_style(a)
        if col == 0:
            a.set_ylabel("Area within reach (%)")
        if row == 1:
            a.set_xlabel("Radius (km)")
        cv.text(x - 9.5, y + h + 1.3, f"({letter(i)}) {PAIR_LABEL[pair]}", fontsize=fs.FS_TITLE + 0.2,
                fontweight="bold", va="bottom")
        cv.panel(f"{letter(i)}_{pair.replace('|', '_')}", x - 10.5, y - 7.0, x + w + 1.0, y + h + 5.0)
    rows = [("line", (c, 0.8, ls), f"from the {lev}" + (" (30 m, true)" if lev == "pixel" else "'s point"))
            for lev, (c, ls, _) in LEVEL_STYLE.items()]
    cv.legend(x0 - 9.5, 9.0, [("Distance measured", rows)], header=None)
    cv.text(58.0, 5.0, "Same land, same plants. A municipality (or region) counts whole when its\nrepresentative "
            "point is within the radius. Red: the gap at 30 km between reading\nreach from statistical units and "
            "from the 30 m pixels.", fontsize=fs.FS_CRED, color=fs.MUTED, va="center", linespacing=1.3)
    cv.panel("legend", 0.5, 0.0, FW - 0.5, 10.5)
    return cv


# ---------------------------------------------------------------------------
# x06 — intermediate regions: share within reach
# ---------------------------------------------------------------------------
X06_MAPS = [("sugarcane|sugarcane_mill", 30, "Sugarcane within 30 km of a mill"),
            ("soybean|biodiesel", 50, "Soybean within 50 km of a biodiesel plant"),
            ("corn_2nd|corn_ethanol", 50, "2nd-season corn within 50 km of a corn ethanol plant"),
            ("all_crops|biogas", 30, "All crops within 30 km of a biogas plant")]
SHARE_EDGES = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0001]
SHARE_LABELS = ["< 10 %", "10–25 %", "25–50 %", "50–75 %", "75–90 %", "≥ 90 %"]


@register("x06", "x06_regions_within_reach")
def x06_regions(ctx: Ctx):
    from matplotlib import colormaps

    from bbp.figures import nice_scale

    rgt = ctx.t("rgint")
    if rgt is None:
        return None
    rg = ctx.regions()
    ext = ctx.ext
    FW, gap = fs.FULL_W, 3.0
    w = (FW - 2.0 - gap) / 2
    S = nice_scale((ext[1] - ext[0]) / w)
    MW, MH = (ext[1] - ext[0]) / S, (ext[3] - ext[2]) / S
    TT, LEG = 4.4, 14.5
    FH = 2 * (MH + TT) + LEG
    cv = Canvas(FW, FH)
    cmap = colormaps["YlGnBu"]
    colors = [cmap(v) for v in np.linspace(0.12, 0.95, len(SHARE_LABELS))]
    min_ha = 5_000.0
    for i, (pair, r, title) in enumerate(X06_MAPS):
        col, row = i % 2, i // 2
        x = 1.0 + col * (w + gap)
        y = FH - (row + 1) * (MH + TT) + 0.2
        a = cv.map_ax(x, y, ext, S)
        t = rgt[rgt["pair"] == pair].set_index("cd_rgint")
        share = t[f"within_{r}km_ha"] / t["area_ha"]
        m = rg.assign(share=rg["cd_rgint"].map(share), area=rg["cd_rgint"].map(t["area_ha"]).fillna(0))
        little = m["area"] < min_ha
        m[little].plot(ax=a, color="#f3f3f3", linewidth=0, zorder=1)
        cls = np.digitize(m.loc[~little, "share"].fillna(0), SHARE_EDGES[1:-1])
        m.loc[~little].assign(c=[colors[k] for k in cls]).plot(ax=a, color=[colors[k] for k in cls], linewidth=0,
                                                                zorder=1.2)
        rg.boundary.plot(ax=a, color="white", linewidth=0.18, zorder=2)
        ctx.states.boundary.plot(ax=a, color="#4d4d4d", linewidth=0.28, zorder=3)
        ctx.country.boundary.plot(ax=a, color=fs.COUNTRY_EDGE, linewidth=0.45, zorder=3.2)
        fam = pair.split("|")[1]
        sub = pair.split("|")[2] if pair.count("|") > 1 else None
        if ctx.plants is not None:
            P = ctx.plants[(ctx.plants["family"] == fam) & ctx.plants["operating"].fillna(True).astype(bool)]
            if sub:
                P = P[P["subtype"] == sub]
            a.scatter(P["x"], P["y"], s=0.9, c=PLANT_INK, linewidths=0, zorder=5)
        map_frame(a, ext)
        tot = t["area_ha"].sum()
        nat = t[f"within_{r}km_ha"].sum() / tot if tot else np.nan
        panel_title(cv, x, y + MH + 0.4, f"({letter(i)}) {title}")
        cv.text(x + MW - 0.5, y + 1.0, f"Brazil: {_pct(nat)}\nof {tot / 1e6:.1f} Mha", fontsize=fs.FS_ITEM,
                ha="right", va="bottom", linespacing=1.2)
        cv.panel(f"{letter(i)}_{pair.replace('|', '_')}_{r}km", x, y, x + MW, y + MH + TT - 0.2)
        ctx.ck.add(f"x06_{pair}_{r}km", True, f"Brazil {nat:.1%} of {tot / 1e6:.2f} Mha", nat)
    rows = [("patch", {"fc": c, "ec": "#7a7a7a"}, lab) for c, lab in zip(colors, SHARE_LABELS, strict=True)]
    cv.legend(1.0, LEG - 1.2, [("Share of the region's crop area within the radius", rows[:3])], header=None)
    cv.legend(30.0, LEG - 1.2 - fs.ROW, [("", rows[3:])], header=None)
    cv.legend(56.0, LEG - 1.2 - fs.ROW, [("", [("patch", {"fc": "#f3f3f3", "ec": "#7a7a7a"},
                                                f"< {min_ha / 1e3:.0f} kha of the crop"),
                                               ("dot", (1.5, PLANT_INK, PLANT_INK, 0), "Plant (operating)")])],
              header=None)
    cv.north(102.0, 6.5)
    cv.scale(106.0, 6.3, S, seg_km=1000, n=1, bh=0.6)
    cv.text(1.0, 0.5, "133 intermediate regions (IBGE 2017). Shares from 30 m pixel distances to the nearest "
            "operating plant (great circle). EPSG:5880. " + fs.CREDIT_CARTO,
            fontsize=fs.FS_CRED - 0.2, color=fs.MUTED, va="bottom")
    cv.panel("legend", 0.5, 0.0, FW - 0.5, LEG - 0.2)
    return cv


# ---------------------------------------------------------------------------
# x07 — intermediate-region typology
# ---------------------------------------------------------------------------
REGION_COLORS = {"N": "#1b9e77", "NE": "#d95f02", "CO": "#7570b3", "SE": "#e7298a", "S": "#66a61e"}
REGION_NAME = {"N": "North", "NE": "Northeast", "CO": "Centre-West", "SE": "Southeast", "S": "South"}
UF_REGION = {**dict.fromkeys(["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "N"),
             **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "NE"),
             **dict.fromkeys(["DF", "GO", "MT", "MS"], "CO"), **dict.fromkeys(["ES", "MG", "RJ", "SP"], "SE"),
             **dict.fromkeys(["PR", "RS", "SC"], "S")}


@register("x07", "x07_region_typology")
def x07_typology(ctx: Ctx):
    rgt, rgp = ctx.t("rgint"), ctx.t("rgint_plants")
    if rgt is None:
        return None
    FW, FH = fs.FULL_W, 77.0
    cv = Canvas(FW, FH)
    panels = [("all_crops|biogas", 30, "(a) All crops vs biogas plants, 30 km"),
              ("soybean|biodiesel", 50, "(b) Soybean vs biodiesel plants, 50 km")]
    W, H = 50.0, 52.0
    for i, (pair, r, title) in enumerate(panels):
        x, y = 12.0 + i * (W + 16.0), 17.0
        a = cv.ax(x, y, W, H)
        t = rgt[rgt["pair"] == pair].copy()
        t = t[t["area_ha"] > 10_000]
        t["share"] = t[f"within_{r}km_ha"] / t["area_ha"]
        t["reg"] = t["uf"].map(UF_REGION)
        fam = pair.split("|")[1]
        n = rgp[rgp["family"] == fam].set_index("cd_rgint")["n"] if rgp is not None else pd.Series(dtype=float)
        t["n"] = t["cd_rgint"].map(n).fillna(0)
        for reg, c in REGION_COLORS.items():
            s = t[t["reg"] == reg]
            a.scatter(s["area_ha"] / 1e3, s["share"], s=2.5 + 3.0 * np.sqrt(s["n"]), c=c, alpha=0.85,
                      edgecolors="white", linewidths=0.3, label=REGION_NAME[reg], zorder=3)
        a.set_xscale("log")
        a.set_ylim(-0.03, 1.03)
        a.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
        xm = np.median(t["area_ha"] / 1e3)
        a.axvline(xm, color=fs.HAIR, lw=0.5, zorder=1)
        a.axhline(0.5, color=fs.HAIR, lw=0.5, zorder=1)
        big_far = t[(t["area_ha"] / 1e3 > xm) & (t["share"] < 0.5)].nlargest(5, "area_ha")
        for j, (_, rw) in enumerate(big_far.sort_values("share").iterrows()):
            a.annotate(str(rw["nm_rgint"]), (rw["area_ha"] / 1e3, rw["share"]),
                       xytext=(xm * 0.9, 0.12 + 0.075 * j), textcoords="data", ha="right", va="center",
                       fontsize=fs.FS_TEXT - 1.2, color=fs.INK,
                       arrowprops={"arrowstyle": "-", "lw": 0.35, "color": fs.MUTED, "shrinkA": 1, "shrinkB": 1.5})
        a.text(0.99, 0.49, "large, beyond reach", transform=a.transAxes, ha="right", va="top",
               fontsize=fs.FS_TEXT - 0.6, color=fs.DETAIL, style="italic")
        a.text(0.99, 0.99, "large, within reach", transform=a.transAxes, ha="right", va="top",
               fontsize=fs.FS_TEXT - 0.6, color=fs.MUTED, style="italic")
        a.set_xlim(8, None)
        a.set_xlabel("Crop area of the intermediate region (kha, log)")
        if i == 0:
            a.set_ylabel("Share within the radius")
        axis_style(a)
        cv.text(x - 10.0, y + H + 1.5, title, fontsize=fs.FS_LABEL, fontweight="bold", va="bottom")
        cv.panel(f"{letter(i)}_{pair.replace('|', '_')}", x - 11.0, y - 8.0, x + W + 1.0, y + H + 6.0)
        ctx.ck.add(f"x07_{pair}_large_far", True, f"{len(t[(t['area_ha'] / 1e3 > xm) & (t['share'] < 0.5)])} "
                   "regions above the median area with < 50 % in reach")
    rows = [("dot", (6, c, "white", 0.3), REGION_NAME[k]) for k, c in REGION_COLORS.items()]
    cv.legend(2.0, 7.4, [("", rows[:3])], header=None)
    cv.legend(26.0, 7.4, [("", rows[3:])], header=None)
    cv.text(54.0, 4.0, "Marker area proportional to number of plants of the family in the region.\nOne point per intermediate "
            "region (IBGE, 133) with > 10 kha of the crop.", fontsize=fs.FS_CRED, color=fs.MUTED, va="center",
            linespacing=1.3)
    cv.panel("legend", 0.5, 0.0, FW - 0.5, 7.5)
    return cv


# ---------------------------------------------------------------------------
# x08 — registers compared; biomethane plants against the mandate
# ---------------------------------------------------------------------------
@register("x08", "x08_registers_biomethane")
def x08_registers(ctx: Ctx):

    from bbp.plants import read_conab_mills, read_epe_mills
    from bbp.reach import haversine_km

    if ctx.plants is None:
        return None
    cfg = ctx.cfg
    cx = {k: cfg.raw_root / (v["path"] if isinstance(v, dict) else v)
          for k, v in (cfg.section("plants").get("registers") or {}).items()}
    anp = ctx.plants[ctx.plants["family"] == "sugarcane_mill"].to_crs("EPSG:4674")
    anp_op = anp[anp["operating"].fillna(True).astype(bool)]
    regs = {"ANP 2026 (operating)": anp_op}
    if cx.get("conab") and cx["conab"].exists():
        regs["CONAB 2025 (active)"] = read_conab_mills(cx["conab"])
    if cx.get("epe") and cx["epe"].exists():
        regs["MapBiomas / EPE 2018"] = read_epe_mills(cx["epe"])
    FW, FH = fs.FULL_W, 70.0
    cv = Canvas(FW, FH)
    # (a) mills per state, three registers
    a = cv.ax(10.0, 13.0, 36.0, FH - 20.0)
    ufcol = {k: ("UF" if "UF" in g else "uf") for k, g in regs.items()}
    counts = pd.DataFrame({k: g[ufcol[k]].value_counts() for k, g in regs.items()}).fillna(0)
    top = counts.sum(axis=1).nlargest(10).index[::-1]
    shades = ["#1a1a1a", "#8c8c8c", "#cfcfcf"]
    bh = 0.8 / len(regs)
    for j, k in enumerate(regs):
        a.barh(np.arange(len(top)) + (j - (len(regs) - 1) / 2) * bh, counts.loc[top, k], height=bh,
               color=shades[j], label=k, edgecolor="white", linewidth=0.2)
    a.set_yticks(range(len(top)), top)
    a.set_xlabel("Cane mills")
    a.legend(loc="lower right", fontsize=fs.FS_ITEM - 0.3, handlelength=1.0, borderaxespad=0.2)
    a.grid(axis="x", color=fs.FAINT, lw=0.4)
    cv.text(1.0, FH - 1.0, "(a) Cane mills per state", fontsize=fs.FS_LABEL, fontweight="bold", va="top")
    cv.panel("a_mills_per_state", 0.5, 2.0, 48.0, FH - 0.3)
    # (b) do the registers locate the same mills? distance from each ANP mill to the nearest unit of the others
    b = cv.ax(56.0, 13.0, 26.0, FH - 20.0)
    lab_b = []
    for j, (k, g) in enumerate(list(regs.items())[1:]):
        lon, lat = g.geometry.x.to_numpy(), g.geometry.y.to_numpy()
        d = np.array([haversine_km(x_, y_, lon, lat).min() for x_, y_ in zip(anp_op["lon"], anp_op["lat"],
                                                                             strict=True)])
        xs = np.sort(d)
        b.plot(np.maximum(xs, 0.01), np.arange(1, len(xs) + 1) / len(xs), color=shades[j + 1] if j else "#555555",
               lw=0.9, ls="-" if j == 0 else (0, (3, 1.2)))
        lab_b.append((k, float((d <= 2).mean()), float((d <= 10).mean())))
    b.set_xscale("log")
    b.set_xlim(0.05, 200)
    b.set_ylim(0, 1)
    b.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
    b.axvline(2, color=fs.HAIR, lw=0.5)
    b.set_xlabel("Nearest unit in the other register (km)")
    b.set_ylabel("ANP mills (cumulative)")
    axis_style(b)
    for j, (k, s2, _s10) in enumerate(lab_b):
        b.text(0.97, 0.50 - j * 0.07, f"{k.split(' (')[0].replace('MapBiomas / ', '')}: {_pct(s2)} ≤ 2 km",
               transform=b.transAxes, fontsize=fs.FS_TEXT - 1.0, va="top", ha="right",
               color="#555555" if j == 0 else "#8c8c8c")
    b.text(0.97, 0.34, "EPE points reused\nby ANP: dashed\ncurve starts high", transform=b.transAxes,
           fontsize=fs.FS_TEXT - 1.4, va="top", ha="right", color=fs.MUTED, linespacing=1.2)
    cv.text(48.5, FH - 1.0, "(b) Same mills, same place?", fontsize=fs.FS_LABEL, fontweight="bold", va="top")
    cv.panel("b_register_match", 47.0, 2.0, 84.0, FH - 0.3)
    # (c) biomethane plants: capacity, estimated output, the mandate
    c = cv.ax(98.0, 13.0, 32.0, FH - 20.0)
    bm = ctx.plants[ctx.plants["family"] == "biomethane"]
    bm = bm[bm["operating"].astype(bool)].sort_values("capacity")
    snaps = ctx.proc / "plants_biomethane_snapshots.parquet"
    if len(bm):
        mill_sites = ctx.plants.loc[ctx.plants["family"] == "sugarcane_mill", "site_id"]
        cane = bm["site_id"].isin(mill_sites) | bm["name"].str.contains(
            "COCAL|RAÍZEN|RAIZEN|ADECOAGRO|SANTA CRUZ|BIOENERGIA", case=False)
        yy = np.arange(len(bm))
        c.barh(yy, bm["capacity"] / 1e3, color=np.where(cane, fs.CROP_COLORS["sugarcane"], "#5a5a5a"), height=0.75,
               edgecolor="none", label="Authorised capacity")
        c.set_yticks(yy, [f"{u}" for u in bm["uf"]], fontsize=fs.FS_TEXT - 1.4)
        c.set_xlabel("Biomethane (1,000 m³/day)")
        c.grid(axis="x", color=fs.FAINT, lw=0.4)
        lines = [f"{len(bm)} plants", f"authorised {bm['capacity'].sum() / 1e6:.2f} M m³/d"]
        if snaps.exists():
            s = pd.read_parquet(snaps).iloc[-1]
            lines += [f"produced {s['produced_m3d'] / 1e6:.2f} M m³/d ({s['snapshot']})"]
        c.text(0.98, 0.03, "\n".join(lines), transform=c.transAxes, ha="right", va="bottom",
               fontsize=fs.FS_TEXT - 1.0, linespacing=1.25, bbox={"fc": "white", "ec": "none", "alpha": 0.9,
                                                                    "pad": 0.6})
        ctx.ck.add("x08_biomethane", True, "; ".join(lines), float(bm["capacity"].sum()))
    cv.text(86.0, FH - 1.0, "(c) Biomethane plants (green: cane mills)", fontsize=fs.FS_LABEL, fontweight="bold",
            va="top")
    cv.panel("c_biomethane", 85.0, 2.0, FW - 0.5, FH - 0.3)
    cv.text(1.0, 0.6, "Data: ANP ethanol plants; ANP biomethane open data (capacity and production, snapshot set in "
            "pipeline.yaml); CONAB 2025;\nMapBiomas infrastructure (EPE webmap 2018). The mandate ruler is drawn "
            "separately (Phase 3).", fontsize=fs.FS_CRED,
            color=fs.MUTED, va="bottom", linespacing=1.3)
    for k, s2, s10 in lab_b:
        ctx.ck.add(f"x08_match_{k.split(' ')[0]}", True, f"{s2:.1%} of ANP mills within 2 km, {s10:.1%} within 10 km",
                   s2)
    return cv


# ---------------------------------------------------------------------------
def build(cfg: Config, ck: CheckLog, only: list[str] | None = None) -> list:
    """`bbp explore`: every exploratory figure, each element also on its own."""
    import traceback

    from bbp.figures import out_dir

    fs.setup()
    ctx = Ctx(cfg, ck)
    folder = out_dir(cfg) / "explore"
    spec = cfg.section("paper")
    formats = spec.get("figure_formats", ["pdf", "svg", "png", "tif"])
    formats = [f for f in formats if f != "tif"]
    dpi = int(spec.get("dpi", 600))
    written = []
    for key, (name, func) in EXPLORE.items():
        if only and key not in only:
            continue
        log.info("Explore %s: %s", key, name)
        try:
            cv = func(ctx)
        except Exception as exc:  # one broken figure must not stop the others
            log.error("%s", traceback.format_exc())
            ck.add(f"{key}_built", False, f"{type(exc).__name__}: {exc}")
            continue
        if cv is None:
            ck.add(f"{key}_built", False, "inputs missing (run bbp plants / bbp reach)", severity="warn")
            continue
        ck.add(f"{key}_size", cv.fw <= 132.5 and cv.fh <= fs.MAX_H, f"{cv.fw:.1f} × {cv.fh:.1f} mm", severity="warn")
        written += cv.save(folder, name, formats, dpi, panels_dir=folder / "panels" / name)
    return written
