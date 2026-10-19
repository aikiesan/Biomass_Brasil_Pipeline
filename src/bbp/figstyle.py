"""House style of the paper figures: the NIPE/Unicamp cartographic style (e-methane paper, Fig 2),
at Geocarto International's sizes.

* The whole figure is laid out in MILLIMETRES on one canvas (`Canvas`): every axis is placed in mm,
  and an overlay axis in mm (0..FW, 0..FH) carries legends, scale bars, connectors and credits.
* Maps are drawn at a FIXED map scale S (metres per mm) in SIRGAS 2000 / Brazil Polyconic
  (EPSG:5880, display only), so panels of a figure share one true scale bar.
* Under each map a strip: two-tone north arrow, alternating 0–500–1000 km scale bar, credits.
* Legend as a structured block: "LEGEND" header, group titles, 2.45 mm row pitch, 4 mm handles.
* Arial-metric sans (Liberation Sans in the Docker image, so output is identical everywhere).
  Sizes at final scale: legend items 5.4 pt, group titles 5.8 pt, "LEGEND" 6.8 pt bold, credits
  4.7 pt, state labels 4.5 pt; chart text 6–7 pt.
* One colour per crop in every figure (Okabe–Ito as in the NIPE figure). The palette validator
  flags the yellow (citrus) as too light for a bare line and citrus/sugarcane as close under
  deuteranopia, so crop lines carry dark-edged markers, light colours get a dark casing, perennials
  are dashed and lines are labelled directly.
* Export: PDF + SVG + PNG 600 dpi + TIFF 600 dpi LZW, white background; final size logged in mm.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("bbp")

MM = 1 / 25.4
FULL_W = 132.0                  # Geocarto text block (mm)
HALF_W = 90.0
MAX_H = 190.0

FONTS = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
FS_ITEM, FS_TITLE, FS_HEAD = 5.4, 5.8, 6.8      # legend items / group titles / "LEGEND"
FS_CRED, FS_STATE, FS_SCALE = 4.7, 4.5, 4.7
FS_TEXT, FS_LABEL = 6.0, 6.5                     # chart ticks / chart axis labels and panel titles
ROW, GROUP_GAP, HW, TXT = 2.45, 2.3, 4.0, 1.3   # legend row pitch, gap between groups, handle width, gap (mm)
STRIP = 12.0                                     # band under a map: north arrow, scale bar, credits (mm)

INK = "#1a1a1a"
MUTED = "#555555"
STATE_EDGE, COUNTRY_EDGE, STATE_LABEL = "#8a8a8a", "#4d4d4d", "#4a4a4a"
HAIR = "#bfbfbf"
FAINT = "#ececec"
DETAIL = "#c1121f"
DETAIL_LS = (0, (2, 1.2))
CREDIT_PROJ = "Projection: SIRGAS 2000 / Brazil Polyconic (EPSG:5880)"
CREDIT_CARTO = "Cartography: NIPE/Unicamp"

CROP_COLORS = {"soybean": "#0173b2", "corn_2nd": "#e69f00", "sugarcane": "#029e73", "coffee": "#d55e00",
               "citrus": "#ece133", "cotton": "#56b4e9", "rice": "#cc78a7", "corn_1st": "#9a9a9a"}
LIGHT = {"citrus", "cotton", "corn_2nd"}         # need a dark casing as lines on white
CROP_MARKERS = {"soybean": "o", "corn_2nd": "s", "sugarcane": "^", "coffee": "D", "citrus": "v", "cotton": "P",
                "rice": "X", "corn_1st": "."}
CROP_SHORT = {"soybean": "Soybean", "corn_2nd": "Corn (2nd season)", "sugarcane": "Sugarcane", "coffee": "Coffee",
              "citrus": "Citrus", "cotton": "Cotton", "rice": "Rice", "corn_1st": "Corn (1st season)"}
PERENNIAL = {"sugarcane", "coffee", "citrus"}
DECLARED_COLOR = MUTED

# Land-cover display groups (bbp.landcover): the paper crops in their crop colours, saturated; everything
# else in light neutrals, so the crops carry the map. Soybean followed by a 2nd-season crop takes the colour
# of that crop (corn orange, cotton blue): the field holds two residues a year.
LANDCOVER_COLORS = {
    "soybean": CROP_COLORS["soybean"], "soy_corn2": CROP_COLORS["corn_2nd"], "soy_cotton2": CROP_COLORS["cotton"],
    "sugarcane": CROP_COLORS["sugarcane"], "coffee": CROP_COLORS["coffee"], "citrus": CROP_COLORS["citrus"],
    "cotton": CROP_COLORS["cotton"], "rice": CROP_COLORS["rice"],
    "other_crops": "#c2b394", "pasture": "#efe7d2", "plantation": "#d4cdb8", "natural": "#e1e7db",
    "water": "#c8dae6", "built": "#bdbdbd",
}
LANDCOVER_LEGEND = [            # (display groups drawn with one swatch, label); soybean + cotton shares cotton's
    (("soybean",), "Soybean"),
    (("soy_corn2",), "Soybean + corn (2nd season)"),
    (("sugarcane",), "Sugarcane"),
    (("coffee",), "Coffee"),
    (("citrus",), "Citrus"),
    (("cotton", "soy_cotton2"), "Cotton (incl. 2nd season)"),
    (("rice",), "Rice"),
    (("other_crops",), "Other crops"),
    (("pasture",), "Pasture and mosaic of uses"),
    (("plantation",), "Forest plantation"),
    (("natural",), "Natural vegetation"),
    (("water",), "Water"),
    (("built",), "Urban and other built-up"),
]


def setup() -> str:
    """Apply the rcParams; returns the font family actually used."""
    import matplotlib as mpl
    from matplotlib import font_manager

    mpl.use("Agg")
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in FONTS if f in names), "DejaVu Sans")
    if family not in ("Arial", "Liberation Sans", "Helvetica"):
        log.warning("no Arial-metric font installed; using %r (rebuild the Docker image)", family)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONTS, "font.size": FS_TEXT,
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "axes.titlesize": FS_LABEL, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 3,
        "axes.labelsize": FS_LABEL, "axes.labelcolor": INK, "axes.edgecolor": MUTED, "axes.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "axes.axisbelow": True,
        "axes.facecolor": "none",
        "grid.color": FAINT, "grid.linewidth": 0.4,
        "xtick.labelsize": FS_TEXT, "ytick.labelsize": FS_TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.major.pad": 1.6, "ytick.major.pad": 1.6,
        "legend.frameon": False, "legend.fontsize": FS_ITEM,
        "lines.linewidth": 0.9, "lines.markersize": 3.0, "patch.linewidth": 0.4, "hatch.linewidth": 0.35,
        "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white", "figure.dpi": 150,
    })
    return family


# ---------------------------------------------------------------------------
# crops
# ---------------------------------------------------------------------------
def crop_style(crop: str) -> dict:
    return {"color": CROP_COLORS.get(crop, MUTED), "marker": CROP_MARKERS.get(crop, "o"),
            "linestyle": (0, (3.5, 1.4)) if crop in PERENNIAL else "-"}


def crop_line(ax, x, y, crop: str, lw: float = 0.9, ms: float = 3.0, marker: bool = True, **kw):
    """A crop's line: its colour, dashed if perennial, dark-edged marker, dark casing if the colour is light."""
    import matplotlib.patheffects as pe

    st = crop_style(crop)
    effects = [pe.Stroke(linewidth=lw + 0.7, foreground=MUTED), pe.Normal()] if crop in LIGHT else None
    (line,) = ax.plot(x, y, color=st["color"], ls=st["linestyle"], lw=lw, marker=st["marker"] if marker else None,
                      ms=ms, mec=INK, mew=0.3, path_effects=effects, **kw)
    return line


def label(crop: str) -> str:
    return CROP_SHORT.get(crop, crop)


def direct_labels(ax, ends: list[tuple[str, float, float, str]], x_pad: float = 0.0, min_gap: float = None,
                  fontsize: float = FS_TEXT):
    """Labels at line ends, spread symmetrically around their lines; leader lines where moved."""
    if not ends:
        return
    y0, y1 = ax.get_ylim()
    gap = min_gap if min_gap is not None else (y1 - y0) * 0.055
    order = sorted((e for e in ends if np.isfinite(e[2])), key=lambda e: e[2])
    pos = [e[2] for e in order]
    for _ in range(200):
        moved = False
        for i in range(1, len(pos)):
            d = pos[i] - pos[i - 1]
            if d < gap - 1e-12:
                pos[i - 1] -= (gap - d) / 2
                pos[i] += (gap - d) / 2
                moved = True
        lo = y0 + gap * 0.3
        if pos and pos[0] < lo:
            pos = [p + lo - pos[0] for p in pos]
        if not moved:
            break
    for (text, x, y, _color), yy in zip(order, pos, strict=True):
        ax.annotate(text, (x, y), xytext=(x + x_pad, yy), textcoords="data", va="center", ha="left",
                    fontsize=fontsize, color=INK, annotation_clip=False,
                    arrowprops=None if abs(yy - y) < gap * 0.25 else
                    {"arrowstyle": "-", "lw": 0.4, "color": MUTED, "shrinkA": 0, "shrinkB": 0})


# ---------------------------------------------------------------------------
# the millimetre canvas
# ---------------------------------------------------------------------------
class Canvas:
    """A figure of FW × FH mm; `ax()` places axes in mm, `ov` is an overlay axis in mm."""

    def __init__(self, fw: float, fh: float):
        import matplotlib.pyplot as plt

        self.fw, self.fh = float(fw), float(fh)
        self.fig = plt.figure(figsize=(fw * MM, fh * MM))
        self.panels: dict[str, tuple[float, float, float, float]] = {}
        self._hide: dict[str, list] = {}
        self._connectors: list = []
        self.ov = self.ax(0, 0, fw, fh, z=10)
        self.ov.set_xlim(0, fw)
        self.ov.set_ylim(0, fh)
        self.ov.axis("off")

    def ax(self, x: float, y: float, w: float, h: float, z: float = 1, transparent: bool = True):
        a = self.fig.add_axes((x / self.fw, y / self.fh, w / self.fw, h / self.fh), zorder=z)
        if transparent:
            a.patch.set_alpha(0)
        return a

    def map_ax(self, x: float, y: float, extent: tuple[float, float, float, float], scale: float, z: float = 1):
        """Map axes at (x, y) mm showing `extent` (x0, x1, y0, y1, metres) at `scale` metres per mm."""
        x0, x1, y0, y1 = extent
        a = self.ax(x, y, (x1 - x0) / scale, (y1 - y0) / scale, z=z)
        map_frame(a, extent)
        return a

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", FS_ITEM)
        kw.setdefault("color", INK)
        return self.ov.text(x, y, s, **kw)

    # ---- map furniture (all in mm on the overlay) ------------------------------
    def legend(self, x: float, y_top: float, groups: list, header: str | None = "LEGEND") -> float:
        """Structured legend. groups: [(title, [(kind, spec, label), ...]), ...];
        kind: line (color, lw[, ls]) | lineon (ground, color, lw) | dot (size, fc, ec, lw)
        | patch ({"fc", "hatch", "ec"} or a colour) | marker (crop name).
        Returns the y of the last row (mm)."""
        from matplotlib.patches import Rectangle

        ov = self.ov
        y = y_top
        if header:
            ov.text(x, y, header, fontsize=FS_HEAD, fontweight="bold", va="top", ha="left", color=INK)
            y -= 3.9
        for gi, (title, rows) in enumerate(groups):
            if gi:
                y -= GROUP_GAP
            if title:
                ov.text(x, y, title, fontsize=FS_TITLE, va="center", ha="left", color=INK)
                y -= ROW
            for kind, spec, text in rows:
                hx = x + 0.6 + HW / 2
                if kind == "line":
                    ls = spec[2] if len(spec) > 2 else "-"
                    ov.plot([x + 0.6, x + 0.6 + HW], [y, y], color=spec[0], lw=spec[1], ls=ls, solid_capstyle="butt")
                elif kind == "lineon":                         # a line on a ground colour (map hairlines)
                    bg, lc_, lw_ = spec
                    ov.add_patch(Rectangle((x + 0.6, y - 0.85), HW, 1.7, fc=bg, ec="#7a7a7a", lw=0.4))
                    ov.plot([x + 0.6, x + 0.6 + HW], [y, y], color=lc_, lw=lw_, solid_capstyle="butt")
                elif kind == "dot":
                    s_, fc, ec, lw = spec
                    ov.scatter([hx], [y], s=s_, c=fc, edgecolors=ec, linewidths=lw, zorder=3)
                elif kind == "patch":
                    d = spec if isinstance(spec, dict) else {"fc": spec}
                    ov.add_patch(Rectangle((x + 0.6, y - 0.85), HW, 1.7, fc=d["fc"], ec=d.get("ec", "#7a7a7a"),
                                           lw=0.4, hatch=d.get("hatch")))
                elif kind == "marker":
                    crop_line(ov, [x + 0.6, x + 0.6 + HW], [y, y], spec, marker=False)
                    st = crop_style(spec)
                    ov.plot([hx], [y], st["marker"], color=st["color"], ms=3.0, mec=INK, mew=0.3)
                ov.text(x + 0.6 + HW + TXT, y, text, fontsize=FS_ITEM, va="center", ha="left", color=INK)
                y -= ROW
        return y + ROW

    def north(self, x: float, y: float, h: float = 4.0):
        import matplotlib.pyplot as plt

        ov = self.ov
        ov.add_patch(plt.Polygon([[x, y + h], [x - 0.32 * h, y - 0.45 * h], [x, y - 0.15 * h]], fc=INK, ec=INK,
                                 lw=0.3))
        ov.add_patch(plt.Polygon([[x, y + h], [x + 0.32 * h, y - 0.45 * h], [x, y - 0.15 * h]], fc="white", ec=INK,
                                 lw=0.3))
        ov.text(x, y + h + 0.3, "N", ha="center", va="bottom", fontsize=FS_TITLE, fontweight="bold")

    def scale(self, x: float, y: float, scale: float, seg_km: float = 500, n: int = 2, bh: float = 0.9,
              z: float = 2.0, compact: bool = False):
        """Alternating bar: n segments of seg_km at `scale` metres per mm."""
        from matplotlib.patches import Rectangle

        seg = seg_km * 1000 / scale
        for i in range(n):
            self.ov.add_patch(Rectangle((x + i * seg, y), seg, bh, fc=INK if i % 2 == 0 else "white", ec=INK, lw=0.35,
                                        zorder=z))
        if compact:                                   # one label after the bar: short bars on small insets
            self.ov.text(x + n * seg + 0.8, y + bh / 2, f"{seg_km * n:g} km", ha="left", va="center",
                         fontsize=FS_SCALE, color=INK, zorder=z + 0.1)
            return n * seg
        for i in range(n + 1):
            v = seg_km * i
            self.ov.text(x + i * seg, y - 0.6, f"{v:g}" + (" km" if i == n else ""), ha="center", va="top",
                         fontsize=FS_SCALE, color=INK, zorder=z + 0.1)
        return n * seg

    def credits(self, x: float, y_top: float, data: str, extra: str | None = None, narrow: bool = False):
        proj = CREDIT_PROJ.replace(" / Brazil", " /\nBrazil") if narrow else CREDIT_PROJ
        lines = [proj, f"Data: {data}"] + ([extra] if extra else []) + [CREDIT_CARTO]
        self.ov.text(x, y_top, "\n".join(lines), fontsize=FS_CRED, color=MUTED, va="top", ha="left",
                     linespacing=1.35)

    def strip(self, left: float, scale: float, data: str, y0: float = 0.0, seg_km: float = 500):
        """The NIPE strip under a map (12 mm): north arrow, scale bar, credits."""
        self.north(left + 3.0, y0 + 3.4)
        w = self.scale(left + 6.5, y0 + 3.2, scale, seg_km)
        self.credits(left + 6.5 + w + 6.0, y0 + 7.1, data)

    def connect(self, p0: tuple[float, float], p1: tuple[float, float]):
        (ln,) = self.ov.plot([p0[0], p1[0]], [p0[1], p1[1]], color=DETAIL, lw=0.4, ls=DETAIL_LS, zorder=2)
        self._connectors.append(ln)

    # ---- elements exported one by one, for manual layout --------------------------
    def panel(self, name: str, x0: float, y0: float, x1: float, y1: float, pad: float = 0.6, hide=()):
        """Register an element of the figure (mm box); `save(..., panels_dir=)` also writes it on its own,
        with the artists in `hide` (e.g. an inset drawn over it) left out."""
        self.panels[name] = (max(0.0, x0 - pad), max(0.0, y0 - pad), min(self.fw, x1 + pad), min(self.fh, y1 + pad))
        self._hide[name] = list(hide)

    def snapshot(self) -> tuple:
        """Mark the artists drawn so far; `since(mark)` lists those drawn after it (to hide them in a panel)."""
        ov = self.ov
        return (len(self.fig.axes), len(ov.texts), len(ov.patches), len(ov.lines), len(ov.collections))

    def since(self, mark: tuple) -> list:
        ov = self.ov
        return (list(self.fig.axes[mark[0]:]) + list(ov.texts[mark[1]:]) + list(ov.patches[mark[2]:])
                + list(ov.lines[mark[3]:]) + list(ov.collections[mark[4]:]))

    def save_panels(self, folder: Path, formats=("pdf", "svg", "png"), dpi: int = 600) -> list[Path]:
        """Each registered element cropped from the figure (connectors hidden), at final size; a
        `<folder>/layout.json` keeps where each sat (mm), to rebuild or rearrange the figure by hand."""
        import json

        from matplotlib.transforms import Bbox

        if not self.panels:
            return []
        folder.mkdir(parents=True, exist_ok=True)
        for ln in self._connectors:
            ln.set_visible(False)
        paths = []
        try:
            for key, (x0, y0, x1, y1) in self.panels.items():
                bb = Bbox.from_extents(x0 * MM, y0 * MM, x1 * MM, y1 * MM)
                hidden = [h for h in self._hide.get(key, []) if h.get_visible()]
                for h in hidden:
                    h.set_visible(False)
                for fmt in formats:
                    p = folder / f"{key}.{fmt}"
                    kw = {"facecolor": "white", "bbox_inches": bb, "pad_inches": 0}
                    if fmt in ("png", "tif", "pdf"):
                        kw["dpi"] = dpi
                    self.fig.savefig(p, **kw)
                    paths.append(p)
                for h in hidden:
                    h.set_visible(True)
        finally:
            for ln in self._connectors:
                ln.set_visible(True)
        layout = {"figure_mm": [self.fw, self.fh],
                  "panels_mm": {k: {"x": round(b[0], 2), "y": round(b[1], 2), "w": round(b[2] - b[0], 2),
                                    "h": round(b[3] - b[1], 2)} for k, b in self.panels.items()}}
        (folder / "layout.json").write_text(json.dumps(layout, indent=1), encoding="utf-8")
        log.info("  %d elements in %s", len(self.panels), folder)
        return paths

    def save(self, folder: Path, name: str, formats=("pdf", "svg", "png", "tif"), dpi: int = 600,
             panels_dir: Path | None = None, panel_formats=("pdf", "svg", "png")) -> list[Path]:
        import matplotlib.pyplot as plt

        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        if panels_dir is not None:
            self.save_panels(panels_dir, [f for f in panel_formats if f in formats] or ["png"], dpi)
        for fmt in formats:
            p = folder / f"{name}.{fmt}"
            kw = {"facecolor": "white"}
            if fmt in ("png", "tif", "tiff"):
                kw["dpi"] = dpi
            if fmt in ("tif", "tiff"):
                kw["pil_kwargs"] = {"compression": "tiff_lzw"}
            if fmt == "pdf":
                kw["dpi"] = dpi                      # resolution of rasterised layers (choropleths, chips)
            self.fig.savefig(p, **kw)
            paths.append(p)
        plt.close(self.fig)
        log.info("  %s: %.1f x %.1f mm", name, self.fw, self.fh)
        return paths


def map_frame(a, extent: tuple[float, float, float, float]):
    """No axis frame on maps (NIPE): limits, no ticks, no spines, no GeoPandas labels."""
    a.set_xlim(extent[0], extent[1])
    a.set_ylim(extent[2], extent[3])
    a.set_aspect("auto")                   # the axes already have the extent's aspect at scale S
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel("")
    a.set_ylabel("")
    for s in a.spines.values():
        s.set_visible(False)


def detail_border(a):
    for s in a.spines.values():
        s.set_visible(True)
        s.set_edgecolor(DETAIL)
        s.set_linestyle(DETAIL_LS)
        s.set_linewidth(0.6)


# ---------------------------------------------------------------------------
# legibility checks: greyscale and deuteranopia renderings of the PNG
# ---------------------------------------------------------------------------
# Machado, Oliveira & Fernandes (2009), deuteranomaly severity 1.0, linear RGB
_DEUTAN = np.array([[0.367322, 0.860646, -0.227968],
                    [0.280085, 0.672501, 0.047413],
                    [-0.011820, 0.042940, 0.968881]])


def _to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.round(255 * np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)).astype(np.uint8)


def simulate(png: Path, out_dir: Path, max_px: int = 1600) -> list[Path]:
    """Greyscale and deuteranopia versions of a figure PNG, for a visual check."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    im = Image.open(png).convert("RGB")
    im.thumbnail((max_px, max_px))
    grey = out_dir / f"{png.stem}_grey.png"
    im.convert("L").save(grey)
    deut = _to_srgb(_to_linear(np.asarray(im, dtype=float)) @ _DEUTAN.T)
    dpath = out_dir / f"{png.stem}_deutan.png"
    Image.fromarray(deut).save(dpath)
    return [grey, dpath]
