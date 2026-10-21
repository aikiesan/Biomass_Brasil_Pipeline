"""Figure captions (data/<paper.out_dir>/captions.md), with the numbers of the run filled in.

Numbers come from numbers.json when `bbp paper` wrote it, otherwise from the tables; a caption whose
number is not available says so ("[n/a]") rather than inventing one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from bbp.config import Config

log = logging.getLogger("bbp")


def _get(d: dict, *keys, default="[n/a]"):
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d


def _pct(v, digits=0) -> str:
    return f"{v:.{digits}%}" if isinstance(v, (int, float)) else str(v)


def _fig01(cfg: Config, facts: dict) -> str:
    """Fig 1 caption from what the figure itself measured (bbp.figures sets data.facts['fig01'])."""
    spec = cfg.section("paper").get("fig1") or {}
    f = (facts or {}).get(int(spec.get("year", cfg.years[-1]))) or {}
    if not f:
        return "Study area."
    year = f["year"]
    sc = (f" The second-crop layer (beta) splits off soybean followed by 2nd-season corn "
          f"({f['soy_corn2_mha']:.1f} of the {f['soybean_mha']:.1f} Mha of soybean) or cotton."
          if f.get("second_crop") and f.get("soybean_mha") else "")
    dev = f.get("max_dev_statistics")
    check = (f" The display groups reproduce the national Collection 11 area of every crop class to within "
             f"{max(abs(dev), 1e-4):.2%}." if isinstance(dev, (int, float)) else "")
    sp = next((d for d in f.get("details", []) if d.get("uf")), None)
    corner = [d for d in f.get("details", []) if not d.get("uf")]
    det = ""
    if sp or corner:
        names = [f"({chr(98 + i)}) {d['title']}" for i, d in enumerate(corner + ([sp] if sp else []))]
        det = (f" {' and '.join(names)}, with the municipal mesh; dots: the seat of each intermediate region (centre "
               "of its urban area), plus named towns.")
    return (f"Study area at three resolutions. (a) Land cover of Brazil in {year}, MapBiomas Collection 11 (30 m), "
            "drawn in display groups: every 480 m cell takes the group that covers most of it (crops in the colours "
            "used throughout; natural vegetation, pasture and forest plantation in MapBiomas legend colours), and "
            f"each printed pixel shows the cell under it.{sc} Lines: the {f['states']} states and the "
            f"{f['regions']} intermediate regions; italic: the biomes (MapBiomas).{det}{check} Top right: Brazil and the EU-27 on one Lambert equal-area scale (areas: "
            "IBGE mesh, Natural Earth 1:50 m). SIRGAS 2000 / Brazil Polyconic (EPSG:5880).")


def _hot_spots(year, hs: dict) -> str:
    head = (f"Where N₁ per km² clusters ({year}): (a) Getis–Ord Gi* z-scores (binary weights with the unit itself), "
            "(b) local Moran clusters (LISA, p < 0.05")
    if not hs:
        return head + ").  [n/a: run bbp figures]"
    lisa = hs.get("lisa") or {}
    return (f"{head}, {hs['permutations']} conditional permutations, no correction for multiple tests); queen "
            f"contiguity on the municipal mesh. Global Moran's I {hs['morans_i']:.2f} (p = {hs['p_value']:.3f}); "
            f"Gi* at 95 %: {hs['gi_hot_95']} hot and {hs['gi_cold_95']} cold municipalities; LISA: "
            f"{lisa.get('HH', 0)} high–high, {lisa.get('LL', 0)} low–low.")


def captions(cfg: Config, nums: dict, facts: dict | None = None) -> dict[str, str]:
    """Captions keyed by figure name (the file name without its number)."""
    facts = facts or {}
    p0, p1 = cfg.years[0], cfg.years[-1]
    sy = _get(nums, "study_area", "summary_years", default=[])
    s0, s1 = (sy[0], sy[-1]) if sy else ("[n/a]", "[n/a]")
    dec = _get(nums, "h3", "source_effect_by_decile", "all_cells", default=[])
    top = dec[-1] if dec else {}
    bottom = dec[0] if dec else {}
    min_area = cfg.section("reconcile").get("min_area_ha", 0)
    demo = cfg.section("demo")
    year = demo.get("year", p1)
    ty = demo.get("typology", {})
    yld = cfg.section("plants").get("reach", {}).get("cane_yield_t_ha", "[n/a]")
    return {
        # ---- paper ---------------------------------------------------------------------------------------------
        "study_area": _fig01(cfg, facts.get("fig01")),
        "workflow":
            "The open workflow (bbp) drawn as a transit map. Each data family is a line (CP2b factors, MapBiomas "
            "satellite data, IBGE statistics, plant registers); each stage is a station (round: tabular stage; "
            "square: 30 m pass, a command of its own whose tables publish releases), with the number of invariant "
            "checks it runs. The lines flow into one trunk where the families are compared (observed vs declared "
            "area, H1–H3) and used (demonstration), ending at publish (release, manifest, STAC). Red numbers: what "
            "the checks caught in the source data, marked where it was caught. Every number in the text comes from numbers.json and the "
            "check logs written by the same run.",
        "agreement_scale":
            f"Do the satellite (MapBiomas) and the statistics (IBGE PAM) agree, and at what scale? (H1), mean "
            f"{s0}–{s1}. (a) Share of the declared area that lies in units where the two sources are within ±25 % "
            "of each other (|observed/declared − 1| ≤ 0.25), from the municipality to the state (AMCs, not shown, "
            "match the municipalities). (b) Ratio of the national totals, observed/declared; grey band: the same "
            f"±25 %. Units where neither source reaches {min_area:g} ha are not scored; a unit with area in one "
            "source only counts as disagreeing. Dashed: (semi-)perennials. The share of units within ±25 % is in "
            "Table 2.",
        "log_ratio_maps":
            f"Where the satellite sees more or less than the statistics: observed (MapBiomas) / declared (IBGE PAM) "
            f"area per municipality, median {s0}–{s1}, for soybean, second-season corn and sugarcane. Grey: the "
            "two sources agree within ±25 %, as in Fig 3; red: the satellite sees more; blue: less. Hatched or "
            f"dotted: the crop is reported by one source only. Blank: neither source reaches {min_area:g} ha. "
            "Below each map: the share of the declared area within ±25 %. The other crops: Fig S15. SIRGAS 2000 / "
            "Brazil Polyconic.",
        "h2_drivers":
            f"Why the sources diverge (H2): municipal ln(A_obs/A_dec), {s0}–{s1}, one mixed model per crop "
            "(rows). Left: share of the variance explained by the predictors (marginal R²), by the intermediate "
            "region (random intercept) and left unexplained, and Moran's I of the residuals (queen "
            "contiguity; * p < 0.05). Right: standardised effect of each predictor, change in ln(A_obs/A_dec) per "
            "standard deviation, in the colours of Fig 4 (red: the satellite sees more; blue: less); faded and "
            "light: the 95 % confidence interval crosses zero; —: predictor not in the model.",
        "source_vs_factors":
            f"Which matters more for municipal N₁: the area source or the conversion factors? (H3), mean "
            f"{s0}–{s1}. Dots: how much municipal N₁ changes when observed area replaces declared area "
            "(multiplicative, either direction; municipalities with both > 0), for the typical municipality "
            "(open, median) and the 10 % most divergent (filled, 90th percentile). Grey band: how much N₁ changes "
            "over the literature ranges of all conversion factors jointly (min to max). Right: the two as a "
            "ratio, area source ÷ factors (Table 4); red where the area source moves N₁ more.",
        "distance_to_plants":
            f"Crops within reach of existing plants, MapBiomas Collection 11 {year}, 30 m pixels (second-season "
            "corn: second-crop layer). Top: share of each crop's area by straight-line (great-circle) distance to "
            "the nearest operating plant that uses it, with the share within 30 km, the median distance and the "
            "crop area. Bottom: the share within 30 km per municipality, one map per crop–plant pair; dots: "
            "operating plants. Road distances are longer. The plants by register: Fig S16.",
        "plants_by_register":
            f"The plants by family and register, over the crops they draw on (MapBiomas Collection 11 {year}); "
            "marker area proportional to capacity; open markers: construction, expansion or inactive.",
        "maup_access":
            f"The same question read at three supports (MAUP): share of each crop's area within 30 km of a "
            f"paired plant, {year}, measured from the 30 m pixels (reference), from the land of the municipality "
            "and from the land of the intermediate region (a unit's crop is spread like its land: area-fraction "
            "rule). Right: the region minus the pixel, in percentage points (red: 3 points or more). Other radii "
            "and the centroid and representative-point rules: table reach_scale of the release.",
        "typology_mandate":
            f"A demonstration. (a) Types of crop composition of the 133 intermediate regions ({year}; centred "
            "log-ratio of the crop shares, Ward clustering, number of types chosen by stability); grey: less "
            f"than {ty.get('min_crop_kha', '[n/a]')} kha of crops. (b) N₁ of each region's crop residues "
            "(observed area, CP2b reference factors) against the share of its crop area within 30 km of a "
            "biogas plant; marker area: biogas plants in the region; ringed and numbered: priority regions (top "
            "quartile of N₁, under 25 % within 30 km; shaded). Types coloured after their dominant crop, as in "
            "Figs 3–7, and listed by N₁ (number of regions). (c) Biomethane mandate on one axis, million m³ per "
            "day: targets (filled: official volumes, CNPE Res. 4/2026 and MME Ordinance 942/2026; open: "
            "2028–2030, derived from the 2027 base), ANP production and authorised capacity at the cut-off "
            "(flows), and N₁ of the crops within 30 km of the existing biomethane plants (stock; band: factor "
            "envelope). The stock is theoretical methane, not an available supply.",
        # ---- supplement ------------------------------------------------------------------------------------------
        "national_area":
            f"National crop area {p0}–{p1}: observed (MapBiomas Collection 11; second-season corn from the "
            f"second-crop layer) and declared (IBGE PAM). Shaded: {s0}–{s1}, the years summarised in Figs 3–6.",
        "example_municipalities":
            "Example municipalities on the MapBiomas Collection 11 coverage map (30 m), with the observed and "
            "declared area of the crop. Chosen by rule (fig06_candidates.csv): (a) soybean with intense double "
            "cropping, (b) a perennial with much of its area near the municipal border, (c) coffee where the "
            "area destined for harvest far exceeds the mapped area, (d) a municipality where the sources agree. "
            "DRAFT: candidates to be confirmed.",
        "state_detail":
            f"As Fig 4 for São Paulo and Minas Gerais (median {s0}–{s1}): the dashed box on each national map "
            "marks the detail on its right, drawn at its own scale (bar beside it).",
        "rank_shift":
            "Rank correlation (Spearman) of municipal N₁ computed from the two area sources, and the share of "
            "the top 10 % of municipalities that changes, per year.",
        "scatters":
            "Municipal second-season corn area (left) and municipal N₁, all crops (right), observed vs declared, "
            f"{s1}; dashed: 1:1. Shading: number of municipalities (log).",
        "ccc_scale":
            f"Lin's concordance correlation coefficient between observed and declared area by scale, mean "
            f"{s0}–{s1}. It stays near 1 because a few large units dominate it, which is why Fig 3 uses the "
            "share within ±25 % instead.",
        "source_effect_deciles":
            f"Does the area source matter where the potential is? Municipalities ranked by declared N₁ (mean "
            f"{s0}–{s1}, crops with an observed area) into deciles: median and 90th percentile of the "
            f"multiplicative difference between N₁ from observed and declared area (top) and the share of "
            f"national N₁ in each decile (bottom). The top decile holds {_pct(top.get('share_of_national_n1'))} "
            f"of national N₁ with a median difference of ×{_exp(top.get('median_abs_log'))}; the bottom decile "
            f"×{_exp(bottom.get('median_abs_log'))}. The same with crop cells below {min_area:g} ha removed: "
            "Table 5.",
        "sp_mills_catchments":
            f"São Paulo sugarcane mills and the cane around them ({year}, 30 m): the radius each mill needs to "
            f"fill its crushing class at {yld} t/ha, after Cervi et al. (2019).",
        "mill_catchments":
            f"Sugarcane mills and their catchments, Brazil ({year}, 30 m pixels): (a) the radius each mill needs "
            f"at {yld} t/ha, (b) cane area within each radius, (c) cane within 30 km over the cane the mill "
            "needs, by crushing class, (d) share of the cane area by the number of mills within 30 km.",
        "regions_within_reach":
            "Share of each crop's area within reach of the plants that use it, per intermediate region.",
        "region_typology":
            "Typology of the intermediate regions in full: (a) types, (b) share of all crops within 30 km of a "
            "biogas plant, (c) N₁ against that share, priority regions numbered and listed.",
        "registers_biomethane":
            "The plant registers compared: (a) cane mills per state in ANP, CONAB and MapBiomas/EPE; (b) distance "
            "from each ANP mill to the nearest unit of the other register; (c) the ANP biomethane plants by "
            "authorised capacity (green: at a cane mill).",
        "n1_by_level":
            f"N₁ per km² ({year}, observed area, CP2b reference factors) at three levels on the same classes "
            "(quantiles of the municipal density): (a) state, (b) intermediate region, (c) municipality.",
        "n1_hot_spots": _hot_spots(year, facts.get("n1_hot_spots") or {}),
        "log_ratio_maps_all_crops":
            f"Fig 4 for every crop: observed / declared area per municipality, median {s0}–{s1}, on the same "
            "classes and scale.",
    }


def _exp(v) -> str:
    import math

    return f"{math.exp(v):.2f}" if isinstance(v, (int, float)) else "[n/a]"


def write_captions(cfg: Config, data, path: Path) -> Path:
    nums_path = path.parent / "numbers.json"
    nums = json.loads(nums_path.read_text(encoding="utf-8")) if nums_path.exists() else {}
    from bbp.figures import FIGURES, SUPPLEMENT, file_name

    caps = captions(cfg, nums, getattr(data, "facts", None))
    lines = ["# Figure captions", "", f"Generated by bbp figures (profile {cfg.profile}); numbers from "
             f"{'numbers.json' if nums else 'no numbers.json: run bbp paper'}.", ""]
    for key in [*FIGURES, *SUPPLEMENT]:
        name = (FIGURES.get(key) or SUPPLEMENT[key])[0]
        lines += [f"**Figure {key}** ({file_name(key)}). {caps.get(name, '[no caption]')}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  wrote %s", path)
    return path
