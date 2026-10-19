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
            "used throughout, other cover in light neutrals), and each printed pixel shows the cell under it."
            f"{sc} Lines: the {f['states']} states, the {f['regions']} intermediate regions and the biomes "
            f"(MapBiomas).{det}{check} Right: Brazil and the EU-27 on one Lambert equal-area scale; São Paulo State, "
            "the United Kingdom and Germany side by side on one scale (areas of the outlines drawn: IBGE mesh, "
            "Natural Earth 1:50 m). SIRGAS 2000 / Brazil Polyconic (EPSG:5880).")


def captions(cfg: Config, nums: dict, facts: dict | None = None) -> dict[str, str]:
    p0, p1 = cfg.years[0], cfg.years[-1]
    sy = _get(nums, "study_area", "summary_years", default=[])
    s0, s1 = (sy[0], sy[-1]) if sy else ("[n/a]", "[n/a]")
    dec = _get(nums, "h3", "source_effect_by_decile", "all_cells", default=[])
    top = dec[-1] if dec else {}
    bottom = dec[0] if dec else {}
    min_area = cfg.section("reconcile").get("min_area_ha", 0)
    return {
        "fig01_study_area": _fig01(cfg, (facts or {}).get("fig01")),
        "fig02_workflow":
            "The open workflow (bbp): sources grouped by agency, the stages with the number of invariant checks "
            "each runs, the products, and what the checks caught in the source data. Every number in the text "
            "comes from numbers.json, written by the same run.",
        "fig03_national_area":
            f"National crop area {p0}–{p1}: observed (MapBiomas Collection 11; second-season corn from the "
            f"second-crop layer) and declared (IBGE PAM). Shaded: {s0}–{s1}, the years summarised in Figs 4, 5 "
            "and 7–10.",
        "fig04_agreement_scale":
            f"Agreement between observed and declared area from municipality to state: share of the units "
            f"where the two sources are within ±25 % of each other, weighted by declared area (left) and by "
            f"number of units (right), mean {s0}–{s1}. Units where neither source reaches {min_area:g} ha are "
            "not scored; a unit with area in one source only counts as disagreeing. Dashed: (semi-)perennials.",
        "fig05_log_ratio_maps":
            f"Observed / declared area per municipality, median {s0}–{s1}, on classes symmetric on the log "
            "scale (±10, 25, 50, 100 %); all maps at the same scale. Hatched: the crop is reported by one source "
            f"only. Blank: neither source reaches {min_area:g} ha. SIRGAS 2000 / Brazil Polyconic.",
        "fig06_example_municipalities":
            "Example municipalities on the MapBiomas Collection 11 coverage map (30 m), with the observed and "
            "declared area of the crop. Chosen by rule (fig06_candidates.csv): (a) soybean with intense double "
            "cropping, (b) a perennial with much of its area near the municipal border, (c) coffee where the "
            "area destined for harvest far exceeds the mapped area, (d) a municipality where the sources agree. "
            "DRAFT: candidates to be confirmed.",
        "fig07_h2_variance":
            f"Why the sources diverge (H2): share of the variance of the municipal ln(A_obs/A_dec), {s0}–{s1}, "
            "explained by the predictors (marginal R²), by the intermediate region (random intercept) and left "
            "as residual, per crop. Right: Moran's I of the residuals (queen contiguity; * p < 0.05).",
        "fig08_h2_effects":
            "Standardised effects of the predictors in the mixed model: change in ln(A_obs/A_dec) per standard "
            "deviation of each predictor, with 95 % confidence intervals, per crop.",
        "fig09_source_vs_factors":
            f"Area source against conversion factors (H3): multiplicative difference between municipal N₁ "
            f"computed from observed and from declared area, {s0}–{s1} (municipality-years with both > 0; "
            "thin line 10th–90th percentile, thick line interquartile range, open circle median, tick 90th "
            "percentile), against the spread of N₁ over the literature factor ranges (grey band: all factors "
            "jointly, min to max; darker: BMP alone).",
        "fig10_source_effect_deciles":
            f"Does the area source matter where the potential is? (C4, droppable.) Municipalities ranked by "
            f"declared N₁ (mean {s0}–{s1}, crops with an observed area) into deciles: median and 90th percentile "
            f"of the multiplicative difference between N₁ from observed and declared area (top) and the share "
            f"of national N₁ in each decile (bottom). The top decile holds {_pct(top.get('share_of_national_n1'))} "
            f"of national N₁ with a median difference of ×{_exp(top.get('median_abs_log'))}; the bottom decile "
            f"×{_exp(bottom.get('median_abs_log'))}. The same with crop cells below {min_area:g} ha removed: "
            "Table 5.",
        "figS1_state_detail":
            f"As Fig 5 for São Paulo and Minas Gerais (median {s0}–{s1}): the dashed box on each national map "
            "marks the detail on its right, drawn at its own scale (bar beside it).",
        "figS2_rank_shift":
            "Rank correlation (Spearman) of municipal N₁ computed from the two area sources, and the share of "
            "the top 10 % of municipalities that changes, per year.",
        "figS3_scatters":
            "Municipal second-season corn area (left) and municipal N₁, all crops (right), observed vs declared, "
            f"{s1}; dashed: 1:1. Shading: number of municipalities (log).",
        "figS4_ccc_scale":
            f"Lin's concordance correlation coefficient between observed and declared area by scale, mean "
            f"{s0}–{s1}. It stays near 1 because a few large units dominate it, which is why Fig 4 uses the "
            "share within ±25 % instead.",
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
        name = file_name(key)
        label = f"Figure {key}" if isinstance(key, int) else f"Figure {key}"
        lines += [f"**{label}.** {caps.get(name, '')}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  wrote %s", path)
    return path
