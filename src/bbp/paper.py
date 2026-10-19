"""Every number of the paper, from the pipeline: `paper/numbers.json` and Tables 1–4.

    bbp paper                  numbers.json + tables (+ figures, see bbp.figures)

Reads data/processed (the full run), keeps the profile's years and crops, and writes
data/<paper.out_dir>/: numbers.json, tables/table<N>_<name>.csv and .md. The text of the
paper quotes numbers.json; nothing is typed by hand.

* Table 1 — data sources: version, period used, access, licence (facts from the profile,
  periods from the data).
* Table 2 — agreement metrics by scale and crop (mean over the summary years).
* Table 3 — H2 models: fit, region variance, out-of-region RF R², Moran's I; and the
  standardised mixed-model coefficients (3b).
* Table 4 — source effect vs factor effect on N1 by crop (mean over the summary years).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import bbp
from bbp.checks import CheckLog
from bbp.config import Config

log = logging.getLogger("bbp")

SCALES = ["municipality", "amc", "rgint", "uf", "nation"]
SCALE_LABELS = {"municipality": "Municipality", "amc": "AMC", "rgint": "Intermediate region", "uf": "State",
                "nation": "Brazil"}
AGREEMENT_COLS = ["n_units", "ratio_total", "median_log_ratio", "share_within_10pct", "share_within_25pct",
                  "area_share_within_25pct", "rel_abs_error", "lin_ccc", "spearman_rho"]
PREDICTOR_LABELS = {"log_mun_area_ha": "Municipal area (log)", "crop_share": "Crop share of municipality",
                    "log_double_crop": "Double cropping (log)", "irrigated_share": "Irrigated share",
                    "border_share_1km": "Crop within 1 km of border", "log_landsat_valid_obs": "Clear Landsat obs. (log)",
                    "dominant_biome": "Biome", "year": "Year", "new_municipality": "New municipality"}
PROCESSED = ["agreement_by_scale", "unit_ratios_nation", "municipality_attributes", "n1_municipal",
             "source_vs_factor_effect", "rank_shift", "crop_area", "h2_mixed_effects", "h2_model_fit",
             "h2_rf_importance", "h2_moran"]
OPTIONAL = {"h2_mixed_effects", "h2_model_fit", "h2_rf_importance", "h2_moran"}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def out_dir(cfg: Config) -> Path:
    d = cfg.data_dir / cfg.section("paper").get("out_dir", "paper")
    d.mkdir(parents=True, exist_ok=True)
    return d


def summary_years(cfg: Config) -> list[int]:
    years = cfg.section("paper").get("summary_years") or cfg.section("reconcile").get("reference_years") \
        or cfg.years[-5:]
    return [int(y) for y in years if int(y) in cfg.years]


def restrict(df: pd.DataFrame, years: list[int] | None, crops: list[str] | None) -> pd.DataFrame:
    """Rows of the profile's years and crops (tables without those columns pass through)."""
    out = df
    if years is not None and "year" in out:
        out = out[out["year"].astype(int).isin(years)]
    if crops is not None and "crop" in out:
        out = out[out["crop"].astype(str).isin(crops)]
    if "crop" in out and isinstance(out["crop"].dtype, pd.CategoricalDtype):
        out = out.assign(crop=out["crop"].astype(str))
    return out.reset_index(drop=True)


def load_tables(cfg: Config, ck: CheckLog, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """data/processed tables restricted to the profile; missing optional (H2) tables are skipped with a warning."""
    crops = list(cfg.crops)
    tables = {}
    for name in names or PROCESSED:
        p = cfg.data_dir / "processed" / f"{name}.parquet"
        optional = name in OPTIONAL
        if not ck.add(f"input_{name}", p.exists(), "" if p.exists() else f"{p.name} missing — run the stage first",
                      severity="warn" if optional else "error"):
            continue
        raw = pd.read_parquet(p)
        df = restrict(raw, cfg.years, crops)
        if "year" in raw:
            missing = sorted(set(cfg.years) - set(raw["year"].astype(int)))
            # MapBiomas second crop and the H2 models do not cover every year by design
            ck.add(f"input_{name}_covers_period", not missing or name.startswith("h2_") or name == "rank_shift",
                   f"years absent: {missing}" if missing else f"{cfg.years[0]}–{cfg.years[-1]}", missing,
                   severity="warn")
        tables[name] = df
    return tables


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def _years_label(years) -> str:
    years = sorted({int(y) for y in years})
    if not years:
        return "—"
    return str(years[0]) if len(years) == 1 else f"{years[0]}–{years[-1]}"


def source_periods(cfg: Config) -> dict[str, str]:
    """Period used of each source, read from the interim tables (years outside the profile dropped)."""
    keep = set(cfg.years)
    out: dict[str, str] = {}

    def period(*parts: str) -> str:
        p = cfg.data_dir.joinpath(*parts)
        if not p.exists():
            return "not built"
        years = pd.read_parquet(p, columns=["year"])["year"].astype(int).unique().tolist()
        kept = [y for y in years if y in keep]
        return _years_label(kept) if kept else f"outside the period (built: {_years_label(years)})"

    for key, parts in {"mapbiomas_municipal_xlsx": ("interim", "mapbiomas_coverage_municipal.parquet"),
                       "second_crop_tif_dir": ("interim", "second_crop_municipal.parquet"),
                       "mapbiomas_irrigation_xlsx": ("interim", "mapbiomas_irrigation_municipal.parquet"),
                       "coverage_tif_dir": ("interim", "border_share_municipal.parquet")}.items():
        out[key] = period(*parts)
    for table in cfg.sidra_tables():
        out[f"sidra_{table}"] = period("interim", "sidra", f"t{table}.parquet")
    out["municipalities_shp"] = "—"
    out["conversion_factors"] = "—"
    return out


def table_sources(cfg: Config) -> pd.DataFrame:
    """Table 1. Facts the files cannot tell (version, access, licence) come from the profile."""
    periods = source_periods(cfg)
    files: dict[str, list[str]] = {}
    for key, p in cfg.input_files():
        files.setdefault(key, []).append(p.name)
    for key in ("second_crop_tif_dir", "coverage_tif_dir"):
        if key in cfg.raw.get("inputs", {}):
            files[key] = [Path(cfg.raw["inputs"][key]).name + "/"]
    files["conversion_factors"] = ["config/conversion_factors.yaml"]
    rows = []
    for s in cfg.section("paper").get("sources", []):
        rows.append({"source": s["source"], "version": s.get("version", ""), "period": periods.get(s["key"], "—"),
                     "access": s.get("access", ""), "licence": s.get("licence", ""),
                     "files": "; ".join(files.get(s["key"], []))})
    return pd.DataFrame(rows)


def table_agreement(agreement: pd.DataFrame, years: list[int], crops: list[str]) -> pd.DataFrame:
    """Table 2: mean over `years` of each metric, per crop and scale (scales from fine to coarse)."""
    a = agreement[agreement["year"].isin(years)]
    t = a.groupby(["crop", "scale"], observed=True)[AGREEMENT_COLS].mean().reset_index()
    t["n_units"] = t["n_units"].round().astype("int64")
    t["crop"] = pd.Categorical(t["crop"], [c for c in crops if c in set(t["crop"])], ordered=True)
    t["scale"] = pd.Categorical(t["scale"], [s for s in SCALES if s in set(t["scale"])], ordered=True)
    t = t.sort_values(["crop", "scale"]).reset_index(drop=True)
    t.insert(2, "years", _years_label(years))
    return t


def table_h2(fit: pd.DataFrame, moran: pd.DataFrame, importance: pd.DataFrame | None,
             crops: list[str]) -> pd.DataFrame:
    """Table 3: one row per crop."""
    t = fit.merge(moran[["crop", "morans_i", "p_value"]].rename(columns={"p_value": "moran_p"}), on="crop", how="left")
    if importance is not None and len(importance):
        top = importance.sort_values("importance_mean", ascending=False).drop_duplicates("crop")
        t = t.merge(top[["crop", "predictor"]].rename(columns={"predictor": "rf_top_predictor"}), on="crop", how="left")
    cols = ["crop", "years", "n", "municipalities", "regions", "model", "r2_marginal", "r2_conditional",
            "region_variance_share", "rf_r2_grouped_cv", "rf_top_predictor", "morans_i", "moran_p", "predictors"]
    t = t[[c for c in cols if c in t]]
    order = {c: i for i, c in enumerate(crops)}
    return t.sort_values("crop", key=lambda s: s.map(order)).reset_index(drop=True)


def table_coefficients(coefs: pd.DataFrame, crops: list[str]) -> pd.DataFrame:
    """Table 3b: standardised fixed effects (1 SD -> change in log-ratio) of the numeric predictors and biome."""
    c = coefs[~coefs["term"].str.startswith(("Intercept", "C(year)"))].copy()
    c["term"] = c["term"].str.replace(r"C\(dominant_biome\)\[T\.(.+)\]", r"biome: \1", regex=True)
    order = {k: i for i, k in enumerate(crops)}
    c = c.sort_values(["crop", "term"], key=lambda s: s.map(order) if s.name == "crop" else s)
    return c[["crop", "term", "estimate", "std_error", "ci_low", "ci_high", "p_value"]].reset_index(drop=True)


def table_effects(sve: pd.DataFrame, years: list[int], crops: list[str]) -> pd.DataFrame:
    """Table 4: the area-source effect against the conversion-factor envelope, as |log-ratios|.

    `source_vs_factor_ratio` > 1: switching the activity-data source moves municipal N1 more
    (median) than the joint spread of the literature factors.
    """
    s = sve[sve["year"].isin(years)].copy()
    s["ratio_of_totals"] = s["sum_obs_nm3"] / s["sum_dec_nm3"]
    cols = ["n_both", "ratio_of_totals", "source_median_abs_log", "source_p90_abs_log", "source_share_gt10pct",
            "factor_envelope_log", "factor_oat_g_log", "factor_oat_ts_log", "factor_oat_vs_ts_log",
            "factor_oat_bmp_log"]
    t = s.groupby("crop")[cols].mean().reset_index()
    t["n_both"] = t["n_both"].round().astype("int64")
    t["source_vs_factor_ratio"] = t["source_median_abs_log"] / t["factor_envelope_log"]
    t["source_p90_vs_factor_ratio"] = t["source_p90_abs_log"] / t["factor_envelope_log"]
    order = {c: i for i, c in enumerate(crops)}
    t = t.sort_values("crop", key=lambda x: x.map(order)).reset_index(drop=True)
    t.insert(1, "years", _years_label(years))
    return t


def source_effect_by_decile(n1: pd.DataFrame, years: list[int], area: pd.DataFrame | None = None,
                            min_area_ha: float = 0.0, n_bins: int = 10) -> pd.DataFrame:
    """C4: does the area source matter where the potential is? |ln(N1_obs / N1_dec)| by decile of municipal N1.

    Municipal N1 = sum over the crops with an observed area (declared-only crops excluded: identical in
    both variants by construction), per year, then the mean over `years`. Only (municipality, year, crop)
    cells where both variants exist enter; with `min_area_ha` > 0 also only cells where either source
    reports at least that area. Municipalities with N1_dec > 0 and N1_obs > 0 are ranked by N1_dec into
    `n_bins` equal-count groups (1 = smallest). Per decile: municipalities, share of national N1_dec,
    median and p90 of |ln(N1_obs / N1_dec)|.
    """
    n = n1[n1["year"].isin(years) & ~n1["activity_declared_only"].astype(bool)]
    w = n.pivot_table(index=["geocode", "year", "crop"], columns="variant", values="n1_ref_nm3",
                      observed=True).dropna(subset=["dec", "obs"])
    if min_area_ha > 0 and area is not None:
        a = area.assign(crop=area["crop"].astype(str)).set_index(["geocode", "year", "crop"])
        big = np.maximum(a["a_obs_ha"].fillna(0), a["a_dec_ha"].fillna(0)) >= min_area_ha
        w = w[big.reindex(w.index).fillna(False).to_numpy()]
    mun = w.groupby(level=["geocode", "year"]).sum().groupby(level="geocode").mean()
    mun = mun[(mun["dec"] > 0) & (mun["obs"] > 0)]
    if mun.empty:
        return pd.DataFrame()
    mun["abs_log"] = np.abs(np.log(mun["obs"] / mun["dec"]))
    mun["decile"] = pd.qcut(mun["dec"].rank(method="first"), n_bins, labels=range(1, n_bins + 1)).astype(int)
    total = mun["dec"].sum()
    out = mun.groupby("decile").agg(municipalities=("dec", "size"), n1_dec_min_nm3=("dec", "min"),
                                    n1_dec_max_nm3=("dec", "max"), n1_dec_sum_nm3=("dec", "sum"),
                                    median_abs_log=("abs_log", "median"),
                                    p90_abs_log=("abs_log", lambda v: v.quantile(0.9))).reset_index()
    out.insert(4, "share_of_national_n1", out.pop("n1_dec_sum_nm3") / total)
    out["min_area_ha"] = min_area_ha
    out["years"] = _years_label(years)
    return out


# ---------------------------------------------------------------------------
# numbers.json
# ---------------------------------------------------------------------------
def _num(x: Any, digits: int = 6) -> Any:
    """JSON-safe value: floats to `digits` significant digits, NaN -> None, numpy scalars to Python."""
    if isinstance(x, dict):
        return {str(k): _num(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_num(v, digits) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if not math.isfinite(x) else float(f"{x:.{digits}g}")
    return x


def _summary(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {}
    return {"mean": s.mean(), "min": s.min(), "max": s.max(), "min_at": s.idxmin(), "max_at": s.idxmax()}


def numbers(cfg: Config, t: dict[str, pd.DataFrame], years: list[int]) -> dict:
    crops = list(cfg.crops)
    attrs = t["municipality_attributes"]
    mun = attrs[attrs["is_municipality"]] if "is_municipality" in attrs else attrs
    out: dict[str, Any] = {
        "study_area": {
            "municipalities": int(len(mun)),
            "new_municipalities": int(mun["created_after_start"].sum()),
            "amc": int(mun["amc_id"].nunique()),
            "amc_with_more_than_one_municipality": int((mun.groupby("amc_id").size() > 1).sum()),
            "intermediate_regions": int(mun["cd_rgint"].nunique()),
            "states": int(mun["uf"].nunique()),
            "period": [cfg.years[0], cfg.years[-1]],
            "summary_years": years,
            "crops": crops,
            "crops_area_only": [c for c in crops if not cfg.crops[c].get("feedstock")],
            "crops_declared_only": [c for c in crops if not (cfg.crops[c].get("mapbiomas_class")
                                                             or cfg.crops[c].get("mapbiomas_second_crop"))],
        }
    }

    # ---- H1 ------------------------------------------------------------------
    nat = t["unit_ratios_nation"]
    ratio = nat.pivot_table(index="year", columns="crop", values="ratio", observed=True)
    obs = nat.pivot_table(index="year", columns="crop", values="a_obs_ha", observed=True)
    dec = nat.pivot_table(index="year", columns="crop", values="a_dec_ha", observed=True)
    h1: dict[str, Any] = {"national_ratio": {}, "national_ratio_summary": {}, "national_area_mha": {},
                          "by_scale": {}}
    for c in [c for c in crops if c in ratio]:
        h1["national_ratio"][c] = ratio[c].dropna().to_dict()
        h1["national_ratio_summary"][c] = {**_summary(ratio[c]),
                                           "summary_years_mean": ratio[c].reindex(years).mean()}
        h1["national_area_mha"][c] = {"observed": (obs[c] / 1e6).dropna().to_dict(),
                                      "declared": (dec[c] / 1e6).dropna().to_dict()}
    tab2 = table_agreement(t["agreement_by_scale"], years, crops)
    for r in tab2.itertuples(index=False):
        h1["by_scale"].setdefault(str(r.crop), {})[str(r.scale)] = {k: getattr(r, k) for k in AGREEMENT_COLS}
    out["h1"] = h1

    # ---- H2 ------------------------------------------------------------------
    if "h2_model_fit" in t and "h2_moran" in t:
        tab3 = table_h2(t["h2_model_fit"], t["h2_moran"], t.get("h2_rf_importance"), crops)
        h2: dict[str, Any] = {}
        coefs = table_coefficients(t["h2_mixed_effects"], crops) if "h2_mixed_effects" in t else None
        for r in tab3.to_dict("records"):
            crop = r.pop("crop")
            if coefs is not None:
                cc = coefs[coefs["crop"] == crop].set_index("term")
                r["coefficients"] = {k: {"estimate": v["estimate"], "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                                         "p_value": v["p_value"]} for k, v in cc.to_dict("index").items()}
            h2[crop] = r
        out["h2"] = h2

    # ---- H3 ------------------------------------------------------------------
    n1 = t["n1_municipal"]
    tot = n1.groupby(["variant", "year"], observed=True)["n1_ref_nm3"].sum() / 1e9
    obs_only = n1[~n1["activity_declared_only"]].groupby(["variant", "year"], observed=True)["n1_ref_nm3"].sum() / 1e9
    h3: dict[str, Any] = {
        "n1_national_bnm3_ch4": {v: tot[v].to_dict() for v in ("dec", "obs") if v in tot.index.get_level_values(0)},
        "n1_national_bnm3_ch4_observed_crops": {v: obs_only[v].to_dict() for v in ("dec", "obs")
                                                if v in obs_only.index.get_level_values(0)},
    }
    if {"dec", "obs"} <= set(tot.index.get_level_values(0)):
        r = (tot["obs"] / tot["dec"]).reindex(cfg.years)
        h3["n1_national_ratio_obs_dec"] = {**_summary(r), "summary_years_mean": r.reindex(years).mean()}
    tab4 = table_effects(t["source_vs_factor_effect"], years, crops)
    h3["source_vs_factor"] = {row.pop("crop"): row for row in tab4.drop(columns="years").to_dict("records")}
    rs = t["rank_shift"].set_index("year")
    h3["rank_shift"] = rs[["n_units", "spearman_rho", "top_decile_turnover"]].to_dict("index")
    h3["rank_shift_summary"] = {"spearman_rho": _summary(rs["spearman_rho"]),
                                "top_decile_turnover": _summary(rs["top_decile_turnover"]),
                                "summary_years_mean": rs.reindex(years)[["spearman_rho", "top_decile_turnover"]]
                                .mean().to_dict()}
    if "crop_area" in t:
        min_area = float(cfg.section("reconcile").get("min_area_ha", 0))
        h3["source_effect_by_decile"] = {
            "note": "C4 (droppable): municipal N1 over crops with an observed area, mean of the summary years",
            "all_cells": source_effect_by_decile(n1, years).drop(columns="years").to_dict("records"),
            f"cells_min_{min_area:g}ha": source_effect_by_decile(n1, years, t["crop_area"], min_area)
            .drop(columns="years").to_dict("records"),
        }
    out["h3"] = h3
    out["validation"] = validation_numbers(cfg)
    return out


def validation_numbers(cfg: Config) -> dict:
    """Invariant checks of the run: counts per stage, and the second-crop national comparison."""
    out: dict[str, Any] = {"checks": {}}
    for p in sorted((cfg.data_dir / "logs").glob("checks_*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        if data["stage"] in ("paper", "figures", "publish", "sanitation"):  # sanitation is not in the paper
            continue
        cs = data["checks"]
        out["checks"][data["stage"]] = {"total": len(cs), "passed": sum(c["passed"] for c in cs),
                                        "failed": sum(not c["passed"] and c["severity"] == "error" for c in cs),
                                        "warnings": sum(not c["passed"] and c["severity"] == "warn" for c in cs)}
        if data["stage"] == "second_crop":
            sc = {c["name"]: c["value"] for c in cs if c["name"].endswith("_vs_mapbiomas_series")
                  and isinstance(c["value"], (int, float))}
            years = [int(n.split("_")[-4]) for n in sc]
            in_period = {n: v for n, v, y in zip(sc, sc.values(), years, strict=True) if y in cfg.years}
            if in_period:
                out["second_crop_vs_mapbiomas_series_max_abs_rel"] = max(abs(v) for v in in_period.values())
                out["second_crop_vs_mapbiomas_series_n"] = len(in_period)
    return out


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------
def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    """A GitHub-style table (no tabulate dependency); floats to `digits` decimals, NaN as an em dash."""
    def fmt(v):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "—"
        if isinstance(v, (float, np.floating)):
            return f"{v:,.{digits}f}"
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            return f"{v:,}"
        return str(v).replace("|", "\\|")

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows]) + "\n"


def write_table(df: pd.DataFrame, folder: Path, name: str, title: str, note: str = "") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{name}.csv", index=False)
    md = f"**{title}**\n\n{to_markdown(df)}" + (f"\n{note}\n" if note else "")
    (folder / f"{name}.md").write_text(md, encoding="utf-8")
    log.info("  wrote %s (.csv, .md; %d rows)", folder / name, len(df))
    return folder / f"{name}.csv"


def build(cfg: Config, ck: CheckLog) -> Path:
    from bbp.publish import git_commit, sha256

    spec = cfg.section("paper")
    want = spec.get("factor_version")
    have = cfg.factors.get("source", {}).get("version")
    ck.add("paper_factor_version", want is None or want == have, f"profile wants {want!r}, conversion_factors.yaml "
           f"is {have!r}")
    if cfg.profile is None:
        ck.add("paper_profile", False, "no --profile given: using pipeline.yaml as is", severity="warn")
    years = summary_years(cfg)
    ck.add("paper_summary_years_in_period", bool(years), f"summary years {years}", years)
    t = load_tables(cfg, ck)
    ck.raise_if_failed()
    crops = list(cfg.crops)
    folder = out_dir(cfg)
    tables = folder / "tables"

    t1 = table_sources(cfg)
    absent = t1.loc[t1["period"].str.startswith(("not built", "outside")), "source"].tolist()
    ck.add("table1_every_source_covers_the_period", not absent, f"not built or outside the period: {absent}",
           absent, severity="warn")
    write_table(t1, tables, "table1_data_sources", "Table 1. Data sources.")

    t2 = table_agreement(t["agreement_by_scale"], years, crops)
    write_table(t2, tables, "table2_agreement_by_scale",
                f"Table 2. Agreement between observed (MapBiomas) and declared (IBGE PAM) area by scale, "
                f"mean over {_years_label(years)}.",
                "ratio_total = Σ observed / Σ declared; shares are of the units scored (either source ≥ the "
                "minimum area); rel_abs_error = Σ|obs − dec| / Σ dec.")
    # cross-table invariant: the national ratio in Table 2 is the one in unit_ratios_nation
    nat2 = t2[t2["scale"] == "nation"].set_index("crop")["ratio_total"]
    nat = t["unit_ratios_nation"]
    natu = nat[nat["year"].isin(years)].groupby("crop")["ratio"].mean()
    diff = (nat2.astype(float) - natu.reindex(nat2.index.astype(str)).to_numpy()).abs().max()
    ck.add("table2_nation_matches_unit_ratios", bool(diff < 1e-9), f"max |diff| = {diff:.2e}", float(diff))

    if "h2_model_fit" in t and "h2_moran" in t:
        t3 = table_h2(t["h2_model_fit"], t["h2_moran"], t.get("h2_rf_importance"), crops)
        write_table(t3, tables, "table3_h2_models",
                    "Table 3. Why the sources diverge (H2): mixed model, random forest and Moran's I per crop.",
                    "Response: municipal ln(A_obs / A_dec). R² marginal/conditional after Nakagawa & Schielzeth; "
                    "rf_r2_grouped_cv: out-of-region R² (folds grouped by intermediate region); Moran's I of the "
                    "mixed-model residuals (queen contiguity, permutation p-value).")
        if "h2_mixed_effects" in t:
            write_table(table_coefficients(t["h2_mixed_effects"], crops), tables, "table3b_h2_coefficients",
                        "Table 3b. Standardised fixed effects (change in log-ratio per SD of the predictor).")
        ck.add("table3_every_area_crop_modelled", set(t3["crop"]) >= set(t["agreement_by_scale"]["crop"]),
               f"modelled: {sorted(t3['crop'])}", severity="warn")
    else:
        ck.add("table3_h2_models", False, "H2 tables absent: run `bbp run analyse`", severity="warn")

    t4 = table_effects(t["source_vs_factor_effect"], years, crops)
    write_table(t4, tables, "table4_source_vs_factor",
                f"Table 4. Effect of the area source on municipal N1 compared with the conversion-factor "
                f"envelope, mean over {_years_label(years)}.",
                "Effects as |ln| ratios: source = |ln(N1_obs / N1_dec)| over municipalities with both; factor "
                "envelope = ln(N1_max / N1_min), all factors jointly; OAT = one factor at a time.")

    min_area = float(cfg.section("reconcile").get("min_area_ha", 0))
    t5 = pd.concat([source_effect_by_decile(t["n1_municipal"], years),
                    source_effect_by_decile(t["n1_municipal"], years, t["crop_area"], min_area)], ignore_index=True)
    write_table(t5, tables, "table5_source_effect_by_decile",
                f"Table 5 (C4, droppable). Area-source effect on municipal N1 by decile of N1_dec, mean over "
                f"{_years_label(years)}.",
                "Crops with an observed area only; decile 10 = largest municipal potential. Rows with "
                f"min_area_ha = {min_area:g} keep only crop cells where either source reports at least that area.")
    ck.add("table5_deciles_complete", len(t5) == 20 and abs(t5.groupby("min_area_ha")["share_of_national_n1"]
                                                           .sum() - 1).max() < 1e-9,
           f"{len(t5)} rows; shares add to 1 per variant")

    nums = numbers(cfg, t, years)
    nums = {"meta": {"generated": dt.datetime.now().isoformat(timespec="seconds"), "bbp_version": bbp.__version__,
                     "git_commit": git_commit(cfg.repo_root), "profile": cfg.profile,
                     "factor_version": have,
                     "units": {"area": "ha (Mha where named)", "n1": "billion Nm³ CH4 per year (methane, not biogas)",
                               "ratio": "observed / declared", "log": "natural log"},
                     "inputs_sha256": {f"processed/{n}.parquet": sha256(cfg.data_dir / "processed" / f"{n}.parquet")
                                       for n in t}},
            **nums}
    nums = _num(nums)
    text = json.dumps(nums, indent=2, ensure_ascii=False, allow_nan=False)
    path = folder / "numbers.json"
    path.write_text(text, encoding="utf-8")
    ck.add("numbers_json_valid", json.loads(text) == nums)
    log.info("  wrote %s (%d bytes)", path, len(text))
    return path
