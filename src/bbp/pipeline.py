"""Stage orchestration. Each stage reads its inputs, writes Parquet, and records invariant checks."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bbp.checks import CheckLog
from bbp.config import Config

log = logging.getLogger("bbp")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _is_fresh(out: Path, inputs: list[Path]) -> bool:
    if not out.exists():
        return False
    newest = max((p.stat().st_mtime for p in inputs if p.exists()), default=0)
    return out.stat().st_mtime >= newest


def _read(cfg: Config, *parts: str) -> pd.DataFrame:
    path = cfg.data_dir.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run the stage that produces it first")
    return pd.read_parquet(path)


def _write(df: pd.DataFrame, cfg: Config, *parts: str) -> Path:
    path = cfg.out(*parts)
    df.to_parquet(path, index=False)
    log.info("  wrote %s (%s rows)", path.relative_to(cfg.repo_root) if path.is_relative_to(cfg.repo_root) else path, f"{len(df):,}")
    return path


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
def stage_ingest(cfg: Config, ck: CheckLog, force: bool = False) -> None:
    from bbp.ingest import mapbiomas, municipalities, sidra

    non_mun = [int(x) for x in cfg.changes.get("non_municipal_units", [])]

    # ---- municipal mesh -----------------------------------------------------
    shp = cfg.input_path("municipalities_shp")
    out_attr, out_geom = cfg.out("interim", "municipalities.parquet"), cfg.out("interim", "municipalities_geom.parquet")
    if force or not (_is_fresh(out_attr, [shp]) and out_geom.exists()):
        log.info("Ingest: municipal mesh %s", shp.name)
        gdf = municipalities.read_mesh(shp, non_mun)
        gdf.to_parquet(out_geom, index=False)  # GeoParquet
        _write(municipalities.attributes(gdf), cfg, "interim", "municipalities.parquet")
    mesh = pd.read_parquet(out_attr)
    n_mun = int(mesh["is_municipality"].sum())
    ck.add("mesh_unique_geocodes", mesh["geocode"].is_unique)
    ck.add("mesh_municipality_count", 5565 <= n_mun <= 5575, f"{n_mun} municipalities", n_mun)
    mesh_codes = set(mesh["geocode"])

    # ---- MapBiomas coverage ---------------------------------------------------
    xlsx = cfg.input_path("mapbiomas_municipal_xlsx")
    out_mb = cfg.out("interim", "mapbiomas_coverage_municipal.parquet")
    if force or not _is_fresh(out_mb, [xlsx]):
        mb = mapbiomas.read_coverage_xlsx(xlsx, cfg.section("mapbiomas")["sheet"], cfg.years)
        _write(mb, cfg, "interim", "mapbiomas_coverage_municipal.parquet")
    mb = pd.read_parquet(out_mb)
    legend = mapbiomas.read_legend(cfg.input_path("mapbiomas_legend_csv"))
    legend.to_csv(cfg.out("interim", "mapbiomas_legend.csv"), index=False)
    in_data = mb.drop_duplicates("class_id").set_index("class_id")["class_label"].astype(str).to_dict()
    pd.Series(in_data, name="class_label").rename_axis("class_id").sort_index().to_csv(cfg.out("interim", "mapbiomas_classes_in_data.csv"))
    unknown = {int(k): v for k, v in in_data.items() if k not in set(legend["class_id"]) and k != 0}
    ck.add("mapbiomas_classes_in_legend_csv", not unknown, f"classes in the data but not in the legend CSV: {unknown}",
           severity="warn")
    # the crop classes must still mean what crops.yaml assumes (class codes can change between collections)
    for cid, expected in cfg.section("mapbiomas").get("expected_labels", {}).items():
        label = in_data.get(int(cid), "<absent>")
        ck.add(f"mapbiomas_class_{cid}_is_{expected}", expected.lower() in label.lower(), f"class {cid} = {label!r}")
    mb_codes = set(mb["geocode"].unique())
    ck.add("mapbiomas_geocodes_in_mesh", mb_codes <= mesh_codes, f"{len(mb_codes - mesh_codes)} codes not in mesh",
           sorted(mb_codes - mesh_codes)[:20])
    missing = sorted(set(mesh.loc[mesh["is_municipality"], "geocode"]) - mb_codes)
    ck.add("mesh_municipalities_in_mapbiomas", not missing, f"{len(missing)} missing", missing[:20], severity="warn")
    tot = mb.groupby(["geocode", "year"])["area_ha"].sum().unstack()
    drift = ((tot.max(axis=1) - tot.min(axis=1)) / tot.mean(axis=1)).max()
    ck.add("mapbiomas_municipal_area_constant_over_years", drift < 1e-3, f"max relative drift {drift:.2e}", float(drift))
    ibge = mesh.set_index("geocode")["area_km2_ibge"] * 100
    rel = (tot.mean(axis=1) / ibge.reindex(tot.index) - 1).abs()
    ck.add("mapbiomas_area_vs_ibge_area", rel.median() < 0.01,
           f"median |MB/IBGE − 1| = {rel.median():.4f}, max = {rel.max():.3f}", float(rel.median()), severity="warn")
    years_present = sorted(mb["year"].unique().tolist())
    ck.add("mapbiomas_years", years_present == cfg.years, f"{years_present[0]}–{years_present[-1]}", years_present)

    # ---- MapBiomas irrigation systems (H2 predictor) ----------------------------
    if "mapbiomas_irrigation_xlsx" in cfg.raw["inputs"]:
        irr_xlsx = cfg.input_path("mapbiomas_irrigation_xlsx")
        out_irr = cfg.out("interim", "mapbiomas_irrigation_municipal.parquet")
        if force or not _is_fresh(out_irr, [irr_xlsx]):
            _write(mapbiomas.read_irrigation_xlsx(irr_xlsx, cfg.years), cfg, "interim",
                   "mapbiomas_irrigation_municipal.parquet")
        irr = pd.read_parquet(out_irr)
        collections = sorted(irr["collection"].unique().tolist())
        ck.add("irrigation_same_collection_as_coverage", collections == [int(cfg.section("mapbiomas")["collection"])],
               f"irrigation collection(s) {collections}", collections)
        outside = sorted(set(irr["geocode"]) - mesh_codes)
        ck.add("irrigation_geocodes_in_mesh", not outside, f"{len(outside)} not in mesh", outside[:20])
        ck.add("irrigation_non_negative", bool((irr["area_ha"] >= 0).all()))
        per_mun = irr.groupby(["geocode", "year"])["area_ha"].sum()
        mun_area = mb.groupby(["geocode", "year"])["area_ha"].sum()
        over = per_mun[per_mun > mun_area.reindex(per_mun.index) * 1.001]
        ck.add("irrigation_within_municipal_area", over.empty, f"{len(over)} municipality-years exceed the municipal area",
               [list(map(int, k)) for k in over.index[:20]])

    # ---- clear Landsat observations (H2 cloud proxy; docs/gee/landsat_valid_observations.js) --
    if "landsat_valid_obs_dir" in cfg.raw["inputs"]:
        obs_dir = cfg.input_path("landsat_valid_obs_dir")
        csvs = sorted(obs_dir.glob("landsat_valid_obs_*.csv")) if obs_dir.is_dir() else []
        if ck.add("landsat_valid_obs_files", bool(csvs), f"{len(csvs)} CSV in {obs_dir.name} (run the GEE script)",
                  len(csvs), severity="warn"):
            obs = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
            obs = obs.rename(columns={"CD_MUN": "geocode", "clear_obs_mean": "landsat_valid_obs"})
            obs["geocode"], obs["year"] = obs["geocode"].astype("int64"), obs["year"].astype("int64")
            outside = sorted(set(obs["geocode"]) - mesh_codes)
            ck.add("landsat_valid_obs_geocodes_in_mesh", not outside, f"{len(outside)} not in mesh", outside[:20])
            ck.add("landsat_valid_obs_unique", not obs.duplicated(["geocode", "year"]).any())
            ck.add("landsat_valid_obs_one_masking", obs["crop_mask"].nunique() == 1,
                   f"crop_mask values {sorted(obs['crop_mask'].astype(str).unique())}")
            _write(obs[["geocode", "year", "landsat_valid_obs", "pixels"]], cfg, "interim",
                   "landsat_valid_observations.parquet")

    # ---- SIDRA tables (PAM, PPM) ---------------------------------------------
    tables = cfg.sidra_tables()
    drop_vars = cfg.raw.get("sidra_drop_variables", [])
    for table, spec in tables.items():
        _ingest_sidra_table(cfg, ck, table, spec, drop_vars, mesh_codes, force)

    # PAM rows used by the area comparison: tables 5457 and 839 plus any table named in crops.yaml
    pam_tables = sorted({5457, 839} | {int(c["pam_table"]) for c in cfg.crops.values()})
    missing_tables = [t for t in pam_tables if t not in tables]
    ck.add("pam_tables_configured", not missing_tables, f"crops.yaml needs tables not in `sidra`: {missing_tables}")
    sources = [cfg.out("interim", "sidra", f"t{t}.parquet") for t in pam_tables if t in tables]
    out_pam = cfg.out("interim", "pam.parquet")
    if force or not _is_fresh(out_pam, sources):
        pam = sidra.tidy(pd.concat([pd.read_parquet(p) for p in sources], ignore_index=True), cfg.crops)
        _write(pam, cfg, "interim", "pam.parquet")
    pam = pd.read_parquet(out_pam)
    for crop, spec in cfg.crops.items():
        has = ((pam["crop"] == crop) & (pam["variable"] == "area_planted_ha")).any()
        ck.add(f"pam_has_{crop}", has, f"table {spec['pam_table']} product {spec['pam_product']!r}")
    _check_corn_consistency(pam, ck)
    log.info("PAM years: %s", sorted(pam["year"].unique().tolist()))

    # ---- second crop (produced by `bbp second-crop`) --------------------------
    sc = cfg.data_dir / "interim" / "second_crop_municipal.parquet"
    ck.add("second_crop_table_present", sc.exists(),
           "" if sc.exists() else "run `bbp second-crop` to compute it from the rasters", severity="warn")


def _ingest_sidra_table(cfg: Config, ck: CheckLog, table: int, spec: dict, drop_vars: list[str],
                        mesh_codes: set[int], force: bool) -> None:
    from bbp.ingest import sidra

    files = spec["files"]
    absent = [str(p) for p in files if not p.exists()]
    t = f"sidra_{table}"
    if not ck.add(f"{t}_files_exist", not absent, f"missing: {absent}"):
        return
    out = cfg.out("interim", "sidra", f"t{table}.parquet")
    meta_path = out.with_suffix(".meta.json")
    if force or not (_is_fresh(out, files) and meta_path.exists()):
        log.info("Ingest: SIDRA table %s (%s, %d files)", table, spec.get("survey", "?"), len(files))
        df, meta = sidra.read_table(table, files, spec.get("products"), spec.get("exclude_products", []), drop_vars)
        df.insert(1, "survey", spec.get("survey"))
        _write(df, cfg, "interim", "sidra", f"t{table}.parquet")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    df = pd.read_parquet(out)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    codes = set(df["geocode"].unique().tolist())
    ck.add(f"{t}_geocodes_in_mesh", codes <= mesh_codes, f"{len(codes - mesh_codes)} not in mesh", sorted(codes - mesh_codes)[:20])
    ck.add(f"{t}_non_negative", bool((df["value"].dropna() >= 0).all()))
    unparsed = int(df["flag"].astype(str).str.startswith("unparsed").sum())
    ck.add(f"{t}_all_values_parsed", unparsed == 0, f"{unparsed} unparsed cells")
    years = sorted(df["year"].unique().tolist())
    ck.add(f"{t}_years_contiguous", years == list(range(years[0], years[-1] + 1)), f"{years[0]}–{years[-1]}", years)
    wanted = set(spec.get("products") or [])
    absent_products = sorted(wanted - set(df["product"].astype(str).unique()))
    ck.add(f"{t}_has_configured_products", not absent_products, f"not found (renamed by IBGE?): {absent_products}")
    for prod in spec.get("exclude_products", []):
        ck.add(f"{t}_excludes_{prod}", not (df["product"] == prod).any())
    for sub in spec.get("subsets", []):
        bad = sidra.subset_violations(df, sub["total"], sub["parts"])
        cases = bad.reset_index()[["geocode", "year"]].astype(int).values.tolist()
        ck.add(f"{t}_{'+'.join(sub['parts'])}_within_{sub['total']}", bad.empty,
               f"{len(bad)} municipality-years where the parts exceed the total (kept as published)", cases[:50],
               severity=sub.get("severity", "error"))
    for prod in spec.get("expected_empty", []):
        filled = int(((df["product"] == prod) & (df["value"] > 0)).sum())
        ck.add(f"{t}_{prod}_is_empty", filled == 0,
               f"{filled} non-zero cells: IBGE now fills this column; revisit how its total is built", severity="warn")
    ck.add(f"{t}_overlap", True, f"{meta['overlapping_cells_replaced']} cells covered by more than one file "
           f"(the later file was kept)", meta["overlapping_cells_replaced"])
    ck.add(f"{t}_{years[-1]}_is_preliminary", False,
           f"IBGE publishes the newest year ({years[-1]}) as preliminary", years[-1], severity="warn")
    stale = sidra.stale_preliminary(df, meta["file_years"])
    ck.add(f"{t}_{years[-1] - 1}_not_from_a_preliminary_export", not stale,
           f"{years[-1] - 1} was the newest (preliminary) year in {sorted(stale)}; IBGE may have revised it since — "
           f"re-export {years[-1] - 1}", stale, severity="warn")


def _check_corn_consistency(pam: pd.DataFrame, ck: CheckLog) -> None:
    """Table 839 (1st + 2nd season) must add up to its own total and to table 5457 corn."""
    a = pam[(pam["variable"] == "area_planted_ha")]
    t839 = a[a["table"] == 839].pivot_table(index=["geocode", "year"], columns="product", values="value", aggfunc="sum")
    if {"Total", "Milho (em grão) - 1ª safra", "Milho (em grão) - 2ª safra"} <= set(t839.columns):
        parts = t839["Milho (em grão) - 1ª safra"].fillna(0) + t839["Milho (em grão) - 2ª safra"].fillna(0)
        ok = (t839["Total"].isna()) | (np.abs(parts - t839["Total"]) <= 1)
        ck.add("pam839_seasons_sum_to_total", ok.mean() > 0.999, f"{(~ok).sum()} mismatching rows", severity="warn")
        c5457 = a[(a["table"] == 5457) & (a["product"] == "Milho (em grão)")].set_index(["geocode", "year"])["value"]
        j = pd.concat([t839["Total"], c5457.rename("t5457")], axis=1, join="inner").dropna()
        if len(j):
            ok2 = np.abs(j["Total"] - j["t5457"]) <= 1
            ck.add("pam839_total_equals_5457_corn", ok2.mean() > 0.99, f"{(~ok2).sum()} of {len(j)} rows differ", severity="warn")


# ---------------------------------------------------------------------------
# sanitation (SNIS to 2022, SINISA from 2023) — urban waste and sewage
# ---------------------------------------------------------------------------
def _cached(cfg: Config, name: str, sources: list[Path], force: bool, build) -> tuple[pd.DataFrame, dict]:
    """interim/sanitation/<name>.parquet + .meta.json, rebuilt when a source is newer."""
    out = cfg.out("interim", "sanitation", f"{name}.parquet")
    meta_path = out.with_suffix(".meta.json")
    if force or not (_is_fresh(out, sources) and meta_path.exists()):
        df, meta = build()
        _write(df, cfg, "interim", "sanitation", f"{name}.parquet")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return pd.read_parquet(out), json.loads(meta_path.read_text(encoding="utf-8"))


def _published_check(ck: CheckLog, name: str, ours: dict, published: dict, tol: float) -> None:
    """Municipal sums vs the survey's own totals, for every unit and field both sides have."""
    diffs = []
    for unit, fields in published.items():
        for f, pub in fields.items():
            got = ours.get(unit, {}).get(f)
            if pub and got is not None:
                diffs.append((abs(got - pub) / abs(pub), unit, f, got, pub))
    if not diffs:
        ck.add(name, False, "no comparable totals")
        return
    worst = max(diffs)
    bad = [d for d in diffs if d[0] > tol]
    ck.add(name, not bad, f"{len(diffs)} totals compared, {len(bad)} off by more than {tol:.1%}; worst "
           f"{worst[1]} {worst[2]}: ours {worst[3]:,.2f} vs published {worst[4]:,.2f} ({worst[0]:.3%})",
           [list(d[1:3]) + [round(d[0], 6)] for d in sorted(bad, reverse=True)[:20]])


def stage_sanitation(cfg: Config, ck: CheckLog, force: bool = False) -> None:
    from bbp.ingest import sanitation as san

    spec = cfg.section("sanitation")
    if not spec or not cfg.sanitation:
        ck.add("sanitation_configured", False, "no `sanitation` in pipeline.yaml or no config/sanitation.yaml",
               severity="warn")
        return
    xw = san.crosswalk(cfg.sanitation["variables"])
    identities = cfg.sanitation.get("identities", {})
    tol = float(spec.get("published_tolerance", 0.01))
    mesh = _read(cfg, "interim", "municipalities.parquet")
    mesh = mesh[mesh["is_municipality"]]
    uf_of = mesh.set_index("geocode")["uf"]
    region_of = mesh.set_index("geocode")["region"]
    mesh_codes = set(mesh["geocode"])
    by_code6 = san.code6_index(mesh_codes)
    ck.add("mesh_code6_prefixes_unique", True, f"{len(by_code6)} municipalities")
    long = []

    def common_checks(t: str, df: pd.DataFrame, meta: dict) -> None:
        ck.add(f"{t}_fields_present", not meta["missing_fields"], f"absent: {meta['missing_fields']}",
               meta["missing_fields"], severity="warn")
        ck.add(f"{t}_all_values_parsed", meta["n_unparsed"] == 0, f"{meta['n_unparsed']} unparsed cells",
               meta["unparsed"])
        neg = df[df["value"] < 0]
        ck.add(f"{t}_non_negative", neg.empty, f"{len(neg)} negative values (kept as published)",
               neg[["geocode", "field", "value"]].head(20).values.tolist(), severity="warn")

    # ---- SNIS -----------------------------------------------------------------
    snis_ids = [f for i in identities.get("snis", []) for f in (i["total"], *i["parts"])]
    snis_fields = san.fields_for(xw, "SNIS", "snis", [*snis_ids, "POP_TOT"])
    for year, path in cfg.snis_files().items():
        t = f"snis_{year}"
        if not ck.add(f"{t}_file_exists", path is not None, f"no file matching {spec['snis']['pattern']}"):
            continue
        df, meta = _cached(cfg, t, [path], force, lambda p=path, y=year: san.read_snis(p, y, snis_fields))
        df["geocode"] = df["code6"].map(by_code6)
        unresolved = sorted(df.loc[df["geocode"].isna(), "code6"].unique().tolist())
        ck.add(f"{t}_codes_resolve_to_mesh", not unresolved, f"{len(unresolved)} 6-digit codes not in the mesh",
               unresolved[:20])
        df = df.dropna(subset=["geocode"]).astype({"geocode": "int64"})
        dup = int(df.duplicated(["geocode", "field"]).sum())
        ck.add(f"{t}_one_row_per_municipality", dup == 0, f"{dup} duplicated cells")
        common_checks(t, df, meta)
        ck.add(f"{t}_one_total_row", meta["summary_rows"] == 1, f"{meta['summary_rows']} 'TOTAL da AMOSTRA' rows")
        ours = san.sums_by_uf(df, uf_of, list(meta["totals"]))
        _published_check(ck, f"{t}_matches_its_total_row", ours,
                         {"BR": {f: v for f, v in meta["totals"].items() if f in ("POP_TOT", "CO111", "CO119", "ES006")}},
                         tol)
        for ident in identities.get("snis", []):
            bad = san.identity_breaks(df, ident["total"], ident["parts"])
            ck.add(f"{t}_{ident['total']}_equals_{'+'.join(ident['parts'])}", bad.empty,
                   f"{len(bad)} municipalities where SNIS's own total differs from its parts (kept as published)",
                   bad.reset_index()[["geocode"]].values.ravel().tolist()[:20], severity="warn")
        long.append(df.assign(survey="SNIS", module="snis")[san.COLUMNS])

    # ---- SINISA ---------------------------------------------------------------
    sinisa_ids = [f for i in identities.get("sinisa", []) for f in (i["total"], *i["parts"])]
    for year, ys in cfg.sinisa_years().items():
        for module in ("waste", "sewage"):
            t = f"sinisa_{year}_{module}"
            src = ys.get(module)
            if not ck.add(f"{t}_file_exists", src is not None and src["file"].exists(),
                          f"missing: {src and src['file']}"):
                continue
            fields = san.fields_for(xw, "SINISA", module, sinisa_ids if module == "waste" else [])
            df, meta = _cached(cfg, t, [src["file"]], force,
                               lambda s=src, y=year, f=fields: san.read_sinisa(s["file"], s["sheet"], y, f))
            outside = sorted(set(df["geocode"]) - mesh_codes)
            ck.add(f"{t}_geocodes_in_mesh", not outside, f"{len(outside)} not in the mesh", outside[:20])
            ck.add(f"{t}_one_row_per_municipality", meta["duplicates"] == 0,
                   f"{meta['rows']} rows, {meta['duplicates']} duplicated geocodes")
            common_checks(t, df, meta)
            if module == "waste":
                expected = ys.get("waste_respondents")
                ck.add(f"{t}_respondents", expected is None or meta["respondents"] == expected,
                       f"{meta['respondents']} municipalities answered the module; SINISA reports {expected}",
                       meta["respondents"])
                for ident in identities.get("sinisa", []):
                    bad = san.identity_breaks(df, ident["total"], ident["parts"])
                    ck.add(f"{t}_{ident['total']}_equals_{'+'.join(ident['parts'])}", bad.empty,
                           f"{len(bad)} municipalities where SINISA's own total differs from its parts "
                           "(kept as published)", bad.reset_index()[["geocode"]].values.ravel().tolist()[:20],
                           severity="warn")
            pub = ys.get(f"{module}_published")
            if pub is not None:
                if ck.add(f"{t}_published_file_exists", pub["file"].exists(), f"missing: {pub['file']}"):
                    published = san.read_sinisa_published(pub["file"], pub["sheet"], fields)
                    _published_check(ck, f"{t}_matches_published_consolidation",
                                     san.sums_by_uf(df, uf_of, fields, region_of), published, tol)
            long.append(df.assign(survey="SINISA", module=module)[san.COLUMNS])

    if not long:
        ck.add("sanitation_has_data", False, "no SNIS or SINISA year could be read")
        return
    h = san.harmonise(pd.concat(long, ignore_index=True), xw)
    dup = int(h.duplicated(["geocode", "year", "variable"]).sum())
    ck.add("sanitation_one_value_per_municipality_year_variable", dup == 0, f"{dup} duplicates")
    years = sorted(h["year"].unique().tolist())
    ck.add("sanitation_years_contiguous", years == list(range(years[0], years[-1] + 1)),
           f"{years[0]}–{years[-1]}", years)
    _write(h, cfg, "processed", "sanitation.parquet")
    cont = san.continuity(h)
    _write(cont, cfg, "processed", "sanitation_continuity.parquet")
    lo, hi = spec.get("continuity_band", [0.8, 1.25])
    for r in cont.itertuples():
        ck.add(f"sanitation_continuity_{r.variable}", lo <= r.median_ratio <= hi,
               f"{r.n_both} municipalities report both {r.snis_year} (SNIS) and {r.sinisa_year} (SINISA): "
               f"median ratio {r.median_ratio:.3f} (p10 {r.p10_ratio:.3f}, p90 {r.p90_ratio:.3f}), "
               f"ratio of totals {r.ratio_of_totals:.3f}, {r.share_within_25pct:.0%} within ±25 %",
               round(float(r.median_ratio), 4), severity="warn")
    summary = h.groupby(["variable", "year"], observed=True).size().unstack("year")
    log.info("Sanitation: municipalities reporting each variable per year\n%s", summary.to_string())


# ---------------------------------------------------------------------------
# border share (raster) — separate command, like second crop: it runs for hours
# ---------------------------------------------------------------------------
def stage_border_share(cfg: Config, ck: CheckLog, years: list[int] | None = None) -> None:
    import geopandas as gpd

    from bbp.ingest import border_share as bs
    from bbp.ingest import second_crop as sc

    spec = cfg.section("border_share")
    sources = bs.find_year_sources(cfg.input_path("coverage_tif_dir"), spec["tif_pattern"], years or spec["years"])
    ck.add("coverage_rasters_found", bool(sources), f"{len(sources)} years: {sorted(sources)}", sorted(sources))
    if not sources:
        return
    # a year exported in tiles (Earth Engine) is read through a VRT mosaic built in data/interim
    rasters = {}
    for year, src in sources.items():
        if isinstance(src, list):
            vrt = cfg.out("interim", "coverage_vrt", f"{Path(spec['tif_pattern'].format(year=year)).stem}.vrt")
            try:
                meta = bs.build_vrt(src, vrt)
            except ValueError as exc:
                ck.add(f"coverage_{year}_tiles_form_one_mosaic", False, str(exc))
                continue
            ck.add(f"coverage_{year}_tiles_form_one_mosaic", True,
                   f"{meta['tiles']} tiles -> {meta['width']:,} x {meta['height']:,} px")
            rasters[year] = vrt
        else:
            rasters[year] = src
    classes = {int(c): crop for crop, s in cfg.crops.items() for c in (s.get("mapbiomas_class") or [])}
    coverage = pd.read_parquet(cfg.data_dir / "interim" / "mapbiomas_coverage_municipal.parquet")
    mesh = None
    tol = float(spec.get("collection_tolerance", 0.005))
    out = cfg.out("interim", "border_share_municipal.parquet")
    kept = [pd.read_parquet(out)] if out.exists() else []
    for year, path in sorted(rasters.items()):
        key = bs.grid_key(path)
        grid = cfg.out("interim", f"municipality_grid_{key}.tif")
        if not grid.exists():
            if mesh is None:
                mesh = gpd.read_parquet(cfg.out("interim", "municipalities_geom.parquet"))
                mesh = mesh[mesh["is_municipality"]]
            sc.build_label_grid(mesh, path, grid, window=spec.get("window", 2048))
        table = bs.border_area(grid, {year: path}, classes, spec.get("distances_m", [1000, 2000]),
                               window=spec.get("window", 2048),
                               checkpoint=cfg.out("interim", f"border_share_checkpoint_{year}.npz"))
        # guard: the rasters must be the collection the areas come from (see collection_agreement)
        agree = bs.collection_agreement(table, coverage, classes)
        med = float(agree["rel"].abs().median()) if len(agree) else float("nan")
        ok = ck.add(f"border_share_{year}_raster_matches_collection_{cfg.section('mapbiomas')['collection']}",
                    med <= tol, f"median |raster/statistics − 1| = {med:.3%} over {len(agree)} municipality-crops "
                    f"(≤ {tol:.1%} expected for the same collection; 1–3 % for another)", med)
        if not ok:
            _write(table, cfg, "interim", f"border_share_rejected_{year}.parquet")
            continue
        kept = [k[k["year"] != year] for k in kept] + [table]
        share = table[table["crop"] == bs.ALL]
        log.info("Border share %d: median share of crop area within 1 km %.3f, 2 km %.3f", year,
                 share["border_share_1km"].median(), share["border_share_2km"].median())
    if kept:
        _write(pd.concat(kept, ignore_index=True).sort_values(["year", "crop", "geocode"]), cfg,
               "interim", "border_share_municipal.parquet")


# ---------------------------------------------------------------------------
# second crop (raster) — separate command because it runs for hours
# ---------------------------------------------------------------------------
def stage_second_crop(cfg: Config, ck: CheckLog, years: list[int] | None = None, grid_only: bool = False) -> None:
    import geopandas as gpd

    from bbp.ingest import second_crop as sc

    spec = cfg.section("second_crop")
    tif_dir = cfg.input_path("second_crop_tif_dir")
    rasters = sc.find_rasters(tif_dir, spec["tif_pattern"], years or cfg.years)
    ck.add("second_crop_rasters_found", bool(rasters), f"{len(rasters)} rasters: {sorted(rasters)}", sorted(rasters))
    if not rasters:
        return
    grid = cfg.out("interim", "municipality_grid.tif")
    if not grid.exists():
        mesh = gpd.read_parquet(cfg.out("interim", "municipalities_geom.parquet"))
        sc.build_label_grid(mesh, next(iter(rasters.values())), grid, window=spec.get("window", 2048))
    if grid_only:
        return

    classes = {int(k): v for k, v in spec["classes"].items()}
    table, diag = sc.zonal_area(grid, rasters, classes, window=spec.get("window", 2048),
                                checkpoint=cfg.out("interim", "second_crop_checkpoint.npz"))
    out = cfg.out("interim", "second_crop_municipal.parquet")
    if out.exists() and years:  # partial rerun: replace only the recomputed years
        old = pd.read_parquet(out)
        table = pd.concat([old[~old["year"].isin(table["year"].unique())], table], ignore_index=True)
    _write(table.sort_values(["year", "class", "geocode"]), cfg, "interim", "second_crop_municipal.parquet")
    sc.save_diagnostics(diag, cfg.out("logs", "second_crop_diagnostics.json"))

    # invariant: national totals vs the series published by MapBiomas
    nat_csv = cfg.input_path("second_crop_national_csv")
    if nat_csv.exists():
        nat = sc.read_national_series(nat_csv)
        tol = float(spec.get("national_tolerance", 0.02))
        mine = table.groupby(["year", "class"])["area_ha"].sum()
        for cls, label in spec.get("national_series_labels", {}).items():
            ref = nat[nat["label"] == label].set_index("year")["area_ha"]
            for y in sorted(set(ref.index) & set(table["year"])):
                total = mine.get((y, cls), 0.0) + diag["unassigned_ha"][int(y)].get(cls, 0.0)
                rel = total / ref[y] - 1
                ck.add(f"second_crop_{cls}_{y}_vs_mapbiomas_series", abs(rel) <= tol,
                       f"pipeline {total / 1e6:.3f} Mha vs MapBiomas {ref[y] / 1e6:.3f} Mha ({rel:+.2%})", rel)


# ---------------------------------------------------------------------------
# processing plants (one national register; bbp reach reads it)
# ---------------------------------------------------------------------------
def stage_plants(cfg: Config, ck: CheckLog) -> None:
    from bbp.plants import build_plants

    if not cfg.section("plants").get("sources"):
        ck.add("plants_configured", False, "no plants.sources in pipeline.yaml", severity="warn")
        return
    build_plants(cfg, ck)


# ---------------------------------------------------------------------------
# harmonise
# ---------------------------------------------------------------------------
def stage_harmonise(cfg: Config, ck: CheckLog) -> None:
    from bbp import harmonise as hm

    mesh = _read(cfg, "interim", "municipalities.parquet")
    pam = _read(cfg, "interim", "pam.parquet")
    changes = cfg.changes.get("changes", [])

    problems = hm.check_changes_against_mesh(changes, mesh)
    ck.add("municipality_changes_match_mesh", not problems, "; ".join(problems))

    first = hm.first_record_year(pam)
    gaps = hm.data_gaps(pam)
    _write(gaps, cfg, "processed", "pam_gaps.parquet")
    listed = {int(c["code"]) for c in changes}
    gap_mun = sorted(set(gaps["geocode"]) - listed)
    ck.add("pam_whole_municipality_gaps", not gap_mun,
           f"{len(gap_mun)} municipalities have years with no PAM statistics at all ('...'); "
           f"see processed/pam_gaps.parquet. Check none is an unlisted new municipality.", gap_mun[:50], severity="warn")
    first_year = {int(c["code"]): int(first.get(int(c["code"]), c.get("first_pam_year") or c["installed"])) for c in changes}
    for c in changes:
        log.info("  %s (%s): first PAM record %s", c["name"], c["code"], first_year[int(c["code"])])

    mun = mesh[mesh["is_municipality"]].copy()
    amc = hm.build_amc(mun["geocode"], changes)
    attrs = mun.merge(amc, on="geocode", how="left")
    attrs["first_pam_year"] = attrs["geocode"].map(first).astype("Int64")
    attrs["created_after_start"] = attrs["geocode"].isin(listed)
    _write(attrs, cfg, "processed", "municipality_attributes.parquet")
    mism = hm.mesh_mismatch_flags(cfg.years, changes, first_year)
    _write(mism, cfg, "processed", "mesh_mismatch.parquet")
    ck.add("amc_partition", attrs["amc_id"].notna().all() and attrs["geocode"].is_unique)


# ---------------------------------------------------------------------------
# area
# ---------------------------------------------------------------------------
def stage_area(cfg: Config, ck: CheckLog) -> None:
    from bbp.area import crop_area_table

    attrs = _read(cfg, "processed", "municipality_attributes.parquet")
    mb = _read(cfg, "interim", "mapbiomas_coverage_municipal.parquet")
    pam = _read(cfg, "interim", "pam.parquet")
    sc_path = cfg.data_dir / "interim" / "second_crop_municipal.parquet"
    sc = pd.read_parquet(sc_path) if sc_path.exists() else None
    ck.add("area_second_crop_available", sc is not None, "corn_2nd observed area will be empty", severity="warn")

    area = crop_area_table(attrs["geocode"], cfg.years, cfg.crops, mb, pam, sc)
    _write(area, cfg, "processed", "crop_area.parquet")

    mun_area = mb.groupby(["geocode", "year"])["area_ha"].sum()
    obs_tot = area.groupby(["geocode", "year"])["a_obs_ha"].sum()
    j = pd.concat([obs_tot, mun_area], axis=1, join="inner")
    ratio = (j.iloc[:, 0] / j.iloc[:, 1]).max()
    # second-season corn sits on soybean fields, so the sum of crops can exceed the mapped crop area,
    # but never twice the municipality
    ck.add("observed_crop_area_within_municipality", ratio <= 2.0, f"max sum(crops)/municipal area = {ratio:.3f}", float(ratio))
    ck.add("area_non_negative", bool((area[["a_obs_ha", "a_dec_ha"]].fillna(0) >= 0).all().all()))
    summary = (area.groupby(["crop", "year"], observed=True)[["a_obs_ha", "a_dec_ha"]].sum(min_count=1) / 1e6).round(3)
    log.info("National area (Mha):\n%s", summary.unstack("crop").to_string())


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------
def stage_reconcile(cfg: Config, ck: CheckLog) -> None:
    from bbp import reconcile as rc

    area = _read(cfg, "processed", "crop_area.parquet")
    attrs = _read(cfg, "processed", "municipality_attributes.parquet")
    mism = _read(cfg, "processed", "mesh_mismatch.parquet")
    mb = _read(cfg, "interim", "mapbiomas_coverage_municipal.parquet")
    spec = cfg.section("reconcile")
    scales = spec.get("scales", ["municipality", "rgint", "uf", "nation"])
    if "amc" not in scales:
        scales = [scales[0], "amc", *scales[1:]] if scales[0] == "municipality" else ["amc", *scales]

    metrics, units = rc.agreement_by_scale(area, attrs, scales, mism, spec.get("tolerances", [0.1, 0.25]),
                                           float(spec.get("min_area_ha", 0)))
    _write(metrics, cfg, "processed", "agreement_by_scale.parquet")
    for scale, tab in units.items():
        _write(tab, cfg, "processed", f"unit_ratios_{scale}.parquet")

    # invariant: every scale built from whole AMCs adds up to the same national total
    totals = {s: units[s].groupby(["crop", "year"], observed=True)[["a_obs_ha", "a_dec_ha"]].sum() for s in units if s != "municipality"}
    ref = totals.get("nation")
    if ref is not None:
        for s, t in totals.items():
            diff = (t - ref.reindex(t.index)).abs().max().max()
            ck.add(f"scale_{s}_sums_to_nation", diff < 1.0, f"max |diff| = {diff:.3f} ha", float(diff))

    def optional(*parts: str) -> pd.DataFrame | None:
        p = cfg.data_dir.joinpath(*parts)
        ck.add(f"predictor_input_{p.stem}", p.exists(), "" if p.exists() else f"{p.name} not built yet", severity="warn")
        return pd.read_parquet(p) if p.exists() else None

    pred = rc.predictors(area, attrs, mb, irrigation=optional("interim", "mapbiomas_irrigation_municipal.parquet"),
                         border=optional("interim", "border_share_municipal.parquet"),
                         observations=optional("interim", "landsat_valid_observations.parquet"))
    _write(pred, cfg, "processed", "predictors.parquet")
    nat = metrics[metrics["scale"] == "nation"].pivot_table(index="year", columns="crop", values="ratio_total", observed=True)
    log.info("National ratio observed/declared:\n%s", nat.round(3).to_string())


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------
def stage_inventory(cfg: Config, ck: CheckLog) -> None:
    from bbp import inventory as inv

    area = _read(cfg, "processed", "crop_area.parquet")
    attrs = _read(cfg, "processed", "municipality_attributes.parquet")

    for fs, spec in cfg.factors["feedstocks"].items():
        ordered = all(spec[f]["min"] <= spec[f]["ref"] <= spec[f]["max"] and spec[f]["min"] > 0 for f in inv.FACTORS)
        ck.add(f"factors_{fs}_ordered_positive", ordered)
    missing = [s["feedstock"] for s in cfg.crops.values() if s.get("feedstock") and s["feedstock"] not in cfg.factors["feedstocks"]]
    ck.add("factors_cover_all_feedstocks", not missing, f"missing: {missing}")

    n1 = inv.n1_table(area, attrs, cfg.crops, cfg.factors, cfg.section("inventory").get("yield_fallback", ["rgint", "uf", "nation"]))
    _write(n1, cfg, "processed", "n1_municipal.parquet")

    # identity: where both sources give the same area, both variants give the same N1
    a = area[(area["a_obs_ha"] == area["a_dec_ha"]) & (area["a_dec_ha"] > 0) & (area["prod_t"] > 0)][["geocode", "year", "crop"]]
    if len(a):
        w = n1.pivot_table(index=["geocode", "year", "crop"], columns="variant", values="n1_ref_nm3", observed=True)
        w = w.join(a.assign(crop=a["crop"].astype(str)).set_index(["geocode", "year", "crop"]), how="inner")
        ok = np.isclose(w["dec"], w["obs"], rtol=1e-9).all()
        ck.add("n1_identity_equal_areas", bool(ok), f"{len(w)} cells with A_obs == A_dec")

    _write(inv.source_vs_factor_effect(n1), cfg, "processed", "source_vs_factor_effect.parquet")
    _write(inv.rank_shift(n1), cfg, "processed", "rank_shift.parquet")
    fb = n1[(n1["variant"] == "obs")]["yield_source"].value_counts(dropna=False).to_dict()
    log.info("Yield source for the observed variant: %s", fb)


# ---------------------------------------------------------------------------
# analyse (H2 models)
# ---------------------------------------------------------------------------
def stage_analyse(cfg: Config, ck: CheckLog) -> None:
    import geopandas as gpd

    from bbp import analyse as an

    units = _read(cfg, "processed", "unit_ratios_municipality.parquet")
    pred = _read(cfg, "processed", "predictors.parquet")
    border_p = cfg.data_dir / "interim" / "border_share_municipal.parquet"
    border = pd.read_parquet(border_p) if border_p.exists() else None
    spec = cfg.section("analyse")
    years = spec.get("years") or cfg.section("reconcile").get("reference_years", cfg.years[-5:])
    crops = spec.get("crops") or [c for c, s in cfg.crops.items() if s.get("mapbiomas_class") or s.get("mapbiomas_second_crop")]
    min_n = int(spec.get("min_observations", 200))
    for name in ("irrigated_share", "border_share_1km", "landsat_valid_obs"):
        present = name in pred or (name == "border_share_1km" and border is not None)
        ck.add(f"h2_predictor_{name}", present, "" if present else "not available: model runs without it",
               severity="warn")

    geom = gpd.read_parquet(cfg.data_dir / "interim" / "municipalities_geom.parquet").set_index("geocode").geometry
    neighbours = an.queen_neighbours(geom)

    coefs, fits, imps, morans = [], [], [], []
    for crop in crops:
        df = an.model_frame(units, pred, border, years, crop)
        if not ck.add(f"h2_{crop}_enough_observations", len(df) >= min_n, f"{len(df)} municipality-years", len(df),
                      severity="warn"):
            continue
        c, stats, resid = an.mixed_model(df)
        ck.add(f"h2_{crop}_mixed_model_converged", stats["converged"], severity="warn")
        imp, rf_stats = an.random_forest(df)
        res_mun = resid.groupby(df["geocode"]).mean()
        mi = an.morans_i(res_mun, neighbours)
        coefs.append(c.assign(crop=crop))
        imps.append(imp.assign(crop=crop))
        morans.append({"crop": crop, **mi})
        fits.append({"crop": crop, "years": f"{min(years)}-{max(years)}", "predictors": ",".join(df.attrs["numeric"]),
                     **stats, **{k: v for k, v in rf_stats.items() if k != "rf_r2_folds"}})
        log.info("H2 %-10s n=%6d  R²m=%.3f R²c=%.3f  RF R²(cv)=%.3f  Moran's I=%.3f (p=%.3f)", crop, stats["n"],
                 stats["r2_marginal"], stats["r2_conditional"], rf_stats["rf_r2_grouped_cv"], mi["morans_i"],
                 mi["p_value"])
    if not fits:
        ck.add("h2_any_model", False, "no crop had enough observations")
        return
    _write(pd.concat(coefs, ignore_index=True), cfg, "processed", "h2_mixed_effects.parquet")
    _write(pd.DataFrame(fits), cfg, "processed", "h2_model_fit.parquet")
    _write(pd.concat(imps, ignore_index=True), cfg, "processed", "h2_rf_importance.parquet")
    _write(pd.DataFrame(morans), cfg, "processed", "h2_moran.parquet")


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------
def stage_publish(cfg: Config, ck: CheckLog) -> Path:
    from bbp.publish import publish

    return publish(cfg, ck)


STAGE_FUNCS = {
    "ingest": stage_ingest,
    "sanitation": stage_sanitation,
    "plants": stage_plants,
    "harmonise": stage_harmonise,
    "area": stage_area,
    "reconcile": stage_reconcile,
    "inventory": stage_inventory,
    "analyse": stage_analyse,
    "publish": stage_publish,
}
