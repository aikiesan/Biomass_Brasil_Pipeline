"""bbp paper / bbp figures on a synthetic run (no source data): every table, number and figure is built."""

import json

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from bbp import figures, paper
from bbp.checks import CheckLog
from bbp.config import load_config
from bbp.inventory import n1_table, rank_shift, source_vs_factor_effect
from bbp.reconcile import agreement_by_scale

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]   # 2025 is outside the paper profile
CROPS = {"soybean": "mapbiomas_coverage", "sugarcane": "mapbiomas_coverage", "corn_1st": "none"}


def _attrs(n=12):
    codes = 3500000 + np.arange(n) * 7
    return pd.DataFrame({"geocode": codes, "name": [f"M{i}" for i in range(n)],
                         "uf": ["SP"] * (n // 2) + ["MG"] * (n - n // 2),
                         "cd_rgint": pd.array([3501] * (n // 2) + [3101] * (n - n // 2), dtype="Int64"),
                         "is_municipality": True, "amc_id": codes, "amc_size": 1, "created_after_start": False})


def _area(attrs, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for crop, src in CROPS.items():
        for y in YEARS:
            dec = rng.uniform(100, 5000, len(attrs))
            obs = np.nan * dec if src == "none" else dec * np.exp(rng.normal(0, 0.3, len(attrs)))
            rows.append(pd.DataFrame({"geocode": attrs["geocode"], "year": y, "crop": crop, "a_obs_ha": obs,
                                      "obs_source": src, "a_dec_ha": dec, "prod_t": dec * 3.0}))
    return pd.concat(rows, ignore_index=True)


def _write_run(data_dir, cfg):
    """data/processed + interim mesh + logs, as a full run would leave them."""
    (data_dir / "processed").mkdir(parents=True)
    (data_dir / "interim").mkdir()
    (data_dir / "logs").mkdir()
    attrs = _attrs()
    area = _area(attrs)
    metrics, units = agreement_by_scale(area, attrs, ["municipality", "amc", "rgint", "uf", "nation"], None,
                                        [0.1, 0.25], 0.0)
    n1 = n1_table(area, attrs, {c: cfg.crops[c] for c in CROPS}, cfg.factors, ["rgint", "uf", "nation"])
    crops_obs = [c for c, s in CROPS.items() if s != "none"]
    terms = ["Intercept", "C(year)[T.2021]", "log_mun_area_ha", "crop_share", "log_double_crop",
             "irrigated_share", "C(dominant_biome)[T.Cerrado]"]
    coefs = pd.DataFrame([{"term": t, "estimate": 0.1, "std_error": 0.02, "p_value": 0.01, "ci_low": 0.06,
                           "ci_high": 0.14, "crop": c} for c in crops_obs for t in terms])
    fit = pd.DataFrame([{"crop": c, "years": "2020-2024", "predictors": "log_mun_area_ha,crop_share", "n": 60,
                         "municipalities": 12, "regions": 2, "model": "mixed", "r2_marginal": 0.2,
                         "r2_conditional": 0.6, "region_variance_share": 0.4, "converged": True,
                         "rf_r2_grouped_cv": 0.1} for c in crops_obs])
    imp = pd.DataFrame([{"predictor": p, "importance_mean": 0.1 * k, "importance_sd": 0.01, "crop": c}
                        for c in crops_obs for k, p in enumerate(["crop_share", "dominant_biome", "year"])])
    moran = pd.DataFrame([{"crop": c, "morans_i": 0.12, "expected": -0.09, "p_value": 0.02, "n": 12}
                          for c in crops_obs])
    pred = pd.DataFrame({"geocode": np.repeat(attrs["geocode"], len(YEARS)), "year": YEARS * len(attrs),
                         "crop_share": 0.2})
    tables = {"agreement_by_scale": metrics, "municipality_attributes": attrs, "n1_municipal": n1, "crop_area": area,
              "source_vs_factor_effect": source_vs_factor_effect(n1), "rank_shift": rank_shift(n1),
              "h2_mixed_effects": coefs, "h2_model_fit": fit, "h2_rf_importance": imp, "h2_moran": moran,
              "predictors": pred, **{f"unit_ratios_{s}": t for s, t in units.items()}}
    for name, df in tables.items():
        df.to_parquet(data_dir / "processed" / f"{name}.parquet", index=False)
    geom = [box(-48 - (i % 4) * 0.6, -22 - (i // 4) * 0.6, -47.5 - (i % 4) * 0.6, -21.5 - (i // 4) * 0.6)
            for i in range(len(attrs))]
    import geopandas as gpd

    gpd.GeoDataFrame(attrs[["geocode", "name", "uf", "cd_rgint", "is_municipality"]], geometry=geom,
                     crs="EPSG:4674").to_parquet(data_dir / "interim" / "municipalities_geom.parquet")
    (data_dir / "logs" / "checks_ingest.json").write_text(json.dumps(
        {"stage": "ingest", "checks": [{"name": "a", "passed": True, "severity": "error", "detail": "",
                                        "value": None}]}), encoding="utf-8")
    return area


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("BBP_DATA_DIR", str(tmp_path))
    cfg = load_config(profile="paper")
    area = _write_run(tmp_path, cfg)
    return cfg, area


def test_profile_restricts_years_and_crops():
    cfg = load_config(profile="paper")
    assert (cfg.years[0], cfg.years[-1]) == (2008, 2024)
    assert list(cfg.crops)[:2] == ["soybean", "corn_2nd"] and cfg.profile == "paper"
    assert cfg.section("paper")["factor_version"] == cfg.factors["source"]["version"]
    assert load_config().profile is None and load_config().years[-1] >= 2024


def test_unknown_profile_and_unknown_crop_are_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(profile="no_such_profile")
    bad = tmp_path / "bad.yaml"
    bad.write_text("crops: [soybean, banana]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="banana"):
        load_config(profile=str(bad))


def test_restrict_drops_years_and_crops_outside_the_profile():
    df = pd.DataFrame({"year": [2024, 2025, 2024], "crop": pd.Categorical(["soybean", "soybean", "wheat"])})
    out = paper.restrict(df, [2024], ["soybean"])
    assert len(out) == 1 and not isinstance(out["crop"].dtype, pd.CategoricalDtype)


def test_numbers_are_json_safe():
    assert paper._num({"a": np.float64(np.nan), 2020: np.int64(3), "b": [1 / 3]}) == \
        {"a": None, "2020": 3, "b": [0.333333]}


def test_markdown_table():
    md = paper.to_markdown(pd.DataFrame({"crop": ["soy|bean"], "n": [1234], "x": [0.12345], "y": [np.nan]}))
    lines = md.splitlines()
    assert lines[1] == "|---|---:|---:|---:|"
    assert lines[2] == "| soy\\|bean | 1,234 | 0.123 | — |"


def test_table4_compares_source_and_factor_effects():
    sve = pd.DataFrame({"year": [2020, 2021, 2019], "crop": ["soybean"] * 3, "n_both": [10, 12, 99],
                        "sum_obs_nm3": [90.0, 110.0, 1.0], "sum_dec_nm3": [100.0, 100.0, 1.0],
                        "source_median_abs_log": [0.2, 0.4, 9], "source_p90_abs_log": [1.0, 1.0, 9],
                        "source_share_gt10pct": [0.5, 0.5, 1], "factor_envelope_log": [0.6, 0.6, 0.6],
                        **{f"factor_oat_{f}_log": [0.1, 0.1, 0.1] for f in ("g", "ts", "vs_ts", "bmp")}})
    t = paper.table_effects(sve, [2020, 2021], ["soybean"]).iloc[0]
    assert t["n_both"] == 11 and t["ratio_of_totals"] == pytest.approx(1.0)
    assert t["source_vs_factor_ratio"] == pytest.approx(0.3 / 0.6)


def test_paper_build_writes_numbers_and_tables(run):
    cfg, area = run
    ck = CheckLog("paper")
    path = paper.build(cfg, ck)
    assert not ck.failures, [c.name for c in ck.failures]
    nums = json.loads(path.read_text(encoding="utf-8"))
    assert nums["meta"]["profile"] == "paper" and nums["study_area"]["municipalities"] == 12
    # the national ratio quoted in the text is the one of the data, for profile years only
    a = area[(area["crop"] == "soybean") & (area["year"] == 2022)]
    assert nums["h1"]["national_ratio"]["soybean"]["2022"] == pytest.approx(a["a_obs_ha"].sum() / a["a_dec_ha"].sum(),
                                                                          rel=1e-5)
    assert "2025" not in nums["h1"]["national_ratio"]["soybean"]
    assert nums["h3"]["source_vs_factor"]["corn_1st"]["source_median_abs_log"] == 0  # declared only
    assert set(nums["h2"]) == {"soybean", "sugarcane"}
    tables = sorted(p.name for p in (path.parent / "tables").iterdir())
    assert {"table1_data_sources.csv", "table2_agreement_by_scale.md", "table3_h2_models.csv",
            "table3b_h2_coefficients.csv", "table4_source_vs_factor.md"} <= set(tables)
    t2 = pd.read_csv(path.parent / "tables" / "table2_agreement_by_scale.csv")
    assert list(t2.loc[t2["crop"] == "soybean", "scale"]) == ["municipality", "amc", "rgint", "uf", "nation"]


def test_paper_build_refuses_another_factor_version(run):
    cfg, _ = run
    cfg.raw["paper"]["factor_version"] = "something else"
    with pytest.raises(Exception, match="paper_factor_version"):
        paper.build(cfg, CheckLog("paper"))


def test_figures_are_written_without_h2_tables_or_rasters(run, tmp_path):
    cfg, _ = run
    for name in ("h2_mixed_effects", "h2_model_fit", "h2_rf_importance", "h2_moran"):
        (tmp_path / "processed" / f"{name}.parquet").unlink()
    cfg.raw["inputs"]["coverage_tif_dir"] = str(tmp_path / "no_rasters")   # Fig 6 has nothing to draw
    cfg.raw["paper"]["figure_formats"] = ["png"]
    cfg.raw["paper"]["dpi"] = 60
    ck = CheckLog("figures")
    written = figures.build(cfg, ck)
    assert not ck.failures, [(c.name, c.detail) for c in ck.failures]
    skipped = {6, 7, 8}                                   # coverage raster and H2 tables absent
    expected = [figures.file_name(k) + ".png" for k in [*figures.FIGURES, *figures.SUPPLEMENT] if k not in skipped]
    assert sorted(p.name for p in written) == sorted(expected)
    assert any(c.name == "fig7_built" and not c.passed and c.severity == "warn" for c in ck.checks)
    assert (tmp_path / "paper" / "captions.md").read_text(encoding="utf-8").count("**Figure") ==         len(figures.FIGURES) + len(figures.SUPPLEMENT)
    assert (tmp_path / "paper" / "fig06_candidates.csv").exists()
    assert any((tmp_path / "paper" / "figures_check").glob("*_deutan.png"))


def test_figures_are_drawn_at_final_size(run):
    import matplotlib.pyplot as plt

    from bbp import figstyle

    cfg, _ = run
    figstyle.setup()
    data = figures._Data(cfg, CheckLog("figures"))
    cv = figures.fig04_agreement_scale(cfg, CheckLog("figures"), data)
    w, h = cv.fig.get_size_inches()
    plt.close(cv.fig)
    assert (cv.fw, w * 25.4) == (132.0, pytest.approx(132, abs=0.01)) and h * 25.4 < 190


def test_maps_share_one_true_scale():
    from bbp.figstyle import Canvas

    cv = Canvas(100, 50)
    a = cv.map_ax(5, 5, (0, 2_000_000, 0, 1_000_000), scale=40_000)        # 2,000 km at 40 km/mm -> 50 mm
    bbox = a.get_position()
    assert bbox.width * 100 == pytest.approx(50) and bbox.height * 50 == pytest.approx(25)
    assert cv.scale(0, 0, 40_000, seg_km=500, n=2) == pytest.approx(25)     # 1,000 km bar = 25 mm
    assert figures.nice_scale(39_400) == 40_000


def test_log_ratio_classes_are_symmetric_and_keep_one_source_apart():
    edges, colors, labels = figures.ratio_classes()
    assert np.allclose(edges, -edges[::-1]) and len(colors) == len(labels) == len(edges) + 1
    assert np.allclose(np.exp(edges[4:]), [1.1, 1.25, 1.5, 2.0])          # ±10, 25, 50, 100 %
    cls = figures.classify(pd.Series([-np.inf, 0.0, np.inf, np.nan, -5.0, 5.0]), edges)
    assert list(cls[:3]) == [figures.DEC_ONLY, 4, figures.OBS_ONLY] and np.isnan(cls[3])
    assert list(cls[4:]) == [0, 8]                                          # extreme but finite stays a colour


def test_municipal_log_ratio_skips_small_units():
    units = pd.DataFrame({"unit": ["1", "1", "2", "3"], "year": [2020, 2021, 2020, 2020], "crop": "soybean",
                          "a_obs_ha": [100.0, 300.0, 0.0, 10.0], "a_dec_ha": [100.0, 100.0, 80.0, 10.0]})
    lr = figures.municipal_log_ratio(units, "soybean", [2020, 2021], min_area=50)
    assert lr[1] == pytest.approx(np.log(3) / 2) and lr[2] == -np.inf and 3 not in lr.index


def test_source_effect_by_decile():
    rows = []
    for g in range(1, 21):
        for v, f in (("dec", 1.0), ("obs", 2.0 if g <= 10 else 1.1)):
            rows.append({"geocode": g, "year": 2022, "crop": "soybean", "variant": v, "n1_ref_nm3": g * 100.0 * f,
                         "activity_declared_only": False})
        rows.append({"geocode": g, "year": 2022, "crop": "corn_1st", "variant": "dec", "n1_ref_nm3": 1e9,
                     "activity_declared_only": True})
        rows.append({"geocode": g, "year": 2022, "crop": "corn_1st", "variant": "obs", "n1_ref_nm3": 1e9,
                     "activity_declared_only": True})
    n1 = pd.DataFrame(rows)
    d = paper.source_effect_by_decile(n1, [2022])
    assert list(d["municipalities"]) == [2] * 10                            # declared-only crops left out
    assert d["share_of_national_n1"].sum() == pytest.approx(1.0)
    assert d["median_abs_log"].iloc[0] == pytest.approx(np.log(2)) and d["median_abs_log"].iloc[-1] ==         pytest.approx(np.log(1.1))
    area = pd.DataFrame({"geocode": range(1, 21), "year": 2022, "crop": "soybean", "a_obs_ha": 10.0,
                         "a_dec_ha": [10.0] * 10 + [100.0] * 10})
    d50 = paper.source_effect_by_decile(n1, [2022], area, 50)
    assert d50["municipalities"].sum() == 10                                # cells below 50 ha dropped


def test_graticule_labels_where_meridians_cross_the_frame():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyproj import Transformer

    from bbp.cartography import MAP_CRS, graticule
    from bbp.figstyle import map_frame

    tr = Transformer.from_crs("EPSG:4326", MAP_CRS, always_xy=True)
    (x0, x1), (y0, y1) = tr.transform([-60, -40], [-25, -5])
    fig, ax = plt.subplots()
    map_frame(ax, (x0, x1, y0, y1))
    ticks = graticule(ax, step=10, sides=("bottom", "left"))
    plt.close(fig)
    assert "50°W" in ticks["bottom"][1] and "10°S" in ticks["left"][1] and "20°S" in ticks["left"][1]
