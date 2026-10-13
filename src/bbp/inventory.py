"""Theoretical biomethane potential (N1) from crop residues, for both activity-data sources.

    N1 [Nm³ CH4/yr] = P × g × TS × VS/TS × BMP

Yield always comes from PAM (Y = production / planted area), so that
N1_obs = A_obs × Y × factors differs from N1_dec = P × factors only through
the AREA. Where a municipality has observed area but no usable declared yield,
Y falls back to the intermediate region, then the state, then the nation.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

FACTORS = ("g", "ts", "vs_ts", "bmp")


def factor_product(spec: dict, levels: dict[str, str]) -> float:
    return float(np.prod([spec[f][levels[f]] for f in FACTORS]))


def factor_envelope(spec: dict) -> dict[str, float]:
    """Reference, all-min, all-max and one-at-a-time products of the conversion factors."""
    ref = {f: "ref" for f in FACTORS}
    out = {"ref": factor_product(spec, ref),
           "min": factor_product(spec, {f: "min" for f in FACTORS}),
           "max": factor_product(spec, {f: "max" for f in FACTORS})}
    for f, lv in itertools.product(FACTORS, ("min", "max")):
        out[f"oat_{f}_{lv}"] = factor_product(spec, {**ref, f: lv})
    return out


def yields_with_fallback(area: pd.DataFrame, attrs: pd.DataFrame, levels: list[str]) -> pd.DataFrame:
    """Y = prod_t / a_dec_ha per (geocode, year, crop), with hierarchical fallback."""
    df = area.merge(attrs[["geocode", "cd_rgint", "uf"]], on="geocode", how="left")
    df["nation"] = "BR"
    level_col = {"rgint": "cd_rgint", "uf": "uf", "nation": "nation"}
    ok = (df["a_dec_ha"] > 0) & (df["prod_t"] > 0)
    df["yield_t_ha"] = np.where(ok, df["prod_t"] / df["a_dec_ha"], np.nan)
    df["yield_source"] = np.where(ok, "municipality", None)
    for lvl in levels:
        col = level_col[lvl]
        grp = (df[ok].groupby([col, "year", "crop"], observed=True)[["prod_t", "a_dec_ha"]].sum()
               .assign(y=lambda t: t["prod_t"] / t["a_dec_ha"])["y"].rename(f"y_{lvl}").reset_index())
        df = df.merge(grp, on=[col, "year", "crop"], how="left")
        fill = df["yield_t_ha"].isna() & df[f"y_{lvl}"].notna()
        df.loc[fill, "yield_t_ha"] = df.loc[fill, f"y_{lvl}"]
        df.loc[fill, "yield_source"] = lvl
        df = df.drop(columns=f"y_{lvl}")
    return df.drop(columns=["cd_rgint", "uf", "nation"])


def n1_table(area: pd.DataFrame, attrs: pd.DataFrame, crops: dict, factors: dict, fallback: list[str]) -> pd.DataFrame:
    """Long table: one row per (geocode, year, crop, variant) with N1 at ref/min/max and OAT bounds."""
    with_feed = [c for c, s in crops.items() if s.get("feedstock")]
    df = yields_with_fallback(area[area["crop"].isin(with_feed)].copy(), attrs, fallback)
    df["crop"] = df["crop"].astype(str)
    frames = []
    for crop in with_feed:
        spec = crops[crop]
        fs = spec["feedstock"]
        env = factor_envelope(factors["feedstocks"][fs])
        sub = df[df["crop"] == crop]
        declared_only = sub["obs_source"].eq("none")
        prod_obs = np.where(declared_only, sub["prod_t"], sub["a_obs_ha"] * sub["yield_t_ha"])
        for variant, prod in (("dec", sub["prod_t"].to_numpy()), ("obs", prod_obs)):
            block = sub[["geocode", "year", "crop", "yield_source"]].copy()
            block["feedstock"] = fs
            block["variant"] = variant
            block["production_t"] = prod
            block["activity_declared_only"] = declared_only.to_numpy()
            for k, v in env.items():
                block[f"n1_{k}_nm3"] = block["production_t"] * v
            frames.append(block)
    out = pd.concat(frames, ignore_index=True)
    out["variant"] = out["variant"].astype("category")
    return out


def source_vs_factor_effect(n1: pd.DataFrame) -> pd.DataFrame:
    """Per crop and year: how much does switching the area source move N1, compared
    with the spread of the literature conversion factors?

    Both are expressed as log-ratios so they are directly comparable:
      source effect  = |ln(N1_obs / N1_dec)| per municipality (median, p90)
      factor effect  = ln(N1_max / N1_min) (all factors jointly) and per factor (OAT)
    """
    wide = n1.pivot_table(index=["geocode", "year", "crop"], columns="variant", values="n1_ref_nm3", observed=True)
    wide = wide.dropna(subset=["dec", "obs"])
    rows = []
    env_cols = [c for c in n1.columns if c.startswith("n1_") and c != "n1_ref_nm3"]
    env = n1[n1["variant"] == "dec"].groupby("crop")[env_cols + ["n1_ref_nm3"]].sum()
    for (year, crop), g in wide.groupby(level=["year", "crop"]):
        pos = g[(g["dec"] > 0) & (g["obs"] > 0)]
        lr = np.abs(np.log(pos["obs"] / pos["dec"]))
        e = env.loc[crop]
        row = {
            "year": year, "crop": crop, "n_units": len(g), "n_both": len(pos),
            "sum_dec_nm3": g["dec"].sum(), "sum_obs_nm3": g["obs"].sum(),
            "source_median_abs_log": float(lr.median()) if len(lr) else np.nan,
            "source_p90_abs_log": float(lr.quantile(0.9)) if len(lr) else np.nan,
            "source_share_gt10pct": float((lr > np.log(1.1)).mean()) if len(lr) else np.nan,
            "factor_envelope_log": float(np.log(e["n1_max_nm3"] / e["n1_min_nm3"])),
        }
        for f in FACTORS:
            row[f"factor_oat_{f}_log"] = float(np.log(e[f"n1_oat_{f}_max_nm3"] / e[f"n1_oat_{f}_min_nm3"]))
        rows.append(row)
    return pd.DataFrame(rows)


def rank_shift(n1: pd.DataFrame) -> pd.DataFrame:
    """Per year: does the municipal ranking of total N1 (all crops) change with the area source?"""
    from scipy import stats

    from bbp.metrics import top_decile_turnover

    # keep only (municipality, year, crop) cells where BOTH variants exist, so a crop
    # missing from one source in a given year cannot bias the totals
    wide = n1.pivot_table(index=["year", "geocode", "crop"], columns="variant", values="n1_ref_nm3", observed=True)
    tot = wide.dropna(subset=["dec", "obs"]).groupby(level=["year", "geocode"]).sum()
    rows = []
    for year, g in tot.groupby(level="year"):
        g = g.dropna()
        g = g[(g["dec"] > 0) | (g["obs"] > 0)]
        if len(g) < 3:
            continue
        rows.append({
            "year": year, "n_units": len(g),
            "spearman_rho": float(stats.spearmanr(g["dec"], g["obs"]).statistic),
            "top_decile_turnover": top_decile_turnover(g["dec"].droplevel(0), g["obs"].droplevel(0)),
        })
    return pd.DataFrame(rows)
