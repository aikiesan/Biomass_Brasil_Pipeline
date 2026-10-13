"""Agreement between observed and declared crop area across spatial scales (H1)
and the unit-level tables used to explain the divergence (H2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bbp.metrics import agreement_table

SCALE_KEYS = {"municipality": "geocode", "amc": "amc_id", "rgint": "cd_rgint", "uf": "uf", "nation": "nation"}


def aggregate(area: pd.DataFrame, attrs: pd.DataFrame, scale: str, mismatch: pd.DataFrame | None = None) -> pd.DataFrame:
    """Sum observed and declared area to a spatial scale.

    At municipality scale, (geocode, year) pairs whose boundaries differ between
    the sources (see harmonise.mesh_mismatch_flags) are dropped; every coarser
    scale contains whole AMCs and is unaffected.
    """
    key = SCALE_KEYS[scale]
    df = area[area["obs_source"] != "none"]
    df = df[df["a_obs_ha"].notna() & df["a_dec_ha"].notna()]
    if scale == "municipality" and mismatch is not None and len(mismatch):
        m = df.set_index(["geocode", "year"]).index.isin(mismatch.set_index(["geocode", "year"]).index)
        df = df[~m]
    df = df.merge(attrs[["geocode", "amc_id", "cd_rgint", "uf"]], on="geocode", how="left")
    df["nation"] = "BR"
    out = df.groupby([key, "year", "crop"], observed=True, as_index=False)[["a_obs_ha", "a_dec_ha"]].sum()
    out = out.rename(columns={key: "unit"})
    out["unit"] = out["unit"].astype(str)
    out.insert(0, "scale", scale)
    return out


def unit_ratios(agg: pd.DataFrame) -> pd.DataFrame:
    out = agg.copy()
    both = (out["a_obs_ha"] > 0) & (out["a_dec_ha"] > 0)
    out["ratio"] = np.where(out["a_dec_ha"] > 0, out["a_obs_ha"] / out["a_dec_ha"].where(out["a_dec_ha"] > 0), np.nan)
    out["log_ratio"] = np.where(both, np.log(out["a_obs_ha"].where(both) / out["a_dec_ha"].where(both)), np.nan)
    out["presence"] = np.select(
        [both, (out["a_obs_ha"] > 0) & (out["a_dec_ha"] == 0), (out["a_obs_ha"] == 0) & (out["a_dec_ha"] > 0)],
        ["both", "obs_only", "dec_only"], default="neither",
    )
    return out


def agreement_by_scale(area, attrs, scales, mismatch, tolerances, min_area_ha) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    metrics, units = [], {}
    for scale in scales:
        agg = aggregate(area, attrs, scale, mismatch)
        units[scale] = unit_ratios(agg)
        tab = agreement_table(agg, ["scale", "crop", "year"], tolerances=tuple(tolerances),
                              min_area=min_area_ha if scale != "nation" else 0.0)
        metrics.append(tab)
    return pd.concat(metrics, ignore_index=True), units


def predictors(area: pd.DataFrame, attrs: pd.DataFrame, mapbiomas: pd.DataFrame) -> pd.DataFrame:
    """Municipality x year explanatory variables for the divergence model (H2).

    - log municipal area
    - dominant biome (largest MapBiomas area share)
    - agricultural share of the municipality (MapBiomas classes of the crop group)
    - double-cropping intensity: observed second-season corn / observed soybean
    - created_after_start (new municipality) flag
    Border-proximity and Landsat-observation-count predictors need raster
    work and are added by a later stage (see docs/decisions.md).
    """
    tot = mapbiomas.groupby(["geocode", "year"], as_index=False)["area_ha"].sum().rename(columns={"area_ha": "mun_area_ha"})
    biome = (mapbiomas.groupby(["geocode", "biome"], observed=True)["area_ha"].sum().reset_index()
             .sort_values("area_ha").drop_duplicates("geocode", keep="last")[["geocode", "biome"]]
             .rename(columns={"biome": "dominant_biome"}))
    crop_classes = [20, 39, 40, 41, 46, 47, 48, 62, 35]
    agri = (mapbiomas[mapbiomas["class_id"].isin(crop_classes)]
            .groupby(["geocode", "year"], as_index=False)["area_ha"].sum().rename(columns={"area_ha": "crop_area_ha"}))
    wide = area.pivot_table(index=["geocode", "year"], columns="crop", values="a_obs_ha", observed=True)
    dc = pd.DataFrame(index=wide.index)
    if {"corn_2nd", "soybean"} <= set(wide.columns):
        dc["double_crop_intensity"] = wide["corn_2nd"] / wide["soybean"].where(wide["soybean"] > 0)
    dc = dc.reset_index()

    out = tot.merge(agri, on=["geocode", "year"], how="left").merge(dc, on=["geocode", "year"], how="left")
    out["crop_area_ha"] = out["crop_area_ha"].fillna(0.0)
    out["crop_share"] = out["crop_area_ha"] / out["mun_area_ha"]
    out["log_mun_area_ha"] = np.log(out["mun_area_ha"])
    out = out.merge(biome, on="geocode", how="left")
    out = out.merge(attrs[["geocode", "uf", "cd_rgint", "created_after_start"]], on="geocode", how="left")
    return out
