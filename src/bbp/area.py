"""Crop area per municipality and year from both sources (observed vs declared)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pam_variable(pam: pd.DataFrame, table: int, product: str, variable: str, name: str) -> pd.DataFrame:
    sel = pam[(pam["table"] == table) & (pam["product"] == product) & (pam["variable"] == variable)]
    return sel[["geocode", "year", "value", "flag"]].rename(columns={"value": name, "flag": f"{name}_flag"})


def crop_area_table(
    municipalities: pd.Series,
    years: list[int],
    crops: dict,
    mapbiomas: pd.DataFrame,
    pam: pd.DataFrame,
    second_crop: pd.DataFrame | None,
) -> pd.DataFrame:
    mb_years = set(mapbiomas["year"].unique().tolist())
    sc_years = set(second_crop["year"].unique().tolist()) if second_crop is not None else set()
    mb_by_class = mapbiomas.groupby(["geocode", "year", "class_id"], as_index=False)["area_ha"].sum()

    frames = []
    for crop, spec in crops.items():
        grid = pd.MultiIndex.from_product([municipalities, years], names=["geocode", "year"]).to_frame(index=False)
        grid["crop"] = crop

        # ---- observed -----------------------------------------------------
        # a coverage class, a second-crop class, or both (summed: e.g. cotton = first-season class 62
        # + second-season cotton). A year counts as observed only if every named source covers it.
        parts, available, sources = [], pd.Series(True, index=grid.index), []
        if spec.get("mapbiomas_class"):
            parts.append(mb_by_class[mb_by_class["class_id"].isin(spec["mapbiomas_class"])])
            available &= grid["year"].isin(mb_years)
            sources.append("mapbiomas_coverage")
        if spec.get("mapbiomas_second_crop"):
            if second_crop is not None:
                parts.append(second_crop[second_crop["class"] == spec["mapbiomas_second_crop"]])
            available &= grid["year"].isin(sc_years)
            sources.append("mapbiomas_second_crop")
        if not sources:
            available &= False
        obs = (pd.concat([p_[["geocode", "year", "area_ha"]] for p_ in parts], ignore_index=True)
               if parts else pd.DataFrame({"geocode": pd.Series(dtype="int64"), "year": pd.Series(dtype="int64"),
                                           "area_ha": pd.Series(dtype="float64")}))
        obs = obs.groupby(["geocode", "year"], as_index=False)["area_ha"].sum().rename(columns={"area_ha": "a_obs_ha"})
        obs_source = "+".join(sources) or "none"
        grid = grid.merge(obs, on=["geocode", "year"], how="left")
        grid["a_obs_ha"] = np.where(available, grid["a_obs_ha"].fillna(0.0), np.nan)
        grid["obs_source"] = obs_source

        # ---- declared -----------------------------------------------------
        t, p = spec["pam_table"], spec["pam_product"]
        for var, name in (("area_planted_ha", "a_dec_ha"), ("area_harvested_ha", "a_harv_ha"), ("production_t", "prod_t")):
            grid = grid.merge(_pam_variable(pam, t, p, var, name), on=["geocode", "year"], how="left")
        prelim = pam.loc[(pam["table"] == t) & pam["preliminary"].astype(bool), "year"].unique() if "preliminary" in pam else []
        grid["dec_preliminary"] = grid["year"].isin(prelim)
        frames.append(grid)

    out = pd.concat(frames, ignore_index=True)
    for col in ("a_dec_ha_flag", "a_harv_ha_flag", "prod_t_flag"):
        out[col] = out[col].fillna("no_record")
    out = out.rename(columns={"a_dec_ha_flag": "dec_flag", "a_harv_ha_flag": "harv_flag", "prod_t_flag": "prod_flag"})
    out["crop"] = out["crop"].astype("category")
    return out.sort_values(["crop", "geocode", "year"]).reset_index(drop=True)
