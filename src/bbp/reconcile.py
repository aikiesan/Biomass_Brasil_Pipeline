"""Agreement between observed and declared crop area across spatial scales (H1)
and the unit-level tables used to explain the divergence (H2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bbp.landcover import GROUPS
from bbp.metrics import agreement_table, lin_ccc

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


DIRECTIONS = ("both_up", "both_down", "obs_up_dec_down", "obs_down_dec_up", "flat")
# covers against which the trend gap is read: the Fig 1 natural group, with pasture and mosaic apart
_LC = {g.key: g.classes for g in GROUPS}
TREND_COVERS = {"natural": _LC["natural"], "pasture": (15,), "mosaic": (21,),
                "other_crops": _LC["other_crops"], "plantation": _LC["plantation"]}


def trend_units(units: pd.DataFrame, windows: list[list[int]], min_area_ha: float) -> pd.DataFrame:
    """Change in observed and declared area between two windows, per unit and crop.

    Each source is averaged over the years of a window; a unit is scored only when
    both sources report every year of both windows (so a year dropped for a mesh
    mismatch never biases one window) and both window means of both sources reach
    `min_area_ha`. The change is the log of the later mean over the earlier one.
    """
    (y0a, y0b), (y1a, y1b) = windows
    parts = []
    for tag, (a, b) in (("0", (y0a, y0b)), ("1", (y1a, y1b))):
        w = units[units["year"].between(a, b)]
        g = w.groupby(["scale", "unit", "crop"], observed=True).agg(
            **{f"a_obs_ha_{tag}": ("a_obs_ha", "mean"), f"a_dec_ha_{tag}": ("a_dec_ha", "mean"),
               f"n_years_{tag}": ("year", "nunique")})
        parts.append(g[g[f"n_years_{tag}"] == b - a + 1].drop(columns=f"n_years_{tag}"))
    t = parts[0].join(parts[1], how="inner").reset_index()
    area_cols = ["a_obs_ha_0", "a_obs_ha_1", "a_dec_ha_0", "a_dec_ha_1"]
    t = t[(t[area_cols] >= min_area_ha).all(axis=1)].copy()
    t["dlog_obs"] = np.log(t["a_obs_ha_1"] / t["a_obs_ha_0"])
    t["dlog_dec"] = np.log(t["a_dec_ha_1"] / t["a_dec_ha_0"])
    so, sd = np.sign(t["dlog_obs"]), np.sign(t["dlog_dec"])
    t["direction"] = np.select([(so > 0) & (sd > 0), (so < 0) & (sd < 0), (so > 0) & (sd < 0), (so < 0) & (sd > 0)],
                               DIRECTIONS[:4], default="flat")
    t["crop"] = t["crop"].astype(str)
    return t.reset_index(drop=True)


def trend_agreement(trend: pd.DataFrame) -> pd.DataFrame:
    """Per scale and crop: do the two sources agree on the direction and size of the change?

    `dlog_*_total` is the change of the summed area over the scored units, the
    "story" each source tells for the whole set; the rest describe the units.
    """
    rows = []
    for (scale, crop), g in trend.groupby(["scale", "crop"], sort=False):
        n = len(g)
        share = g["direction"].value_counts(normalize=True)
        rows.append({
            "scale": scale, "crop": crop, "n_units": n,
            "share_same_sign": float(share.get("both_up", 0) + share.get("both_down", 0)),
            **{f"share_{d}": float(share.get(d, 0)) for d in DIRECTIONS[:4]},
            "spearman_rho": float(g["dlog_obs"].corr(g["dlog_dec"], method="spearman")) if n >= 3 else np.nan,
            "lin_ccc": lin_ccc(g["dlog_obs"].to_numpy(), g["dlog_dec"].to_numpy()) if n >= 3 else np.nan,
            "median_dlog_obs": float(g["dlog_obs"].median()), "median_dlog_dec": float(g["dlog_dec"].median()),
            "dlog_obs_total": float(np.log(g["a_obs_ha_1"].sum() / g["a_obs_ha_0"].sum())),
            "dlog_dec_total": float(np.log(g["a_dec_ha_1"].sum() / g["a_dec_ha_0"].sum())),
        })
    return pd.DataFrame(rows)


def trend_by_magnitude(trend: pd.DataFrame, edges: list[float]) -> pd.DataFrame:
    """Direction agreement by size of the change: is the disagreement where little changed?

    The size is the mean |Δlog| of the two sources; `edges` are the inner bin limits
    (Δlog), so 0.25 is a change of about 28 %. `area_share` is the share of the late
    declared area of the scale and crop held by the units of the bin.
    """
    t = trend.copy()
    t["magnitude"] = t[["dlog_obs", "dlog_dec"]].abs().mean(axis=1)
    bins = [0.0, *edges, np.inf]
    t["bin"] = pd.cut(t["magnitude"], bins, labels=False, include_lowest=True)
    rows = []
    for (scale, crop, b), g in t.groupby(["scale", "crop", "bin"], sort=True):
        tot = t.loc[(t["scale"] == scale) & (t["crop"] == crop), "a_dec_ha_1"].sum()
        rows.append({"scale": scale, "crop": crop, "bin": int(b), "dlog_low": bins[int(b)], "dlog_high": bins[int(b) + 1],
                     "n_units": len(g), "share_same_sign": float(g["direction"].isin(["both_up", "both_down"]).mean()),
                     "area_share": float(g["a_dec_ha_1"].sum() / tot) if tot > 0 else np.nan})
    return pd.DataFrame(rows)


def landcover_change(mapbiomas: pd.DataFrame, windows: list[list[int]], covers: dict[str, tuple[int, ...]]) -> pd.DataFrame:
    """Change in the share of each cover in the municipality (percentage points) between the windows, per MapBiomas."""
    cls = {c: k for k, v in covers.items() for c in v}
    m = mapbiomas.assign(cover=mapbiomas["class_id"].map(cls))
    tot = m.groupby(["geocode", "year"])["area_ha"].sum()
    share = (m.dropna(subset=["cover"]).groupby(["geocode", "year", "cover"])["area_ha"].sum()
             .unstack("cover").reindex(columns=list(covers)).fillna(0.0).div(tot, axis=0).dropna())

    def mean(a, b):
        return share[share.index.get_level_values("year").isin(range(a, b + 1))].groupby(level="geocode").mean()

    (a0, b0), (a1, b1) = windows
    out = (mean(a1, b1) - mean(a0, b0)) * 100
    return out.add_prefix("dpp_").reset_index()


def trend_landcover(trend: pd.DataFrame, change: pd.DataFrame, n_quantiles: int = 4) -> pd.DataFrame:
    """Does the land-cover change in the municipality go with the gap between the two trends?

    gap = Δlog(map) − Δlog(statistics). Per crop and cover: Spearman ρ(gap, Δshare) over
    the municipalities, and the municipalities split in quantiles of Δshare with their
    median gap and direction agreement (quantile 1 = the cover lost most).
    """
    t = trend[trend["scale"] == "municipality"].copy()
    t["geocode"] = t["unit"].astype(change["geocode"].dtype)
    t["gap"] = t["dlog_obs"] - t["dlog_dec"]
    t = t.merge(change, on="geocode", how="inner")
    rows = []
    for crop, g in t.groupby("crop", sort=False):
        for col in [c for c in change if c.startswith("dpp_")]:
            rho = float(g["gap"].corr(g[col], method="spearman")) if len(g) >= 3 else np.nan
            q = pd.qcut(g[col].rank(method="first"), n_quantiles, labels=False) if len(g) >= n_quantiles else None
            for k, h in (g.groupby(q) if q is not None else []):
                rows.append({"crop": crop, "cover": col[4:], "spearman_rho": rho, "quantile": int(k) + 1,
                             "n_units": len(h), "median_dpp": float(h[col].median()), "median_gap": float(h["gap"].median()),
                             "share_same_sign": float(h["direction"].isin(["both_up", "both_down"]).mean())})
    return pd.DataFrame(rows)


def predictors(area: pd.DataFrame, attrs: pd.DataFrame, mapbiomas: pd.DataFrame,
               irrigation: pd.DataFrame | None = None, border: pd.DataFrame | None = None,
               observations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Municipality x year explanatory variables for the divergence model (H2).

    - log municipal area
    - dominant biome (largest MapBiomas area share)
    - agricultural share of the municipality (MapBiomas classes of the crop group)
    - double-cropping intensity: observed second-season corn / observed soybean
    - created_after_start (new municipality) flag
    - irrigated area and its share of the crop area (MapBiomas irrigation systems), when given
    - border share: crop area within 1 and 2 km of the municipal boundary / crop area
      (from the coverage rasters, `bbp border-share`), when given; per municipality x year x crop
      it is joined by the analysis, here only the all-crops value
    - Landsat valid observations (cloud proxy, `docs/gee/landsat_valid_observations.js`), when given
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
    if irrigation is not None:
        irr = irrigation.groupby(["geocode", "year"], as_index=False)["area_ha"].sum().rename(
            columns={"area_ha": "irrigated_area_ha"})
        irr["geocode"], irr["year"] = irr["geocode"].astype(out["geocode"].dtype), irr["year"].astype(out["year"].dtype)
        out = out.merge(irr, on=["geocode", "year"], how="left")
        out["irrigated_area_ha"] = out["irrigated_area_ha"].fillna(0.0)  # absent from the sheet = no irrigation mapped
        out["irrigated_share"] = (out["irrigated_area_ha"] / out["crop_area_ha"].where(out["crop_area_ha"] > 0)).clip(upper=1)
    if border is not None:
        b = border[border["crop"] == "all"][["geocode", "year", "border_share_1km", "border_share_2km"]]
        out = out.merge(b, on=["geocode", "year"], how="left")
    if observations is not None:
        out = out.merge(observations[["geocode", "year", "landsat_valid_obs"]], on=["geocode", "year"], how="left")
    return out
