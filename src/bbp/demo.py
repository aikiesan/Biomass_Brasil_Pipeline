"""Using the system (`bbp demo`): a typology of the intermediate regions and a ruler for the biomethane mandate.

**Typology.** Each intermediate region is described by the composition of its observed crop area (MapBiomas
Col. 11, `demo.year`): soybean, second-season corn, sugarcane, coffee, citrus, cotton, rice and other crops
(classes 41, 48, 35). Second-season corn is counted as a season of its own, so the parts are shares of the cropped
area summed over the two seasons. Compositions are closed (shares sum to 1); parts below a detection limit (`detection_share`, 0.1 % of the region's
crop area) are rounded zeros, replaced multiplicatively by δ = 0.65 × the limit (Martín-Fernández et al. 2003), and the
composition is transformed by the centred log-ratio (CLR), so Euclidean distance is the Aitchison distance.
Regions with less than `min_crop_kha` of crops are left out (their composition is noise) and labelled
`minor agriculture`. Clusters: Ward's hierarchical clustering on the CLR coordinates. k is the most stable partition (largest
adjusted Rand index between Ward and k-means, fixed seed) among the k of `k_range` whose smallest cluster holds at
least `min_cluster` regions (ties within 0.02 of the best ARI go to the larger mean silhouette); the mean
silhouette and Calinski–Harabasz index are reported for every k. Each region is then crossed with the share of its crop area within reach of the paired plants
(`reach_units`, intermediate-region level) and the number of plants, and with its N1 (observed variant, ref
factors). Priority regions: N1 in the top quartile and less than `priority_share` of all crops within
`priority_km` of a biogas plant.

**Mandate ruler.** The targets and their official volumes are configuration (`demo.mandate`, each with its act,
URL and access date): there is no conversion to do, the acts state the volume. Indicative targets without an
official volume get one from the 2027 base (volume / percentage), flagged `derived`. Flows: ANP authorised
capacity and production at each snapshot (`plants_biomethane_snapshots`). Stock: the theoretical methane of the
crop residues (N1, observed variant) within reach of existing plants, per municipality N1 × the share of its crop
area within r km (`reach_hist`), with the factor envelope (min–max) — a stock, drawn apart from the flows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("bbp")

PARTS = {"soybean": (39,), "sugarcane": (20,), "coffee": (46,), "citrus": (47,), "rice": (40,), "cotton": (62,),
         "other": (41, 48, 35)}
SECOND = {"corn_2nd": "corn_2nd", "cotton": "cotton_2nd"}     # part: second-crop class name
ORDER = ["soybean", "corn_2nd", "sugarcane", "coffee", "citrus", "cotton", "rice", "other"]
N1_CROPS = ["soybean", "corn_1st", "corn_2nd", "sugarcane", "coffee", "citrus"]


def _spec(cfg) -> dict:
    return cfg.section("demo") or {}


# ---------------------------------------------------------------------------
# compositions
# ---------------------------------------------------------------------------
def region_composition(cov: pd.DataFrame, second: pd.DataFrame, mun: pd.DataFrame, year: int) -> pd.DataFrame:
    """Crop area (ha) per intermediate region and part. `cov`: geocode, year, class_id, area_ha (Col. 11
    statistics); `second`: geocode, year, class, area_ha (second-crop zonal statistics); `mun`: geocode, cd_rgint."""
    rg = mun.set_index("geocode")["cd_rgint"]
    c = cov[cov["year"] == year]
    cols = {}
    for part, classes in PARTS.items():
        s = c[c["class_id"].isin(classes)].groupby("geocode")["area_ha"].sum()
        cols[part] = s
    s2 = second[second["year"] == year]
    for part, cls in SECOND.items():
        add = s2[s2["class"] == cls].groupby("geocode")["area_ha"].sum()
        cols[part] = cols.get(part, pd.Series(dtype=float)).add(add, fill_value=0)
    m = pd.DataFrame(cols).fillna(0.0)
    m = m.groupby(rg.reindex(m.index).to_numpy()).sum()
    m.index.name = "cd_rgint"
    return m[ORDER]


def closure(x: pd.DataFrame) -> pd.DataFrame:
    return x.div(x.sum(axis=1), axis=0)


def replace_zeros(shares: pd.DataFrame, detection: float = 1e-3) -> tuple[pd.DataFrame, float]:
    """Multiplicative replacement with a detection limit: parts below `detection` are rounded zeros, set to
    δ = 0.65 × detection; the other parts are rescaled so each row still sums to 1 and keep their ratios."""
    d = 0.65 * float(detection)
    x = shares.to_numpy(dtype=float).copy()
    z = x < detection
    keep = np.where(z, 0.0, x)
    scale = (1 - z.sum(axis=1, keepdims=True) * d) / keep.sum(axis=1, keepdims=True)
    x = np.where(z, d, keep * scale)
    return pd.DataFrame(x, index=shares.index, columns=shares.columns), d


def clr(shares: pd.DataFrame) -> pd.DataFrame:
    lg = np.log(shares.to_numpy(dtype=float))
    return pd.DataFrame(lg - lg.mean(axis=1, keepdims=True), index=shares.index, columns=shares.columns)


def cluster(z: pd.DataFrame, k_range=(3, 8), seed: int = 0, select: str = "stability",
            min_cluster: int = 10) -> tuple[pd.Series, pd.DataFrame]:
    """Ward clusters on the CLR coordinates. Returns (labels 1..k, diagnostics per k: mean silhouette,
    Calinski–Harabasz, adjusted Rand index against k-means, smallest cluster).

    `select`: "stability" (default) = the k whose Ward partition agrees best with k-means (ARI), among the k whose
    smallest cluster has at least `min_cluster` units; "silhouette" = the largest mean silhouette under the same
    constraint. Without the constraint the silhouette of these data keeps growing with k (ever smaller clusters)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score

    X = z.to_numpy()
    L = linkage(X, method="ward")
    rows, labels = [], {}
    for k in range(int(k_range[0]), int(k_range[1]) + 1):
        lab = fcluster(L, k, criterion="maxclust")
        km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(X)
        rows.append({"k": k, "silhouette": float(silhouette_score(X, lab)),
                     "calinski_harabasz": float(calinski_harabasz_score(X, lab)),
                     "ari_kmeans": float(adjusted_rand_score(lab, km)),
                     "smallest_cluster": int(np.bincount(lab)[1:].min())})
        labels[k] = lab
    diag = pd.DataFrame(rows)
    ok = diag[diag["smallest_cluster"] >= min_cluster]
    ok = ok if len(ok) else diag
    if select == "stability":                   # ties (ARI within 0.02 of the best) go to the larger silhouette
        ok = ok[ok["ari_kmeans"] >= ok["ari_kmeans"].max() - 0.02]
    best = int(ok.loc[ok["silhouette"].idxmax(), "k"])
    diag["chosen"] = diag["k"] == best
    lab = labels[best]
    # number clusters by total size, largest first, so labels are stable between runs
    order = pd.Series(lab).value_counts().index
    remap = {old: i + 1 for i, old in enumerate(order)}
    return pd.Series([remap[v] for v in lab], index=z.index, name="cluster"), diag


def cluster_names(shares: pd.DataFrame, labels: pd.Series) -> dict[int, str]:
    """A name per cluster from its mean composition: the parts above 20 %, largest first (else the largest two)."""
    out = {}
    for c, g in shares.groupby(labels):
        m = g.mean().sort_values(ascending=False)
        top = [k for k, v in m.items() if v >= 0.2] or list(m.index[:2])
        out[int(c)] = " + ".join(top[:3])
    return out


# ---------------------------------------------------------------------------
# the typology table
# ---------------------------------------------------------------------------
def typology(cfg, ck) -> dict[str, pd.DataFrame]:
    spec = _spec(cfg).get("typology", {})
    year = int(_spec(cfg).get("year", 2024))
    d = cfg.data_dir
    mun = pd.read_parquet(d / "interim" / "municipalities.parquet")
    mun = mun[mun["is_municipality"]]
    cov = pd.read_parquet(d / "interim" / "mapbiomas_coverage_municipal.parquet")
    second = pd.read_parquet(d / "interim" / "second_crop_municipal.parquet")
    comp = region_composition(cov, second, mun, year)
    total = comp.sum(axis=1)
    min_ha = float(spec.get("min_crop_kha", 10)) * 1e3
    keep = total >= min_ha
    ck.add("demo_regions", True, f"{int(keep.sum())} of {len(comp)} intermediate regions with ≥ {min_ha / 1e3:g} kha of "
           f"crops are clustered; {int((~keep).sum())} are 'minor agriculture'", int(keep.sum()))
    shares = closure(comp[keep])
    det = float(spec.get("detection_share", 1e-3))
    repl, delta = replace_zeros(shares, det)
    ck.add("demo_zero_replacement", bool(np.allclose(repl.sum(axis=1), 1)),
           f"detection limit {det:.1%} of the region's crop area, δ = {delta:.2e}; "
           f"{int((shares < det).to_numpy().sum())} parts below it replaced", delta)
    z = clr(repl)
    ck.add("demo_clr_rows_sum_to_zero", bool(np.allclose(z.sum(axis=1), 0, atol=1e-9)), "CLR coordinates sum to 0")
    labels, diag = cluster(z, tuple(spec.get("k_range", (3, 8))), int(spec.get("seed", 0)),
                           spec.get("select", "stability"), int(spec.get("min_cluster", 10)))
    best = diag[diag["chosen"]].iloc[0]
    kr = tuple(spec.get("k_range", (3, 8)))
    ck.add("demo_k_inside_range", kr[0] < int(best["k"]) < kr[1],
           f"chosen k = {int(best['k'])} inside the range {kr} (a k at the edge means the range is too narrow)",
           int(best["k"]), severity="warn")
    ck.add("demo_clusters", int(best["smallest_cluster"]) >= int(spec.get("min_cluster", 10)),
           f"k = {int(best['k'])} by {spec.get('select', 'stability')} (Ward vs k-means ARI {best['ari_kmeans']:.2f}; "
           f"mean silhouette {best['silhouette']:.3f}; Calinski–Harabasz {best['calinski_harabasz']:.0f}); smallest "
           f"cluster {int(best['smallest_cluster'])} regions", int(best["k"]), severity="warn")
    names = cluster_names(shares, labels)

    # crossing: share within reach, plants, N1
    units = pd.read_parquet(d / "processed" / "reach_units.parquet")
    rg_units = units[(units["level"] == "rgint")].copy()
    rg_units["cd_rgint"] = rg_units["unit"].astype("int64")
    rows = pd.DataFrame(index=comp.index)
    rows["crop_area_ha"] = total
    for c in ORDER:
        rows[f"share_{c}"] = closure(comp)[c]
    rows["cluster"] = labels.reindex(rows.index)
    rows["cluster_name"] = rows["cluster"].map(names).fillna("minor agriculture")
    rows.loc[~keep.reindex(rows.index, fill_value=False), "cluster"] = 0
    for pair, r in [(p["pair"], p["radius_km"]) for p in spec.get("reach", [
            {"pair": "all_crops|biogas", "radius_km": 30}, {"pair": "all_crops|biomethane", "radius_km": 30},
            {"pair": "soybean|biodiesel", "radius_km": 50}, {"pair": "corn_2nd|corn_ethanol", "radius_km": 50},
            {"pair": "sugarcane|sugarcane_mill", "radius_km": 30}])]:
        s = rg_units[(rg_units["pair"] == pair) & (rg_units["radius_km"] == float(r))].set_index("cd_rgint")
        key = pair.replace("|", "__")
        rows[f"within_{int(r)}km__{key}"] = s["share"]
        rows[f"plants__{pair.split('|')[1]}"] = s["n_plants"]
    n1 = pd.read_parquet(d / "processed" / "n1_municipal.parquet",
                         columns=["geocode", "year", "crop", "variant", "n1_ref_nm3"])
    n1 = n1[(n1["year"] == year) & (n1["variant"] == "obs")]
    rg = mun.set_index("geocode")["cd_rgint"]
    rows["n1_ref_nm3"] = n1.groupby(n1["geocode"].map(rg))["n1_ref_nm3"].sum()
    names_rg = mun.drop_duplicates("cd_rgint").set_index("cd_rgint")
    rows["nm_rgint"] = names_rg["nm_rgint"]
    rows["uf"] = names_rg["uf"]
    rows = rows.reset_index()

    # priorities: large residue, little of it within reach
    pk = float(spec.get("priority_km", 30))
    col = f"within_{int(pk)}km__all_crops__biogas"
    q = float(spec.get("priority_n1_quantile", 0.75))
    thr = rows.loc[rows["cluster"] > 0, "n1_ref_nm3"].quantile(q)
    ps = float(spec.get("priority_share", 0.25))
    rows["priority"] = (rows["n1_ref_nm3"] >= thr) & (rows[col] < ps)
    pr = rows[rows["priority"]].sort_values("n1_ref_nm3", ascending=False)
    ck.add("demo_priority_regions", True,
           f"{len(pr)} regions with N1 in the top {100 * (1 - q):.0f} % and < {ps:.0%} of all crops within {pk:g} km "
           f"of a biogas plant: " + "; ".join(f"{r.nm_rgint} ({r.uf})" for r in pr.head(12).itertuples()),
           pr["nm_rgint"].tolist())
    for name in spec.get("expected_priorities") or []:
        hit = pr["nm_rgint"].str.contains(name, case=False, regex=False).any()
        ck.add(f"demo_priority_{name.lower().replace(' ', '_')}", True,
               f"{name}: {'a priority region' if hit else 'not a priority region under these thresholds'}", bool(hit))
    comp_out = pd.concat([shares, z.add_prefix("clr_")], axis=1).reset_index()
    return {"typology": rows, "typology_k": diag, "typology_composition": comp_out}


# ---------------------------------------------------------------------------
# the mandate ruler
# ---------------------------------------------------------------------------
def mandate_targets(spec: list[dict]) -> pd.DataFrame:
    """Targets with official or derived volumes (m³/day). A target without an official volume gets
    pct × (volume / pct of the reference year), flagged `derived`."""
    t = pd.DataFrame(spec).sort_values("year").reset_index(drop=True)
    t["volume_m3d"] = t.get("volume_m3d")
    t["implied_base_m3d"] = t["volume_m3d"] / (t["ghg_reduction_pct"] / 100)
    ref = t[t.get("base_reference", pd.Series(False, index=t.index)).fillna(False).astype(bool)]
    base = float(ref["implied_base_m3d"].iloc[0]) if len(ref) else float(t["implied_base_m3d"].dropna().iloc[-1])
    t["volume_source"] = np.where(t["volume_m3d"].notna(), "official", "derived")
    t.loc[t["volume_m3d"].isna(), "volume_m3d"] = t["ghg_reduction_pct"] / 100 * base
    return t


def stock_within_reach(hist: pd.DataFrame, n1: pd.DataFrame, pair: str, radii_km: list[float]) -> pd.DataFrame:
    """N1 (Nm³ CH4/yr; ref, min, max) of the municipalities' crop residues within r km of the pair's plants:
    per municipality N1 × the share of its crop area within r km."""
    h = hist[hist["pair"] == pair]
    tot = h.groupby("geocode")["area_ha"].sum()
    rows = []
    for r in radii_km:
        share = (h[h["bin_km"] < r].groupby("geocode")["area_ha"].sum() / tot).reindex(tot.index).fillna(0)
        m = n1.set_index("geocode")
        s = share.reindex(m.index).fillna(0.0)
        rows.append({"pair": pair, "radius_km": float(r),
                     **{c.replace("_nm3", "_nm3_yr"): float((m[c] * s).sum()) for c in ("n1_ref_nm3", "n1_min_nm3",
                                                                                      "n1_max_nm3")},
                     "n1_total_ref_nm3_yr": float(m["n1_ref_nm3"].sum())})
    out = pd.DataFrame(rows)
    for c in ("n1_ref", "n1_min", "n1_max"):
        out[f"{c}_m3d"] = out[f"{c}_nm3_yr"] / 365.0
    return out


def mandate(cfg, ck) -> dict[str, pd.DataFrame]:
    spec = _spec(cfg).get("mandate", {})
    d = cfg.data_dir
    t = mandate_targets(spec.get("targets") or [])
    for r in t[t["volume_source"] == "official"].itertuples():
        ck.add(f"demo_mandate_{r.year}_source", bool(getattr(r, "url", "")) and bool(getattr(r, "accessed", "")),
               f"{r.year}: {r.ghg_reduction_pct:g} % = {r.volume_m3d:,.0f} m³/d ({r.status}; {r.act})",
               float(r.volume_m3d))
    off = t[t["volume_source"] == "official"]
    if len(off) > 1:
        b = off["implied_base_m3d"]
        ck.add("demo_mandate_bases_agree", float(b.max() / b.min() - 1) <= 0.05,
               "implied natural-gas base of the official volumes (volume / %): "
               + ", ".join(f"{y}: {v / 1e6:.2f} M m³/d" for y, v in zip(off["year"], b, strict=True)),
               [round(float(v)) for v in b], severity="warn")
    snaps_p = d / "processed" / "plants_biomethane_snapshots.parquet"
    snaps = pd.read_parquet(snaps_p) if snaps_p.exists() else pd.DataFrame()
    if len(snaps):
        cur = snaps.iloc[-1]
        for r in t.itertuples():
            ck.add(f"demo_mandate_{r.year}_coverage", True,
                   f"{r.year} ({r.volume_source}): produced {cur['produced_m3d'] / r.volume_m3d:.0%}, authorised "
                   f"{cur['authorised_m3d'] / r.volume_m3d:.0%} of {r.volume_m3d / 1e3:,.1f} thousand m³/d "
                   f"(ANP {cur['snapshot']})", round(float(cur["produced_m3d"] / r.volume_m3d), 4))
    # stock: N1 within reach
    year = int(_spec(cfg).get("year", 2024))
    n1 = pd.read_parquet(d / "processed" / "n1_municipal.parquet",
                         columns=["geocode", "year", "crop", "variant", "n1_ref_nm3", "n1_min_nm3", "n1_max_nm3"])
    n1 = n1[(n1["year"] == year) & (n1["variant"] == "obs") & n1["crop"].isin(N1_CROPS)]
    n1 = n1.groupby("geocode")[["n1_ref_nm3", "n1_min_nm3", "n1_max_nm3"]].sum().reset_index()
    hist = pd.read_parquet(d / "processed" / "reach_hist.parquet")
    st = pd.concat([stock_within_reach(hist, n1, s["pair"], s.get("radii_km", [30]))
                    for s in spec.get("stock", [{"pair": "all_crops|biomethane", "radii_km": [30, 100]}])],
                   ignore_index=True)
    bad = st[(st["n1_min_nm3_yr"] > st["n1_ref_nm3_yr"] + 1e-6) | (st["n1_ref_nm3_yr"] > st["n1_max_nm3_yr"] + 1e-6)
             | (st["n1_ref_nm3_yr"] > st["n1_total_ref_nm3_yr"] + 1e-6)]
    ck.add("demo_stock_within_envelope", bad.empty, "min ≤ ref ≤ max ≤ total for every pair and radius", len(bad))
    for r in st.itertuples():
        ck.add(f"demo_stock_{r.pair}_{r.radius_km:g}km", True,
               f"N1 within {r.radius_km:g} km: {r.n1_ref_m3d / 1e6:.2f} M Nm³ CH4/d ({r.n1_min_m3d / 1e6:.2f}–"
               f"{r.n1_max_m3d / 1e6:.2f}); {r.n1_ref_nm3_yr / r.n1_total_ref_nm3_yr:.1%} of the national N1",
               round(float(r.n1_ref_m3d), 0))
    return {"mandate_targets": t, "mandate_stock": st}


def run(cfg, ck=None) -> dict[str, Path]:
    """`bbp demo`."""
    from bbp.checks import CheckLog

    ck = ck if ck is not None else CheckLog("demo")
    need = [cfg.data_dir / "processed" / f"{n}.parquet" for n in ("reach_units", "reach_hist", "n1_municipal")]
    if not ck.add("demo_inputs", all(p.exists() for p in need), "reach_units, reach_hist, n1_municipal "
                  "(run bbp run, bbp reach)"):
        return {}
    out = cfg.data_dir / "processed"
    written = {}
    for name, df in {**typology(cfg, ck), **mandate(cfg, ck)}.items():
        p = out / f"demo_{name}.parquet"
        df.to_parquet(p)
        written[name] = p
    return written


__all__ = ["closure", "replace_zeros", "clr", "cluster", "mandate_targets", "stock_within_reach", "run"]
