# Methods (as implemented)

This file describes what the code does, stage by stage. It is the reference
for §2 of the paper; any change to the method must be reflected here.

## 1. Units

* Reference mesh: IBGE Malha Municipal 2025 (5,571 municipalities; the two
  lagoon units 4300001/4300002 are kept for area checks only).
* All joins use the 7-digit IBGE geocode. MapBiomas Collection 11 statistics
  are keyed by geocode, so no name matching is needed (Collection 10 was keyed
  by name).
* Aggregation scales: municipality → AMC → intermediate region (133) → state → nation.

### Minimum comparable areas (AMC)

MapBiomas reports every year on the current mesh; PAM reports a municipality
only from its first year of existence. Six municipalities were installed in or
after the study period (`config/municipality_changes.yaml`): five in 2013 and
Boa Esperança do Norte (MT) in 2025 (split from Nova Ubiratã and Sorriso; first
PAM year is therefore 2025, i.e. not yet published). Each is merged with its
parent(s) into one AMC. At municipality scale, the new municipality and its
parents are excluded for the years before its first PAM record
(`processed/mesh_mismatch.parquet`); every coarser scale contains whole AMCs
and is unaffected.

## 2. Observed area (MapBiomas)

* **Coverage classes** (Collection 11 municipal statistics, hectares, summed
  over biomes): soybean 39, sugarcane 20, coffee 46, citrus 47, rice 40.
  Areas are those published by MapBiomas.
* **Second-season corn** from the second-crop rasters (pixel value 1 = corn;
  62 = cotton; 41 = other second-season crops / cover). Area is computed by the
  pipeline:
  1. the IBGE mesh is rasterised once onto the exact raster grid
     (EPSG:4326, 0.000269°; pixel-centre rule, one municipality per pixel);
  2. each pixel is weighted by its **geodesic area on the WGS84 ellipsoid**
     (`pyproj.Geod`), which depends only on latitude and is computed per row;
  3. areas are accumulated per municipality, class and year in 2,048-pixel
     windows.
  The national totals are checked against the series published by MapBiomas:
  for 2008–2024 the pipeline reproduces the published corn, cotton and other
  second-crop totals to better than 0.01 %, and no second-crop pixel falls
  outside the municipal mesh.
  SIRGAS 2000 (EPSG:4674) and WGS84 differ by < 1 m in Brazil; the mesh is
  used as published.
* First-season corn has no MapBiomas class (it sits inside classes 39/41) and
  is therefore declared-only: both inventory variants use PAM for it.

## 3. Declared area and production (PAM)

* Table 5457: area planted or destined for harvest, area harvested, quantity
  produced (2008–2024).
* Table 839: corn by season (1st/2nd), same variables (2008–2024).
* SIDRA symbols are preserved as flags: `-` → 0 (true zero); `...`, `..`, `X`
  → missing (never zero). Cells missing in one source are dropped from both
  sources before any aggregation, so totals at every scale compare like with like.
* Known conventions: perennials (coffee, citrus) are reported as area *destined
  for harvest*, which excludes young orchards that MapBiomas maps; PAM
  "Laranja" covers oranges only, while MapBiomas class 47 covers all citrus.

## 4. Agreement metrics (H1)

For each scale × crop × year, over units where max(A_obs, A_dec) ≥ 50 ha
(no threshold at national scale):

* ratio of totals ΣA_obs/ΣA_dec;
* log-ratio ln(A_obs/A_dec) on units with both > 0: median, mean |·|, p10, p90;
* share of units (and of declared area) with |A_obs/A_dec − 1| ≤ 10 % and ≤ 25 %
  (units with area in only one source count as outside);
* relative absolute error Σ|A_obs − A_dec| / ΣA_dec;
* Lin's concordance correlation coefficient and Spearman ρ;
* counts of units present in both sources, in one only.

## 5. Explaining divergence (H2) — inputs

`processed/predictors.parquet` (municipality × year): municipal area (log),
dominant biome, share of the municipality under crops (MapBiomas classes 20,
35, 39, 40, 41, 46, 47, 48, 62), double-crop intensity (observed
second-season corn / observed soybean), new-municipality flag. Model fitting
(mixed model, RF, Moran's I) is done in analysis notebooks on these tables,
not inside the pipeline. Border-proximity and Landsat-observation predictors
are planned (see `decisions.md`).

## 6. N1 (H3)

    N1 [Nm³ CH₄ yr⁻¹] = P × g × TS × VS/TS × BMP

* P_dec = PAM production. P_obs = A_obs × Y, with **Y = PAM production / PAM
  planted area** so that the two variants differ only through area. When a
  municipality has observed area but no usable declared yield, Y comes from the
  intermediate region, then the state, then the nation (`yield_source`).
* Residues: soybean straw, corn stover (1st and 2nd season), sugarcane straw
  (g on a dry-matter basis, TS = 1), coffee husk, citrus pulp. Rice has no
  residue in the CP2b parameter set and enters the area comparison only.
* Factor uncertainty: all-min, all-max and one-at-a-time (g, TS, VS/TS, BMP at
  min/max, others at ref).
* Source vs factor effect (per crop and year): median and p90 of
  |ln(N1_obs/N1_dec)| across municipalities, compared with ln(N1_max/N1_min)
  and the OAT widths. Rank shift: Spearman ρ and top-decile turnover of total
  municipal N1, using only municipality × crop cells present in both variants.
* National N1 totals are computed but are **not** a result of the paper.
