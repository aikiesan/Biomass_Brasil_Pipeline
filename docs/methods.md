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
* **Cotton**: coverage class 62 (first-season cotton) **plus** second-season
  cotton from the second-crop rasters (pixel 62), because PAM reports one
  cotton area for both seasons. A year is observed only when both sources cover
  it, so cotton has no observed area in 2025 until a 2025 second-crop raster
  exists. Cotton has no CP2b residue factor: area comparison only.
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

## 3. Declared area and production (PAM) and livestock (PPM)

Every SIDRA table is listed under `sidra:` in `config/pipeline.yaml` with the
exports that make up its series. The ingest reads them in the order listed;
where two exports cover the same municipality × year × product × variable, the
one listed later wins. Each table is written to `interim/sidra/t<table>.parquet`
(long: geocode, year, product, variable, unit, value, flag, preliminary,
source_file). Rendimento médio (yield) is dropped everywhere: it is derived.

* Table 5457: area planted or destined for harvest, area harvested, quantity
  produced (2008–2025), seven products (the 2025 export has all 86; only the
  seven of the earlier exports are kept so that every year covers the same set).
* Table 839: corn by season (1st/2nd), same variables (2008–2025).
* Tables 1612 (temporary crops: sugarcane, soybean, corn, rice, herbaceous
  cotton) and 1613 (permanent crops: coffee total/arábica/canephora, orange),
  2008–2025. 1613's "área destinada à colheita" is stored as `area_planted_ha`,
  as in 5457. Cotton's declared area comes from 1612 (it is not in the 5457
  exports). Arábica + canephora are checked not to exceed the coffee total.
* PPM: herds (3939), animal products (74), milked cows (94), aquaculture (3940,
  from 2013), 2008–2025. Units are kept per row (`unit`), taken from the
  variable text or, for 74 and 3940, from the product text. "Total" of 74 and
  3940 is not ingested (it adds incompatible units). "Suíno - matrizes" and
  "Galináceos - galinhas" are subsets of their totals and must never be added
  to them (checked). The fish aggregate of 3940 ("Peixes") is empty in every
  year, so fish totals must be summed from the species (a warning is raised if
  IBGE starts filling it).
* **Preliminary year.** IBGE publishes the newest year of every table as
  preliminary and revises it at the next release. The ingest sets
  `preliminary = True` for that year, carries it into `crop_area`
  (`dec_preliminary`), and warns when the year before it still comes from an
  export in which it was the newest (i.e. the preliminary values, possibly
  revised since).
* The area comparison uses PAM tables 5457 and 839 and any table named in
  `crops.yaml` (currently 1612, for cotton): `interim/pam.parquet`.
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
  (g on a dry-matter basis, TS = 1), coffee husk, citrus pulp. Rice and cotton
  have no residue in the CP2b parameter set and enter the area comparison only.
* Factor uncertainty: all-min, all-max and one-at-a-time (g, TS, VS/TS, BMP at
  min/max, others at ref).
* Source vs factor effect (per crop and year): median and p90 of
  |ln(N1_obs/N1_dec)| across municipalities, compared with ln(N1_max/N1_min)
  and the OAT widths. Rank shift: Spearman ρ and top-decile turnover of total
  municipal N1, using only municipality × crop cells present in both variants.
* National N1 totals are computed but are **not** a result of the paper.

## 7. Municipal sanitation (platform feed, not part of the paper)

`bbp run sanitation` builds one series of urban solid waste, sewage and
population per municipality from the two federal sanitation surveys. It feeds
the PILAR-2b platform (`bbp.v_sanitation`); no crop stage reads it.

* **Sources.** SNIS "Consolidado por Município" 2008–2022 (one CSV per year)
  and SINISA 2023–2024 (waste module: *Formulário Manejo*, municipal sheet;
  sewage module: *Gestão Técnica de Esgoto*, municipal base). SINISA replaced
  SNIS from reference year 2023; the two share no field codes.
* **Harmonised variables.** `config/sanitation.yaml` maps one field of each
  survey onto each variable (e.g. SNIS CO119 and SINISA GTR1028 →
  `msw_collected_t`) and records the evidence: the SNIS indicator that
  SINISA's glossary pairs with the SINISA indicator built on that field.
  Pairs are `equivalent` (same quantity) or `similar` (same intent, different
  scope, written down); four SNIS fields have no SINISA counterpart per
  municipality (`snis_only`).
* **Keys.** SNIS is keyed by the 6-digit municipality code in every year (its
  7-digit column is empty before ~2019); the code is resolved through the
  mesh, whose 6-digit prefixes are unique. SINISA is keyed by the 7-digit code.
* **Blank is not zero.** Reporting is voluntary; only reported cells become
  rows. A reported zero stays a zero.
* **Checks.** Municipal sums reproduce each survey's own totals: SNIS's
  "TOTAL da AMOSTRA" row (population, CO111, CO119, ES006) and SINISA's
  consolidated UF / macro-region / Brazil sewage sheets (GTE1002, GTE1014), to
  1 %. The number of SINISA waste respondents equals the one in SINISA's
  report. CO119 = CO111 + CO115 and GTR1028 = GTR1025 + GTR1027 are checked;
  breaks are reported and kept as published.
* **Series break.** For every variable both surveys measure,
  `sanitation_continuity` compares the last SNIS year with the first SINISA
  year over the municipalities that report both: median and p10/p90 of the
  ratio SINISA/SNIS, ratio of totals, share within ±25 %. A median outside
  `continuity_band` (0.8–1.25) is a warning.
