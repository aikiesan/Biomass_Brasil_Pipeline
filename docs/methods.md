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

### Agreement on the change

The sources can agree on the level and still tell different stories of change.
Each source is averaged over two windows (`reconcile.trend_windows`, default
2008–2012 and 2020–2024), at every scale. A unit is scored only when both sources
report every year of both windows, so a year dropped for a mesh mismatch never
biases one window, and when all four window means reach 100 ha
(`trend_min_area_ha`). The change of each source is Δlog = ln(mean late / mean early).

* direction class per unit: both up, both down, map up and statistics down, or
  the reverse (an exact zero change is "flat" and left out of the four shares);
* per scale × crop (`trend_agreement`): share with the same sign, share in each
  class, Spearman ρ and Lin's CCC of Δlog, median Δlog per source, and Δlog of
  the summed area of the scored units (the story each source tells for the set);
* by size of the change (`trend_by_magnitude`): units binned by the mean |Δlog|
  of the two sources at ln 1.10, ln 1.25, ln 1.50 (10, 25, 50 %), with the
  direction agreement and the share of the late declared area in each bin. Is
  the disagreement where little changed?
* against land-cover change (`landcover_change`, `trend_landcover`): the change
  in the municipal share (pp, MapBiomas) of natural vegetation (the Fig 1 group),
  pasture (15), mosaic of uses (21), other crops and forest plantation, read
  against the trend gap Δlog(map) − Δlog(PAM). Per crop and cover: Spearman ρ
  over the municipalities, and quartiles of the cover change (1 = lost most)
  with their median gap and direction agreement.

Checks: the windows are ordered and inside the period, the direction shares sum
to at most one, the nation is one unit per crop, the magnitude bins cover the
whole area, and the cover changes of a municipality sum to at most 100 pp.
Exploratory figures: x09 (level vs change), x10 (trend maps, soybean and
second-season corn), x11 (soybean → pasture → native vegetation chain).

## 5. Explaining divergence (H2)

Response: the municipal log-ratio ln(A_obs/A_dec) per crop for the reference
years (2020–2024), where both sources report the crop.

Predictors (`processed/predictors.parquet`), municipality × year:

* log municipal area; dominant biome; share of the municipality under crops
  (MapBiomas classes 20, 35, 39, 40, 41, 46, 47, 48, 62); double-crop intensity
  (observed second-season corn / observed soybean, log1p, capped at 10);
  new-municipality flag.
* **Irrigated share**: MapBiomas Col. 11 irrigation systems (center pivot and
  other systems, municipal statistics) / crop area.
* **Border share** (`bbp border-share`): share of the crop's observed area
  within 1 km (and 2 km) of a boundary with another municipality, from the
  Col. 11 coverage rasters. The mesh is rasterised onto each raster grid;
  boundary pixels have a 4-neighbour in another municipality (coastline and
  national border excluded); distances are Euclidean in metres (pixel size at
  the window's central latitude) in windows with a halo wider than 2 km; areas
  are geodesic per pixel row. The run is rejected when the raster does not
  reproduce the Col. 11 municipal statistics (median |raster/stats − 1| ≤ 0.5 %).
* **Clear Landsat observations** (cloud proxy): mean number of clear Landsat
  5/7/8/9 Collection 2 observations per crop pixel and year (QA_PIXEL: no fill,
  dilated cloud, cirrus, cloud, shadow or snow), from Earth Engine.

Models (`bbp run analyse`, fixed seed):

* **Mixed model** (statsmodels MixedLM, REML): standardised numeric predictors,
  biome and year as fixed effects, random intercept for the intermediate
  region. Marginal and conditional R² after Nakagawa & Schielzeth (2013).
  Biome levels with < 30 municipality-years are pooled; when the region
  variance is not estimable the model falls back to OLS with standard errors
  clustered by region (recorded in `h2_model_fit.model`).
* **Random forest** (300 trees, min. 5 per leaf): R² in 5-fold cross-validation
  grouped by intermediate region (no region in both train and test) and
  permutation importance on the held-out folds.
* **Moran's I** of the mixed-model residuals averaged per municipality, queen
  contiguity, row-standardised, 999 permutations.

Outputs: `h2_mixed_effects`, `h2_model_fit`, `h2_rf_importance`, `h2_moran`.

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

## 7. Processing plants (verification 2)

`bbp run plants` builds one national register (`interim/plants.parquet`, GeoParquet; `processed/plants.parquet`).

* **Sources.** ANP ethanol plants (cane and corn) and biodiesel plants (Painel Dinâmico, points); ANP biomethane
  open data (capacity per plant and month, production per state and month); CIBiogás biogas plants, ANEEL SIGA
  biomass power plants and MAPA SIF slaughterhouses (points, through the MapBiomas infrastructure layers).
* **One row per plant function.** `family` is the function (sugarcane mill, corn ethanol, biodiesel, biogas,
  biomethane, biomass power, slaughterhouse); capacity stays in the unit of its source (`capacity_unit`). A flex
  ethanol plant (cane + corn) is one source record in two families; both rows share one `site_id`, so the plant
  is counted once as a site. Rows of other registers that are the same site are linked to it: each bagasse power
  plant joins its nearest sugarcane mill and each biogas power plant its nearest CIBiogás biogas plant, within
  `site_link_km` (2 km); an ANP biomethane plant joins the single CIBiogás GNR/biomethane plant of its
  municipality, if there is one. Linking is nearest-only, so a power plant between two mills never merges them.
* **Status.** `operating` = in operation or expanding (ANP, ANEEL, CIBiogás, SIF statuses); `status_reason`
  says why not. ANP biomethane: the plants recorded in the snapshot month (`plants.sources.anp_biomethane.snapshot`)
  operate; a plant whose last record is older is kept, not operating (e.g. Gasgrid / ZEG, São Paulo: no record
  since 06/2022). The biomethane open data has no coordinates: plants are placed at the municipality's
  representative point (`located = municipality`).
* **Biomethane snapshots** (`processed/plants_biomethane_snapshots`): per snapshot month, plants, authorised
  capacity, biogas capacity and processed biogas (m³/d) and production (m³ in the month and m³/d). The
  production file mixes two notations: a point is a decimal separator ('4324235.558') except in values of one to
  three digits followed by groups of exactly three ('173.037' = 173,037 m³), which are read as thousands and
  checked against the output implied by processed biogas.
* **Checks** (`checks_plants.json`): every source found and its cut-off recorded; coordinates present and inside
  Brazil (coastal points: nearest municipality within 5 km); each point inside the municipality the source
  declares (warning above 5 % per source); possible duplicates (same family and source, < 2 km, different sites);
  capacity present where the family needs it; count by family and source; biomethane snapshot against the
  figures quoted for it; the ANP operating mills against the CONAB 2025 and MapBiomas/EPE 2018 registers (share
  with a unit within 2 km and 10 km, `processed/plants_register_match`).

## 8. Crop area within reach of existing plants (verification 2)

`bbp reach` (30 m, resumable; `plants.reach` in `pipeline.yaml`).

* **Crops** from the MapBiomas Collection 11 coverage raster and the second-crop raster of the same year
  (2024): sugarcane (class 20), soybean (39), second-season corn (second-crop value 1, whatever the first-season
  class), all crops (coverage classes 20, 35, 39, 40, 41, 46, 47, 48, 62 or second-crop values 1, 41, 62; a pixel
  counts once). Pixel area is geodesic (WGS84, per row). Each pixel belongs to the municipality whose polygon
  holds its centre (the label grid of `bbp border-share`).
* **Pairs** (crop → family): sugarcane → sugarcane mill, second-season corn → corn ethanol, soybean → biodiesel,
  all crops → biogas, all crops → biomethane. Only operating plants count.
* **Distance** is the straight-line distance from each pixel centre to the nearest plant of the family: a great
  circle on the mean Earth sphere (R = 6371.0088 km), found with a KD tree on unit vectors. It differs from the
  WGS84 geodesic by < 0.6 % (< 200 m at 30 km; checked on random pairs), whereas a planar distance in SIRGAS 2000 /
  Brazil Polyconic would be off by up to 5.5 % at the edges of the country. Buffers are therefore geodesic
  circles; no road network is used (a declared choice: the same distance at every resolution).
* **Accumulation.** Per pair, municipality and 1 km distance bin to 150 km (one overflow bin beyond), the crop
  area (`reach_hist`); nationally, 1 km bins to 2000 km (`reach_distance`); the number of plants of the family
  within 30 km (`reach_competition`, capped at 10). The area within r km of a unit is the sum of its bins below
  r, at every level (municipality, intermediate region, state, nation: `reach_units`, with mean and median
  distance and the plants in the unit), so all levels add up to the same national total (checked).
* **Sugarcane mills** (`reach_mills`): the cane area in 1 km rings around every mill to 150 km; the cane a mill
  needs = crushing-class midpoint / 75 t ha⁻¹; required radius (smallest radius holding that area), supply
  ratio within each radius, the share of the cane within 30 km that also has another operating mill within 30 km
  (contested), and the number of other operating mills within 30 km. The same cane may supply several mills
  (as in Cervi et al. 2019); contested shares say how much.
* **Aggregation check.** The same distances measured from the centre of each pixel's 16 × 16 cell (~480 m)
  give the shares a land-cover aggregation would give (`reach_aggregation_check`); the difference is reported in
  percentage points. The cells are also written as a raster (`reach_cells_f16.tif`: crop area and area within
  30 km per crop and pair).
* **Checks** (`checks_reach.json`): every crop's area in the pass against the published statistics (Col. 11
  coverage classes; second-crop zonal statistics), share of crop area outside the mesh, every pair holding its
  whole crop, levels adding up, shares in [0, 1] and growing with the radius, the sphere vs WGS84, the
  aggregation effect, and the headline numbers (share within each radius, mills short of cane, competition).

## 9. Scale as a property of the system (MAUP)

`bbp scale` (after `bbp reach`; `plants.scale` in `pipeline.yaml`). The same land and the same plants, reach
measured at three resolutions, for every pair and for radii of 10–2000 km plus an infinite radius.

* **Pixel** (reference): each 30 m crop pixel at its own distance (`reach_hist` below 150 km, the national 1 km
  curve `reach_distance` above).
* **Municipality** and **intermediate region**: what a study with statistical units knows, the unit's crop area
  (the 30 m area summed per unit, so every resolution holds the same land) and its geometry.
  * `area_fraction` (main rule): the crop is spread over the unit like its land; the share within r km is the
    share of the unit's land within r km of the nearest operating plant. Land area comes from the ~480 m cells
    of `reach_cells_f16.tif` (cell centre in the municipality, geodesic cell area); per municipality it matches
    the IBGE area to a median of 0.15 %. A unit with no cell falls back to its centroid.
  * `centroid` (sensitivity): the unit counts whole when its centroid (taken in SIRGAS 2000 / Brazil Polyconic)
    is within r km.
  * `representative_point`: the same with a point inside the polygon, the rule of the exploratory x05.
* **Outputs:** `reach_scale` (pair × radius × resolution × rule, national) and `reach_scale_units` (per
  municipality and intermediate region).
* **Checks** (`checks_scale.json`): every resolution and rule reaches 100 % at an infinite radius; the area
  within grows with the radius; the same crop area at every resolution; the land grid against the IBGE areas;
  the headline gaps at 30 km; and the exploratory x05 (representative points, 480 m cells, April register)
  against the new values.

## 10. Using the system: typology and mandate ruler

`bbp demo` (`demo` in `pipeline.yaml`).

* **Typology.** Each intermediate region is described by the composition of its observed crop area in 2024
  (MapBiomas Col. 11 statistics and the second-crop layer): soybean, second-season corn, sugarcane, coffee,
  citrus, cotton (both seasons), rice and other crops (classes 41, 48, 35). Second-season corn is a season of its
  own, so the parts are shares of the area cropped over the two seasons. Regions with less than 10 kha of crops
  are left out ("minor agriculture"). Parts below 0.1 % of a region's crop area are rounded zeros, replaced
  multiplicatively by δ = 0.65 × 0.1 %; the compositions are transformed by the centred log-ratio (CLR), so
  Euclidean distance is the Aitchison distance. Ward's clustering on the CLR coordinates; the number of types is
  the most stable partition (largest adjusted Rand index between Ward and k-means, fixed seed) among k = 2–12
  whose smallest type holds at least 10 regions, ties within 0.02 going to the larger mean silhouette (the
  silhouette alone keeps growing with k on these data). Each region is crossed with the share of its crops within
  reach of each family (`reach_units`), its plants and its N1 (observed variant, ref factors). Priority regions:
  N1 in the top quartile and less than 25 % of all crops within 30 km of a biogas plant.
* **Mandate ruler.** Targets and volumes as stated by the acts (`demo.mandate.targets`, each with its act, URL
  and access date): 0.5 % in 2026 = 504.8 thousand m³/d (CNPE Res. 4/2026, art. 3) and 1 % in 2027 =
  995.2 thousand m³/d (MME Ordinance 942/2026, public consultation). The MME's indicative 3, 4 and 5 % for
  2028–2030 have no official volume and get one from the 2027 base (volume / %), flagged `derived`. No official
  target exists for 2035. Flows: ANP authorised capacity and production at the snapshot (§7). Stock: the N1 of
  the crop residues within r km of existing plants, per municipality N1 × the share of its crop area within r km
  (`reach_hist`), with the factor envelope — theoretical methane, drawn apart from the flows, not a supply.
* **Checks** (`checks_demo.json`): zero replacement closes every composition; CLR rows sum to zero; the chosen k
  lies inside the range and its smallest type is large enough; every official volume has its act, URL and access
  date; the bases implied by the two official volumes agree within 5 %; the stock lies within its envelope and
  below the national N1.

## 11. Municipal sanitation (platform feed, not part of the paper)

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
