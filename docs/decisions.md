# Decisions log

One line per decision, newest first. Open items are at the end.

## Taken

| Date | Decision | Why |
|---|---|---|
| 2026-10-21 | Trend agreement (H1) between two five-year windows, 2008–2012 and 2020–2024, each source averaged over the window; change = ln(late / early) | Five-year means damp single-year noise in both sources; the earliest and latest full windows of the second-crop rasters (to 2024) |
| 2026-10-21 | A unit enters the trend only with every year of both windows in both sources and ≥ 100 ha in all four means | A year dropped for a mesh mismatch would bias one window; tiny areas give large, meaningless log changes |
| 2026-10-21 | Size of the change = mean \|Δlog\| of the two sources, bins at 10, 25, 50 %; area weights from the late declared area | Tests whether the disagreement on direction sits where little changed, without favouring either source |
| 2026-10-21 | Trend gap read against the MapBiomas cover change: natural vegetation (Fig 1 group), pasture and mosaic kept apart, other crops, plantation | Pasture → crop conversion and clearing are the two stories the map may see before the statistics do |
| 2026-10-19 | Typology: CLR of the regional crop composition, Ward clustering, k by stability (Ward vs k-means ARI) with every type ≥ 10 regions; detection limit 0.1 % for zeros | On these data the silhouette grows with k down to 2-region types; stability under a minimum size gives an interpretable k (5) |
| 2026-10-19 | Mandate volumes taken as stated by the acts (CNPE Res. 4/2026 art. 3: 504.8 thousand m³/d; MME Ord. 942/2026: 995.2 thousand m³/d); 2028–2030 derived from the 2027 base and labelled indicative | No conversion needed; the exploratory 0.31 M m³/d made the 2026 coverage look 1.63× larger than it is (504.8 / 310) |
| 2026-10-19 | Mandate stock = N1 within reach of existing plants (theoretical), never added to the flows | Stock and flow are different quantities; N2–N4 stay out of this paper |
| 2026-10-19 | Scale (MAUP): main rule = area fraction (a unit's crop spread like its land); centroid and representative point as sensitivity; pixel as reference | A statistical unit knows its crop area and its geometry, not where the crop sits inside it; the point rules are what most regional studies do |
| 2026-10-19 | Unit land area from the ~480 m cells (cell centre in the municipality) | Matches the IBGE municipal areas to a median 0.15 %; a 30 m pass over all land would add hours for no visible gain |
| 2026-10-19 | Reach measured from the 30 m pixels (coverage + second crop 2024), not from aggregated cells; the ~480 m cells kept only as a check of the aggregation effect | The pixel is the reference resolution of the funnel; the check shows how much aggregation would move the result |
| 2026-10-19 | Distance = great circle on the mean sphere (straight line, geodesic buffers), not planar EPSG:5880 | Polyconic distorts a 30 km radius by up to 5.5 % at the edges of Brazil; the sphere is within 0.6 % of WGS84 everywhere |
| 2026-10-19 | Second-season corn = every corn pixel of the second-crop raster, whatever the first-season class | The register question is about all second-season corn, not only corn after soybean (14.07 vs 14.75 Mha in 2024) |
| 2026-10-19 | Flex ethanol plants: one site, two functions (cane mill and corn ethanol) | Counted once; each function keeps its own capacity and pairing |
| 2026-10-19 | ANP biomethane from the open data, snapshot 08/2026; April 2026 kept as a record; plants absent from the snapshot kept as not operating (Gasgrid/ZEG, SP, last record 06/2022) | One cut-off for the paper; nothing silently dropped |
| 2026-10-19 | Plant register at the August 2026 cut-off has 1,940 rows (1,631 sites) instead of the 1,938 rows of the exploratory register | Biomethane 22 rows (21 operating + Gasgrid) instead of 20 from the April extract; one CIBiogás point outside Brazil dropped as before |
| 2026-10-19 | Biodiesel plants under construction (ANP status Construção) do not operate | As for ethanol plants; 69 of 77 operate |
| 2026-10-19 | Reach pairs of the paper: cane → mill, 2nd-season corn → corn ethanol, soybean → biodiesel, all crops → biogas, all crops → biomethane; pasture → slaughterhouse and cane → bagasse power left out of the default | The five conclusions of verification 2; other pairs can be added in `plants.reach.pairs` |
| 2026-10-15 | N1_rec (reconciled variant) out of the paper | Keeps H3 declared vs observed; reconciliation is future work |
| 2026-10-15 | Census of Agriculture 2017 out of the paper | Not on disk; mentioned in the discussion as future validation |
| 2026-10-15 | Border share from **Collection 11** coverage rasters, reference years 2020–2024 (`bbp border-share`); the Collection 10 rasters of the ABIOVE project (ILUC_NIPE) are not used | One collection for the whole paper; the Col. 10 rasters differ from the Col. 11 statistics by 1–3 % per municipality. A guard rejects rasters that do not reproduce the Col. 11 municipal statistics |
| 2026-10-15 | Border = boundary between two municipalities; coastline and national border excluded | No neighbour to misattribute area to |
| 2026-10-15 | Cloud proxy = mean clear Landsat 5/7/8/9 C2 observations per crop pixel and year, exported from Earth Engine (`docs/gee/landsat_valid_observations.js`) | MapBiomas classifies each year from these scenes; fewer clear observations = harder classification |
| 2026-10-15 | Irrigated share from MapBiomas Col. 11 irrigation systems (municipal statistics) | Same collection as the coverage areas; checked (collection, mesh, within municipal area) |
| 2026-10-15 | H2 = mixed model (random intercept: intermediate region; R² after Nakagawa & Schielzeth) + random forest with region-grouped CV and permutation importance + Moran's I of residuals (queen, 999 permutations); fixed seed; OLS with region-clustered SE when the region variance is not estimable | Explanation, out-of-region prediction and residual spatial structure are three different questions |
| 2026-10-15 | SNIS 2008–2022 and SINISA 2023–2024 ingested by a separate `sanitation` stage into one table on harmonised variables (`config/sanitation.yaml`), loaded as `bbp.sanitation` | One parser and one set of checks for the urban series across the 2023 survey change; PILAR-2b reads `bbp.v_sanitation` as it reads `v_sidra` |
| 2026-10-15 | Field pairs SNIS↔SINISA justified by SINISA's own glossary (indicator ↔ "similar no SNIS"), labelled `equivalent` / `similar`; continuity measured, never assumed | The surveys share no codes; a silent 1:1 mapping would hide scope changes (e.g. GTR1025 includes selective routes) |
| 2026-10-15 | SNIS identity breaks (15 municipality-years, CO119 ≠ CO111 + CO115) and SINISA's (2 in 2024) kept as published, reported as warnings | Source data-entry errors; not ours to correct |
| 2026-10-15 | Sanitation is a platform feed, outside the paper's scope | The paper is about crop activity data (H1–H3) |
| 2026-10-13 | All SIDRA tables in one `sidra:` registry; exports listed in order, the later one wins on overlap | One reader for PAM and PPM; a new year is one more file, not a re-export of the series |
| 2026-10-13 | PAM/PPM 2025 (exported 2026-10-13) included and flagged `preliminary` | IBGE marks the newest year as preliminary; flag rather than wait a year |
| 2026-10-13 | PPM 3939/74/94/3940 and PAM 1612/1613 ingested | The same pipeline rebuilds the PILAR-2b municipal series (crops and livestock) |
| 2026-10-13 | Cotton added (area only): observed = class 62 + second-crop cotton; declared = PAM 1612 "Algodão herbáceo" | PAM does not split cotton by season; no CP2b cotton residue factor. TODO: confirm in the Collection 11 ATBD that class 62 maps first-season cotton only (else this double-counts) |
| 2026-10-13 | IBGE inconsistencies (4 herd subsets > totals) kept as published, reported as warnings | Not ours to correct; listed in `checks_ingest.json` |
| 2026-10-13 | Observed area for coverage classes = MapBiomas Collection 11 municipal statistics (tabular), not our own raster count | Matches MapBiomas's published figures; fast; geocode-keyed |
| 2026-10-13 | Second-season corn from the second-crop rasters, geodesic area per pixel (WGS84), computed by the pipeline | No municipal table published for this layer; area method is reproducible and checked against the national series |
| 2026-10-13 | Period 2008–2025 for MapBiomas and PAM/PPM; second-crop rasters to 2024 | No 2025 second-crop raster: corn_2nd and cotton are unobserved in 2025 |
| 2026-10-13 | Boa Esperança do Norte (5101837): parents Nova Ubiratã (~80 %) and Sorriso (~20 %), installed 2025 | STF decision Oct 2023; first PAM year 2025 |
| 2026-10-13 | Yield always from PAM; N1_obs = A_obs × Y_PAM | Isolates the area divergence (outline §2.7) |
| 2026-10-13 | Sugarcane residue = straw only (field residue); bagasse/filter cake/vinasse excluded | Industrial residues are not tied to field area |
| 2026-10-13 | Rice: area comparison only | No rice residue in the CP2b parameter set |
| 2026-10-13 | Cells missing in either source are dropped from both before aggregation | Like-with-like totals at every scale |

## Open (from the paper outline)

3. Second-crop layer verification: mention the proprietary cross-check or use only PAM 839 + MapBiomas accuracy?
4. Final crop list — cotton added as area-only (see above); does it go in the paper?
5. Area method: geodesic (implemented for the rasters) vs EPSG:5880 — report the Polyconic bias per biome as a methods check (not implemented yet).
6. H2 predictors: border share and cloud proxy implemented; waiting for the Col. 11 coverage rasters 2020–2024 and the GEE export.
7. Outputs still to add from outline §2.8: COG for the aggregated rasters, GeoParquet next to the GeoPackage (PostGIS load and STAC done).
8. Re-export 2024 of every SIDRA table: the 2024 values on disk come from exports in which 2024 was still preliminary (warning `sidra_<t>_2024_not_from_a_preliminary_export`).
9. Cut-off dates of the ANP ethanol/biodiesel exports and of the MapBiomas infrastructure layers (CIBiogás, ANEEL, SIF): not recorded yet (`cutoff: null`, warning in `checks_plants.json`).
