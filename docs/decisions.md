# Decisions log

One line per decision, newest first. Open items are at the end.

## Taken

| Date | Decision | Why |
|---|---|---|
| 2026-10-13 | Observed area for coverage classes = MapBiomas Collection 11 municipal statistics (tabular), not our own raster count | Matches MapBiomas's published figures; fast; geocode-keyed |
| 2026-10-13 | Second-season corn from the second-crop rasters, geodesic area per pixel (WGS84), computed by the pipeline | No municipal table published for this layer; area method is reproducible and checked against the national series |
| 2026-10-13 | Period 2008–2025 for MapBiomas; PAM to 2024 | Collection 11 adds 2025; PAM 2025 not yet ingested |
| 2026-10-13 | Boa Esperança do Norte (5101837): parents Nova Ubiratã (~80 %) and Sorriso (~20 %), installed 2025 | STF decision Oct 2023; first PAM year 2025 |
| 2026-10-13 | Yield always from PAM; N1_obs = A_obs × Y_PAM | Isolates the area divergence (outline §2.7) |
| 2026-10-13 | Sugarcane residue = straw only (field residue); bagasse/filter cake/vinasse excluded | Industrial residues are not tied to field area |
| 2026-10-13 | Rice: area comparison only | No rice residue in the CP2b parameter set |
| 2026-10-13 | Cells missing in either source are dropped from both before aggregation | Like-with-like totals at every scale |

## Open (from the paper outline)

1. Reconciled variant (N1_rec, IPOPT) — include or stay with declared vs observed?
2. Census of Agriculture 2017 as a third reference for one year?
3. Second-crop layer verification: mention the proprietary cross-check or use only PAM 839 + MapBiomas accuracy?
4. Final crop list (cotton: MapBiomas class 62 exists but PAM 5457 export has no cotton — export table 5457 with "Algodão herbáceo" if wanted).
5. Area method: geodesic (implemented for the rasters) vs EPSG:5880 — report the Polyconic bias per biome as a methods check (not implemented yet).
6. H2 predictors still to add: share of crop area near the municipal border; Landsat valid-observation count (cloud proxy).
7. Outputs still to add from outline §2.8: PostGIS load, COG + STAC catalogue.
