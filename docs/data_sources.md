# Data sources

Record the access date and version of every input here when you add or
refresh it. The release `manifest.json` stores the SHA-256 of each file, so
a later run can prove it used the same bytes.

| Source | File(s) | Version / period | Access | Notes |
|---|---|---|---|---|
| IBGE Malha Municipal | `BR_Municipios_2025.shp` | 2025 | — | 5,573 features (5,571 municipalities + 2 lagoons) |
| MapBiomas Brasil statistics | `MAPBIOMAS_BRAZIL-COL.11-BIOME_STATE_MUNICIPALITY.xlsx`, sheet `COVERAGE_11` | Collection 11, 1985–2025, published Aug 2026 | — | keyed by geocode; one row per municipality × biome × class |
| MapBiomas legend | `legend_code_mapbiomas_brazil_collection_11.csv` | Collection 11 | — | class 13 (Herbaceous-Shrub Mosaic) appears in the statistics but not in this CSV; labels are taken from the sheet itself |
| MapBiomas second crop (beta) | `{year}_agriculture_agricultural_use_second_crop_*.tif` | 2008–2024, 30 m, EPSG:4326, 131 072 × 131 072 px | — | pixel 1 = corn, 62 = cotton, 41 = other / cover crops |
| MapBiomas second-crop national series | `Série temporal de Agricultura • Uso agrícola - Segunda safra por classe • 2000 - 2024.csv` | platform export | — | invariant check only |
| IBGE PAM | `tabela5457.csv` (2008–2014), `tabela5457 (1).csv` (2015–2024) | SIDRA table 5457 | — | variables: planted, harvested, production, value; products: rice, coffee, sugarcane, orange, corn, soybean, wheat |
| IBGE PAM corn by season | `PAM_tabela839_milho_por_safra.xlsx` | SIDRA table 839, 2008–2024 | — | SP 2009–2010 not available by season (IBGE note 3) |
| CP2b conversion factors | `conversion_factors.yaml` (copy of canonical v1.0 + Método CP2b v5.1) | — | — | `bbp check-factors` compares with the canonical YAML |

## Re-exporting SIDRA tables

The parser reads SIDRA exports as produced by the SIDRA web interface (CSV or
XLSX, all variables in one file, municipalities in rows, *year × product* in
columns). It is robust to the trailing-cell trimming of the merged year row.
When PAM 2025 is published, export tables 5457 and 839 for 2025 and append
the files to `inputs.pam_5457_csv` / `inputs.pam_839_xlsx` (both accept one
file or a list), then rerun `bbp run --force`. Where two files cover the same
municipality × year × product × variable, the file listed later wins, so list
the newest export last.

## Known source issues found by the checks

* Table 839 total vs table 5457 corn: 1,206 of ~92,000 municipality-years differ
  by more than 1 ha (IBGE revisions between tables) — reported, not corrected.
* 106 municipalities have at least one year where every PAM 5457 cell is `...`
  (no statistics published); listed in `processed/pam_gaps.parquet`.
* Fernando de Noronha (2605459) is not in the MapBiomas municipal statistics.
