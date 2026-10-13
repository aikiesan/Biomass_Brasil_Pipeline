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
| IBGE PAM | `tabela5457.csv` (2008–2014), `tabela5457 (1).csv` (2015–2024), `SIDRA_2025/tabela5457_PAM_2025.xlsx` | SIDRA table 5457 | 2025 file: 2026-10-13 | variables: planted, harvested, production (+ value to 2024); products kept: rice, coffee, sugarcane, orange, corn, soybean, wheat |
| IBGE PAM corn by season | `PAM_tabela839_milho_por_safra.xlsx` (2008–2024), `SIDRA_2025/tabela839_PAM_2025.xlsx` | SIDRA table 839 | 2025 file: 2026-10-13 | SP 2009–2010 not available by season (IBGE note 3) |
| IBGE PAM temporary crops | `PAM_1612_1613_raw/TABELA_1612_*.xlsx` (2008–2024), `SIDRA_2025/tabela1612_PAM_2025.xlsx` | SIDRA table 1612 | 2025 file: 2026-10-13 | products kept: sugarcane, soybean, corn, rice, herbaceous cotton |
| IBGE PAM permanent crops | `PAM_1612_1613_raw/TABELA_1613_*.xlsx` (2008–2024), `SIDRA_2025/tabela1613_PAM_2025.xlsx` | SIDRA table 1613 | 2025 file: 2026-10-13 | products kept: coffee (total, arábica, canephora), orange. `TABELA_1613_2016_A_2020.xlsx` holds 2016–2019 only |
| IBGE PPM herds | `TABELA_3939_PPM_2008A2024.xlsx`, `SIDRA_2025/tabela3939_PPM_2025.xlsx` | SIDRA table 3939 | 2025 file: 2026-10-13 | head on 31 Dec; sow data from 2013 |
| IBGE PPM animal products | `TABELA_74_2008A2024.xlsx`, `SIDRA_2025/tabela74_PPM_2025.xlsx` | SIDRA table 74 | 2025 file: 2026-10-13 | units in the product name; "Total" not ingested |
| IBGE PPM milked cows | `TABELA_94_VACAS_ORDENHADAS_PPM_2008_A2024.xlsx`, `SIDRA_2025/tabela94_PPM_2025.xlsx` | SIDRA table 94 | 2025 file: 2026-10-13 | no product dimension |
| IBGE PPM aquaculture | `TABELA_3940_2008A2024.xlsx` (2013–2024), `SIDRA_2025/tabela3940_PPM_2025.xlsx` | SIDRA table 3940 | 2025 file: 2026-10-13 | fish aggregate empty: sum the species |
| CP2b conversion factors | `conversion_factors.yaml` (copy of canonical v1.0 + Método CP2b v5.1) | — | — | `bbp check-factors` compares with the canonical YAML |

## Re-exporting SIDRA tables

The parser reads SIDRA exports as produced by the SIDRA web interface (CSV or
XLSX, all variables in one file, municipalities in rows, *year × product* in
columns). It is robust to the trailing-cell trimming of the merged year row.
For a new year (e.g. PAM/PPM 2026), export each table for the new year **and
the year before it** (IBGE revises the previous, preliminary year at every
release), save the files under `04_IBGE_Tabelas/SIDRA_<year>/`, append them
to that table's `files` in the `sidra:` section, and rerun
`bbp run --from ingest`. Where two files cover the same municipality × year ×
product × variable, the file listed later wins, so list the newest export
last. Level: Município; municipalities in rows, *year × product* in columns;
CSV or XLSX. The parser also reads tables with no product dimension (e.g. 94).

## Known source issues found by the checks

* Table 839 total vs table 5457 corn: 1,206 of 97,613 municipality-years (2008–2025) differ
  by more than 1 ha (IBGE revisions between tables) — reported, not corrected.
* 107 municipalities (outside the listed new ones) have at least one year where every PAM 5457 cell is `...`
  (no statistics published); listed in `processed/pam_gaps.parquet`.
* Fernando de Noronha (2605459) is not in the MapBiomas municipal statistics.
* PPM 3939: 4 municipality-years where a subset exceeds its total: sows above
  all swine in Itaquara-BA 2021; hens above all poultry in Primavera-PE 2020,
  Japeri-RJ 2017 and Andradina-SP 2023 — reported, not corrected.
* PPM 3940: the fish aggregate ("Peixes", "Peixes (Quilogramas)") is empty in
  every year; sum the species.
* PAM 2025: Divinópolis de Goiás (5208301) has no statistics ("...") for 2025.
* The 2024 values of every table come from exports in which 2024 was the
  newest, preliminary year: re-export 2024 (decisions, open item 8).
