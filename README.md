# Biomass Brasil Pipeline (`bbp`)

**Same fields, different numbers.** An open, annually updatable pipeline that
compares the two sources of crop *activity data* used in Brazilian biomass
inventories — satellite land cover (**MapBiomas**) and official agricultural
statistics (**IBGE PAM**) — for all 5,571 municipalities, from 2008 onwards,
and carries the difference through to the theoretical biomethane potential (N1)
of crop residues.

Developed at the Centro Paulista de Estudos de Biogás e Bioprodutos (CP2b),
NIPE – Unicamp, as the reproducible basis of the paper prepared for *Geocarto
International* (special issue FOSS4G Europe 2026). The data products feed the
PILAR-2b platform.

```
 MapBiomas Col. 11 ─┐                      ┌─ agreement by scale (H1)
 2nd-crop rasters  ─┤                      ├─ divergence predictors (H2)
 IBGE PAM 5457/839 ─┼─ ingest → harmonise → area → reconcile → inventory → publish
 IBGE mesh 2025    ─┘                                          │          │
 CP2b factors ─────────────────────────────────────────────────┘          └─ CSV · GeoPackage · manifest
```

## What it answers

| Hypothesis | Question | Stage | Main output |
|---|---|---|---|
| H1 – scale | Do the sources agree in aggregate and diverge per municipality? | `reconcile` | `agreement_by_scale` (municipality → AMC → intermediate region → state → nation) |
| H2 – structure | Is the divergence systematic and explainable? | `reconcile` | `unit_ratios_*`, `predictors` |
| H3 – consequence | Does switching source move N1 and the municipal ranking as much as the conversion-factor uncertainty does? | `inventory` | `n1_municipal`, `source_vs_factor_effect`, `rank_shift` |

## Install

Python ≥ 3.10.

```bash
git clone https://github.com/aikiesan/Biomass_Brasil_Pipeline.git
cd Biomass_Brasil_Pipeline
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

## Configure the inputs

Raw data are **not** in the repository. `config/pipeline.yaml` lists every
input relative to `raw_root` (default: the folder that contains the
repository). To point elsewhere, create `config/paths.local.yaml` (git-ignored):

```yaml
raw_root: "C:/Users/<you>/Documents/Reposicionamento_Submissão_ESD"
```

Then check that every file is found:

```bash
bbp info
```

| Input | Source | Used for |
|---|---|---|
| `BR_Municipios_2025.shp` | IBGE Malha Municipal 2025 | units, label grid, maps |
| `MAPBIOMAS_BRAZIL-COL.11-BIOME_STATE_MUNICIPALITY.xlsx` | MapBiomas Collection 11 statistics | observed area, 2008–2025 |
| `*_agriculture_agricultural_use_second_crop_*.tif` | MapBiomas second-crop layer (beta) | observed 2nd-season corn, 2008–2024 |
| `tabela5457*.csv` | IBGE SIDRA, PAM table 5457 | declared area and production |
| `PAM_tabela839_milho_por_safra.xlsx` | IBGE SIDRA, PAM table 839 | corn by season |
| `config/conversion_factors.yaml` | CP2b canonical parameters | g, TS, VS/TS, BMP (min/ref/max) |

See [`docs/data_sources.md`](docs/data_sources.md) for versions, access dates
and how to re-export the SIDRA tables.

## Run

```bash
bbp run ingest                 # parse every source once into data/interim/
bbp second-crop                # raster zonal statistics (long; resumable; run once per new raster year)
bbp run --from harmonise       # harmonise → area → reconcile → inventory → publish
```

`bbp run` alone runs every tabular stage in order. `bbp check-factors` confirms
that `conversion_factors.yaml` still matches the CP2b canonical parameter file.

Every stage records **invariant checks** in `data/logs/checks_<stage>.json`, for example:

* MapBiomas municipal area is constant across years (|drift| < 0.1 %);
* each crop class still carries the expected label (guards against class renumbering between collections);
* PAM table 839 seasons add up to the table total;
* second-crop national totals match the series published by MapBiomas (± 2 %);
* every aggregation scale sums to the same national total;
* where both sources give the same area, both N1 variants are identical.

A failed `error` check stops the run; `warn` checks are reported and shipped in the release manifest.

## Docker and the database

```bash
copy .env.example .env                   # once (cp on Git Bash)
docker compose up -d db                  # local PostGIS 15 on localhost:5433
docker compose run --rm bbp run          # the pipeline in a pinned environment
docker compose run --rm bbp load         # load results into schema bbp as a new release
bbp load --target prod --yes --keep 2    # publish to the production database (PILAR-2b)
```

`bbp load` writes only inside schema `bbp`, versions every load in
`bbp.release`, and exposes the current release through `bbp.v_*` views for
PILAR-2b. See [`docs/deployment.md`](docs/deployment.md) for Supabase, the
Unicamp VM (SSH tunnel) and the read-only role.

## Outputs

| Table (`data/processed/`) | Grain | Content |
|---|---|---|
| `crop_area` | municipality × year × crop | `a_obs_ha` (MapBiomas), `a_dec_ha`, `a_harv_ha`, `prod_t` (PAM) + SIDRA flags |
| `agreement_by_scale` | scale × crop × year | ratio of totals, log-ratio percentiles, share within ±10/25 %, Lin's CCC, Spearman ρ |
| `unit_ratios_<scale>` | unit × crop × year | A_obs, A_dec, ratio, log-ratio, presence class |
| `predictors` | municipality × year | municipal area, dominant biome, crop share, double-crop intensity, new-municipality flag |
| `n1_municipal` | municipality × year × crop × variant | N1 (Nm³ CH₄/yr) at ref/min/max and one-at-a-time bounds |
| `source_vs_factor_effect` | crop × year | \|ln(N1_obs/N1_dec)\| vs ln(N1_max/N1_min) and per-factor widths |
| `rank_shift` | year | Spearman ρ and top-decile turnover of the municipal N1 ranking |

`bbp run publish` writes `data/release/bbp_v<version>_<date>/` with gzipped CSVs,
a GeoPackage for the latest common year and `manifest.json` (git commit,
configuration, SHA-256 of every input and output, check summary) — ready to
deposit on Zenodo/REDU.

## Repository layout

```
config/       pipeline.yaml · crops.yaml · conversion_factors.yaml · municipality_changes.yaml
src/bbp/      ingest/ (sources) · harmonise · area · reconcile · inventory · publish · checks · cli
tests/        synthetic-data unit tests (run in CI; no source data needed)
docs/         methods · data sources · decisions log · deployment
docker/       initdb scripts for the local PostGIS · roles.sql for production
Dockerfile · compose.yaml · .env.example
data/         generated, git-ignored
```

## Scope

In: crop *stocks* per year (area, production) and N1 for crop residues.
Out (other CP2b work): the N2–N4 availability cascade, logistics and hubs
(ESD paper); land-use transitions and iLUC matrices (ABIOVE project).

## Credits and licence

MapBiomas data © Projeto MapBiomas, used under CC-BY 4.0 — cite the collection
and the second-crop layer as indicated on mapbiomas.org. IBGE data are public.
Conversion factors from the CP2b canonical parameter set.

Funding: FAPESP 2024/01112-1 (CP2b) and 2025/08745-2.

Code licensed under **GPL-3.0** (see `LICENSE`). Cite via `CITATION.cff`.
