# Deployment: from the pipeline to PILAR-2b

```
 your computer (Docker Desktop or a venv)          database                     web VM (Debian, nginx + PM2)
 ┌───────────────────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────────┐
 │ raw data (1.2 GB rasters, xlsx, csv)  │   │ schema bbp               │   │ PILAR-2b backend (FastAPI)   │
 │ bbp run · bbp second-crop · publish   │──▶│  release (versions)      │◀──│  reads bbp.v_* views         │
 │ bbp load --target prod --yes          │   │  crop_area, n1_…, …      │   │ PILAR-2b frontend (Next.js)  │
 │ local PostGIS (compose, :5433)        │   │  v_* = current release   │   │                              │
 └───────────────────────────────────────┘   └──────────────────────────┘   └──────────────────────────────┘
          heavy compute, once per update         Supabase today; or           serves only — never touches
                                                 PostgreSQL on the VM         the raw data
```

The pipeline never runs on the web VM. It computes on your machine and
*loads* finished tables; the platform only reads them.

## 1. Local stack (Docker Desktop)

```bash
copy .env.example .env                 # Windows (cp on Git Bash); edit if needed
docker compose build                   # once, and after changing pyproject.toml
docker compose up -d db                # PostGIS 15 on localhost:5433, data in a named volume
docker compose run --rm bbp info       # inputs found?
docker compose run --rm bbp run        # full pipeline inside the container
docker compose run --rm bbp load --with-geometry --simplify 0.0005
docker compose run --rm bbp db-status
docker compose run --rm --entrypoint pytest bbp   # tests inside the image
```

* Raw data: the parent folder of the repository, mounted read-only at
  `/data/raw` (change with `BBP_RAW_HOST` in `.env`).
* Outputs go to `./data` on the host, exactly as when running without Docker.
* `src/` and `config/` are mounted, so code/YAML edits apply without rebuilding.
* Open the local database in QGIS or DBeaver: host `localhost`, port `5433`,
  database `cp2b_maps`, user `bbp`, password from `.env`. With
  `--with-geometry`, `bbp.municipality_geom` joins to every table on `geocode`.

Without Docker the same commands work from the venv (`bbp load` then reads
`BBP_DATABASE_URL` from `.env`, which points to the compose database on 5433).

## 2. What `bbp load` writes

| Object | Content |
|---|---|
| `bbp.release` | one row per load: version, label, git commit, years, row counts, publish manifest (input SHA-256, checks), `is_current` |
| `bbp.<table>` | `municipality_attributes`, `crop_area`, `agreement_by_scale`, `unit_ratios` (all scales, `scale` column), `predictors`, `n1_municipal`, `source_vs_factor_effect`, `rank_shift` — each with `release_id` |
| `bbp.v_<table>` | the same tables filtered to the current release — **what PILAR-2b reads** |
| `bbp.municipality_geom` | optional (`--with-geometry`), for local use; production already has municipalities |

* One load is one transaction: readers see the old release until the new one is complete.
* Releases are never overwritten. `--keep N` deletes all but the newest N (cascade).
* Default is compact: rows where a municipality has neither observed nor declared
  area/production for a crop are skipped, and the AMC copy of the unit ratios is
  omitted (≈ 225 MB per release instead of ≈ 600 MB). `--full` loads everything.
* Columns are added automatically when a new version adds them; none are dropped.

## 3. Production database

### Option A — Supabase (current PILAR-2b database)

1. In the Supabase SQL editor, run `docker/sql/roles.sql` (edit the passwords).
   If the backend uses an existing role, grant that role instead (see the end of the file).
2. In `.env`: `BBP_PROD_DATABASE_URL=postgresql://bbp_loader:<pw>@<project>.pooler.supabase.com:5432/postgres?sslmode=require`
3. `bbp load --target prod --yes --keep 2 --label "Col11 + PAM 2024"`
4. Size: the free plan caps the database at 500 MB — use `--keep 1` there.
5. Only if the frontend queries Supabase's REST API directly: add `bbp` to
   *Settings → API → Exposed schemas*. The FastAPI backend does not need this.

### Option B — PostgreSQL + PostGIS on the Unicamp VM (no Supabase)

On the VM (Debian 12), once:

```bash
sudo apt install postgresql-15 postgresql-15-postgis-3
sudo -u postgres createdb cp2b_maps
sudo -u postgres psql -d cp2b_maps -c "CREATE EXTENSION postgis"
sudo -u postgres psql -d cp2b_maps -f docker/sql/roles.sql
```

PostgreSQL stays bound to `localhost` on the VM (no port opened). From your
computer, open a tunnel through the SSH port you already use, then load:

```bash
ssh -N -p <ssh-port> -L 5434:localhost:5432 <user>@<vm-host>    # leave it open
# .env:  BBP_PROD_DATABASE_URL=postgresql://bbp_loader:<pw>@localhost:5434/cp2b_maps
bbp load --target prod --yes --keep 3
# or from Docker: ...@host.docker.internal:5434/cp2b_maps
```

Port 5434 is deliberate: anything that is not the compose database (5433)
requires `--yes`, so a tunnel to production can never be written by accident.
Back up with `pg_dump -n bbp` (or the whole database) from a cron job on the VM.

## 4. Reading from PILAR-2b

```sql
-- observed vs declared soybean area, current release, one year
SELECT geocode, a_obs_ha, a_dec_ha
FROM bbp.v_crop_area WHERE crop = 'soybean' AND year = 2024;

-- municipal log-ratio map layer joined to the platform's municipality table
SELECT m.*, u.log_ratio
FROM public.municipalities m
JOIN bbp.v_unit_ratios u ON u.scale = 'municipality' AND u.unit = m.ibge_code::text
WHERE u.crop = 'soybean' AND u.year = 2024;

-- which release is live, and from which commit
SELECT release_id, label, git_commit, loaded_at FROM bbp.release WHERE is_current;
```

(`public.municipalities.ibge_code` is illustrative — use the platform's actual
table and key.)
