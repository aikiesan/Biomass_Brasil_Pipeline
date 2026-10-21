"""Load processed tables into PostgreSQL/PostGIS as a versioned release.

Design (follows the CP2b data-model rule "a new version never overwrites the
previous one; it creates a new partition"):

* everything lives in one schema (default ``bbp``); nothing outside it is touched;
* ``bbp.release`` has one row per load, with the git commit, the publish
  manifest (input hashes, checks) and row counts;
* every data table carries ``release_id`` (FK, ON DELETE CASCADE);
* ``bbp.v_<table>`` views expose the release flagged ``is_current`` — this is
  what PILAR-2b should read;
* one load = one transaction: the platform sees either the old or the new
  release, never a half-loaded one.

The target is a libpq URL taken from --url, or BBP_DATABASE_URL in the
environment or in the repository's .env file. Any host other than the local
Docker database needs an explicit --yes.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from bbp.config import Config

log = logging.getLogger("bbp")

# Hosts/ports treated as "your own local database" (no --yes needed): the compose
# service `db` from inside Docker, and the compose port 5433 from the host.
# An SSH tunnel to a production server also appears as localhost, so tunnels
# must use another port (the docs use 5434) and therefore always need --yes.
# `cp2b-db-dev` is the PILAR-2b development database (NewLook docker compose), reached
# from the `bbp-pilar2b` compose service through the newlook_default network.
LOCAL_TARGETS = {("db", 5432), ("localhost", 5433), ("127.0.0.1", 5433), ("::1", 5433), ("cp2b-db-dev", 5432)}
TARGET_VARS = {"local": "BBP_DATABASE_URL", "prod": "BBP_PROD_DATABASE_URL", "pilar2b": "BBP_PILAR2B_DATABASE_URL"}

# processed table -> (source parquet(s), index columns)
TABLES: dict[str, tuple[list[str], list[str]]] = {
    "municipality_attributes": (["municipality_attributes"], ["geocode"]),
    "crop_area": (["crop_area"], ["geocode", "year", "crop"]),
    "agreement_by_scale": (["agreement_by_scale"], ["scale", "crop", "year"]),
    "unit_ratios": (["unit_ratios_municipality", "unit_ratios_amc", "unit_ratios_rgint", "unit_ratios_uf"],
                    ["scale", "unit", "year", "crop"]),
    "predictors": (["predictors"], ["geocode", "year"]),
    "n1_municipal": (["n1_municipal"], ["geocode", "year", "crop", "variant"]),
    "source_vs_factor_effect": (["source_vs_factor_effect"], ["crop", "year"]),
    "rank_shift": (["rank_shift"], ["year"]),
    # IBGE SIDRA tables (interim/sidra/t<table>.parquet), long; the tables listed in
    # database.sidra_tables. PILAR-2b reads PAM/PPM from bbp.v_sidra.
    "sidra": ([], ["sidra_table", "year", "variable"]),
    # SNIS (to 2022) + SINISA (from 2023) on harmonised variables; PILAR-2b reads bbp.v_sanitation.
    "sanitation": (["sanitation"], ["variable", "year", "geocode"]),
    "sanitation_continuity": (["sanitation_continuity"], ["variable"]),
    # H2 results (analyse)
    "h2_mixed_effects": (["h2_mixed_effects"], ["crop"]),
    "h2_model_fit": (["h2_model_fit"], ["crop"]),
    "h2_rf_importance": (["h2_rf_importance"], ["crop"]),
    "h2_moran": (["h2_moran"], ["crop"]),
}
SIDRA_DROP_COLUMNS = ["name", "variable_raw"]  # municipality names live in the platform; raw label = variable
N1_COMPACT = ["geocode", "year", "crop", "feedstock", "variant", "production_t", "yield_source",
              "activity_declared_only", "n1_ref_nm3", "n1_min_nm3", "n1_max_nm3"]


# ---------------------------------------------------------------------------
# connection
# ---------------------------------------------------------------------------
def read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def resolve_url(cfg: Config, url: str | None, target: str = "local") -> str:
    """--url wins; otherwise the target's variable (TARGET_VARS: local, prod, pilar2b),
    from the environment or the repository's .env file."""
    var = TARGET_VARS[target]
    url = url or os.environ.get(var) or read_dotenv(cfg.repo_root / ".env").get(var)
    if not url:
        raise SystemExit(f"No database URL: pass --url or set {var} (environment or .env; see .env.example)")
    return url


def describe(url: str) -> str:
    """URL without the password, for logs."""
    p = urlparse(url)
    return f"{p.scheme}://{p.username or ''}@{p.hostname}:{p.port or 5432}{p.path}"


def is_local(url: str) -> bool:
    p = urlparse(url)
    return ((p.hostname or "localhost"), p.port or 5432) in LOCAL_TARGETS


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
def pg_type(series: pd.Series) -> str:
    kind = series.dtype.kind
    if kind == "b" or str(series.dtype) == "boolean":
        return "boolean"
    if kind in "iu" or str(series.dtype) in ("Int64", "Int32", "Int16"):
        return "bigint"
    if kind == "f":
        return "double precision"
    return "text"


# value columns that make a row "empty" when all are 0 or missing (compact mode drops those rows)
VALUE_COLS = {
    "crop_area": ["a_obs_ha", "a_dec_ha", "a_harv_ha", "prod_t"],
    "unit_ratios": ["a_obs_ha", "a_dec_ha"],
    "n1_municipal": ["production_t"],
}


def _drop_empty(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    vals = df[[c for c in cols if c in df.columns]]
    return df[(vals.fillna(0) != 0).any(axis=1)]


def collect(cfg: Config, full_n1: bool = False, compact: bool = True) -> dict[str, pd.DataFrame]:
    """Processed tables to load. Compact mode (default) drops rows where every value
    is zero or missing (crops a municipality does not grow) and the AMC copy of the
    unit ratios; --full loads everything."""
    frames: dict[str, pd.DataFrame] = {}
    sidra = collect_sidra(cfg)
    if sidra is not None:
        frames["sidra"] = sidra
    for name, (sources, _) in TABLES.items():
        if name == "sidra":
            continue
        parts = []
        for src in sources:
            if compact and src == "unit_ratios_amc":
                continue
            p = cfg.data_dir / "processed" / f"{src}.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p))
        if not parts:
            log.warning("  %s: no processed table found, skipped", name)
            continue
        df = pd.concat(parts, ignore_index=True)
        if name == "n1_municipal" and not full_n1:
            df = df[[c for c in N1_COMPACT if c in df.columns]]
        if compact and name in VALUE_COLS:
            df = _drop_empty(df, VALUE_COLS[name])
        for c in df.columns:
            if isinstance(df[c].dtype, pd.CategoricalDtype):
                df[c] = df[c].astype(str)
        frames[name] = df
    return frames


def collect_sidra(cfg: Config) -> pd.DataFrame | None:
    """The SIDRA tables named in database.sidra_tables, minus database.sidra_exclude_variables.

    Every row is kept, zeros and missing cells included: SIDRA's "-" (a measured zero) and
    "..."/"X" (no value) are different facts, and the flag column says which.
    """
    db = cfg.section("database")
    tables = db.get("sidra_tables", [])
    exclude = set(db.get("sidra_exclude_variables", []))
    parts = []
    for t in tables:
        p = cfg.data_dir / "interim" / "sidra" / f"t{t}.parquet"
        if not p.exists():
            log.warning("  sidra table %s: %s not found (run `bbp run ingest`), skipped", t, p.name)
            continue
        df = pd.read_parquet(p)
        df = df[~df["variable"].isin(list(exclude))].drop(columns=[c for c in SIDRA_DROP_COLUMNS if c in df])
        parts.append(df.rename(columns={"table": "sidra_table"}))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df


def _ensure_table(cur, schema: str, name: str, df: pd.DataFrame, index_cols: list[str]) -> None:
    cols = ",\n  ".join(f'"{c}" {pg_type(df[c])}' for c in df.columns)
    cur.execute(f'CREATE TABLE IF NOT EXISTS {schema}."{name}" (\n'
                f'  release_id integer NOT NULL REFERENCES {schema}.release(release_id) ON DELETE CASCADE,\n  {cols})')
    # schema evolution: add new columns, never drop old ones
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s",
                (schema, name))
    existing = {r[0] for r in cur.fetchall()}
    for c in df.columns:
        if c not in existing:
            cur.execute(f'ALTER TABLE {schema}."{name}" ADD COLUMN "{c}" {pg_type(df[c])}')
    idx = "".join(f', "{c}"' for c in index_cols if c in df.columns)
    cur.execute(f'CREATE INDEX IF NOT EXISTS "{name}_release_idx" ON {schema}."{name}" (release_id{idx})')


def _copy(cur, schema: str, name: str, df: pd.DataFrame, chunk: int = 200_000) -> None:
    columns = ["release_id", *df.columns.drop("release_id")]
    quoted = ", ".join(f'"{c}"' for c in columns)
    sql = f"COPY {schema}.\"{name}\" ({quoted}) FROM STDIN WITH (FORMAT csv, NULL '\\N')"
    with cur.copy(sql) as copy:
        for start in range(0, len(df), chunk):
            buf = io.StringIO()
            df.iloc[start:start + chunk][columns].to_csv(buf, header=False, index=False, na_rep="\\N")
            copy.write(buf.getvalue())


def _views(cur, schema: str, names: list[str]) -> None:
    for name in names:
        cur.execute(f'DROP VIEW IF EXISTS {schema}."v_{name}"')
        cur.execute(f'CREATE VIEW {schema}."v_{name}" AS SELECT t.* FROM {schema}."{name}" t '
                    f'JOIN {schema}.release r USING (release_id) WHERE r.is_current')


def _latest_manifest(cfg: Config) -> dict:
    rel = sorted((cfg.data_dir / "release").glob("bbp_v*/manifest.json"))
    if not rel:
        return {}
    m = json.loads(rel[-1].read_text(encoding="utf-8"))
    m.pop("config", None)  # keep the row small; the config is in git
    return m


# ---------------------------------------------------------------------------
# geometry (optional; for local inspection in QGIS — production already has municipalities)
# ---------------------------------------------------------------------------
def load_geometry(cur, cfg: Config, schema: str, simplify: float) -> int:
    import geopandas as gpd
    import shapely

    gdf = gpd.read_parquet(cfg.data_dir / "interim" / "municipalities_geom.parquet")
    gdf = gdf[gdf["is_municipality"]]
    if simplify > 0:
        gdf["geometry"] = gdf.geometry.simplify(simplify, preserve_topology=True)
    geom = shapely.force_2d(gdf.geometry.values)
    geom = [g if g.geom_type == "MultiPolygon" else shapely.MultiPolygon([g]) for g in geom]
    srid = gdf.crs.to_epsg() or 4674
    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'postgis'")
    if cur.fetchone() is None:
        cur.execute("CREATE EXTENSION postgis")
    cur.execute(f"DROP TABLE IF EXISTS {schema}.municipality_geom")
    cur.execute(f"CREATE TABLE {schema}.municipality_geom (geocode bigint PRIMARY KEY, name text, uf text, "
                f"geom geometry(MultiPolygon, {srid}))")
    wkb = shapely.to_wkb(shapely.set_srid(geom, srid), hex=True, include_srid=True)
    df = pd.DataFrame({"geocode": gdf["geocode"].to_numpy(), "name": gdf["name"].to_numpy(),
                       "uf": gdf["uf"].to_numpy(), "geom": wkb})
    with cur.copy(f"COPY {schema}.municipality_geom (geocode, name, uf, geom) FROM STDIN WITH (FORMAT csv)") as copy:
        buf = io.StringIO()
        df.to_csv(buf, header=False, index=False)
        copy.write(buf.getvalue())
    cur.execute(f"CREATE INDEX IF NOT EXISTS municipality_geom_gix ON {schema}.municipality_geom USING gist (geom)")
    return len(df)


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def load(cfg: Config, url: str, *, label: str = "", keep: int = 0, full_n1: bool = False, compact: bool = True,
         with_geometry: bool = False, simplify: float = 0.0, dry_run: bool = False) -> int | None:
    import psycopg

    import bbp
    from bbp.publish import git_commit

    schema = cfg.section("database").get("schema", "bbp")
    frames = collect(cfg, full_n1=full_n1, compact=compact)
    counts = {k: len(v) for k, v in frames.items()}
    log.info("Target: %s (schema %s)", describe(url), schema)
    for k, v in counts.items():
        log.info("  %-26s %10s rows", k, f"{v:,}")
    if dry_run:
        return None

    with psycopg.connect(url) as conn:  # one transaction; commits on success, rolls back on error
        with conn.cursor() as cur:
            # check first: CREATE ... IF NOT EXISTS still needs CREATE on the database,
            # which a restricted loader role (docker/sql/roles.sql) does not have
            cur.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,))
            if cur.fetchone() is None:
                cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(f"""CREATE TABLE IF NOT EXISTS {schema}.release (
                release_id serial PRIMARY KEY,
                version text NOT NULL,
                label text,
                git_commit text,
                year_start integer,
                year_end integer,
                loaded_at timestamptz NOT NULL DEFAULT now(),
                is_current boolean NOT NULL DEFAULT false,
                row_counts jsonb,
                manifest jsonb)""")
            cur.execute(f"""INSERT INTO {schema}.release (version, label, git_commit, year_start, year_end, row_counts, manifest)
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING release_id""",
                        (bbp.__version__, label or f"load {dt.date.today().isoformat()}", git_commit(cfg.repo_root),
                         cfg.years[0], cfg.years[-1], json.dumps(counts), json.dumps(_latest_manifest(cfg), default=str)))
            release_id = cur.fetchone()[0]
            for name, df in frames.items():
                df = df.copy()
                df.insert(0, "release_id", release_id)
                _ensure_table(cur, schema, name, df.drop(columns="release_id"), TABLES[name][1])
                _copy(cur, schema, name, df)
                log.info("  loaded %s", name)
            cur.execute(f"UPDATE {schema}.release SET is_current = (release_id = %s)", (release_id,))
            _views(cur, schema, list(frames))
            if with_geometry:
                n = load_geometry(cur, cfg, schema, simplify)
                log.info("  loaded municipality_geom (%d municipalities)", n)
            if keep > 0:
                cur.execute(f"""DELETE FROM {schema}.release WHERE release_id NOT IN (
                    SELECT release_id FROM {schema}.release ORDER BY release_id DESC LIMIT %s)""", (keep,))
                if cur.rowcount:
                    log.info("  pruned %d old release(s)", cur.rowcount)
    log.info("Release %d is now current in %s", release_id, describe(url))
    return release_id


def status(url: str, schema: str = "bbp") -> pd.DataFrame:
    import psycopg

    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{schema}.release",))
        if cur.fetchone()[0] is None:
            return pd.DataFrame()
        cur.execute(f"SELECT release_id, version, label, git_commit, loaded_at, is_current, row_counts "
                    f"FROM {schema}.release ORDER BY release_id")
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
