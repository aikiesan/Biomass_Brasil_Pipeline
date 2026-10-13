import os

import pandas as pd
import pytest

from bbp.config import load_config
from bbp.load import _drop_empty, collect_sidra, is_local, pg_type, read_dotenv


def test_helpers(tmp_path):
    assert is_local("postgresql://bbp:x@localhost:5433/cp2b_maps")
    assert is_local("postgresql://bbp:x@db:5432/cp2b_maps")
    assert not is_local("postgresql://u:x@db.abc.supabase.co:5432/postgres")
    assert not is_local("postgresql://u:x@localhost:5434/cp2b_maps")  # SSH tunnel to a server
    assert is_local("postgresql://u:x@cp2b-db-dev:5432/cp2b_maps")    # PILAR-2b dev container
    assert not is_local("postgresql://u:x@host.docker.internal:5434/cp2b_maps")
    env = tmp_path / ".env"
    env.write_text("# comment\nBBP_DATABASE_URL='postgresql://a@b/c'\n", encoding="utf-8")
    assert read_dotenv(env)["BBP_DATABASE_URL"] == "postgresql://a@b/c"
    df = pd.DataFrame({"i": [1], "f": [1.5], "b": [True], "s": ["x"]})
    assert [pg_type(df[c]) for c in df] == ["bigint", "double precision", "boolean", "text"]
    d = pd.DataFrame({"a": [0, None, 3], "b": [None, 0, 0]})
    assert len(_drop_empty(d, ["a", "b"])) == 1


def _sidra_parquet(data_dir, table=3939):
    d = data_dir / "interim" / "sidra"
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"table": [table] * 3, "survey": ["PPM"] * 3, "geocode": [1, 1, 2], "name": ["A", "A", "B"],
                       "year": [2025] * 3, "product": ["Bovino", "Bovino", "Bovino"],
                       "variable": ["Efetivo dos rebanhos (Cabeças)", "value_thousand_brl", "Efetivo dos rebanhos (Cabeças)"],
                       "variable_raw": ["x"] * 3, "unit": ["Cabeças", None, "Cabeças"], "value": [10.0, 5.0, None],
                       "flag": ["ok", "ok", "not_available"], "preliminary": [True] * 3, "source_file": ["f.xlsx"] * 3})
    for c in ("product", "variable", "unit", "flag"):
        df[c] = df[c].astype("category")
    df.to_parquet(d / f"t{table}.parquet")


def test_collect_sidra_keeps_missing_cells_and_drops_excluded_variables(tmp_path, monkeypatch):
    _sidra_parquet(tmp_path)
    monkeypatch.setenv("BBP_DATA_DIR", str(tmp_path))
    cfg = load_config()
    cfg.raw["database"] = {"sidra_tables": [3939, 74], "sidra_exclude_variables": ["value_thousand_brl"]}
    df = collect_sidra(cfg)  # table 74 absent: skipped with a warning
    assert len(df) == 2                                   # the "..." cell is kept (flag says why it is empty)
    assert "sidra_table" in df and "table" not in df      # "table" is an SQL keyword
    assert "name" not in df and "variable_raw" not in df
    assert df.loc[df["geocode"] == 2, "flag"].item() == "not_available"


URL = os.environ.get("BBP_TEST_DATABASE_URL")


@pytest.mark.skipif(not URL, reason="set BBP_TEST_DATABASE_URL to run the PostGIS integration test")
def test_load_roundtrip(tmp_path, monkeypatch):
    import psycopg

    from bbp.load import load, status

    proc = tmp_path / "processed"
    proc.mkdir()
    pd.DataFrame({"geocode": [1, 2], "year": [2020, 2020], "crop": ["soybean", "soybean"],
                  "a_obs_ha": [10.0, 0.0], "a_dec_ha": [12.0, 0.0], "a_harv_ha": [12.0, None],
                  "prod_t": [36.0, None], "obs_source": ["mapbiomas_coverage"] * 2, "dec_flag": ["ok", "zero"]}
                 ).to_parquet(proc / "crop_area.parquet")
    pd.DataFrame({"year": [2020], "n_units": [2], "spearman_rho": [0.9], "top_decile_turnover": [0.0]}
                 ).to_parquet(proc / "rank_shift.parquet")
    _sidra_parquet(tmp_path)
    monkeypatch.setenv("BBP_DATA_DIR", str(tmp_path))
    cfg = load_config()
    cfg.raw.setdefault("database", {}).update(schema="bbp_test", sidra_tables=[3939])

    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS bbp_test CASCADE")
    r1 = load(cfg, URL, label="first")
    r2 = load(cfg, URL, label="second", keep=1)
    assert r2 == r1 + 1
    with psycopg.connect(URL) as conn:
        n = conn.execute("SELECT count(*) FROM bbp_test.v_crop_area").fetchone()[0]
        releases = conn.execute("SELECT count(*) FROM bbp_test.release").fetchone()[0]
        sidra = conn.execute("SELECT count(*), count(value) FROM bbp_test.v_sidra WHERE sidra_table = 3939").fetchone()
    assert n == 1          # compact mode dropped the all-zero row
    assert releases == 1   # --keep 1 pruned the first release (cascade)
    assert sidra == (2, 1)  # value_thousand_brl excluded; the "..." cell kept as NULL
    st = status(URL, "bbp_test")
    assert st["is_current"].tolist() == [True]
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("DROP SCHEMA bbp_test CASCADE")


def test_git_head_without_git_binary(tmp_path):
    from bbp.publish import _read_git_head

    g = tmp_path / ".git"
    (g / "refs" / "heads").mkdir(parents=True)
    (g / "HEAD").write_text("ref: refs/heads/main\n")
    (g / "refs" / "heads" / "main").write_text("abc123\n")
    assert _read_git_head(tmp_path) == "abc123"
    (g / "refs" / "heads" / "main").unlink()
    (g / "packed-refs").write_text("# pack-refs\ndef456 refs/heads/main\n")
    assert _read_git_head(tmp_path) == "def456"
