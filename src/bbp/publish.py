"""Build a versioned release folder: CSV tables, a GeoPackage and a manifest with checksums."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

import bbp
from bbp.checks import CheckLog
from bbp.config import Config

log = logging.getLogger("bbp")

TABLES = [
    "municipality_attributes", "crop_area", "agreement_by_scale", "unit_ratios_municipality", "unit_ratios_rgint",
    "unit_ratios_uf", "predictors", "n1_municipal", "source_vs_factor_effect", "rank_shift",
]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _inputs(cfg: Config) -> list[Path]:
    paths: list[Path] = []
    for key in cfg.raw["inputs"]:
        p = cfg.input_path(key)
        paths.extend(p if isinstance(p, list) else [p])
    tif_dir = cfg.input_path("second_crop_tif_dir")
    if tif_dir.is_dir():
        paths.extend(sorted(tif_dir.glob("*.tif")))
    return [p for p in paths if p.is_file()]


def publish(cfg: Config, ck: CheckLog) -> Path:
    stamp = dt.date.today().isoformat()
    rel = cfg.out("release", f"bbp_v{bbp.__version__}_{stamp}", "x").parent
    rel.mkdir(parents=True, exist_ok=True)

    files = {}
    for name in TABLES:
        src = cfg.data_dir / "processed" / f"{name}.parquet"
        if not src.exists():
            ck.add(f"release_has_{name}", False, "table missing", severity="warn")
            continue
        df = pd.read_parquet(src)
        dst = rel / f"{name}.csv.gz"
        df.to_csv(dst, index=False, compression="gzip")
        files[dst.name] = {"rows": len(df), "columns": list(df.columns), "sha256": sha256(dst)}

    gpkg = _municipal_layer(cfg, rel)
    if gpkg is not None:
        files[gpkg.name] = {"sha256": sha256(gpkg)}

    checks = {}
    for p in sorted((cfg.data_dir / "logs").glob("checks_*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        checks[data["stage"]] = {
            "passed": sum(c["passed"] for c in data["checks"]),
            "failed": [c["name"] for c in data["checks"] if not c["passed"] and c["severity"] == "error"],
            "warnings": [c["name"] for c in data["checks"] if not c["passed"] and c["severity"] == "warn"],
        }

    log.info("Hashing input files (MapBiomas rasters take a moment)...")
    manifest = {
        "pipeline": "Biomass Brasil Pipeline (bbp)",
        "version": bbp.__version__,
        "git_commit": git_commit(cfg.repo_root),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "years": [cfg.years[0], cfg.years[-1]],
        "config": cfg.raw,
        "inputs": {str(p.relative_to(cfg.raw_root)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                   for p in _inputs(cfg)},
        "outputs": files,
        "checks": checks,
        "credits": {
            "MapBiomas": "Projeto MapBiomas — Collection 11 of the annual land use and land cover maps of Brazil and the "
                         "second-crop (beta) layer, accessed via mapbiomas.org.",
            "IBGE": "IBGE — Produção Agrícola Municipal (PAM), SIDRA tables 5457 and 839; Malha Municipal 2025.",
            "CP2b": "Conversion factors from the CP2b canonical parameter set (FAPESP 2024/01112-1).",
        },
    }
    with open(rel / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)
    log.info("Release written to %s", rel)
    return rel


def _municipal_layer(cfg: Config, rel: Path) -> Path | None:
    """One GeoPackage layer with a wide summary for the latest year that has both sources."""
    import geopandas as gpd

    units = cfg.data_dir / "processed" / "unit_ratios_municipality.parquet"
    mesh = cfg.data_dir / "interim" / "municipalities_geom.parquet"
    if not (units.exists() and mesh.exists()):
        return None
    u = pd.read_parquet(units)
    year = int(u["year"].max())
    w = u[u["year"] == year].pivot_table(index="unit", columns="crop", values=["a_obs_ha", "a_dec_ha", "log_ratio"], observed=True)
    w.columns = [f"{v}_{c}" for v, c in w.columns]
    w.index = w.index.astype("int64")
    gdf = gpd.read_parquet(mesh).merge(w, left_on="geocode", right_index=True, how="left")
    gdf["year"] = year
    out = rel / f"bbp_municipal_{year}.gpkg"
    # GeoPackage is SQLite: write it on a local temp disk and copy, because SQLite
    # cannot lock files on some network/virtual file systems (e.g. mounted shares)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / out.name
        gdf.to_file(local, driver="GPKG", layer=f"agreement_{year}")
        shutil.copyfile(local, out)
    return out
