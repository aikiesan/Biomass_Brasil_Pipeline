"""STAC catalogue of a release folder (synthetic release, no source data)."""

import json

import pandas as pd
import pystac

from bbp.checks import CheckLog
from bbp.config import load_config
from bbp.publish import sha256
from bbp.stac import multihash_sha256, write_catalog


def _release(tmp_path):
    rel = tmp_path / "release" / "bbp_v0.1.0_2026-10-15"   # dotted name: must not be taken for a file
    rel.mkdir(parents=True)
    outputs = {}
    for name, df in {"rank_shift": pd.DataFrame({"year": [2020, 2021], "spearman_rho": [0.9, 0.95]}),
                     "h2_moran": pd.DataFrame({"crop": ["soybean"], "morans_i": [0.1]})}.items():
        p = rel / f"{name}.csv.gz"
        df.to_csv(p, index=False, compression="gzip")
        outputs[p.name] = {"rows": len(df), "columns": list(df.columns), "sha256": sha256(p)}
    gpkg = rel / "bbp_municipal_2024.gpkg"
    gpkg.write_bytes(b"not really a geopackage")
    outputs[gpkg.name] = {"sha256": sha256(gpkg)}
    manifest = {"version": "0.1.0", "git_commit": "abc123", "years": [2008, 2024], "outputs": outputs,
                "credits": {"MapBiomas": "Projeto MapBiomas", "IBGE": "IBGE"}}
    (rel / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return rel, manifest


def test_catalogue_describes_every_release_file(tmp_path, monkeypatch):
    monkeypatch.setenv("BBP_DATA_DIR", str(tmp_path))
    cfg = load_config()
    rel, manifest = _release(tmp_path)
    ck = CheckLog("publish")
    root = write_catalog(cfg, rel, manifest, ck)
    assert root == rel / "catalog.json" and not ck.failures, [c.name for c in ck.failures]
    assert not (rel.parent / "catalog.json").exists()

    cat = pystac.Catalog.from_file(str(root))
    col = next(cat.get_children())
    assert col.extent.temporal.intervals[0][0].year == 2008 and col.extent.temporal.intervals[0][1].year == 2024
    assert col.assets["manifest"].extra_fields["file:checksum"] == multihash_sha256(sha256(rel / "manifest.json"))
    items = {i.id: i for i in cat.get_items(recursive=True)}
    assert set(items) == {"rank_shift", "h2_moran", "bbp_municipal_2024"}
    a = items["rank_shift"].assets["data"]
    assert a.extra_fields["table:row_count"] == 2
    assert [c["name"] for c in a.extra_fields["table:columns"]] == ["year", "spearman_rho"]
    assert a.extra_fields["file:checksum"] == "1220" + manifest["outputs"]["rank_shift.csv.gz"]["sha256"]
    assert not a.href.startswith("/")                                  # self-contained
    assert items["bbp_municipal_2024"].datetime.year == 2024


def test_multihash_prefix():
    assert multihash_sha256("ab" * 32) == "1220" + "ab" * 32
