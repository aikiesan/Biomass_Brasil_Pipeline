"""STAC catalogue of a release folder (pystac), written by `publish` next to the files it describes.

    release/bbp_v<version>_<date>/
        catalog.json                       root catalogue
        <collection>/collection.json       the release: extent, providers, manifest as an asset
        <collection>/<item>/<item>.json    one item per table (CSV) and one for the GeoPackage

Every asset carries its size and SHA-256 (file extension, multihash) and, for tables, the row
count and column names (table extension), so the catalogue can be checked against the files
without the manifest. The catalogue is self-contained: every href is relative.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

from bbp.checks import CheckLog
from bbp.config import Config

log = logging.getLogger("bbp")

TABLE_TITLES = {
    "municipality_attributes": "Municipalities: region, state, AMC, first PAM year",
    "crop_area": "Observed (MapBiomas) and declared (IBGE PAM) area per municipality, year and crop",
    "agreement_by_scale": "Agreement metrics by scale, crop and year (H1)",
    "unit_ratios_municipality": "Observed / declared area per municipality",
    "unit_ratios_rgint": "Observed / declared area per intermediate region",
    "unit_ratios_uf": "Observed / declared area per state",
    "predictors": "Municipal predictors of the divergence (H2)",
    "n1_municipal": "Theoretical biomethane potential N1 (Nm³ CH4/yr), both area sources, factor envelope",
    "source_vs_factor_effect": "Area-source effect vs conversion-factor envelope on N1 (H3)",
    "rank_shift": "Rank correlation and top-decile turnover of municipal N1 (H3)",
    "sanitation": "Municipal sanitation series (SNIS 2008–2022, SINISA 2023–)",
    "sanitation_continuity": "SNIS -> SINISA continuity per variable",
    "h2_mixed_effects": "H2 mixed model: standardised fixed effects",
    "h2_model_fit": "H2 model fit per crop (mixed model, random forest)",
    "h2_rf_importance": "H2 random forest permutation importance",
    "h2_moran": "H2 Moran's I of the mixed-model residuals",
    "plants": "Processing plants (ANP, CIBiogás, ANEEL, MAPA SIF): one row per plant function, sites linked",
    "plants_biomethane_snapshots": "ANP biomethane plants, authorised capacity and production per snapshot month",
    "plants_register_match": "ANP cane mills against the CONAB and MapBiomas/EPE registers (share within 2/10 km)",
    "reach_units": "Crop area within reach of existing plants, by radius, at municipality, region, state, nation",
    "reach_hist": "Crop area per municipality in 1 km bins of distance to the nearest plant (30 m pixels)",
    "reach_distance": "National crop area by distance to the nearest plant, 1 km bins to 2000 km",
    "reach_competition": "Crop area by number of plants within the competition radius",
    "reach_mills": "Sugarcane mills: cane within each radius, required radius, supply ratio, contested cane",
    "reach_aggregation_check": "Share within reach from 30 m pixels vs from ~480 m cells",
}


def multihash_sha256(hex_digest: str) -> str:
    """STAC file:checksum is a multihash: 0x12 (sha2-256), 0x20 (32 bytes), then the digest."""
    return "1220" + hex_digest


def mesh_bbox(cfg: Config) -> list[float]:
    """Bounding box of the municipal mesh (SIRGAS 2000 ~ WGS 84 at this scale), from the GeoParquet metadata."""
    import pyarrow.parquet as pq

    path = cfg.data_dir / "interim" / "municipalities_geom.parquet"
    if path.exists():
        meta = pq.read_metadata(path).metadata or {}
        geo = json.loads(meta.get(b"geo", b"{}"))
        bbox = geo.get("columns", {}).get("geometry", {}).get("bbox")
        if bbox:
            return [float(v) for v in bbox]
    return [-73.99, -33.76, -28.84, 5.28]  # Brazil, IBGE mesh 2025


def _bbox_polygon(b: list[float]) -> dict[str, Any]:
    return {"type": "Polygon", "coordinates": [[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]]]}


def _asset(href: Path, media_type: str, title: str, roles: list[str], info: dict):
    import pystac
    from pystac.extensions.file import FileExtension

    a = pystac.Asset(href=str(href), media_type=media_type, title=title, roles=roles)
    fx = FileExtension.ext(a, add_if_missing=False)
    fx.size = href.stat().st_size
    if info.get("sha256"):
        fx.checksum = multihash_sha256(info["sha256"])
    return a


def build_catalog(cfg: Config, rel: Path, manifest: dict) -> Any:
    """pystac Catalog for the release folder `rel` described by `manifest` (not yet saved)."""
    import pystac
    from pystac.extensions.file import FileExtension
    from pystac.extensions.table import Column, TableExtension

    years = manifest["years"]
    start = dt.datetime(int(years[0]), 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(int(years[1]), 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc)
    bbox = mesh_bbox(cfg)
    version = manifest["version"]
    cid = rel.name.replace("_", "-").replace(".", "-")

    catalog = pystac.Catalog(
        id="bbp", title="Biomass Brasil Pipeline (bbp)",
        description="Municipal crop area from satellite land cover (MapBiomas) and agricultural statistics "
                    "(IBGE PAM) for Brazil, their agreement, and the crop-residue biomethane potential N1. "
                    "Every release is regenerated by the open pipeline (GPL-3.0).")
    providers = [
        pystac.Provider("Projeto MapBiomas", roles=[pystac.ProviderRole.LICENSOR, pystac.ProviderRole.PRODUCER],
                        url="https://mapbiomas.org", description=manifest["credits"].get("MapBiomas")),
        pystac.Provider("IBGE", roles=[pystac.ProviderRole.PRODUCER], url="https://sidra.ibge.gov.br",
                        description=manifest["credits"].get("IBGE")),
        pystac.Provider("CP2b / NIPE-UNICAMP", roles=[pystac.ProviderRole.PROCESSOR, pystac.ProviderRole.HOST],
                        description=f"bbp {version}, commit {manifest.get('git_commit') or 'unknown'}"),
    ]
    collection = pystac.Collection(
        id=cid, title=f"bbp release {version} ({rel.name})",
        description=f"Release {rel.name} of the Biomass Brasil Pipeline, {years[0]}–{years[1]}. Inputs, their "
                    "SHA-256 and the invariant checks of the run are in the manifest asset.",
        extent=pystac.Extent(pystac.SpatialExtent([bbox]), pystac.TemporalExtent([[start, end]])),
        license="other", providers=providers, keywords=["biomass", "biomethane", "MapBiomas", "IBGE", "PAM",
                                                        "Brazil", "municipality"],
        extra_fields={"bbp:version": version, "bbp:git_commit": manifest.get("git_commit")},
    )
    manifest_path = rel / "manifest.json"
    if manifest_path.exists():
        from bbp.publish import sha256

        collection.add_asset("manifest", _asset(manifest_path, pystac.MediaType.JSON, "Release manifest",
                                                ["metadata"], {"sha256": sha256(manifest_path)}))
    FileExtension.add_to(collection)
    catalog.add_child(collection)

    for name, info in manifest["outputs"].items():
        path = rel / name
        if not path.exists():
            continue
        if name.endswith(".gpkg"):
            year = int(path.stem.rsplit("_", 1)[-1])
            item = pystac.Item(id=path.stem, geometry=_bbox_polygon(bbox), bbox=bbox,
                               datetime=dt.datetime(year, 7, 1, tzinfo=dt.timezone.utc),
                               properties={"title": f"Municipal layer {year}: observed and declared area, log-ratio",
                                           "start_datetime": f"{year}-01-01T00:00:00Z",
                                           "end_datetime": f"{year}-12-31T23:59:59Z"})
            item.add_asset("data", _asset(path, pystac.MediaType.GEOPACKAGE, f"bbp_municipal_{year}",
                                          ["data"], info))
        else:
            table = name.removesuffix(".csv.gz")
            item = pystac.Item(id=table, geometry=_bbox_polygon(bbox), bbox=bbox, datetime=None,
                               start_datetime=start, end_datetime=end,
                               properties={"title": TABLE_TITLES.get(table, table)})
            asset = _asset(path, "text/csv", f"{table}.csv.gz (gzip)", ["data"], info)
            item.add_asset("data", asset)
            tx = TableExtension.ext(item.assets["data"], add_if_missing=True)
            tx.row_count = int(info["rows"])
            tx.columns = [Column({"name": c}) for c in info["columns"]]
        FileExtension.add_to(item)
        collection.add_item(item)
    return catalog


def write_catalog(cfg: Config, rel: Path, manifest: dict, ck: CheckLog) -> Path:
    import pystac

    catalog = build_catalog(cfg, rel, manifest)
    catalog.normalize_hrefs(rel.as_posix() + "/")  # trailing slash: a dotted folder name is not a file
    catalog.make_all_asset_hrefs_relative()
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    root = rel / "catalog.json"

    # read it back: every asset resolves to a file of the recorded size, one item per release file
    back = pystac.Catalog.from_file(str(root))
    items = list(back.get_items(recursive=True))
    missing, wrong = [], []
    for item in items:
        for key, a in item.assets.items():
            p = Path(a.get_absolute_href())
            if not p.exists():
                missing.append(f"{item.id}/{key}")
            elif a.extra_fields.get("file:size") != p.stat().st_size:
                wrong.append(f"{item.id}/{key}")
    ck.add("stac_items_match_release_files", len(items) == len(manifest["outputs"]),
           f"{len(items)} items for {len(manifest['outputs'])} files", len(items))
    ck.add("stac_assets_resolve", not missing, f"missing: {missing}", missing)
    ck.add("stac_asset_sizes", not wrong, f"size differs: {wrong}", wrong)
    hrefs = [a.href for i in items for a in i.assets.values()]
    ck.add("stac_self_contained", all(not h.startswith(("/", "file:")) and ":" not in h[:3] for h in hrefs),
           "asset hrefs are relative")
    log.info("STAC catalogue: %s (%d items)", root, len(items))
    return root
