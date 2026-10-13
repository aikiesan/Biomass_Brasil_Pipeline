"""IBGE municipal mesh -> attribute table + geometry layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RENAME = {
    "CD_MUN": "geocode",
    "NM_MUN": "name",
    "SIGLA_UF": "uf",
    "CD_UF": "cd_uf",
    "CD_RGINT": "cd_rgint",
    "NM_RGINT": "nm_rgint",
    "CD_RGI": "cd_rgi",
    "NM_RGI": "nm_rgi",
    "SIGLA_RG": "region",
    "AREA_KM2": "area_km2_ibge",
}


def read_mesh(path: Path, non_municipal: list[int]):
    import geopandas as gpd

    gdf = gpd.read_file(path)
    missing = set(RENAME) - set(gdf.columns)
    if missing:
        raise ValueError(f"Municipal mesh {path.name} lacks columns: {sorted(missing)}")
    gdf = gdf[list(RENAME) + ["geometry"]].rename(columns=RENAME)
    gdf["geocode"] = gdf["geocode"].astype("int64")
    for col in ("cd_uf", "cd_rgint", "cd_rgi"):  # empty for the two lagoon units
        gdf[col] = pd.to_numeric(gdf[col]).astype("Int64")
    gdf["is_municipality"] = ~gdf["geocode"].isin(non_municipal)
    return gdf.sort_values("geocode").reset_index(drop=True)


def attributes(gdf) -> pd.DataFrame:
    return pd.DataFrame(gdf.drop(columns="geometry"))
