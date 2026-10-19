"""Processing plants of Brazil in one table: where residues are already handled, or could be.

Sources (paths relative to raw_root, see `plants:` in the paper profile):

* ANP, Painel Dinâmico — sugarcane and corn ethanol plants (lat/lon, type, status, cane crushing class,
  corn capacity, ethanol capacity) and biodiesel plants (capacity m³/day).
* ANP, biomethane open data — authorised biomethane plants (capacity m³/day; municipality only, so placed
  at the municipality's representative point).
* MapBiomas infrastructure layers (Col. 11 update): biogas plants (CIBiogás BiogasMap: category, use,
  scale), biomass thermal power plants (ANEEL SIGA: fuel, power), slaughterhouses under federal inspection
  (MAPA SIF), natural-gas delivery points and transport pipelines (context).
* CONAB, active sugar and ethanol mills 2025 — the second cane-mill register, for the cross-check.

Every plant becomes one row: family, subtype, status, capacity in its own unit, the residues it relates to
(field: left on the land, e.g. cane straw; processing: produced at the plant, e.g. bagasse, vinasse) and the
land-cover groups of `bbp.landcover` that supply it (`feeds`), used by `bbp.reach`.
"""

from __future__ import annotations

import logging
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("bbp")

SOY = ("soybean", "soy_corn2", "soy_cotton2")
FAMILIES = {             # family: (label, land-cover groups that supply it, residue link)
    "sugarcane_mill": ("Sugarcane mill (sugar/ethanol)", ("sugarcane",), "field + processing"),
    "corn_ethanol": ("Corn ethanol plant", ("soy_corn2",), "field + processing"),
    "biodiesel": ("Biodiesel plant", SOY, "processing"),
    "biogas": ("Biogas plant", (), "processing"),
    "biomethane": ("Biomethane plant (ANP)", (), "processing"),
    "biomass_power": ("Biomass power plant (UTE)", (), "processing"),
    "slaughterhouse": ("Slaughterhouse (SIF)", ("pasture",), "processing"),
}
FAMILY_ORDER = list(FAMILIES)

# ANP crushing classes (t of cane per season) -> representative value (class midpoint; open class 6.5 Mt)
CANE_CLASS_T = {"Menor que 1.500.000": 1.0e6, "Entre 1.500.000 e 2.500.000": 2.0e6,
                "Entre 2.500.000 e 3.500.000": 3.0e6, "Entre 3.500.000 e 4.500.000": 4.0e6,
                "Entre 4.500.000 e 5.500.000": 5.0e6, "Maior que 5.500.000": 6.5e6}
CANE_TYPES = {"Etanol 1G e Açúcar", "Etanol Cana 1G", "Açúcar", "Etanol 1G", "Etanol 1G, 2G e Açúcar", "Etanol 2G",
              "Etanol Milho Flex"}
CORN_TYPES = {"Etanol Milho Full", "Etanol Milho Flex"}
UTE_SUBTYPE = {"Bagaço de Cana de Açúcar": "bagasse", "Casca de Arroz": "agro_other", "Capim Elefante": "agro_other",
               "Biogás-AGR": "biogas", "Biogás - RA": "biogas", "Biogás - RU": "biogas", "Biogás - Floresta": "biogas",
               "Óleos vegetais": "agro_other", "Etanol": "agro_other", "Resíduos Sólidos Urbanos - RU": "urban",
               "Carvão - RU": "urban"}                     # the rest: forestry (wood, black liquor, charcoal)
COLUMNS = ["plant_id", "family", "subtype", "name", "uf", "municipality", "status", "operating", "lon", "lat",
           "capacity", "capacity_unit", "capacity_note", "residue_link", "source", "located"]


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z0-9]+", " ", s).strip()


def _num(v) -> float:
    """'5,500.00' (ANEEL), '1.300' or 1300.0 -> float; '-' or blank -> nan."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    t = str(v).strip()
    if t in ("", "-"):
        return np.nan
    if re.fullmatch(r"[\d,]+\.\d+", t):
        t = t.replace(",", "")
    elif re.fullmatch(r"[\d.]+,\d+", t):
        t = t.replace(".", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return np.nan


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def _shp(path: Path):
    import geopandas as gpd

    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs("EPSG:4674")
    g = g.to_crs("EPSG:4674")
    if (g.geom_type == "MultiPoint").any():
        g["geometry"] = g.geometry.representative_point()
    return g


def _frame(rows: dict, n: int) -> pd.DataFrame:
    df = pd.DataFrame({c: rows.get(c, [None] * n) for c in COLUMNS if c != "plant_id"})
    return df


# ---------------------------------------------------------------------------
# readers, one per source
# ---------------------------------------------------------------------------
def read_anp_ethanol(path: Path) -> pd.DataFrame:
    d = _read_csv(path)
    col = {c: c for c in d.columns}
    cls = next(c for c in col if c.startswith("Classe de Capacidade"))
    corn = next(c for c in col if c.startswith("Capacidade de processamento de milho"))
    anh = next(c for c in col if "Anidro" in c)
    hyd = next(c for c in col if "Hidratado" in c)
    status = d["Situação"].astype(str)
    out = []
    for fam, types in (("sugarcane_mill", CANE_TYPES), ("corn_ethanol", CORN_TYPES)):
        s = d[d["Tipo"].isin(types)]
        n = len(s)
        if fam == "sugarcane_mill":
            cap = s[cls].map(CANE_CLASS_T).astype(float)
            unit, note = "t cane/season", "ANP crushing class, midpoint"
        else:
            cap = s[corn].map(_num)
            unit, note = "t corn/year", "ANP corn processing capacity"
        ethanol = s[anh].map(_num).fillna(0) + s[hyd].map(_num).fillna(0)
        out.append(_frame({
            "family": [fam] * n, "subtype": s["Tipo"].values, "name": s["Nome"].values, "uf": s["UF"].values,
            "municipality": s["Cidade"].values, "status": status.loc[s.index].values,
            "operating": (status.loc[s.index] == "Em operação").values | (status.loc[s.index] == "Ampliação").values,
            "lon": s["Longitude"].map(_num).values, "lat": s["Latitude"].map(_num).values, "capacity": cap.values,
            "capacity_unit": [unit] * n, "capacity_note": [f"{note}; ethanol {e:,.0f} m³/day" for e in ethanol],
            "residue_link": [FAMILIES[fam][2]] * n, "source": ["ANP ethanol plants"] * n, "located": ["point"] * n},
            n))
    return pd.concat(out, ignore_index=True)


def read_anp_biodiesel(path: Path) -> pd.DataFrame:
    d = _read_csv(path)
    cap = next(c for c in d.columns if "m3/dia" in c)
    aut = next((c for c in d.columns if c.startswith("Autoriza")), None)
    n = len(d)
    st = d[aut].astype(str).values if aut else ["?"] * n
    return _frame({
        "family": ["biodiesel"] * n, "subtype": ["biodiesel"] * n, "name": d["Nome"].values, "uf": d["UF"].values,
        "municipality": d["Município"].values, "status": st, "operating": [True] * n,
        "lon": d["Longitude"].map(_num).values, "lat": d["Latitude"].map(_num).values,
        "capacity": d[cap].map(_num).values, "capacity_unit": ["m³ biodiesel/day"] * n,
        "capacity_note": ["ANP authorised capacity"] * n, "residue_link": ["processing"] * n,
        "source": ["ANP biodiesel plants"] * n, "located": ["point"] * n}, n)


def read_anp_biomethane(path: Path, municipalities: pd.DataFrame) -> pd.DataFrame:
    """ANP biomethane plants (one row each, latest month); placed at the municipality's representative point."""
    d = _read_csv(path)
    key = municipalities.assign(k=municipalities["uf"] + "|" + municipalities["name"].map(_norm)).set_index("k")
    k = d["uf"].astype(str) + "|" + d["municipio"].map(_norm)
    hit = k.map(lambda x: x in key.index)
    n = len(d)
    lon = np.where(hit, k.map(lambda x: key.at[x, "lon"] if x in key.index else np.nan), np.nan)
    lat = np.where(hit, k.map(lambda x: key.at[x, "lat"] if x in key.index else np.nan), np.nan)
    if (~hit).any():
        log.warning("biomethane: %d plants with no municipality match: %s", int((~hit).sum()),
                    ", ".join(k[~hit].tolist()))
    vol = d.get("vol_biogas_m3d")
    return _frame({
        "family": ["biomethane"] * n, "subtype": ["biomethane"] * n, "name": d["razao_social"].values,
        "uf": d["uf"].values, "municipality": d["municipio"].values, "status": d["mesano"].astype(str).values,
        "operating": [True] * n, "lon": lon, "lat": lat, "capacity": d["cap_biometano_m3d"].map(_num).values,
        "capacity_unit": ["m³ biomethane/day"] * n,
        "capacity_note": [f"ANP authorised; biogas use {u}%" for u in d.get("util_pct", [None] * n)]
        if vol is not None else ["ANP authorised"] * n,
        "residue_link": ["processing"] * n, "source": ["ANP biomethane plants"] * n,
        "located": ["municipality"] * n}, n)


def read_biogas(path: Path) -> pd.DataFrame:
    g = _shp(path)
    n = len(g)
    sub = g["Categoria"].map({"Agropecuária": "agriculture", "Indústria": "industry", "RSU ou Esgoto": "urban"})
    return _frame({
        "family": ["biogas"] * n, "subtype": sub.fillna("other").values, "name": g.get("Label", [None] * n),
        "uf": g["Estado"].values, "municipality": g["Municipio"].values, "status": g["Situacao"].values,
        "operating": (g["Situacao"] == "Em operação").values, "lon": g.geometry.x.values, "lat": g.geometry.y.values,
        "capacity": g["Escala_pro"].map({"Pequeno porte": 1, "Médio porte": 2, "Grande porte": 3}).values,
        "capacity_unit": ["scale class (1 small, 2 medium, 3 large)"] * n,
        "capacity_note": ("use: " + g["Principal_"].astype(str)).values, "residue_link": ["processing"] * n,
        "source": ["MapBiomas / CIBiogás biogas plants"] * n, "located": ["point"] * n}, n)


def read_biomass_power(path: Path) -> pd.DataFrame:
    g = _shp(path)
    fuel = next(c for c in g.columns if c.startswith("Combust"))
    pot = next(c for c in g.columns if c.startswith("Potência O") or c.startswith("Potencia O"))
    n = len(g)
    mun = g.get("Município", g.get("Municipio"))
    return _frame({
        "family": ["biomass_power"] * n, "subtype": g[fuel].map(UTE_SUBTYPE).fillna("forestry").values,
        "name": g["Empreendim"].values, "uf": g["UF"].values,
        "municipality": mun.astype(str).str.split(" - ").str[0].values if mun is not None else [None] * n,
        "status": g["Fase"].values, "operating": (g["Fase"] == "Operação").values,
        "lon": g.geometry.x.values, "lat": g.geometry.y.values, "capacity": g[pot].map(_num).values,
        "capacity_unit": ["kW"] * n, "capacity_note": ("fuel: " + g[fuel].astype(str)).values,
        "residue_link": ["processing"] * n, "source": ["MapBiomas / ANEEL SIGA biomass UTE"] * n,
        "located": ["point"] * n}, n)


def read_slaughterhouses(path: Path) -> pd.DataFrame:
    g = _shp(path)
    n = len(g)
    c1 = g["classe_1"].astype(str).str.upper()
    sub = np.select([c1.str.contains("BOVIN"), c1.str.contains("SU[IÍ]N"), c1.str.contains("AVE|FRANGO")],
                    ["cattle", "pig", "poultry"], "other")
    return _frame({
        "family": ["slaughterhouse"] * n, "subtype": sub, "name": g["razao_soci"].values, "uf": g["uf"].values,
        "municipality": g["municipio"].values, "status": g["situacao"].values,
        "operating": (g["situacao"] == "Ativo").values, "lon": g.geometry.x.values, "lat": g.geometry.y.values,
        "capacity": [np.nan] * n, "capacity_unit": ["—"] * n, "capacity_note": g["classe_1"].astype(str).values,
        "residue_link": ["processing"] * n, "source": ["MapBiomas / MAPA SIF slaughterhouses"] * n,
        "located": ["point"] * n}, n)


def read_conab_mills(path: Path):
    """CONAB active sugar/ethanol units 2025 (a zipped shapefile), for the register cross-check."""
    import geopandas as gpd

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            shp = next(n for n in z.namelist() if n.endswith(".shp"))
        g = gpd.read_file(f"zip://{path.as_posix()}!{shp}")
    else:
        g = gpd.read_file(path)
    return g.to_crs("EPSG:4674")


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
def _spec(cfg) -> dict:
    return cfg.section("plants") or {}


def source_paths(cfg) -> dict[str, Path]:
    sp = _spec(cfg).get("sources", {})
    return {k: cfg.raw_root / v for k, v in sp.items()}


def _municipality_points(cfg) -> pd.DataFrame:
    import geopandas as gpd

    attrs = pd.read_parquet(cfg.data_dir / "interim" / "municipalities.parquet", columns=["geocode", "name", "uf",
                                                                                          "cd_rgint", "nm_rgint"])
    geom = gpd.read_parquet(cfg.data_dir / "interim" / "municipalities_geom.parquet", columns=["geocode", "geometry"])
    pt = geom.set_index("geocode").geometry.representative_point()
    attrs["lon"], attrs["lat"] = attrs["geocode"].map(pt.x), attrs["geocode"].map(pt.y)
    return attrs


def build_plants(cfg, ck=None):
    """Read every source, keep plants inside Brazil, attach municipality and intermediate region (spatial join)
    and write data/interim/plants.parquet (GeoParquet, EPSG:4674). Returns the GeoDataFrame."""
    import geopandas as gpd

    paths = source_paths(cfg)
    mun = _municipality_points(cfg)
    readers = {"anp_ethanol": read_anp_ethanol, "anp_biodiesel": read_anp_biodiesel,
               "biogas": read_biogas, "biomass_power": read_biomass_power, "slaughterhouses": read_slaughterhouses,
               "anp_biomethane": lambda p: read_anp_biomethane(p, mun)}
    parts = []
    for key, reader in readers.items():
        p = paths.get(key)
        ok = p is not None and p.exists()
        if ck is not None:
            ck.add(f"plants_{key}_found", ok, str(p) if p else "not configured", severity="warn")
        if not ok:
            continue
        df = reader(p)
        parts.append(df)
        log.info("  plants: %-16s %5d rows (%s)", key, len(df), p.name)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df["family"] = pd.Categorical(df["family"], FAMILY_ORDER)
    df = df.sort_values(["family", "uf", "name"], kind="stable").reset_index(drop=True)
    df["plant_id"] = [f"{f[:4]}{i:04d}" for i, f in enumerate(df["family"].astype(str))]
    g = gpd.GeoDataFrame(df[COLUMNS], geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4674")
    g = g[g.geometry.notna() & np.isfinite(g["lon"]) & np.isfinite(g["lat"])]
    polys = gpd.read_parquet(cfg.data_dir / "interim" / "municipalities_geom.parquet", columns=["geocode", "geometry"])
    j = gpd.sjoin(g, polys.to_crs("EPSG:4674"), how="left", predicate="within").drop(columns="index_right")
    j = j[~j.index.duplicated()]
    # points on the coast or a border (just outside every polygon): nearest municipality within 5 km
    miss = j["geocode"].isna()
    if miss.any():
        near = gpd.sjoin_nearest(j.loc[miss, COLUMNS + ["geometry"]].to_crs("EPSG:5880"),
                                 polys.to_crs("EPSG:5880"), max_distance=5000, how="left").drop(columns="index_right")
        near = near[~near.index.duplicated()]
        j.loc[miss, "geocode"] = near["geocode"].values
    attrs = mun.set_index("geocode")
    j["geocode"] = j["geocode"].astype("Int64")
    j["cd_rgint"] = j["geocode"].map(attrs["cd_rgint"]).astype("Int64")
    j["nm_rgint"] = j["geocode"].map(attrs["nm_rgint"])
    outside = j["geocode"].isna()
    if ck is not None:
        ck.add("plants_inside_brazil", bool(outside.sum() <= 0.01 * len(j)),
               f"{int(outside.sum())} of {len(j)} plants outside every municipality (dropped)", int(outside.sum()),
               severity="warn")
    j = j[~outside].copy()
    out = cfg.data_dir / "interim" / "plants.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    j.to_parquet(out)
    if ck is not None:
        by = j.groupby("family", observed=True).size()
        ck.add("plants_written", len(j) > 0, "; ".join(f"{k} {v}" for k, v in by.items()), int(len(j)))
    return j


def load_plants(cfg):
    import geopandas as gpd

    p = cfg.data_dir / "interim" / "plants.parquet"
    return gpd.read_parquet(p) if p.exists() else None
