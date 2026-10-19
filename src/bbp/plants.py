"""Processing plants of Brazil in one register: where crop residues are already handled, or could be.

`bbp run plants` (or `bbp plants`). Sources (paths relative to raw_root, `plants.sources` in pipeline.yaml):

* ANP, Painel Dinâmico — sugarcane and corn ethanol plants (lat/lon, type, status, cane crushing class,
  corn capacity, ethanol capacity) and biodiesel plants (capacity m³/day).
* ANP, biomethane open data (Painel Dinâmico do Biometano, CSV) — authorised capacity per plant and month,
  production per state and month. The register takes the plants of one snapshot month (`snapshot`); a plant
  whose last record is older is kept but marked not operating. Plants are placed at the municipality's
  representative point (the open data has no coordinates).
* MapBiomas infrastructure layers (Col. 11 update): biogas plants (CIBiogás BiogasMap: category, use,
  scale), biomass thermal power plants (ANEEL SIGA: fuel, power), slaughterhouses under federal inspection
  (MAPA SIF).
* CONAB active sugar and ethanol mills 2025 and the MapBiomas ethanol layer (EPE webmap 2018): second and
  third cane-mill registers, used only to cross-check where the ANP mills are.

One row per plant *function* (family): a flex ethanol plant (cane + corn) is one row in each family, and both
rows share one `site_id`. Rows of different sources that describe the same site (a bagasse power plant inside a
cane mill, a biogas plant that feeds an ANP biomethane plant) are linked to one `site_id` as well
(`plants.site_links`); nothing is dropped, so every family keeps its own source and capacity.
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

FAMILIES = {             # family: (label, residue link)
    "sugarcane_mill": ("Sugarcane mill (sugar/ethanol)", "field + processing"),
    "corn_ethanol": ("Corn ethanol plant", "field + processing"),
    "biodiesel": ("Biodiesel plant", "processing"),
    "biogas": ("Biogas plant", "processing"),
    "biomethane": ("Biomethane plant (ANP)", "processing"),
    "biomass_power": ("Biomass power plant (UTE)", "processing"),
    "slaughterhouse": ("Slaughterhouse (SIF)", "processing"),
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
UF_OF_STATE = {
    "Acre": "AC", "Alagoas": "AL", "Amapá": "AP", "Amazonas": "AM", "Bahia": "BA", "Ceará": "CE",
    "Distrito Federal": "DF", "Espírito Santo": "ES", "Goiás": "GO", "Maranhão": "MA", "Mato Grosso": "MT",
    "Mato Grosso do Sul": "MS", "Minas Gerais": "MG", "Pará": "PA", "Paraíba": "PB", "Paraná": "PR",
    "Pernambuco": "PE", "Piauí": "PI", "Rio de Janeiro": "RJ", "Rio Grande do Norte": "RN",
    "Rio Grande do Sul": "RS", "Rondônia": "RO", "Roraima": "RR", "Santa Catarina": "SC", "São Paulo": "SP",
    "Sergipe": "SE", "Tocantins": "TO",
}
COLUMNS = ["plant_id", "site_id", "record_id", "family", "subtype", "name", "uf", "municipality", "status",
           "operating", "status_reason", "lon", "lat", "capacity", "capacity_unit", "capacity_note", "residue_link",
           "source", "cutoff", "located"]
# ANP biomethane open data (column names after stripping)
BM_MONTH, BM_NAME, BM_CNPJ, BM_STATE, BM_MUN = "Mês/Ano", "Razão Social", "CNPJ", "Estado", "Município"
BM_CAP = "Capacidade Autorizada de Produção de Biometano (m³/d)"
BM_CAP_BIOGAS = "Capacidade Processamento de Biogás(m³/d)"
BM_VOL_BIOGAS = "Volume Processado de Biogás (m³/d)"
BM_PROD, BM_PRODUCT = "Produção (m³)", "Produto"


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z0-9]+", " ", s).strip()


def _num(v) -> float:
    """'5,500.00' (ANEEL), '1.300,5' or 1300.0 -> float; '-' or blank -> nan."""
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


THOUSANDS = re.compile(r"\d{1,3}(\.\d{3})+")


def production_m3(v) -> tuple[float, bool]:
    """A value of the ANP production CSV -> (m³, read with a thousands separator?).

    The file mixes two notations: '4324235.558' (point = decimal) and '173.037' (point = thousands: the plant
    behind it processes ~10,000 m³ of biogas a day, so 173 m³ a month is impossible). One to three digits
    followed by groups of exactly three after points is read as thousands; `production_check` tests the
    result against the output implied by the capacity file."""
    t = str(v).strip()
    if THOUSANDS.fullmatch(t):
        return float(t.replace(".", "")), True
    return _num(t), False


def _read_csv(path: Path, **kw) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            d = pd.read_csv(path, sep=None, engine="python", encoding=enc, **kw)
            d.columns = [str(c).strip() for c in d.columns]
            return d
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
    return pd.DataFrame({c: rows.get(c, [None] * n) for c in COLUMNS if c not in ("plant_id", "site_id")})


# ---------------------------------------------------------------------------
# readers, one per source
# ---------------------------------------------------------------------------
def read_anp_ethanol(path: Path) -> pd.DataFrame:
    d = _read_csv(path)
    cls = next(c for c in d.columns if c.startswith("Classe de Capacidade"))
    corn = next(c for c in d.columns if c.startswith("Capacidade de processamento de milho"))
    anh = next(c for c in d.columns if "Anidro" in c)
    hyd = next(c for c in d.columns if "Hidratado" in c)
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
        st = status.loc[s.index]
        operating = st.isin(["Em operação", "Ampliação"])
        out.append(_frame({
            "record_id": ("anp_ethanol:" + s["FID"].astype(str)).values, "family": [fam] * n,
            "subtype": s["Tipo"].values, "name": s["Nome"].values, "uf": s["UF"].values,
            "municipality": s["Cidade"].values, "status": st.values, "operating": operating.values,
            "status_reason": np.where(operating, "", "ANP status: " + st).tolist(),
            "lon": s["Longitude"].map(_num).values, "lat": s["Latitude"].map(_num).values, "capacity": cap.values,
            "capacity_unit": [unit] * n, "capacity_note": [f"{note}; ethanol {e:,.0f} m³/day" for e in ethanol],
            "residue_link": [FAMILIES[fam][1]] * n, "source": ["ANP ethanol plants"] * n, "located": ["point"] * n},
            n))
    return pd.concat(out, ignore_index=True)


def read_anp_biodiesel(path: Path) -> pd.DataFrame:
    d = _read_csv(path)
    cap = next(c for c in d.columns if "m3/dia" in c)
    aut = next((c for c in d.columns if c.startswith("Autoriza")), None)
    n = len(d)
    st = d[aut].astype(str) if aut else pd.Series(["Operação"] * n)
    operating = st.isin(["Operação", "Ampliação"]).to_numpy()      # as for ethanol: construction is not operating
    rid = ("anp_biodiesel:" + d[d.columns[0]].astype(str)).values
    return _frame({
        "record_id": rid, "family": ["biodiesel"] * n, "subtype": ["biodiesel"] * n, "name": d["Nome"].values,
        "uf": d["UF"].values, "municipality": d["Município"].values, "status": st.values, "operating": operating,
        "status_reason": np.where(operating, "", "ANP status: " + st).tolist(),
        "lon": d["Longitude"].map(_num).values, "lat": d["Latitude"].map(_num).values,
        "capacity": d[cap].map(_num).values, "capacity_unit": ["m³ biodiesel/day"] * n,
        "capacity_note": ["ANP authorised capacity"] * n, "residue_link": ["processing"] * n,
        "source": ["ANP biodiesel plants"] * n, "located": ["point"] * n}, n)


def read_biomethane_capacity(path: Path) -> pd.DataFrame:
    """ANP biomethane capacity open data, long: one row per plant (CNPJ) and month, numbers parsed."""
    d = _read_csv(path, dtype=str)
    d["month"] = pd.to_datetime(d[BM_MONTH], format="%m/%Y")
    for c in (BM_CAP, BM_CAP_BIOGAS, BM_VOL_BIOGAS):
        d[c] = d[c].map(_num)
    d["uf"] = d[BM_STATE].map(UF_OF_STATE)
    return d


def read_anp_biomethane(path: Path, municipalities: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    """One row per ANP biomethane plant ever recorded up to `snapshot` (MM/YYYY), from its latest record.

    Operating = recorded in the snapshot month; a plant last seen earlier is kept, not operating, with the reason
    (e.g. Gasgrid, São Paulo: no record since 06/2022). Placed at the municipality's representative point."""
    d = read_biomethane_capacity(path)
    snap = pd.to_datetime(snapshot, format="%m/%Y")
    d = d[d["month"] <= snap]
    last = d.sort_values("month").groupby(BM_CNPJ, as_index=False).tail(1).reset_index(drop=True)
    key = municipalities.assign(k=municipalities["uf"] + "|" + municipalities["name"].map(_norm)).set_index("k")
    k = last["uf"].astype(str) + "|" + last[BM_MUN].map(_norm)
    hit = k.isin(key.index)
    if (~hit).any():
        log.warning("biomethane: %d plants with no municipality match: %s", int((~hit).sum()),
                    ", ".join(k[~hit].tolist()))
    n = len(last)
    on = (last["month"] == snap).to_numpy()
    seen = last["month"].dt.strftime("%m/%Y")
    out = _frame({
        "record_id": ("anp_biomethane:" + last[BM_CNPJ].astype(str)).values, "family": ["biomethane"] * n,
        "subtype": ["biomethane"] * n, "name": last[BM_NAME].values, "uf": last["uf"].values,
        "municipality": last[BM_MUN].values, "status": ("last record " + seen).values, "operating": on,
        "status_reason": np.where(on, "", "no ANP record since " + seen).tolist(),
        "lon": k.map(lambda x: key.at[x, "lon"] if x in key.index else np.nan).values,
        "lat": k.map(lambda x: key.at[x, "lat"] if x in key.index else np.nan).values,
        "capacity": last[BM_CAP].values, "capacity_unit": ["m³ biomethane/day"] * n,
        "capacity_note": [f"ANP authorised; biogas capacity {b:,.0f} m³/d, processed {v:,.0f} m³/d"
                          for b, v in zip(last[BM_CAP_BIOGAS], last[BM_VOL_BIOGAS], strict=True)],
        "residue_link": ["processing"] * n, "source": ["ANP biomethane open data"] * n,
        "cutoff": [snapshot] * n, "located": ["municipality"] * n}, n)
    return out


def biomethane_snapshots(capacity: Path, production: Path | None, months: list[str]) -> pd.DataFrame:
    """Per snapshot month: plants recorded, authorised capacity, biogas capacity and processed biogas (m³/d)
    from the capacity file; produced biomethane (m³ in the month and m³/d) from the production file."""
    d = read_biomethane_capacity(capacity)
    prod = None
    if production is not None and Path(production).exists():
        p = _read_csv(production, dtype=str)
        parsed = p[BM_PROD].map(production_m3)
        p["m3"] = [v for v, _ in parsed]
        p["thousands"] = [t for _, t in parsed]
        p["month"] = pd.to_datetime(p[BM_MONTH], format="%m/%Y")
        prod = p
    rows = []
    for m in months:
        mm = pd.to_datetime(m, format="%m/%Y")
        s = d[d["month"] == mm]
        days = mm.days_in_month
        row = {"snapshot": m, "plants": int(s[BM_CNPJ].nunique()), "authorised_m3d": float(s[BM_CAP].sum()),
               "biogas_capacity_m3d": float(s[BM_CAP_BIOGAS].sum()),
               "biogas_processed_m3d": float(s[BM_VOL_BIOGAS].sum()),
               "implied_output_m3d": float((s[BM_VOL_BIOGAS] * s[BM_CAP] / s[BM_CAP_BIOGAS]).sum())}
        if prod is not None:
            q = prod[prod["month"] == mm]
            row |= {"produced_m3_month": float(q["m3"].sum()), "produced_m3d": float(q["m3"].sum()) / days,
                    "produced_compressed_m3_month": float(q.loc[q[BM_PRODUCT] != "BIOMETANO", "m3"].sum()),
                    "production_rows_thousands": int(q["thousands"].sum())}
        rows.append(row)
    return pd.DataFrame(rows)


def production_check(capacity: Path, production: Path, factor: float = 10.0) -> pd.DataFrame:
    """State × month: produced biomethane vs the output implied by the capacity file (processed biogas ×
    biomethane/biogas capacity × days). Rows outside [1/factor, factor] are returned with `outside`."""
    d = read_biomethane_capacity(capacity)
    d["implied_m3"] = d[BM_VOL_BIOGAS] * d[BM_CAP] / d[BM_CAP_BIOGAS] * d["month"].dt.days_in_month
    imp = d.groupby(["month", "uf"])["implied_m3"].sum()
    p = _read_csv(production, dtype=str)
    p["m3"] = [production_m3(v)[0] for v in p[BM_PROD]]
    p["month"] = pd.to_datetime(p[BM_MONTH], format="%m/%Y")
    p["uf"] = p[BM_STATE].map(UF_OF_STATE)
    got = p.groupby(["month", "uf"])["m3"].sum()
    t = pd.concat({"produced_m3": got, "implied_m3": imp}, axis=1).dropna().reset_index()
    t = t[t["implied_m3"] > 0]
    t["ratio"] = t["produced_m3"] / t["implied_m3"]
    t["outside"] = (t["ratio"] > factor) | (t["ratio"] < 1 / factor)
    return t


def read_biogas(path: Path) -> pd.DataFrame:
    g = _shp(path)
    n = len(g)
    sub = g["Categoria"].map({"Agropecuária": "agriculture", "Indústria": "industry", "RSU ou Esgoto": "urban"})
    on = (g["Situacao"] == "Em operação").to_numpy()
    use = g["Principal_"].astype(str)
    gnr = (use == "GNR/Biometano") | (g.get("GNR_Biomet", pd.Series(["Não"] * n)) == "Sim")
    return _frame({
        "record_id": [f"cibiogas:{i}" for i in range(n)], "family": ["biogas"] * n,
        "subtype": sub.fillna("other").values, "name": g.get("Label", [None] * n), "uf": g["Estado"].values,
        "municipality": g["Municipio"].values, "status": g["Situacao"].values, "operating": on,
        "status_reason": np.where(on, "", "CIBiogás status: " + g["Situacao"].astype(str)).tolist(),
        "lon": g.geometry.x.values, "lat": g.geometry.y.values,
        "capacity": g["Escala_pro"].map({"Pequeno porte": 1, "Médio porte": 2, "Grande porte": 3}).values,
        "capacity_unit": ["scale class (1 small, 2 medium, 3 large)"] * n,
        "capacity_note": ("use: " + use + np.where(gnr, "; GNR/biomethane", "")).values,
        "residue_link": ["processing"] * n, "source": ["MapBiomas / CIBiogás biogas plants"] * n,
        "located": ["point"] * n}, n)


def read_biomass_power(path: Path) -> pd.DataFrame:
    g = _shp(path)
    fuel = next(c for c in g.columns if c.startswith("Combust"))
    pot = next(c for c in g.columns if c.startswith("Potência O") or c.startswith("Potencia O"))
    n = len(g)
    mun = g.get("Município", g.get("Municipio"))
    on = (g["Fase"] == "Operação").to_numpy()
    rid = ("aneel:" + g["CEG"].astype(str)).values if "CEG" in g else [f"aneel:{i}" for i in range(n)]
    return _frame({
        "record_id": rid, "family": ["biomass_power"] * n,
        "subtype": g[fuel].map(UTE_SUBTYPE).fillna("forestry").values,
        "name": g["Empreendim"].values, "uf": g["UF"].values,
        "municipality": mun.astype(str).str.split(" - ").str[0].values if mun is not None else [None] * n,
        "status": g["Fase"].values, "operating": on,
        "status_reason": np.where(on, "", "ANEEL phase: " + g["Fase"].astype(str)).tolist(),
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
    on = (g["situacao"] == "Ativo").to_numpy()
    return _frame({
        "record_id": [f"sif:{i}" for i in range(n)], "family": ["slaughterhouse"] * n, "subtype": sub,
        "name": g["razao_soci"].values, "uf": g["uf"].values, "municipality": g["municipio"].values,
        "status": g["situacao"].values, "operating": on,
        "status_reason": np.where(on, "", "SIF status: " + g["situacao"].astype(str)).tolist(),
        "lon": g.geometry.x.values, "lat": g.geometry.y.values,
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


def read_epe_mills(path: Path):
    """MapBiomas ethanol plants (EPE webmap 2018 for most points), operating only."""
    import geopandas as gpd

    g = gpd.read_file(path).to_crs("EPSG:4674")
    return g[g["Situacao"] == "Operando"] if "Situacao" in g else g


# ---------------------------------------------------------------------------
# sites, duplicates, municipality, register cross-check
# ---------------------------------------------------------------------------
def unit_xyz(lon, lat) -> np.ndarray:
    lo, la = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


R_EARTH_KM = 6371.0088


def km_to_chord(km) -> float:
    return float(2 * np.sin(np.asarray(km, float) / (2 * R_EARTH_KM)))


def _close_pairs(a: pd.DataFrame, b: pd.DataFrame | None, km: float) -> list[tuple[int, int]]:
    """Index pairs (i in a, j in b) closer than km (great circle); b=None: pairs within a (i < j)."""
    from scipy.spatial import cKDTree

    if len(a) == 0 or (b is not None and len(b) == 0):
        return []
    ta = cKDTree(unit_xyz(a["lon"], a["lat"]))
    r = km_to_chord(km)
    if b is None:
        return [(a.index[i], a.index[j]) for i, j in sorted(ta.query_pairs(r))]
    tb = cKDTree(unit_xyz(b["lon"], b["lat"]))
    return [(a.index[i], b.index[j]) for i, js in enumerate(ta.query_ball_tree(tb, r)) for j in js]


def _nearest_within(a: pd.DataFrame, b: pd.DataFrame, km: float) -> list[tuple[int, int]]:
    """(i in a, j in b): the nearest row of b to each row of a, where it is closer than km."""
    from scipy.spatial import cKDTree

    if len(a) == 0 or len(b) == 0:
        return []
    d, k = cKDTree(unit_xyz(b["lon"], b["lat"])).query(unit_xyz(a["lon"], a["lat"]),
                                                        distance_upper_bound=km_to_chord(km))
    return [(a.index[i], b.index[kk]) for i, (dd, kk) in enumerate(zip(d, k, strict=True)) if np.isfinite(dd)]


def _select(df: pd.DataFrame, spec: str) -> pd.DataFrame:
    """'family' or 'family:subtype'."""
    fam, _, sub = spec.partition(":")
    s = df[df["family"].astype(str) == fam]
    return s[s["subtype"] == sub] if sub else s


def assign_sites(df: pd.DataFrame, links: list[list[str]], link_km: float) -> pd.Series:
    """site_id per row: rows of one source record (flex ethanol plants) share a site; so do point-located rows
    of the families in a `links` pair closer than link_km, and an ANP biomethane plant (located only by
    municipality) with the single CIBiogás GNR/biomethane plant of the same municipality, if there is one."""
    parent = {i: i for i in df.index}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for _, idx in df.groupby("record_id").groups.items():
        idx = list(idx)
        for j in idx[1:]:
            union(idx[0], j)
    pts = df[(df["located"] == "point") & np.isfinite(df["lon"].astype(float)) & np.isfinite(df["lat"].astype(float))]
    for a, b in links:
        # each row of b joins only its nearest row of a: a power plant between two mills must not merge them
        for j, i in _nearest_within(_select(pts, b), _select(pts, a), link_km):
            union(i, j)
    if "geocode" in df:
        bm = df[df["family"].astype(str) == "biomethane"]
        gnr = df[(df["family"].astype(str) == "biogas")
                 & df["capacity_note"].astype(str).str.contains("GNR/biomethane", regex=False)]
        for i, row in bm.iterrows():
            cand = gnr[gnr["geocode"] == row["geocode"]]
            if len(cand) == 1:
                union(i, cand.index[0])
    roots = pd.Series({i: find(i) for i in df.index})
    order = {r: k for k, r in enumerate(pd.unique(roots.loc[df.index]))}
    return roots.loc[df.index].map(lambda r: f"site{order[r]:04d}")


def duplicate_pairs(df: pd.DataFrame, km: float) -> pd.DataFrame:
    """Pairs of rows of the same family and source closer than km, on different sites (possible duplicates)."""
    out = []
    pts = df[df["located"] == "point"]
    for (fam, src), g in pts.groupby([pts["family"].astype(str), "source"]):
        for i, j in _close_pairs(g, None, km):
            if df.at[i, "site_id"] != df.at[j, "site_id"]:
                out.append({"family": fam, "source": src, "plant_a": df.at[i, "plant_id"],
                            "plant_b": df.at[j, "plant_id"], "name_a": df.at[i, "name"], "name_b": df.at[j, "name"]})
    return pd.DataFrame(out, columns=["family", "source", "plant_a", "plant_b", "name_a", "name_b"])


def municipality_agreement(df: pd.DataFrame, municipalities: pd.DataFrame) -> pd.Series:
    """For point-located rows: does the point fall in the municipality the source declares? True / False;
    <NA> where the declared name is not in the mesh (or the source declares none)."""
    key = municipalities.assign(k=municipalities["uf"].astype(str) + "|" + municipalities["name"].map(_norm))
    code = key.drop_duplicates("k").set_index("k")["geocode"]
    k = df["uf"].astype(str) + "|" + df["municipality"].map(_norm)
    declared = k.map(code)
    out = pd.Series(pd.NA, index=df.index, dtype="boolean")
    ok = declared.notna() & df["geocode"].notna() & (df["located"] == "point")
    out[ok] = declared[ok].astype("int64") == df.loc[ok, "geocode"].astype("int64")
    return out


def register_match(mills: pd.DataFrame, other, km: tuple[float, ...] = (2.0, 10.0)) -> dict[str, float]:
    """Share of `mills` (lon/lat) with a point of `other` (GeoDataFrame) within each distance (km)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(unit_xyz(other.geometry.x.to_numpy(), other.geometry.y.to_numpy()))
    d, _ = tree.query(unit_xyz(mills["lon"], mills["lat"]))
    dkm = 2 * R_EARTH_KM * np.arcsin(np.clip(d / 2, 0, 1))
    return {f"within_{k:g}km": float((dkm <= k).mean()) for k in km}


# ---------------------------------------------------------------------------
# the register
# ---------------------------------------------------------------------------
def _spec(cfg) -> dict:
    return cfg.section("plants") or {}


def _source(spec) -> dict:
    return spec if isinstance(spec, dict) else {"path": spec}


def source_paths(cfg) -> dict[str, Path]:
    """Every configured plant file (sources, their companions and the cross-check registers)."""
    sp = _spec(cfg)
    out = {}
    for key, s in (sp.get("sources") or {}).items():
        s = _source(s)
        out[key] = cfg.raw_root / s["path"]
        if s.get("production"):
            out[f"{key}_production"] = cfg.raw_root / s["production"]
    for key, v in (sp.get("registers") or {}).items():
        out[key] = cfg.raw_root / _source(v)["path"]
    return out


def _municipality_points(cfg) -> pd.DataFrame:
    import geopandas as gpd

    attrs = pd.read_parquet(cfg.data_dir / "interim" / "municipalities.parquet",
                            columns=["geocode", "name", "uf", "cd_rgint", "nm_rgint", "is_municipality"])
    attrs = attrs[attrs["is_municipality"]].drop(columns="is_municipality")
    geom = gpd.read_parquet(cfg.data_dir / "interim" / "municipalities_geom.parquet", columns=["geocode", "geometry"])
    pt = geom.set_index("geocode").geometry.representative_point()
    attrs["lon"], attrs["lat"] = attrs["geocode"].map(pt.x), attrs["geocode"].map(pt.y)
    return attrs


def build_plants(cfg, ck=None):
    """Read every source, keep plants inside Brazil, attach municipality and intermediate region, link sites,
    run the checks and write data/interim/plants.parquet (GeoParquet, EPSG:4674), data/processed/plants.parquet
    (no geometry) and the biomethane snapshots. Returns the GeoDataFrame."""
    import geopandas as gpd

    from bbp.checks import CheckLog

    ck = ck if ck is not None else CheckLog("plants")
    spec = _spec(cfg)
    sources = {k: _source(v) for k, v in (spec.get("sources") or {}).items()}
    mun = _municipality_points(cfg)
    bm = sources.get("anp_biomethane", {})
    snapshot = bm.get("snapshot")
    readers = {"anp_ethanol": read_anp_ethanol, "anp_biodiesel": read_anp_biodiesel,
               "biogas": read_biogas, "biomass_power": read_biomass_power, "slaughterhouses": read_slaughterhouses,
               "anp_biomethane": lambda p: read_anp_biomethane(p, mun, snapshot)}
    parts = []
    for key, reader in readers.items():
        s = sources.get(key)
        p = cfg.raw_root / s["path"] if s else None
        ok = p is not None and p.exists()
        ck.add(f"plants_{key}_found", ok, str(p) if p else "not configured", severity="warn")
        if not ok:
            continue
        df = reader(p)
        cutoff = s.get("snapshot") or s.get("cutoff")
        df["cutoff"] = df["cutoff"].where(df["cutoff"].notna(), cutoff)
        ck.add(f"plants_{key}_cutoff_recorded", bool(cutoff), f"cut-off: {cutoff or 'not recorded in pipeline.yaml'}",
               cutoff, severity="warn")
        parts.append(df)
        log.info("  plants: %-16s %5d rows (%s)", key, len(df), p.name)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df["family"] = pd.Categorical(df["family"], FAMILY_ORDER)
    df = df.sort_values(["family", "uf", "name"], kind="stable").reset_index(drop=True)
    df["plant_id"] = [f"{f[:4]}{i:04d}" for i, f in enumerate(df["family"].astype(str))]
    for c in ("lon", "lat", "capacity"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- inside Brazil: municipality by point, nearest within 5 km for coastal points -------------------
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4674")
    located = np.isfinite(g["lon"]) & np.isfinite(g["lat"])
    ck.add("plants_have_coordinates", bool(located.mean() >= 0.99),
           f"{int((~located).sum())} rows without coordinates (dropped)", int((~located).sum()), severity="warn")
    g = g[located]
    polys = gpd.read_parquet(cfg.data_dir / "interim" / "municipalities_geom.parquet", columns=["geocode", "geometry"])
    polys = polys[polys["geocode"].isin(mun["geocode"])].to_crs("EPSG:4674")
    j = gpd.sjoin(g, polys, how="left", predicate="within").drop(columns="index_right")
    j = j[~j.index.duplicated()]
    miss = j["geocode"].isna()
    if miss.any():
        near = gpd.sjoin_nearest(j.loc[miss, [c for c in j.columns if c != "geocode"]].to_crs("EPSG:5880"),
                                 polys.to_crs("EPSG:5880"), max_distance=5000, how="left").drop(columns="index_right")
        near = near[~near.index.duplicated()]
        j.loc[miss, "geocode"] = near["geocode"].values
    attrs = mun.set_index("geocode")
    j["geocode"] = j["geocode"].astype("Int64")
    j["cd_rgint"] = j["geocode"].map(attrs["cd_rgint"]).astype("Int64")
    j["nm_rgint"] = j["geocode"].map(attrs["nm_rgint"])
    outside = j["geocode"].isna()
    ck.add("plants_inside_brazil", bool(outside.sum() <= 0.01 * len(j)),
           f"{int(outside.sum())} of {len(j)} plants outside every municipality (dropped): "
           + ", ".join(j.loc[outside, "plant_id"].head(10)), int(outside.sum()), severity="warn")
    j = j[~outside].copy()

    # ---- the declared municipality --------------------------------------------------------------------
    j["municipality_match"] = municipality_agreement(j, mun)
    for src, s in j[j["located"] == "point"].groupby("source"):
        known = s["municipality_match"].notna()
        bad = int((s["municipality_match"] == False).sum())  # noqa: E712 (nullable boolean)
        share = bad / max(int(known.sum()), 1)
        ck.add(f"plants_point_in_declared_municipality_{_norm(src).lower().replace(' ', '_')}", share <= 0.05,
               f"{src}: {bad} of {int(known.sum())} points outside the municipality the source declares "
               f"({share:.1%}); {int((~known).sum())} declared names not in the mesh", round(share, 4),
               severity="warn")

    # ---- sites (flex plants, the same site in two registers) and duplicates -----------------------------
    j["site_id"] = assign_sites(j, spec.get("site_links") or [], float(spec.get("site_link_km", 2.0)))
    flex = j.groupby("record_id")["family"].nunique()
    ck.add("plants_flex_counted_once", bool((j.groupby("record_id")["site_id"].nunique() == 1).all()),
           f"{int((flex > 1).sum())} source records in two families (flex ethanol), one site each",
           int((flex > 1).sum()))
    dup = duplicate_pairs(j, float(spec.get("duplicate_km", 2.0)))
    by = dup.groupby("family").size().to_dict() if len(dup) else {}
    ck.add("plants_possible_duplicates", len(dup) <= 0.05 * len(j),
           f"{len(dup)} pairs of the same family and source < {spec.get('duplicate_km', 2.0)} km apart on different "
           f"sites: {by}", by, severity="warn")
    ck.add("plants_sites", True, f"{j['site_id'].nunique()} sites for {len(j)} plant rows "
           f"({len(j) - j['site_id'].nunique()} rows share a site)", int(j["site_id"].nunique()))

    # ---- counts, capacity ------------------------------------------------------------------------------
    counts = j.groupby([j["family"].astype(str), "source"]).agg(rows=("plant_id", "size"),
                                                              operating=("operating", "sum")).reset_index()
    ck.add("plants_count_by_family_and_source", len(counts) > 0,
           "; ".join(f"{r.family} [{r.source}] {r.rows} ({int(r.operating)} operating)" for r in counts.itertuples()),
           {f"{r.family}|{r.source}": [int(r.rows), int(r.operating)] for r in counts.itertuples()})
    required = set(spec.get("capacity_required") or [])
    for fam, s in j.groupby(j["family"].astype(str)):
        n_miss = int(s["capacity"].isna().sum())
        ck.add(f"plants_{fam}_capacity", fam not in required or n_miss == 0,
               f"{n_miss} of {len(s)} without capacity" + ("" if fam in required else " (capacity not required)"),
               n_miss, severity="warn")
    inactive = j[~j["operating"].astype(bool)]
    ck.add("plants_not_operating", True, f"{len(inactive)} rows not operating; biomethane: "
           + "; ".join(f"{r.name} ({r.status_reason})" for r in inactive[inactive["family"] == "biomethane"].itertuples()),
           int(len(inactive)))

    # ---- biomethane snapshots ----------------------------------------------------------------------------
    out_proc = cfg.data_dir / "processed"
    out_proc.mkdir(parents=True, exist_ok=True)
    if bm and snapshot:
        cap = cfg.raw_root / bm["path"]
        prod = cfg.raw_root / bm["production"] if bm.get("production") else None
        months = [*(bm.get("history") or []), snapshot]
        snaps = biomethane_snapshots(cap, prod, months)
        snaps.to_parquet(out_proc / "plants_biomethane_snapshots.parquet")
        for r in snaps.itertuples():
            ck.add(f"plants_biomethane_{r.snapshot.replace('/', '_')}", True,
                   f"{r.plants} plants, {r.authorised_m3d / 1e6:.3f} M m³/d authorised"
                   + (f", {r.produced_m3d / 1e6:.3f} M m³/d produced" if "produced_m3d" in snaps else ""),
                   r._asdict())
        exp = bm.get("expected") or {}
        cur = snaps[snaps["snapshot"] == snapshot].iloc[0]
        for k, v in exp.items():
            if k in cur and pd.notna(cur[k]):
                rel = float(cur[k]) / float(v) - 1
                ck.add(f"plants_biomethane_{k}_matches_expected", abs(rel) <= float(bm.get("expected_tolerance", 0.02)),
                       f"{cur[k]:,.0f} vs expected {v:,.0f} ({rel:+.1%})", round(rel, 4), severity="warn")
        if prod is not None and prod.exists():
            pc = production_check(cap, prod, factor=10.0)       # a misread separator is a factor of 1,000
            n_th = int(snaps["production_rows_thousands"].sum()) if "production_rows_thousands" in snaps else 0
            ck.add("plants_biomethane_production_notation", True,
                   f"{n_th} production values in the snapshot months written with a point as thousands separator "
                   "(e.g. '173.037'), read as thousands", n_th)
            med = pc.groupby("uf")["ratio"].median().round(2).to_dict()
            ck.add("plants_biomethane_production_vs_capacity", not pc["outside"].any(),
                   f"{int(pc['outside'].sum())} of {len(pc)} state-months outside ×10 of the output implied by "
                   f"processed biogas; median produced/implied per state: {med}", med, severity="warn")

    # ---- the cross-check: where are the ANP mills in the other registers? ---------------------------------
    regs = spec.get("registers") or {}
    mills = j[(j["family"].astype(str) == "sugarcane_mill") & j["operating"].astype(bool)]
    match_rows = []
    for key, r in regs.items():
        r = _source(r)
        p = cfg.raw_root / r["path"]
        if not p.exists():
            ck.add(f"plants_register_{key}_found", False, str(p), severity="warn")
            continue
        other = read_conab_mills(p) if key == "conab" else read_epe_mills(p)
        m = register_match(mills, other)
        match_rows.append({"register": key, "label": r.get("label", key), "units": len(other),
                           "anp_mills": len(mills), **m})
        lo = r.get("min_share_2km")
        ck.add(f"plants_anp_mills_in_{key}", lo is None or m["within_2km"] >= float(lo),
               f"{m['within_2km']:.1%} of {len(mills)} operating ANP mills have a {r.get('label', key)} unit within "
               f"2 km ({m['within_10km']:.1%} within 10 km; {len(other)} units)", round(m["within_2km"], 4),
               severity="warn")
    if match_rows:
        pd.DataFrame(match_rows).to_parquet(out_proc / "plants_register_match.parquet")

    out = cfg.data_dir / "interim" / "plants.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    j = j[[c for c in COLUMNS if c in j] + ["municipality_match", "geocode", "cd_rgint", "nm_rgint", "geometry"]]
    j.to_parquet(out)
    pd.DataFrame(j.drop(columns="geometry")).to_parquet(out_proc / "plants.parquet")
    ck.add("plants_written", len(j) > 0, f"{len(j)} rows, {j['site_id'].nunique()} sites", int(len(j)))
    return j


def load_plants(cfg):
    import geopandas as gpd

    p = cfg.data_dir / "interim" / "plants.parquet"
    return gpd.read_parquet(p) if p.exists() else None


def select_plants(plants: pd.DataFrame, family: str, subtype: str | None = None, operating_only: bool = True):
    sel = plants[plants["family"].astype(str) == family]
    if subtype:
        sel = sel[sel["subtype"] == subtype]
    if operating_only and "operating" in sel:
        sel = sel[sel["operating"].fillna(True).astype(bool)]
    return sel
