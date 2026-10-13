"""IBGE SIDRA exports (PAM, PPM, any municipal table) -> long table.

SIDRA exports (CSV or XLSX) share one layout, repeated once per variable:

    Variável - <name> (<unit>)
    Nível | # | Cód. | Município | Ano x Produto ...      <- skipped
          |   |      |           | 2008 |      | 2009 ...   <- years (merged cells)
          |   |      |           | Arroz| Café | Arroz ...  <- products
    MU    | 1 |1100015| Alta Floresta ... | 12 | - | ...    <- data
    ...
    Legenda / notes                                         <- ignored

Tables with no product dimension (e.g. PPM 94, milked cows) have no product
row: data follow the year row directly. Their product is set to the variable
name.

The unit is taken from the variable text, "(Hectares)", or, when the variable
has none (PPM 74 and 3940 put it in the product: "Leite (Mil litros)"), from
the product text.

SIDRA special symbols are kept as flags and never turned into zeros, except
"-" which SIDRA defines as a true zero:

    "-"    zero              -> 0.0
    "..."  not available     -> NaN
    ".."   not applicable    -> NaN
    "X"    suppressed        -> NaN
"""

from __future__ import annotations

import csv
import re
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

YEAR_RE = re.compile(r"^(19|20)\d\d$")
CODE_RE = re.compile(r"^\d{7}$")
UNIT_RE = re.compile(r"\(([^()]*)\)\s*$")

SYMBOLS = {"-": ("zero", 0.0), "...": ("not_available", np.nan), "..": ("not_applicable", np.nan), "X": ("suppressed", np.nan)}

VARIABLES = {
    "Área plantada ou destinada à colheita (Hectares)": "area_planted_ha",
    "Área plantada (Hectares)": "area_planted_ha",
    "Área colhida (Hectares)": "area_harvested_ha",
    "Quantidade produzida (Toneladas)": "production_t",
    "Valor da produção (Mil Reais)": "value_thousand_brl",
    "Rendimento médio da produção (Quilogramas por Hectare)": "yield_kg_ha",
    "Área destinada à colheita (Hectares)": "area_planted_ha",
}


def unit_of(variable_raw: str, product: str) -> str | None:
    m = UNIT_RE.search(variable_raw) or UNIT_RE.search(product)
    return m.group(1).strip() if m else None


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def parse_value(raw: str) -> tuple[float, str]:
    if raw in SYMBOLS:
        flag, value = SYMBOLS[raw]
        return value, flag
    if raw == "":
        return np.nan, "empty"
    try:
        return float(raw), "ok"
    except ValueError:
        pass
    try:  # Brazilian formatting: 1.234,5
        return float(raw.replace(".", "").replace(",", ".")), "ok"
    except ValueError:
        return np.nan, f"unparsed:{raw}"


def parse_sidra_rows(rows: Iterable[Sequence], table: int, variable: str | None = None,
                     products_keep: Collection[str] | None = None) -> pd.DataFrame:
    """Parse rows of a SIDRA export. `variable` is used when the sheet has no 'Variável -' line.

    `products_keep`, when given, keeps only those products (columns are skipped while parsing, so a
    table with dozens of products stays cheap to read).
    """
    records: list[tuple] = []
    var = variable
    years: list[str | None] | None = None
    products: list[str] | None = None

    for row in rows:
        cells = [_cell(c) for c in row]
        if not any(cells):
            continue
        first = cells[0]
        if first.startswith("Variável - "):
            var = first[len("Variável - "):].strip()
            years, products = None, None
            continue
        if var is None:
            continue

        year_cols = [i for i, c in enumerate(cells) if YEAR_RE.match(c)]
        if year_cols and years is None and not any(CODE_RE.match(c) for c in cells[:4]):
            years = [None] * len(cells)
            current = None
            for i, c in enumerate(cells):
                if YEAR_RE.match(c):
                    current = c
                if i >= year_cols[0]:
                    years[i] = current
            continue
        code_i = next((i for i, c in enumerate(cells[:4]) if CODE_RE.match(c)), None)
        if years is not None and products is None:
            if code_i is None:
                products = cells
                continue
            products = [var] * max(len(years), len(cells))  # no product dimension: data follow the year row
        if products is None or code_i is None:
            continue
        code = int(cells[code_i])
        name = cells[code_i + 1] if code_i + 1 < len(cells) else ""
        last_year = next((y for y in reversed(years) if y is not None), None)
        for j in range(code_i + 2, len(cells)):
            # the year row is a merged cell: exporters trim its trailing empty cells,
            # so columns past its end belong to the last year
            year = years[j] if j < len(years) else last_year
            if year is None or j >= len(products) or not products[j]:
                continue
            if products_keep is not None and products[j] not in products_keep:
                continue
            value, flag = parse_value(cells[j])
            records.append((code, name, int(year), products[j], var, value, flag))

    df = pd.DataFrame.from_records(records, columns=["geocode", "name", "year", "product", "variable_raw", "value", "flag"])
    df["table"] = table
    df["variable"] = df["variable_raw"].map(VARIABLES).fillna(df["variable_raw"])
    pairs = df[["variable_raw", "product"]].drop_duplicates()
    units = {(v, p): unit_of(v, p) for v, p in zip(pairs["variable_raw"], pairs["product"], strict=True)}
    df["unit"] = [units[k] for k in zip(df["variable_raw"], df["product"], strict=True)]
    return df


def read_sidra_csv(path: Path, table: int, products_keep: Collection[str] | None = None) -> pd.DataFrame:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    df = parse_sidra_rows(rows, table, products_keep=products_keep)
    df["source_file"] = Path(path).name
    return df


def read_sidra_xlsx(path: Path, table: int, products_keep: Collection[str] | None = None) -> pd.DataFrame:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    frames = []
    try:
        for ws in wb.worksheets:
            if ws.title.lower().startswith("nota"):
                continue
            df = parse_sidra_rows(ws.iter_rows(values_only=True), table, products_keep=products_keep)
            if not df.empty:
                frames.append(df)
    finally:
        wb.close()
    out = pd.concat(frames, ignore_index=True) if frames else parse_sidra_rows([], table)
    out["source_file"] = Path(path).name
    return out


def read_sidra(path: Path, table: int, products_keep: Collection[str] | None = None) -> pd.DataFrame:
    """CSV or XLSX, by extension."""
    if Path(path).suffix.lower() == ".csv":
        return read_sidra_csv(path, table, products_keep)
    return read_sidra_xlsx(path, table, products_keep)


def tidy(df: pd.DataFrame, crops: dict) -> pd.DataFrame:
    """Attach the pipeline crop key and drop duplicate (geocode, year, product, variable) rows."""
    product_to_crop = {
        (c["pam_table"], c["pam_product"]): key for key, c in crops.items() if c.get("pam_product")
    }
    df = df.copy()
    for col in ("product", "variable", "name", "unit", "flag", "source_file", "variable_raw"):
        if col in df and isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(object)
    if "source_file" not in df:
        df["source_file"] = None
    df["geocode"], df["year"] = df["geocode"].astype("int64"), df["year"].astype("int64")
    df["crop"] = [product_to_crop.get((t, p)) for t, p in zip(df["table"], df["product"], strict=True)]
    df = df.drop_duplicates(["table", "geocode", "year", "product", "variable"], keep="last")
    for col, default in (("unit", None), ("preliminary", False)):
        if col not in df:
            df[col] = default
    cols = ["table", "geocode", "name", "year", "product", "crop", "variable", "unit", "value", "flag", "preliminary",
            "source_file"]
    return df[cols].sort_values(["table", "variable", "product", "geocode", "year"]).reset_index(drop=True)


KEY = ["geocode", "year", "product", "variable"]
CATEGORICAL = ["name", "product", "variable", "variable_raw", "unit", "flag", "source_file"]


def read_table(table: int, files: Sequence[Path], products: Collection[str] | None = None,
               exclude_products: Collection[str] = (), drop_variables: Collection[str] = ()) -> tuple[pd.DataFrame, dict]:
    """All exports of one SIDRA table as one long frame, plus metadata for the checks.

    Files are read in the order given; where two cover the same cell the LATER file wins. The newest
    year of the series is flagged `preliminary` (IBGE: "Os dados do último ano divulgado são
    RESULTADOS PRELIMINARES").
    """
    frames, file_years = [], {}
    for path in files:
        df = read_sidra(path, table, products)
        df = df[~df["product"].isin(list(exclude_products)) & ~df["variable"].isin(list(drop_variables))]
        file_years[Path(path).name] = [int(df["year"].min()), int(df["year"].max())] if len(df) else None
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)
    df = df.drop_duplicates(KEY, keep="last").reset_index(drop=True)
    df["preliminary"] = df["year"] == df["year"].max() if len(df) else pd.Series(dtype=bool)
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    df["geocode"] = df["geocode"].astype("int32")
    df["year"] = df["year"].astype("int16")
    meta = {"file_years": file_years, "overlapping_cells_replaced": n_raw - len(df)}
    return df, meta


def stale_preliminary(df: pd.DataFrame, file_years: dict) -> dict[str, int]:
    """Cells of the year before the newest that come from an export in which that year was the newest.

    IBGE revises the preliminary year at the next release, so such cells may be outdated: re-export
    that year. Returns {source_file: n_cells}.
    """
    if df.empty:
        return {}
    prev = int(df["year"].max()) - 1
    stale_files = {f for f, yrs in file_years.items() if yrs and yrs[1] == prev}
    sel = df[(df["year"] == prev) & df["source_file"].isin(stale_files)]
    return {str(k): int(v) for k, v in sel["source_file"].value_counts().items() if v}


def subset_violations(df: pd.DataFrame, total: str, parts: Sequence[str], tol: float = 1.0) -> pd.DataFrame:
    """(geocode, year, variable) where the parts add up to more than the total (+tol)."""
    w = df[df["product"].isin([total, *parts])].pivot_table(index=["geocode", "year", "variable"], columns="product",
                                                             values="value", aggfunc="first", observed=True)
    if total not in w or not set(parts) <= set(w.columns):
        return w.iloc[0:0]
    s = w[list(parts)].sum(axis=1, min_count=len(parts))
    bad = s.notna() & w[total].notna() & (s > w[total] + tol)
    return w[bad]
