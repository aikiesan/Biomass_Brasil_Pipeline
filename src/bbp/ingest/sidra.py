"""IBGE SIDRA exports (PAM tables 5457 and 839) -> long table.

SIDRA exports (CSV or XLSX) share one layout, repeated once per variable:

    Variável - <name> (<unit>)
    Nível | # | Cód. | Município | Ano x Produto ...      <- skipped
          |   |      |           | 2008 |      | 2009 ...   <- years (merged cells)
          |   |      |           | Arroz| Café | Arroz ...  <- products
    MU    | 1 |1100015| Alta Floresta ... | 12 | - | ...    <- data
    ...
    Legenda / notes                                         <- ignored

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
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

YEAR_RE = re.compile(r"^(19|20)\d\d$")
CODE_RE = re.compile(r"^\d{7}$")

SYMBOLS = {"-": ("zero", 0.0), "...": ("not_available", np.nan), "..": ("not_applicable", np.nan), "X": ("suppressed", np.nan)}

VARIABLES = {
    "Área plantada ou destinada à colheita (Hectares)": "area_planted_ha",
    "Área plantada (Hectares)": "area_planted_ha",
    "Área colhida (Hectares)": "area_harvested_ha",
    "Quantidade produzida (Toneladas)": "production_t",
    "Valor da produção (Mil Reais)": "value_thousand_brl",
    "Rendimento médio da produção (Quilogramas por Hectare)": "yield_kg_ha",
}


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


def parse_sidra_rows(rows: Iterable[Sequence], table: int, variable: str | None = None) -> pd.DataFrame:
    """Parse rows of a SIDRA export. `variable` is used when the sheet has no 'Variável -' line."""
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
        if years is not None and products is None:
            products = cells
            continue
        if products is None:
            continue

        code_i = next((i for i, c in enumerate(cells[:4]) if CODE_RE.match(c)), None)
        if code_i is None:
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
            value, flag = parse_value(cells[j])
            records.append((code, name, int(year), products[j], var, value, flag))

    df = pd.DataFrame.from_records(records, columns=["geocode", "name", "year", "product", "variable_raw", "value", "flag"])
    df["table"] = table
    df["variable"] = df["variable_raw"].map(VARIABLES).fillna(df["variable_raw"])
    return df


def read_sidra_csv(path: Path, table: int) -> pd.DataFrame:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    df = parse_sidra_rows(rows, table)
    df["source_file"] = Path(path).name
    return df


def read_sidra_xlsx(path: Path, table: int) -> pd.DataFrame:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    frames = []
    try:
        for ws in wb.worksheets:
            if ws.title.lower().startswith("nota"):
                continue
            df = parse_sidra_rows(ws.iter_rows(values_only=True), table)
            if not df.empty:
                frames.append(df)
    finally:
        wb.close()
    out = pd.concat(frames, ignore_index=True)
    out["source_file"] = Path(path).name
    return out


def tidy(df: pd.DataFrame, crops: dict) -> pd.DataFrame:
    """Attach the pipeline crop key and drop duplicate (geocode, year, product, variable) rows."""
    product_to_crop = {
        (c["pam_table"], c["pam_product"]): key for key, c in crops.items() if c.get("pam_product")
    }
    df = df.copy()
    if "source_file" not in df:
        df["source_file"] = None
    df["crop"] = [product_to_crop.get((t, p)) for t, p in zip(df["table"], df["product"], strict=True)]
    df = df.drop_duplicates(["table", "geocode", "year", "product", "variable"], keep="last")
    cols = ["table", "geocode", "name", "year", "product", "crop", "variable", "value", "flag", "source_file"]
    return df[cols].sort_values(["table", "variable", "product", "geocode", "year"]).reset_index(drop=True)
