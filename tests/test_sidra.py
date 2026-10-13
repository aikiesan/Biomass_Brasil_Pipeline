import csv

import numpy as np
import pandas as pd

from bbp.ingest.sidra import (
    parse_sidra_rows,
    parse_value,
    read_table,
    stale_preliminary,
    subset_violations,
    tidy,
    unit_of,
)

ROWS = [
    ["Tabela 5457 - Área plantada ..."],
    ["Variável - Área plantada ou destinada à colheita (Hectares)"],
    ["Nível", "#", "Cód.", "Unidade da Federação e Município", "Ano x Produto"],
    # merged year cells; exporters trim the trailing empties of this row
    ["Nível", "#", "Cód.", "Unidade da Federação e Município", "2013", "", "2014"],
    ["Nível", "#", "Cód.", "Unidade da Federação e Município", "Soja (em grão)", "Laranja", "Soja (em grão)", "Laranja"],
    ["MU", "1", "3550308", "São Paulo (SP)", "...", "-", "12", "X"],
    ["MU", "2", "5107925", "Sorriso (MT)", "600000", "..", "610000.5", "-"],
    [],
    ["Variável - Quantidade produzida (Toneladas)"],
    ["Nível", "#", "Cód.", "Unidade da Federação e Município", "2013"],
    ["Nível", "#", "Cód.", "Unidade da Federação e Município", "Soja (em grão)"],
    ["MU", "1", "5107925", "Sorriso (MT)", "1800000"],
    ["Fonte: IBGE - Produção Agrícola Municipal"],
    ["..", "Valor não se aplica."],
]


def test_symbols():
    assert parse_value("-") == (0.0, "zero")
    v, f = parse_value("...")
    assert np.isnan(v) and f == "not_available"
    assert parse_value("1.234,5") == (1234.5, "ok")
    assert parse_value("42") == (42.0, "ok")


def test_parse_blocks_and_trimmed_year_row():
    df = parse_sidra_rows(ROWS, 5457)
    assert set(df["variable"]) == {"area_planted_ha", "production_t"}
    a = df[df["variable"] == "area_planted_ha"]
    assert len(a) == 8  # 2 municipalities x 2 years x 2 products
    # columns beyond the trimmed year row belong to the last year (2014)
    laranja_2014 = a[(a["geocode"] == 3550308) & (a["year"] == 2014) & (a["product"] == "Laranja")]
    assert laranja_2014["flag"].item() == "suppressed"
    sorriso = a[(a["geocode"] == 5107925) & (a["year"] == 2014) & (a["product"] == "Soja (em grão)")]
    assert sorriso["value"].item() == 610000.5
    p = df[df["variable"] == "production_t"]
    assert p["value"].item() == 1_800_000


def test_tidy_later_file_wins_on_overlap():
    # a table split over several exports (e.g. PAM 2025 added as its own file): where two files
    # cover the same cell, the one listed later in pipeline.yaml (the newer export) is kept
    old = parse_sidra_rows(ROWS, 5457).assign(source_file="old.csv")
    new = old[(old["geocode"] == 5107925) & (old["year"] == 2013)].assign(value=1.0, source_file="new.csv")
    t = tidy(pd.concat([old, new], ignore_index=True), {})
    cell = t[(t["geocode"] == 5107925) & (t["year"] == 2013) & (t["product"] == "Soja (em grão)")]
    assert (cell["source_file"] == "new.csv").all() and (cell["value"] == 1.0).all()
    assert len(t) == len(old)


def test_tidy_maps_crops():
    df = parse_sidra_rows(ROWS, 5457)
    crops = {"soybean": {"pam_table": 5457, "pam_product": "Soja (em grão)"}}
    t = tidy(df, crops)
    assert set(t.loc[t["product"] == "Soja (em grão)", "crop"]) == {"soybean"}
    assert t.loc[t["product"] == "Laranja", "crop"].isna().all()


# PPM 94 layout: one variable, no product row — data follow the year row directly
ROWS_NO_PRODUCT = [
    ["Tabela 94 - Vacas ordenhadas"],
    ["Variável - Vacas ordenhadas (Cabeças)"],
    ["Nível", "Cód.", "Município", "Ano"],
    ["", "", "", "2024", "2025"],
    ["MU", "1100015", "Alta Floresta D'Oeste (RO)", "2464", "2500"],
    ["MU", "1100023", "Ariquemes (RO)", "...", "-"],
]


def test_table_without_product_row_keeps_first_municipality():
    df = parse_sidra_rows(ROWS_NO_PRODUCT, 94)
    assert len(df) == 4
    assert set(df["geocode"]) == {1100015, 1100023}  # the first data row is not taken as a header
    assert set(df["product"]) == {"Vacas ordenhadas (Cabeças)"}
    assert set(df["unit"]) == {"Cabeças"}
    assert df.loc[(df["geocode"] == 1100015) & (df["year"] == 2025), "value"].item() == 2500


def test_products_keep_and_units():
    df = parse_sidra_rows(ROWS, 5457, products_keep={"Laranja"})
    assert set(df["product"]) == {"Laranja"}
    assert set(df["unit"]) == {"Hectares"}  # from the variable, not from "Soja (em grão)"
    assert unit_of("Produção de origem animal", "Leite (Mil litros)") == "Mil litros"
    assert unit_of("Produção da aquicultura", "Peixes") is None


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return path


def _ppm_rows(years, values):
    return [
        ["Variável - Efetivo dos rebanhos (Cabeças)"],
        ["Nível", "Cód.", "Município", "Ano x Tipo de rebanho"],
        ["", "", "", *[y for y in years for _ in range(2)]],
        ["", "", "", *(["Suíno - total", "Suíno - matrizes de suínos"] * len(years))],
        ["MU", "1100015", "A (RO)", *values],
    ]


def test_read_table_later_file_wins_and_flags_preliminary(tmp_path):
    old = _write_csv(tmp_path / "old_2023_2024.csv", _ppm_rows(["2023", "2024"], ["100", "10", "200", "20"]))
    new = _write_csv(tmp_path / "new_2025.csv", _ppm_rows(["2025"], ["300", "30"]))
    df, meta = read_table(3939, [old, new])
    assert sorted(df["year"].unique()) == [2023, 2024, 2025]
    assert df.loc[df["preliminary"], "year"].unique().tolist() == [2025]
    # 2024 was the newest (preliminary) year of old_2023_2024.csv: flagged as possibly revised since
    assert stale_preliminary(df, meta["file_years"]) == {"old_2023_2024.csv": 2}

    revised = _write_csv(tmp_path / "revised_2024_2025.csv", _ppm_rows(["2024", "2025"], ["210", "21", "300", "30"]))
    df, meta = read_table(3939, [old, revised])
    assert meta["overlapping_cells_replaced"] == 2
    v2024 = df[(df["year"] == 2024) & (df["product"] == "Suíno - total")]
    assert v2024["value"].item() == 210 and v2024["source_file"].item() == "revised_2024_2025.csv"
    assert stale_preliminary(df, meta["file_years"]) == {}


def test_subset_violations():
    df = parse_sidra_rows(_ppm_rows(["2024", "2025"], ["100", "10", "50", "60"]), 3939)
    bad = subset_violations(df, "Suíno - total", ["Suíno - matrizes de suínos"])
    assert bad.index.get_level_values("year").tolist() == [2025]
