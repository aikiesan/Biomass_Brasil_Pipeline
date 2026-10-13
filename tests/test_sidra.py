import numpy as np

from bbp.ingest.sidra import parse_sidra_rows, parse_value, tidy

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


def test_tidy_maps_crops():
    df = parse_sidra_rows(ROWS, 5457)
    crops = {"soybean": {"pam_table": 5457, "pam_product": "Soja (em grão)"}}
    t = tidy(df, crops)
    assert set(t.loc[t["product"] == "Soja (em grão)", "crop"]) == {"soybean"}
    assert t.loc[t["product"] == "Laranja", "crop"].isna().all()
