import numpy as np
import pandas as pd
import pytest

from bbp.config import load_config
from bbp.ingest.sanitation import (
    clean_code,
    code6_index,
    continuity,
    crosswalk,
    harmonise,
    identity_breaks,
    parse_number,
    parse_sinisa_published,
    parse_sinisa_rows,
    parse_snis_rows,
    sums_by_uf,
)

SNIS_ROWS = [
    ["Código do IBGE", "Código do Município", "Município", "POP_TOT - População total", "CO111 - RDO",
     "CO115 - RPU", "CO119 - Total"],
    ["", "355030", "São Paulo", "11.451.245", "3.000.000,00", "1.000.000,00", "4.000.000,00"],
    ["", "510792", "Sorriso", "110.635", "", "", ""],                       # did not report waste
    ["", "130260", "Manaus", "2.063.547", "0", "10", "500"],               # a reported zero; broken identity
    ["TOTAL da AMOSTRA:", "", "", "13.625.427", "3.000.000,00", "1.000.010,00", "4.000.500,00"],
]

SINISA_ROWS = [
    ["MINISTÉRIO DAS CIDADES"],
    ["Legenda", "Data de Geração: 18/03/2025"],
    ["RESPONDEU", "CÓDIGO DO IBGE", "MUNICÍPIO", "POPULAÇÃO TOTAL", "Massa RDO", "Massa LU", "Massa RSU"],
    [None, None, None, None, "Toneladas/Ano", "Toneladas/Ano", "Toneladas/Ano"],
    ["Sim/Não", "Cod_IBGE", "Nom_Mun", "DFE0001", "GTR1025*", "GTR1027 ", "GTR1028\xa0"],
    ["Sim", 3550308, "São Paulo", 11451999, 3100000.5, 900000, 4000000.5],
    ["Não", "5107925", "Sorriso", "110635", "null", "null", "null"],
    ["Sim", 1302603, "Manaus", 2063689, 0, 10, 5],
    [None, None, "Fonte: SINISA"],
]


def test_cells():
    assert clean_code("GTR1028* ") == "GTR1028"
    assert clean_code("CO119 - Quantidade total") == "CO119"
    assert clean_code(" CAD1000\xa0") == "CAD1000"
    assert parse_number("8.000,00") == (8000.0, "ok")
    assert parse_number("203.080", brazilian=True) == (203080.0, "ok")   # SNIS: a lone dot is thousands
    assert parse_number("11.95") == (11.95, "ok")                         # SINISA text: decimal point
    assert parse_number(12) == (12.0, "ok")
    assert parse_number("0") == (0.0, "ok")              # a reported zero stays a zero
    for blank in ("", None, "null", " NULL "):
        v, f = parse_number(blank)
        assert np.isnan(v) and f == "not_reported"      # blank is never zero
    assert parse_number("abc")[1].startswith("unparsed")


def test_snis_rows_skip_blanks_and_keep_the_total_row_aside():
    df, meta = parse_snis_rows(iter(SNIS_ROWS), 2022, ["POP_TOT", "CO111", "CO115", "CO119", "ES006"])
    assert meta["summary_rows"] == 1
    assert meta["totals"]["CO119"] == 4000500.0
    assert meta["missing_fields"] == ["ES006"]
    assert set(df["code6"]) == {355030, 510792, 130260}
    assert not ((df["code6"] == 510792) & (df["field"] == "CO111")).any()   # Sorriso did not report
    assert df.loc[(df["code6"] == 130260) & (df["field"] == "CO111"), "value"].item() == 0.0
    assert (df["year"] == 2022).all()


def test_code6_index():
    assert code6_index([3550308, 5107925])[355030] == 3550308
    with pytest.raises(ValueError):
        code6_index([3550308, 3550309])


def test_sinisa_rows():
    df, meta = parse_sinisa_rows(iter(SINISA_ROWS), 2023, ["DFE0001", "GTR1025", "GTR1027", "GTR1028", "GTR9999"])
    assert meta["rows"] == 3 and meta["respondents"] == 2 and meta["duplicates"] == 0
    assert meta["missing_fields"] == ["GTR9999"]
    assert set(df["geocode"]) == {3550308, 5107925, 1302603}
    assert not ((df["geocode"] == 5107925) & (df["field"] == "GTR1028")).any()   # "null" is not reported
    assert df.loc[(df["geocode"] == 5107925) & (df["field"] == "DFE0001"), "value"].item() == 110635.0


def test_identity_breaks():
    df, _ = parse_sinisa_rows(iter(SINISA_ROWS), 2023, ["GTR1025", "GTR1027", "GTR1028"])
    bad = identity_breaks(df, "GTR1028", ["GTR1025", "GTR1027"])
    assert bad.reset_index()["geocode"].tolist() == [1302603]       # 5 vs 0 + 10
    assert identity_breaks(df[df["geocode"] != 1302603], "GTR1028", ["GTR1025", "GTR1027"]).empty


def test_published_consolidation():
    rows = [["Módulo"], ["UF/Macrorregião", "Sigla", "Volume tratado"], [None, None, "1.000 m³/ano"],
            ["UF/Macrorregião", "Sigla", "GTE1014"], ["Brasil", "BR", 100.5], ["Sudeste", "SE", 60.0],
            ["São Paulo", "SP", "40,5"], ["Sergipe", "SE", 1.0], ["Nota", None, None]]
    # Sudeste and Sergipe share "SE": the macro-region is told apart by name
    assert parse_sinisa_published(iter(rows), ["GTE1014"]) == {
        "BR": {"GTE1014": 100.5}, "R:SE": {"GTE1014": 60.0}, "SP": {"GTE1014": 40.5}, "SE": {"GTE1014": 1.0}}


def test_sums_by_uf():
    df = pd.DataFrame({"geocode": [3550308, 3509502, 1302603], "field": ["X"] * 3, "value": [1.0, 2.0, 4.0]})
    uf = pd.Series({3550308: "SP", 3509502: "SP", 1302603: "AM"})
    assert sums_by_uf(df, uf, ["X"]) == {"AM": {"X": 4.0}, "SP": {"X": 3.0}, "BR": {"X": 7.0}}
    region = pd.Series({3550308: "SE", 3509502: "SE", 1302603: "N"})
    assert sums_by_uf(df, uf, ["X"], region)["R:SE"] == {"X": 3.0}


def test_crosswalk_from_config_is_one_to_one():
    cfg = load_config()
    xw = crosswalk(cfg.sanitation["variables"])
    assert not xw.duplicated(["survey", "module", "field"]).any()
    msw = xw[xw["variable"] == "msw_collected_t"].set_index("survey")["field"].to_dict()
    assert msw == {"SNIS": "CO119", "SINISA": "GTR1028"}
    assert set(xw["match"]) <= {"equivalent", "similar", "snis_only"}


def test_harmonise_and_continuity():
    xw = crosswalk({"msw_collected_t": {"unit": "t/yr", "match": "equivalent", "snis": "CO119",
                                        "sinisa": {"field": "GTR1028", "module": "waste"}}})
    long = pd.DataFrame({
        "geocode": [1, 2, 3, 1, 2, 4, 1],
        "year": [2022, 2022, 2022, 2023, 2023, 2023, 2023],
        "survey": ["SNIS"] * 3 + ["SINISA"] * 4,
        "module": ["snis"] * 3 + ["waste"] * 4,
        "field": ["CO119"] * 3 + ["GTR1028"] * 3 + ["GTR1025"],   # GTR1025 is not in this crosswalk
        "value": [100.0, 200.0, 50.0, 110.0, 180.0, 70.0, 90.0],
        "source_file": ["a"] * 7,
    })
    h = harmonise(long, xw)
    assert len(h) == 6 and set(h["variable"]) == {"msw_collected_t"}
    c = continuity(h).iloc[0]
    assert (c["snis_year"], c["sinisa_year"], c["n_both"]) == (2022, 2023, 2)
    assert c["median_ratio"] == pytest.approx((1.1 + 0.9) / 2)
    assert c["ratio_of_totals"] == pytest.approx(290 / 300)
