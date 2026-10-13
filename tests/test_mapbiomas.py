import pytest

from bbp.ingest.mapbiomas import parse_coverage_rows

HEADER = ["ID", "country", "biome", "region", "state", "geocode", "municipality", "municipality-state", "class",
          "class_level_0", "class_level_1", "class_level_2", "class_level_3", "class_level_4", "y2007", "y2008", "y2009"]


def row(geocode, biome, cls, label, *vals):
    return [1, "Brasil", biome, "Sul", "SP", geocode, "X", "X - SP", cls, "a", "b", "c", "d", label, *vals]


def test_parse_keeps_requested_years_and_labels():
    rows = [
        HEADER,
        row(3550308, "Mata Atlântica", 39, "3.2.1.1. Soybean", 1.0, 2.0, 3.0),
        row(3550308, "Cerrado", 39, "3.2.1.1. Soybean", 10.0, 20.0, None),
        row(5107925, "Amazônia", 20, "3.2.1.2. Sugar cane", 5.0, 6.0, 7.0),
    ]
    df = parse_coverage_rows(rows, [2008, 2009, 2010])
    assert sorted(df["year"].unique()) == [2008, 2009]  # 2010 absent from the sheet, 2007 not requested
    assert len(df) == 5  # one None cell dropped
    soy = df[(df["geocode"] == 3550308) & (df["year"] == 2008)]
    assert soy["area_ha"].sum() == 22.0  # summed over biomes later; both biome rows kept
    assert set(df["class_label"].astype(str)) == {"3.2.1.1. Soybean", "3.2.1.2. Sugar cane"}


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        parse_coverage_rows([["geocode", "y2008"], [1, 2]], [2008])
