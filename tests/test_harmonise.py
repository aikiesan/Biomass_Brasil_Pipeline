import pandas as pd

from bbp.harmonise import build_amc, check_changes_against_mesh, mesh_mismatch_flags

CHANGES = [
    {"code": 5006275, "name": "Paraíso das Águas", "uf": "MS", "parents": [5000203, 5002951, 5003256], "installed": 2013},
    {"code": 5101837, "name": "Boa Esperança do Norte", "uf": "MT", "parents": [5107925, 5106240], "installed": 2025},
]
CODES = pd.Series([5000203, 5002951, 5003256, 5006275, 5107925, 5106240, 5101837, 3550308])


def test_amc_groups():
    amc = build_amc(CODES, CHANGES).set_index("geocode")
    assert amc.loc[5006275, "amc_id"] == 5000203 and amc.loc[5006275, "amc_size"] == 4
    assert amc.loc[5101837, "amc_id"] == 5101837 and amc.loc[5106240, "amc_id"] == 5101837
    assert amc.loc[3550308, "amc_size"] == 1


def test_mismatch_years():
    m = mesh_mismatch_flags([2012, 2013, 2024], CHANGES, {5006275: 2013})
    pairs = set(map(tuple, m.to_numpy()))
    assert (5006275, 2012) in pairs and (5003256, 2012) in pairs
    assert (5006275, 2013) not in pairs
    assert (5107925, 2024) in pairs  # Boa Esperança do Norte only reported from 2025


def test_changes_checked_against_mesh():
    mesh = pd.DataFrame({"geocode": CODES, "name": ["a", "b", "c", "Paraíso das Águas", "e", "f", "Boa Esperança", "SP"],
                         "uf": ["MS", "MS", "MS", "MS", "MT", "MT", "MT", "SP"]})
    problems = check_changes_against_mesh(CHANGES, mesh)
    assert len(problems) == 1 and "5101837" in problems[0]
