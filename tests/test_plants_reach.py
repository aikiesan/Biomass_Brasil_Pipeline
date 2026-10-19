"""bbp.plants (parsing), bbp.reach (rings, required radius, distances) and the per-element export of figures."""

import numpy as np
import pandas as pd
import pytest

from bbp import landcover as lc
from bbp import plants, reach

RES = 0.00026949458523585647


def test_numbers_in_every_source_format():
    assert plants._num("5,500.00") == 5500.0           # ANEEL
    assert plants._num("1.300,5") == 1300.5            # Brazilian
    assert plants._num(1300.0) == 1300.0
    assert np.isnan(plants._num("-")) and np.isnan(plants._num(None))
    assert plants._norm("São João da Boa Vista") == "SAO JOAO DA BOA VISTA"


def test_anp_ethanol_splits_cane_and_corn(tmp_path):
    csv = tmp_path / "e.csv"
    csv.write_text(
        "FID;Região;UF;Nome;Cidade;Latitude;Longitude;Tipo;Situação;"
        "Classe de Capacidade de processamento de cana (toneladas);Capacidade de processamento de milho (toneladas);"
        "Capacidade de Produção de Anidro (m3/dia);Capacidade de Produção de Hidratado (m3/dia);x;y\n"
        "1;SE;SP;A;Ribeirão Preto;-21.1;-47.8;Etanol 1G e Açúcar;Em operação;Entre 2.500.000 e 3.500.000;0;100;200;0;0\n"
        "2;CO;MT;B;Sorriso;-12.5;-55.7;Etanol Milho Flex;Construção;Menor que 1.500.000;500000;0;300;0;0\n"
        "3;CO;MT;C;Sinop;-11.8;-55.5;Etanol Milho Full;Em operação;0;800000;0;600;0;0\n", encoding="utf-8")
    d = plants.read_anp_ethanol(csv)
    cane, corn = d[d["family"] == "sugarcane_mill"], d[d["family"] == "corn_ethanol"]
    assert sorted(cane["name"]) == ["A", "B"] and sorted(corn["name"]) == ["B", "C"]     # flex: both
    assert cane.set_index("name").loc["A", "capacity"] == 3.0e6
    assert corn.set_index("name").loc["C", "capacity"] == 800000
    assert not cane.set_index("name").loc["B", "operating"]


def _grid(tmp_path):
    """A counts raster: sugarcane in a 0.1° block, pasture elsewhere (factor 4 cells)."""
    import rasterio
    from rasterio.transform import from_origin

    cov = np.full((800, 800), 15, dtype=np.uint8)                   # pasture
    cov[380:420, 380:420] = 20                                     # sugarcane block in the middle
    with rasterio.open(tmp_path / "cov.tif", "w", driver="GTiff", height=800, width=800, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(-48.0, -21.0, RES, RES), nodata=0) as dst:
        dst.write(cov, 1)
    lc.aggregate(tmp_path / "cov.tif", tmp_path / "out", "g", factors=(4,), band_rows=160, chunk_cols=160,
                 counts=True)
    return reach.Grid(lc.counts_path(tmp_path / "out", "g", 4)), cov


def test_rings_hold_every_pixel_within_reach(tmp_path):
    grid, cov = _grid(tmp_path)
    assert grid.factor == 4
    cane_ha = grid.area_ha("sugarcane").sum()
    assert cane_ha == pytest.approx(1600 * grid.px_ha.mean(), rel=1e-3)
    centre = (-48.0 + 400 * RES, -21.0 - 400 * RES)
    p = pd.DataFrame({"lon": [centre[0]], "lat": [centre[1]]})
    rings = reach.plant_rings(grid, p, ring_km=1.0, max_km=30.0)
    gi = grid.groups.index("sugarcane")
    assert rings[0, gi].sum() == pytest.approx(cane_ha, rel=1e-6)    # the block is 1.2 km wide: all within 30 km
    assert rings[0, gi, 2:].sum() == 0
    need = np.array([cane_ha / 2, cane_ha * 10])
    rr = reach.required_radius(np.cumsum(rings[[0, 0], gi], axis=1), need, 1.0)
    assert rr[0] in (1.0, 2.0) and np.isinf(rr[1])


def test_nearest_distances_are_great_circle():
    from scipy.spatial import cKDTree

    tree = cKDTree(reach.unit_xyz(np.array([-47.0]), np.array([-22.0])))
    d = reach.nearest_km(tree, np.array([-47.0, -46.0]), np.array([-21.0, -22.0]))[:, 0]
    assert d[0] == pytest.approx(111.2, abs=0.3)
    assert d[1] == pytest.approx(reach.haversine_km(-47.0, -22.0, -46.0, -22.0), rel=1e-6)
    assert np.isinf(reach.nearest_km(tree, np.array([-40.0]), np.array([-22.0]), upper_km=50)[0, 0])


def test_elements_are_written_one_by_one(tmp_path):
    import json

    from bbp.figstyle import Canvas, setup

    setup()
    cv = Canvas(60, 40)
    a = cv.ax(5, 5, 20, 20)
    a.plot([0, 1], [0, 1])
    cv.connect((25, 25), (40, 30))
    cv.panel("left", 5, 5, 25, 25, hide=[])
    cv.panel("right", 35, 5, 55, 35)
    paths = cv.save(tmp_path, "fig", formats=("png",), dpi=100, panels_dir=tmp_path / "panels")
    assert [p.name for p in paths] == ["fig.png"]
    assert sorted(p.name for p in (tmp_path / "panels").iterdir()) == ["layout.json", "left.png", "right.png"]
    lay = json.loads((tmp_path / "panels" / "layout.json").read_text(encoding="utf-8"))
    assert lay["figure_mm"] == [60.0, 40.0] and lay["panels_mm"]["left"]["w"] == pytest.approx(21.2)
