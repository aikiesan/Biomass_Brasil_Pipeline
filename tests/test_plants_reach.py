"""bbp.plants (parsing, sites, checks) and bbp.reach (the 30 m pass and its tables), on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from bbp import plants, reach
from bbp.checks import CheckLog

RES = 0.00026949458523585647           # MapBiomas 30 m pixel (degrees)
X0, Y0 = -48.0, -21.0                  # top-left corner of the synthetic grid


def test_numbers_in_every_source_format():
    assert plants._num("5,500.00") == 5500.0           # ANEEL
    assert plants._num("1.300,5") == 1300.5            # Brazilian
    assert plants._num("8000,00") == 8000.0            # ANP biomethane capacity
    assert plants._num(1300.0) == 1300.0
    assert np.isnan(plants._num("-")) and np.isnan(plants._num(None))
    assert plants._norm("São João da Boa Vista") == "SAO JOAO DA BOA VISTA"


def test_production_reads_points_as_thousands_only_in_groups_of_three():
    assert plants.production_m3("173.037") == (173037.0, True)            # MG, Aug 2026
    assert plants.production_m3("1.234.567") == (1234567.0, True)
    assert plants.production_m3("4324235.558") == (4324235.558, False)    # point = decimal
    assert plants.production_m3("1138977") == (1138977.0, False)
    assert plants.production_m3("159494.82") == (159494.82, False)


def test_anp_ethanol_flex_is_one_record_in_two_families(tmp_path):
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
    assert sorted(cane["name"]) == ["A", "B"] and sorted(corn["name"]) == ["B", "C"]
    assert d.loc[d["name"] == "B", "record_id"].nunique() == 1                  # flex: one record, two rows
    assert cane.set_index("name").loc["A", "capacity"] == 3.0e6
    assert not cane.set_index("name").loc["B", "operating"]
    assert cane.set_index("name").loc["B", "status_reason"] == "ANP status: Construção"


def _capacity_csv(tmp_path):
    p = tmp_path / "cap.csv"
    p.write_text(
        "Mês/Ano,Razão Social,CNPJ,Região,Estado,Município,Capacidade Autorizada de Produção de Biometano (m³/d),"
        "Capacidade Processamento de Biogás(m³/d),Volume Processado de Biogás (m³/d),Volume Processado/Capacidade (%) \n"
        '06/2022,OLD GAS S.A,111,SUDESTE,São Paulo,SAO PAULO,"30000,00",60000,"0,000",0\n'
        '04/2026,A LTDA,222,SUDESTE,São Paulo,NARANDIBA,"20000,00",40000,"20000,000",50\n'
        '08/2026,A LTDA,222,SUDESTE,São Paulo,NARANDIBA,"20000,00",40000,"30000,000",75\n'
        '08/2026,B LTDA,333,SUDESTE,Minas Gerais,TUPACIGUARA,"10000,00",20000,"10000,000",50\n',
        encoding="utf-8")
    q = tmp_path / "prod.csv"
    q.write_text("Mês/Ano,Região,Estado,Produto,Produção (m³)\n"
                 "08/2026,SUDESTE,São Paulo,BIOMETANO,465000.5\n"
                 "08/2026,SUDESTE,Minas Gerais,BIOMETANO,155.000\n"
                 "04/2026,SUDESTE,São Paulo,BIOMETANO,300000\n", encoding="utf-8")
    return p, q


def test_biomethane_snapshot_keeps_stale_plants_as_not_operating(tmp_path):
    cap, prod = _capacity_csv(tmp_path)
    mun = pd.DataFrame({"geocode": [3550308, 3532009, 3169604], "name": ["São Paulo", "Narandiba", "Tupaciguara"],
                        "uf": ["SP", "SP", "MG"], "lon": [-46.6, -51.5, -48.7], "lat": [-23.5, -22.4, -18.6]})
    d = plants.read_anp_biomethane(cap, mun, "08/2026").set_index("name")
    assert len(d) == 3
    assert d.loc["A LTDA", "operating"] and d.loc["B LTDA", "operating"]
    assert not d.loc["OLD GAS S.A", "operating"]
    assert d.loc["OLD GAS S.A", "status_reason"] == "no ANP record since 06/2022"
    assert d.loc["A LTDA", "lon"] == -51.5 and d.loc["B LTDA", "uf"] == "MG"
    s = plants.biomethane_snapshots(cap, prod, ["04/2026", "08/2026"]).set_index("snapshot")
    assert s.loc["04/2026", "plants"] == 1 and s.loc["08/2026", "plants"] == 2
    assert s.loc["08/2026", "authorised_m3d"] == 30000
    assert s.loc["08/2026", "produced_m3_month"] == pytest.approx(465000.5 + 155000)   # MG read as thousands
    assert s.loc["08/2026", "produced_m3d"] == pytest.approx((465000.5 + 155000) / 31)
    assert s.loc["08/2026", "production_rows_thousands"] == 1
    pc = plants.production_check(cap, prod).set_index(["uf"])
    assert pc.loc["MG", "ratio"].max() == pytest.approx(155000 / (10000 * 0.5 * 31))


def _rows(**cols):
    n = len(next(iter(cols.values())))
    base = {"record_id": [f"r{i}" for i in range(n)], "subtype": [""] * n, "located": ["point"] * n,
            "capacity_note": [""] * n}
    return pd.DataFrame(base | cols)


def test_sites_link_nearest_only_and_keep_mills_apart():
    # two mills 3 km apart, a bagasse power plant 1.5 km from each (between them), a flex record in two rows
    km = 1 / 111.2
    d = _rows(family=["sugarcane_mill", "sugarcane_mill", "biomass_power", "sugarcane_mill", "corn_ethanol"],
              subtype=["", "", "bagasse", "", ""], lon=[-47.0, -47.0 + 3 * km, -47.0 + 1.4 * km, -50.0, -50.0],
              lat=[-21.0, -21.0, -21.0, -15.0, -15.0], record_id=["m1", "m2", "u1", "flex", "flex"])
    sites = plants.assign_sites(d, [["sugarcane_mill", "biomass_power:bagasse"]], 2.0)
    assert sites[0] != sites[1]                          # the power plant does not merge the two mills
    assert sites[2] == sites[0]                          # it joins its nearest mill
    assert sites[3] == sites[4]                          # flex: one site
    assert sites.nunique() == 3


def test_duplicates_and_declared_municipality():
    d = _rows(family=["biogas", "biogas", "biogas"], source=["S"] * 3, name=["a", "b", "c"],
              lon=[-47.0, -47.005, -48.0], lat=[-21.0, -21.0, -21.0], plant_id=["p0", "p1", "p2"],
              site_id=["s0", "s1", "s2"], uf=["SP", "SP", "SP"], municipality=["Ribeirão Preto", "Sertãozinho", "X"],
              geocode=[3543402, 3543402, 3500000])
    dup = plants.duplicate_pairs(d, 2.0)
    assert len(dup) == 1 and set(dup.iloc[0][["plant_a", "plant_b"]]) == {"p0", "p1"}
    mun = pd.DataFrame({"geocode": [3543402, 3551702], "name": ["Ribeirão Preto", "Sertãozinho"], "uf": ["SP", "SP"]})
    m = plants.municipality_agreement(d, mun)
    assert m[0] == True and m[1] == False and pd.isna(m[2])    # noqa: E712 (nullable boolean)


def test_register_match_shares():
    import geopandas as gpd

    mills = pd.DataFrame({"lon": [-47.0, -48.0], "lat": [-21.0, -21.0]})
    other = gpd.GeoDataFrame(geometry=gpd.points_from_xy([-47.0 + 1 / 111.2], [-21.0]), crs="EPSG:4674")
    m = plants.register_match(mills, other)
    assert m["within_2km"] == 0.5 and m["within_10km"] == 0.5


# ---------------------------------------------------------------------------
# the 30 m pass
# ---------------------------------------------------------------------------
def _write(path, arr, x0=X0, y0=Y0, dtype="uint8"):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                       crs="EPSG:4326", transform=from_origin(x0, y0, RES, RES), nodata=0) as dst:
        dst.write(arr.astype(dtype), 1)


def _scene(tmp_path):
    """640 x 640 px (~19 km). Two municipalities (left / right half). Cane block in the left, soybean block in the
    right; second-season corn on part of the soybean, in a second-crop raster offset by (8, 16) pixels."""
    n = 640
    lab = np.ones((n, n), dtype=np.uint16)
    lab[:, n // 2:] = 2
    lab[:5, :] = 0                                    # a strip outside the mesh
    cov = np.full((n, n), 15, dtype=np.uint8)         # pasture
    cov[100:200, 100:200] = 20                        # cane
    cov[0:3, 100:200] = 20                            # cane outside the mesh
    cov[300:500, 400:600] = 39                        # soybean
    sc = np.zeros((n + 40, n + 40), dtype=np.uint8)   # second crop, larger, origin 16 rows up and 8 columns left
    sc[16 + 300:16 + 400, 8 + 400:8 + 600] = 1        # corn on the upper half of the soybean
    _write(tmp_path / "lab.tif", lab, dtype="uint16")
    pd.DataFrame({"label": [1, 2], "geocode": [3500001, 3500002]}).to_csv(tmp_path / "lab.lookup.csv", index=False)
    _write(tmp_path / "cov.tif", cov)
    _write(tmp_path / "sc.tif", sc, x0=X0 - 8 * RES, y0=Y0 + 16 * RES)
    return lab, cov


def _px(row, col):
    return X0 + (col + 0.5) * RES, Y0 - (row + 0.5) * RES


def _plants():
    mill = _px(150, 150)                                  # centre of the cane block
    far_lon = X0 + 0.5                                    # ~52 km east of the grid
    return pd.DataFrame({
        "plant_id": ["mill0", "mill1", "corn0", "bio0"], "site_id": ["s0", "s1", "s2", "s3"],
        "family": ["sugarcane_mill", "sugarcane_mill", "corn_ethanol", "biodiesel"],
        "subtype": [""] * 4, "name": ["M0", "M1", "C0", "B0"], "uf": ["SP"] * 4, "municipality": ["x"] * 4,
        "geocode": [3500001, 3500001, 3500002, 3500002], "cd_rgint": [3501] * 4, "nm_rgint": ["R"] * 4,
        "status": ["op"] * 4, "operating": [True, False, True, True],
        "lon": [mill[0], mill[0], far_lon, far_lon], "lat": [mill[1], mill[1] - 0.05, Y0 - 0.12, Y0 - 0.12],
        "capacity": [1.0e6, 2.0e6, 1e5, 100.0], "capacity_unit": ["t cane/season"] * 2 + ["t", "m3"]})


def _spec(**kw):
    crops = {"sugarcane": ((20,), ()), "soybean": ((39,), ()), "corn_2nd": ((), (1,)),
             "all_crops": ((20, 39), (1,))}
    pairs = [("sugarcane", "sugarcane_mill"), ("corn_2nd", "corn_ethanol"), ("soybean", "biodiesel"),
             ("all_crops", "sugarcane_mill")]
    return reach.Spec(crops=crops, pairs=pairs, radii_km=[10, 30, 50, 75], max_km=100, ring_max_km=60,
                      window=160, check_factor=16, **kw)


def _run(tmp_path, spec, **kw):
    return reach.accumulate(tmp_path / "lab.tif", tmp_path / "cov.tif", tmp_path / "sc.tif", _plants(), spec,
                            tmp_path / "ck.npz", cells_tif=tmp_path / "cells.tif", save_every=1, **kw)


def _mun():
    return pd.DataFrame({"geocode": [3500001, 3500002], "cd_rgint": [3501, 3501], "nm_rgint": ["R", "R"],
                         "uf": ["SP", "SP"]})


def test_pass_areas_distances_and_levels(tmp_path):
    from rasterio.transform import from_origin

    from bbp.ingest.second_crop import row_pixel_area_m2

    _scene(tmp_path)
    spec = _spec()
    acc, meta = _run(tmp_path, spec)
    assert meta["complete"]
    px_ha = row_pixel_area_m2(from_origin(X0, Y0, RES, RES), 640) / 1e4
    keys = list(spec.crops)
    cane_in = px_ha[100:200].sum() * 100
    assert acc.crop_area[keys.index("sugarcane")].sum() == pytest.approx(cane_in, rel=1e-9)
    assert acc.unassigned[keys.index("sugarcane")] == pytest.approx(px_ha[0:3].sum() * 100, rel=1e-9)
    corn = px_ha[300:400].sum() * 200                       # the offset second crop lands on the soybean
    assert acc.crop_area[keys.index("corn_2nd")].sum() == pytest.approx(corn, rel=1e-9)
    assert acc.crop_area[keys.index("all_crops")].sum() == pytest.approx(cane_in + px_ha[300:500].sum() * 200,
                                                                        rel=1e-9)   # corn on soy counts once
    t = reach.build_tables(acc, meta, spec, _plants(), _mun())
    u = t["units"]
    nat = u[u["level"] == "nation"].set_index(["pair", "radius_km"])
    # the operating mill sits in the cane block (~3 km wide): all cane within 10 km
    assert nat.loc[("sugarcane|sugarcane_mill", 10.0), "share"] == pytest.approx(1.0)
    # the corn ethanol plant is ~40-55 km from the corn: nothing within 30 km, everything within 75 km
    assert nat.loc[("corn_2nd|corn_ethanol", 30.0), "share"] == 0.0
    assert nat.loc[("corn_2nd|corn_ethanol", 75.0), "share"] == pytest.approx(1.0)
    d = reach.haversine_km(X0 + 0.5, Y0 - 0.12, *_px(350, 500))
    assert nat.loc[("corn_2nd|corn_ethanol", 30.0), "mean_km"] == pytest.approx(d, abs=4.0)
    # levels add up; the municipality of the cane holds all of it
    mun_rows = u[(u["level"] == "municipality") & (u["pair"] == "sugarcane|sugarcane_mill") & (u["radius_km"] == 10)]
    assert mun_rows.set_index("unit").loc["3500001", "crop_area_ha"] == pytest.approx(cane_in)
    for lev in ("municipality", "rgint", "uf"):
        s = u[u["level"] == lev].groupby(["pair", "radius_km"])["within_ha"].sum()
        assert np.allclose(s.sort_index().to_numpy(), nat["within_ha"].sort_index().to_numpy())
    ck = CheckLog("reach")
    reach.check_tables(ck, acc, meta, spec, t, {"sugarcane": cane_in + acc.unassigned[0]})
    assert not ck.failures, [c.name for c in ck.failures]


def test_mill_rings_required_radius_and_contested_cane(tmp_path):
    _scene(tmp_path)
    spec = _spec()
    acc, meta = _run(tmp_path, spec)
    t = reach.build_tables(acc, meta, spec, _plants(), _mun())
    m = t["mills"].set_index("plant_id")
    cane = acc.crop_area[0].sum()
    # mill0 in the block: all cane within 10 km; the block (~3 km) needs a radius of ~2 km for all of it
    assert m.loc["mill0", "cane_ha_10km"] == pytest.approx(cane, rel=1e-9)
    need = 1.0e6 / 75
    assert m.loc["mill0", "need_ha"] == pytest.approx(need)
    assert m.loc["mill0", "supply_ratio_30km"] == pytest.approx(cane / need)
    assert m.loc["mill0", "required_radius_km"] == np.inf if cane < need else m.loc["mill0", "required_radius_km"] <= 3
    # mill1 (5.5 km south, not operating) still gets its rings; only one mill operates: nothing contested
    assert m.loc["mill1", "cane_ha_10km"] == pytest.approx(cane, rel=1e-9)
    assert m.loc["mill0", "contested_share_30km"] == 0.0
    assert m.loc["mill0", "other_mills_30km"] == 0


def test_pass_resumes_to_the_same_result(tmp_path):
    _scene(tmp_path)
    spec = _spec()
    full, _ = reach.accumulate(tmp_path / "lab.tif", tmp_path / "cov.tif", tmp_path / "sc.tif", _plants(), spec,
                               tmp_path / "ckA.npz", save_every=1)
    part, meta = reach.accumulate(tmp_path / "lab.tif", tmp_path / "cov.tif", tmp_path / "sc.tif", _plants(), spec,
                                  tmp_path / "ckB.npz", save_every=1, stop_after=7)
    assert not meta["complete"] and part.windows_done == 7
    rest, meta = reach.accumulate(tmp_path / "lab.tif", tmp_path / "cov.tif", tmp_path / "sc.tif", _plants(), spec,
                                  tmp_path / "ckB.npz", save_every=1)
    assert meta["complete"]
    assert np.allclose(rest.hist, full.hist) and np.allclose(rest.rings, full.rings)
    assert np.allclose(rest.comp, full.comp) and np.allclose(rest.nat, full.nat)


def test_aggregated_cells_and_check(tmp_path):
    import rasterio

    _scene(tmp_path)
    spec = _spec()
    acc, meta = _run(tmp_path, spec)
    with rasterio.open(tmp_path / "cells.tif") as ds:
        assert ds.res[0] == pytest.approx(16 * RES) and ds.count == len(meta["bands"])
        cane = ds.read(1)
        within = ds.read(len(spec.crops) + 1)          # sugarcane|sugarcane_mill within 30 km
    assert cane.sum() == pytest.approx(acc.crop_area[0].sum(), rel=1e-5)
    assert within.sum() == pytest.approx(cane.sum(), rel=1e-5)
    t = reach.build_tables(acc, meta, spec, _plants(), _mun())
    assert t["aggregation_check"]["diff_pp"].abs().max() < 1.0


def test_great_circle_within_three_quarters_of_a_percent_of_wgs84():
    ck = CheckLog("reach")
    reach.geodesic_check(ck, pd.DataFrame({"lon": [-47.0, -60.0, -35.0], "lat": [-21.0, 2.0, -8.0]}))
    assert ck.checks[0].passed and ck.checks[0].value < 0.0075


def test_median_from_bins():
    h = np.array([[0, 2, 2, 0], [4, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
    med = reach._median_from_bins(h, 1.0)
    assert med[0] == pytest.approx(2.0) and med[1] == pytest.approx(0.5) and np.isnan(med[2])   # last: overflow


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
