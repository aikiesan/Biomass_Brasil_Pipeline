"""The shipped configuration must be internally consistent."""

from bbp.config import load_config
from bbp.inventory import FACTORS


def test_config_loads_and_is_consistent():
    cfg = load_config()
    assert cfg.years[0] <= cfg.years[-1]
    feed = cfg.factors["feedstocks"]
    for crop, spec in cfg.crops.items():
        assert spec["pam_table"] in (5457, 839), crop
        if spec.get("feedstock"):
            assert spec["feedstock"] in feed, crop
    for fs, spec in feed.items():
        for f in FACTORS:
            assert 0 < spec[f]["min"] <= spec[f]["ref"] <= spec[f]["max"], (fs, f)
    for ch in cfg.changes["changes"]:
        assert len(str(ch["code"])) == 7 and ch["parents"]
