"""Command line interface.

    bbp info                          resolved paths and whether each input exists
    bbp run                           every stage, in order
    bbp run area reconcile            selected stages
    bbp run --from reconcile          a stage and everything after it
    bbp second-crop [--years 2023 2024] [--grid-only]
    bbp check-factors                 compare conversion_factors.yaml with the CP2b canonical YAML
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

from bbp import STAGES, __version__
from bbp.checks import CheckFailed, CheckLog
from bbp.config import load_config

log = logging.getLogger("bbp")


def _setup_logging(data_dir: Path, verbose: bool) -> Path:
    logdir = data_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / f"bbp_{dt.datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    root = logging.getLogger("bbp")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(logfile, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    return logfile


def _run_stage(name: str, func, cfg, keep_going: bool, **kw) -> bool:
    ck = CheckLog(name)
    log.info("=== stage: %s ===", name)
    try:
        func(cfg, ck, **kw)
    finally:
        ck.write(cfg.data_dir / "logs" / f"checks_{name}.json")
    n_fail = len(ck.failures)
    n_warn = sum(1 for c in ck.checks if not c.passed and c.severity == "warn")
    log.info("=== %s done: %d checks, %d failed, %d warnings ===", name, len(ck.checks), n_fail, n_warn)
    if n_fail and not keep_going:
        ck.raise_if_failed()
    return n_fail == 0


def cmd_info(cfg) -> int:
    print(f"bbp {__version__}")
    print(f"repository : {cfg.repo_root}")
    print(f"raw_root   : {cfg.raw_root}  ({'ok' if cfg.raw_root.exists() else 'MISSING'})")
    print(f"data_dir   : {cfg.data_dir}")
    print(f"years      : {cfg.years[0]}–{cfg.years[-1]}")
    for key in cfg.raw["inputs"]:
        paths = cfg.input_path(key)
        for p in paths if isinstance(paths, list) else [paths]:
            print(f"  [{'ok' if p.exists() else 'MISSING':7}] {key}: {p}")
    return 0


def cmd_check_factors(cfg) -> int:
    import yaml

    src = cfg.raw_root / cfg.factors["source"]["canonical_yaml"]
    if not src.exists():
        print(f"Canonical YAML not found: {src}")
        return 1
    canon = yaml.safe_load(src.read_text(encoding="utf-8"))["feedstocks"]
    pairs = {"ts": ("ts", 100), "vs_ts": ("vs_of_ts", 100), "bmp": ("bmp", 1)}
    levels = {"min": "min", "ref": "medio", "max": "max"}
    drift = 0
    for fs, spec in cfg.factors["feedstocks"].items():
        if fs not in canon:
            print(f"  {fs}: not in canonical YAML")
            drift += 1
            continue
        for mine, (theirs, scale) in pairs.items():
            if fs == "PALHA" and mine == "ts":
                continue  # straw g is already on a dry-matter basis (TS fixed at 1)
            for lv, clv in levels.items():
                a, b = spec[mine][lv], canon[fs][theirs][clv] / scale
                if abs(a - b) > 1e-9:
                    print(f"  {fs}.{mine}.{lv}: pipeline {a} vs canonical {b}")
                    drift += 1
    print("conversion factors match the canonical YAML" if drift == 0 else f"{drift} difference(s)")
    return 0 if drift == 0 else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bbp", description="Biomass Brasil Pipeline")
    ap.add_argument("--config", type=Path, default=None, help="config directory (default: <repo>/config)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("info")
    sub.add_parser("check-factors")
    r = sub.add_parser("run")
    r.add_argument("stages", nargs="*", default=[], help=f"any of: {', '.join(STAGES)}")
    r.add_argument("--from", dest="from_stage", choices=STAGES)
    r.add_argument("--force", action="store_true", help="re-read sources even if cached interim files are newer")
    r.add_argument("--keep-going", action="store_true", help="continue after a failed check")
    s = sub.add_parser("second-crop")
    s.add_argument("--years", type=int, nargs="*")
    s.add_argument("--grid-only", action="store_true")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "info":
        return cmd_info(cfg)
    if args.command == "check-factors":
        return cmd_check_factors(cfg)

    logfile = _setup_logging(cfg.data_dir, args.verbose)
    log.info("bbp %s — log: %s", __version__, logfile)
    from bbp import pipeline

    try:
        if args.command == "second-crop":
            _run_stage("second_crop", pipeline.stage_second_crop, cfg, keep_going=True,
                       years=args.years, grid_only=args.grid_only)
            return 0
        unknown = [st for st in args.stages if st not in STAGES]
        if unknown:
            ap.error(f"unknown stage(s) {unknown}; choose from {STAGES}")
        stages = args.stages or STAGES
        if args.from_stage:
            stages = STAGES[STAGES.index(args.from_stage):]
        for st in STAGES:
            if st in stages:
                kw = {"force": args.force} if st == "ingest" else {}
                _run_stage(st, pipeline.STAGE_FUNCS[st], cfg, args.keep_going, **kw)
    except CheckFailed as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
