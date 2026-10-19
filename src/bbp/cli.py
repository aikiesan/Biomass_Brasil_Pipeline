"""Command line interface.

    bbp info                          resolved paths and whether each input exists
    bbp run                           every stage, in order
    bbp run area reconcile            selected stages
    bbp run --from reconcile          a stage and everything after it
    bbp second-crop [--years 2023 2024] [--grid-only]
    bbp border-share [--years 2024]   crop area near municipal borders (coverage rasters; long, resumable)
    bbp landcover [--years 2024 2025] [--budget 150]
                                      Fig 1 land-cover display rasters from the 30 m coverage (resumable)
    bbp plants                        processing plants of Brazil in one register (same as bbp run plants)
    bbp reach [--budget 600]          crop area within reach of the plants from the 30 m pixels (long; resumable)
    bbp scale                         the same reach at the pixel, the municipality and the region (MAUP)
    bbp explore [--only x01 x04]      exploratory figures, each element also on its own (data/paper/explore)
    bbp check-factors                 compare conversion_factors.yaml with the CP2b canonical YAML
    bbp paper [--no-figures]          paper numbers.json, Tables 1–4 and Figs 1–6 (profile `paper` by default)
    bbp figures [--only 3 4]          Figs 1–6 only
    bbp --profile paper <command>     merge config/profiles/paper.yaml over pipeline.yaml
    bbp load [--yes] [--dry-run]      load data/processed into PostgreSQL/PostGIS (schema bbp) as a new release
    bbp db-status                     list the releases in the database
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
    for key, p in cfg.input_files():
        print(f"  [{'ok' if p.exists() else 'MISSING':7}] {key}: {p}")
    from bbp.plants import source_paths

    for key, p in source_paths(cfg).items():
        print(f"  [{'ok' if p.exists() else 'MISSING':7}] plants.{key}: {p}")
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
    ap.add_argument("--profile", default=None,
                    help="config/profiles/<name>.yaml merged over pipeline.yaml (paper/figures default: paper)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("info")
    sub.add_parser("check-factors")
    r = sub.add_parser("run")
    r.add_argument("stages", nargs="*", default=[], help=f"any of: {', '.join(STAGES)}")
    r.add_argument("--from", dest="from_stage", choices=STAGES)
    r.add_argument("--force", action="store_true", help="re-read sources even if cached interim files are newer")
    r.add_argument("--keep-going", action="store_true", help="continue after a failed check")
    ld = sub.add_parser("load", help="load processed tables into PostGIS as a new versioned release")
    ld.add_argument("--target", choices=["local", "prod", "pilar2b"], default="local",
                    help="local = BBP_DATABASE_URL, prod = BBP_PROD_DATABASE_URL, pilar2b = BBP_PILAR2B_DATABASE_URL "
                         "(the PILAR-2b development database; environment or .env)")
    ld.add_argument("--url", help="libpq URL; overrides --target")
    ld.add_argument("--yes", action="store_true", help="required for any database that is not local")
    ld.add_argument("--dry-run", action="store_true", help="show what would be loaded and exit")
    ld.add_argument("--label", default="", help="free text stored with the release")
    ld.add_argument("--keep", type=int, default=0, help="keep only the N most recent releases (0 = keep all)")
    ld.add_argument("--full-n1", action="store_true", help="load every N1 column (default: ref/min/max only)")
    ld.add_argument("--full", action="store_true",
                    help="keep empty rows (crops a municipality does not grow) and the AMC unit ratios")
    ld.add_argument("--with-geometry", action="store_true", help="also load bbp.municipality_geom (for QGIS/local use)")
    ld.add_argument("--simplify", type=float, default=0.0, help="simplify geometry tolerance in degrees (e.g. 0.0005)")
    st = sub.add_parser("db-status", help="list releases in the database")
    st.add_argument("--target", choices=["local", "prod", "pilar2b"], default="local")
    st.add_argument("--url")
    s = sub.add_parser("second-crop")
    s.add_argument("--years", type=int, nargs="*")
    s.add_argument("--grid-only", action="store_true")
    bsp = sub.add_parser("border-share", help="crop area near municipal borders, from the coverage rasters (long)")
    bsp.add_argument("--years", type=int, nargs="*", help="default: border_share.years in pipeline.yaml")
    lcp = sub.add_parser("landcover", help="land-cover display rasters of Fig 1 (data/interim/landcover)")
    lcp.add_argument("--years", type=int, nargs="*", help="default: paper.fig1.year and its variants")
    lcp.add_argument("--budget", type=float, default=None,
                     help="stop after this many seconds and resume on the next call (default: run to the end)")
    sub.add_parser("plants", help="processing plants (ANP, CIBiogás, ANEEL, MAPA) -> data/interim/plants.parquet")
    rp = sub.add_parser("reach", help="crop area within reach of the plants, 30 m (long; resumable)")
    rp.add_argument("--budget", type=float, default=None,
                    help="stop after this many seconds and resume on the next call (default: run to the end)")
    sub.add_parser("scale", help="reach at three resolutions: pixel, municipality, intermediate region (needs bbp reach)")
    xp = sub.add_parser("explore", help="exploratory figures on the plants and their surroundings")
    xp.add_argument("--only", nargs="*", help="figure keys, e.g. x01 x04 (default: all)")
    pp = sub.add_parser("paper", help="numbers.json, Tables 1–4 and the figures of the paper")
    pp.add_argument("--no-figures", action="store_true", help="numbers and tables only")
    fg = sub.add_parser("figures", help="Figs 1–6 of the paper")
    fg.add_argument("--only", type=int, nargs="*", help="figure numbers (default: all)")

    args = ap.parse_args(argv)
    profile = args.profile or ("paper" if args.command in ("paper", "figures", "landcover", "plants", "reach",
                                                                 "scale", "explore") else None)
    cfg = load_config(args.config, profile=profile)
    if args.command == "info":
        return cmd_info(cfg)
    if args.command == "check-factors":
        return cmd_check_factors(cfg)
    if args.command == "db-status":
        from bbp.load import resolve_url, status

        print(status(resolve_url(cfg, args.url, args.target), cfg.section("database").get("schema", "bbp")).to_string(index=False))
        return 0

    logfile = _setup_logging(cfg.data_dir, args.verbose)
    log.info("bbp %s — log: %s%s", __version__, logfile, f" — profile: {cfg.profile}" if cfg.profile else "")
    from bbp import pipeline

    try:
        if args.command == "load":
            from bbp.load import describe, is_local, load, resolve_url

            url = resolve_url(cfg, args.url, args.target)
            if not args.yes and not args.dry_run and (args.target == "prod" or not is_local(url)):
                log.error("Refusing to load into %s without --yes (not a local database).", describe(url))
                return 2
            load(cfg, url, label=args.label, keep=args.keep, full_n1=args.full_n1, compact=not args.full,
                 with_geometry=args.with_geometry, simplify=args.simplify, dry_run=args.dry_run)
            return 0
        if args.command == "second-crop":
            _run_stage("second_crop", pipeline.stage_second_crop, cfg, keep_going=True,
                       years=args.years, grid_only=args.grid_only)
            return 0
        if args.command in ("paper", "figures"):
            from bbp import figures, paper

            ok = True
            if args.command == "paper":
                ok = _run_stage("paper", lambda c, ck: paper.build(c, ck), cfg, keep_going=False)
            if args.command == "figures" or not args.no_figures:
                ok = _run_stage("figures", lambda c, ck: figures.build(c, ck, only=getattr(args, "only", None)), cfg,
                                keep_going=True) and ok
            return 0 if ok else 1
        if args.command == "landcover":
            from bbp.figures import build_landcover

            ok = _run_stage("landcover", lambda c, ck: build_landcover(c, ck, args.years, args.budget), cfg,
                            keep_going=True)
            return 0 if ok else 1
        if args.command in ("plants", "reach", "scale", "explore"):
            if args.command == "plants":
                func = pipeline.stage_plants
            elif args.command == "scale":
                from bbp import scale

                func = scale.run
            elif args.command == "reach":
                from bbp import reach

                func = lambda c, ck: reach.run(c, ck, budget=args.budget)  # noqa: E731
            else:
                from bbp import explore

                func = lambda c, ck: explore.build(c, ck, only=args.only)  # noqa: E731
            ok = _run_stage(args.command, func, cfg, keep_going=True)
            return 0 if ok else 1
        if args.command == "border-share":
            ok = _run_stage("border_share", pipeline.stage_border_share, cfg, keep_going=True, years=args.years)
            return 0 if ok else 1
        unknown = [st for st in args.stages if st not in STAGES]
        if unknown:
            ap.error(f"unknown stage(s) {unknown}; choose from {STAGES}")
        stages = args.stages or STAGES
        if args.from_stage:
            stages = STAGES[STAGES.index(args.from_stage):]
        for st in STAGES:
            if st in stages:
                kw = {"force": args.force} if st in ("ingest", "sanitation") else {}
                _run_stage(st, pipeline.STAGE_FUNCS[st], cfg, args.keep_going, **kw)
    except CheckFailed as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
