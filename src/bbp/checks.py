"""Invariant checks.

Every stage records its checks through a `CheckLog`. A failed check with
severity "error" stops the run (unless --keep-going); "warn" is reported
and written to data/logs/checks_<stage>.json so it shows up in the release.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("bbp")


class CheckFailed(RuntimeError):
    pass


@dataclass
class Check:
    name: str
    passed: bool
    severity: str  # "error" | "warn"
    detail: str = ""
    value: Any = None


@dataclass
class CheckLog:
    stage: str
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "", value: Any = None, severity: str = "error") -> bool:
        passed = bool(passed)
        self.checks.append(Check(name, passed, severity, detail, _jsonable(value)))
        level = logging.INFO if passed else (logging.ERROR if severity == "error" else logging.WARNING)
        status = "PASS" if passed else ("FAIL" if severity == "error" else "WARN")
        log.log(level, "[%s] %s %s%s", self.stage, status, name, f" — {detail}" if detail else "")
        return passed

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"stage": self.stage, "checks": [asdict(c) for c in self.checks]}, fh, indent=2, ensure_ascii=False)

    def raise_if_failed(self) -> None:
        if self.failures:
            names = ", ".join(c.name for c in self.failures)
            raise CheckFailed(f"{len(self.failures)} check(s) failed in stage '{self.stage}': {names}")


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)
