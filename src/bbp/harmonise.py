"""Harmonise municipal codes across sources and build Minimum Comparable Areas (AMC)."""

from __future__ import annotations

import pandas as pd


def build_amc(geocodes: pd.Series, changes: list[dict]) -> pd.DataFrame:
    """Union-find over (new municipality, parents). amc_id = smallest geocode in the group."""
    parent: dict[int, int] = {int(g): int(g) for g in geocodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for ch in changes:
        for p in ch["parents"]:
            union(int(ch["code"]), int(p))
    out = pd.DataFrame({"geocode": list(parent)})
    out["amc_id"] = [find(g) for g in out["geocode"]]
    out["amc_size"] = out.groupby("amc_id")["geocode"].transform("size")
    return out


def first_record_year(pam: pd.DataFrame) -> pd.Series:
    """First year in which a municipality has any record other than 'not available' in PAM 5457."""
    rec = pam[(pam["table"] == 5457) & (pam["flag"] != "not_available")]
    return rec.groupby("geocode")["year"].min()


def data_gaps(pam: pd.DataFrame) -> pd.DataFrame:
    """(geocode, year) where every PAM 5457 cell is 'not available' ("..."): no statistics that year.

    This includes municipalities not yet installed, but also published gaps
    (e.g. whole-municipality '...' blocks), so it is a report, not a mesh rule.
    """
    t = pam[pam["table"] == 5457]
    na = t.assign(na=t["flag"].eq("not_available")).groupby(["geocode", "year"])["na"].all()
    return na[na].reset_index()[["geocode", "year"]]


def check_changes_against_mesh(changes: list[dict], mesh: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty = all codes and names match the mesh)."""
    names = dict(zip(mesh["geocode"], mesh["name"], strict=True))
    ufs = dict(zip(mesh["geocode"], mesh["uf"], strict=True))
    problems = []
    for ch in changes:
        code = int(ch["code"])
        if code not in names:
            problems.append(f"{code} ({ch['name']}) not in mesh")
            continue
        if names[code] != ch["name"] or ufs[code] != ch["uf"]:
            problems.append(f"{code}: config says {ch['name']}/{ch['uf']}, mesh says {names[code]}/{ufs[code]}")
        for p in ch["parents"]:
            if int(p) not in names:
                problems.append(f"parent {p} of {code} not in mesh")
    return problems


def mesh_mismatch_flags(years: list[int], changes: list[dict], first_year: dict[int, int]) -> pd.DataFrame:
    """(geocode, year) pairs where PAM and MapBiomas use different boundaries.

    Before a new municipality's first PAM year, its area is reported inside its
    parents in PAM but separately in MapBiomas: the new municipality and all its
    parents are flagged for those years.
    """
    rows = []
    for ch in changes:
        fy = first_year.get(int(ch["code"])) or ch.get("first_pam_year") or ch["installed"]
        for y in years:
            if y < int(fy):
                for g in [ch["code"], *ch["parents"]]:
                    rows.append((int(g), y))
    return pd.DataFrame(rows, columns=["geocode", "year"]).drop_duplicates()
