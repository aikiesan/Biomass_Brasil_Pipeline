"""Agreement metrics between observed (satellite) and declared (statistics) values."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return np.nan
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    denom = vx + vy + (mx - my) ** 2
    return float(2 * cov / denom) if denom > 0 else np.nan


def agreement(obs: np.ndarray, dec: np.ndarray, tolerances=(0.10, 0.25), min_area: float = 0.0) -> dict:
    """Metrics over a set of spatial units (one crop, one year, one scale).

    Units where both sources are below `min_area` are not scored. A unit with
    area in only one source counts as outside every tolerance band.
    """
    obs, dec = np.asarray(obs, float), np.asarray(dec, float)
    valid = ~(np.isnan(obs) | np.isnan(dec))
    obs, dec = obs[valid], dec[valid]
    scored = np.maximum(obs, dec) >= max(min_area, 1e-9)
    o, d = obs[scored], dec[scored]
    both = (o > 0) & (d > 0)
    out = {
        "n_units": int(scored.sum()),
        "n_both": int(both.sum()),
        "n_obs_only": int(((o > 0) & (d == 0)).sum()),
        "n_dec_only": int(((o == 0) & (d > 0)).sum()),
        "sum_obs_ha": float(o.sum()),
        "sum_dec_ha": float(d.sum()),
    }
    out["ratio_total"] = out["sum_obs_ha"] / out["sum_dec_ha"] if out["sum_dec_ha"] > 0 else np.nan
    if both.any():
        lr = np.log(o[both] / d[both])
        out["median_log_ratio"] = float(np.median(lr))
        out["mean_abs_log_ratio"] = float(np.mean(np.abs(lr)))
        out["p10_log_ratio"], out["p90_log_ratio"] = (float(v) for v in np.percentile(lr, [10, 90]))
    else:
        out.update(median_log_ratio=np.nan, mean_abs_log_ratio=np.nan, p10_log_ratio=np.nan, p90_log_ratio=np.nan)
    n = max(out["n_units"], 1)
    rel = np.full(len(o), np.inf)
    rel[both] = np.abs(o[both] / d[both] - 1)
    for tol in tolerances:
        within = rel <= tol
        key = f"{int(round(tol * 100))}"
        out[f"share_within_{key}pct"] = float(within.sum() / n) if out["n_units"] else np.nan
        tot = d.sum()
        out[f"area_share_within_{key}pct"] = float(d[within].sum() / tot) if tot > 0 else np.nan
    out["rel_abs_error"] = float(np.abs(o - d).sum() / d.sum()) if d.sum() > 0 else np.nan
    out["lin_ccc"] = lin_ccc(o, d)
    out["spearman_rho"] = float(stats.spearmanr(o, d).statistic) if out["n_units"] > 2 else np.nan
    return out


def agreement_table(df: pd.DataFrame, by: list[str], obs: str = "a_obs_ha", dec: str = "a_dec_ha", **kw) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key, strict=True)), **agreement(g[obs].to_numpy(), g[dec].to_numpy(), **kw)})
    return pd.DataFrame(rows)


def top_decile_turnover(a: pd.Series, b: pd.Series) -> float:
    """Share of the top-10% units under `a` that are not in the top 10% under `b`."""
    k = max(int(round(len(a) * 0.10)), 1)
    top_a = set(a.nlargest(k).index)
    top_b = set(b.nlargest(k).index)
    return 1 - len(top_a & top_b) / k
