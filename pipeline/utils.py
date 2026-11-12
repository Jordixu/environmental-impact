"""Shared helpers for the interconsultations emissions analysis pipeline.

Logic that is reused across step modules belongs here: effect sizes and the
route-clustered bootstrap, weighted descriptive statistics, region geography,
facility name cleaning, and generic table I/O. No step module should
re-implement any of this on its own.

The statistical methods are Mann-Whitney U with Cliff's delta as the
non-parametric effect size (using the Romano et al., 2006 interpretation
thresholds), Cohen's d reported for reference only, and a route-clustered
percentile bootstrap for confidence intervals. The clustering matters
because every interconsultation on the same origin-destination route shares
the same distance, so those rows are not independent observations.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

import config

# =============================================================================
# Logging
# =============================================================================


def log(msg: str) -> None:
    """Single logging entry point for the pipeline.

    Every step module should call this instead of a bare `print`, so that if
    output ever needs redirecting or reformatting, there is one place to do it.
    """
    print(msg)


# =============================================================================
# Effect sizes and statistics
# =============================================================================


def cliffs_delta(a: np.ndarray, b: np.ndarray, u_stat: float | None = None) -> float:
    """Cliff's delta, the non-parametric effect size that matches the
    Mann-Whitney U test (equivalent to the rank-biserial correlation).

    delta = P(X > Y) - P(X < Y) = 2*U / (n1*n2) - 1, in [-1, 1], where U is
    the Mann-Whitney U statistic for `a` relative to `b` (group 1 vs group
    2). The sign carries meaning: a negative delta means `a` tends to be
    smaller than `b`. Pass an already-computed `u_stat` if you have one, to
    avoid redoing the Mann-Whitney U test on large samples.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n1, n2 = len(a), len(b)
    if u_stat is None:
        u_stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 2 * u_stat / (n1 * n2) - 1


def interpret_cliffs_delta(delta: float) -> str:
    """Interpret Cliff's delta using the Romano et al. (2006) thresholds for
    |delta|: negligible below 0.147, small below 0.33, medium below 0.474,
    large above that.
    """
    abs_d = abs(delta)
    if abs_d < 0.147:
        return "negligible"
    elif abs_d < 0.33:
        return "small"
    elif abs_d < 0.474:
        return "medium"
    else:
        return "large"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size: (mean(a) - mean(b)) / pooled_sd.

    Reported as a reference value only. It assumes normality, and that does
    not hold for the distance and emissions variables analyzed here (see the
    Shapiro-Wilk test in s07). So this is not the headline effect size for a
    Mann-Whitney U comparison; use cliffs_delta() for that instead.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n1, n2 = len(a), len(b)
    mean1, mean2 = np.mean(a), np.mean(b)
    sd1, sd2 = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_sd


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d using the conventional thresholds for |d|:
    negligible below 0.2, small below 0.5, medium below 0.8, large above that.
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict | None:
    """Weighted descriptive statistics, computed without ever materializing
    an exploded DataFrame: a 1-D array is expanded with np.repeat (the
    weights are integer counts of how many interconsultations share each
    value), and order statistics are read off that array directly.

    Returns a dict with keys sum, count, mean, median, min, max, p25, p75,
    p90 and std, or None if there are no valid (non-NaN) observations.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values)
    v, w = values[mask], weights[mask].astype(int)
    total_n = int(w.sum())
    if total_n == 0:
        return None
    weighted_sum = float((v * w).sum())
    weighted_mean = weighted_sum / total_n
    expanded = np.repeat(v, w)
    return {
        "sum": weighted_sum,
        "count": total_n,
        "mean": weighted_mean,
        "median": float(np.median(expanded)),
        "min": float(v.min()),
        "max": float(v.max()),
        "p25": float(np.percentile(expanded, 25)),
        "p75": float(np.percentile(expanded, 75)),
        "p90": float(np.percentile(expanded, 90)),
        "std": float(np.std(expanded, ddof=1)) if total_n > 1 else 0.0,
    }


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Quantile `q` (in [0, 1]) of the weighted distribution: each value v_i
    counted w_i times. Uses the same np.repeat + np.percentile convention as
    weighted_stats' p25/p75/p90, generalized to an arbitrary quantile.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float).astype(int)
    expanded = np.repeat(values, weights)
    return float(np.percentile(expanded, 100 * q))


def weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    """Median of the data expanded by integer weights (value v_i repeated
    w_i times), without ever building that expanded array.

    This picks the value at the weighted midpoint directly, via cumulative
    sums and a single searchsorted, which is a different convention from
    weighted_quantile(..., q=0.5)'s np.percentile interpolation. The two are
    not unified: this is the fast version that cluster_bootstrap calls
    thousands of times per analysis, and it has to stay bit-identical to the
    one that produced the published bootstrap confidence intervals.
    """
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    order = np.argsort(v)
    cw = np.cumsum(w[order])
    return float(v[order][np.searchsorted(cw, cw[-1] / 2.0)])


def _weighted_cliffs_delta(a: np.ndarray, wa: np.ndarray, b: np.ndarray, wb: np.ndarray) -> float:
    """Cliff's delta for weighted observations (each value a_i counts wa_i
    times). Gives the same result as cliffs_delta() on the expanded data,
    but without ever expanding it. Internal helper for cluster_bootstrap.
    """
    order = np.argsort(b)
    b_sorted = b[order]
    cum_w = np.concatenate([[0.0], np.cumsum(wb[order])])
    total_b = cum_w[-1]
    below = cum_w[np.searchsorted(b_sorted, a, side="left")]
    below_or_equal = cum_w[np.searchsorted(b_sorted, a, side="right")]
    above = total_b - below_or_equal
    return ((wa * below).sum() - (wa * above).sum()) / (wa.sum() * total_b)


def cluster_bootstrap(
    cap_df: pd.DataFrame,
    non_df: pd.DataFrame,
    var_name: str,
    scale_factor: float,
    n_resamples: int,
    confidence_level: float,
    random_state: int,
) -> dict:
    """Route-clustered percentile bootstrap for the mean difference, the
    median difference and Cliff's delta (group 1 "cap_df" minus group 2
    "non_df").

    `cap_df` / `non_df` hold one row per route-year cell (the value is
    constant within the cell), with columns config.ROUTE_KEY, `var_name` and
    config.WEIGHT_COL. Whole routes (grouped by config.ROUTE_KEY) get
    resampled with replacement within each group: every interconsultation on
    the same route shares the same distance, so they are not independent
    observations, and every cell of a sampled route is kept together. The
    result is deterministic for a given random_state.

    Returns a dict with keys mean_diff_ci, median_diff_ci and
    cliffs_delta_ci, each a (low, high) tuple at the given confidence level.
    """
    rng = np.random.default_rng(random_state)

    def prep(df_: pd.DataFrame):
        v = df_[var_name].to_numpy(dtype=float) * scale_factor
        w = df_[config.WEIGHT_COL].to_numpy(dtype=float)
        groups = list(df_.groupby(config.ROUTE_KEY).indices.values())
        return v, w, groups

    va, wa, ga = prep(cap_df)
    vb, wb, gb = prep(non_df)
    out = np.empty((n_resamples, 3))
    for i in range(n_resamples):
        if (i + 1) % 500 == 0:
            log(f"    cluster_bootstrap progress: {i + 1:,}/{n_resamples:,} resamples")
        ia = np.concatenate([ga[j] for j in rng.integers(0, len(ga), len(ga))])
        ib = np.concatenate([gb[j] for j in rng.integers(0, len(gb), len(gb))])
        a, w1 = va[ia], wa[ia]
        b, w2 = vb[ib], wb[ib]
        out[i, 0] = np.average(a, weights=w1) - np.average(b, weights=w2)
        out[i, 1] = weighted_median(a, w1) - weighted_median(b, w2)
        out[i, 2] = _weighted_cliffs_delta(a, w1, b, w2)

    alpha = 1 - confidence_level
    lo = np.percentile(out, 100 * alpha / 2, axis=0)
    hi = np.percentile(out, 100 * (1 - alpha / 2), axis=0)
    return {
        "mean_diff_ci": (float(lo[0]), float(hi[0])),
        "median_diff_ci": (float(lo[1]), float(hi[1])),
        "cliffs_delta_ci": (float(lo[2]), float(hi[2])),
    }


# =============================================================================
# Geography
# =============================================================================

_EARTH_RADIUS_M = 6_378_137.0  # WGS84 equatorial radius


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in metres."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a)))


def short_region_name(name):
    """Strip the 'Regió Sanitària ' / 'Regió sanitària ' prefix from a
    health region name. Non-string values pass through unchanged.
    """
    if not isinstance(name, str):
        return name
    return name.replace("Regió Sanitària ", "").replace("Regió sanitària ", "")


def classify_region_type(region_name) -> str:
    """Classify a health region as "Capital" (the three Barcelona health
    regions) or "Non-capital" (everything else).
    """
    name = short_region_name(region_name)
    return "Capital" if name in config.CAPITAL_REGIONS else "Non-capital"


def to_8_regions(region_name):
    """Collapse the three Barcelona health regions into a single
    "Barcelona" category; the other seven health regions pass through
    unchanged, as do missing values.
    """
    if pd.isna(region_name):
        return region_name
    return "Barcelona" if "Barcelona" in str(region_name) else region_name


def _clean_region_name(name):
    """Fix a known encoding artefact in the regions GeoJSON: 'Penedès'
    sometimes shows up with a mangled accent. Any name containing the
    unaccented stem gets normalized back to the correctly accented form.
    """
    if pd.isna(name):
        return name
    if "Pened" in name:
        return "Penedès"
    return name


def assign_region(df: pd.DataFrame, lat_col: str, lon_col: str, out_col: str) -> pd.DataFrame:
    """Spatially assign each row of `df` to a health region with a point-
    in-polygon join against the regions GeoJSON named in
    config.REGIONS_GEOJSON (under config.DATA_DIR).

    Adds `out_col` holding the region name (config.REGION_NAME_FIELD from
    the GeoJSON) and returns a plain DataFrame: the geometry and sjoin
    bookkeeping columns are dropped before it comes back.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    regions = gpd.read_file(config.data_path(config.REGIONS_GEOJSON))
    if regions.crs is None:
        regions = regions.set_crs(epsg=4326)

    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")

    joined = gpd.sjoin(
        gdf,
        regions[[config.REGION_NAME_FIELD, "geometry"]],
        how="left",
        predicate="within",
    )
    joined[out_col] = joined[config.REGION_NAME_FIELD].apply(_clean_region_name)

    drop_cols = ["geometry", "index_right", config.REGION_NAME_FIELD]
    drop_cols = [c for c in drop_cols if c in joined.columns]
    return pd.DataFrame(joined.drop(columns=drop_cols))


# =============================================================================
# Facility name cleaning
# =============================================================================
# These lookup tables are the single source of truth for name cleaning,
# reconciled from the historical preprocessing notebook and the 2025
# integration script (both used identical tables). Facility names appear
# here because the whole point of these functions is to standardize facility
# name strings for geocoding: this is a naming-normalization lookup, not a
# count or record tied to any individual.
#
# Stage 1: specific, individually handled name corrections.
INDIVIDUAL_CHANGES = {
    "Unitat Salut Laboral Vallès Oriental i Vallès Occidental": "Hospital Mollet",
    "Institut Diagnòstic Imatge Tarragona (Ressonància Magnètica Nuclear)": "Hospital Universitari Joan XXIII",
    "PAC Lleida Rural Sud": "CAP Albatarrec",
    "Unitat de Rehabilitació Vilafranca": "CAP Vilafranca Nord",
    "Equip pediàtric territorial Moianès": "CAP Moià",
    "Hospital del Mar - H. de l'Esperança": "Hospital de l'Esperança",
    "Trèvol, residència de paràlisi cerebral": "La Muntanyeta Tarragona",
}

# Stage 2: regex-based removals (service/unit prefixes that do not identify
# the facility itself).
REGEX_REMOVALS = {
    "service_keywords": r"^(Servei de|Unitat de|Equip de|Consultes Externes)\s+",
    # Trailing parenthetical qualifier: a district, a former name, or a
    # service description tacked onto the facility name. Stripping it is
    # what lets a facility resolve to a single geocodable site. It affects
    # 12 of the 672 raw names and brings them down to the 659 distinct
    # locations the analysis uses; two entries of one hospital group in
    # different towns collapse into a single location, which is why the
    # count drops by one more than the number of names changed.
    #
    # This runs after the stage 1 literal replacements, so a name whose
    # parenthetical is part of an explicit individual correction gets
    # handled there first and is left alone here.
    "trailing_parenthetical": r"\s*\([^)]*\)\s*$",
}

# Stage 3: general abbreviation/term replacements, applied with word
# boundaries so they do not match inside longer words.
GENERAL_REPLACEMENTS = {
    "EAP": "CAP",
    "CAC": "CAP",
    "CSM": "Centre de Salut Mental",
    "ASSIR": "CAP",
    "Unitat Rehabilitació": "CAP",
    "PAC": "CAP",
    "Diagnòstic Imatge": "CAP",
    "CUAP": "Centre d'Urgències d'Atenció Primària",
}

# Stage 4: plain substrings to drop.
SUBSTRINGS_TO_REMOVE = [
    "U.Suport Espec.",
    "Unitat Suport Especialistes",
    "Unitat Suport Especialitats",
    "- Internament",
]

# A small number of facility names denote two co-managed physical sites
# sharing a single administrative code. Both the historical stream (s01)
# and the 2025 refresh stream (s03) split these into two separate location
# columns. The names live here, not in either step module, for the same
# reason as INDIVIDUAL_CHANGES above: this module is the only place facility
# names are allowed to appear.
MULTI_LOCATION_PATTERNS = [
    "Hospital General de l'Hospitalet de Llobregat i Hospital de Sant Joan Despí Moisès Broggi",
    "HC Sant Jaume Calella i HC de Blanes",
]


def split_multi_location(series: pd.Series) -> pd.Series:
    """Split a cleaned facility-name series into a list of locations each.

    For the handful of names that exactly match (after stripping) an entry
    in MULTI_LOCATION_PATTERNS, this splits on ' i ' into the component
    sites; any other name comes back unchanged as a single-element list.
    Matching happens on the already-cleaned name (see clean_facility_names).
    """

    def _split_one(name):
        name = "" if pd.isna(name) else str(name)
        stripped = name.strip()
        for pattern in MULTI_LOCATION_PATTERNS:
            if stripped == pattern.strip() and " i " in name:
                return [part.strip() for part in name.split(" i ")]
        return [name]

    return series.apply(_split_one)


def clean_facility_names(series: pd.Series, case_sensitive: bool) -> pd.Series:
    """Standardize facility names for geocoding, in stages: strip, then
    INDIVIDUAL_CHANGES (literal replacements), REGEX_REMOVALS,
    GENERAL_REPLACEMENTS (word-boundary), SUBSTRINGS_TO_REMOVE, and finally
    collapse whitespace.

    `case_sensitive` controls whether the replacements match case
    sensitively. See is_nonphysical() below for why the two data streams in
    this pipeline deliberately pass different values here.
    """
    s = series.astype(str).str.strip()
    for original, new in INDIVIDUAL_CHANGES.items():
        s = s.str.replace(original, new, regex=False, case=case_sensitive)
    for pattern in REGEX_REMOVALS.values():
        s = s.str.replace(pattern, "", regex=True, case=case_sensitive)
    for old, new in GENERAL_REPLACEMENTS.items():
        pattern = r"\b" + re.escape(old) + r"\b"
        s = s.str.replace(pattern, new, regex=True, case=case_sensitive)
    for sub in SUBSTRINGS_TO_REMOVE:
        s = s.str.replace(sub, "", regex=False, case=case_sensitive)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def is_nonphysical(series: pd.Series, case_sensitive: bool) -> pd.Series:
    """True where any of config.NONPHYSICAL_KEYWORDS matches the series.

    The historical data stream (the original export, cleaned by s01) calls
    this on the already-cleaned facility names with case_sensitive=False,
    while the 2025 refresh stream (integrated by s03) calls it on the raw
    facility names with case_sensitive=True. The two call sites look
    inconsistent, but they are not a mismatch to fix: this is exactly how
    the two streams were filtered when the published Figure 1 counts were
    produced, so changing either value would silently shift those counts.
    Both call sites need to keep their own case_sensitive setting.
    """
    pattern = "|".join(re.escape(kw) for kw in config.NONPHYSICAL_KEYWORDS)
    return series.astype(str).str.contains(pattern, case=case_sensitive, na=False)


# =============================================================================
# Generic table I/O
# =============================================================================


def resolve_input(name: str) -> Path:
    """Locate a pipeline artifact: prefer a freshly generated one in
    OUTPUT_DIR, otherwise fall back to the frozen copy in DATA_DIR.

    This fallback is what lets any single step run on its own against the
    frozen intermediates, and what lets the whole pipeline run without any
    API calls. A step that consumes a previous step's artifact should
    resolve it with this helper instead of hard-coding DATA_DIR or
    OUTPUT_DIR.
    """
    p = config.output_path(name)
    if p.exists():
        return p
    # Route caches live in a cache/ subdirectory so they don't clutter the
    # results folder; check there too before falling back to the frozen copy.
    cached = config.OUTPUT_DIR / "cache" / name
    if cached.exists():
        return cached
    return config.data_path(name)


def year_column(df: pd.DataFrame) -> str:
    """Return the name of the year column present in `df`.

    The raw exports label the year `config.YEAR_COL`, while the analysis
    dataset written by the emissions step exposes the same values under the
    tidier name "year". A step may be handed either one (the raw
    intermediates when run against the frozen files, or the analysis
    dataset when run after the emissions step), so the column name is
    resolved instead of assumed. The two always hold identical values.
    """
    if "year" in df.columns:
        return "year"
    if config.YEAR_COL in df.columns:
        return config.YEAR_COL
    raise KeyError(
        f"No year column found: expected 'year' or {config.YEAR_COL!r} "
        f"in columns {list(df.columns)[:12]}..."
    )


def read_table(path) -> pd.DataFrame:
    """Read a DataFrame from `path`, dispatching on its suffix: .parquet,
    .xlsx, .csv, or .txt (the last one read as CSV, matching the 2025
    export format).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".xlsx":
        return pd.read_excel(path)
    elif suffix in (".csv", ".txt"):
        return pd.read_csv(path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file extension for read_table: {suffix}")


def write_table(df: pd.DataFrame, path) -> None:
    """Write `df` to `path`, creating parent directories as needed and
    dispatching on the suffix: .parquet, .xlsx, or .csv.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".xlsx":
        df.to_excel(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file extension for write_table: {suffix}")


# =============================================================================
# Manuscript number collection
# =============================================================================
# An in-process collector for every number the manuscript reports. Step
# modules call record_number() as they compute each figure, and
# s10_tables.py flushes the collector to manuscript_numbers.csv through
# write_numbers().

_NUMBERS: list[dict] = []


_NUMBER_COLUMNS = ["name", "value", "unit", "source_step"]


def record_number(name: str, value, unit: str, source_step: str) -> None:
    """Append one manuscript number to the in-process collector."""
    _NUMBERS.append({"name": name, "value": value, "unit": unit, "source_step": source_step})


def flush_numbers(step_name: str) -> Path:
    """Write this step's recorded numbers to OUTPUT_DIR/numbers/<step>.csv.

    The collector only lives for the duration of one process, so a step run
    on its own would otherwise lose everything it recorded. Each step
    persists its own fragment, and the tables step later merges all the
    fragments into one manuscript_numbers.csv. That keeps the numbers
    complete whether the pipeline runs end to end or one step at a time.

    Returns the path written, and clears the collector afterward, so a
    process that runs several steps in sequence does not attribute one
    step's numbers to the next.
    """
    path = config.output_path("numbers", f"{step_name}.csv")
    pd.DataFrame(_NUMBERS, columns=_NUMBER_COLUMNS).to_csv(path, index=False)
    log(f"Recorded {len(_NUMBERS)} numbers to {path.name}")
    _NUMBERS.clear()
    return path


def collect_numbers() -> pd.DataFrame:
    """Merge every per-step fragment in OUTPUT_DIR/numbers/ with whatever is
    still sitting in the in-process collector.

    Later entries win on a duplicated name, so if a step recomputes a value
    another step already recorded, that does not leave two conflicting rows.
    """
    frames = []
    numbers_dir = config.OUTPUT_DIR / "numbers"
    if numbers_dir.is_dir():
        for frag in sorted(numbers_dir.glob("*.csv")):
            frames.append(pd.read_csv(frag))
    if _NUMBERS:
        frames.append(pd.DataFrame(_NUMBERS, columns=_NUMBER_COLUMNS))
    if not frames:
        return pd.DataFrame(columns=_NUMBER_COLUMNS)
    out = pd.concat(frames, ignore_index=True)[_NUMBER_COLUMNS]
    return out.drop_duplicates(subset="name", keep="last").sort_values("name").reset_index(drop=True)


def write_numbers(path) -> None:
    """Write the merged manuscript numbers to a CSV at `path`, with exactly
    these columns: name, value, unit, source_step.
    """
    out = collect_numbers()
    write_table(out, path)
    log(f"Wrote {len(out)} manuscript numbers to {Path(path).name}")
