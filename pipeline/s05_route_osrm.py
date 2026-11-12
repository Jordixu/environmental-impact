"""Step 05: OSRM routing and ORS/OSRM agreement analysis.

Re-computes the same round-trip routes as `s04_route_ors`, this time using
OSRM (`config.OSRM_BASE_URL`) as an independent routing engine, and then
asks how much the choice of routing engine actually matters for this
study's results.

`config.OSRM_BASE_URL` points at the public OSRM demo instance, not a
self-hosted server, so requests are rate-limited (`config.OSRM_SLEEP_S`
between calls) under that service's fair-use policy, the same as any other
public API used elsewhere in this pipeline. The legacy notebook's text
called the server self-hosted with no rate limit, but the URL it actually
called was the public demo server all along; this module keeps the rate
limit and describes the server honestly.

The agreement analysis has two parts. The population-weighted (N-weighted)
total distance and time implied by each service is the number that
actually matters for the paper's conclusions: every reported headline
total is a sum of route distance or time multiplied by the number of
affected interconsultations (N), so that weighted total is what bounds how
much the choice of engine could move those figures. Alongside it, a
route-level comparison runs on routes deduplicated by `config.ROUTE_KEY`
(one row per unique origin-destination pair), unweighted. Route distance
and time depend only on the origin-destination pair, not on the year, sex
or age group of the affected interconsultations, so the row-level data
(one row per origin x destination x year x sex x age-group cell) repeats
the same route-level comparison on average about 46 times. Testing
significance at that row-level grain would overstate the sample size
through pseudo-replication; deduplicating to the unique-route grain is
what fixes it.

The same two cache-correctness fixes from `s04_route_ors` apply here too: a
cached failure (km=None) is treated as a cache miss, not a hit, and stays
eligible for a retry, and the retry loop backs off exponentially, which the
legacy notebook never did.

Input:
  geocoded_interconsultations.parquet (resolved via
  utils.resolve_input)
  analysis_dataset.parquet (resolved via utils.resolve_input; the s06
  output population used for the agreement analysis)
  route_cache_osrm.pkl (resolved via utils.resolve_input; the frozen cache
  in DATA_DIR is never modified)

Output:
  OUTPUT_DIR/routed_osrm.parquet
  OUTPUT_DIR/route_table_osrm.parquet
  OUTPUT_DIR/cache/route_cache_osrm.pkl
  OUTPUT_DIR/ors_osrm_agreement.xlsx
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd
import requests
from scipy import stats

import config
from pipeline import utils

INPUT_FILENAME = "geocoded_interconsultations.parquet"
CACHE_FILENAME = "route_cache_osrm.pkl"
DISTANCES_OUTPUT = "routed_osrm.parquet"
ROUTES_OUTPUT = "route_table_osrm.parquet"
AGREEMENT_OUTPUT = "ors_osrm_agreement.xlsx"

REQUIRED_COORD_COLUMNS = [
    "origin_latitude", "origin_longitude", "destination_latitude", "destination_longitude",
]

# One row per origin/destination/year/UP_DES/UP_DES_1/SEXE/GRUP_EDAT cell;
# verified strictly one-to-one against analysis_dataset.parquet with zero
# misses once GRUP_EDAT is normalized to the same dtype on both sides.
AGREEMENT_MERGE_KEY = [
    "OC_UP_ORI", "DER_UP_SCS_DES", "UP_DES", "UP_DES_1", "SEXE", "GRUP_EDAT", "year",
]

MAX_RETRIES = 3

_EMPTY_ENTRY = {"km": None, "duration_min": None, "geometry": None}

# Previously published reference values for the route-level distance
# comparison, reported alongside the figures this run produces so a
# disagreement is visible rather than silent.
_PUBLISHED_REFERENCE = {
    "Pearson r": 0.999,
    "Spearman rho": 0.998,
    "Bland-Altman bias (km)": 1.17,
    "Parametric LoA low (km)": -8.23,
    "Parametric LoA high (km)": 10.57,
    "Rank-biserial r": 0.150,
}
_REFERENCE_TOLERANCE = 0.05  # relative tolerance for the disagreement flag


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 05 - route unique origin/destination pairs with OSRM and report "
        "its agreement with the ORS routing from s04."
    )
    parser.add_argument(
        "--use-cache",
        dest="use_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only the cached routes; never call the OSRM API (default: True).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        default=False,
        help="Permit live OSRM calls for pairs missing from the cache or cached as a past failure.",
    )
    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
# Cache helpers
# -----------------------------------------------------------------------------


def _is_valid_cache_entry(entry) -> bool:
    """True if `entry` is a cached routing result with an actual distance.

    A cached entry with `km=None` (a past failure) does not count as a hit,
    so it stays eligible for a retry. Same fix as s04_route_ors.
    """
    return isinstance(entry, dict) and entry.get("km") is not None


def _load_cache(path) -> dict:
    if path.exists():
        with open(path, "rb") as f:
            cache = pickle.load(f)
        utils.log(f"Loaded OSRM cache: {len(cache):,} entries from {path}")
        return cache
    utils.log(f"No existing OSRM cache found at {path}; starting empty.")
    return {}


# -----------------------------------------------------------------------------
# Pair construction (shared shape with s04_route_ors; kept separate because
# the two modules route against distinct APIs with distinct cache schemas)
# -----------------------------------------------------------------------------


def _has_alt_destination_columns(df: pd.DataFrame) -> bool:
    return "destination_latitude_2" in df.columns and "destination_longitude_2" in df.columns


def _build_pairs(df: pd.DataFrame) -> list[tuple]:
    has_alt = _has_alt_destination_columns(df)
    pairs = set()
    for row in df.itertuples(index=False):
        o = (row.origin_longitude, row.origin_latitude)
        d1 = (row.destination_longitude, row.destination_latitude)
        pairs.add((o, d1))
        pairs.add((d1, o))
        if has_alt:
            lat2, lon2 = row.destination_latitude_2, row.destination_longitude_2
            if pd.notna(lat2) and pd.notna(lon2):
                d2 = (lon2, lat2)
                pairs.add((o, d2))
                pairs.add((d2, o))
    return list(pairs)


# -----------------------------------------------------------------------------
# Routing
# -----------------------------------------------------------------------------


def _route_with_retry(pair: tuple) -> dict:
    """Route one coordinate pair against config.OSRM_BASE_URL, retrying
    with exponential backoff on failure. Sleeps config.OSRM_SLEEP_S after a
    successful call to respect the public server's fair-use policy;
    sleeps a longer backoff before a retry.
    """
    origin, dest = pair
    coords = f"{origin[0]},{origin[1]};{dest[0]},{dest[1]}"
    url = f"{config.OSRM_BASE_URL}/route/v1/driving/{coords}"
    params = {"overview": "full", "geometries": "geojson"}

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            data = response.json()
            if data.get("code") != "Ok" or not data.get("routes"):
                raise RuntimeError(data.get("message", "no route found"))
            route = data["routes"][0]
            entry = {
                "km": route["distance"] / 1000.0,
                "duration_min": route["duration"] / 60.0,
                "geometry": json.dumps(route["geometry"]),
            }
            time.sleep(config.OSRM_SLEEP_S)
            return entry
        except Exception as exc:  # noqa: BLE001 - any OSRM/network failure is retried the same way
            last_err = exc
            utils.log(f"  OSRM routing attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            time.sleep(config.OSRM_SLEEP_S * (2 ** attempt))
    utils.log(f"  Giving up on this pair after {MAX_RETRIES} attempts: {last_err}")
    return {"km": None, "duration_min": None, "geometry": None, "error": str(last_err)}


def _resolve_pairs(pairs: list[tuple], cache: dict, allow_network: bool) -> tuple[dict, dict]:
    """Resolve every pair against `cache`, updating it in place. See
    s04_route_ors._resolve_pairs for the identical cache-hit semantics.
    """
    stats = {
        "total_pairs": len(pairs), "hits": 0, "retried_failures": 0,
        "new_pairs": 0, "missing": 0, "routed_live": 0,
    }
    pending = []
    for pair in pairs:
        entry = cache.get(pair)
        if _is_valid_cache_entry(entry):
            stats["hits"] += 1
        else:
            pending.append(pair)
            if entry is not None:
                stats["retried_failures"] += 1
            else:
                stats["new_pairs"] += 1

    utils.log(
        f"Cache hits: {stats['hits']:,}/{stats['total_pairs']:,} pairs "
        f"({stats['retried_failures']:,} cached failures, {stats['new_pairs']:,} never cached)"
    )

    if not allow_network:
        for pair in pending:
            if pair not in cache:
                cache[pair] = dict(_EMPTY_ENTRY)
        stats["missing"] = len(pending)
        if pending:
            utils.log(
                f"--use-cache: {len(pending):,} pairs have no usable cached route and will NOT "
                f"be called live; pass --refresh to route them. Recorded as unroutable."
            )
    else:
        utils.log(f"--refresh: routing {len(pending):,} pending pairs live via OSRM...")
        for i, pair in enumerate(pending, 1):
            cache[pair] = _route_with_retry(pair)
            stats["routed_live"] += 1
            if i % 100 == 0 or i == len(pending):
                utils.log(f"  Routed {i:,}/{len(pending):,} pending pairs...")

    return cache, stats


# -----------------------------------------------------------------------------
# Applying routes to the dataset
# -----------------------------------------------------------------------------


def _apply_routes(df: pd.DataFrame, pair_results: dict) -> pd.DataFrame:
    """Assign outbound/return/round-trip distance and duration, selecting
    the shorter round trip when an alternative destination exists. Mirrors
    s04_route_ors._apply_routes (ported from the legacy OSRM notebook,
    which duplicated the same logic rather than sharing it with the ORS
    notebook).
    """
    has_alt = _has_alt_destination_columns(df)

    km_outbound, km_return, km_roundtrip = [], [], []
    dur_outbound, dur_return, dur_roundtrip = [], [], []

    for row in df.itertuples(index=False):
        o = (row.origin_longitude, row.origin_latitude)
        d1 = (row.destination_longitude, row.destination_latitude)

        r1_out = pair_results.get((o, d1))
        r1_ret = pair_results.get((d1, o))
        valid1 = bool(r1_out) and bool(r1_ret) and r1_out.get("km") is not None and r1_ret.get("km") is not None
        dist1 = (r1_out["km"] + r1_ret["km"]) if valid1 else None

        r2_out = r2_ret = None
        valid2 = False
        dist2 = None
        if has_alt:
            lat2, lon2 = row.destination_latitude_2, row.destination_longitude_2
            if pd.notna(lat2) and pd.notna(lon2):
                d2 = (lon2, lat2)
                r2_out = pair_results.get((o, d2))
                r2_ret = pair_results.get((d2, o))
                valid2 = bool(r2_out) and bool(r2_ret) and r2_out.get("km") is not None and r2_ret.get("km") is not None
                if valid2:
                    dist2 = r2_out["km"] + r2_ret["km"]

        if valid1 and valid2:
            use_1 = dist1 <= dist2
        elif valid1:
            use_1 = True
        elif valid2:
            use_1 = False
        else:
            km_outbound.append(None)
            km_return.append(None)
            km_roundtrip.append(None)
            dur_outbound.append(None)
            dur_return.append(None)
            dur_roundtrip.append(None)
            continue

        sel_out, sel_ret, total_km = (r1_out, r1_ret, dist1) if use_1 else (r2_out, r2_ret, dist2)
        km_outbound.append(sel_out["km"])
        km_return.append(sel_ret["km"])
        km_roundtrip.append(total_km)
        dur_out, dur_ret = sel_out["duration_min"], sel_ret["duration_min"]
        dur_outbound.append(dur_out)
        dur_return.append(dur_ret)
        dur_roundtrip.append(dur_out + dur_ret)

    df = df.copy()
    df["km_outbound_osrm"] = km_outbound
    df["km_return_osrm"] = km_return
    df["km_roundtrip_osrm"] = km_roundtrip
    df["duration_outbound_min_osrm"] = dur_outbound
    df["duration_return_min_osrm"] = dur_return
    df["duration_roundtrip_min_osrm"] = dur_roundtrip
    return df


def _build_routes_table(pair_results: dict) -> pd.DataFrame:
    records = [
        {"origin_lon": o[0], "origin_lat": o[1], "dest_lon": d[0], "dest_lat": d[1], **res}
        for (o, d), res in pair_results.items()
    ]
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Agreement analysis
# -----------------------------------------------------------------------------


def _build_osrm_merge_table(df_out: pd.DataFrame) -> pd.DataFrame:
    """Reduce the row-level OSRM output to the columns needed to merge it
    onto analysis_dataset.parquet for the agreement analysis.
    """
    df = df_out.copy()
    df["year"] = df[config.YEAR_COL]
    cols = AGREEMENT_MERGE_KEY + ["km_roundtrip_osrm", "duration_roundtrip_min_osrm"]
    return df[cols]


def _merge_for_agreement(analysis_df: pd.DataFrame, osrm_df: pd.DataFrame) -> pd.DataFrame:
    """One-to-one merge of the OSRM distances onto the final analysis
    population, on AGREEMENT_MERGE_KEY. GRUP_EDAT is normalized to a
    nullable integer dtype on both sides first: one side is stored as
    float64 and the other as int64 upstream, which otherwise silently
    yields an empty join.
    """
    a = analysis_df.copy()
    o = osrm_df.copy()
    a["GRUP_EDAT"] = a["GRUP_EDAT"].astype("Int64")
    o["GRUP_EDAT"] = o["GRUP_EDAT"].astype("Int64")
    return a.merge(o, on=AGREEMENT_MERGE_KEY, how="inner", validate="one_to_one")


def _population_weighted_totals(merged: pd.DataFrame) -> dict:
    """Population-weighted (N-weighted) total distance and time implied by
    each service, and the resulting percent difference: the quantity that
    bounds the impact on this study's headline totals.
    """
    total_km_ors = float((merged["km_roundtrip"] * merged["N"]).sum())
    total_km_osrm = float((merged["km_roundtrip_osrm"] * merged["N"]).sum())
    total_min_ors = float((merged["duration_roundtrip_min"] * merged["N"]).sum())
    total_min_osrm = float((merged["duration_roundtrip_min_osrm"] * merged["N"]).sum())
    return {
        "merged_rows": len(merged),
        "total_km_ors": total_km_ors,
        "total_km_osrm": total_km_osrm,
        "pct_diff_km": (total_km_ors - total_km_osrm) / total_km_ors * 100,
        "total_min_ors": total_min_ors,
        "total_min_osrm": total_min_osrm,
        "pct_diff_min": (total_min_ors - total_min_osrm) / total_min_ors * 100,
    }


def _matched_pairs_rank_biserial(diff: np.ndarray) -> float:
    """Non-parametric effect size matching the Wilcoxon signed-rank test
    (rank-biserial correlation on matched pairs). Range [-1, 1]; sign
    follows diff = a - b.
    """
    d = np.asarray(diff)
    d = d[d != 0]
    if len(d) == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(d))
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    return float((r_plus - r_minus) / (r_plus + r_minus))


def _bootstrap_ci(values: np.ndarray, statistic_func, n_resamples: int, random_state: int) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    values = np.asarray(values)
    n = len(values)
    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = values[rng.integers(0, n, size=n)]
        boot[i] = statistic_func(sample)
    alpha = 1 - config.CONFIDENCE_LEVEL
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Pearson correlation weighting each route by `w` (its total N)."""
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    return float(cov / np.sqrt(vx * vy))


def _route_level_comparison(df_routes: pd.DataFrame, col_a: str, col_b: str, label: str, unit: str) -> dict:
    """Route-level (deduplicated, unweighted) paired comparison between the
    ORS column `col_a` and the OSRM column `col_b`: normality, Wilcoxon
    signed-rank test, matched-pairs rank-biserial effect size with a
    bootstrap CI, median/mean difference, Pearson/Spearman correlation, a
    population-weighted Pearson correlation, and Bland-Altman agreement
    (parametric and percentile limits of agreement).

    Routes missing either engine's value are dropped first, so the comparison
    runs on pairwise-complete observations. A route can be missing an OSRM
    distance when OSRM declines the coordinate pair while ORS routes it, and
    leaving those in silently turns the mean, the correlation and the
    rank-biserial effect size into NaN: a single unroutable pair is enough to
    empty every one of them. The count dropped is returned as
    `n_routes_dropped` and logged, so a comparison resting on fewer routes than
    the run reports is visible rather than implied.
    """
    paired = df_routes[[col_a, col_b, "N"]].dropna(subset=[col_a, col_b])
    n_dropped = len(df_routes) - len(paired)
    if n_dropped:
        utils.log(
            f"  {label}: {n_dropped} of {len(df_routes)} routes dropped for lacking "
            f"a value from one engine; comparison uses {len(paired)} routes."
        )
    a = paired[col_a].to_numpy(dtype=float)
    b = paired[col_b].to_numpy(dtype=float)
    diff = a - b
    n = len(a)

    sample_n = min(n, config.SHAPIRO_SUBSAMPLE)
    if n > config.SHAPIRO_SUBSAMPLE:
        idx = np.random.default_rng(config.RANDOM_SEED).choice(n, sample_n, replace=False)
        sample_a, sample_b = a[idx], b[idx]
    else:
        sample_a, sample_b = a, b
    shapiro_a = stats.shapiro(sample_a)
    shapiro_b = stats.shapiro(sample_b)

    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(a, b)

    r_rb = _matched_pairs_rank_biserial(diff)
    rb_ci = _bootstrap_ci(diff, _matched_pairs_rank_biserial, config.N_BOOTSTRAP, config.RANDOM_SEED)

    median_diff = float(np.median(diff))
    median_ci = _bootstrap_ci(diff, np.median, config.N_BOOTSTRAP, config.RANDOM_SEED)
    mean_diff = float(np.mean(diff))

    pearson_r, pearson_p = stats.pearsonr(a, b)
    spearman_r, spearman_p = stats.spearmanr(a, b)
    weighted_r = _weighted_pearson(a, b, paired["N"].to_numpy(dtype=float))

    bias = mean_diff
    sd_diff = float(np.std(diff, ddof=1))
    loa_param = (bias - 1.96 * sd_diff, bias + 1.96 * sd_diff)
    loa_pct = tuple(float(v) for v in np.percentile(diff, [2.5, 97.5]))

    return {
        "label": label, "unit": unit, "n_routes": n, "n_routes_dropped": n_dropped,
        "shapiro_a_w": float(shapiro_a.statistic), "shapiro_a_p": float(shapiro_a.pvalue),
        "shapiro_b_w": float(shapiro_b.statistic), "shapiro_b_p": float(shapiro_b.pvalue),
        "wilcoxon_statistic": float(wilcoxon_stat), "wilcoxon_p": float(wilcoxon_p),
        "rank_biserial_r": r_rb, "rank_biserial_ci_low": rb_ci[0], "rank_biserial_ci_high": rb_ci[1],
        "median_diff": median_diff, "median_diff_ci_low": median_ci[0], "median_diff_ci_high": median_ci[1],
        "mean_diff": mean_diff,
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "weighted_pearson_r": weighted_r,
        "ba_bias": bias, "ba_sd_diff": sd_diff,
        "ba_loa_parametric_low": loa_param[0], "ba_loa_parametric_high": loa_param[1],
        "ba_loa_percentile_low": loa_pct[0], "ba_loa_percentile_high": loa_pct[1],
    }


def _reference_check_rows(dist_result: dict) -> pd.DataFrame:
    """Compare the observed route-level distance statistics against the
    previously published reference values, flagging any disagreement
    beyond _REFERENCE_TOLERANCE (relative).
    """
    observed = {
        "Pearson r": dist_result["pearson_r"],
        "Spearman rho": dist_result["spearman_r"],
        "Bland-Altman bias (km)": dist_result["ba_bias"],
        "Parametric LoA low (km)": dist_result["ba_loa_parametric_low"],
        "Parametric LoA high (km)": dist_result["ba_loa_parametric_high"],
        "Rank-biserial r": dist_result["rank_biserial_r"],
    }
    rows = []
    for name, published in _PUBLISHED_REFERENCE.items():
        obs = observed[name]
        denom = abs(published) if published != 0 else 1.0
        rel_diff = abs(obs - published) / denom
        flag = "OK" if rel_diff <= _REFERENCE_TOLERANCE else "DISAGREES"
        rows.append((name, published, obs, rel_diff, flag))
    rows.append((
        "Wilcoxon p-value", "<0.001", dist_result["wilcoxon_p"], None,
        "OK" if dist_result["wilcoxon_p"] < 0.001 else "DISAGREES",
    ))
    return pd.DataFrame(rows, columns=["Metric", "Published", "Observed", "RelativeDiff", "Flag"])


def _metric_rows(result: dict) -> pd.DataFrame:
    unit = result["unit"]
    rows = [
        ("Unique routes (n)", result["n_routes"], None),
        ("Shapiro-Wilk W (ORS)", result["shapiro_a_w"], None),
        ("Shapiro-Wilk p (ORS)", result["shapiro_a_p"], None),
        ("Shapiro-Wilk W (OSRM)", result["shapiro_b_w"], None),
        ("Shapiro-Wilk p (OSRM)", result["shapiro_b_p"], None),
        ("Wilcoxon statistic", result["wilcoxon_statistic"], None),
        ("Wilcoxon p-value", result["wilcoxon_p"], None),
        ("Rank-biserial r (matched pairs)", result["rank_biserial_r"], None),
        ("Rank-biserial 95% CI low", result["rank_biserial_ci_low"], None),
        ("Rank-biserial 95% CI high", result["rank_biserial_ci_high"], None),
        ("Median difference (ORS - OSRM)", result["median_diff"], unit),
        ("Median difference 95% CI low", result["median_diff_ci_low"], unit),
        ("Median difference 95% CI high", result["median_diff_ci_high"], unit),
        ("Mean difference (reference only, assumes normality)", result["mean_diff"], unit),
        ("Pearson r", result["pearson_r"], None),
        ("Pearson p-value", result["pearson_p"], None),
        ("Spearman rho", result["spearman_r"], None),
        ("Spearman p-value", result["spearman_p"], None),
        ("Population-weighted Pearson r (routes weighted by total N)", result["weighted_pearson_r"], None),
        ("Bland-Altman bias (mean difference)", result["ba_bias"], unit),
        ("Bland-Altman SD of differences", result["ba_sd_diff"], unit),
        ("Parametric 95% limits of agreement, low", result["ba_loa_parametric_low"], unit),
        ("Parametric 95% limits of agreement, high", result["ba_loa_parametric_high"], unit),
        ("Percentile 95% limits of agreement, low", result["ba_loa_percentile_low"], unit),
        ("Percentile 95% limits of agreement, high", result["ba_loa_percentile_high"], unit),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Notes"])


def _write_agreement_workbook(path, pop_weighted: dict, dist_result: dict, time_result: dict) -> None:
    readme = pd.DataFrame({"README": [
        "Agreement between OpenRouteService (ORS, the primary routing engine used "
        "throughout this study) and OSRM (an independent routing engine) on the same "
        "origin-destination pairs.",
        "Sheet population_weighted: the population-weighted (N-weighted) total distance "
        "and travel time implied by each service -- the quantity that bounds the impact "
        "on every headline, N-weighted figure in the analysis.",
        "Sheets route_level_distance / route_level_time: a route-level (deduplicated by "
        "origin-destination pair, unweighted) paired comparison, to avoid the "
        "pseudo-replication that would result from comparing the same route repeated "
        "once per year/sex/age-group stratum.",
        "Sheet reference_check: the route-level distance statistics against previously "
        "published reference values, flagged where they disagree.",
    ]})

    pop_rows = pd.DataFrame([
        ("Merged rows (one-to-one, analysis_dataset x OSRM)", pop_weighted["merged_rows"], None),
        ("Population-weighted total distance, ORS", pop_weighted["total_km_ors"], "km"),
        ("Population-weighted total distance, OSRM", pop_weighted["total_km_osrm"], "km"),
        ("Distance difference (ORS - OSRM), percent of ORS", pop_weighted["pct_diff_km"], "%"),
        ("Population-weighted total time, ORS", pop_weighted["total_min_ors"], "min"),
        ("Population-weighted total time, OSRM", pop_weighted["total_min_osrm"], "min"),
        ("Time difference (ORS - OSRM), percent of ORS", pop_weighted["pct_diff_min"], "%"),
    ], columns=["Metric", "Value", "Notes"])

    with pd.ExcelWriter(path) as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        pop_rows.to_excel(writer, sheet_name="population_weighted", index=False)
        _metric_rows(dist_result).to_excel(writer, sheet_name="route_level_distance", index=False)
        _metric_rows(time_result).to_excel(writer, sheet_name="route_level_time", index=False)
        _reference_check_rows(dist_result).to_excel(writer, sheet_name="reference_check", index=False)

    utils.log(f"Saved: {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    utils.log("=" * 60)
    utils.log("Step 05 - OSRM routing and ORS/OSRM agreement analysis")
    utils.log("=" * 60)

    input_path = utils.resolve_input(INPUT_FILENAME)
    df = utils.read_table(input_path)
    utils.log(f"Loaded {len(df):,} rows from {input_path}")

    missing_cols = [c for c in REQUIRED_COORD_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    before = len(df)
    df_complete = df.dropna(subset=REQUIRED_COORD_COLUMNS).copy()
    utils.log(f"Rows with complete coordinates: {len(df_complete):,} of {before:,}")

    cache_path = utils.resolve_input(CACHE_FILENAME)
    cache = _load_cache(cache_path)

    pairs = _build_pairs(df_complete)
    utils.log(f"Unique origin-destination pairs required: {len(pairs):,}")

    allow_network = (not args.use_cache) or args.refresh
    cache, stats_ = _resolve_pairs(pairs, cache, allow_network)
    utils.log(
        f"Pairs: {stats_['hits']:,} cache hits, {stats_['retried_failures']:,} cached failures, "
        f"{stats_['new_pairs']:,} never cached, {stats_['missing']:,} left unroutable this run, "
        f"{stats_['routed_live']:,} routed live this run."
    )

    df_out = _apply_routes(df_complete, cache)

    cache_out_path = config.output_path("cache", CACHE_FILENAME)
    with open(cache_out_path, "wb") as f:
        pickle.dump(cache, f)
    utils.log(f"Saved updated cache: {cache_out_path} ({len(cache):,} entries)")

    routes_df = _build_routes_table(cache)
    utils.write_table(routes_df, config.output_path(ROUTES_OUTPUT))
    utils.log(f"Saved: {config.output_path(ROUTES_OUTPUT)}")

    utils.write_table(df_out, config.output_path(DISTANCES_OUTPUT))
    utils.log(f"Saved: {config.output_path(DISTANCES_OUTPUT)}")

    n_missing_km = int(df_out["km_roundtrip_osrm"].isna().sum())
    utils.log(f"Rows without a routed OSRM round-trip distance: {n_missing_km:,} of {len(df_out):,}")

    utils.record_number("osrm_unique_pairs", len(pairs), "pairs", "s05_route_osrm")
    utils.record_number("osrm_cache_hits", stats_["hits"], "pairs", "s05_route_osrm")
    utils.record_number("osrm_pairs_unroutable", stats_["missing"], "pairs", "s05_route_osrm")
    utils.record_number("osrm_rows_without_distance", n_missing_km, "rows", "s05_route_osrm")

    utils.log("Loading the analysis dataset for the ORS/OSRM agreement analysis...")
    analysis_path = utils.resolve_input("analysis_dataset.parquet")
    analysis_df = utils.read_table(analysis_path)
    utils.log(f"Loaded {len(analysis_df):,} rows from {analysis_path}")

    osrm_merge = _build_osrm_merge_table(df_out)
    merged = _merge_for_agreement(analysis_df, osrm_merge)
    utils.log(f"Merged {len(merged):,} rows (one-to-one) for the agreement analysis.")

    pop_weighted = _population_weighted_totals(merged)
    utils.log(
        f"Population-weighted total distance: ORS {pop_weighted['total_km_ors']:,.2f} km, "
        f"OSRM {pop_weighted['total_km_osrm']:,.2f} km ({pop_weighted['pct_diff_km']:.3f}% lower)"
    )
    utils.log(
        f"Population-weighted total time: ORS {pop_weighted['total_min_ors']:,.2f} min, "
        f"OSRM {pop_weighted['total_min_osrm']:,.2f} min ({pop_weighted['pct_diff_min']:.3f}% lower)"
    )

    agg_dict = {
        "km_roundtrip": "first", "km_roundtrip_osrm": "first",
        "duration_roundtrip_min": "first", "duration_roundtrip_min_osrm": "first",
        "N": "sum",
    }
    df_routes = merged.groupby(config.ROUTE_KEY).agg(agg_dict).reset_index()
    utils.log(f"Unique routes for the route-level comparison: {len(df_routes):,}")

    dist_result = _route_level_comparison(df_routes, "km_roundtrip", "km_roundtrip_osrm", "Distance", "km")
    time_result = _route_level_comparison(
        df_routes, "duration_roundtrip_min", "duration_roundtrip_min_osrm", "Time", "min"
    )

    agreement_path = config.output_path(AGREEMENT_OUTPUT)
    _write_agreement_workbook(agreement_path, pop_weighted, dist_result, time_result)

    utils.record_number("osrm_weighted_total_km", pop_weighted["total_km_osrm"], "km", "s05_route_osrm")
    utils.record_number("osrm_weighted_total_min", pop_weighted["total_min_osrm"], "min", "s05_route_osrm")
    utils.record_number("ors_osrm_pct_diff_km", pop_weighted["pct_diff_km"], "%", "s05_route_osrm")
    utils.record_number("ors_osrm_pct_diff_min", pop_weighted["pct_diff_min"], "%", "s05_route_osrm")
    utils.record_number("ors_osrm_route_level_n", dist_result["n_routes"], "routes", "s05_route_osrm")
    utils.record_number("ors_osrm_distance_pearson_r", dist_result["pearson_r"], "unitless", "s05_route_osrm")
    utils.record_number(
        "ors_osrm_distance_rank_biserial", dist_result["rank_biserial_r"], "unitless", "s05_route_osrm"
    )
    utils.record_number("ors_osrm_distance_ba_bias_km", dist_result["ba_bias"], "km", "s05_route_osrm")

    utils.log("Done.")
    utils.flush_numbers("s05_route_osrm")


if __name__ == "__main__":
    main()
