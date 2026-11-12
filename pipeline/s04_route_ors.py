"""Step 04: ORS routing.

Computes real road-network distance and duration for every unique
origin-destination coordinate pair in the referral dataset, using the
OpenRouteService (ORS) API with the `config.ORS_PROFILE` profile. Outbound
and return legs are routed as two independent calls and summed into a
round-trip total: there is no "multiply by two" shortcut, since a road
route is not generally symmetric. When a derivation has two candidate
destinations, both are routed and the shorter round trip is kept.

Routes are cached by coordinate pair, so re-running this step, or running
the rest of the pipeline standalone, never needs the network (see
`--use-cache` / `--refresh` below). The cache key is
`((origin_lon, origin_lat), (dest_lon, dest_lat))`, longitude first in each
tuple, verified against the frozen cache file. Outbound and return are
therefore two separate entries with reversed key order, since a route need
not be symmetric.

Two defects in the legacy notebook are fixed here. First, a cache hit used
to be decided by key presence alone, so a pair that had previously failed
and was cached with `km=None` was never retried on a later run; here, a
cached entry with no distance counts as a miss, not a hit, so it stays
eligible for a retry whenever live calls are permitted. Second, the legacy
notebook held a literal placeholder API-key string and compared it against
a slightly different literal when deciding whether to fall back to the
environment variable, so that fallback never actually triggered. This
module takes the key only from `config.ors_api_key()`, which raises a clear
error if the environment variable is unset; no placeholder string appears
anywhere in this file.

By default (`--use-cache`), this step never calls the API: any pair missing
from the cache, or cached as a past failure, is recorded as unroutable and
the run continues, reporting how many pairs were affected. Pass `--refresh`
to permit live ORS calls for those pairs. Calls are rate-limited by
`config.ORS_SLEEP_S` between requests, with a retry-with-backoff loop that
the legacy notebook never had.

Input:
  geocoded_interconsultations.parquet (resolved via
  utils.resolve_input)
  route_cache_ors.pkl (resolved via utils.resolve_input; the frozen cache in
  DATA_DIR is never touched, the updated cache is written under
  OUTPUT_DIR/cache/)

Output:
  OUTPUT_DIR/routed_ors.parquet - the input dataset plus the
    routed distance/duration/geometry columns.
  OUTPUT_DIR/route_table_ors.parquet - one row per unique routed
    coordinate pair (reference table).
  OUTPUT_DIR/cache/route_cache_ors.pkl - the updated route cache.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import openrouteservice
import pandas as pd

import config
from pipeline import utils

INPUT_FILENAME = "geocoded_interconsultations.parquet"
CACHE_FILENAME = "route_cache_ors.pkl"
DISTANCES_OUTPUT = "routed_ors.parquet"
ROUTES_OUTPUT = "route_table_ors.parquet"

REQUIRED_COORD_COLUMNS = [
    "origin_latitude", "origin_longitude", "destination_latitude", "destination_longitude",
]

MAX_RETRIES = 3

_EMPTY_ENTRY = {"km": None, "duracio_min": None, "geometry": None}

# How often to flush the cache to disk while routing live.
CHECKPOINT_EVERY = 25


class QuotaExhausted(RuntimeError):
    """The ORS account is out of quota, so no further request can succeed."""


def _is_quota_error(exc: Exception) -> bool:
    """True for "you have no quota left", false for "you are going too fast".

    ORS answers a daily-quota refusal with 403 and a rate-limit refusal with
    429. The two need opposite handling: a 429 clears if you wait a few
    seconds, a 403 does not clear until the quota window resets, so retrying
    it just burns wall-clock. Matching on the message as well as the status
    keeps an unrelated 403 (a bad key, say) on the ordinary retry path, where
    its error text ends up in the cache entry.
    """
    text = str(exc).lower()
    return "quota" in text and ("403" in text or "exceeded" in text)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 04 - route unique origin/destination pairs with OpenRouteService."
    )
    parser.add_argument(
        "--use-cache",
        dest="use_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only the cached routes; never call the ORS API (default: True).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        default=False,
        help="Permit live ORS calls for pairs missing from the cache or cached as a past failure.",
    )
    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
# Cache helpers
# -----------------------------------------------------------------------------


def _is_valid_cache_entry(entry) -> bool:
    """True if `entry` is a cached routing result with an actual distance.

    A cached entry with `km=None` (a past failure) does not count as a hit.
    That is the fix for the legacy "cache hit decided by key presence alone"
    bug.
    """
    return isinstance(entry, dict) and entry.get("km") is not None


def _load_cache(path) -> dict:
    if path.exists():
        with open(path, "rb") as f:
            cache = pickle.load(f)
        utils.log(f"Loaded ORS cache: {len(cache):,} entries from {path}")
        return cache
    utils.log(f"No existing ORS cache found at {path}; starting empty.")
    return {}


def _save_cache(cache: dict, path) -> None:
    """Write the cache, via a temporary file so an interrupted write cannot
    leave a half-written pickle behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    tmp.replace(path)


# -----------------------------------------------------------------------------
# Pair construction
# -----------------------------------------------------------------------------


def _has_alt_destination_columns(df: pd.DataFrame) -> bool:
    return "destination_latitude_2" in df.columns and "destination_longitude_2" in df.columns


def _build_pairs(df: pd.DataFrame) -> list[tuple]:
    """Unique (origin, destination) coordinate pairs to route, both
    directions, including the alternative destination when present.
    """
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


def _route_with_retry(client: "openrouteservice.Client", pair: tuple) -> dict:
    """Route one coordinate pair, retrying with exponential backoff on
    failure. Sleeps config.ORS_SLEEP_S after a successful call to respect
    the rate limit; sleeps a longer backoff before a retry.
    """
    origin, dest = pair
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            route = client.directions(
                coordinates=[list(origin), list(dest)],
                profile=config.ORS_PROFILE,
                format="geojson",
            )
            feat = route["features"][0]
            seg = feat["properties"]["segments"][0]
            entry = {
                "km": seg["distance"] / 1000.0,
                "duracio_min": seg["duration"] / 60.0,
                "geometry": json.dumps(feat["geometry"]),
            }
            time.sleep(config.ORS_SLEEP_S)
            return entry
        except Exception as exc:  # noqa: BLE001 - any ORS/network failure is retried the same way
            if _is_quota_error(exc):
                raise QuotaExhausted(str(exc)) from exc
            last_err = exc
            utils.log(f"  ORS routing attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            time.sleep(config.ORS_SLEEP_S * (2 ** attempt))
    utils.log(f"  Giving up on this pair after {MAX_RETRIES} attempts: {last_err}")
    return {"km": None, "duracio_min": None, "geometry": None, "error": str(last_err)}


def _resolve_pairs(
    pairs: list[tuple], cache: dict, allow_network: bool, checkpoint=None
) -> tuple[dict, dict]:
    """Resolve every pair against `cache`, updating it in place.

    Returns (cache, stats). `cache` doubles as the full pair-results dict:
    valid hits are left untouched, and misses (never cached, or cached as a
    past failure) are either routed live (if `allow_network`) or recorded
    as unroutable.

    `checkpoint`, when given, is called with the cache every
    CHECKPOINT_EVERY live routes. A full refresh is thousands of requests
    against a daily quota, so losing the lot to a dropped connection an hour
    in is worth one pickle write per 25 routes.
    """
    stats = {
        "total_pairs": len(pairs), "hits": 0, "retried_failures": 0,
        "new_pairs": 0, "missing": 0, "routed_live": 0,
        "quota_stopped": False, "not_attempted": 0,
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
        utils.log(f"--refresh: routing {len(pending):,} pending pairs live via ORS...")
        client = openrouteservice.Client(
            key=config.ors_api_key(), base_url=config.ORS_BASE_URL
        )
        utils.log(f"  ORS endpoint: {config.ORS_BASE_URL}")
        for i, pair in enumerate(pending, 1):
            try:
                cache[pair] = _route_with_retry(client, pair)
            except QuotaExhausted as exc:
                # Stop at once. Every later pair would fail the same way, and
                # the pairs already routed are paid for, so save and report
                # how many are left rather than grinding through the rest.
                stats["quota_stopped"] = True
                stats["not_attempted"] = len(pending) - i + 1
                if checkpoint is not None:
                    checkpoint(cache)
                utils.log(
                    f"  ORS quota exhausted after {stats['routed_live']:,} routes "
                    f"this run ({exc})."
                )
                utils.log(
                    f"  {stats['not_attempted']:,} pairs were not attempted and are not "
                    f"cached as failures. Re-run with --refresh once the quota resets; "
                    f"the routes already obtained are saved and will not be re-requested."
                )
                break
            stats["routed_live"] += 1
            if i % 20 == 0 or i == len(pending):
                utils.log(f"  Routed {i:,}/{len(pending):,} pending pairs...")
            if checkpoint is not None and (i % CHECKPOINT_EVERY == 0 or i == len(pending)):
                checkpoint(cache)

    return cache, stats


# -----------------------------------------------------------------------------
# Applying routes to the dataset
# -----------------------------------------------------------------------------


def _apply_routes(df: pd.DataFrame, pair_results: dict) -> pd.DataFrame:
    """Assign outbound/return/round-trip distance, duration and geometry to
    every row, selecting the shorter round trip when an alternative
    destination exists. The selection logic matches the legacy notebook.
    """
    has_alt = _has_alt_destination_columns(df)

    km_outbound, km_return, km_roundtrip = [], [], []
    dur_outbound, dur_return, dur_roundtrip = [], [], []
    geoms_outbound, geoms_return = [], []

    for row in df.itertuples(index=False):
        o = (row.origin_longitude, row.origin_latitude)
        d1 = (row.destination_longitude, row.destination_latitude)

        r1_out = pair_results.get((o, d1))
        r1_ret = pair_results.get((d1, o))
        cand1_total = None
        if r1_out and r1_ret and r1_out.get("km") is not None and r1_ret.get("km") is not None:
            cand1_total = r1_out["km"] + r1_ret["km"]

        cand2_total = None
        r2_out = r2_ret = None
        if has_alt:
            lat2, lon2 = row.destination_latitude_2, row.destination_longitude_2
            if pd.notna(lat2) and pd.notna(lon2):
                d2 = (lon2, lat2)
                r2_out = pair_results.get((o, d2))
                r2_ret = pair_results.get((d2, o))
                if r2_out and r2_ret and r2_out.get("km") is not None and r2_ret.get("km") is not None:
                    cand2_total = r2_out["km"] + r2_ret["km"]

        use_route2 = False
        if cand1_total is not None and cand2_total is not None:
            use_route2 = cand2_total < cand1_total
        elif cand1_total is None and cand2_total is not None:
            use_route2 = True

        if use_route2:
            sel_out, sel_ret, total_km = r2_out, r2_ret, cand2_total
        else:
            sel_out, sel_ret, total_km = r1_out, r1_ret, cand1_total

        if sel_out and sel_ret and total_km is not None:
            km_outbound.append(sel_out["km"])
            km_return.append(sel_ret["km"])
            km_roundtrip.append(total_km)
            dur_out, dur_ret = sel_out["duracio_min"], sel_ret["duracio_min"]
            dur_outbound.append(dur_out)
            dur_return.append(dur_ret)
            dur_roundtrip.append(dur_out + dur_ret)
            geoms_outbound.append(sel_out["geometry"])
            geoms_return.append(sel_ret["geometry"])
        else:
            km_outbound.append(None)
            km_return.append(None)
            km_roundtrip.append(None)
            dur_outbound.append(None)
            dur_return.append(None)
            dur_roundtrip.append(None)
            geoms_outbound.append(None)
            geoms_return.append(None)

    df = df.copy()
    df["km_outbound"] = km_outbound
    df["km_return"] = km_return
    df["km_roundtrip"] = km_roundtrip
    df["duration_outbound_min"] = dur_outbound
    df["duration_return_min"] = dur_return
    df["duration_roundtrip_min"] = dur_roundtrip
    df["route_geometry_outbound"] = geoms_outbound
    df["route_geometry_return"] = geoms_return
    return df


def _build_routes_table(pair_results: dict) -> pd.DataFrame:
    """One row per unique routed coordinate pair, for the reference output."""
    records = [
        {"origin_lon": o[0], "origin_lat": o[1], "dest_lon": d[0], "dest_lat": d[1], **res}
        for (o, d), res in pair_results.items()
    ]
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    utils.log("=" * 60)
    utils.log("Step 04 - ORS routing")
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
    cache_out_path = config.output_path("cache", CACHE_FILENAME)
    cache, stats = _resolve_pairs(
        pairs, cache, allow_network, checkpoint=lambda c: _save_cache(c, cache_out_path)
    )
    utils.log(
        f"Pairs: {stats['hits']:,} cache hits, {stats['retried_failures']:,} cached failures, "
        f"{stats['new_pairs']:,} never cached, {stats['missing']:,} left unroutable this run, "
        f"{stats['routed_live']:,} routed live this run."
    )
    if stats["quota_stopped"]:
        utils.log(
            f"INCOMPLETE: {stats['not_attempted']:,} coordinate pairs still have no route "
            f"because the ORS quota ran out. The outputs written below cover only the pairs "
            f"routed so far, so do not treat the downstream numbers as final until a "
            f"--refresh run reports 0 pairs left."
        )

    df_out = _apply_routes(df_complete, cache)

    _save_cache(cache, cache_out_path)
    utils.log(f"Saved updated cache: {cache_out_path} ({len(cache):,} entries)")

    routes_df = _build_routes_table(cache)
    utils.write_table(routes_df, config.output_path(ROUTES_OUTPUT))
    utils.log(f"Saved: {config.output_path(ROUTES_OUTPUT)}")

    utils.write_table(df_out, config.output_path(DISTANCES_OUTPUT))
    utils.log(f"Saved: {config.output_path(DISTANCES_OUTPUT)}")

    n_missing_km = int(df_out["km_roundtrip"].isna().sum())
    utils.log(f"Rows without a routed round-trip distance: {n_missing_km:,} of {len(df_out):,}")

    utils.record_number("ors_unique_pairs", len(pairs), "pairs", "s04_route_ors")
    utils.record_number("ors_cache_hits", stats["hits"], "pairs", "s04_route_ors")
    utils.record_number("ors_pairs_unroutable", stats["missing"], "pairs", "s04_route_ors")
    utils.record_number("ors_rows_without_distance", n_missing_km, "rows", "s04_route_ors")

    utils.log("Done.")
    utils.flush_numbers("s04_route_ors")


if __name__ == "__main__":
    main()
