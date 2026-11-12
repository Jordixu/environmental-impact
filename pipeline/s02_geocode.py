"""Step 2: geocode locations and merge coordinates into the derivations dataset.

This module has three operating modes.

1. Default (no flags): the reproducible path, with no network access at
   all. It reuses the frozen, hand-validated ``cleaned_locations_geocoded.parquet``
   (located via ``utils.resolve_input``: a freshly produced copy in
   OUTPUT_DIR if one exists, otherwise the frozen copy shipped in DATA_DIR),
   merges its ``validated_*`` coordinate fields onto every row of the
   preprocessed derivations dataset, and writes:

       OUTPUT_DIR/cleaned_locations_geocoded.parquet
       OUTPUT_DIR/geocoded_interconsultations.parquet

   This path reproduces every published result. No API key is needed, and
   none is read.

2. ``--geocode``: a historical, opt-in path that re-derives raw coordinates
   from scratch with the Google Geocoding API. It is not needed to
   reproduce the published results, which come from the frozen,
   hand-validated file above; it exists only so the original
   data-collection step can be re-run if that is ever needed. Running it
   requires both the ``--geocode`` flag and a resolved
   ``config.google_api_key()``; without either, the module makes no
   network call. The legacy notebook this replaces had defined a batch
   geocoding function but never called it, used a literal placeholder API
   key, and never set a real success status, so none of that was
   reproducible or verifiable. This implementation actually calls the API,
   rate-limits and retries requests, and records a real ``geocoding_status``
   so a fresh run can be checked. Coordinates obtained this way are not
   validated: the ``validated_*`` fields used by the merge are only
   populated by the manual review tool, ``s02b_validate_locations.py``.

3. ``--resource-osm``: a second, opt-in path that re-sources a validated
   coordinate from OpenStreetMap/Nominatim instead of Google. This is what
   produced the published coordinates for 202 of the 659 hand-validated
   locations (the ones that were never actually moved during hand
   validation): their frozen result is already in the file the default
   path loads, so this flag is not needed to reproduce it, only to repeat
   or extend that pass. Pass ``--resource-osm-names PATH`` with a CSV
   ``location_name`` column to limit which locations are re-queried; left
   out, every location with a ``validated_address`` is in scope, and the
   count is logged. Each candidate query result is cached to
   OUTPUT_DIR/nominatim_resource_cache.json so re-running never re-queries.
   Requires the geopy package; without the flag, the module never imports
   it.

Run as a module:

    python -m pipeline.s02_geocode
    python -m pipeline.s02_geocode --geocode
    python -m pipeline.s02_geocode --resource-osm
    python -m pipeline.s02_geocode --resource-osm --resource-osm-names names.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

import config
from pipeline import utils

# -----------------------------------------------------------------------------
# Historical, opt-in Google geocoding
# -----------------------------------------------------------------------------
# These constants govern only the --geocode path. They are not part of
# config.py because this path is historical and opt-in: it is not exercised
# by the default, reproducible run of the pipeline.

GOOGLE_GEOCODE_SLEEP_S = 0.2  # throttle: 5 requests/second, well under quota
GOOGLE_GEOCODE_MAX_RETRIES = 3
GOOGLE_GEOCODE_BACKOFF_S = 1.5
GOOGLE_GEOCODE_REGION_BIAS = "Catalunya, Espana"
# Catalonia bounding box, used as a sanity check on returned coordinates
# (lat_min, lat_max, lon_min, lon_max).
CATALONIA_BBOX = (40.5, 42.9, 0.2, 3.4)

# A result status that is worth retrying rather than treating as a final
# failure (transient, quota-related).
_RETRYABLE_STATUSES = {"OVER_QUERY_LIMIT", "UNKNOWN_ERROR"}

LOCATION_SCHEMA_DEFAULTS = {
    "validated": False,
    "validated_latitude": pd.NA,
    "validated_longitude": pd.NA,
    "validated_address": pd.NA,
    "abs_region_code": pd.NA,
    "abs_region_name": pd.NA,
    "flagged": False,
    "flag_reason": pd.NA,
}


def _ensure_location_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing columns the rest of the pipeline expects on the
    locations table, with neutral defaults. Idempotent.
    """
    df = df.copy()
    for col, default in LOCATION_SCHEMA_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    return df


def geocode_google_address(location_name: str, api_key: str, session, region_bias: str = GOOGLE_GEOCODE_REGION_BIAS):
    """Historical path: call the Google Geocoding API once for a single
    location name, retrying on transient failures (network errors,
    timeouts, and the quota-related statuses in _RETRYABLE_STATUSES).
    Returns (longitude, latitude, formatted_address), or
    (None, None, None) if the location could not be resolved.
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": f"{location_name}, {region_bias}",
        "key": api_key,
        "region": "es",
        "language": "ca",
    }
    last_error = None
    for attempt in range(1, GOOGLE_GEOCODE_MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # network error, timeout, bad JSON, etc.
            last_error = exc
            time.sleep(GOOGLE_GEOCODE_BACKOFF_S * attempt)
            continue

        status = payload.get("status")
        if status == "OK" and payload.get("results"):
            top = payload["results"][0]
            loc = top["geometry"]["location"]
            return loc["lng"], loc["lat"], top.get("formatted_address")
        if status in _RETRYABLE_STATUSES:
            last_error = status
            time.sleep(GOOGLE_GEOCODE_BACKOFF_S * attempt)
            continue
        # ZERO_RESULTS, REQUEST_DENIED, INVALID_REQUEST: not retryable.
        return None, None, None

    utils.log(f"  geocoding failed for one location after {GOOGLE_GEOCODE_MAX_RETRIES} attempts: {last_error}")
    return None, None, None


def run_google_geocoding(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """Historical, opt-in: geocode every row of `df` still pending geocoding
    (geocoded is False, or latitude/longitude are missing) using the Google
    Geocoding API, one request at a time, rate-limited to
    GOOGLE_GEOCODE_SLEEP_S between calls.

    Unlike the sanitised legacy notebook, which defined this step but never
    invoked it and never set a real status, this function sets
    `geocoding_status` to "success" or "failed" and `geocoded` to True only
    on genuine success, so a fresh run can be checked. The health-region
    columns are filled in using `utils.assign_region` against the current
    regions file (config.REGIONS_GEOJSON); that file carries only a region
    name (field "nom"), not the numeric region code the legacy geojson
    used, so abs_region_code is left unset here. None of this affects the
    published results, which come from the hand-validated coordinates, and
    s06 reassigns regions by spatial join regardless.
    """
    out = _ensure_location_schema(df)
    for col, default in [("latitude", pd.NA), ("longitude", pd.NA), ("address", pd.NA), ("geocoding_status", pd.NA)]:
        if col not in out.columns:
            out[col] = default

    pending = out.index[(~out["geocoded"].astype(bool)) | out["latitude"].isna() | out["longitude"].isna()]
    utils.log(f"Geocoding {len(pending)} of {len(out)} locations via the Google Geocoding API (historical path).")

    import requests

    session = requests.Session()
    n_success = 0
    n_failed = 0
    for i, idx in enumerate(pending, start=1):
        name = str(out.loc[idx, "location_name"])
        lon, lat, address = geocode_google_address(name, api_key, session)
        in_bbox = (
            lon is not None
            and lat is not None
            and CATALONIA_BBOX[0] <= lat <= CATALONIA_BBOX[1]
            and CATALONIA_BBOX[2] <= lon <= CATALONIA_BBOX[3]
        )
        if in_bbox:
            out.loc[idx, "longitude"] = lon
            out.loc[idx, "latitude"] = lat
            out.loc[idx, "address"] = address
            out.loc[idx, "geocoded"] = True
            out.loc[idx, "geocoding_status"] = "success"
            n_success += 1
        else:
            out.loc[idx, "geocoding_status"] = "failed"
            n_failed += 1
        if i % 25 == 0 or i == len(pending):
            utils.log(f"  geocoding progress: {i}/{len(pending)}")
        time.sleep(GOOGLE_GEOCODE_SLEEP_S)

    utils.log(f"Geocoding complete: {n_success} succeeded, {n_failed} failed.")
    utils.record_number("s02_google_geocode_success", n_success, "count", "s02_geocode")
    utils.record_number("s02_google_geocode_failed", n_failed, "count", "s02_geocode")

    geocoded_mask = out["latitude"].notna() & out["longitude"].notna()
    if geocoded_mask.any():
        regioned = utils.assign_region(out.loc[geocoded_mask], "latitude", "longitude", "abs_region_name")
        out.loc[regioned.index, "abs_region_name"] = regioned["abs_region_name"].values
    return out


# -----------------------------------------------------------------------------
# Opt-in OpenStreetMap/Nominatim re-sourcing
# -----------------------------------------------------------------------------
# Governs only the --resource-osm path. Not part of config.py for the same
# reason as the Google constants above: this path is historical and opt-in,
# not exercised by the default, reproducible run.

OSM_RESOURCE_CACHE_NAME = "nominatim_resource_cache.json"
OSM_RESOURCE_MAX_RETRIES = 3
OSM_RESOURCE_RETRY_BACKOFF_S = 3.0  # increasing sleep between retries of one query


def _osm_candidate_queries(name: str, address) -> list[str]:
    """Query strings to try against Nominatim, most specific first.

    The facility name with its address is the best shot. Failing that, the
    address alone usually resolves to the right street, the name with the
    town is a reasonable fallback, and the name with the tail of the
    address (usually town, province, country) is a last attempt for places
    whose full address string does not geocode on its own.
    """
    name = "" if pd.isna(name) else str(name).strip()
    address = "" if pd.isna(address) else str(address).strip()
    queries = []
    if name and address:
        queries.append(f"{name}, {address}")
    if address:
        queries.append(address)
    if name:
        queries.append(f"{name}, Catalunya, Spain")
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if name and len(parts) >= 2:
        queries.append(f"{name}, {', '.join(parts[-3:])}")
    seen, out = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _load_osm_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_osm_cache(cache: dict, path: Path) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def _select_osm_targets(locations: pd.DataFrame, names_path: str | None) -> pd.DataFrame:
    """Pick which rows to re-source against Nominatim.

    The one-off pass that produced the published coordinates picked its
    259 candidates by comparing validated_latitude/longitude against the
    raw latitude/longitude the original Google geocoding returned, but that
    raw pair has since been dropped from the locations file, so that
    comparison can no longer be reconstructed here. An explicit names file
    is the precise way to repeat that pass, or to target any other list of
    locations. Without one, every location with a validated_address is
    treated as in scope, and the count is logged so this never silently
    re-sources everything without saying so.
    """
    if names_path:
        names_df = utils.read_table(Path(names_path))
        wanted = set(names_df["location_name"].astype(str))
        target = locations[locations["location_name"].astype(str).isin(wanted)].copy()
        utils.log(
            f"--resource-osm-names matched {len(target)} of {len(wanted)} requested location names."
        )
    else:
        target = locations[locations["validated_address"].notna()].copy()
        utils.log(
            f"No --resource-osm-names given: re-sourcing all {len(target)} locations "
            "that have a validated_address."
        )
    return target


def resource_osm_coordinates(locations: pd.DataFrame, names_path: str | None) -> pd.DataFrame:
    """OPT-IN: re-query Nominatim for the coordinate of selected locations,
    replacing validated_latitude/validated_longitude only where the new
    result lands within config.NOMINATIM_AUTO_ACCEPT_M of the one already
    held. Anything farther away, or with no result at all, is left
    untouched for manual review.

    Nominatim's usage policy is respected: one request per second at most
    (config.NOMINATIM_SLEEP_S after every live request), a descriptive user
    agent (config.nominatim_user_agent()), and no parallel requests. A
    timeout or service error is retried up to OSM_RESOURCE_MAX_RETRIES
    times with an increasing sleep before the query is given up on. Every
    query result, including an empty one, is cached to
    OUTPUT_DIR/nominatim_resource_cache.json immediately, so a re-run never
    repeats a request the cache already answered.
    """
    from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
    from geopy.geocoders import Nominatim

    target = _select_osm_targets(locations, names_path)

    cache_path = config.OUTPUT_DIR / OSM_RESOURCE_CACHE_NAME
    cache = _load_osm_cache(cache_path)
    geolocator = Nominatim(user_agent=config.nominatim_user_agent(), timeout=20)

    out = locations.copy()
    n_found = 0
    n_accepted = 0
    distances = []
    for i, row in enumerate(target.itertuples(), start=1):
        hit = None
        for q in _osm_candidate_queries(row.location_name, row.validated_address):
            if q in cache:
                res = cache[q]
            else:
                res = {}
                for attempt in range(OSM_RESOURCE_MAX_RETRIES):
                    try:
                        loc = geolocator.geocode(q, exactly_one=True, country_codes="es")
                        res = {"lat": loc.latitude, "lon": loc.longitude} if loc else {}
                        break
                    except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError):
                        if attempt == OSM_RESOURCE_MAX_RETRIES - 1:
                            res = {}
                        else:
                            time.sleep(OSM_RESOURCE_RETRY_BACKOFF_S * (attempt + 1))
                cache[q] = res
                _save_osm_cache(cache, cache_path)
                time.sleep(config.NOMINATIM_SLEEP_S)
            if res:
                hit = res
                break

        if hit:
            n_found += 1
            d = utils.haversine_m(row.validated_latitude, row.validated_longitude, hit["lat"], hit["lon"])
            distances.append(d)
            if d <= config.NOMINATIM_AUTO_ACCEPT_M:
                out.loc[row.Index, "validated_latitude"] = hit["lat"]
                out.loc[row.Index, "validated_longitude"] = hit["lon"]
                n_accepted += 1

        if i % 25 == 0 or i == len(target):
            utils.log(f"  OSM re-sourcing progress: {i}/{len(target)}, {n_accepted} accepted so far")

    n_queried = len(target)
    n_no_result = n_queried - n_found
    n_review = n_queried - n_accepted
    utils.log(
        f"OSM re-sourcing complete: {n_queried} queried, {n_no_result} returned nothing, "
        f"{n_accepted} accepted (within {config.NOMINATIM_AUTO_ACCEPT_M:.0f} m), "
        f"{n_review} need manual review."
    )
    if distances:
        d_arr = pd.Series(distances)
        utils.log(
            "  distance between the held coordinate and the Nominatim result, for the "
            f"{len(distances)} with a result (m): median {d_arr.median():.1f}, "
            f"p90 {d_arr.quantile(0.9):.1f}, max {d_arr.max():.1f}"
        )
    utils.record_number("s02_osm_resourced_queried", n_queried, "count", "s02_geocode")
    utils.record_number("s02_osm_resourced_accepted", n_accepted, "count", "s02_geocode")
    return out


# -----------------------------------------------------------------------------
# Merge validated coordinates into the derivations dataset
# -----------------------------------------------------------------------------


def _column_or_na(df: pd.DataFrame, col: str, index: pd.Index) -> pd.Series:
    """Return `df[col]` if present, else an all-NA series aligned to `index`."""
    if col in df.columns:
        return df[col]
    return pd.Series(pd.NA, index=index)


def build_merged_dataset(processed: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    """Merge hand-validated coordinates onto every row of the preprocessed
    derivations dataset, for both the origin and the destination location.

    Reproduces the integration logic of the legacy coordinate-merge step
    (02_Geocoding_Coordinates notebook, the "integrate with preprocessed
    data" section): the displayed location name is always the original,
    uncleaned name, while the coordinates, address and health-region
    fields come only from the hand-validated validated_* columns, never
    from the raw, unvalidated latitude/longitude, looked up by the cleaned
    name used for geocoding (ORIGIN_NAME_1 / DESTINATION_NAME_1). A
    location without a validated coordinate ends up with a missing
    coordinate here rather than falling back to the raw geocoded one.
    """
    mapping_cols = [
        c
        for c in (
            "validated_latitude",
            "validated_longitude",
            "validated_address",
            "abs_region_code",
            "abs_region_name",
        )
        if c in locations.columns
    ]
    lut = (
        locations[["location_name"] + mapping_cols]
        .drop_duplicates(subset="location_name")
        .set_index("location_name")
    )

    out = processed.copy()
    out["origin_location_name"] = out.get("ORIGIN_NAME_ORIGINAL", out.get("UP_DES"))
    out["destination_location_name"] = out.get("DESTINATION_NAME_ORIGINAL", out.get("UP_DES_1"))

    def attach(prefix: str, key_col: str) -> None:
        joined = out[[key_col]].join(lut, on=key_col)
        lat = _column_or_na(joined, "validated_latitude", out.index)
        lon = _column_or_na(joined, "validated_longitude", out.index)
        has_coords = (lat.notna() & lon.notna()).fillna(False)

        out[f"{prefix}_latitude"] = lat.where(has_coords)
        out[f"{prefix}_longitude"] = lon.where(has_coords)
        out[f"{prefix}_address"] = _column_or_na(joined, "validated_address", out.index).where(has_coords)
        out[f"{prefix}_abs_region_code"] = _column_or_na(joined, "abs_region_code", out.index).where(has_coords)
        out[f"{prefix}_abs_region_name"] = _column_or_na(joined, "abs_region_name", out.index).where(has_coords)
        out[f"{prefix}_geocoded"] = has_coords

    attach("origin", "ORIGIN_NAME_1")
    attach("destination", "DESTINATION_NAME_1")
    return out


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Geocode locations and merge coordinates into the derivations "
            "dataset. Default: reuse the frozen, hand-validated geocoded "
            "locations, no network access. --geocode: historical, opt-in "
            "re-geocoding via the Google Geocoding API (requires "
            "GOOGLE_MAPS_API_KEY)."
        )
    )
    parser.add_argument(
        "--geocode",
        action="store_true",
        help=(
            "HISTORICAL, OPT-IN: call the Google Geocoding API to re-geocode "
            "every pending location from scratch. Requires GOOGLE_MAPS_API_KEY. "
            "Not needed to reproduce the published results; the resulting "
            "coordinates are also not validated until s02b_validate_locations.py "
            "is run."
        ),
    )
    parser.add_argument(
        "--resource-osm",
        action="store_true",
        help=(
            "OPT-IN: re-query OpenStreetMap/Nominatim for the coordinate of "
            "selected validated locations, and replace it when the new "
            "result lands within config.NOMINATIM_AUTO_ACCEPT_M of the one "
            "already held. Requires the geopy package. Not needed to "
            "reproduce the published results, which already reflect this "
            "pass; this flag exists so it can be repeated or extended. "
            "Combine with --resource-osm-names to limit which locations are "
            "re-queried."
        ),
    )
    parser.add_argument(
        "--resource-osm-names",
        metavar="PATH",
        default=None,
        help=(
            "With --resource-osm: path to a CSV with a 'location_name' "
            "column naming which locations to re-source. If omitted, every "
            "location with a validated_address is re-sourced, and the count "
            "is logged."
        ),
    )
    args = parser.parse_args()

    if args.geocode:
        try:
            api_key = config.google_api_key()
        except RuntimeError as exc:
            utils.log(f"ERROR: --geocode requires a Google Maps API key. {exc}")
            sys.exit(1)

        loc_path = utils.resolve_input("cleaned_locations_for_geocoding.parquet")
        utils.log(f"Loading locations pending geocoding from {loc_path}")
        locations = utils.read_table(loc_path)
        locations = run_google_geocoding(locations, api_key)
    else:
        loc_path = utils.resolve_input("cleaned_locations_geocoded.parquet")
        utils.log(f"Reusing frozen, hand-validated geocoded locations from {loc_path} (no API calls).")
        locations = utils.read_table(loc_path)

    if args.resource_osm_names and not args.resource_osm:
        utils.log("WARNING: --resource-osm-names has no effect without --resource-osm. Ignoring it.")

    if args.resource_osm:
        locations = resource_osm_coordinates(locations, args.resource_osm_names)

    locations = _ensure_location_schema(locations)

    utils.record_number("s02_total_locations", len(locations), "count", "s02_geocode")
    utils.record_number("s02_validated_locations", int(locations["validated"].sum()), "count", "s02_geocode")
    utils.record_number("s02_flagged_locations", int(locations["flagged"].sum()), "count", "s02_geocode")

    loc_out_path = config.output_path("cleaned_locations_geocoded.parquet")
    utils.write_table(locations, loc_out_path)
    utils.log(f"Wrote {loc_out_path}")

    processed_path = utils.resolve_input("processed_export.parquet")
    utils.log(f"Loading preprocessed derivations from {processed_path}")
    processed = utils.read_table(processed_path)

    merged = build_merged_dataset(processed, locations)

    total_rows = len(merged)
    origin_matched = int(merged["origin_geocoded"].sum())
    destination_matched = int(merged["destination_geocoded"].sum())
    utils.record_number("s02_merged_rows", total_rows, "count", "s02_geocode")
    utils.record_number("s02_origin_coords_matched", origin_matched, "count", "s02_geocode")
    utils.record_number("s02_destination_coords_matched", destination_matched, "count", "s02_geocode")
    utils.log(
        f"Merged {total_rows:,} rows: origin coordinates matched for "
        f"{origin_matched:,} ({origin_matched / total_rows * 100:.1f}%), "
        f"destination coordinates matched for {destination_matched:,} "
        f"({destination_matched / total_rows * 100:.1f}%)."
    )

    merged_out_path = config.output_path("geocoded_interconsultations.parquet")
    utils.write_table(merged, merged_out_path)
    utils.log(f"Wrote {merged_out_path}")
    utils.flush_numbers("s02_geocode")


if __name__ == "__main__":
    main()
