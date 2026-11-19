"""Step 2: geocode locations and merge coordinates into the derivations dataset.

This module has two operating modes.

1. Default (no flags): the reproducible path, with no network access at
   all. It reuses the frozen, hand-validated ``cleaned_locations_geocoded.parquet``
   (located via ``utils.resolve_input``: a freshly produced copy in
   OUTPUT_DIR if one exists, otherwise the frozen copy shipped in DATA_DIR),
   merges its ``validated_*`` coordinate fields onto every row of the
   preprocessed derivations dataset, and writes:

       OUTPUT_DIR/cleaned_locations_geocoded.parquet
       OUTPUT_DIR/geocoded_interconsultations.parquet

   This path reproduces every published result. No network access and no
   API key are needed.

2. ``--geocode``: geocodes every pending location from scratch with
   Nominatim, the OpenStreetMap geocoding service. This is how the frozen
   file above is produced in the first place: ``--geocode`` writes raw
   ``latitude``/``longitude``/``address`` fields, the interactive tool
   ``s02b_validate_locations.py`` then has a reviewer check each one on an
   OpenStreetMap basemap and writes the ``validated_*`` fields, and that
   validated file is frozen. Coordinates from this path are therefore not
   used by the merge until they have been validated.

   Queries are restricted to Spain and to a bounding box around Catalonia
   (``config.CATALONIA_BBOX``), and a result is accepted only if it falls
   inside one of the health-region polygons in ``config.REGIONS_GEOJSON``,
   i.e. inside Catalonia. Nominatim's usage policy is respected: one
   request per ``config.NOMINATIM_SLEEP_S`` at most, a descriptive user
   agent (``config.nominatim_user_agent()``), and no parallel requests.
   Every query result, including an empty one, is cached to
   OUTPUT_DIR/cache/nominatim_geocode_cache.json, so a re-run never
   repeats a request the cache already answered. No API key is needed.
   Requires the geopy package; without the flag, the module never imports
   it.

Run as a module:

    python -m pipeline.s02_geocode
    python -m pipeline.s02_geocode --geocode
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import config
from pipeline import utils

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


# -----------------------------------------------------------------------------
# Opt-in Nominatim (OpenStreetMap) geocoding
# -----------------------------------------------------------------------------
# These constants govern only the --geocode path, which the default,
# reproducible run never exercises.

NOMINATIM_GEOCODE_CACHE_NAME = "nominatim_geocode_cache.json"
NOMINATIM_MAX_RETRIES = 3
NOMINATIM_RETRY_BACKOFF_S = 3.0  # increasing sleep between retries of one query


def _nominatim_queries(name) -> list[str]:
    """Query strings to try against Nominatim for one location, most
    specific first: the cleaned facility name qualified with the region,
    then the bare name. Both are bounded to Catalonia by the request
    itself, so the bare name is a fallback for names whose OpenStreetMap
    entry does not match the qualified string, not a looser search area.
    """
    name = "" if pd.isna(name) else str(name).strip()
    if not name:
        return []
    queries = [f"{name}, Catalunya", name]
    seen, out = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict, path: Path) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def _query_nominatim(geolocator, query: str) -> dict:
    """One forward-geocoding request, bounded to Spain and to
    config.CATALONIA_BBOX. Retries timeouts and service errors with an
    increasing sleep. Returns {"lat", "lon", "address"}, or {} for no result.
    """
    from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable

    lat_min, lat_max, lon_min, lon_max = config.CATALONIA_BBOX
    for attempt in range(NOMINATIM_MAX_RETRIES):
        try:
            loc = geolocator.geocode(
                query,
                exactly_one=True,
                country_codes=config.NOMINATIM_COUNTRY_CODE,
                viewbox=[(lat_max, lon_min), (lat_min, lon_max)],
                bounded=True,
            )
            if loc is None:
                return {}
            return {"lat": loc.latitude, "lon": loc.longitude, "address": loc.address}
        except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError) as exc:
            if attempt == NOMINATIM_MAX_RETRIES - 1:
                utils.log(f"  Nominatim query failed after {NOMINATIM_MAX_RETRIES} attempts: {exc}")
                return {}
            time.sleep(NOMINATIM_RETRY_BACKOFF_S * (attempt + 1))
    return {}


def run_nominatim_geocoding(df: pd.DataFrame) -> pd.DataFrame:
    """Geocode every row of `df` still pending geocoding (geocoded is False,
    or latitude/longitude are missing) with Nominatim, one request at a
    time.

    For each location the queries from _nominatim_queries are tried in
    order and the first result is kept. A result is then accepted only if
    a point-in-polygon join against the health-region boundaries places it
    inside Catalonia; one that lands outside every region is discarded.
    `geocoding_status` records the outcome for every attempted row:
    "success", "no_result" or "outside_catalonia", and `geocoded` is True
    only on success. The health-region name is filled in for successful
    rows; it is for the validator's information only, since s06 reassigns
    regions by spatial join from the validated coordinates regardless.
    """
    from geopy.geocoders import Nominatim

    out = _ensure_location_schema(df)
    for col, default in [
        ("geocoded", False),
        ("latitude", pd.NA),
        ("longitude", pd.NA),
        ("address", pd.NA),
        ("geocoding_status", pd.NA),
    ]:
        if col not in out.columns:
            out[col] = default

    pending = out.index[(~out["geocoded"].fillna(False).astype(bool)) | out["latitude"].isna() | out["longitude"].isna()]
    utils.log(f"Geocoding {len(pending)} of {len(out)} locations with Nominatim (OpenStreetMap).")

    cache_path = config.output_path("cache", NOMINATIM_GEOCODE_CACHE_NAME)
    cache = _load_cache(cache_path)
    geolocator = Nominatim(user_agent=config.nominatim_user_agent(), timeout=20)

    for i, idx in enumerate(pending, start=1):
        hit = {}
        for q in _nominatim_queries(out.loc[idx, "location_name"]):
            if q in cache:
                res = cache[q]
            else:
                res = _query_nominatim(geolocator, q)
                cache[q] = res
                _save_cache(cache, cache_path)
                time.sleep(config.NOMINATIM_SLEEP_S)
            if res:
                hit = res
                break

        if hit:
            out.loc[idx, "latitude"] = hit["lat"]
            out.loc[idx, "longitude"] = hit["lon"]
            out.loc[idx, "address"] = hit.get("address")
            out.loc[idx, "geocoding_status"] = "success"
        else:
            out.loc[idx, "geocoding_status"] = "no_result"

        if i % 25 == 0 or i == len(pending):
            utils.log(f"  geocoding progress: {i}/{len(pending)}")

    # Inside-Catalonia check: point-in-polygon against the health regions.
    found = out.index.isin(pending) & (out["geocoding_status"] == "success")
    if found.any():
        regioned = utils.assign_region(out.loc[found], "latitude", "longitude", "abs_region_name")
        regioned = regioned[~regioned.index.duplicated(keep="first")]
        out.loc[regioned.index, "abs_region_name"] = regioned["abs_region_name"].values
        outside = regioned.index[regioned["abs_region_name"].isna()]
        out.loc[outside, ["latitude", "longitude", "address"]] = pd.NA
        out.loc[outside, "geocoding_status"] = "outside_catalonia"

    attempted = out.loc[pending, "geocoding_status"]
    n_success = int((attempted == "success").sum())
    n_no_result = int((attempted == "no_result").sum())
    n_outside = int((attempted == "outside_catalonia").sum())
    out.loc[pending, "geocoded"] = attempted == "success"

    utils.log(
        f"Geocoding complete: {n_success} succeeded, {n_no_result} returned no result, "
        f"{n_outside} fell outside Catalonia and were discarded. All of them still "
        "need manual validation with s02b_validate_locations.py."
    )
    utils.record_number("s02_nominatim_geocode_success", n_success, "count", "s02_geocode")
    utils.record_number("s02_nominatim_geocode_no_result", n_no_result, "count", "s02_geocode")
    utils.record_number("s02_nominatim_geocode_outside_catalonia", n_outside, "count", "s02_geocode")
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
            "locations, no network access. --geocode: geocode every pending "
            "location from scratch with Nominatim (OpenStreetMap); no API key."
        )
    )
    parser.add_argument(
        "--geocode",
        action="store_true",
        help=(
            "OPT-IN: geocode every pending location from scratch with "
            "Nominatim (OpenStreetMap), bounded to Catalonia. Needs network "
            "access and the geopy package, but no API key. The resulting "
            "coordinates are not used by the merge until they have been "
            "validated with s02b_validate_locations.py."
        ),
    )
    args = parser.parse_args()

    if args.geocode:
        loc_path = utils.resolve_input("cleaned_locations_for_geocoding.parquet")
        utils.log(f"Loading locations pending geocoding from {loc_path}")
        locations = utils.read_table(loc_path)
        locations = run_nominatim_geocoding(locations)
    else:
        loc_path = utils.resolve_input("cleaned_locations_geocoded.parquet")
        utils.log(f"Reusing frozen, hand-validated geocoded locations from {loc_path} (no network access).")
        locations = utils.read_table(loc_path)

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
