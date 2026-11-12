"""s01: Preprocess the historical derivations export.

Reads the raw historical export (raw_export_historical.xlsx) and the
provider-unit catalogue (UP_codis.xlsx), standardizes origin and destination
facility names, drops interconsultations that are not tied to a physical
service, splits the handful of names that denote two co-managed sites, and
joins the provider-unit catalogue onto the resulting unique location list.

This module cleans facility names first, with
utils.clean_facility_names(..., case_sensitive=False), and only then checks
the cleaned names against config.NONPHYSICAL_KEYWORDS, case-insensitively,
via utils.is_nonphysical(..., case_sensitive=False). The 2025 refresh stream
in s03_integrate_2025.py runs the opposite order on both axes: it matches
against the raw names, before cleaning, case-sensitively. Neither
convention is a mismatch waiting to be tidied up. Each produced its half of
the published Figure 1 funnel counts, so aligning one stream with the other
would silently change those counts rather than fix anything. See
utils.is_nonphysical for the shared explanation.

Output:
    cleaned_locations_for_geocoding.parquet
        One row per unique cleaned location name, with placeholder
        geocoding fields and provider-unit catalogue codes/descriptions.
    processed_export.parquet
        The full interconsultation-level dataset with cleaned, standardized
        and split origin/destination names attached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from pipeline import utils

RAW_FILE = "raw_export_historical.xlsx"
UP_CODIS_FILE = "UP_codis.xlsx"

OUTPUT_LOCATIONS_FILE = "cleaned_locations_for_geocoding.parquet"
OUTPUT_PROCESSED_FILE = "processed_export.parquet"

ORIGIN_COL = "UP_DES"
DESTINATION_COL = "UP_DES_1"

# Column in UP_codis.xlsx (after stripping whitespace from the header row)
# holding the facility name to match against.
UP_CODIS_NAME_COL = "descripció_subtipus_UP"
UP_CODIS_CODE_COLS = {
    "codi_ep": "codi EP",
    "descripcio_ep": "descripció EP",
    "codi_tipus_up": "codi tipus UP",
    "descripcio_tipus_up": "descripció tipus UP",
    "codi_subtipus_up": "codi_subtipus_UP",
}


def _drop_missing_origin_or_destination(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop rows with no usable origin or destination name: NaN, empty, or
    whitespace-only strings in either UP_DES or UP_DES_1. Returns the
    filtered frame plus the number of records and interconsultations
    dropped, for the Figure 1 funnel.
    """
    before_records = len(df)
    before_n = int(df[config.WEIGHT_COL].sum())

    df = df.dropna(subset=[ORIGIN_COL, DESTINATION_COL], how="any")
    blank = (
        df[ORIGIN_COL].astype(str).str.strip().isin(["", "nan"])
        | df[DESTINATION_COL].astype(str).str.strip().isin(["", "nan"])
    )
    df = df[~blank].copy()

    dropped_records = before_records - len(df)
    dropped_n = before_n - int(df[config.WEIGHT_COL].sum())
    return df, dropped_records, dropped_n


def _load_up_codis_maps(path) -> tuple[dict, dict, dict, dict, dict]:
    """Build the five code/description lookup dicts from UP_codis.xlsx,
    keyed by the (stripped) facility-description string. All five dicts
    share the same keys, built in the same iteration order, which the
    matching logic in _match_catalogue relies on (see its docstring).
    """
    up_codis = utils.read_table(path)
    up_codis.columns = up_codis.columns.str.strip()

    code_ep, desc_ep, code_tipus, desc_tipus, code_subtipus = {}, {}, {}, {}, {}
    names = up_codis[UP_CODIS_NAME_COL].astype(str).str.strip()
    for name, (_, row) in zip(names, up_codis.iterrows()):
        if not name or name == "nan":
            continue
        code_ep[name] = row.get(UP_CODIS_CODE_COLS["codi_ep"])
        desc_ep[name] = row.get(UP_CODIS_CODE_COLS["descripcio_ep"])
        code_tipus[name] = row.get(UP_CODIS_CODE_COLS["codi_tipus_up"])
        desc_tipus[name] = row.get(UP_CODIS_CODE_COLS["descripcio_tipus_up"])
        code_subtipus[name] = row.get(UP_CODIS_CODE_COLS["codi_subtipus_up"])
    return code_ep, desc_ep, code_tipus, desc_tipus, code_subtipus


def _match_catalogue_key(cleaned_name: str, catalogue_names: list[str]) -> str | None:
    """Find the UP_codis catalogue key that best matches `cleaned_name`.

    Reproduces the original heuristic, in order:
      1. an exact match against a catalogue key;
      2. a catalogue key that also appears (as a raw origin/destination
         name, pre-cleaning) in `catalogue_names`, and whose lowercase form
         is a substring either way of the cleaned name's lowercase form, or
         shares a word longer than 3 characters with it;
      3. any catalogue key (longer than 3 characters) that is a substring
         either way of the cleaned name's lowercase form.

    `catalogue_names` is pre-sorted by the caller so that, unlike the
    original notebook (which iterated an unordered set, making the result
    depend on Python's randomized string hashing), the chosen match is
    deterministic across runs for ambiguous cases.

    All five UP_codis lookup dicts share the same keys (see
    _load_up_codis_maps), so the matched key is looked up once and reused
    for all five code/description columns rather than being recomputed
    per column, as the original did; the result is identical because the
    match depends only on the key set, not on which dict's values are
    being read.
    """
    if cleaned_name in catalogue_names:
        return cleaned_name

    cleaned_lower = cleaned_name.lower()
    cleaned_words = [w for w in cleaned_lower.split() if len(w) > 3]
    for candidate in catalogue_names:
        candidate_lower = candidate.lower()
        if (
            candidate_lower in cleaned_lower
            or cleaned_lower in candidate_lower
            or any(word in candidate_lower for word in cleaned_words)
        ):
            return candidate

    for candidate in catalogue_names:
        candidate_lower = candidate.lower()
        if len(candidate_lower) > 3 and (
            candidate_lower in cleaned_lower or cleaned_lower in candidate_lower
        ):
            return candidate
    return None


def _build_locations_catalogue(
    unique_locations: list[str],
    up_codis_path,
) -> pd.DataFrame:
    """Build the cleaned_locations_for_geocoding output: one row per unique
    cleaned location name, with placeholder geocoding fields and
    provider-unit catalogue codes/descriptions joined in.
    """
    locations_df = pd.DataFrame(
        {
            "location_name": unique_locations,
            "geocoded": False,
            "latitude": None,
            "longitude": None,
            "address": None,
            "geocoding_status": None,
        }
    )

    code_ep, desc_ep, code_tipus, desc_tipus, code_subtipus = _load_up_codis_maps(up_codis_path)
    catalogue_names = sorted(code_ep.keys())

    matched_keys = [_match_catalogue_key(name, catalogue_names) for name in unique_locations]
    locations_df["codi_ep"] = [code_ep.get(k) if k else None for k in matched_keys]
    locations_df["descripcio_ep"] = [desc_ep.get(k) if k else None for k in matched_keys]
    locations_df["codi_tipus_up"] = [code_tipus.get(k) if k else None for k in matched_keys]
    locations_df["descripcio_tipus_up"] = [desc_tipus.get(k) if k else None for k in matched_keys]
    locations_df["codi_subtipus_up"] = [code_subtipus.get(k) if k else None for k in matched_keys]
    return locations_df


LIST_COLUMNS = ("ORIGIN_LOCATIONS", "DESTINATION_LOCATIONS")


def _normalize_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the multi-location columns hold uniform Python lists so parquet
    stores them as a list type instead of as text.

    Parquet rejects an object column that mixes lists with non-lists, so every
    cell is coerced to a list: a missing value becomes an empty list and a bare
    string becomes a single-element list. Writing these as real lists keeps the
    stored schema identical to the rest of the pipeline's inputs, which matters
    because a later step appends freshly split rows to rows read back from this
    file; if one side were text and the other lists, the concatenated column
    could not be written at all.
    """
    df = df.copy()
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: list(x) if isinstance(x, (list, tuple, np.ndarray))
                else ([] if (x is None or (isinstance(x, float) and pd.isna(x))) else [x])
            )
    return df


def main() -> None:
    raw_path = config.data_path(RAW_FILE)
    up_codis_path = config.data_path(UP_CODIS_FILE)

    df = utils.read_table(raw_path)
    raw_records = len(df)
    raw_interconsultations = int(df[config.WEIGHT_COL].sum())
    utils.log(f"Loaded {raw_records:,} raw records from {RAW_FILE} "
              f"({raw_interconsultations:,} interconsultations).")
    utils.record_number("s01_raw_records", raw_records, "records", "s01_preprocess")
    utils.record_number(
        "s01_raw_interconsultations", raw_interconsultations, "interconsultations", "s01_preprocess"
    )

    # Figure 1 funnel, step 1: drop rows with no usable origin or destination name.
    df, dropped_missing_records, dropped_missing_n = _drop_missing_origin_or_destination(df)
    utils.log(
        f"Dropped {dropped_missing_records:,} records "
        f"({dropped_missing_n:,} interconsultations) for missing origin or destination."
    )
    utils.record_number(
        "s01_dropped_missing_origin_destination_records",
        dropped_missing_records,
        "records",
        "s01_preprocess",
    )
    utils.record_number(
        "s01_dropped_missing_origin_destination_interconsultations",
        dropped_missing_n,
        "interconsultations",
        "s01_preprocess",
    )

    df["ORIGIN_NAME_ORIGINAL"] = df[ORIGIN_COL].astype(str)
    df["DESTINATION_NAME_ORIGINAL"] = df[DESTINATION_COL].astype(str)

    # Clean the names first (case-insensitive lookup tables), then test the
    # cleaned names against NONPHYSICAL_KEYWORDS, also case-insensitively.
    # This ordering is what produced the published Figure 1 historical
    # counts; see the module docstring and utils.is_nonphysical for why the
    # 2025 stream in s03 runs the opposite order on purpose.
    df["ORIGIN_NAME_CLEAN"] = utils.clean_facility_names(df[ORIGIN_COL], case_sensitive=False)
    df["DESTINATION_NAME_CLEAN"] = utils.clean_facility_names(df[DESTINATION_COL], case_sensitive=False)

    # Figure 1 funnel, step 2: drop non-physical or virtual services.
    before_nonphys_records = len(df)
    before_nonphys_n = int(df[config.WEIGHT_COL].sum())
    nonphysical = utils.is_nonphysical(df["ORIGIN_NAME_CLEAN"], case_sensitive=False) | utils.is_nonphysical(
        df["DESTINATION_NAME_CLEAN"], case_sensitive=False
    )
    df = df[~nonphysical].copy()
    dropped_nonphysical_records = before_nonphys_records - len(df)
    dropped_nonphysical_n = before_nonphys_n - int(df[config.WEIGHT_COL].sum())
    utils.log(
        f"Dropped {dropped_nonphysical_records:,} records "
        f"({dropped_nonphysical_n:,} interconsultations) as non-physical services."
    )
    utils.record_number(
        "s01_dropped_nonphysical_records", dropped_nonphysical_records, "records", "s01_preprocess"
    )
    utils.record_number(
        "s01_dropped_nonphysical_interconsultations",
        dropped_nonphysical_n,
        "interconsultations",
        "s01_preprocess",
    )

    # Split the handful of names that denote two co-managed sites.
    # This runs once. The legacy notebook re-ran cleaning, filtering and
    # splitting three times across its cells, but the whole sequence is
    # idempotent, so repeating it would not have changed the result.
    df["ORIGIN_LOCATIONS"] = utils.split_multi_location(df["ORIGIN_NAME_CLEAN"])
    df["DESTINATION_LOCATIONS"] = utils.split_multi_location(df["DESTINATION_NAME_CLEAN"])
    df["ORIGIN_NAME_1"] = df["ORIGIN_LOCATIONS"].apply(lambda locs: locs[0] if locs else None)
    df["ORIGIN_NAME_2"] = df["ORIGIN_LOCATIONS"].apply(lambda locs: locs[1] if len(locs) > 1 else None)
    df["DESTINATION_NAME_1"] = df["DESTINATION_LOCATIONS"].apply(lambda locs: locs[0] if locs else None)
    df["DESTINATION_NAME_2"] = df["DESTINATION_LOCATIONS"].apply(lambda locs: locs[1] if len(locs) > 1 else None)

    utils.log(f"Final processed dataset: {len(df):,} records, {int(df[config.WEIGHT_COL].sum()):,} interconsultations.")
    utils.write_table(_normalize_list_columns(df), config.output_path(OUTPUT_PROCESSED_FILE))
    utils.log(f"Wrote {OUTPUT_PROCESSED_FILE}")

    # Unique cleaned locations, joined with the provider-unit catalogue.
    unique_locations = set()
    for col in ("ORIGIN_NAME_1", "ORIGIN_NAME_2", "DESTINATION_NAME_1", "DESTINATION_NAME_2"):
        unique_locations.update(df[col].dropna().unique())
    unique_locations = sorted(
        loc for loc in unique_locations if str(loc).strip() not in ("", "nan", "None")
    )

    locations_df = _build_locations_catalogue(unique_locations, up_codis_path)
    utils.write_table(locations_df, config.output_path(OUTPUT_LOCATIONS_FILE))
    matched = int(locations_df["codi_ep"].notna().sum())
    utils.log(
        f"Wrote {OUTPUT_LOCATIONS_FILE}: {len(locations_df):,} unique locations, "
        f"{matched:,} matched to a provider-unit catalogue code."
    )
    utils.flush_numbers("s01_preprocess")


if __name__ == "__main__":
    main()
