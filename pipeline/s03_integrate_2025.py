"""s03: Integrate the complete 2025 export into the geocoded dataset.

The historical export (raw_export_historical.xlsx) carries a partial,
incomplete slice of 2025 referrals. A later, complete 2025 export
(raw_export_2025.txt) supersedes it. This step reconciles the two
according to config.YEAR_2025_MODE:

  - "refresh_replaces_partial" (the default; reproduces every published
    number): drop every row with YEAR_COL == 2025 from the existing
    dataset, then append the complete 2025 export, matched onto the
    existing geocoding by origin/destination health-centre code. There is
    no re-geocoding here: the 2025 export carries the same codes as the
    historical one, so its coordinates can be looked up instead of
    re-resolved.
  - "concatenate": append the complete 2025 export without first dropping
    the existing 2025 rows. This double counts the partial 2025 period,
    since every 2025 interconsultation already in the existing dataset
    gets counted a second time, and it changes every reported number
    derived from this output. It exists only as a sensitivity or
    regression check that the mode switch itself works, not to reproduce
    a published result.

2025-stream cleaning convention: this module matches
config.NONPHYSICAL_KEYWORDS against the raw UP_DES / UP_DES_1 values,
before any name cleaning, case-sensitively, via
utils.is_nonphysical(..., case_sensitive=True). s01_preprocess.py's
historical stream runs the opposite order on both axes: it cleans names
first, then matches case-insensitively. Neither convention is an oversight
waiting to be tidied up. Each produced its half of the published Figure 1
funnel counts, so swapping either one to match the other stream would
silently change those counts rather than correct anything. See
utils.is_nonphysical for the shared explanation.

Input:
    utils.resolve_input("geocoded_interconsultations.parquet")
        The existing geocoded dataset (s02's output, or the frozen copy).
    config.data_path("raw_export_2025.txt")
        The complete 2025 export. Comma-delimited UTF-8, with its header
        carrying two columns both literally named "UP_DES"; pandas renames
        the second to "UP_DES.1", which is renamed here to "UP_DES_1" to
        match the destination-name column used throughout the rest of the
        pipeline.

Output:
    OUTPUT_DIR/geocoded_interconsultations.parquet
        The integrated dataset. It overwrites the file this step reads,
        under OUTPUT_DIR; the frozen copy under DATA_DIR is never touched.
    OUTPUT_DIR/unroutable_2025.csv
        2025 rows whose origin and/or destination code did not resolve to a
        known coordinate via the lookup described below.

Data notes:
    - GRUP_EDAT is int64 in the 2025 export but float64 in the historical,
      Excel-derived stream. Both are normalised to a nullable Int64 before
      the final concatenation; an unresolved dtype mismatch here is the
      kind of thing that causes a silent zero-overlap join further down
      the pipeline.
    - The existing dataset may or may not carry the rurality columns
      (is_rural_origin, is_rural_destination, origin_municipality,
      destination_municipality): that enrichment step feeds no published
      result and has been dropped from some generations of the pipeline.
      They are carried through when present, but never required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from pipeline import utils

EXISTING_FILE = "geocoded_interconsultations.parquet"
RAW_2025_FILE = "raw_export_2025.txt"
UNMATCHED_FILE = "unroutable_2025.csv"

ORIGIN_CODE_COL = "OC_UP_ORI"
DEST_CODE_COL = "DER_UP_SCS_DES"
ORIGIN_COL = "UP_DES"
DESTINATION_COL = "UP_DES_1"

# Columns pulled from the existing dataset into the per-code coordinate
# lookups. The optional ones are included only if the existing dataset
# happens to carry them (see the module docstring).
_ORIGIN_LOOKUP_COLS = [
    "origin_latitude", "origin_longitude",
    "origin_abs_region_code", "origin_abs_region_name",
    "origin_geocoded", "origin_address", "origin_location_name",
]
_ORIGIN_OPTIONAL_COLS = ["origin_municipality", "is_rural_origin"]
_DEST_LOOKUP_COLS = [
    "destination_latitude", "destination_longitude",
    "destination_abs_region_code", "destination_abs_region_name",
    "destination_geocoded", "destination_address", "destination_location_name",
]
_DEST_OPTIONAL_COLS = ["destination_municipality", "is_rural_destination"]

_UNMATCHED_EXPORT_COLS = [
    ORIGIN_CODE_COL, ORIGIN_COL, DEST_CODE_COL, DESTINATION_COL, "N",
    "origin_latitude", "origin_longitude", "destination_latitude", "destination_longitude",
]


def _build_coordinate_lookups(existing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the origin/destination coordinate lookups keyed by
    health-centre code (OC_UP_ORI / DER_UP_SCS_DES), one row per code.

    Built from the full existing dataset, including the partial 2025 rows
    that are about to be dropped under "refresh_replaces_partial", not only
    from the rows that will be retained. A handful of origin/destination
    codes appear only among those partial-2025 rows (not among
    2019/2023/2024), so restricting the lookup to the retained rows
    measurably undercounts matches against the 2025 refresh: verified at
    206 rows / 307 interconsultations fewer than the regression target.
    This matches the behaviour of the legacy integration script, which
    also builds its lookups from the full existing dataset before removing
    the old 2025 rows.
    """
    origin_cols = _ORIGIN_LOOKUP_COLS + [c for c in _ORIGIN_OPTIONAL_COLS if c in existing.columns]
    dest_cols = _DEST_LOOKUP_COLS + [c for c in _DEST_OPTIONAL_COLS if c in existing.columns]

    origin_lookup = (
        existing.dropna(subset=["origin_latitude", "origin_longitude"])
        .groupby(ORIGIN_CODE_COL)[origin_cols]
        .first()
    )
    dest_lookup = (
        existing.dropna(subset=["destination_latitude", "destination_longitude"])
        .groupby(DEST_CODE_COL)[dest_cols]
        .first()
    )
    return origin_lookup, dest_lookup


def _load_and_clean_2025(raw_path) -> tuple[pd.DataFrame, int, int]:
    """Load the complete 2025 export and apply the 2025-stream cleaning
    convention: coerce dtypes, drop non-physical services (matched against
    the raw names, case-sensitively; see the module docstring), then clean
    and split facility names the same way s01 does for the historical
    stream (case-insensitively, at this later stage).

    Returns the cleaned frame plus the records and interconsultations
    dropped as non-physical.
    """
    df = utils.read_table(raw_path)
    if "UP_DES.1" in df.columns:
        df = df.rename(columns={"UP_DES.1": DESTINATION_COL})

    df[ORIGIN_CODE_COL] = pd.to_numeric(df[ORIGIN_CODE_COL], errors="coerce").astype("Int64")
    df[DEST_CODE_COL] = pd.to_numeric(df[DEST_CODE_COL], errors="coerce").astype("Int64")
    df["GRUP_EDAT"] = pd.to_numeric(df["GRUP_EDAT"], errors="coerce").astype("Int64")
    df["N"] = pd.to_numeric(df["N"], errors="coerce").astype("Int64")
    df[config.YEAR_COL] = df[config.YEAR_COL].astype(int)

    before_records = len(df)
    before_n = int(df["N"].sum())

    # 2025-stream convention: raw names, case-sensitive, before cleaning.
    nonphysical = utils.is_nonphysical(df[ORIGIN_COL], case_sensitive=True) | utils.is_nonphysical(
        df[DESTINATION_COL], case_sensitive=True
    )
    df = df[~nonphysical].copy()
    dropped_records = before_records - len(df)
    dropped_n = before_n - int(df["N"].sum())

    df["ORIGIN_NAME_ORIGINAL"] = df[ORIGIN_COL].astype(str)
    df["DESTINATION_NAME_ORIGINAL"] = df[DESTINATION_COL].astype(str)

    df["ORIGIN_NAME_CLEAN"] = utils.clean_facility_names(df[ORIGIN_COL], case_sensitive=False)
    df["DESTINATION_NAME_CLEAN"] = utils.clean_facility_names(df[DESTINATION_COL], case_sensitive=False)

    df["ORIGIN_LOCATIONS"] = utils.split_multi_location(df["ORIGIN_NAME_CLEAN"])
    df["DESTINATION_LOCATIONS"] = utils.split_multi_location(df["DESTINATION_NAME_CLEAN"])
    df["ORIGIN_NAME_1"] = df["ORIGIN_LOCATIONS"].apply(lambda locs: locs[0] if locs else None)
    df["ORIGIN_NAME_2"] = df["ORIGIN_LOCATIONS"].apply(lambda locs: locs[1] if len(locs) > 1 else None)
    df["DESTINATION_NAME_1"] = df["DESTINATION_LOCATIONS"].apply(lambda locs: locs[0] if locs else None)
    df["DESTINATION_NAME_2"] = df["DESTINATION_LOCATIONS"].apply(lambda locs: locs[1] if len(locs) > 1 else None)

    return df, dropped_records, dropped_n


def _match_coordinates(
    df_new: pd.DataFrame, origin_lookup: pd.DataFrame, dest_lookup: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Match each 2025 row onto its origin/destination coordinates by code
    lookup (no re-geocoding). Returns the merged frame and a boolean mask
    of rows with both coordinates resolved.
    """
    df = df_new.merge(origin_lookup, left_on=ORIGIN_CODE_COL, right_index=True, how="left", suffixes=("", "_lookup"))
    df = df.merge(dest_lookup, left_on=DEST_CODE_COL, right_index=True, how="left", suffixes=("", "_lookup"))
    matched = df["origin_latitude"].notna() & df["destination_latitude"].notna()
    return df, matched


def _export_unmatched(df: pd.DataFrame, matched: pd.Series, path) -> tuple[int, int]:
    """Write the 2025 rows that failed to resolve a full coordinate match
    to `path`, and return their record and interconsultation counts for the
    Figure 1 funnel.
    """
    unmatched_df = df[~matched].copy()
    cols = [c for c in _UNMATCHED_EXPORT_COLS if c in unmatched_df.columns]
    utils.write_table(unmatched_df[cols], path)
    return len(unmatched_df), int(unmatched_df["N"].sum())


def _normalize_list_cell(x):
    """Normalise one ORIGIN_LOCATIONS/DESTINATION_LOCATIONS cell to a plain
    Python list, or None if missing. See the call site for why this is
    needed before writing to parquet.
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return x
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    return x


def _align_to_schema(df: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    """Reindex `df` onto `target_columns`: columns the target schema has
    that `df` lacks are added as all-missing; columns `df` has that the
    target schema lacks are dropped. Mirrors the legacy integration
    script's column alignment ahead of concatenation.
    """
    df = df.copy()
    for col in target_columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[target_columns]


def main() -> None:
    utils.log("=" * 60)
    utils.log("Step 03 - Integrate the complete 2025 export")
    utils.log("=" * 60)

    mode = config.YEAR_2025_MODE
    utils.log(f"YEAR_2025_MODE = {mode!r}")

    existing_path = utils.resolve_input(EXISTING_FILE)
    existing = utils.read_table(existing_path)
    existing_by_year = existing.groupby(config.YEAR_COL)["N"].agg(["count", "sum"])
    utils.log(f"Loaded {len(existing):,} existing rows from {existing_path}")
    utils.log(f"Existing rows by year:\n{existing_by_year}")

    origin_lookup, dest_lookup = _build_coordinate_lookups(existing)
    utils.log(
        f"Coordinate lookups: {len(origin_lookup):,} origin codes, {len(dest_lookup):,} destination codes"
    )

    if mode == "concatenate":
        df_keep = existing.copy()
        utils.log(
            "YEAR_2025_MODE='concatenate': keeping the existing 2025 rows and appending the "
            "refresh on top. This double counts the partial 2025 period and changes every "
            "reported number derived from this output; see the module docstring."
        )
    elif mode == "refresh_replaces_partial":
        before_records = len(existing)
        before_n = int(existing["N"].sum())
        df_keep = existing[existing[config.YEAR_COL] != 2025].copy()
        utils.log(
            f"Dropped {before_records - len(df_keep):,} existing 2025 rows "
            f"({before_n - int(df_keep['N'].sum()):,} interconsultations); "
            f"{len(df_keep):,} rows retained from {sorted(set(config.YEARS) - {2025})}."
        )
    else:
        raise ValueError(f"Unknown config.YEAR_2025_MODE: {mode!r}")

    raw_path = config.data_path(RAW_2025_FILE)
    df_new, dropped_nonphys_records, dropped_nonphys_n = _load_and_clean_2025(raw_path)
    utils.log(
        f"2025 refresh: dropped {dropped_nonphys_records:,} records "
        f"({dropped_nonphys_n:,} interconsultations) as non-physical services; "
        f"{len(df_new):,} rows remain."
    )
    utils.record_number(
        "s03_2025_dropped_nonphysical_records", dropped_nonphys_records, "records", "s03_integrate_2025"
    )
    utils.record_number(
        "s03_2025_dropped_nonphysical_interconsultations",
        dropped_nonphys_n,
        "interconsultations",
        "s03_integrate_2025",
    )

    df_new, matched = _match_coordinates(df_new, origin_lookup, dest_lookup)

    unmatched_path = config.output_path(UNMATCHED_FILE)
    unmatched_records, unmatched_n = _export_unmatched(df_new, matched, unmatched_path)
    utils.log(
        f"Exported {unmatched_records:,} rows without a full coordinate match "
        f"({unmatched_n:,} interconsultations) to {unmatched_path}"
    )
    utils.record_number(
        "s03_dropped_no_coordinate_match_records", unmatched_records, "records", "s03_integrate_2025"
    )
    utils.record_number(
        "s03_dropped_no_coordinate_match_interconsultations",
        unmatched_n,
        "interconsultations",
        "s03_integrate_2025",
    )

    df_new_matched = df_new[matched].copy()
    df_new_matched = _align_to_schema(df_new_matched, list(df_keep.columns))

    # Normalise GRUP_EDAT to a nullable integer on both sides before
    # concatenating (see the module docstring's data notes).
    df_keep["GRUP_EDAT"] = df_keep["GRUP_EDAT"].astype("Int64")
    df_new_matched["GRUP_EDAT"] = df_new_matched["GRUP_EDAT"].astype("Int64")

    df_combined = pd.concat([df_keep, df_new_matched], ignore_index=True)

    # ORIGIN_LOCATIONS/DESTINATION_LOCATIONS hold list-valued cells (from
    # split_multi_location); the existing dataset stores them as numpy
    # arrays (parquet's native list type), while the freshly matched 2025
    # rows produce plain Python lists. Normalise both to plain lists (and
    # missing values to None) so the column is a single, homogeneous type
    # before writing: a stray numpy.ndarray/list mix in the same object
    # column is what pyarrow rejects as "cannot mix list and non-list,
    # non-null values".
    for col in ("ORIGIN_LOCATIONS", "DESTINATION_LOCATIONS"):
        if col in df_combined.columns:
            df_combined[col] = df_combined[col].apply(_normalize_list_cell)

    df_combined[ORIGIN_CODE_COL] = df_combined[ORIGIN_CODE_COL].astype(int)
    df_combined[DEST_CODE_COL] = df_combined[DEST_CODE_COL].astype(int)
    df_combined["N"] = df_combined["N"].astype(int)

    total_n = int(df_combined["N"].sum())
    by_year_final = df_combined.groupby(config.YEAR_COL)["N"].agg(["count", "sum"])
    utils.log(f"Integrated dataset: {len(df_combined):,} rows, {total_n:,} interconsultations")
    utils.log(f"By year:\n{by_year_final}")

    utils.record_number("s03_output_rows", len(df_combined), "rows", "s03_integrate_2025")
    utils.record_number("s03_output_interconsultations", total_n, "interconsultations", "s03_integrate_2025")

    out_path = config.output_path(EXISTING_FILE)
    utils.write_table(df_combined, out_path)
    utils.log(f"Saved: {out_path}")
    utils.flush_numbers("s03_integrate_2025")


if __name__ == "__main__":
    main()
