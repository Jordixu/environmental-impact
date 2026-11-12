"""Step 06: Emissions calculation.

Computes greenhouse gas emissions (CO2, CH4, N2O and CO2-equivalent) for
every routed interconsultation, using year-specific MITECO emission factors
weighted by the national passenger-car fleet's diesel/gasoline split (DGT
fleet composition; see config.py for sources and shares).

This step turns the routed distance dataset into the analysis dataset that
the rest of the pipeline, and the manuscript's headline numbers, are built
from.

Input:
  routed_ors.parquet (resolved via utils.resolve_input, so
  this step runs standalone against the frozen intermediate, or against a
  freshly routed file if s04/s05 have just produced one).

Output:
  OUTPUT_DIR/analysis_dataset.parquet - one row per origin/destination/
    year/UP_DES/UP_DES_1/SEXE/GRUP_EDAT cell, with distance, time and
    emission columns, plus the health-region assignment.
  OUTPUT_DIR/emissions_summary.xlsx - descriptive statistics per route,
    per interconsultation, and per health region (both the 10-region and
    the 8-region Barcelona-collapsed scheme).

Methodology:
  - Fleet composition (turismos): config.DIESEL_SHARE / GASOLINE_SHARE /
    OTHER_SHARE, derived from DGT 2025 table 5.2. The "other" share,
    dominated by battery-electric vehicles, carries a zero tailpipe
    emission factor.
  - Emission factors are year-specific (config.FACTORS), sourced from
    MITECO.
  - CO2-equivalent uses the published, three-decimal config.CO2EQ_BY_FUEL
    table directly, not recomputed from FACTORS and the GWP values, because
    that table is what the published results were actually produced from,
    not the underlying arithmetic. The agreement between the two is checked
    below, but the arithmetic is never swapped in for the real figures.
  - Round-trip distances (km_roundtrip) are used throughout.
  - Per-interconsultation statistics are weighted by N (the number of
    interconsultations represented by each row) and computed by expanding
    with np.repeat instead of materializing an exploded DataFrame.
  - Round trips of 200 m or less, and age groups above GRUP_EDAT 21 (110
    years or more), are excluded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from pipeline import utils

# -----------------------------------------------------------------------------
# Emission factor helpers
# -----------------------------------------------------------------------------
# Fleet-weighted factors are a diesel/gasoline blend of config.FACTORS (for
# CO2, CH4, N2O individually) and of config.CO2EQ_BY_FUEL (for CO2eq, kept
# as its own published table rather than recomputed from FACTORS and the
# GWP values below; see the module docstring).


def _weighted_factor(year: int, gas: str) -> float:
    """Fleet-weighted emission factor (kg/km) for `gas` ("CO2", "CH4" or
    "N2O") in `year`. Falls back to config.FACTOR_FALLBACK_YEAR for a year
    absent from config.FACTORS.
    """
    diesel = config.FACTORS["diesel"].get(year, config.FACTORS["diesel"][config.FACTOR_FALLBACK_YEAR])
    gasoline = config.FACTORS["gasoline"].get(year, config.FACTORS["gasoline"][config.FACTOR_FALLBACK_YEAR])
    return config.DIESEL_SHARE * diesel[gas] + config.GASOLINE_SHARE * gasoline[gas]


def _weighted_co2eq(year: int) -> float:
    """Fleet-weighted CO2-equivalent factor (kg/km) for `year`, from the
    published config.CO2EQ_BY_FUEL table. Falls back to
    config.FACTOR_FALLBACK_YEAR for a year absent from that table.
    """
    diesel = config.CO2EQ_BY_FUEL["diesel"].get(year, config.CO2EQ_BY_FUEL["diesel"][config.FACTOR_FALLBACK_YEAR])
    gasoline = config.CO2EQ_BY_FUEL["gasoline"].get(year, config.CO2EQ_BY_FUEL["gasoline"][config.FACTOR_FALLBACK_YEAR])
    return config.DIESEL_SHARE * diesel + config.GASOLINE_SHARE * gasoline


def _assert_co2eq_provenance(tolerance: float = 0.001) -> None:
    """Assert that the published config.CO2EQ_BY_FUEL table agrees with
    config.FACTORS combined with the GWP values (config.GWP_CO2/CH4/N2O),
    to within `tolerance` (relative), for every fuel-year pair.

    This documents the provenance of CO2EQ_BY_FUEL: it is a separately
    published, rounded table, not a value this pipeline derives. The
    arithmetic below exists only to check that agreement holds. The actual
    CO2eq figures reported by this step always come from CO2EQ_BY_FUEL
    itself (see _weighted_co2eq), never from this recomputation.
    """
    for fuel, by_year in config.FACTORS.items():
        for year, factors in by_year.items():
            recomputed = (
                factors["CO2"] * config.GWP_CO2
                + factors["CH4"] * config.GWP_CH4
                + factors["N2O"] * config.GWP_N2O
            )
            published = config.CO2EQ_BY_FUEL[fuel][year]
            rel_diff = abs(recomputed - published) / published
            if rel_diff > tolerance:
                raise AssertionError(
                    f"CO2EQ_BY_FUEL[{fuel!r}][{year}] = {published} disagrees with "
                    f"FACTORS + GWP ({recomputed}) by {rel_diff:.4%}, "
                    f"exceeding the {tolerance:.2%} tolerance"
                )
    utils.log("Verified: CO2EQ_BY_FUEL agrees with FACTORS + GWP within 0.1% for every fuel-year pair.")


# -----------------------------------------------------------------------------
# Loading and per-row computation
# -----------------------------------------------------------------------------

# Column names before config.COLUMN_RENAMES is applied. These stay stable
# per the contract; they are selected into the final dataset below.
_FINAL_COLUMNS_BEFORE_RENAME = [
    "year", "OC_UP_ORI", "DER_UP_SCS_DES", "UP_DES", "UP_DES_1", "SEXE", "GRUP_EDAT", "N",
    "km_roundtrip", "duration_roundtrip_min", "total_km", "temps_total",
    "co2_factor", "ch4_factor", "n2o_factor", "co2eq_factor",
    "emissio_co2_kg", "emissio_ch4_kg", "emissio_n2o_kg", "emissio_co2eq_kg",
    "region_8", "region_10",
    "origin_abs_region_name",
    "km_per_person", "emissio_co2_kg_per_person", "emissio_ch4_kg_per_person",
    "emissio_n2o_kg_per_person", "emissio_co2eq_kg_per_person", "temps_total_per_person",
]

# Facility locations, labels and routed geometry. The mapping figure draws facility
# positions, route lines and a per-interconsultation emissions scale from these, so the
# analysis dataset must carry them through.
#
# These columns stay out of the repository but not out of the pipeline: the analysis
# dataset is written under OUTPUT_DIR, which is excluded by .gitignore and is never
# committed or distributed. The published aggregate extract is built by the tables step,
# which applies small-count suppression and omits coordinates entirely.
_GEOMETRY_COLUMNS = [
    "origin_location_name", "destination_location_name",
    "origin_latitude", "origin_longitude",
    "destination_latitude", "destination_longitude",
    "route_geometry_outbound", "route_geometry_return",
]


def _load_routed_dataset() -> pd.DataFrame:
    """Read the routed dataset (origin/destination distances and
    durations), resolving it against OUTPUT_DIR first and falling back to
    the frozen copy in DATA_DIR.
    """
    path = utils.resolve_input("routed_ors.parquet")
    df = utils.read_table(path)
    utils.log(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from {path}")
    return df


def _compute_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Derive year and emission factors, compute per-row emission and time
    totals (steps b and c of the contract), ahead of filtering.

    Distance and duration are read from km_roundtrip / duration_roundtrip_min.
    Emission values multiply distance by the year-and-fleet-weighted factor
    (per interconsultation) and by N (the number of interconsultations
    represented by the row) for the totals.
    """
    df = df.copy()
    dist_col, dur_col = "km_roundtrip", "duration_roundtrip_min"

    df[dist_col] = pd.to_numeric(df[dist_col], errors="coerce")
    df[dur_col] = pd.to_numeric(df[dur_col], errors="coerce")
    df["N"] = pd.to_numeric(df["N"], errors="coerce").fillna(1)

    # b) derive year, map year-and-fleet-weighted factors onto each row.
    df["year"] = df[config.YEAR_COL]
    df["co2_factor"] = df["year"].map(lambda y: _weighted_factor(y, "CO2"))
    df["ch4_factor"] = df["year"].map(lambda y: _weighted_factor(y, "CH4"))
    df["n2o_factor"] = df["year"].map(lambda y: _weighted_factor(y, "N2O"))
    df["co2eq_factor"] = df["year"].map(_weighted_co2eq)

    # c) total emissions per row (distance * factor * N) and the
    # per-interconsultation single-trip rates (distance * factor, no N).
    df["emissio_co2_kg"] = df[dist_col] * df["co2_factor"] * df["N"]
    df["emissio_ch4_kg"] = df[dist_col] * df["ch4_factor"] * df["N"]
    df["emissio_n2o_kg"] = df[dist_col] * df["n2o_factor"] * df["N"]
    df["emissio_co2eq_kg"] = df[dist_col] * df["co2eq_factor"] * df["N"]
    df["total_km"] = df[dist_col] * df["N"]
    df["temps_total"] = df[dur_col] * df["N"]

    df["km_per_person"] = df[dist_col]
    df["emissio_co2_kg_per_person"] = df[dist_col] * df["co2_factor"]
    df["emissio_ch4_kg_per_person"] = df[dist_col] * df["ch4_factor"]
    df["emissio_n2o_kg_per_person"] = df[dist_col] * df["n2o_factor"]
    df["emissio_co2eq_kg_per_person"] = df[dist_col] * df["co2eq_factor"]
    df["temps_total_per_person"] = df[dur_col]

    return df


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """d) apply the Figure 1 funnel filters, in order: round trips longer
    than MIN_ROUNDTRIP_KM, then GRUP_EDAT <= MAX_AGE_GROUP.

    A NaN round-trip distance fails the "> MIN_ROUNDTRIP_KM" comparison
    directly, so rows with no routed distance are dropped by the same
    filter without a separate notna() step.
    """
    before = len(df)
    df = df[df["km_roundtrip"] > config.MIN_ROUNDTRIP_KM].copy()
    utils.log(f"After round-trip > {config.MIN_ROUNDTRIP_KM} km filter: {len(df):,} rows (dropped {before - len(df):,})")

    before = len(df)
    df = df[df["GRUP_EDAT"] <= config.MAX_AGE_GROUP].copy()
    utils.log(f"After GRUP_EDAT <= {config.MAX_AGE_GROUP} filter: {len(df):,} rows (dropped {before - len(df):,})")
    return df


def _assign_regions(df: pd.DataFrame) -> pd.DataFrame:
    """e) assign health regions by spatial join, then derive the 8-region
    (Barcelona-collapsed) and 10-region schemes.

    Rows whose origin point does not fall in any mapped health region, or
    falls in an excluded pseudo-region (config.REGION_EXCLUDE, e.g. "No
    consta"), are dropped: the analysis dataset must carry a real health
    region for every row.
    """
    df = utils.assign_region(df, "origin_latitude", "origin_longitude", "origin_abs_region_name")

    before = len(df)
    valid = df["origin_abs_region_name"].notna() & ~df["origin_abs_region_name"].isin(config.REGION_EXCLUDE)
    df = df[valid].copy()
    dropped = before - len(df)
    if dropped:
        utils.log(f"Dropped {dropped:,} rows with no mapped health region (null or excluded).")

    df["region_10"] = df["origin_abs_region_name"]
    df["region_8"] = df["origin_abs_region_name"].apply(utils.to_8_regions)
    return df


# -----------------------------------------------------------------------------
# Summary workbook (emissions_summary.xlsx)
# -----------------------------------------------------------------------------

_REGION_VARIABLES = {
    "km": "km_per_person",
    "co2": "emissio_co2_kg_per_person",
    "ch4": "emissio_ch4_kg_per_person",
    "n2o": "emissio_n2o_kg_per_person",
    "co2eq": "emissio_co2eq_kg_per_person",
    "time": "temps_total_per_person",
}


def _unweighted_stats(values: np.ndarray) -> dict:
    """Plain (unweighted) descriptive statistics for an array that already
    holds one value per unit (e.g. one total per route): count, mean,
    median, min, max, p75, p90, dev (sample standard deviation).
    """
    s = values[~np.isnan(values)]
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "median": float(np.median(s)),
        "min": float(s.min()),
        "max": float(s.max()),
        "p75": float(np.percentile(s, 75)),
        "p90": float(np.percentile(s, 90)),
        "dev": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
    }


def _per_route_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Per-route descriptive statistics: routes are grouped by the
    destination facility codes (UP_DES, UP_DES_1), the per-interconsultation
    rates are averaged per route, and multiplied by the summed N to get a
    route total; the sheet then describes the distribution of those totals
    across routes.
    """
    agg = df.groupby(["UP_DES", "UP_DES_1"]).agg(
        km_per_person=("km_per_person", "mean"),
        emissio_co2_kg_per_person=("emissio_co2_kg_per_person", "mean"),
        emissio_ch4_kg_per_person=("emissio_ch4_kg_per_person", "mean"),
        emissio_n2o_kg_per_person=("emissio_n2o_kg_per_person", "mean"),
        emissio_co2eq_kg_per_person=("emissio_co2eq_kg_per_person", "mean"),
        temps_total_per_person=("temps_total_per_person", "mean"),
        N=("N", "sum"),
    )
    agg["km_tot"] = agg["km_per_person"] * agg["N"]
    agg["co2_tot"] = agg["emissio_co2_kg_per_person"] * agg["N"]
    agg["ch4_tot"] = agg["emissio_ch4_kg_per_person"] * agg["N"]
    agg["n2o_tot"] = agg["emissio_n2o_kg_per_person"] * agg["N"]
    agg["co2eq_tot"] = agg["emissio_co2eq_kg_per_person"] * agg["N"]
    agg["time_tot"] = agg["temps_total_per_person"] * agg["N"]

    rows = []
    for col in ["km_tot", "co2_tot", "ch4_tot", "n2o_tot", "co2eq_tot", "time_tot"]:
        rows.append({"metrica": col, **_unweighted_stats(agg[col].dropna().to_numpy(dtype=float))})
    return pd.DataFrame(rows)


def _per_interconsultation_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Per-interconsultation descriptive statistics, weighted by N, for
    each of the six emission/distance/time variables. The "metrica" label
    uses the renamed (interconsultation) column names via
    config.COLUMN_RENAMES.
    """
    weights = df["N"].to_numpy(dtype=float)
    cols = [
        "km_per_person", "emissio_co2_kg_per_person", "emissio_ch4_kg_per_person",
        "emissio_n2o_kg_per_person", "emissio_co2eq_kg_per_person", "temps_total_per_person",
    ]
    rows = []
    for col in cols:
        stats = utils.weighted_stats(df[col].to_numpy(dtype=float), weights)
        if stats is not None:
            rows.append({"metrica": config.COLUMN_RENAMES[col], **stats})
    return pd.DataFrame(rows)


def _region_stats(df: pd.DataFrame, region_col: str) -> dict[str, pd.DataFrame]:
    """Weighted per-region statistics for each variable in
    _REGION_VARIABLES, one DataFrame per variable with a "region" column,
    sorted by region name.
    """
    out = {}
    regions = df[region_col].unique()
    for var_name, col_name in _REGION_VARIABLES.items():
        rows = []
        for region in regions:
            mask = df[region_col] == region
            v = df.loc[mask, col_name].to_numpy(dtype=float)
            w = df.loc[mask, "N"].to_numpy(dtype=float)
            stats = utils.weighted_stats(v, w)
            if stats is not None:
                rows.append({"region": region, **stats})
        out[var_name] = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    return out


def _region_summary_sheet(region_stats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine the per-variable region statistics into one summary sheet,
    one row per region, with count/sum/mean/median for each variable.
    """
    all_regions = sorted(set().union(*(set(df_["region"]) for df_ in region_stats.values())))
    rows = []
    for region in all_regions:
        row = {"region": region}
        for var, df_ in region_stats.items():
            match = df_[df_["region"] == region]
            if len(match):
                row[f"{var}_count"] = match["count"].iloc[0]
                row[f"{var}_sum"] = match["sum"].iloc[0]
                row[f"{var}_mean"] = match["mean"].iloc[0]
                row[f"{var}_median"] = match["median"].iloc[0]
            else:
                row[f"{var}_count"] = 0
                row[f"{var}_sum"] = 0
                row[f"{var}_mean"] = float("nan")
                row[f"{var}_median"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _write_summary_workbook(df: pd.DataFrame, path) -> None:
    """Write emissions_summary.xlsx with the 16-sheet layout of the legacy
    resum_emissions.xlsx: per_route, per_interconsultation (the legacy
    per_person, renamed), six region_10_* sheets, six region_8_* sheets,
    and the two combined region_summary sheets. Sheet names are ASCII.
    """
    region_stats_10 = _region_stats(df, "region_10")
    region_stats_8 = _region_stats(df, "region_8")

    with pd.ExcelWriter(path) as writer:
        _per_route_sheet(df).to_excel(writer, sheet_name="per_route", index=False)
        _per_interconsultation_sheet(df).to_excel(writer, sheet_name="per_interconsultation", index=False)

        for var, region_df in region_stats_10.items():
            region_df.to_excel(writer, sheet_name=f"region_10_{var}"[:31], index=False)
        for var, region_df in region_stats_8.items():
            region_df.to_excel(writer, sheet_name=f"region_8_{var}"[:31], index=False)

        _region_summary_sheet(region_stats_10).to_excel(writer, sheet_name="region_summary_10", index=False)
        _region_summary_sheet(region_stats_8).to_excel(writer, sheet_name="region_summary_8", index=False)

    utils.log(f"Saved: {path}")


# -----------------------------------------------------------------------------
# Headline numbers
# -----------------------------------------------------------------------------


def _record_headline_numbers(df: pd.DataFrame) -> None:
    """Record every headline number this step produces via
    utils.record_number, for later collection into manuscript_numbers.csv.
    """
    n_sum = int(df["N"].sum())
    n_rows = len(df)
    n_routes = df.groupby(config.ROUTE_KEY).ngroups
    n_route_years = df.groupby(config.ROUTE_KEY + ["year"]).ngroups
    total_km = float(df["total_km"].sum())
    total_co2 = float(df["emissio_co2_kg"].sum())
    total_ch4 = float(df["emissio_ch4_kg"].sum())
    total_n2o = float(df["emissio_n2o_kg"].sum())
    total_co2eq = float(df["emissio_co2eq_kg"].sum())

    region_type = df["origin_abs_region_name"].apply(utils.classify_region_type)
    n_by_region_type = df.groupby(region_type)["N"].sum()
    capital_n = int(n_by_region_type.get("Capital", 0))
    noncapital_n = int(n_by_region_type.get("Non-capital", 0))

    utils.record_number("analysis_dataset_rows", n_rows, "rows", "s06_emissions")
    utils.record_number("total_interconsultations", n_sum, "interconsultations", "s06_emissions")
    utils.record_number("unique_routes", n_routes, "routes", "s06_emissions")
    utils.record_number("route_year_groups", n_route_years, "route-year groups", "s06_emissions")
    utils.record_number("total_roundtrip_km", total_km, "km", "s06_emissions")
    utils.record_number("total_co2_kg", total_co2, "kg", "s06_emissions")
    utils.record_number("total_ch4_kg", total_ch4, "kg", "s06_emissions")
    utils.record_number("total_n2o_kg", total_n2o, "kg", "s06_emissions")
    utils.record_number("total_co2eq_kg", total_co2eq, "kg", "s06_emissions")
    utils.record_number("capital_interconsultations", capital_n, "interconsultations", "s06_emissions")
    utils.record_number("noncapital_interconsultations", noncapital_n, "interconsultations", "s06_emissions")

    utils.log(f"Rows: {n_rows:,} | N sum: {n_sum:,} | unique routes: {n_routes:,} | route-year groups: {n_route_years:,}")
    utils.log(f"Total km: {total_km:,.4f} | CO2: {total_co2:,.6f} kg | N2O: {total_n2o:,.6f} kg")
    utils.log(f"Capital N: {capital_n:,} | Non-capital N: {noncapital_n:,}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    utils.log("=" * 60)
    utils.log("Step 06 - Emissions calculation")
    utils.log("=" * 60)

    _assert_co2eq_provenance()

    raw = _load_routed_dataset()
    df = _compute_rows(raw)
    df = _apply_filters(df)
    df = _assign_regions(df)

    _record_headline_numbers(df)

    utils.log("Writing emissions summary workbook...")
    _write_summary_workbook(df, config.output_path("emissions_summary.xlsx"))

    keep = _FINAL_COLUMNS_BEFORE_RENAME + [c for c in _GEOMETRY_COLUMNS if c in df.columns]
    missing = [c for c in _GEOMETRY_COLUMNS if c not in df.columns]
    if missing:
        utils.log(f"WARNING: routed dataset lacks {len(missing)} geometry column(s): "
                  f"{', '.join(missing)}. The mapping figure will not be reproducible.")
    final = df[keep].rename(columns=config.COLUMN_RENAMES)
    out_path = config.output_path("analysis_dataset.parquet")
    utils.write_table(final, out_path)
    utils.log(f"Saved: {out_path}")

    utils.log("Done.")
    utils.flush_numbers("s06_emissions")


if __name__ == "__main__":
    main()
