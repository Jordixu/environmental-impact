"""Step 10 - Manuscript tables, published data extract and number inventory.

This is the final step of the pipeline. It produces every table the
manuscript reports (Table 1-5, S1-S4 Table), the published route-year data
extract (S1_Data.xlsx) and the machine-readable inventory of every number
the manuscript cites (manuscript_numbers.csv).

Inputs (all resolved with utils.resolve_input, so this step runs standalone
against the frozen intermediates in DATA_DIR, or against freshly generated
ones in OUTPUT_DIR):
  analysis_dataset.parquet                   (s06)
  routed_osrm.parquet     (OSRM distances, for S1_Data)
  statistics_capital_vs_noncapital.xlsx       (s07; S2 Table, S4 Table)
  sensitivity_analysis.xlsx                   (s08; Table 4, Table 5)

The sensitivity workbook has no legacy counterpart, and at the time this was
written s08 had not been built yet in this checkout. If it is absent, Table
4 and Table 5 are skipped with a clear warning instead of the step crashing.

Outputs (all under OUTPUT_DIR/tables, plus manuscript_numbers.csv under
OUTPUT_DIR):
  Table1.xlsx .. Table5.xlsx, S1_Table.xlsx .. S4_Table.xlsx, S1_Data.xlsx,
  ../manuscript_numbers.csv

Units are interconsultations throughout: every table header, row label and
column name says "interconsultation" or "per interconsultation". N is the
number of interconsultations a row represents.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
from pipeline import utils

SOURCE_STEP = "s10_tables"

# -----------------------------------------------------------------------------
# Table 1: per-interconsultation descriptive statistics by region type
# -----------------------------------------------------------------------------
# (metric key, column in the analysis dataset, scale factor, display label,
# unit label used in manuscript_numbers)
TABLE1_METRICS = [
    ("distance", "km_per_interconsultation", 1.0, "Distance per interconsultation (km)", "km"),
    ("co2", "co2_kg_per_interconsultation", 1.0, "CO₂ per interconsultation (kg)", "kg"),
    ("n2o", "n2o_kg_per_interconsultation", 1000.0, "N₂O per interconsultation (g)", "g"),
    ("time", "minutes_per_interconsultation", 1.0, "Total time per interconsultation (min)", "min"),
]
TABLE1_REGIONS = [("overall", "Overall", None), ("capital", "Capital", "Capital"),
                   ("noncapital", "Non-capital", "Non-capital")]

# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def load_analysis_dataset() -> pd.DataFrame:
    """Load the analysis dataset produced by s06_emissions and attach the
    region_type classification (Capital / Non-capital) used throughout.
    """
    path = utils.resolve_input("analysis_dataset.parquet")
    utils.log(f"Loading analysis dataset: {path}")
    df = utils.read_table(path)
    df["region_type"] = df["origin_health_region"].apply(utils.classify_region_type)
    return df


def load_osrm_dataset() -> pd.DataFrame | None:
    """Load the OSRM-routed distances/durations needed for S1_Data, at the
    same cell grain as the analysis dataset (OC_UP_ORI, DER_UP_SCS_DES,
    UP_DES, UP_DES_1, SEXE, GRUP_EDAT, year). Returns None with a warning if
    the file cannot be found.
    """
    path = utils.resolve_input("routed_osrm.parquet")
    if not path.exists():
        utils.log(f"WARNING: OSRM distances file not found ({path}); S1_Data will omit the OSRM columns.")
        return None
    utils.log(f"Loading OSRM distances: {path}")
    df = utils.read_table(path)
    df = df.rename(columns={config.YEAR_COL: "year"})
    return df


def load_statistics_workbook() -> dict[str, pd.DataFrame] | None:
    """Load the Capital vs Non-capital statistics workbook written by s07
    (Summary and Route_level sheets, needed for S2 Table and S4 Table).
    Returns None with a warning if the file cannot be found.
    """
    path = utils.resolve_input("statistics_capital_vs_noncapital.xlsx")
    if not path.exists():
        utils.log(f"WARNING: {path.name} not found; S2 Table and S4 Table will be skipped.")
        return None
    utils.log(f"Loading statistics workbook: {path}")
    return pd.read_excel(path, sheet_name=None)


def load_sensitivity_workbook() -> dict[str, pd.DataFrame] | None:
    """Load the sensitivity analysis workbook written by s08 (substitution
    rate and transport mode scenarios, needed for Table 4 and Table 5).
    Returns None with a warning if the file cannot be found: s08_sensitivity
    has no legacy counterpart and may not have been run yet.
    """
    path = utils.resolve_input("sensitivity_analysis.xlsx")
    if not path.exists():
        utils.log(f"WARNING: {path.name} not found; Table 4 and Table 5 will be skipped.")
        return None
    utils.log(f"Loading sensitivity workbook: {path}")
    return pd.read_excel(path, sheet_name=None)


def _find_sheet(sheets: dict[str, pd.DataFrame], candidates: list[str]) -> pd.DataFrame | None:
    """Return the first sheet in `sheets` whose name matches (case
    insensitive) one of `candidates`, or None if no candidate matches.
    """
    lower_map = {name.lower(): name for name in sheets}
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return sheets[lower_map[key]]
    return None


# -----------------------------------------------------------------------------
# Table 1
# -----------------------------------------------------------------------------


def build_table1(df: pd.DataFrame) -> pd.DataFrame:
    """Per-interconsultation descriptive statistics (mean, median, min, max,
    SD, P75, P90), weighted by N, stratified by region type.
    """
    utils.log("Building Table 1 (descriptive statistics by region type)")
    rows = []
    for metric_key, col, scale, label, unit in TABLE1_METRICS:
        for region_key, region_label, region_filter in TABLE1_REGIONS:
            sub = df if region_filter is None else df.loc[df["region_type"] == region_filter]
            values = sub[col].to_numpy(dtype=float) * scale
            weights = sub["N"].to_numpy(dtype=float)
            stats = utils.weighted_stats(values, weights)
            if stats is None:
                continue
            rows.append({
                "Metric": label,
                "Region": region_label,
                "Mean": stats["mean"],
                "Median": stats["median"],
                "Min": stats["min"],
                "Max": stats["max"],
                "Std. Dev": stats["std"],
                "P75": stats["p75"],
                "P90": stats["p90"],
            })
            utils.record_number(f"table1_{metric_key}_{region_key}_n", stats["count"], "interconsultations", SOURCE_STEP)
            for stat_key in ("mean", "median", "min", "max", "std", "p75", "p90"):
                utils.record_number(f"table1_{metric_key}_{region_key}_{stat_key}", stats[stat_key], unit, SOURCE_STEP)
    table = pd.DataFrame(rows, columns=["Metric", "Region", "Mean", "Median", "Min", "Max", "Std. Dev", "P75", "P90"])
    # IQR (P75 - P25) for the manuscript's distance anchor; P25 is recorded
    # separately even though the table itself does not carry a P25 column.
    for metric_key, col, scale, label, unit in TABLE1_METRICS:
        values = df[col].to_numpy(dtype=float) * scale
        weights = df["N"].to_numpy(dtype=float)
        stats = utils.weighted_stats(values, weights)
        if stats is not None:
            utils.record_number(f"table1_{metric_key}_overall_p25", stats["p25"], unit, SOURCE_STEP)
    return table


# -----------------------------------------------------------------------------
# Table 2: median per-interconsultation outcomes by year
# -----------------------------------------------------------------------------

TABLE2_METRICS = [
    ("distance", "km_per_interconsultation", 1.0, "Distance (km)", "km"),
    ("co2", "co2_kg_per_interconsultation", 1.0, "CO2 (kg)", "kg"),
    ("n2o", "n2o_kg_per_interconsultation", 1000.0, "N2O (g)", "g"),
    ("duration", "minutes_per_interconsultation", 1.0, "Duration (min)", "min"),
]


def build_table2(df: pd.DataFrame) -> pd.DataFrame:
    """Median distance, CO2, N2O and travel time per interconsultation, by
    study year.
    """
    utils.log("Building Table 2 (median outcomes by year)")
    rows = []
    for year in config.YEARS:
        sub = df.loc[df["year"] == year]
        if sub.empty:
            continue
        row = {"Year": year}
        for metric_key, col, scale, label, unit in TABLE2_METRICS:
            stats = utils.weighted_stats(sub[col].to_numpy(dtype=float) * scale, sub["N"].to_numpy(dtype=float))
            median = stats["median"] if stats is not None else float("nan")
            row[label] = median
            utils.record_number(f"table2_{metric_key}_median_{year}", median, unit, SOURCE_STEP)
        rows.append(row)
    return pd.DataFrame(rows, columns=["Year"] + [label for _, _, _, label, _ in TABLE2_METRICS])


# -----------------------------------------------------------------------------
# Table 3: annual volumes and totals
# -----------------------------------------------------------------------------

TABLE3_METRICS = [
    ("distance", "km_per_interconsultation", 1.0, "Distance (km)", "km"),
    ("co2", "co2_kg_per_interconsultation", 1.0, "CO2 (kg)", "kg"),
    ("n2o", "n2o_kg_per_interconsultation", 1000.0, "N2O (g)", "g"),
]


def build_table3(df: pd.DataFrame) -> pd.DataFrame:
    """Annual interconsultation volumes and modelled environmental outcomes:
    mean, median and total for distance, CO2 and N2O, plus mean/median/total
    for travel time (time total reported in hours, matching the manuscript).
    """
    utils.log("Building Table 3 (annual volumes and totals)")
    year_cols = [y for y in config.YEARS if (df["year"] == y).any()]
    metric_rows: dict[str, dict] = {}

    def set_cell(metric: str, year: int, value) -> None:
        metric_rows.setdefault(metric, {"Metric": metric})
        metric_rows[metric][year] = value

    for year in year_cols:
        sub = df.loc[df["year"] == year]
        weights = sub["N"].to_numpy(dtype=float)
        n_sum = int(sub["N"].sum())
        set_cell("Interconsultations", year, n_sum)
        utils.record_number(f"table3_interconsultations_{year}", n_sum, "interconsultations", SOURCE_STEP)

        for metric_key, col, scale, label, unit in TABLE3_METRICS:
            stats = utils.weighted_stats(sub[col].to_numpy(dtype=float) * scale, weights)
            if stats is None:
                continue
            set_cell(f"{label} Mean", year, stats["mean"])
            set_cell(f"{label} Median", year, stats["median"])
            set_cell(f"{label} Total", year, stats["sum"])
            utils.record_number(f"table3_{metric_key}_mean_{year}", stats["mean"], unit, SOURCE_STEP)
            utils.record_number(f"table3_{metric_key}_median_{year}", stats["median"], unit, SOURCE_STEP)
            utils.record_number(f"table3_{metric_key}_total_{year}", stats["sum"], unit, SOURCE_STEP)

        time_stats = utils.weighted_stats(sub["minutes_per_interconsultation"].to_numpy(dtype=float), weights)
        if time_stats is not None:
            set_cell("Time (min) Mean", year, time_stats["mean"])
            set_cell("Time (min) Median", year, time_stats["median"])
            hours_total = time_stats["sum"] / 60.0
            set_cell("Time (hours) Total", year, hours_total)
            utils.record_number(f"table3_time_mean_{year}", time_stats["mean"], "min", SOURCE_STEP)
            utils.record_number(f"table3_time_median_{year}", time_stats["median"], "min", SOURCE_STEP)
            utils.record_number(f"table3_time_total_{year}", hours_total, "hours", SOURCE_STEP)

    order = ["Interconsultations",
              "Distance (km) Mean", "Distance (km) Median", "Distance (km) Total",
              "CO2 (kg) Mean", "CO2 (kg) Median", "CO2 (kg) Total",
              "N2O (g) Mean", "N2O (g) Median", "N2O (g) Total",
              "Time (min) Mean", "Time (min) Median", "Time (hours) Total"]
    rows = [metric_rows[m] for m in order if m in metric_rows]
    return pd.DataFrame(rows, columns=["Metric"] + year_cols)


# -----------------------------------------------------------------------------
# Table 4: substitution-rate sensitivity (from s08's workbook)
# -----------------------------------------------------------------------------

_SUBSTITUTION_SHEET_CANDIDATES = [
    "Substitution_Rate", "SubstitutionRate", "Substitution_rate_sensitivity", "Table4", "Substitution",
]


def build_table4(sensitivity: dict[str, pd.DataFrame] | None) -> pd.DataFrame | None:
    """Sensitivity of avoided travel/CO2/N2O to the assumed interconsultation
    substitution rate, read verbatim from s08's workbook. Returns None (with
    a warning already logged by the caller) if the workbook or the expected
    sheet is unavailable.
    """
    if sensitivity is None:
        return None
    sheet = _find_sheet(sensitivity, _SUBSTITUTION_SHEET_CANDIDATES)
    if sheet is None:
        utils.log(
            "WARNING: no substitution-rate sheet found in sensitivity_analysis.xlsx "
            f"(looked for: {', '.join(_SUBSTITUTION_SHEET_CANDIDATES)}; "
            f"available sheets: {', '.join(sensitivity.keys())}). Table 4 skipped."
        )
        return None
    utils.log("Building Table 4 (substitution-rate sensitivity)")
    table = sheet.copy()
    for _, row in table.iterrows():
        rate_label = str(row.get("Substitution Rate", row.iloc[0]))
        key = rate_label.split("%")[0].strip().replace(".", "")
        for col in table.columns[1:]:
            utils.record_number(f"table4_{key}pct_{col.lower().replace(' ', '_')}", row[col], col, SOURCE_STEP)
    return table


# -----------------------------------------------------------------------------
# Table 5: transport-mode sensitivity (from s08's workbook)
# -----------------------------------------------------------------------------

_SCENARIO_SHEET_CANDIDATES = ["Scenario_Summary", "Transport_Mode", "Table5", "Scenario"]


def build_table5(sensitivity: dict[str, pd.DataFrame] | None) -> pd.DataFrame | None:
    """Sensitivity of total avoided CO2 to the transport-mode assumption,
    read verbatim from s08's workbook (the published Table 5 keeps the first
    five scenarios; any additional scenario sheets are recorded but not
    added to the published rows). Returns None if unavailable.
    """
    if sensitivity is None:
        return None
    sheet = _find_sheet(sensitivity, _SCENARIO_SHEET_CANDIDATES)
    if sheet is None:
        utils.log(
            "WARNING: no transport-mode scenario sheet found in sensitivity_analysis.xlsx "
            f"(looked for: {', '.join(_SCENARIO_SHEET_CANDIDATES)}; "
            f"available sheets: {', '.join(sensitivity.keys())}). Table 5 skipped."
        )
        return None
    utils.log("Building Table 5 (transport-mode sensitivity)")
    table = sheet.copy()
    scenario_col = table.columns[0]
    for _, row in table.iterrows():
        key = str(row[scenario_col]).split(".")[0].strip().replace(" ", "_").lower()
        for col in table.columns[1:]:
            utils.record_number(f"table5_{key}_{col.lower().replace(' ', '_').replace('%', 'pct')}", row[col], col, SOURCE_STEP)
    return table


# -----------------------------------------------------------------------------
# S1 Table: emission factors by fuel and year, fleet-weighted
# -----------------------------------------------------------------------------


def _fleet_weighted_factor(year: int, gas: str) -> float:
    """Fleet-weighted emission factor (kg/km) for `gas` in `year`, from
    config.FACTORS blended by config.DIESEL_SHARE / GASOLINE_SHARE (the
    same formula s06_emissions applies to the analysis dataset; recomputed
    here from the published config constants alone so S1 Table does not
    depend on the analysis dataset being available).
    """
    fallback = config.FACTOR_FALLBACK_YEAR
    diesel = config.FACTORS["diesel"].get(year, config.FACTORS["diesel"][fallback])
    gasoline = config.FACTORS["gasoline"].get(year, config.FACTORS["gasoline"][fallback])
    return config.DIESEL_SHARE * diesel[gas] + config.GASOLINE_SHARE * gasoline[gas]


def _fleet_weighted_co2eq(year: int) -> float:
    """Fleet-weighted CO2-equivalent factor (kg/km) for `year`, from the
    published config.CO2EQ_BY_FUEL table blended the same way.
    """
    fallback = config.FACTOR_FALLBACK_YEAR
    diesel = config.CO2EQ_BY_FUEL["diesel"].get(year, config.CO2EQ_BY_FUEL["diesel"][fallback])
    gasoline = config.CO2EQ_BY_FUEL["gasoline"].get(year, config.CO2EQ_BY_FUEL["gasoline"][fallback])
    return config.DIESEL_SHARE * diesel + config.GASOLINE_SHARE * gasoline


def build_s1_table() -> pd.DataFrame:
    """Private-vehicle emission factors by fuel type and year (MITECO,
    2025), with fleet-weighted values (fleet composition from DGT, 2025).
    """
    utils.log("Building S1 Table (emission factors)")
    rows = []
    for fuel, share, label in (
        ("diesel", config.DIESEL_SHARE, f"Diesel ({config.DIESEL_SHARE * 100:.1f}%)"),
        ("gasoline", config.GASOLINE_SHARE, f"Gasoline ({config.GASOLINE_SHARE * 100:.1f}%)"),
    ):
        for year in config.YEARS:
            factors = config.FACTORS[fuel].get(year, config.FACTORS[fuel][config.FACTOR_FALLBACK_YEAR])
            co2eq = config.CO2EQ_BY_FUEL[fuel].get(year, config.CO2EQ_BY_FUEL[fuel][config.FACTOR_FALLBACK_YEAR])
            rows.append({
                "Fuel type (share)": label,
                "Year": year,
                "CO₂ (kg/km)": factors["CO2"],
                "CH₄ (g/km)": factors["CH4"] * 1000,
                "N₂O (g/km)": factors["N2O"] * 1000,
                "CO₂eq (kg/km)": co2eq,
            })
            prefix = f"s1_{fuel}"
            utils.record_number(f"{prefix}_co2_{year}", factors["CO2"], "kg/km", SOURCE_STEP)
            utils.record_number(f"{prefix}_ch4_{year}", factors["CH4"] * 1000, "g/km", SOURCE_STEP)
            utils.record_number(f"{prefix}_n2o_{year}", factors["N2O"] * 1000, "g/km", SOURCE_STEP)
            utils.record_number(f"{prefix}_co2eq_{year}", co2eq, "kg/km", SOURCE_STEP)

    for year in config.YEARS:
        co2 = _fleet_weighted_factor(year, "CO2")
        ch4 = _fleet_weighted_factor(year, "CH4") * 1000
        n2o = _fleet_weighted_factor(year, "N2O") * 1000
        co2eq = _fleet_weighted_co2eq(year)
        rows.append({
            "Fuel type (share)": "Fleet-weighted",
            "Year": year,
            "CO₂ (kg/km)": co2,
            "CH₄ (g/km)": ch4,
            "N₂O (g/km)": n2o,
            "CO₂eq (kg/km)": co2eq,
        })
        utils.record_number(f"s1_fleet_co2_{year}", co2, "kg/km", SOURCE_STEP)
        utils.record_number(f"s1_fleet_ch4_{year}", ch4, "g/km", SOURCE_STEP)
        utils.record_number(f"s1_fleet_n2o_{year}", n2o, "g/km", SOURCE_STEP)
        utils.record_number(f"s1_fleet_co2eq_{year}", co2eq, "kg/km", SOURCE_STEP)

    utils.record_number("gwp_co2", config.GWP_CO2, "GWP100 (AR5)", SOURCE_STEP)
    utils.record_number("gwp_ch4", config.GWP_CH4, "GWP100 (AR5)", SOURCE_STEP)
    utils.record_number("gwp_n2o", config.GWP_N2O, "GWP100 (AR5)", SOURCE_STEP)
    utils.record_number("fleet_diesel_share_pct", config.DIESEL_SHARE * 100, "%", SOURCE_STEP)
    utils.record_number("fleet_gasoline_share_pct", config.GASOLINE_SHARE * 100, "%", SOURCE_STEP)
    utils.record_number("fleet_other_share_pct", config.OTHER_SHARE * 100, "%", SOURCE_STEP)

    return pd.DataFrame(rows, columns=["Fuel type (share)", "Year", "CO₂ (kg/km)", "CH₄ (g/km)",
                                        "N₂O (g/km)", "CO₂eq (kg/km)"])


def build_s1_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Provenance sheet for S1 Table: fleet composition, emission-factor and
    GWP sources, plus a run-time check of the published CH4 total (311.5 kg
    over the full study period).
    """
    total_ch4_kg = float(df["emissio_ch4_kg"].sum())
    utils.record_number("total_ch4_kg_check", total_ch4_kg, "kg", SOURCE_STEP)
    rows = [
        ("Emission factor source", config.EMISSION_FACTOR_SOURCE),
        ("Fleet composition source", config.FLEET_SOURCE),
        ("GWP source", config.GWP_SOURCE),
        ("Diesel fleet share", f"{config.DIESEL_SHARE * 100:.2f}%"),
        ("Gasoline fleet share", f"{config.GASOLINE_SHARE * 100:.2f}%"),
        ("Other (zero-tailpipe) fleet share", f"{config.OTHER_SHARE * 100:.2f}%"),
        ("GWP, CO2 (100-yr)", config.GWP_CO2),
        ("GWP, CH4 (100-yr)", config.GWP_CH4),
        ("GWP, N2O (100-yr)", config.GWP_N2O),
        ("Total CH4 over the study period (computed, kg)", round(total_ch4_kg, 1)),
        ("Total CH4 over the study period (published, kg)", 311.5),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


# -----------------------------------------------------------------------------
# S2 Table: Capital vs Non-capital comparison, six variables
# -----------------------------------------------------------------------------

_S2_VARIABLE_ORDER = [
    ("Distance per interconsultation", "Distance (km)"),
    ("CO2 per interconsultation", "CO₂ (kg)"),
    ("CH4 per interconsultation", "CH₄ (g)"),
    ("N2O per interconsultation", "N₂O (g)"),
    ("CO2eq per interconsultation", "CO₂eq (kg)"),
    ("Time per interconsultation", "Time (min)"),
]


def _p_display(p: float) -> str:
    """Format a p-value the way the manuscript does: '< 0.001' below that
    threshold, else four significant decimals.
    """
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def build_s2_table(stats_sheets: dict[str, pd.DataFrame] | None) -> pd.DataFrame | None:
    """Capital vs Non-capital comparison for the six per-interconsultation
    outcome variables, read from s07's Summary sheet. Cliff's delta and its
    confidence interval keep their published sign (negative: Capital <
    Non-capital); no absolute value is taken anywhere.
    """
    if stats_sheets is None:
        return None
    summary = stats_sheets.get("Summary")
    if summary is None:
        utils.log("WARNING: 'Summary' sheet not found in statistics_capital_vs_noncapital.xlsx; S2 Table skipped.")
        return None
    utils.log("Building S2 Table (Capital vs Non-capital, six variables)")

    by_variable = {row["Variable"]: row for _, row in summary.iterrows()}
    rows = []
    for source_label, display_label in _S2_VARIABLE_ORDER:
        row = by_variable.get(source_label)
        if row is None:
            utils.log(f"WARNING: variable '{source_label}' not found in Summary sheet; skipping from S2 Table.")
            continue
        delta, lo, hi = row["Cliffs_Delta"], row["Cliffs_Delta_CI_Low"], row["Cliffs_Delta_CI_High"]
        p = row["MannWhitney_P"]
        rows.append({
            "Variable (unit)": display_label,
            "Mean Capital": row["Mean_Capital"],
            "Mean Non-Cap.": row["Mean_NonCapital"],
            "Median Capital": row["Median_Capital"],
            "Median Non-Cap.": row["Median_NonCapital"],
            "Cliffs_Delta": delta,
            "Cliffs_Delta_CI_Low": lo,
            "Cliffs_Delta_CI_High": hi,
            "Cliff's delta [95% CI]": f"{delta:.3f} [{lo:.3f}, {hi:.3f}]",
            "MannWhitney_U": row["MannWhitney_U"],
            "MannWhitney_p": p,
            "Mann-Whitney p": _p_display(p),
        })
        key = source_label.lower().replace(" ", "_")
        utils.record_number(f"s2_{key}_mean_capital", row["Mean_Capital"], display_label, SOURCE_STEP)
        utils.record_number(f"s2_{key}_mean_noncapital", row["Mean_NonCapital"], display_label, SOURCE_STEP)
        utils.record_number(f"s2_{key}_median_capital", row["Median_Capital"], display_label, SOURCE_STEP)
        utils.record_number(f"s2_{key}_median_noncapital", row["Median_NonCapital"], display_label, SOURCE_STEP)
        utils.record_number(f"s2_{key}_cliffs_delta", delta, "delta", SOURCE_STEP)
        utils.record_number(f"s2_{key}_cliffs_delta_ci_low", lo, "delta", SOURCE_STEP)
        utils.record_number(f"s2_{key}_cliffs_delta_ci_high", hi, "delta", SOURCE_STEP)
        utils.record_number(f"s2_{key}_mannwhitney_p", p, "p-value", SOURCE_STEP)
    utils.record_number("s2_n_capital", int(summary["N_Capital"].iloc[0]), "interconsultations", SOURCE_STEP)
    utils.record_number("s2_n_noncapital", int(summary["N_NonCapital"].iloc[0]), "interconsultations", SOURCE_STEP)

    cols = ["Variable (unit)", "Mean Capital", "Mean Non-Cap.", "Median Capital", "Median Non-Cap.",
            "Cliffs_Delta", "Cliffs_Delta_CI_Low", "Cliffs_Delta_CI_High", "Cliff's delta [95% CI]",
            "MannWhitney_U", "MannWhitney_p", "Mann-Whitney p"]
    return pd.DataFrame(rows, columns=cols)


# -----------------------------------------------------------------------------
# S3 Table: per-region volume, reference population and CO2 per 100,000
# -----------------------------------------------------------------------------


def build_s3_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-region interconsultation volume, CatSalut reference population,
    mean/median round-trip distance, total avoided CO2 and CO2 per 100,000
    inhabitants, by Catalan health region. Reference populations come only
    from config.REFERENCE_POPULATION.
    """
    utils.log("Building S3 Table (per-region volume and CO2 per 100,000)")
    rows = []
    for region, population in config.REFERENCE_POPULATION.items():
        sub = df.loc[df["origin_health_region"] == region]
        if sub.empty:
            utils.log(f"WARNING: no rows found for health region '{region}'.")
            continue
        weights = sub["N"].to_numpy(dtype=float)
        dist_stats = utils.weighted_stats(sub["km_per_interconsultation"].to_numpy(dtype=float), weights)
        n_sum = int(sub["N"].sum())
        total_co2 = float(sub["co2_kg_per_interconsultation"].to_numpy(dtype=float).dot(weights))
        co2_per_100k = total_co2 / population * 100_000
        rows.append({
            "Health region": region,
            "Interconsultations": n_sum,
            "Reference population": population,
            "Mean dist. (km)": dist_stats["mean"] if dist_stats else float("nan"),
            "Median dist. (km)": dist_stats["median"] if dist_stats else float("nan"),
            "Total CO2 (kg)": total_co2,
            "CO2 per 100,000": co2_per_100k,
        })
        key = region.lower().replace(" ", "_").replace("'", "").replace("è", "e")
        utils.record_number(f"s3_{key}_interconsultations", n_sum, "interconsultations", SOURCE_STEP)
        utils.record_number(f"s3_{key}_reference_population", population, "inhabitants", SOURCE_STEP)
        utils.record_number(f"s3_{key}_mean_distance_km", dist_stats["mean"] if dist_stats else float("nan"), "km", SOURCE_STEP)
        utils.record_number(f"s3_{key}_median_distance_km", dist_stats["median"] if dist_stats else float("nan"), "km", SOURCE_STEP)
        utils.record_number(f"s3_{key}_total_co2_kg", total_co2, "kg", SOURCE_STEP)
        utils.record_number(f"s3_{key}_co2_per_100000", co2_per_100k, "kg CO2 / 100,000 inhabitants", SOURCE_STEP)

    table = pd.DataFrame(rows).sort_values("CO2 per 100,000", ascending=False).reset_index(drop=True)

    # Aggregate capital vs non-capital per-100,000 figures and the
    # highest/lowest region ratio, computed at run time from the same data.
    capital_regions = config.CAPITAL_REGIONS
    capital_population = sum(pop for region, pop in config.REFERENCE_POPULATION.items() if region in capital_regions)
    noncapital_population = config.REFERENCE_POPULATION_TOTAL - capital_population
    capital_co2 = float(table.loc[table["Health region"].isin(capital_regions), "Total CO2 (kg)"].sum())
    noncapital_co2 = float(table.loc[~table["Health region"].isin(capital_regions), "Total CO2 (kg)"].sum())
    capital_per100k = capital_co2 / capital_population * 100_000
    noncapital_per100k = noncapital_co2 / noncapital_population * 100_000
    fold_diff = noncapital_per100k / capital_per100k
    highest = table["CO2 per 100,000"].max()
    lowest = table["CO2 per 100,000"].min()
    highest_lowest_fold = highest / lowest

    utils.record_number("capital_co2_per100k", capital_per100k, "kg CO2 / 100,000 inhabitants", SOURCE_STEP)
    utils.record_number("noncapital_co2_per100k", noncapital_per100k, "kg CO2 / 100,000 inhabitants", SOURCE_STEP)
    utils.record_number("fold_diff_capital_noncapital_per100k", fold_diff, "fold", SOURCE_STEP)
    utils.record_number("fold_diff_highest_lowest_region_per100k", highest_lowest_fold, "fold", SOURCE_STEP)
    utils.record_number("s3_sum_interconsultations_check", int(table["Interconsultations"].sum()), "interconsultations", SOURCE_STEP)
    utils.record_number("s3_sum_total_co2_kg_check", float(table["Total CO2 (kg)"].sum()), "kg", SOURCE_STEP)

    utils.log(
        f"Capital CO2/100k: {capital_per100k:,.0f} | Non-capital CO2/100k: {noncapital_per100k:,.0f} "
        f"({fold_diff:.1f}-fold) | Highest/lowest region ratio: {highest_lowest_fold:.0f}-fold"
    )
    return table


# -----------------------------------------------------------------------------
# S4 Table: route-level Capital vs Non-capital comparison
# -----------------------------------------------------------------------------

_S4_VARIABLE_LABELS = {
    "Round-trip distance (route-level)": "Round-trip distance (km)",
    "Round-trip time (route-level)": "Round-trip travel time (min)",
}


def build_s4_table(stats_sheets: dict[str, pd.DataFrame] | None) -> pd.DataFrame | None:
    """Route-level (unweighted, one observation per unique route) Capital
    vs Non-capital comparison, read from s07's Route_level sheet.
    """
    if stats_sheets is None:
        return None
    route_level = stats_sheets.get("Route_level")
    if route_level is None:
        utils.log("WARNING: 'Route_level' sheet not found in statistics_capital_vs_noncapital.xlsx; S4 Table skipped.")
        return None
    utils.log("Building S4 Table (route-level Capital vs Non-capital)")

    utils.record_number("s4_n_capital_routes", int(route_level["N_Capital"].iloc[0]), "routes", SOURCE_STEP)
    utils.record_number("s4_n_noncapital_routes", int(route_level["N_NonCapital"].iloc[0]), "routes", SOURCE_STEP)

    rows = []
    for _, row in route_level.iterrows():
        display_label = _S4_VARIABLE_LABELS.get(row["Variable"], row["Variable"])
        delta, lo, hi = row["Cliffs_Delta"], row["Cliffs_Delta_CI_Low"], row["Cliffs_Delta_CI_High"]
        p = row["MannWhitney_P"]
        rows.append({
            "Variable": display_label,
            "Median capital": row["Median_Capital"],
            "Median non-capital": row["Median_NonCapital"],
            "Mann-Whitney U": row["MannWhitney_U"],
            "p": _p_display(p),
            "Cliffs_Delta": delta,
            "Cliffs_Delta_CI_Low": lo,
            "Cliffs_Delta_CI_High": hi,
            "Cliff's Delta [95% CI]": f"{delta:.3f} [{lo:.3f}, {hi:.3f}]",
        })
        key = display_label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        utils.record_number(f"s4_{key}_median_capital", row["Median_Capital"], display_label, SOURCE_STEP)
        utils.record_number(f"s4_{key}_median_noncapital", row["Median_NonCapital"], display_label, SOURCE_STEP)
        utils.record_number(f"s4_{key}_mannwhitney_u", row["MannWhitney_U"], "U statistic", SOURCE_STEP)
        utils.record_number(f"s4_{key}_mannwhitney_p", p, "p-value", SOURCE_STEP)
        utils.record_number(f"s4_{key}_cliffs_delta", delta, "delta", SOURCE_STEP)
        utils.record_number(f"s4_{key}_cliffs_delta_ci_low", lo, "delta", SOURCE_STEP)
        utils.record_number(f"s4_{key}_cliffs_delta_ci_high", hi, "delta", SOURCE_STEP)

    cols = ["Variable", "Median capital", "Median non-capital", "Mann-Whitney U", "p",
            "Cliffs_Delta", "Cliffs_Delta_CI_Low", "Cliffs_Delta_CI_High", "Cliff's Delta [95% CI]"]
    return pd.DataFrame(rows, columns=cols)


# -----------------------------------------------------------------------------
# S1_Data.xlsx: published route-year extract
# -----------------------------------------------------------------------------

_S1_DATA_COLUMNS = [
    "year", "origin_code", "origin_facility", "origin_health_region", "region_type",
    "destination_code", "destination_facility", "interconsultations",
    "roundtrip_km_ors", "roundtrip_km_osrm", "roundtrip_min_ors", "roundtrip_min_osrm",
    "co2_kg_per_interconsultation", "ch4_g_per_interconsultation",
    "n2o_g_per_interconsultation", "co2eq_kg_per_interconsultation", "total_co2_kg",
]


def build_s1_data(df: pd.DataFrame, osrm: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the published route x year data extract and its README.

    Grain: (origin_code, destination_code, year), one row per route-year.
    Suppression: route-year cells whose summed interconsultations fall
    below config.SUPPRESSION_THRESHOLD keep the row but have
    `interconsultations` replaced by the literal string config.SUPPRESSION_LABEL
    and `total_co2_kg` blanked. No coordinates and no sex/age breakdown are
    included anywhere in this extract.
    """
    utils.log("Building S1_Data (published route-year extract)")
    cell_key = list(config.CELL_KEY)

    working = df.copy()
    if osrm is not None:
        osrm_cols = cell_key + ["year", "km_roundtrip_osrm", "duration_roundtrip_min_osrm"]
        missing = [c for c in osrm_cols if c not in osrm.columns]
        if missing:
            utils.log(f"WARNING: OSRM file missing columns {missing}; OSRM columns will be blank in S1_Data.")
            osrm = None
        else:
            osrm_small = osrm[osrm_cols].drop_duplicates(subset=cell_key + ["year"])
            before = len(working)
            working = working.merge(osrm_small, on=cell_key + ["year"], how="left")
            missing_osrm = int(working["km_roundtrip_osrm"].isna().sum())
            if missing_osrm:
                utils.log(f"WARNING: {missing_osrm:,} of {before:,} rows had no OSRM match after merge.")
    if osrm is None:
        working["km_roundtrip_osrm"] = np.nan
        working["duration_roundtrip_min_osrm"] = np.nan

    route_key = list(config.ROUTE_KEY)
    grouped = working.groupby(route_key + ["year"], as_index=False).agg(
        origin_health_region=("origin_health_region", "first"),
        origin_facility=("origin_location_name", "first"),
        destination_facility=("destination_location_name", "first"),
        interconsultations=("N", "sum"),
        roundtrip_km_ors=("km_roundtrip", "mean"),
        roundtrip_km_osrm=("km_roundtrip_osrm", "mean"),
        roundtrip_min_ors=("duration_roundtrip_min", "mean"),
        roundtrip_min_osrm=("duration_roundtrip_min_osrm", "mean"),
        co2_kg_per_interconsultation=("co2_kg_per_interconsultation", "mean"),
        ch4_kg_per_interconsultation=("ch4_kg_per_interconsultation", "mean"),
        n2o_kg_per_interconsultation=("n2o_kg_per_interconsultation", "mean"),
        co2eq_kg_per_interconsultation=("co2eq_kg_per_interconsultation", "mean"),
    )
    grouped = grouped.rename(columns={"OC_UP_ORI": "origin_code", "DER_UP_SCS_DES": "destination_code"})
    grouped["region_type"] = grouped["origin_health_region"].apply(utils.classify_region_type)
    grouped["ch4_g_per_interconsultation"] = grouped["ch4_kg_per_interconsultation"] * 1000
    grouped["n2o_g_per_interconsultation"] = grouped["n2o_kg_per_interconsultation"] * 1000
    grouped["total_co2_kg"] = grouped["co2_kg_per_interconsultation"] * grouped["interconsultations"]

    total_rows = len(grouped)
    full_n = int(grouped["interconsultations"].sum())
    full_co2 = float(grouped["total_co2_kg"].sum())

    suppressed_mask = grouped["interconsultations"] < config.SUPPRESSION_THRESHOLD
    n_suppressed = int(suppressed_mask.sum())
    withheld_n = int(grouped.loc[suppressed_mask, "interconsultations"].sum())
    withheld_co2 = float(grouped.loc[suppressed_mask, "total_co2_kg"].sum())
    visible_n = full_n - withheld_n
    visible_co2 = full_co2 - withheld_co2

    grouped["total_co2_kg"] = grouped["total_co2_kg"].where(~suppressed_mask, other=np.nan)
    grouped["interconsultations"] = grouped["interconsultations"].astype(object)
    grouped.loc[suppressed_mask, "interconsultations"] = config.SUPPRESSION_LABEL

    data = grouped[_S1_DATA_COLUMNS].sort_values(["year", "origin_code", "destination_code"]).reset_index(drop=True)

    pct_n_withheld = withheld_n / full_n * 100
    pct_co2_withheld = withheld_co2 / full_co2 * 100
    avg_co2_suppressed = withheld_co2 / withheld_n if withheld_n else float("nan")
    avg_co2_visible = visible_co2 / visible_n if visible_n else float("nan")

    utils.log(
        f"S1_Data: {total_rows:,} route-year rows, {n_suppressed:,} suppressed. "
        f"Withheld: {pct_n_withheld:.2f}% of interconsultations, {pct_co2_withheld:.2f}% of total CO2."
    )

    utils.record_number("s1_data_rows", total_rows, "rows", SOURCE_STEP)
    utils.record_number("s1_data_suppressed_rows", n_suppressed, "rows", SOURCE_STEP)
    utils.record_number("s1_data_visible_interconsultations", visible_n, "interconsultations", SOURCE_STEP)
    utils.record_number("s1_data_visible_total_co2_kg", visible_co2, "kg", SOURCE_STEP)
    utils.record_number("s1_data_withheld_interconsultations_pct", pct_n_withheld, "%", SOURCE_STEP)
    utils.record_number("s1_data_withheld_co2_pct", pct_co2_withheld, "%", SOURCE_STEP)
    utils.record_number("s1_data_avg_co2_per_ic_suppressed", avg_co2_suppressed, "kg", SOURCE_STEP)
    utils.record_number("s1_data_avg_co2_per_ic_visible", avg_co2_visible, "kg", SOURCE_STEP)

    readme_rows = [
        ("Dataset", "S1_Data: published route-year extract of the interconsultations emissions analysis."),
        ("Grain", "One row per unique origin-destination route per study year "
                   "(origin_code, destination_code, year)."),
        ("Rows", f"{total_rows:,} route-year combinations, covering {config.YEARS}."),
        ("Suppression rule", f"Route-year cells with fewer than {config.SUPPRESSION_THRESHOLD} "
                               f"interconsultations keep the row, but 'interconsultations' is "
                               f"replaced by '{config.SUPPRESSION_LABEL}' and 'total_co2_kg' is blank."),
        ("Suppressed rows", f"{n_suppressed:,} of {total_rows:,} route-year rows."),
        ("Interconsultations withheld",
         f"{pct_n_withheld:.2f}% of all interconsultations ({withheld_n:,} of {full_n:,}) "
         "are in suppressed rows and so are not visible in this extract."),
        ("Total CO2 withheld",
         f"{pct_co2_withheld:.2f}% of total avoided CO2 ({withheld_co2:,.0f} of {full_co2:,.0f} kg) "
         "is in suppressed rows and so is not visible in this extract."),
        ("Why the two shares differ",
         f"Suppressed (low-volume) routes average {avg_co2_suppressed:.1f} kg CO2 per "
         f"interconsultation, against {avg_co2_visible:.1f} kg for the rest: low-volume routes "
         "are disproportionately long, so they are suppressed more often than their small share "
         "of interconsultations alone would suggest, and the CO2 share withheld is "
         "correspondingly larger than the interconsultation share withheld."),
        ("Restricted content", "No coordinates and no breakdown by sex or age group are included. "
                                 "Facility names and health regions are included, as in the "
                                 "published extract."),
        ("Columns", ", ".join(_S1_DATA_COLUMNS)),
    ]
    readme = pd.DataFrame(readme_rows, columns=["Field", "Value"])
    return data, readme


# -----------------------------------------------------------------------------
# Headline numbers (recomputed here so manuscript_numbers.csv is complete
# even when this step is run standalone, since each step's record_number
# collector only lives for the duration of its own process).
# -----------------------------------------------------------------------------


def record_headline_numbers(df: pd.DataFrame) -> None:
    n_sum = int(df["N"].sum())
    n_routes = df.groupby(config.ROUTE_KEY).ngroups
    n_route_years = df.groupby(config.ROUTE_KEY + ["year"]).ngroups
    total_km = float(df["total_km"].sum())
    total_co2 = float(df["emissio_co2_kg"].sum())
    total_ch4 = float(df["emissio_ch4_kg"].sum())
    total_n2o = float(df["emissio_n2o_kg"].sum())
    total_co2eq = float(df["emissio_co2eq_kg"].sum())
    n_by_region_type = df.groupby("region_type")["N"].sum()
    capital_n = int(n_by_region_type.get("Capital", 0))
    noncapital_n = int(n_by_region_type.get("Non-capital", 0))
    capital_co2_share = float(df.loc[df["region_type"] == "Capital", "emissio_co2_kg"].sum()) / total_co2 * 100
    noncapital_co2_share = 100 - capital_co2_share

    utils.record_number("total_interconsultations", n_sum, "interconsultations", SOURCE_STEP)
    utils.record_number("unique_routes", n_routes, "routes", SOURCE_STEP)
    utils.record_number("route_year_groups", n_route_years, "route-year groups", SOURCE_STEP)
    utils.record_number("aggregated_records", len(df), "records", SOURCE_STEP)
    utils.record_number("total_avoided_km", total_km, "km", SOURCE_STEP)
    utils.record_number("total_avoided_co2_kg", total_co2, "kg", SOURCE_STEP)
    utils.record_number("total_avoided_ch4_kg", total_ch4, "kg", SOURCE_STEP)
    utils.record_number("total_avoided_n2o_kg", total_n2o, "kg", SOURCE_STEP)
    utils.record_number("total_avoided_co2eq_kg", total_co2eq, "kg", SOURCE_STEP)
    utils.record_number("capital_interconsultations", capital_n, "interconsultations", SOURCE_STEP)
    utils.record_number("noncapital_interconsultations", noncapital_n, "interconsultations", SOURCE_STEP)
    utils.record_number("capital_total_co2_kg", float(df.loc[df["region_type"] == "Capital", "emissio_co2_kg"].sum()), "kg", SOURCE_STEP)
    utils.record_number("noncapital_total_co2_kg", float(df.loc[df["region_type"] == "Non-capital", "emissio_co2_kg"].sum()), "kg", SOURCE_STEP)
    utils.record_number("capital_avoided_co2_share_pct", capital_co2_share, "%", SOURCE_STEP)
    utils.record_number("noncapital_avoided_co2_share_pct", noncapital_co2_share, "%", SOURCE_STEP)
    utils.record_number("capital_interconsultations_share_pct", capital_n / n_sum * 100, "%", SOURCE_STEP)

    utils.log(f"Headline: N={n_sum:,} routes={n_routes:,} route-years={n_route_years:,} "
              f"total_km={total_km:,.0f} total_co2={total_co2:,.0f} kg")


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def _write_multisheet(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    """Write several DataFrames to one workbook, one sheet each, in the
    given order. Sheet names are kept ASCII and truncated to Excel's 31-
    character limit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=name[:31], index=False)
    utils.log(f"Saved: {path}")


def _write_single(path: Path, table: pd.DataFrame) -> None:
    utils.write_table(table, path)
    utils.log(f"Saved: {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    start = time.time()
    utils.log("=" * 70)
    utils.log("s10_tables: manuscript tables, published data extract, number inventory")
    utils.log("=" * 70)

    df = load_analysis_dataset()
    osrm = load_osrm_dataset()
    stats_sheets = load_statistics_workbook()
    sensitivity_sheets = load_sensitivity_workbook()

    record_headline_numbers(df)

    table1 = build_table1(df)
    table2 = build_table2(df)
    table3 = build_table3(df)
    table4 = build_table4(sensitivity_sheets)
    table5 = build_table5(sensitivity_sheets)

    s1_table = build_s1_table()
    s1_sources = build_s1_sources(df)
    s2_table = build_s2_table(stats_sheets)
    s3_table = build_s3_table(df)
    s4_table = build_s4_table(stats_sheets)

    s1_data, s1_readme = build_s1_data(df, osrm)

    _write_single(config.TABLE_DIR / "Table1.xlsx", table1)
    _write_single(config.TABLE_DIR / "Table2.xlsx", table2)
    _write_single(config.TABLE_DIR / "Table3.xlsx", table3)

    skipped = []
    if table4 is not None:
        _write_single(config.TABLE_DIR / "Table4.xlsx", table4)
    else:
        skipped.append("Table4 (substitution-rate sensitivity; sensitivity_analysis.xlsx not found)")
    if table5 is not None:
        _write_single(config.TABLE_DIR / "Table5.xlsx", table5)
    else:
        skipped.append("Table5 (transport-mode sensitivity; sensitivity_analysis.xlsx not found)")

    _write_multisheet(config.TABLE_DIR / "S1_Table.xlsx", {"S1_Table": s1_table, "Sources": s1_sources})

    if s2_table is not None:
        _write_single(config.TABLE_DIR / "S2_Table.xlsx", s2_table)
    else:
        skipped.append("S2_Table (statistics_capital_vs_noncapital.xlsx 'Summary' sheet not found)")

    _write_single(config.TABLE_DIR / "S3_Table.xlsx", s3_table)

    if s4_table is not None:
        _write_single(config.TABLE_DIR / "S4_Table.xlsx", s4_table)
    else:
        skipped.append("S4_Table (statistics_capital_vs_noncapital.xlsx 'Route_level' sheet not found)")

    _write_multisheet(config.TABLE_DIR / "S1_Data.xlsx", {"README": s1_readme, "route_year": s1_data})

    numbers_path = config.output_path("manuscript_numbers.csv")
    utils.write_numbers(numbers_path)
    utils.log(f"Saved: {numbers_path}")

    if skipped:
        utils.log("Skipped tables: " + "; ".join(skipped))
    else:
        utils.log("All tables produced; no inputs were missing.")

    elapsed = time.time() - start
    utils.log(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
