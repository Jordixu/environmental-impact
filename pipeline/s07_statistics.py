"""Inferential statistics: Capital vs Non-capital health regions.

This step reproduces the published comparison between interconsultations
originating in the Capital health regions (the three Barcelona sub-regions)
and those originating everywhere else (Non-capital), plus two analyses with
no legacy counterpart: a route-level comparison for S4 Table and a sex
split of the analysis population.

Three analyses, each written to its own sheet(s) of
OUTPUT_DIR/statistics_capital_vs_noncapital.xlsx:

1. Six-variable comparison (round-trip distance, CO2, CH4, N2O, CO2eq, travel
   time), each expanded to one observation per interconsultation and tested
   with Shapiro-Wilk (normality), Levene (homogeneity of variance),
   Mann-Whitney U (the headline test, since normality is violated) and
   Cliff's delta as its matching non-parametric effect size (Romano et al.,
   2006, for the interpretation thresholds). Cohen's d is also reported, but
   labelled reference-only because it assumes normality. Confidence
   intervals for the mean difference, median difference and Cliff's delta
   come from a route-clustered percentile bootstrap: every interconsultation
   on the same origin-destination route shares the same distance, so routes,
   not individual interconsultations, are the independent unit that must be
   resampled.

2. Route-level comparison (S4 Table): one unweighted row per unique
   origin-destination route, so no population weighting and no
   pseudo-replication from interconsultation counts. Because each route is
   already an independent observation at this grain, the confidence
   intervals here come from an ordinary percentile bootstrap over routes
   (not the clustered bootstrap used in (1), which exists specifically to
   handle route-level non-independence that does not apply at this grain).

3. Sex split of the analysis population: aggregate numerator, denominator
   and share for each sex. No further breakdown is produced.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

import config
from pipeline import utils

# -----------------------------------------------------------------------------
# Variables for the six-variable Capital vs Non-capital comparison.
# (column, sheet/report label, unit, scale factor applied after expansion)
# CH4 and N2O are reported in grams (scale 1000) to match the legacy report;
# distance, CO2 and CO2eq keep their native kg/km scale.
# -----------------------------------------------------------------------------
VARIABLES = [
    ("km_per_interconsultation", "Distance per interconsultation", "km", 1.0),
    ("co2_kg_per_interconsultation", "CO2 per interconsultation", "kg", 1.0),
    ("ch4_kg_per_interconsultation", "CH4 per interconsultation", "g", 1000.0),
    ("n2o_kg_per_interconsultation", "N2O per interconsultation", "g", 1000.0),
    ("co2eq_kg_per_interconsultation", "CO2eq per interconsultation", "kg", 1.0),
    ("minutes_per_interconsultation", "Time per interconsultation", "min", 1.0),
]

# Raw SEXE codes to English labels for the sex split. Only these aggregate
# totals are reported; no breakdown by region, age, year or facility.
SEX_LABELS = {"D": "Women", "H": "Men"}

SOURCE_STEP = "s07_statistics"


# =============================================================================
# Input resolution
# =============================================================================


def load_analysis_dataset() -> pd.DataFrame:
    """Load the analysis dataset produced by s06_emissions.

    Falls back to the legacy frozen file (legacy_analysis_dataset.parquet) with
    config.COLUMN_RENAMES applied, so this step can be exercised before
    s06 has run.
    """
    path = utils.resolve_input("analysis_dataset.parquet")
    if path.exists():
        utils.log(f"Loading analysis dataset: {path.name}")
        return utils.read_table(path)

    legacy_path = config.data_path("legacy_analysis_dataset.parquet")
    utils.log(
        "analysis_dataset.parquet not found in OUTPUT_DIR or DATA_DIR; "
        f"falling back to legacy {legacy_path.name} with COLUMN_RENAMES applied"
    )
    df = utils.read_table(legacy_path)
    return df.rename(columns=config.COLUMN_RENAMES)


# =============================================================================
# 1. Six-variable Capital vs Non-capital comparison
# =============================================================================


def analyze_variable(df: pd.DataFrame, var: str, label: str, unit: str, scale: float) -> dict:
    """Full statistical comparison for one outcome variable. This follows
    the recipe in the pipeline contract (Section 8), reverse engineered
    from the legacy step so every published number reproduces exactly.
    """
    utils.log(f"Analyzing: {label}")

    capital_agg = df.loc[df["region_type"] == "Capital", [var, "N"]].dropna()
    noncap_agg = df.loc[df["region_type"] == "Non-capital", [var, "N"]].dropna()

    # Expand to one observation per interconsultation, then scale.
    capital = np.repeat(capital_agg[var].to_numpy(), capital_agg["N"].to_numpy()) * scale
    noncap = np.repeat(noncap_agg[var].to_numpy(), noncap_agg["N"].to_numpy()) * scale

    n_capital, n_noncap = len(capital), len(noncap)
    utils.log(f"  N: Capital={n_capital:,}, Non-capital={n_noncap:,}")

    mean_capital, mean_noncap = float(np.mean(capital)), float(np.mean(noncap))
    median_capital, median_noncap = float(np.median(capital)), float(np.median(noncap))
    sd_capital = float(np.std(capital, ddof=1))
    sd_noncap = float(np.std(noncap, ddof=1))

    # Shapiro-Wilk: one rng shared by both groups, with Capital drawn first.
    # The order and the generator sharing both matter here: drawing
    # Non-capital first, or using two separate generators, changes the
    # reported W statistics.
    shapiro_rng = np.random.default_rng(config.RANDOM_SEED)
    cap_sample = (
        shapiro_rng.choice(capital, size=config.SHAPIRO_SUBSAMPLE, replace=False)
        if n_capital > config.SHAPIRO_SUBSAMPLE
        else capital
    )
    noncap_sample = (
        shapiro_rng.choice(noncap, size=config.SHAPIRO_SUBSAMPLE, replace=False)
        if n_noncap > config.SHAPIRO_SUBSAMPLE
        else noncap
    )
    w_capital, p_capital = stats.shapiro(cap_sample)
    w_noncap, p_noncap = stats.shapiro(noncap_sample)

    levene_stat, levene_p = stats.levene(capital, noncap)

    u_stat, u_p = stats.mannwhitneyu(capital, noncap, alternative="two-sided")

    delta = utils.cliffs_delta(capital, noncap, u_stat=u_stat)
    delta_interp = utils.interpret_cliffs_delta(delta)

    d = utils.cohens_d(capital, noncap)
    d_interp = utils.interpret_effect_size(d)

    # Confidence intervals come from a route-clustered bootstrap on
    # cell-level data (the route x year x value aggregates), not the
    # expanded arrays, because interconsultations on the same route are
    # not independent observations.
    year_col = utils.year_column(df)

    def cells(zone: str) -> pd.DataFrame:
        sub = df.loc[
            df["region_type"] == zone, config.ROUTE_KEY + [year_col, var, "N"]
        ].dropna()
        return sub.groupby(config.ROUTE_KEY + [year_col, var], as_index=False)["N"].sum()

    ci = utils.cluster_bootstrap(
        cells("Capital"),
        cells("Non-capital"),
        var,
        scale,
        config.N_BOOTSTRAP,
        config.CONFIDENCE_LEVEL,
        config.RANDOM_SEED,
    )
    mean_diff_lo, mean_diff_hi = ci["mean_diff_ci"]
    median_diff_lo, median_diff_hi = ci["median_diff_ci"]
    delta_lo, delta_hi = ci["cliffs_delta_ci"]

    return {
        "Variable": label,
        "Unit": unit,
        "N_Capital": n_capital,
        "N_NonCapital": n_noncap,
        "Mean_Capital": mean_capital,
        "Mean_NonCapital": mean_noncap,
        "Median_Capital": median_capital,
        "Median_NonCapital": median_noncap,
        "SD_Capital": sd_capital,
        "SD_NonCapital": sd_noncap,
        "Shapiro_W_Capital": float(w_capital),
        "Shapiro_P_Capital": float(p_capital),
        "Shapiro_W_NonCapital": float(w_noncap),
        "Shapiro_P_NonCapital": float(p_noncap),
        "Levene_Statistic": float(levene_stat),
        "Levene_P": float(levene_p),
        "MannWhitney_U": float(u_stat),
        "MannWhitney_P": float(u_p),
        "Cliffs_Delta": float(delta),
        "Cliffs_Delta_Interpretation": delta_interp,
        "Cliffs_Delta_CI_Low": delta_lo,
        "Cliffs_Delta_CI_High": delta_hi,
        "Cohens_d_Reference": float(d),
        "Cohens_d_Interpretation": d_interp,
        "MeanDiff_PointEstimate": mean_capital - mean_noncap,
        "MeanDiff_CI_Low": mean_diff_lo,
        "MeanDiff_CI_High": mean_diff_hi,
        "MedianDiff_PointEstimate": median_capital - median_noncap,
        "MedianDiff_CI_Low": median_diff_lo,
        "MedianDiff_CI_High": median_diff_hi,
    }


# Units to record alongside each Summary column that is not already in the
# variable's own physical unit (used by record_number()).
_NON_PHYSICAL_UNITS = {
    "N_Capital": "observations",
    "N_NonCapital": "observations",
    "Shapiro_W_Capital": "W statistic",
    "Shapiro_P_Capital": "p-value",
    "Shapiro_W_NonCapital": "W statistic",
    "Shapiro_P_NonCapital": "p-value",
    "Levene_Statistic": "statistic",
    "Levene_P": "p-value",
    "MannWhitney_U": "U statistic",
    "MannWhitney_P": "p-value",
    "Cliffs_Delta": "delta",
    "Cliffs_Delta_Interpretation": "category",
    "Cliffs_Delta_CI_Low": "delta",
    "Cliffs_Delta_CI_High": "delta",
    "Cohens_d_Reference": "d",
    "Cohens_d_Interpretation": "category",
}


def _record_summary_row(row: dict) -> None:
    """Record every number in one Summary row through utils.record_number."""
    var_key = row["Variable"].lower().replace(" ", "_")
    unit = row["Unit"]
    for col, value in row.items():
        if col in ("Variable", "Unit"):
            continue
        rec_unit = _NON_PHYSICAL_UNITS.get(col, unit)
        utils.record_number(f"stats_{var_key}_{col}", value, rec_unit, SOURCE_STEP)


def build_detail_sheet(result: dict) -> pd.DataFrame:
    """One detail sheet per variable, in the same layout as the legacy
    workbook (Metric / Value / Notes), with ASCII-only labels.
    """
    unit = result["Unit"]
    rows = [
        ["Sample Sizes", "", ""],
        ["Capital", result["N_Capital"], "observations"],
        ["Non-capital", result["N_NonCapital"], "observations"],
        ["", "", ""],
        ["Descriptive Statistics (Capital)", "", unit],
        ["Mean", result["Mean_Capital"], ""],
        ["Median", result["Median_Capital"], ""],
        ["SD", result["SD_Capital"], ""],
        ["", "", ""],
        ["Descriptive Statistics (Non-capital)", "", unit],
        ["Mean", result["Mean_NonCapital"], ""],
        ["Median", result["Median_NonCapital"], ""],
        ["SD", result["SD_NonCapital"], ""],
        ["", "", ""],
        ["Normality Test (Shapiro-Wilk)", "", ""],
        ["Capital W", result["Shapiro_W_Capital"], ""],
        ["Capital P", result["Shapiro_P_Capital"], ""],
        ["Non-capital W", result["Shapiro_W_NonCapital"], ""],
        ["Non-capital P", result["Shapiro_P_NonCapital"], ""],
        ["", "", ""],
        ["Homogeneity of Variance (Levene)", "", ""],
        ["Statistic", result["Levene_Statistic"], ""],
        ["P-value", result["Levene_P"], ""],
        ["", "", ""],
        ["Mann-Whitney U Test", "", ""],
        ["U statistic", result["MannWhitney_U"], ""],
        ["P-value", result["MannWhitney_P"], ""],
        ["", "", ""],
        ["Effect Size (Cliff's delta)", "", ""],
        ["Point estimate", result["Cliffs_Delta"], ""],
        ["95% CI Lower", result["Cliffs_Delta_CI_Low"], ""],
        ["95% CI Upper", result["Cliffs_Delta_CI_High"], ""],
        ["Interpretation", result["Cliffs_Delta_Interpretation"], ""],
        ["", "", ""],
        ["Cohen's d (reference only, assumes normality)", "", ""],
        ["Point estimate", result["Cohens_d_Reference"], ""],
        ["Interpretation", result["Cohens_d_Interpretation"], ""],
        ["", "", ""],
        ["Mean Difference (Capital - Non-capital)", "", unit],
        ["Point estimate", result["MeanDiff_PointEstimate"], ""],
        ["95% CI Lower", result["MeanDiff_CI_Low"], ""],
        ["95% CI Upper", result["MeanDiff_CI_High"], ""],
        ["", "", ""],
        ["Median Difference (Capital - Non-capital)", "", unit],
        ["Point estimate", result["MedianDiff_PointEstimate"], ""],
        ["95% CI Lower", result["MedianDiff_CI_Low"], ""],
        ["95% CI Upper", result["MedianDiff_CI_High"], ""],
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Notes"])


# =============================================================================
# 2. Route-level comparison (S4 Table)
# =============================================================================


def _route_percentile_bootstrap(
    cap_values: dict[str, np.ndarray],
    non_values: dict[str, np.ndarray],
    n_resamples: int,
    confidence_level: float,
    random_state: int,
) -> dict[str, tuple[float, float]]:
    """Ordinary percentile bootstrap over independent routes.

    Unlike cluster_bootstrap (used for the six-variable comparison above),
    no route-clustering or interconsultation weighting is applied here. At
    this grain, each route is already the independent unit (one row per
    unique origin-destination route), so resampling routes with replacement
    is the natural percentile bootstrap. The same per-iteration resample
    indices are shared across all variables passed in, so the joint
    behaviour between, say, distance and time within a route is preserved.

    Returns a dict keyed like the input value dicts, each a (low, high)
    tuple for Cliff's delta's confidence interval at the given level.
    """
    rng = np.random.default_rng(random_state)
    n_cap = len(next(iter(cap_values.values())))
    n_non = len(next(iter(non_values.values())))
    out = {k: np.empty(n_resamples) for k in cap_values}
    for i in range(n_resamples):
        ia = rng.integers(0, n_cap, n_cap)
        ib = rng.integers(0, n_non, n_non)
        for key in cap_values:
            a = cap_values[key][ia]
            b = non_values[key][ib]
            u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
            out[key][i] = 2 * u / (n_cap * n_non) - 1
    alpha = 1 - confidence_level
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo_pct)), float(np.percentile(v, hi_pct))) for k, v in out.items()}


def route_level_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Route-level, unweighted comparison for S4 Table: one row per unique
    origin-destination route (deduplicated on config.ROUTE_KEY), so no
    population weighting and no pseudo-replication from interconsultation
    counts. Reproduces the published n=913/761, median distance and time,
    and their Cliff's delta effect sizes with confidence intervals.
    """
    utils.log("Route-level analysis (S4 Table)")

    cols = config.ROUTE_KEY + ["origin_health_region", "km_roundtrip", "duration_roundtrip_min"]
    routes = df[cols].drop_duplicates(subset=config.ROUTE_KEY).copy()
    routes["region_type"] = routes["origin_health_region"].apply(utils.classify_region_type)

    cap = routes.loc[routes["region_type"] == "Capital"]
    non = routes.loc[routes["region_type"] == "Non-capital"]
    n_cap, n_non = len(cap), len(non)
    utils.log(f"  Unique routes: Capital={n_cap:,}, Non-capital={n_non:,}")

    cap_values = {
        "distance": cap["km_roundtrip"].to_numpy(dtype=float),
        "time": cap["duration_roundtrip_min"].to_numpy(dtype=float),
    }
    non_values = {
        "distance": non["km_roundtrip"].to_numpy(dtype=float),
        "time": non["duration_roundtrip_min"].to_numpy(dtype=float),
    }

    ci = _route_percentile_bootstrap(
        cap_values, non_values, config.N_BOOTSTRAP, config.CONFIDENCE_LEVEL, config.RANDOM_SEED
    )

    rows = []
    notes = (
        "Ordinary percentile bootstrap over independent routes (one row per "
        f"unique route, no weighting), {config.N_BOOTSTRAP:,} resamples, "
        f"seed={config.RANDOM_SEED}, {int(config.CONFIDENCE_LEVEL * 100)}% CI."
    )
    for key, label, unit in (("distance", "Round-trip distance (route-level)", "km"),
                              ("time", "Round-trip time (route-level)", "min")):
        a, b = cap_values[key], non_values[key]
        u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
        delta = utils.cliffs_delta(a, b, u_stat=u_stat)
        delta_lo, delta_hi = ci[key]
        row = {
            "Variable": label,
            "Unit": unit,
            "N_Capital": n_cap,
            "N_NonCapital": n_non,
            "Median_Capital": float(np.median(a)),
            "Median_NonCapital": float(np.median(b)),
            "MannWhitney_U": float(u_stat),
            "MannWhitney_P": float(u_p),
            "Cliffs_Delta": float(delta),
            "Cliffs_Delta_Interpretation": utils.interpret_cliffs_delta(delta),
            "Cliffs_Delta_CI_Low": delta_lo,
            "Cliffs_Delta_CI_High": delta_hi,
            "Notes": notes if key == "distance" else "",
        }
        rows.append(row)

        var_key = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        for col, value in row.items():
            if col in ("Variable", "Unit", "Notes"):
                continue
            rec_unit = "observations" if col.startswith("N_") else (
                "p-value" if col == "MannWhitney_P" else
                "U statistic" if col == "MannWhitney_U" else
                "category" if col == "Cliffs_Delta_Interpretation" else
                "delta" if col.startswith("Cliffs_Delta") else unit
            )
            utils.record_number(f"route_level_{var_key}_{col}", value, rec_unit, SOURCE_STEP)

    return pd.DataFrame(rows, columns=[
        "Variable", "Unit", "N_Capital", "N_NonCapital", "Median_Capital", "Median_NonCapital",
        "MannWhitney_U", "MannWhitney_P", "Cliffs_Delta", "Cliffs_Delta_Interpretation",
        "Cliffs_Delta_CI_Low", "Cliffs_Delta_CI_High", "Notes",
    ])


# =============================================================================
# 3. Sex split
# =============================================================================


def sex_split(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sex split of the analysis population: numerator,
    denominator and share for each sex, reported at the aggregate level
    only (no breakdown by region, age, year or facility).
    """
    utils.log("Sex split")

    totals = df.groupby("SEXE")["N"].sum()
    denominator = int(totals.sum())

    rows = []
    for code, label in SEX_LABELS.items():
        numerator = int(totals.get(code, 0))
        percent = 100.0 * numerator / denominator
        rows.append({
            "Sex": label,
            "Code": code,
            "Numerator": numerator,
            "Denominator": denominator,
            "Percent": percent,
        })
        key = label.lower()
        utils.record_number(f"sex_split_{key}_numerator", numerator, "interconsultations", SOURCE_STEP)
        utils.record_number(f"sex_split_{key}_denominator", denominator, "interconsultations", SOURCE_STEP)
        utils.record_number(f"sex_split_{key}_percent", percent, "%", SOURCE_STEP)

    return pd.DataFrame(rows, columns=["Sex", "Code", "Numerator", "Denominator", "Percent"])


# =============================================================================
# Orchestration
# =============================================================================


def main() -> None:
    start = time.time()
    utils.log("=" * 70)
    utils.log("s07_statistics: Capital vs Non-capital inferential statistics")
    utils.log("=" * 70)

    df = load_analysis_dataset()
    df["region_type"] = df["origin_health_region"].apply(utils.classify_region_type)

    n_capital_rows = int((df["region_type"] == "Capital").sum())
    n_noncap_rows = int((df["region_type"] == "Non-capital").sum())
    utils.log(f"Rows: Capital={n_capital_rows:,}, Non-capital={n_noncap_rows:,}")

    # --- 1. Six-variable comparison ----------------------------------------
    results = []
    for var, label, unit, scale in VARIABLES:
        if var not in df.columns:
            utils.log(f"WARNING: column '{var}' not found, skipping {label}")
            continue
        result = analyze_variable(df, var, label, unit, scale)
        _record_summary_row(result)
        results.append(result)

    summary_cols = [
        "Variable", "Unit", "N_Capital", "N_NonCapital", "Mean_Capital", "Mean_NonCapital",
        "Median_Capital", "Median_NonCapital", "SD_Capital", "SD_NonCapital",
        "Shapiro_W_Capital", "Shapiro_P_Capital", "Shapiro_W_NonCapital", "Shapiro_P_NonCapital",
        "Levene_Statistic", "Levene_P", "MannWhitney_U", "MannWhitney_P",
        "Cliffs_Delta", "Cliffs_Delta_Interpretation", "Cliffs_Delta_CI_Low", "Cliffs_Delta_CI_High",
        "Cohens_d_Reference", "Cohens_d_Interpretation",
        "MeanDiff_PointEstimate", "MeanDiff_CI_Low", "MeanDiff_CI_High",
        "MedianDiff_PointEstimate", "MedianDiff_CI_Low", "MedianDiff_CI_High",
    ]
    summary_df = pd.DataFrame(results, columns=summary_cols)

    # --- 2. Route-level comparison ------------------------------------------
    route_df = route_level_analysis(df)

    # --- 3. Sex split --------------------------------------------------------
    sex_df = sex_split(df)

    # --- Write workbook -------------------------------------------------------
    out_path = config.output_path("statistics_capital_vs_noncapital.xlsx")
    utils.log(f"Writing: {out_path.name}")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for result in results:
            sheet_name = result["Variable"][:31]
            build_detail_sheet(result).to_excel(writer, sheet_name=sheet_name, index=False)
        route_df.to_excel(writer, sheet_name="Route_level", index=False)
        sex_df.to_excel(writer, sheet_name="Sex_split", index=False)

    elapsed = time.time() - start
    utils.log(f"Done in {elapsed:.1f}s")
    utils.flush_numbers("s07_statistics")


if __name__ == "__main__":
    main()
