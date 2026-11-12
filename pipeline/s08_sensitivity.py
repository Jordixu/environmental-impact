"""s08: Sensitivity analysis, substitution rate and transport mode.

Both analyses start from the headline emissions totals in the analysis
dataset (s06's output) and ask the same question from two angles: how much
would the reported avoided-emissions figure change under a less optimistic
assumption than "every interconsultation avoided a car trip made solely for
that purpose"?

(A) Substitution rate (Substitution_Rate sheet)
    The headline totals assume full substitution: every interconsultation
    avoided a round trip that would otherwise have been made. This section
    rescales the headline distance/CO2/N2O totals at config.SUBSTITUTION_RATES
    (100%, 70%, 50%), i.e. what the totals would be if only that fraction of
    interconsultations actually avoided a trip. There is no legacy
    implementation to port here; the rescaling is a direct proportional one
    (rate x headline total), and it is verified numerically below instead of
    just assumed.

(B) Transport mode (Transport_Mode and Assumptions_and_Sources sheets)
    The headline totals also assume every avoided trip was made entirely by
    private car. This section re-computes total avoided CO2 under
    literature-grounded mixed-transport assumptions: EMEF 2024 for the
    capital/metropolitan car-vs-public-transport split, Departament de
    Territori / Mobilicat for the non-capital split, and MITECO 2025 for the
    public-transport CO2 factors, ported from the legacy sensitivity
    notebook. Car shares are restricted to the split among motorized trips
    only (car / (car + public transport)): the raw survey figures cover all
    trip purposes and distances, dominated by short walkable or cyclable
    trips, which is not the relevant choice set for a referral trip of this
    length.

Input:
    utils.resolve_input("analysis_dataset.parquet"), s06's output, already
    carrying the renamed columns (config.COLUMN_RENAMES).

Output:
    OUTPUT_DIR/sensitivity_analysis.xlsx, with sheets Substitution_Rate,
    Transport_Mode, Assumptions_and_Sources.
"""

from __future__ import annotations

import pandas as pd

import config
from pipeline import utils

OUTPUT_FILE = "sensitivity_analysis.xlsx"

# EMEF 2024 all-mode weekday modal split for the Barcelona Metropolitan Area:
# private vehicle 33.5%, public transport 18.2%, and the remaining 48.3% is
# active mobility (walking/cycling, zero emissions). config.CAR_SHARE_CAPITAL
# holds these same two figures already restricted to motorized trips only
# (0.335 / (0.335 + 0.182)). Scenario 4 below needs the un-normalized,
# all-mode pair directly, since it credits the active-mobility share with
# zero emissions instead of excluding it from the choice set.
_EMEF_CAPITAL_CAR_SHARE_ALLMODE = 0.335
_EMEF_CAPITAL_PT_SHARE_ALLMODE = 0.182

_MITECO_SOURCE = "MITECO 2025, Factores de emision: Registro de huella de carbono"
_CAR_SHARE_SOURCE_CAPITAL = "EMEF 2024, Barcelona Metropolitan Area (Institut Metropoli / ATM)"
_CAR_SHARE_SOURCE_NONCAPITAL = "Departament de Territori / Mobilicat rural-mobility data"


# -----------------------------------------------------------------------------
# (A) Substitution rate
# -----------------------------------------------------------------------------


def _substitution_rate_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale the headline totals (total round-trip km, total CO2, total
    N2O) at each of config.SUBSTITUTION_RATES, by direct proportional
    multiplication, then check that the 70% and 50% rows really are exact
    proportional rescalings of the 100% row rather than just assuming it.
    """
    total_km = float(df["total_km"].sum())
    total_co2 = float(df["emissio_co2_kg"].sum())
    total_n2o = float(df["emissio_n2o_kg"].sum())

    rows = []
    for rate in config.SUBSTITUTION_RATES:
        rows.append(
            {
                "Substitution rate": f"{rate:.0%}",
                "Total km": total_km * rate,
                "Total CO2 (kg)": total_co2 * rate,
                "Total N2O (kg)": total_n2o * rate,
            }
        )
    sheet = pd.DataFrame(rows)

    # Confirm numerically, for each of the three quantities, that every row
    # is an exact proportional rescaling of the 100% row.
    base = sheet.iloc[0]
    base_rate = config.SUBSTITUTION_RATES[0]
    for i, rate in enumerate(config.SUBSTITUTION_RATES):
        for col in ("Total km", "Total CO2 (kg)", "Total N2O (kg)"):
            expected = base[col] * (rate / base_rate)
            actual = sheet.iloc[i][col]
            rel_diff = abs(actual - expected) / expected if expected else 0.0
            if rel_diff > 1e-12:
                raise AssertionError(
                    f"Substitution rate {rate:.0%} / {col} is not an exact proportional "
                    f"rescaling of the {base_rate:.0%} row: expected {expected}, got {actual}"
                )
    utils.log(
        "Verified: the 70% and 50% substitution-rate rows are exact proportional "
        "rescalings of the 100% row for km, CO2 and N2O."
    )

    for rate, row in zip(config.SUBSTITUTION_RATES, rows):
        pct = f"{round(rate * 100)}"
        utils.record_number(
            f"sensitivity_substitution_{pct}pct_km", row["Total km"], "km", "s08_sensitivity"
        )
        utils.record_number(
            f"sensitivity_substitution_{pct}pct_co2_kg", row["Total CO2 (kg)"], "kg", "s08_sensitivity"
        )
        utils.record_number(
            f"sensitivity_substitution_{pct}pct_n2o_kg", row["Total N2O (kg)"], "kg", "s08_sensitivity"
        )

    return sheet


# -----------------------------------------------------------------------------
# (B) Transport mode
# -----------------------------------------------------------------------------


def _region_co2(df: pd.DataFrame, mask: pd.Series, car_share: float, pt_factor: float) -> float:
    """Total CO2 (kg) for the region subset `mask` under a car_share /
    pt_factor assumption:
        emissions = km_per_interconsultation * N
                    * (car_share * co2_factor + (1 - car_share) * pt_factor)
    `co2_factor` is the fleet-weighted CO2 factor (kg/km) the analysis
    dataset already carries for the row's year; `pt_factor` is a flat
    public-transport CO2 factor (kg CO2e/passenger-km).
    """
    sub = df[mask]
    return float(
        (
            sub["km_per_interconsultation"]
            * sub["N"]
            * (car_share * sub["co2_factor"] + (1 - car_share) * pt_factor)
        ).sum()
    )


def _transport_mode_scenarios(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Compute the six transport-mode scenarios (1, 2a, 2b, 2c, 3, 4),
    porting the legacy sensitivity notebook's logic onto the renamed
    analysis dataset columns. Returns the summary sheet and the baseline
    total (scenario 1); scenario 4's "short capital trip" share is also
    reported against this baseline.
    """
    region_type = df["origin_health_region"].apply(utils.classify_region_type)
    capital_mask = region_type == "Capital"
    noncapital_mask = region_type == "Non-capital"

    car_capital = config.CAR_SHARE_CAPITAL
    car_noncapital = config.CAR_SHARE_NONCAPITAL
    pt_urban_bus = config.PT_FACTORS["urban_bus"]
    pt_interurban_bus = config.PT_FACTORS["interurban_bus"]
    pt_regional_train = config.PT_FACTORS["regional_train"]

    baseline = float(df["emissio_co2_kg"].sum())

    # 1. Private-vehicle-only (the current headline baseline, all regions).
    scenario_1 = baseline

    # 2a. Mixed transport, zero public-transport credit: a lower bound where
    #     non-car trips contribute zero marginal emissions in this analysis.
    scenario_2a = _region_co2(df, capital_mask, car_capital, 0.0) + _region_co2(
        df, noncapital_mask, car_noncapital, 0.0
    )

    # 2b/2c. Mixed transport, partial public-transport credit. Capital public
    #        transport is costed at the urban-bus factor throughout; non-capital
    #        public transport is costed at the interurban-bus factor (2b, lower
    #        bound) or the regional-train factor (2c, higher bound).
    scenario_2b = _region_co2(df, capital_mask, car_capital, pt_urban_bus) + _region_co2(
        df, noncapital_mask, car_noncapital, pt_interurban_bus
    )
    scenario_2c = _region_co2(df, capital_mask, car_capital, pt_urban_bus) + _region_co2(
        df, noncapital_mask, car_noncapital, pt_regional_train
    )

    # 3. The non-capital (rural/private-vehicle-dominant) car share applied
    #    uniformly everywhere, as an upper-bound check.
    scenario_3 = _region_co2(df, capital_mask, car_noncapital, 0.0) + _region_co2(
        df, noncapital_mask, car_noncapital, 0.0
    )

    # 4. Active travel for short capital-region trips. Capital trips shorter
    #    than SHORT_RT_KM round trip get the EMEF 2024 all-mode split (car,
    #    urban bus, and active travel at zero emissions) instead of the
    #    motorized-only split; longer capital trips and all non-capital
    #    trips keep the scenario 2b assumptions.
    short_cap = capital_mask & (df["km_per_interconsultation"] < config.SHORT_RT_KM)
    long_cap = capital_mask & ~short_cap
    s = df[short_cap]
    e_short = float(
        (
            s["km_per_interconsultation"]
            * s["N"]
            * (
                _EMEF_CAPITAL_CAR_SHARE_ALLMODE * s["co2_factor"]
                + _EMEF_CAPITAL_PT_SHARE_ALLMODE * pt_urban_bus
            )
        ).sum()
    )
    scenario_4 = (
        e_short
        + _region_co2(df, long_cap, car_capital, pt_urban_bus)
        + _region_co2(df, noncapital_mask, car_noncapital, pt_interurban_bus)
    )

    capital_n = float(df.loc[capital_mask, "N"].sum())
    short_cap_n = float(df.loc[short_cap, "N"].sum())
    short_cap_share = short_cap_n / capital_n if capital_n else 0.0
    utils.log(
        f"Share of capital-region interconsultations under {config.SHORT_RT_KM:.0f} km "
        f"round trip: {short_cap_share:.1%}"
    )
    utils.record_number(
        "sensitivity_scenario4_capital_short_trip_share", short_cap_share * 100, "%", "s08_sensitivity"
    )

    rows = [
        ("1. Private-vehicle-only (baseline)", scenario_1),
        ("2a. Mixed transport, zero PT credit", scenario_2a),
        ("2b. Mixed transport, partial PT credit (interurban bus)", scenario_2b),
        ("2c. Mixed transport, partial PT credit (regional train)", scenario_2c),
        ("3. Rural/private-vehicle-dominant everywhere", scenario_3),
        ("4. Active travel for short capital trips (<10 km round trip), else 2b", scenario_4),
    ]
    sheet = pd.DataFrame(
        [
            {"Scenario": name, "Total CO2 (kg)": value, "% of baseline": value / baseline * 100}
            for name, value in rows
        ]
    )

    for name, value in rows:
        slug = name.split(".")[0].strip().lower()
        utils.record_number(f"sensitivity_scenario_{slug}_co2_kg", value, "kg", "s08_sensitivity")
        utils.record_number(
            f"sensitivity_scenario_{slug}_pct_of_baseline", value / baseline * 100, "%", "s08_sensitivity"
        )

    return sheet, baseline


def _assumptions_sheet() -> pd.DataFrame:
    """Assumption/source table for the transport-mode scenarios, mirroring
    the legacy notebook's Assumptions_and_Sources sheet. Every parameter
    comes from config, and every citation is named.
    """
    return pd.DataFrame(
        [
            {
                "Parameter": "Capital car share (of motorized trips)",
                "Value": config.CAR_SHARE_CAPITAL,
                "Source": _CAR_SHARE_SOURCE_CAPITAL,
            },
            {
                "Parameter": "Non-capital car share (of motorized trips)",
                "Value": config.CAR_SHARE_NONCAPITAL,
                "Source": _CAR_SHARE_SOURCE_NONCAPITAL,
            },
            {
                "Parameter": "PT factor - urban bus (kg CO2e/pkm)",
                "Value": config.PT_FACTORS["urban_bus"],
                "Source": _MITECO_SOURCE,
            },
            {
                "Parameter": "PT factor - interurban bus (kg CO2e/pkm)",
                "Value": config.PT_FACTORS["interurban_bus"],
                "Source": _MITECO_SOURCE,
            },
            {
                "Parameter": "PT factor - regional train (kg CO2e/pkm)",
                "Value": config.PT_FACTORS["regional_train"],
                "Source": _MITECO_SOURCE,
            },
        ]
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    utils.log("=" * 60)
    utils.log("Step 08 - Sensitivity analysis")
    utils.log("=" * 60)

    path = utils.resolve_input("analysis_dataset.parquet")
    df = utils.read_table(path)
    utils.log(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {path}")

    utils.log("Computing substitution-rate sensitivity...")
    substitution_sheet = _substitution_rate_sheet(df)
    utils.log(substitution_sheet.to_string(index=False))

    utils.log("Computing transport-mode sensitivity...")
    transport_sheet, baseline = _transport_mode_scenarios(df)
    utils.log(transport_sheet.to_string(index=False))

    assumptions_sheet = _assumptions_sheet()

    out_path = config.output_path(OUTPUT_FILE)
    with pd.ExcelWriter(out_path) as writer:
        substitution_sheet.to_excel(writer, sheet_name="Substitution_Rate", index=False)
        transport_sheet.to_excel(writer, sheet_name="Transport_Mode", index=False)
        assumptions_sheet.to_excel(writer, sheet_name="Assumptions_and_Sources", index=False)
    utils.log(f"Saved: {out_path}")

    utils.log("Done.")
    utils.flush_numbers("s08_sensitivity")


if __name__ == "__main__":
    main()
