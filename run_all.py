"""Orchestrate the full interconsultations emissions analysis pipeline.

Runs every pipeline step, in-process, in the dependency order described
below. That order is not the numeric filename order, because s05 consumes
an artifact that s06 produces (see the comment above STEPS). Each step is
imported as a module and run by calling its own `main()`, the same way it
would run standalone via `python -m pipeline.sNN_name`; any step that takes
CLI flags has those flags set for it based on this script's own options.

`s02b_validate_locations.py`, the interactive coordinate-validation tool, is
intentionally never run here: it opens a notebook-style UI and needs a human
at the controls. Run it directly, if needed, with:

    jupytext --to ipynb pipeline/s02b_validate_locations.py

Usage (from the `publish/` directory):

    py -3.13 run_all.py                       # full run, using frozen caches
    py -3.13 run_all.py --no-use-cache         # permit live ORS/OSRM calls
    py -3.13 run_all.py --list                 # print the step table and exit
    py -3.13 run_all.py --only s06_emissions   # run a single step
    py -3.13 run_all.py --from s07_statistics  # run from a step onward
    py -3.13 run_all.py --continue-on-error    # keep going after a failure
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

_PUBLISH_DIR = Path(__file__).resolve().parent
if str(_PUBLISH_DIR) not in sys.path:
    sys.path.insert(0, str(_PUBLISH_DIR))

# -----------------------------------------------------------------------------
# Step order
# -----------------------------------------------------------------------------
# s04_route_ors routes every origin-destination pair with ORS and writes
# routed_ors.parquet. s06_emissions turns that into
# analysis_dataset.parquet, the renamed, final analysis population.
#
# s05_route_osrm's job is to validate ORS against an independent routing
# engine (OSRM) and to quantify how much that choice matters for this
# study's final, population-weighted results. To do the second half of that
# job it merges its OSRM distances onto analysis_dataset.parquet (s06's
# output), not onto the pre-emissions intermediate, so s05 has to run after
# s06, even though its filename sorts before it. That is deliberate, not a
# mistake: swapping the order breaks s05, since analysis_dataset.parquet
# would not exist yet, instead of just changing a number.
STEPS = [
    dict(
        name="s01_preprocess",
        module="pipeline.s01_preprocess",
        inputs="raw_export_historical.xlsx, UP_codis.xlsx",
        outputs="cleaned_locations_for_geocoding.parquet, processed_export.parquet",
        note="Clean/standardize facility names; drop non-physical-service referrals; split co-managed sites.",
    ),
    dict(
        name="s02_geocode",
        module="pipeline.s02_geocode",
        inputs="cleaned_locations_for_geocoding.parquet (or the frozen cleaned_locations_geocoded.parquet)",
        outputs="cleaned_locations_geocoded.parquet, geocoded_interconsultations.parquet",
        note="Merge the frozen, hand-validated coordinates onto the derivations dataset (no network by default).",
    ),
    dict(
        name="s03_integrate_2025",
        module="pipeline.s03_integrate_2025",
        inputs="geocoded_interconsultations.parquet, raw_export_2025.txt",
        outputs="geocoded_interconsultations.parquet (updated), unroutable_2025.csv",
        note="Replace the partial historical 2025 slice with the complete 2025 export (config.YEAR_2025_MODE).",
    ),
    dict(
        name="s04_route_ors",
        module="pipeline.s04_route_ors",
        inputs="geocoded_interconsultations.parquet, route_cache_ors.pkl",
        outputs="routed_ors.parquet, route_table_ors.parquet, cache/route_cache_ors.pkl",
        note="Route every origin-destination pair with OpenRouteService; reuses the frozen cache by default.",
        cli=True,
    ),
    dict(
        name="s06_emissions",
        module="pipeline.s06_emissions",
        inputs="routed_ors.parquet",
        outputs="analysis_dataset.parquet, emissions_summary.xlsx",
        note="Compute CO2/CH4/N2O/CO2eq emissions per interconsultation; write the final analysis dataset.",
    ),
    dict(
        name="s05_route_osrm",
        module="pipeline.s05_route_osrm",
        inputs="geocoded_interconsultations.parquet, analysis_dataset.parquet, route_cache_osrm.pkl",
        outputs="routed_osrm.parquet, route_table_osrm.parquet, "
        "cache/route_cache_osrm.pkl, ors_osrm_agreement.xlsx",
        note="Validate ORS against OSRM and quantify routing-engine agreement on the final analysis population. "
        "Runs AFTER s06 -- see the comment above this list.",
        cli=True,
    ),
    dict(
        name="s07_statistics",
        module="pipeline.s07_statistics",
        inputs="analysis_dataset.parquet",
        outputs="statistics_capital_vs_noncapital.xlsx",
        note="Capital vs Non-capital inferential statistics, route-level comparison, sex split.",
    ),
    dict(
        name="s08_sensitivity",
        module="pipeline.s08_sensitivity",
        inputs="analysis_dataset.parquet",
        outputs="sensitivity_analysis.xlsx",
        note="Substitution-rate and transport-mode sensitivity analysis.",
    ),
    dict(
        name="s09_figures",
        module="pipeline.s09_figures",
        inputs="analysis_dataset.parquet, regions GeoJSON, raw/intermediate files for the Figure 1 funnel",
        outputs="figures/Fig1.tif, Fig2.tif, Fig3.tif (plus a .pdf of each)",
        note="Render the three manuscript figures: inclusion/exclusion flow chart, box plots, spatial maps.",
    ),
    dict(
        name="s10_tables",
        module="pipeline.s10_tables",
        inputs="analysis_dataset.parquet, routed_osrm.parquet, "
        "statistics_capital_vs_noncapital.xlsx, sensitivity_analysis.xlsx",
        outputs="tables/Table1..5.xlsx, S1..S4_Table.xlsx, S1_Data.xlsx, manuscript_numbers.csv",
        note="Build every manuscript table, the published S1 Data extract, and the number inventory. "
        "The final step.",
    ),
]

STEP_NAMES = [s["name"] for s in STEPS]
STEP_BY_NAME = {s["name"]: s for s in STEPS}

NOT_RUN_HERE_NOTE = (
    "s02b_validate_locations.py is NEVER run by run_all.py: it is an interactive coordinate-validation "
    "tool (ipywidgets + ipyleaflet), written in jupytext percent format. Open it with "
    "`jupytext --to ipynb pipeline/s02b_validate_locations.py` if you need to re-run it by hand."
)


# -----------------------------------------------------------------------------
# argv plumbing
# -----------------------------------------------------------------------------
# Every step module parses its own CLI flags with argparse against
# sys.argv, and main() takes no parameters. To run a step in-process with a
# specific set of flags (rather than re-invoking the interpreter), sys.argv
# is swapped for the duration of the call and restored afterwards.


class _TemporaryArgv:
    def __init__(self, argv: list[str]):
        self._argv = argv
        self._saved: list[str] | None = None

    def __enter__(self) -> "_TemporaryArgv":
        self._saved = sys.argv
        sys.argv = [self._saved[0]] + self._argv
        return self

    def __exit__(self, *exc_info) -> None:
        assert self._saved is not None
        sys.argv = self._saved


def _build_argv(step_name: str, use_cache: bool) -> list[str]:
    """Translate this script's own --use-cache state into the per-step flag
    that reproduces it, for the steps that have one (s04, s05). Steps
    without routing/geocoding flags are always run with their defaults.
    """
    if step_name in ("s04_route_ors", "s05_route_osrm"):
        return ["--use-cache"] if use_cache else ["--no-use-cache"]
    return []


def run_step(step: dict, use_cache: bool) -> None:
    module = importlib.import_module(step["module"])
    argv = _build_argv(step["name"], use_cache) if step.get("cli") else []
    with _TemporaryArgv(argv):
        module.main()


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def print_step_list() -> None:
    print("Pipeline step order (dependency order, not numeric filename order):")
    print()
    for i, step in enumerate(STEPS, start=1):
        print(f"{i:2d}. {step['name']}")
        print(f"      in:  {step['inputs']}")
        print(f"      out: {step['outputs']}")
        print(f"      {step['note']}")
        print()
    print(NOT_RUN_HERE_NOTE)


def print_summary(results: list[tuple[str, str, float]]) -> None:
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    name_width = max((len(name) for name, _, _ in results), default=4)
    print(f"{'step'.ljust(name_width)}  {'status'.ljust(8)}  seconds")
    print(f"{'-' * name_width}  {'-' * 8}  -------")
    for name, status, elapsed in results:
        print(f"{name.ljust(name_width)}  {status.ljust(8)}  {elapsed:7.1f}")
    print("=" * 60)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--use-cache",
        dest="use_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Never call ORS/OSRM/Google; reuse the frozen route caches (default: True). "
        "This is the documented way to reproduce the published results with no API keys.",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Synonym for --use-cache (never call any external API). Kept because the brief names it.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STEP",
        choices=STEP_NAMES,
        help="Run just these steps, in the pipeline's dependency order (not the order given on the command line).",
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        metavar="STEP",
        choices=STEP_NAMES,
        help="Start at this step and continue through the rest of the pipeline.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the ordered step list with inputs/outputs and exit.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going after a failing step (default: stop at the first failure).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.list:
        print_step_list()
        return 0

    use_cache = True if args.skip_api else args.use_cache

    if args.only:
        wanted = set(args.only)
        selected = [s for s in STEPS if s["name"] in wanted]
    elif args.from_step:
        start = STEP_NAMES.index(args.from_step)
        selected = STEPS[start:]
    else:
        selected = STEPS

    print("=" * 60)
    print(f"Running {len(selected)} step(s), use_cache={use_cache}")
    print("=" * 60)

    results: list[tuple[str, str, float]] = []
    for step in selected:
        name = step["name"]
        print(f"\n--- {name} " + "-" * max(0, 56 - len(name)))
        start = time.time()
        try:
            run_step(step, use_cache)
            elapsed = time.time() - start
            results.append((name, "OK", elapsed))
        except Exception:
            elapsed = time.time() - start
            results.append((name, "FAILED", elapsed))
            print(f"\nERROR: step '{name}' raised an exception.", file=sys.stderr)
            traceback.print_exc()
            if args.continue_on_error:
                continue
            print_summary(results)
            raise

    print_summary(results)
    return 1 if any(status == "FAILED" for _, status, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
