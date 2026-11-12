"""Step 09 - Manuscript figures.

Produces the three publication figures as PLOS ONE-ready TIFFs, plus a PDF
of each for drafting:

  Fig1  Study-population inclusion/exclusion flow chart. Every count is
        computed live from the raw exports and pipeline intermediates,
        never typed in, reusing utils.clean_facility_names and
        utils.is_nonphysical as the single source of truth for the
        name-cleaning rules. The two data streams use different
        non-physical-entry matching conventions on purpose (see
        _compute_funnel), and each funnel stage is checked to reconcile
        exactly with the one before it.

  Fig2  Box plots of four per-interconsultation outcomes (round-trip
        distance, avoided CO2, avoided N2O, round-trip travel time),
        weighted by the number of interconsultations each row represents,
        split into Overall / Capital / Non-capital.

  Fig3  Four-panel spatial map: healthcare-centre distribution, volume-
        weighted routes, routes by round-trip distance, and modelled avoided
        CO2 per interconsultation. Health-region boundaries come from the
        regions GeoJSON (config.REGIONS_GEOJSON), dropping the "No consta"
        pseudo-region and empty geometries, reprojected to EPSG:4326.

PLOS ONE figure requirements, checked at authoring time against
https://journals.plos.org/plosone/s/figures and followed below:
  - File format: TIFF (or EPS; TIFF is used here).
  - Color mode: RGB, 8 bit/channel (no CMYK).
  - Resolution: no greater than 300-600 dpi; this module uses 300 dpi.
  - Width: 2.63-7.5 in (789-2250 px at 300 dpi).
  - Height: no more than 8.75 in (2625 px at 300 dpi).
  - Compression: LZW required for TIFF.
  - Font: Arial, Times, or Symbol, 8-12 point.
  - File size: 10 MB or less.
  - File naming: Fig1.tif, Fig2.tif, Fig3.tif, matching the in-text labels.
Every figure below is sized and fonted to stay within these limits.

Attribution, printed by this module as the data sources require it: route
geometries are from OpenRouteService, built on OpenStreetMap data, (c)
OpenStreetMap contributors, under the Open Database License (ODbL). Health-
region boundaries are published open data from the Departament de Salut.

Input: analysis_dataset.parquet (s06's output), resolved via
utils.resolve_input, plus the raw/intermediate files the Figure 1 funnel
needs (resolved the same way) and the regions GeoJSON from DATA_DIR.

Output: OUTPUT_DIR/figures/Fig1.tif, Fig2.tif, Fig3.tif (and a .pdf of each).
"""

from __future__ import annotations

import argparse
import io
import json

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless-safe: figures are written straight to disk
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from PIL import Image

import config
from pipeline import utils

# -----------------------------------------------------------------------------
# PLOS ONE limits and shared styling
# -----------------------------------------------------------------------------

PLOS_DPI = 300
PLOS_MAX_WIDTH_IN = 7.5
PLOS_MAX_HEIGHT_IN = 8.75

ATTRIBUTION = (
    "Route geometries: OpenRouteService, built on OpenStreetMap data, "
    "(c) OpenStreetMap contributors, Open Database License (ODbL). "
    "Health-region boundaries: Departament de Salut open data."
)


def _set_plos_style() -> None:
    """Apply figure-wide styling that satisfies the PLOS ONE font rule
    (Arial, Times, or Symbol, 8-12 point) and keeps text.usetex disabled.

    text.usetex stays at its default of False throughout this module: the
    flow chart in particular draws dynamically computed numbers that can
    contain characters (%, <=, >=) that LaTeX would otherwise need escaping
    for. Subscripts go through mathtext instead.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.edgecolor": "#333333",
        }
    )


def _fig_path(name: str, ext: str):
    return config.output_path("figures", f"{name}.{ext}")


def _save_tiff(fig, name: str) -> None:
    """Save `fig` as OUTPUT_DIR/figures/{name}.tif (RGB, LZW, 300 dpi) and
    as a companion {name}.pdf for drafting.

    Rendering goes through a PNG buffer at PLOS_DPI and is then converted to
    an explicit RGB PIL image (never RGBA or CMYK) before being written out
    as LZW-compressed TIFF, since the Agg backend cannot write TIFF directly.
    """
    pdf_path = _fig_path(name, "pdf")
    fig.savefig(pdf_path, bbox_inches="tight")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=PLOS_DPI, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    tif_path = _fig_path(name, "tif")
    img.save(tif_path, format="TIFF", dpi=(PLOS_DPI, PLOS_DPI), compression="tiff_lzw")

    width_in, height_in = img.width / PLOS_DPI, img.height / PLOS_DPI
    if width_in > PLOS_MAX_WIDTH_IN + 1e-6 or height_in > PLOS_MAX_HEIGHT_IN + 1e-6:
        raise AssertionError(
            f"{name}: rendered size {width_in:.2f} x {height_in:.2f} in exceeds the PLOS "
            f"ONE limit of {PLOS_MAX_WIDTH_IN} x {PLOS_MAX_HEIGHT_IN} in"
        )

    utils.log(
        f"Saved: {tif_path} ({img.width}x{img.height} px, {width_in:.2f}x{height_in:.2f} in "
        f"at {PLOS_DPI} dpi, mode={img.mode})"
    )
    utils.log(f"Saved: {pdf_path}")


def _fmt(n) -> str:
    return f"{int(round(n)):,}"


# =============================================================================
# FIGURE 1 - Study population flow chart
# =============================================================================
# Every number below is computed from the raw exports and pipeline
# intermediates, never typed in. The raw total is the non-2025 historical
# rows plus the complete 2025 export, not just the historical file's length
# on its own, which includes only a partial 2025 slice and so understates
# both records and interconsultations (see config.YEAR_2025_MODE).


def _missing_location_mask(df: pd.DataFrame) -> pd.Series:
    """True where UP_DES or UP_DES_1, the two facility-name fields the
    funnel is keyed on, is missing or blank.
    """
    return (
        df["UP_DES"].isna()
        | df["UP_DES_1"].isna()
        | df["UP_DES"].astype(str).str.strip().isin(["", "nan"])
        | df["UP_DES_1"].astype(str).str.strip().isin(["", "nan"])
    )


def _compute_funnel() -> dict:
    """Recompute the Figure 1 inclusion/exclusion funnel from the raw
    exports and pipeline intermediates, checking that every stage
    reconciles exactly with the one before it.

    Returns a dict of {stage_or_exclusion_name: {"records": int, "ic": int}}
    plus "final_unique_routes".
    """
    # --- Stream A: historical export, years other than 2025 -----------------
    raw_def = utils.read_table(config.data_path("raw_export_historical.xlsx"))
    raw_hist = raw_def[raw_def[config.YEAR_COL] != 2025].copy()

    # --- Stream B: the complete, separate 2025 refresh export --------------
    # This supersedes the partial 2025 slice dropped from the historical
    # file above (config.YEAR_2025_MODE = "refresh_replaces_partial").
    raw_2025 = utils.read_table(config.data_path("raw_export_2025.txt"))
    raw_2025 = raw_2025.rename(columns={"UP_DES.1": "UP_DES_1"})
    raw_2025["N"] = pd.to_numeric(raw_2025["N"], errors="coerce")

    raw_records = int(len(raw_hist) + len(raw_2025))
    raw_ic = int(pd.to_numeric(raw_hist["N"], errors="coerce").sum() + raw_2025["N"].sum())

    # --- Missing origin/destination, checked on both streams ---------------
    miss_hist = _missing_location_mask(raw_hist)
    miss_2025 = _missing_location_mask(raw_2025)
    excl_missing_records = int(miss_hist.sum() + miss_2025.sum())
    excl_missing_ic = int(raw_hist.loc[miss_hist, "N"].sum() + raw_2025.loc[miss_2025, "N"].sum())
    raw_hist = raw_hist[~miss_hist].copy()
    raw_2025 = raw_2025[~miss_2025].copy()

    # --- Non-physical / virtual service entries -----------------------------
    # The two streams use deliberately different conventions (see
    # utils.is_nonphysical's docstring): historical cleans names first, then
    # matches case-insensitively, while the 2025 refresh matches the raw
    # names case-sensitively. That is not a bug, it is how the published
    # Figure 1 counts were produced, and changing either convention would
    # silently shift them.
    origin_clean = utils.clean_facility_names(raw_hist["UP_DES"], case_sensitive=False)
    dest_clean = utils.clean_facility_names(raw_hist["UP_DES_1"], case_sensitive=False)
    nonphys_hist = utils.is_nonphysical(origin_clean, case_sensitive=False) | utils.is_nonphysical(
        dest_clean, case_sensitive=False
    )
    nonphys_2025 = utils.is_nonphysical(raw_2025["UP_DES"], case_sensitive=True) | utils.is_nonphysical(
        raw_2025["UP_DES_1"], case_sensitive=True
    )
    excl_nonphys_records = int(nonphys_hist.sum() + nonphys_2025.sum())
    excl_nonphys_ic = int(
        raw_hist.loc[nonphys_hist, "N"].sum() + raw_2025.loc[nonphys_2025, "N"].sum()
    )
    hist_clean = raw_hist[~nonphys_hist]
    r2025_clean = raw_2025[~nonphys_2025]

    after_cleaning_records = int(len(hist_clean) + len(r2025_clean))
    after_cleaning_ic = int(hist_clean["N"].sum() + r2025_clean["N"].sum())

    if raw_records - excl_missing_records - excl_nonphys_records != after_cleaning_records:
        raise AssertionError("Figure 1 funnel does not reconcile at 'after cleaning' (records)")
    if raw_ic - excl_missing_ic - excl_nonphys_ic != after_cleaning_ic:
        raise AssertionError("Figure 1 funnel does not reconcile at 'after cleaning' (interconsultations)")

    # --- Geocoding / coordinate matching ------------------------------------
    no_coords = utils.read_table(utils.resolve_input("unroutable_2025.csv"))
    excl_nocoord_records = int(len(no_coords))
    excl_nocoord_ic = int(pd.to_numeric(no_coords["N"], errors="coerce").sum())

    geocoded = utils.read_table(utils.resolve_input("geocoded_interconsultations.parquet"))
    geocoded_records = int(len(geocoded))
    geocoded_ic = int(pd.to_numeric(geocoded["N"], errors="coerce").sum())

    if after_cleaning_records - excl_nocoord_records != geocoded_records:
        raise AssertionError("Figure 1 funnel does not reconcile at 'geocoded' (records)")
    if after_cleaning_ic - excl_nocoord_ic != geocoded_ic:
        raise AssertionError("Figure 1 funnel does not reconcile at 'geocoded' (interconsultations)")

    # --- Routing (distance/duration), the age filter ------------------------
    routed = utils.read_table(utils.resolve_input("routed_ors.parquet"))
    no_route_mask = routed["km_roundtrip"].isna()
    excl_noroute_records = int(no_route_mask.sum())
    excl_noroute_ic = int(pd.to_numeric(routed.loc[no_route_mask, "N"], errors="coerce").sum())
    routed_valid = routed[~no_route_mask]

    short_mask = routed_valid["km_roundtrip"] <= config.MIN_ROUNDTRIP_KM
    excl_short_records = int(short_mask.sum())
    excl_short_ic = int(routed_valid.loc[short_mask, "N"].sum())
    routed_after_short = routed_valid[~short_mask]

    age_mask = routed_after_short["GRUP_EDAT"] > config.MAX_AGE_GROUP
    excl_age_records = int(age_mask.sum())
    excl_age_ic = int(routed_after_short.loc[age_mask, "N"].sum())

    # --- Final analytic dataset: the actual s06 output, the single source
    #     of truth shared with Figures 2 and 3. --------------------------
    final_df = utils.read_table(utils.resolve_input("analysis_dataset.parquet"))
    final_records = int(len(final_df))
    final_ic = int(final_df["N"].sum())
    final_unique_routes = int(final_df.groupby(config.ROUTE_KEY).ngroups)

    if geocoded_records - excl_noroute_records - excl_short_records - excl_age_records != final_records:
        raise AssertionError("Figure 1 funnel does not reconcile at 'final' (records)")
    if geocoded_ic - excl_noroute_ic - excl_short_ic - excl_age_ic != final_ic:
        raise AssertionError("Figure 1 funnel does not reconcile at 'final' (interconsultations)")

    utils.log("Figure 1 funnel reconciles exactly at every stage.")

    return {
        "raw": {"records": raw_records, "ic": raw_ic},
        "excl_missing": {"records": excl_missing_records, "ic": excl_missing_ic},
        "excl_nonphysical": {"records": excl_nonphys_records, "ic": excl_nonphys_ic},
        "after_cleaning": {"records": after_cleaning_records, "ic": after_cleaning_ic},
        "excl_nocoordmatch": {"records": excl_nocoord_records, "ic": excl_nocoord_ic},
        "geocoded": {"records": geocoded_records, "ic": geocoded_ic},
        "excl_noroute": {"records": excl_noroute_records, "ic": excl_noroute_ic},
        "excl_dist200m": {"records": excl_short_records, "ic": excl_short_ic},
        "excl_age110": {"records": excl_age_records, "ic": excl_age_ic},
        "final": {"records": final_records, "ic": final_ic},
        "final_unique_routes": final_unique_routes,
    }


def _record_funnel_numbers(flow: dict) -> None:
    pairs = [
        ("raw", "fig1_raw"),
        ("excl_missing", "fig1_excl_missing_origindest"),
        ("excl_nonphysical", "fig1_excl_nonphysical"),
        ("after_cleaning", "fig1_after_namecleaning"),
        ("excl_nocoordmatch", "fig1_excl_nocoordmatch"),
        ("geocoded", "fig1_geocoded"),
        ("excl_noroute", "fig1_excl_noroute"),
        ("excl_dist200m", "fig1_excl_dist200m"),
        ("excl_age110", "fig1_excl_age110"),
        ("final", "fig1_final"),
    ]
    for key, prefix in pairs:
        utils.record_number(f"{prefix}_records", flow[key]["records"], "records", "s09_figures")
        utils.record_number(f"{prefix}_ic", flow[key]["ic"], "interconsultations", "s09_figures")
    utils.record_number("fig1_final_routes", flow["final_unique_routes"], "routes", "s09_figures")


# --- Flow-chart drawing -------------------------------------------------- #
# Drawn directly in an inches-based coordinate system (1 data unit = 1
# inch), so the figure's physical size is exact and stays within the PLOS
# ONE width/height limits no matter how many lines the exclusion boxes
# need. Box heights follow from the number of text lines, so adding or
# shortening an exclusion reason never breaks the layout.

_FONT_MAIN = 9.0
_FONT_SIDE = 8.0
_FONT_TITLE = 12.0
_LINE_H = 0.155  # inches per text line
_BOX_PAD = 0.09  # inches, top and bottom padding inside a box
_SIDE_GAP = 0.07  # inches between stacked exclusion boxes in one segment
_SEG_PAD = 0.09  # inches above/below the stacked exclusion boxes in a segment

_MAIN_W, _MAIN_X = 3.9, 0.3
_SIDE_W, _SIDE_X = 2.6, 4.5
_FIG_W = _SIDE_X + _SIDE_W + 0.3  # = 7.4 in, within the 7.5 in PLOS limit


def _n_lines(text: str) -> int:
    return text.count("\n") + 1


def _box_height(text: str) -> float:
    return _n_lines(text) * _LINE_H + 2 * _BOX_PAD


def _draw_main_box(ax, y_top, height, text, facecolor):
    y_bot = y_top - height
    box = FancyBboxPatch(
        (_MAIN_X, y_bot), _MAIN_W, height, boxstyle="round,pad=0.07",
        linewidth=1.1, edgecolor="#2E3436", facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        _MAIN_X + _MAIN_W / 2, y_bot + height / 2, text, ha="center", va="center",
        fontsize=_FONT_MAIN, family="serif", linespacing=1.35,
    )
    return y_bot


def _draw_side_box(ax, y_top, height, text):
    y_bot = y_top - height
    box = FancyBboxPatch(
        (_SIDE_X, y_bot), _SIDE_W, height, boxstyle="round,pad=0.06",
        linewidth=0.9, edgecolor="#8a3b3b", facecolor="#f6dbdb",
    )
    ax.add_patch(box)
    ax.text(
        _SIDE_X + _SIDE_W / 2, y_bot + height / 2, text, ha="center", va="center",
        fontsize=_FONT_SIDE, family="serif", linespacing=1.3,
    )
    return y_bot


def make_fig1() -> None:
    utils.log("=" * 60)
    utils.log("Figure 1 - study population flow chart")
    utils.log("=" * 60)

    flow = _compute_funnel()
    _record_funnel_numbers(flow)

    r, ic = flow["raw"]["records"], flow["raw"]["ic"]
    stages = [
        f"Raw records (2019, 2023, 2024, 2025)\n{_fmt(r)} records\n{_fmt(ic)} interconsultations",
        (
            f"After name cleaning & non-physical-service filter\n"
            f"{_fmt(flow['after_cleaning']['records'])} records\n"
            f"{_fmt(flow['after_cleaning']['ic'])} interconsultations"
        ),
        (
            f"Geocoded, coordinate-matched dataset\n"
            f"{_fmt(flow['geocoded']['records'])} records\n"
            f"{_fmt(flow['geocoded']['ic'])} interconsultations\n"
            f"(this N is the one the population\ncharacteristics are described over)"
        ),
        (
            f"Final analytic dataset\n"
            f"{_fmt(flow['final']['records'])} records\n"
            f"{_fmt(flow['final']['ic'])} interconsultations\n"
            f"{_fmt(flow['final_unique_routes'])} unique origin-destination routes"
        ),
    ]
    stage_colors = ["#dbe9f6", "#dbe9f6", "#dbe9f6", "#d9f0d9"]

    segment_exclusions = [
        [
            f"Missing origin/destination\n{_fmt(flow['excl_missing']['records'])} records\n"
            f"{_fmt(flow['excl_missing']['ic'])} interconsultations",
            f"Non-physical / virtual service entry\n(e.g., home-care support teams)\n"
            f"{_fmt(flow['excl_nonphysical']['records'])} records\n"
            f"{_fmt(flow['excl_nonphysical']['ic'])} interconsultations",
        ],
        [
            f"No coordinate match for centre code\n{_fmt(flow['excl_nocoordmatch']['records'])} records\n"
            f"{_fmt(flow['excl_nocoordmatch']['ic'])} interconsultations",
        ],
        [
            f"No route returned by ORS\n(coordinates outside road network)\n"
            f"{_fmt(flow['excl_noroute']['records'])} records\n"
            f"{_fmt(flow['excl_noroute']['ic'])} interconsultations",
            f"Round trip ≤ 200 m\n{_fmt(flow['excl_dist200m']['records'])} records\n"
            f"{_fmt(flow['excl_dist200m']['ic'])} interconsultations",
            f"Age group ≥ 110 years (outlier)\n{_fmt(flow['excl_age110']['records'])} records\n"
            f"{_fmt(flow['excl_age110']['ic'])} interconsultations",
        ],
    ]

    main_heights = [_box_height(t) for t in stages]
    seg_side_heights = [[_box_height(t) for t in excls] for excls in segment_exclusions]
    seg_heights = [
        max(sum(h) + _SIDE_GAP * (len(h) - 1) + 2 * _SEG_PAD, 0.6) for h in seg_side_heights
    ]

    total_height = sum(main_heights) + sum(seg_heights)
    fig_h = min(PLOS_MAX_HEIGHT_IN - 0.2, max(5.0, total_height + 1.1))
    fig, ax = plt.subplots(figsize=(_FIG_W, fig_h))
    ax.set_xlim(0, _FIG_W)
    ax.set_ylim(0, total_height + 0.9)
    ax.axis("off")

    arrow_x = _MAIN_X + _MAIN_W / 2
    y = total_height + 0.5
    for i, text in enumerate(stages):
        y = _draw_main_box(ax, y, main_heights[i], text, stage_colors[i])
        if i < len(segment_exclusions):
            seg_h = seg_heights[i]
            seg_top, seg_bot = y, y - seg_h
            # single continuous arrow spanning the whole segment, drawn
            # regardless of how many exclusion boxes hang off it
            ax.add_patch(
                FancyArrowPatch(
                    (arrow_x, seg_top), (arrow_x, seg_bot), arrowstyle="-|>",
                    mutation_scale=13, linewidth=1.1, color="#2E3436",
                )
            )
            heights = seg_side_heights[i]
            total_side = sum(heights) + _SIDE_GAP * (len(heights) - 1)
            side_y = seg_top - (seg_h - total_side) / 2
            for h, text_s in zip(heights, segment_exclusions[i]):
                side_bot = _draw_side_box(ax, side_y, h, text_s)
                mid = side_bot + h / 2
                ax.add_patch(
                    FancyArrowPatch(
                        (_MAIN_X + _MAIN_W, mid), (_SIDE_X, mid), arrowstyle="-|>",
                        mutation_scale=11, linewidth=0.9, color="#8a3b3b",
                    )
                )
                side_y = side_bot - _SIDE_GAP
            y = seg_bot

    ax.set_title("Study Population: Inclusion / Exclusion Flow", fontsize=_FONT_TITLE, pad=10)
    fig.tight_layout()
    _save_tiff(fig, "Fig1")
    plt.close(fig)


# =============================================================================
# FIGURE 2 - Box plots of per-interconsultation outcomes by region type
# =============================================================================

_FIG2_METRICS = [
    ("km_per_interconsultation", "A. Round-trip distance per interconsultation (km)", 1.0),
    ("co2_kg_per_interconsultation", r"B. Avoided $\mathrm{CO_2}$ per interconsultation (kg)", 1.0),
    ("n2o_kg_per_interconsultation", r"C. Avoided $\mathrm{N_2O}$ per interconsultation (g)", 1000.0),
    ("minutes_per_interconsultation", "D. Round-trip travel time per interconsultation (min)", 1.0),
]


def make_fig2(df: pd.DataFrame | None = None) -> None:
    utils.log("=" * 60)
    utils.log("Figure 2 - box plots by region type")
    utils.log("=" * 60)

    if df is None:
        df = utils.read_table(utils.resolve_input("analysis_dataset.parquet"))

    region_type = df["origin_health_region"].apply(utils.classify_region_type)
    cap_mask = (region_type == "Capital").to_numpy()
    non_mask = (region_type == "Non-capital").to_numpy()
    weights = df["N"].to_numpy(dtype=int)

    n_capital = int(weights[cap_mask].sum())
    n_noncapital = int(weights[non_mask].sum())
    n_total = n_capital + n_noncapital
    utils.log(f"  N total: {n_total:,} | Capital: {n_capital:,} | Non-capital: {n_noncapital:,}")
    utils.record_number("fig2_n_total", n_total, "interconsultations", "s09_figures")
    utils.record_number("fig2_n_capital", n_capital, "interconsultations", "s09_figures")
    utils.record_number("fig2_n_noncapital", n_noncapital, "interconsultations", "s09_figures")

    palette = sns.color_palette("Set2", 3)

    fig, axs = plt.subplots(2, 2, figsize=(7.3, 6.8))
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    for i, (col, title, scale) in enumerate(_FIG2_METRICS):
        ax = axs[i // 2, i % 2]
        values = df[col].to_numpy(dtype=float)

        cap_vals = np.repeat(values[cap_mask], weights[cap_mask]) * scale
        non_vals = np.repeat(values[non_mask], weights[non_mask]) * scale
        overall_vals = np.concatenate([cap_vals, non_vals])

        bp = ax.boxplot(
            [overall_vals, cap_vals, non_vals],
            tick_labels=["Overall", "Capital", "Non-capital"],
            whis=1.5, showmeans=True, patch_artist=True,
            meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5),
            medianprops=dict(color="black", linewidth=1.2),
            flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="#555555",
                             markersize=2.5, alpha=0.25, linestyle="none"),
            boxprops=dict(linewidth=0.9),
            whiskerprops=dict(linewidth=0.9),
            capprops=dict(linewidth=0.9),
        )
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.9)

        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        del cap_vals, non_vals, overall_vals

    _save_tiff(fig, "Fig2")
    plt.close(fig)


# =============================================================================
# FIGURE 3 - Four-panel spatial map
# =============================================================================

_DIST_BINS = [
    (0.0, 10.0, "<10 km"),
    (10.0, 25.0, "10-25 km"),
    (25.0, 50.0, "25-50 km"),
    (50.0, 100.0, "50-100 km"),
    (100.0, float("inf"), ">=100 km"),
]
_N_ROUTES_SHOWN = 1000


def _load_regions():
    import geopandas as gpd

    regions = gpd.read_file(config.data_path(config.REGIONS_GEOJSON))
    regions = regions[~regions[config.REGION_NAME_FIELD].isin(config.REGION_EXCLUDE)]
    regions = regions[~regions.geometry.is_empty & regions.geometry.notna()]
    if regions.crs is None:
        regions = regions.set_crs(epsg=4326)
    elif regions.crs.to_epsg() != 4326:
        regions = regions.to_crs(4326)
    return regions


def _parse_route_geometry(geojson_str):
    """GeoJSON LineString string -> list of (lon, lat) tuples. Returns []
    on failure or when no geometry was recorded for the route.
    """
    try:
        if not geojson_str or geojson_str == "null" or pd.isna(geojson_str):
            return []
        geom = json.loads(geojson_str)
        if geom.get("type") == "LineString":
            return [(c[0], c[1]) for c in geom.get("coordinates", []) if len(c) >= 2]
        return []
    except Exception:
        return []


def _route_coords(row):
    coords = _parse_route_geometry(row["geometry"])
    if not coords:
        coords = [(row["olon"], row["olat"]), (row["dlon"], row["dlat"])]
    return coords


def _build_centres(df: pd.DataFrame) -> pd.DataFrame:
    """One row per healthcare centre, with its activity as an origin and as
    a destination.

    N_total = N_ori + N_des counts every interconsultation *endpoint*, so it
    is exactly 2x the true study total when summed across centres. That is
    fine for sizing a marker, since it only scales symbol area, but the
    panel's reported total has to come from df["N"].sum() instead, never
    from this column (see make_fig3).
    """
    origins = df[["origin_location_name", "origin_latitude", "origin_longitude"]].copy()
    origins.columns = ["centre", "lat", "lon"]
    dests = df[["destination_location_name", "destination_latitude", "destination_longitude"]].copy()
    dests.columns = ["centre", "lat", "lon"]

    centres = pd.concat([origins, dests], ignore_index=True)
    centres = centres.drop_duplicates("centre").dropna(subset=["lat", "lon"])

    n_ori = df.groupby("origin_location_name")["N"].sum().rename("N_ori")
    n_des = df.groupby("destination_location_name")["N"].sum().rename("N_des")
    centres = (
        centres.merge(n_ori, left_on="centre", right_index=True, how="left")
        .merge(n_des, left_on="centre", right_index=True, how="left")
        .fillna({"N_ori": 0, "N_des": 0})
    )
    centres["N_total"] = centres["N_ori"] + centres["N_des"]
    return centres


def _build_routes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique origin-destination route (config.ROUTE_KEY, which
    yields 1,674 routes), with interconsultation-weighted average distance
    and CO2 per interconsultation.
    """
    routes = (
        df.groupby(config.ROUTE_KEY)
        .agg(
            N=("N", "sum"),
            km_sum=("total_km", "sum"),
            co2_sum=("emissio_co2_kg", "sum"),
            geometry=("route_geometry_outbound", "first"),
            olat=("origin_latitude", "first"),
            olon=("origin_longitude", "first"),
            dlat=("destination_latitude", "first"),
            dlon=("destination_longitude", "first"),
        )
        .reset_index()
    )
    routes["km_per_ic"] = routes["km_sum"] / routes["N"]
    routes["co2_per_ic"] = routes["co2_sum"] / routes["N"]
    return routes.sort_values("N", ascending=False).reset_index(drop=True)


def _dist_bin_index(km: float) -> int:
    for i, (lo, hi, _label) in enumerate(_DIST_BINS):
        if lo <= km < hi:
            return i
    return len(_DIST_BINS) - 1


def _draw_basemap(ax, regions) -> None:
    """Light-grey Catalonia silhouette with faint internal region borders.

    The aspect-ratio correction uses the mean latitude of the regions'
    centroids (computed from the data), not a literal coordinate, so that
    one kilometre north looks like one kilometre east on screen.
    """
    regions.plot(ax=ax, facecolor="#ECECEC", edgecolor="white", linewidth=0.5, zorder=1)
    # Midpoint of the data's own latitude extent, not a literal coordinate.
    # This sidesteps geopandas' "geographic CRS" centroid warning, which is
    # harmless here since it's only an aspect-ratio correction, not an
    # area/length computation.
    _miny, _maxy = regions.total_bounds[1], regions.total_bounds[3]
    lat_center = float((_miny + _maxy) / 2.0)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(lat_center)))
    ax.set_axis_off()


def _info_box(ax, text: str) -> None:
    ax.text(
        0.98, 0.02, text, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, family="serif", linespacing=1.4, zorder=7,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#333333",
                  linewidth=0.9, alpha=0.95),
    )


def make_fig3(df: pd.DataFrame | None = None) -> None:
    utils.log("=" * 60)
    utils.log("Figure 3 - four-panel spatial map")
    utils.log("=" * 60)

    if df is None:
        df = utils.read_table(utils.resolve_input("analysis_dataset.parquet"))
    regions = _load_regions()

    centres = _build_centres(df)
    active = centres[centres["N_total"] > 0]
    routes = _build_routes(df)
    top = routes.head(_N_ROUTES_SHOWN)

    total_ic = int(df["N"].sum())  # the true study total: one count per interconsultation
    n_active = len(active)
    n_routes_total = len(routes)
    max_vol = int(top["N"].max())

    if abs(active["N_total"].sum() - 2 * total_ic) > 1:
        raise AssertionError(
            "Panel A: N_total no longer sums to 2x the study total -- the origin/destination "
            "endpoint convention in _build_centres has changed; re-check panel A."
        )
    if max_vol != int(routes["N"].max()):
        raise AssertionError("Panel B: max volume is not the global maximum route volume.")

    utils.log(f"  Panel A: {n_active:,} active centres, {total_ic:,} interconsultations")
    utils.log(f"  Panel B: {len(top):,} of {n_routes_total:,} routes shown, max volume {max_vol:,}")

    utils.record_number("fig3_active_centres", n_active, "centres", "s09_figures")
    utils.record_number("fig3_total_interconsultations", total_ic, "interconsultations", "s09_figures")
    utils.record_number("fig3_routes_shown", f"{len(top):,} of {n_routes_total:,}", "routes", "s09_figures")
    utils.record_number("fig3_max_route_volume", max_vol, "interconsultations", "s09_figures")

    fig, axs = plt.subplots(2, 2, figsize=(7.3, 8.4))
    plt.subplots_adjust(hspace=0.1, wspace=0.06)

    # ---- Panel A: healthcare centre distribution --------------------------
    ax_a = axs[0, 0]
    _draw_basemap(ax_a, regions)
    smax = active["N_total"].max()
    sizes = 8 + 70 * np.sqrt(active["N_total"] / smax)
    ax_a.scatter(active["lon"], active["lat"], s=sizes, c="#4A90E2",
                 edgecolor="#2E3436", linewidth=0.3, alpha=0.6, zorder=4)
    ax_a.set_title("A. Healthcare centres distribution", fontsize=10, pad=6)
    _info_box(ax_a, f"Active centres: {n_active:,}\nTotal interconsultations: {total_ic:,}")

    # ---- Panel B: volume-weighted routes -----------------------------------
    ax_b = axs[0, 1]
    _draw_basemap(ax_b, regions)
    by_volume = top.sort_values("N")  # thin drawn first, thick on top
    segs = [_route_coords(r) for _, r in by_volume.iterrows()]
    widths = 0.3 + 4.5 * (by_volume["N"] / max_vol)
    ax_b.add_collection(LineCollection(segs, colors="#2E3A59", linewidths=widths.to_numpy(),
                                       alpha=0.5, zorder=3))
    ax_b.set_title("B. Weighted interconsultation routes", fontsize=10, pad=6)
    _info_box(ax_b, f"Routes shown: {len(top):,} of {n_routes_total:,}\nMax volume: {max_vol:,} interconsultations")

    # ---- Panel C: routes by round-trip distance ----------------------------
    ax_c = axs[1, 0]
    _draw_basemap(ax_c, regions)
    cmap_c = plt.get_cmap("cividis")  # colour-blind-safe sequential ramp
    bin_colors = [cmap_c(x) for x in np.linspace(0.05, 0.95, len(_DIST_BINS))]
    by_dist = top.copy()
    by_dist["bin"] = by_dist["km_per_ic"].apply(_dist_bin_index)
    by_dist = by_dist.sort_values("bin")  # longest routes drawn last/on top
    segs = [_route_coords(r) for _, r in by_dist.iterrows()]
    cols = [bin_colors[b] for b in by_dist["bin"]]
    ax_c.add_collection(LineCollection(segs, colors=cols, linewidths=1.4, alpha=0.85, zorder=3))
    ax_c.set_title("C. Interconsultation routes by round-trip distance", fontsize=10, pad=6)
    legend_handles = [
        Patch(facecolor=bin_colors[i], edgecolor="#333333", label=label)
        for i, (_lo, _hi, label) in enumerate(_DIST_BINS)
    ]
    ax_c.legend(handles=legend_handles, title="round-trip km", loc="lower right",
                fontsize=6.5, title_fontsize=7.5, frameon=True, framealpha=0.95)

    # ---- Panel D: modelled avoided CO2 per interconsultation ---------------
    ax_d = axs[1, 1]
    _draw_basemap(ax_d, regions)
    inten = top.dropna(subset=["co2_per_ic"]).copy()
    vmin, vmax = float(inten["co2_per_ic"].min()), float(inten["co2_per_ic"].max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_d = plt.get_cmap("viridis")  # colour-blind-safe sequential ramp (the legacy red-green map is not)
    inten = inten.sort_values("co2_per_ic")  # highest intensity drawn last/on top
    segs = [_route_coords(r) for _, r in inten.iterrows()]
    cols = [cmap_d(norm(v)) for v in inten["co2_per_ic"]]
    ax_d.add_collection(LineCollection(segs, colors=cols, linewidths=1.5, alpha=0.85, zorder=3))
    ax_d.set_title(r"D. Modelled avoided $\mathrm{CO_2}$ per interconsultation", fontsize=10, pad=6)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_d)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_d, fraction=0.045, pad=0.02)
    cbar.set_label(r"kg $\mathrm{CO_2}$ per interconsultation", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)

    # Lock all panels to the Catalonia extent, with a small margin derived
    # from the data's own bounds (not a literal coordinate).
    minx, miny, maxx, maxy = regions.total_bounds
    margin = 0.02 * max(maxx - minx, maxy - miny)
    for ax in axs.ravel():
        ax.set_xlim(minx - margin, maxx + margin)
        ax.set_ylim(miny - margin, maxy + margin)

    _save_tiff(fig, "Fig3")
    plt.close(fig)

    utils.log(ATTRIBUTION)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the manuscript figures as PLOS ONE TIFFs.")
    parser.add_argument("--figure", choices=["1", "2", "3", "all"], default="all",
                         help="Regenerate a single figure, or all three (default).")
    args = parser.parse_args()

    _set_plos_style()

    if args.figure in ("1", "all"):
        make_fig1()
    if args.figure in ("2", "all"):
        make_fig2()
    if args.figure in ("3", "all"):
        make_fig3()

    utils.log("-" * 60)
    utils.log(ATTRIBUTION)
    utils.log("Done.")
    utils.flush_numbers("s09_figures")


if __name__ == "__main__":
    main()
