# Environmental impact of provider-to-provider telemedicine in the Catalan health system

This pipeline estimates the travel-related greenhouse gas emissions that asynchronous
provider-to-provider interconsultations (e-consults) in the Catalan public health system
may have avoided, over 2019 and 2023-2025.

For each interconsultation it reconstructs the round-trip driving route between the
referring and the receiving facility, works out the CO2, CH4 and N2O that trip would have
produced, and aggregates the result by health region, by year, and by Capital or
Non-Capital status.

## Data availability and control

The record-level administrative data are not public and are not in this repository. That
covers the interconsultation extracts, the patient-level age and sex cells, and the
facility identifiers. They belong to the Catalan health administration (CatSalut /
Departament de Salut), and were accessed under a data-use agreement for this study.

What you get here is the code. Running it end to end means placing the restricted input
files in a local `datafiles/` directory, which is never committed (see `.gitignore`).
Without them the pipeline will not run, and no substitute dataset is shipped in their
place.

The aggregated route-by-year extract published alongside the article (S1 Data, built by
`pipeline/s10_tables.py`) holds no patient-level information. Counts below 10
interconsultations are suppressed, and nothing is broken down by sex or age. That extract
is the public record of this analysis.

## Input files

The pipeline reads a few files from `INTERCONSULT_DATA_DIR`, under the names below. They
are generic on purpose. If your local copies have other names, rename them to these; no
code change is needed.

| Expected name | What it is | Used by |
|---|---|---|
| `raw_export_historical.xlsx` | Historical interconsultation extract (2019 to 2023 and the partial 2025 slice) | `s01_preprocess` |
| `raw_export_2025.txt` | Complete 2025 interconsultation export | `s03_integrate_2025` |
| `UP_codis.xlsx` | Provider-unit catalogue, codes and descriptions | `s01_preprocess` |
| `Regions_sanitàries_20250921.geojson` | Health-region boundaries (Departament de Salut open data) | `s02_geocode`, `s02b_validate_locations` |
| `cleaned_locations_geocoded.parquet` | Hand-validated facility coordinates (frozen) | `s02_geocode` |
| `route_cache_ors.pkl`, `route_cache_osrm.pkl` | Frozen routing caches (frozen) | `s04_route_ors`, `s05_route_osrm` |

The frozen files can sit in `INTERCONSULT_DATA_DIR` or in `INTERCONSULT_OUTPUT_DIR/cache/`.
The pipeline looks in both places.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `INTERCONSULT_DATA_DIR` | `../datafiles` (relative to `publish/`) | Restricted input files and frozen intermediates/caches. Never written to. |
| `INTERCONSULT_OUTPUT_DIR` | `../outputs` (relative to `publish/`) | Every generated artifact: intermediates, route caches, figures, tables. |
| `ORS_API_KEY` | none | OpenRouteService key, only needed with `--no-use-cache`/`--refresh` on `s04_route_ors`. |
| `INTERCONSULT_ORS_BASE_URL` | `https://api.heigit.org/openrouteservice` | OpenRouteService host. The default is HeiGIT's current one; the older `api.openrouteservice.org` still answers but throttles the same key to 200 requests per 24h instead of 2,000. |
| `GOOGLE_MAPS_API_KEY` | none | Google Geocoding API key, only needed for the historical, opt-in `s02_geocode --geocode` path. |
| `INTERCONSULT_NOMINATIM_UA` | `interconsultations-emissions-study/1.0 (research; contact via repository)` | HTTP user agent sent to Nominatim, per its usage policy. |

PowerShell:

```powershell
$env:INTERCONSULT_DATA_DIR = "C:\path\to\datafiles"
$env:INTERCONSULT_OUTPUT_DIR = "C:\path\to\outputs"
$env:ORS_API_KEY = "your-key"
```

bash:

```bash
export INTERCONSULT_DATA_DIR=/path/to/datafiles
export INTERCONSULT_OUTPUT_DIR=/path/to/outputs
export ORS_API_KEY=your-key
```

You do not have to set any of them to reproduce the published results. The defaults and
the frozen route caches are all that requires (see "How to run").

## How to run

From the `publish/` directory, with Python 3.13:

```bash
pip install -r requirements.txt

# Full pipeline, reproducing the published results with no API keys at all
# (the default: --use-cache is on unless you turn it off).
python run_all.py

# Equivalent, explicit form.
python run_all.py --use-cache

# Permit live ORS/OSRM calls for anything missing from the frozen caches
# (requires ORS_API_KEY for the ORS step; OSRM's public demo server needs no key).
python run_all.py --no-use-cache

# Print the ordered step list (inputs, outputs, one-line description) and exit.
python run_all.py --list

# Run a single step, or a subset.
python run_all.py --only s06_emissions
python -m pipeline.s06_emissions

# Start partway through and run to the end.
python run_all.py --from s07_statistics

# Keep going after a failing step instead of stopping at the first one.
python run_all.py --continue-on-error
```

Every step module also runs standalone, the same way `run_all.py` runs it internally:

```bash
python -m pipeline.s01_preprocess
python -m pipeline.s04_route_ors --use-cache
```

When a step needs a previous step's output it asks `pipeline.utils.resolve_input` for it,
which prefers a freshly generated file in `OUTPUT_DIR` and otherwise falls back to the
frozen copy shipped in `DATA_DIR`. That is what lets any single step run on its own, and
what lets the whole pipeline run without touching an API.

## Pipeline steps

Run order follows the dependencies, which is not the numeric filename order:
`s05_route_osrm` runs after `s06_emissions` because its ORS/OSRM agreement analysis is
computed on `analysis_dataset.parquet`, and `s06` is what produces that file. The
pre-emissions intermediate will not do. `run_all.py` encodes the order explicitly, with a
comment at the step list explaining why.

| # | Step | Inputs | Outputs | What it does |
|---|---|---|---|---|
| 1 | `s01_preprocess` | `raw_export_historical.xlsx`, `UP_codis.xlsx` | `cleaned_locations_for_geocoding.parquet`, `processed_export.parquet` | Cleans/standardizes origin and destination facility names, drops interconsultations not tied to a physical service, splits co-managed sites, joins the provider-unit catalogue. |
| 2 | `s02_geocode` | `cleaned_locations_for_geocoding.parquet`, or the frozen `cleaned_locations_geocoded.parquet` | `cleaned_locations_geocoded.parquet`, `geocoded_interconsultations.parquet` | Default: merges the frozen, hand-validated coordinates onto the derivations dataset, no network access. `--geocode` (opt-in, historical): re-derives raw coordinates from the Google Geocoding API; needs `GOOGLE_MAPS_API_KEY` and is not needed to reproduce published results. |
| 2b | `s02b_validate_locations` | geocoded locations, health-region GeoJSON | `validated_*` coordinate/flag columns | **Interactive only, never run by `run_all.py`.** A jupytext percent-format tool (ipywidgets + ipyleaflet + Nominatim reverse geocoding) for manually checking/correcting each location's coordinate. See "Interactive validation tool" below. |
| 3 | `s03_integrate_2025` | `geocoded_interconsultations.parquet`, `raw_export_2025.txt` | `geocoded_interconsultations.parquet` (updated), `unroutable_2025.csv` | Replaces the partial 2025 slice in the historical export with the complete 2025 export, per `config.YEAR_2025_MODE` (default `refresh_replaces_partial`). |
| 4 | `s04_route_ors` | `geocoded_interconsultations.parquet`, `route_cache_ors.pkl` | `routed_ors.parquet`, `route_table_ors.parquet`, `cache/route_cache_ors.pkl` | Routes every unique origin-destination coordinate pair with OpenRouteService (outbound and return legs routed and summed separately); `--use-cache` (default) never calls the API, `--refresh` permits live calls for pairs missing from or previously failed in the cache. |
| 5 | `s06_emissions` | `routed_ors.parquet` | `analysis_dataset.parquet`, `emissions_summary.xlsx` | Computes CO2/CH4/N2O/CO2-equivalent emissions per interconsultation from year-specific MITECO factors weighted by the DGT national fleet's diesel/gasoline split; writes the final, renamed analysis dataset the rest of the pipeline builds on. |
| 6 | `s05_route_osrm` | `geocoded_interconsultations.parquet`, `analysis_dataset.parquet`, `route_cache_osrm.pkl` | `routed_osrm.parquet`, `route_table_osrm.parquet`, `cache/route_cache_osrm.pkl`, `ors_osrm_agreement.xlsx` | Re-routes the same pairs with OSRM and quantifies how much the choice of routing engine matters for the study's population-weighted results, both as a weighted total and as a deduplicated route-level comparison. |
| 7 | `s07_statistics` | `analysis_dataset.parquet` | `statistics_capital_vs_noncapital.xlsx` | Capital vs Non-Capital comparison for six outcomes (Shapiro-Wilk, Levene, Mann-Whitney U, Cliff's delta with route-clustered bootstrap CIs, Cohen's d as reference only), plus a route-level deduplicated comparison and a sex split. |
| 8 | `s08_sensitivity` | `analysis_dataset.parquet` | `sensitivity_analysis.xlsx` | Two sensitivity analyses: rescaling the headline totals at lower substitution rates, and re-computing avoided CO2 under mixed-transport-mode assumptions (car/public-transport shares by region, MITECO public-transport factors). |
| 9 | `s09_figures` | `analysis_dataset.parquet`, regions GeoJSON, raw/intermediate funnel files | `figures/Fig1.tif`, `Fig2.tif`, `Fig3.tif` (+ `.pdf` of each) | Renders the three manuscript figures (inclusion/exclusion flow chart, per-interconsultation box plots, spatial maps) as PLOS ONE-compliant 300 dpi RGB TIFFs. |
| 10 | `s10_tables` | `analysis_dataset.parquet`, `routed_osrm.parquet`, `statistics_capital_vs_noncapital.xlsx`, `sensitivity_analysis.xlsx` | `tables/Table1..5.xlsx`, `tables/S1..S4_Table.xlsx`, `tables/S1_Data.xlsx`, `manuscript_numbers.csv` | Builds every manuscript table, the published route-year data extract (S1 Data), and the machine-readable inventory of every number the manuscript cites. The final step. |

### Interactive validation tool

`pipeline/s02b_validate_locations.py` is written in jupytext "percent" format, so it is a
plain `.py` file that is also a valid notebook. Open it as a notebook with:

```bash
jupytext --to ipynb pipeline/s02b_validate_locations.py
```

It needs `ipywidgets` and `ipyleaflet` on top of the core dependencies. `run_all.py`
never runs it, because it needs a human looking at a map. The coordinates it produces are
a one-time, hand-validated input, frozen afterwards and reused by every later run of
`s02_geocode`.

## Python version and dependencies

Python 3.13. `requirements.txt` pins the exact versions used to produce the published
results:

```bash
pip install -r requirements.txt
```

`jupytext` is listed as optional. You need it only to open `s02b_validate_locations.py`
as a notebook, not to run the pipeline.

## Checks

`checks/scan_for_data.py` walks `publish/` before anything is shared or committed. Run it
from the `publish/` directory:

```bash
py -3.13 checks/scan_for_data.py
py -3.13 checks/scan_for_data.py --strict   # also fail on warnings
```

It fails if it finds anything that has no business being in this repository: data files by
extension, files over 1 MB, literals shaped like an API key, latitude/longitude pairs that
land inside Catalonia, or the names of the restricted inputs. A single number with four or
more decimals is deliberately not enough to trigger the coordinate check, since `config.py`
legitimately carries fleet shares; a plausible latitude and a plausible longitude have to
turn up close together.

It also reports two things without failing, unless you pass `--strict`. One is facility
names found in the code. `pipeline/utils.py` names some of them in its name-cleaning table,
which the pipeline needs; those names are listed in `REVIEWED_UTILS_FACILITY_NAMES` in the
scanner after a manual review. Any other facility name is still reported. The other is
stray "person" or "people" wording outside a legacy-column comment or a rename-map entry.

## Reproducibility notes

- The geocoding step (`s02_geocode --geocode`) is historical and opt-in. The published
  results use the frozen, hand-validated coordinate set
  (`cleaned_locations_geocoded.parquet`), which is what the default, no-flag path reuses.
- The frozen route caches (`route_cache_ors.pkl`, `route_cache_osrm.pkl`) are what make a full
  rerun possible with no API keys at all. `--use-cache`, the default, never calls ORS,
  OSRM or Google.
- The complete 2025 export supersedes a partial 2025 slice that is already present in the
  historical export (`config.YEAR_2025_MODE = "refresh_replaces_partial"`, the default).
  Concatenating the two instead (`"concatenate"`) double counts roughly 361,873
  interconsultations and shifts every reported number. That mode exists only to check that
  the switch itself works, never to reproduce a published result.
- The two input data streams apply the non-physical-service filter with different case
  sensitivity. The historical stream cleans facility names first and then matches
  case-insensitively; the 2025 refresh stream matches the raw names case-sensitively,
  before cleaning. The difference is preserved on purpose, since it is what produced the
  published Figure 1 funnel counts on each stream. Making the two consistent would change
  those counts.
