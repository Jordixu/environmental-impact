"""Shared configuration for the interconsultations emissions analysis pipeline.

Every path, constant, emission factor and reference value used by the
`pipeline.sNN_*` step modules lives here. The module has to import cleanly
with no environment variables set, so anything that needs a secret (an API
key) is wrapped in a function instead of a module-level constant: it is only
evaluated, and only raises, when a step actually calls it.

It also does not touch the input data files themselves, so it can be
imported on its own for testing or documentation.
"""

from __future__ import annotations

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# All data/output locations are environment-variable overridable, with
# defaults expressed relative to the directory containing this file (the
# `publish/` package root), so the pipeline behaves the same regardless of
# the caller's current working directory.

_PUBLISH_DIR = Path(__file__).resolve().parent


def _resolve_dir(env_var: str, default: str) -> Path:
    """Resolve a configurable directory: the env var if it is set, otherwise
    a path relative to the `publish/` directory. Either way, what comes back
    is an absolute, resolved Path.
    """
    raw = os.environ.get(env_var, default)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = _PUBLISH_DIR / candidate
    return candidate.resolve()


DATA_DIR = _resolve_dir("INTERCONSULT_DATA_DIR", "../datafiles")
OUTPUT_DIR = _resolve_dir("INTERCONSULT_OUTPUT_DIR", "../outputs")

FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


def data_path(*parts: str) -> Path:
    """Build a path under DATA_DIR. This does not create any directories:
    the input data is expected to already be there.
    """
    return DATA_DIR.joinpath(*parts)


def output_path(*parts: str) -> Path:
    """Build a path under OUTPUT_DIR, creating parent directories along the
    way so a step module can write to the result directly.
    """
    path = OUTPUT_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
# API keys / secrets
# -----------------------------------------------------------------------------
# These are functions, not module-level constants, so importing this module
# never fails for lack of a secret. Each one raises only when a step that
# actually needs the key calls it.

def ors_api_key() -> str:
    """Return the OpenRouteService API key from ORS_API_KEY.

    Raises RuntimeError with setup instructions if the variable is missing.
    """
    key = os.environ.get("ORS_API_KEY")
    if not key:
        raise RuntimeError(
            "ORS_API_KEY is not set. Obtain a free key from "
            "https://openrouteservice.org/dev/#/signup and set it as an "
            "environment variable before running the routing step."
        )
    return key


def google_api_key() -> str:
    """Return the Google Maps API key from GOOGLE_MAPS_API_KEY.

    Raises RuntimeError with setup instructions if the variable is missing.
    The key is only needed for the historical, opt-in geocoding path.
    """
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY is not set. This key is only required for "
            "the historical, opt-in Google geocoding path (--geocode); the "
            "default pipeline re-uses the frozen geocoded file and does not "
            "need it. Set the environment variable if you intend to "
            "re-geocode from scratch."
        )
    return key


def nominatim_user_agent() -> str:
    """Return the HTTP user agent string to use for Nominatim requests.

    Nominatim's usage policy requires a descriptive, non-generic user agent,
    so this is configurable via INTERCONSULT_NOMINATIM_UA and otherwise falls
    back to a default that identifies this research project.
    """
    return os.environ.get(
        "INTERCONSULT_NOMINATIM_UA",
        "interconsultations-emissions-study/1.0 (research; contact via repository)",
    )


# -----------------------------------------------------------------------------
# Study scope
# -----------------------------------------------------------------------------

YEARS = (2019, 2023, 2024, 2025)
YEAR_COL = "EXTRACT(YEARFROMOC_DATA)"
ROUTE_KEY = ["OC_UP_ORI", "DER_UP_SCS_DES"]  # yields 1,674 unique routes
CELL_KEY = ["OC_UP_ORI", "DER_UP_SCS_DES", "UP_DES", "UP_DES_1", "SEXE", "GRUP_EDAT"]
WEIGHT_COL = "N"  # number of interconsultations represented by each row

# -----------------------------------------------------------------------------
# 2025 data handling
# -----------------------------------------------------------------------------
# The historical export (raw_export_historical.xlsx) only has a partial
# 2025 slice. A later, complete 2025 export (raw_export_2025.txt)
# supersedes it.

YEAR_2025_MODE = "refresh_replaces_partial"
# "refresh_replaces_partial" (the default, and the one that reproduces every
#   published number): drop rows with YEAR_COL == 2025 from the historical
#   export, then append the complete 2025 export. 98.8% of the partial
#   file's 2025 keys reappear in the refresh with equal or higher counts,
#   which is why the refresh is treated as superseding it rather than as a
#   separate slice; concatenating both would double count roughly 361,873
#   interconsultations.
# "concatenate": append without dropping. Changes every reported number.

# -----------------------------------------------------------------------------
# Filters (the Figure 1 funnel, applied in this order)
# -----------------------------------------------------------------------------

MIN_ROUNDTRIP_KM = 0.2  # exclude round trips of 200 m or less
MAX_AGE_GROUP = 21  # exclude GRUP_EDAT > 21, i.e. age 110 years or more
NONPHYSICAL_KEYWORDS = ["RH", "PADES", "Dispositiu", "EAPT", "Equip Residències", "Montsià"]

# -----------------------------------------------------------------------------
# Emission factors
# -----------------------------------------------------------------------------
# Copied from the source calculation exactly as it was computed there. The
# arithmetic below is left unsimplified on purpose: the written form (e.g.
# "0.000 / 1000") records that CH4 and N2O are given in g/km in the source
# and converted to kg/km here, which is what anyone checking provenance
# against the MITECO tables will want to see.

FACTORS = {
    "diesel": {
        2019: {"CO2": 0.162, "CH4": 0.000 / 1000, "N2O": 0.007 / 1000},
        2023: {"CO2": 0.160, "CH4": 0.000 / 1000, "N2O": 0.007 / 1000},
        2024: {"CO2": 0.156, "CH4": 0.000 / 1000, "N2O": 0.007 / 1000},
        2025: {"CO2": 0.156, "CH4": 0.000 / 1000, "N2O": 0.007 / 1000},
    },
    "gasoline": {
        2019: {"CO2": 0.198, "CH4": 0.021 / 1000, "N2O": 0.002 / 1000},
        2023: {"CO2": 0.188, "CH4": 0.019 / 1000, "N2O": 0.002 / 1000},
        2024: {"CO2": 0.184, "CH4": 0.018 / 1000, "N2O": 0.002 / 1000},
        2025: {"CO2": 0.184, "CH4": 0.018 / 1000, "N2O": 0.002 / 1000},
    },
}

# CO2-equivalent factors (kg CO2eq / km) by fuel type and year: the
# published, three-decimal table, used as-is for every result. It agrees
# with FACTORS combined with the GWP values below to within 0.09% per fuel
# and 0.033% fleet-weighted. s06 checks that agreement rather than assuming
# it, so the two are treated as provenance for each other, not as
# alternative ways to compute the same number.
CO2EQ_BY_FUEL = {
    "diesel": {2019: 0.164, 2023: 0.162, 2024: 0.158, 2025: 0.158},
    "gasoline": {2019: 0.199, 2023: 0.189, 2024: 0.185, 2025: 0.185},
}

FACTOR_FALLBACK_YEAR = 2024  # a year absent from the table falls back to 2024

GWP_CO2, GWP_CH4, GWP_N2O = 1, 28, 265
GWP_SOURCE = "IPCC AR5, 100-year horizon"

EMISSION_FACTOR_SOURCE = "MITECO (2025)"

# -----------------------------------------------------------------------------
# Fleet composition
# -----------------------------------------------------------------------------
# Source: DGT, "Parque de vehiculos - Tablas estadisticas auxiliares 2025",
# table 5.2 "Parque por carburante", national passenger cars (turismos).
# Diesel and gasoline vehicles get their MITECO per-km combustion factors.
# "Otros" (mostly battery-electric vehicles) is given zero direct tailpipe
# emissions, so it drops out of the fleet-weighted factor entirely.

DGT_DIESEL, DGT_GASOLINE, DGT_OTHER = 12_742_805, 12_477_598, 568_084
DGT_TOTAL = DGT_DIESEL + DGT_GASOLINE + DGT_OTHER  # 25,788,487

DIESEL_SHARE = DGT_DIESEL / DGT_TOTAL  # 0.49413...
GASOLINE_SHARE = DGT_GASOLINE / DGT_TOTAL  # 0.48384...
OTHER_SHARE = DGT_OTHER / DGT_TOTAL  # 0.02203..., zero tailpipe emissions

FLEET_SOURCE = "DGT, Parque de vehiculos - Tablas estadisticas auxiliares 2025, table 5.2"

# -----------------------------------------------------------------------------
# Regions
# -----------------------------------------------------------------------------

REGIONS_GEOJSON = "Regions_sanitàries_20250921.geojson"  # located in DATA_DIR
REGION_NAME_FIELD = "nom"
REGION_EXCLUDE = {"No consta", ""}
CAPITAL_REGIONS = {
    "Barcelona Ciutat",
    "Barcelona Metropolitana Nord",
    "Barcelona Metropolitana Sud",
}
REGION_TYPE_LABELS = ("Capital", "Non-capital")
# "Non-capital" is the spelling the article uses, so it is the spelling here.
# The legacy code was inconsistent about it, writing "No Capital" in one place
# and "Non-capital" in the other two.

# CatSalut reference populations: public data, and the sole source behind
# every per-100,000 figure and S3 Table.
REFERENCE_POPULATION = {
    "Alt Pirineu i Aran": 73_316,
    "Lleida": 386_837,
    "Camp de Tarragona": 561_104,
    "Terres de l'Ebre": 188_467,
    "Girona": 931_278,
    "Catalunya Central": 444_305,
    "Barcelona Metropolitana Sud": 1_174_403,
    "Barcelona Metropolitana Nord": 2_135_133,
    "Barcelona Ciutat": 1_732_208,
    "Penedès": 488_844,
}
REFERENCE_POPULATION_TOTAL = 8_115_895
REFERENCE_POPULATION_SOURCE = "CatSalut reference population"
# Capital population is 5,041,744; Non-capital is REFERENCE_POPULATION_TOTAL
# minus capital, 3,074,151.

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

N_BOOTSTRAP = 2000
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95
SHAPIRO_SUBSAMPLE = 5000  # seeded with RANDOM_SEED
ALPHA = 0.05

# -----------------------------------------------------------------------------
# Sensitivity analysis
# -----------------------------------------------------------------------------

SUBSTITUTION_RATES = (1.00, 0.70, 0.50)
CAR_SHARE_CAPITAL = 0.335 / (0.335 + 0.182)  # ~0.648, EMEF 2024 (Barcelona metropolitan area)
CAR_SHARE_NONCAPITAL = 0.60 / (0.60 + 0.03)  # ~0.952, Departament de Territori / Mobilicat
PT_FACTORS = {"urban_bus": 0.103, "interurban_bus": 0.028, "regional_train": 0.041}
# kg CO2e per passenger-km, MITECO 2025
SHORT_RT_KM = 10.0
CAR_SHARE_SOURCE_CAPITAL = "EMEF 2024, Barcelona metropolitan area; car share of motorized trips"
CAR_SHARE_SOURCE_NONCAPITAL = "Departament de Territori / Mobilicat rural mobility data"

# -----------------------------------------------------------------------------
# Disclosure control
# -----------------------------------------------------------------------------

SUPPRESSION_THRESHOLD = 10  # route-year cells with fewer interconsultations than this
SUPPRESSION_LABEL = "<10"

# -----------------------------------------------------------------------------
# Routing and geocoding
# -----------------------------------------------------------------------------

ORS_PROFILE = "driving-car"
ORS_SLEEP_S = 2.2

# HeiGIT's current host for OpenRouteService. The older api.openrouteservice.org
# still answers, but it is being retired and the same key is throttled to 200
# requests per 24h there against 2,000 here, so a full refresh is only
# practical against this one. Override with INTERCONSULT_ORS_BASE_URL if the
# host moves again.
ORS_BASE_URL = os.environ.get(
    "INTERCONSULT_ORS_BASE_URL", "https://api.heigit.org/openrouteservice"
)
OSRM_BASE_URL = "https://router.project-osrm.org"
OSRM_SLEEP_S = 1.0
NOMINATIM_SLEEP_S = 1.1
NOMINATIM_AUTO_ACCEPT_M = 100.0  # Phase 3 auto-accept radius
COORD_IDENTICAL_M = 1.0  # Phase 3 "unchanged coordinate" radius

# -----------------------------------------------------------------------------
# Column renames
# -----------------------------------------------------------------------------
# Applied once, in s06, when writing the analysis dataset; everything
# downstream uses the new names only. The `abs_` prefix in the legacy name
# was actually a misnomer: these are health regions (regions sanitàries),
# not basic health areas (àrees bàsiques de salut). The mapping is kept here
# so the private regression check (kept outside publish/) can line up legacy
# outputs with the new ones.

COLUMN_RENAMES = {
    "origin_abs_region_name": "origin_health_region",
    "km_per_person": "km_per_interconsultation",
    "emissio_co2_kg_per_person": "co2_kg_per_interconsultation",
    "emissio_ch4_kg_per_person": "ch4_kg_per_interconsultation",
    "emissio_n2o_kg_per_person": "n2o_kg_per_interconsultation",
    "emissio_co2eq_kg_per_person": "co2eq_kg_per_interconsultation",
    "temps_total_per_person": "minutes_per_interconsultation",
}
