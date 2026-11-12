# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Step 2b: interactive location coordinate validation
#
# This is the one interactive tool in the pipeline. It lets a validator check,
# correct and flag the coordinate obtained for each unique location name,
# using a draggable marker on an OpenStreetMap basemap and a reverse
# geocoding lookup (geopy / Nominatim) to show the address at the current
# marker position.
#
# It is written as a plain `.py` file in jupytext "percent" format (the
# front-matter above and the `# %%` cell markers), so it is both a valid
# Python module and, when paired with jupytext, a notebook that can be
# opened and run cell by cell.
#
# For every unique location it writes back:
# `validated_latitude`, `validated_longitude`, `validated_address`,
# `validated`, `flagged`, `flag_reason`.
#
# The health-region assignment shown while validating is for context only
# and is not persisted here: the published results reassign regions by a
# spatial join in a later step, so nothing downstream depends on this tool
# writing a region column.
#
# Dependencies: this tool requires `ipywidgets` and `ipyleaflet` in addition
# to the core pipeline dependencies (`geopandas`, `shapely`, `geopy`). If
# they are not installed, running this module prints a clear message and
# exits instead of raising an import error.

# %%
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pandas as pd

import config
from pipeline import utils

# %% [markdown]
# ## Optional interactive dependencies
#
# These are imported defensively: a missing package is recorded instead of
# raising at module import time, so the module can still be imported (e.g.
# by a test or by `py_compile`) without ipywidgets/ipyleaflet installed.
# `main()` checks this and exits cleanly before doing anything interactive.

# %%
_IMPORT_ERROR: Exception | None = None

try:
    import ipywidgets as widgets
    from IPython.display import clear_output, display
    from ipyleaflet import FullScreenControl, LayersControl, Map, Marker, basemaps
except ImportError as exc:  # pragma: no cover - exercised only when missing
    _IMPORT_ERROR = exc

try:
    import geopandas as gpd
    from geopy.extra.rate_limiter import RateLimiter
    from geopy.geocoders import Nominatim
    from shapely.geometry import Point
except ImportError as exc:  # pragma: no cover - exercised only when missing
    if _IMPORT_ERROR is None:
        _IMPORT_ERROR = exc

# %% [markdown]
# ## Configuration

# %%
LOCATIONS_FILENAME = "cleaned_locations_geocoded.parquet"
BACKUP_SUFFIX = ".backup.parquet"

# Hook for a later phase: if this file exists in OUTPUT_DIR, iterate only the
# location ids it lists instead of the full table. This is meant for
# re-sourcing coordinates for locations that were flagged for manual
# attention; the re-sourcing itself is out of scope here, only the
# restricted-iteration mechanism is implemented.
REVIEW_LIST_FILENAME = "locations_to_review.csv"
REVIEW_LIST_COLUMN = "location_name"

REVERSE_GEOCODE_DEBOUNCE_S = 0.6

REQUIRED_COLUMNS = ["location_name", "latitude", "longitude", "address"]
VALIDATION_COLUMN_DEFAULTS = {
    "validated": False,
    "validated_latitude": pd.NA,
    "validated_longitude": pd.NA,
    "validated_address": pd.NA,
    "flagged": False,
    "flag_reason": pd.NA,
}


# -----------------------------------------------------------------------------
# NOTE: `utils.resolve_input` is specified in the pipeline contract (section 9)
# but is not yet implemented in pipeline/utils.py. This module is restricted
# to writing only its own file, so this is a local, private mirror of that
# exact contract definition (identical to the one in s02_geocode.py). It
# should be deleted in favour of `utils.resolve_input` once that lands.
# -----------------------------------------------------------------------------


# %% [markdown]
# ## Data loading

# %%
def load_locations() -> tuple[pd.DataFrame, Path]:
    """Load the geocoded locations table and the OUTPUT_DIR path to write it
    back to. Validation always writes to OUTPUT_DIR, never to DATA_DIR: the
    input may be the frozen copy in DATA_DIR (via `_resolve_input`), but the
    write target is always `config.output_path(LOCATIONS_FILENAME)`.
    """
    in_path = utils.resolve_input(LOCATIONS_FILENAME)
    out_path = config.output_path(LOCATIONS_FILENAME)

    df = utils.read_table(in_path)

    rename_map = {}
    if "lat" in df.columns and "latitude" not in df.columns:
        rename_map["lat"] = "latitude"
    if "lon" in df.columns and "longitude" not in df.columns:
        rename_map["lon"] = "longitude"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Once coordinates have been re-sourced, the locations table no longer
    # carries the raw geocoding columns: `validated_latitude` /
    # `validated_longitude` are the coordinate of record, and a stale second
    # copy per location is a trap rather than a convenience. Seed the columns
    # this tool shows as candidates from the validated pair in that case, so a
    # location can still be reviewed and moved on the map.
    if "latitude" not in df.columns and "validated_latitude" in df.columns:
        df["latitude"] = df["validated_latitude"]
    if "longitude" not in df.columns and "validated_longitude" in df.columns:
        df["longitude"] = df["validated_longitude"]
    if "address" not in df.columns:
        df["address"] = df.get("validated_address", pd.NA)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Locations table is missing required columns: {missing}")

    for col, default in VALIDATION_COLUMN_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    return df, out_path


def load_regions():
    """Load the health-region geometry named in config.REGIONS_GEOJSON. Used
    only to show context (which region a dragged marker currently falls in);
    the result is never written back, because s06 reassigns regions by a
    spatial join on the final analysis dataset.
    """
    regions = gpd.read_file(config.data_path(config.REGIONS_GEOJSON))
    if regions.crs is None:
        regions = regions.set_crs(epsg=4326)
    return regions.to_crs("EPSG:4326")


def classify_region_for_display(lon: float, lat: float, regions_gdf) -> str | None:
    """Context-only lookup: the health region name containing (lon, lat), or
    None if outside all regions or the point/region data is unavailable.
    """
    if regions_gdf is None or pd.isna(lon) or pd.isna(lat):
        return None
    point = Point(lon, lat)
    hits = regions_gdf[regions_gdf.contains(point)]
    if hits.empty:
        return None
    name = hits.iloc[0].get(config.REGION_NAME_FIELD)
    if name in config.REGION_EXCLUDE:
        return None
    return name


def load_review_list() -> set[str] | None:
    """If a review list exists in OUTPUT_DIR, return the set of location ids
    it names; otherwise None (meaning: iterate every location). See
    REVIEW_LIST_FILENAME above.
    """
    path = config.output_path(REVIEW_LIST_FILENAME)
    if not path.exists():
        return None
    ids = utils.read_table(path)
    col = REVIEW_LIST_COLUMN if REVIEW_LIST_COLUMN in ids.columns else ids.columns[0]
    return set(ids[col].dropna().astype(str))


# %% [markdown]
# ## Persistence
#
# Every write goes to OUTPUT_DIR. A backup of the table as it existed before
# this tool's first write is taken once per session, also under OUTPUT_DIR.

# %%
_backup_done = False


def ensure_backup(out_path: Path) -> None:
    """Write a one-time backup of `out_path` before the first modification,
    if a prior copy exists there and no backup has been made yet this
    session.
    """
    global _backup_done
    if _backup_done:
        return
    backup_path = out_path.with_name(out_path.stem + BACKUP_SUFFIX)
    if out_path.exists() and not backup_path.exists():
        try:
            utils.write_table(utils.read_table(out_path), backup_path)
            utils.log(f"Backup created at {backup_path}")
        except Exception as exc:
            utils.log(f"Warning: could not create backup: {exc}")
    _backup_done = True


def persist_validation(df: pd.DataFrame, out_path: Path, location_name: str, lat: float, lon: float, address: str) -> None:
    ensure_backup(out_path)
    mask = df["location_name"] == location_name
    df.loc[mask, "validated"] = True
    df.loc[mask, "validated_latitude"] = float(lat)
    df.loc[mask, "validated_longitude"] = float(lon)
    df.loc[mask, "validated_address"] = address
    utils.write_table(df, out_path)


def persist_flag(df: pd.DataFrame, out_path: Path, location_name: str, flagged: bool, reason: str = "") -> None:
    ensure_backup(out_path)
    mask = df["location_name"] == location_name
    df.loc[mask, "flagged"] = flagged
    df.loc[mask, "flag_reason"] = reason if flagged and reason else pd.NA
    utils.write_table(df, out_path)


def remove_validation(df: pd.DataFrame, out_path: Path, location_name: str) -> None:
    ensure_backup(out_path)
    mask = df["location_name"] == location_name
    df.loc[mask, ["validated", "validated_latitude", "validated_longitude", "validated_address"]] = [
        False,
        pd.NA,
        pd.NA,
        pd.NA,
    ]
    utils.write_table(df, out_path)


# %% [markdown]
# ## Coordinate parsing

# %%
def parse_coords(text: str) -> tuple[float, float]:
    """Parse a "lat, lon" string into a (lat, lon) tuple, validating range."""
    if text is None:
        raise ValueError("Empty coordinates")
    s = str(text).strip().replace(";", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Use format: 'lat, lon'")
    lat, lon = float(parts[0]), float(parts[1])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Coordinates out of valid range")
    return lat, lon


def format_address_for_dropdown(address, max_length: int = 85) -> str:
    """Shorten a long address for a dropdown option label, breaking on a
    natural delimiter where possible rather than mid-word.
    """
    if not address:
        return "No address available"
    text = str(address).strip()
    if len(text) <= max_length:
        return text
    for delimiter in (", ", " - ", "; ", " | "):
        if delimiter in text:
            parts = text.split(delimiter)
            truncated = parts[0]
            for part in parts[1:]:
                candidate = truncated + delimiter + part
                if len(candidate) <= max_length - 3:
                    truncated = candidate
                else:
                    break
            if len(truncated) < len(text):
                return truncated + "..."
    return text[: max_length - 3] + "..."


# %% [markdown]
# ## The interactive tool
#
# Everything below builds and displays the widget UI. It is only reachable
# through `main()`, which checks the optional dependencies first.

# %%
def run_validator() -> None:
    """Load the data, build the map/widget UI, and display it.

    Writes back `validated_latitude`, `validated_longitude`,
    `validated_address`, `validated`, `flagged`, `flag_reason` to the
    locations table in OUTPUT_DIR as the validator saves each location.
    """
    df, out_path = load_locations()
    regions_gdf = load_regions()
    review_ids = load_review_list()

    groups = []
    for name, group in df.groupby("location_name"):
        candidates = group[["latitude", "longitude", "address"]].drop_duplicates().to_dict(orient="records")
        groups.append({"location_name": name, "candidates": candidates})
    groups.sort(key=lambda item: str(item["location_name"]).lower())

    if review_ids is not None:
        groups = [g for g in groups if str(g["location_name"]) in review_ids]
        utils.log(f"Review-list mode: {len(groups)} location(s) loaded from {REVIEW_LIST_FILENAME}.")
    else:
        utils.log(f"Loaded {len(groups)} unique locations for validation.")

    def status_dicts():
        validated = df.groupby("location_name")["validated"].max().to_dict()
        flagged = df.groupby("location_name")["flagged"].max().to_dict()
        reasons = df[df["flagged"] == True].groupby("location_name")["flag_reason"].first().to_dict()  # noqa: E712
        reasons = {k: (v if pd.notna(v) else "") for k, v in reasons.items()}
        return validated, flagged, reasons

    loc_validated, loc_flagged, loc_flag_reason = status_dicts()

    def refresh_status():
        nonlocal loc_validated, loc_flagged, loc_flag_reason
        loc_validated, loc_flagged, loc_flag_reason = status_dicts()

    def is_validated(name) -> bool:
        return bool(loc_validated.get(name, False))

    def is_flagged(name) -> bool:
        return bool(loc_flagged.get(name, False))

    def flag_reason_for(name) -> str:
        return loc_flag_reason.get(name, "")

    # Nominatim reverse geocoding, rate-limited per the usage policy.
    geolocator = Nominatim(user_agent=config.nominatim_user_agent(), timeout=10)
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=config.NOMINATIM_SLEEP_S)

    _reverse_state = {"latest": None, "updating": False}

    def schedule_reverse_geocode(lat: float, lon: float) -> None:
        _reverse_state["latest"] = (lat, lon)

        def _run():
            time.sleep(REVERSE_GEOCODE_DEBOUNCE_S)
            if _reverse_state["latest"] != (lat, lon):
                return
            try:
                location = reverse((lat, lon), language="ca") or reverse((lat, lon), language="es")
                if location and location.address:
                    addr_input.value = location.address
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()

    # Map and marker. The initial position comes from the first location's
    # own candidate coordinate, or, failing that, the centroid of the
    # loaded regions; never from a hardcoded coordinate literal.
    if groups and groups[0]["candidates"]:
        start_lat = groups[0]["candidates"][0]["latitude"]
        start_lon = groups[0]["candidates"][0]["longitude"]
    else:
        centroid = regions_gdf.union_all().centroid
        start_lat, start_lon = centroid.y, centroid.x

    m = Map(center=(start_lat, start_lon), zoom=15, basemap=basemaps.OpenStreetMap.Mapnik, scroll_wheel_zoom=True)
    m.layout = widgets.Layout(height="650px", width="100%")
    marker = Marker(location=(start_lat, start_lon), draggable=True)
    m.add_layer(marker)
    m.add_control(LayersControl(position="topright"))
    m.add_control(FullScreenControl())

    def set_coords(lat: float, lon: float, from_marker: bool = False, trigger_reverse: bool = True) -> None:
        _reverse_state["updating"] = True
        try:
            coords_input.value = f"{lat:.6f},{lon:.6f}"
            if not from_marker:
                marker.location = (float(lat), float(lon))
            m.center = (float(lat), float(lon))
            region_label.value = f"Region (context only): {classify_region_for_display(lon, lat, regions_gdf) or 'unknown'}"
            if trigger_reverse:
                schedule_reverse_geocode(float(lat), float(lon))
        finally:
            _reverse_state["updating"] = False

    def on_marker_move(change):
        if change["name"] == "location":
            lat, lon = change["new"]
            set_coords(float(lat), float(lon), from_marker=True, trigger_reverse=True)

    marker.observe(on_marker_move, "location")

    # Widgets
    filter_dropdown = widgets.Dropdown(
        options=["All locations", "Unvalidated only", "Validated only", "Flagged only"],
        value="All locations",
        description="Show:",
    )
    candidate_dropdown = widgets.Dropdown(options=[], description="Candidate:", layout=widgets.Layout(width="600px"))
    coords_input = widgets.Text(description="Lat, Lon:", layout=widgets.Layout(width="400px"))
    addr_input = widgets.Textarea(description="Address:", layout=widgets.Layout(width="600px", height="60px"))
    region_label = widgets.Label(value="Region (context only): -")
    flag_chk = widgets.Checkbox(value=False, description="Flag this location")
    flag_reason_input = widgets.Textarea(
        placeholder="Reason for flagging, e.g. 'unclear location', 'multiple candidates'",
        layout=widgets.Layout(width="600px", height="50px"),
    )
    status_label = widgets.HTML(value="")
    save_confirm = widgets.HTML(value="")
    header_label = widgets.HTML(value="")

    btn_save = widgets.Button(description="Save", button_style="success")
    btn_next = widgets.Button(description="Next", button_style="info")
    btn_prev = widgets.Button(description="Previous", button_style="info")
    btn_next_unvalidated = widgets.Button(description="Next unvalidated", button_style="warning")
    btn_remove = widgets.Button(description="Remove validation", button_style="danger")

    state = {"index": 0, "candidate_map": {}}

    def filtered_groups():
        choice = filter_dropdown.value
        if choice == "Unvalidated only":
            return [g for g in groups if not is_validated(g["location_name"])]
        if choice == "Validated only":
            return [g for g in groups if is_validated(g["location_name"])]
        if choice == "Flagged only":
            return [g for g in groups if is_flagged(g["location_name"])]
        return groups

    def first_unvalidated_index() -> int:
        for i, g in enumerate(filtered_groups()):
            if not is_validated(g["location_name"]):
                return i
        return 0

    def display_location(idx: int) -> None:
        clear_output(wait=True)
        current = filtered_groups()
        if not current:
            display(widgets.VBox([filter_dropdown, widgets.HTML("<i>No locations match the current filter.</i>")]))
            return

        idx = max(0, min(idx, len(current) - 1))
        state["index"] = idx
        item = current[idx]
        name = item["location_name"]
        candidates = item["candidates"]

        state["candidate_map"] = {}
        options = []
        for i, c in enumerate(candidates, start=1):
            label = f"Candidate {i}: {format_address_for_dropdown(c['address'])}"
            options.append(label)
            state["candidate_map"][label] = c
        options.append("Manual entry")
        candidate_dropdown.options = options
        candidate_dropdown.value = options[0]

        mask = df["location_name"] == name
        if is_validated(name):
            lat = float(df.loc[mask, "validated_latitude"].dropna().iloc[0])
            lon = float(df.loc[mask, "validated_longitude"].dropna().iloc[0])
            addr = str(df.loc[mask, "validated_address"].dropna().iloc[0])
        else:
            lat = candidates[0]["latitude"]
            lon = candidates[0]["longitude"]
            addr = candidates[0]["address"] or ""

        set_coords(float(lat), float(lon), from_marker=False, trigger_reverse=False)
        addr_input.value = addr

        flag_chk.value = is_flagged(name)
        flag_reason_input.value = flag_reason_for(name)
        status_label.value = ""
        save_confirm.value = ""

        status_bits = []
        if is_validated(name):
            status_bits.append("<span style='color:green'>Validated</span>")
        if is_flagged(name):
            status_bits.append("<span style='color:#b36b00'>Flagged</span>")
        status_text = " | ".join(status_bits) if status_bits else "Not validated"
        header_label.value = (
            f"<b>Location {idx + 1}/{len(current)}</b> (total unique locations: {len(groups)})"
            f"<br>{name}<br><span style='font-size:12px'>{status_text}</span>"
        )

        left = widgets.VBox(
            [
                filter_dropdown,
                header_label,
                candidate_dropdown,
                addr_input,
                coords_input,
                region_label,
                flag_chk,
                flag_reason_input,
                widgets.HBox([btn_prev, btn_next, btn_next_unvalidated]),
                widgets.HBox([btn_save, btn_remove]),
                status_label,
                save_confirm,
            ],
            layout=widgets.Layout(width="680px", padding="10px"),
        )
        display(widgets.HBox([left, m]))

    def on_candidate_change(change):
        label = change["new"]
        if label == "Manual entry" or label not in state["candidate_map"]:
            return
        c = state["candidate_map"][label]
        set_coords(float(c["latitude"]), float(c["longitude"]), from_marker=False, trigger_reverse=True)
        addr_input.value = c["address"] or ""

    def on_coords_change(change):
        if _reverse_state["updating"]:
            return
        try:
            lat, lon = parse_coords(change["new"])
            set_coords(lat, lon, from_marker=False, trigger_reverse=True)
            status_label.value = ""
        except Exception as exc:
            status_label.value = f"<span style='color:#cc0000'>Invalid coordinates: {exc}</span>"

    def on_save(_):
        current = filtered_groups()
        if state["index"] >= len(current):
            return
        name = current[state["index"]]["location_name"]
        try:
            lat, lon = parse_coords(coords_input.value)
        except Exception as exc:
            status_label.value = f"<span style='color:#cc0000'>Cannot save: {exc}</span>"
            return
        persist_validation(df, out_path, name, lat, lon, addr_input.value)
        if flag_chk.value:
            persist_flag(df, out_path, name, True, flag_reason_input.value)
        refresh_status()
        save_confirm.value = f"<span style='color:green'>Saved ({lat:.6f}, {lon:.6f})</span>"

    def on_remove(_):
        current = filtered_groups()
        if state["index"] >= len(current):
            return
        name = current[state["index"]]["location_name"]
        if is_validated(name):
            remove_validation(df, out_path, name)
            refresh_status()
            display_location(state["index"])
        else:
            save_confirm.value = "<span style='color:gray'>Location is not validated.</span>"

    def on_flag_change(change):
        current = filtered_groups()
        if state["index"] >= len(current):
            return
        name = current[state["index"]]["location_name"]
        persist_flag(df, out_path, name, change["new"], flag_reason_input.value if change["new"] else "")
        refresh_status()

    def advance(step: int, prefer_unvalidated: bool = False) -> None:
        current = filtered_groups()
        if not current:
            return
        idx = state["index"]
        if prefer_unvalidated:
            rng = range(idx + step, len(current) if step > 0 else -1, step)
            for i in rng:
                if not is_validated(current[i]["location_name"]):
                    display_location(i)
                    return
            return
        new_idx = idx + step
        if 0 <= new_idx < len(current):
            display_location(new_idx)

    def on_filter_change(_):
        display_location(first_unvalidated_index())

    candidate_dropdown.observe(on_candidate_change, names="value")
    coords_input.observe(on_coords_change, names="value")
    flag_chk.observe(on_flag_change, names="value")
    btn_save.on_click(on_save)
    btn_remove.on_click(on_remove)
    btn_next.on_click(lambda _: advance(1))
    btn_prev.on_click(lambda _: advance(-1))
    btn_next_unvalidated.on_click(lambda _: advance(1, prefer_unvalidated=True))
    filter_dropdown.observe(on_filter_change, names="value")

    display_location(first_unvalidated_index())


def main() -> None:
    """Entry point: check optional interactive dependencies before doing
    anything else, so a missing ipywidgets/ipyleaflet degrades gracefully
    (clear message, clean exit) instead of raising an import error.
    """
    if _IMPORT_ERROR is not None:
        utils.log(
            "s02b_validate_locations requires the optional interactive "
            "dependencies ipywidgets, ipyleaflet, geopandas and geopy "
            f"(missing or failed to import: {_IMPORT_ERROR}). Install them "
            "(see requirements.txt) and run this module from within "
            "Jupyter, or via `jupytext --to notebook` and a notebook server."
        )
        sys.exit(1)
    run_validator()


# %%
if __name__ == "__main__":
    main()
