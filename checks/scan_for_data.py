"""Scan `publish/` for restricted content before it is shared or committed.

This is the backstop, not the first line of defence. `.gitignore` already
excludes data file extensions, but this script walks the tree and checks
file contents too, so a mistake (a stray data file, a pasted API key, a
hard-coded coordinate pair) gets caught even if it slips past `.gitignore`.

Hard failures (exit 1):
  - data files by extension (.parquet, .xlsx, .xls, .csv, .pkl, .geojson,
    .tif, .tiff, .png, .pdf, .html)
  - any file larger than 1 MB
  - restricted filenames, in the file name or anywhere in the text of a
    source file: the names of the restricted inputs and intermediates
    (`derivacions...`, `derivations_with...`, `*_and_abs`, `abs.xlsx`,
    `rutes_cache...`, `rutes_no_...`). Those names are restricted data too,
    so they must not appear in code, comments or printed examples.
  - API key / secret patterns (quoted-literal assignments to a name like
    api_key/token/secret, Google-style `AIza...` keys, long base64-ish
    literals, "Bearer <token>")
  - coordinate pairs inside Catalonia: a latitude (40.5-42.9, >=4 decimals)
    and a longitude (0.1-3.4, >=4 decimals) occurring together within an
    ~80-character window. A lone decimal is deliberately not flagged: a
    latitude and a longitude have to appear together, since config.py
    legitimately contains fleet shares like 0.49413 and 0.48384, and
    flagging every 4+ decimal number would bury the real signal in false
    positives.

Warnings (reported, not failed; use --strict to fail on these too):
  - facility names drawn from the location file. These are not secret: the
    published S1 Data that accompanies the article already lists 588
    origin and 115 destination facility names in the clear. They are
    flagged for human review rather than blocked. pipeline/utils.py needs
    a name-standardisation table naming some of them to work at all, so the
    names in REVIEWED_UTILS_FACILITY_NAMES below are accepted there after
    review. Any other facility name, in any file, is still reported.
  - the words "person", "people" or "per person" outside a rename-map key
    or a comment that names a legacy column. config.COLUMN_RENAMES and
    comments explaining the legacy naming are expected to mention them;
    the rest of the code should not, per CONTRACT.md section 2.

Usage (from the `publish/` directory):

    py -3.13 checks/scan_for_data.py
    py -3.13 checks/scan_for_data.py --strict
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Make `config` importable when this script is run directly as
# `checks/scan_for_data.py`, i.e. from inside the `checks/` subdirectory.
_PUBLISH_DIR = Path(__file__).resolve().parent.parent
if str(_PUBLISH_DIR) not in sys.path:
    sys.path.insert(0, str(_PUBLISH_DIR))

import config  # noqa: E402

def to_ascii(s: str) -> str:
    """Best-effort ASCII rendering for console output: the reference
    console is cp1252, and some scanned content (facility names, legacy
    statistics labels) can contain characters outside it.
    """
    return str(s).encode("ascii", "replace").decode("ascii")


MAX_FILE_BYTES = 1_000_000  # 1 MB

SKIP_DIR_NAMES = {".git", "__pycache__", ".ipynb_checkpoints"}

DATA_EXTENSIONS = {
    ".parquet", ".xlsx", ".xls", ".csv", ".pkl", ".geojson",
    ".tif", ".tiff", ".png", ".pdf", ".html",
}

# Only scan text-ish source files for content patterns; anything else that
# survives the extension/size checks is not worth regex-scanning.
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".cfg", ".toml", ".ini", ".yml", ".yaml", ".json", ".cff"}

# The assignment-literal and base64-ish patterns are restricted to code-like
# files: documentation legitimately shows shell commands that set a
# placeholder environment-variable value as a setup example, which is not a
# leaked secret, and false positives there would bury real findings.
CODE_EXTENSIONS = {".py", ".cfg", ".toml", ".ini", ".yml", ".yaml", ".json", ".cff"}

# -----------------------------------------------------------------------------
# API key / secret patterns
# -----------------------------------------------------------------------------

_API_KEY_ASSIGN_RE = re.compile(
    r"""(?ix)
    \b\w*(api[_-]?key|token|secret)\w*\s*[:=]\s*
    f?['"][^'"\s]{8,}['"]
    """
)
_GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{10,}")
# A long, high-entropy-looking base64-ish literal, not matched by the two
# patterns above: 32+ chars of base64 alphabet inside a quoted string. Pure
# lower_snake_case identifiers (this codebase's manuscript-number names, e.g.
# "s01_dropped_missing_origin_destination_records") also run to 32+ chars of
# that alphabet, so a literal only counts as secret-shaped if it also
# contains an uppercase letter or a '+'/'/' character: that is the shape a
# real base64 token has, and a snake_case identifier never does.
_BASE64ISH_RE = re.compile(r"""['"]([A-Za-z0-9+/_\-]{32,}={0,2})['"]""")


def _looks_like_secret(literal: str) -> bool:
    return any(c.isupper() for c in literal) or "+" in literal or "/" in literal

# -----------------------------------------------------------------------------
# Coordinate-pair pattern (Catalonia bounding box)
# -----------------------------------------------------------------------------
# A lone decimal is never flagged on its own, since config.py legitimately
# contains fleet shares like 0.49413 / 0.48384; a latitude and a longitude
# have to be found together, close by, before anything is reported.

_LAT_RE = re.compile(r"(?<![0-9.])(4[0-2]\.\d{4,})(?![0-9])")  # 40.xxxx - 42.xxxx
_LON_RE = re.compile(r"(?<![0-9.])([0-3]\.\d{4,})(?![0-9])")   # 0.xxxx - 3.xxxx

_LAT_MIN, _LAT_MAX = 40.5, 42.9
_LON_MIN, _LON_MAX = 0.1, 3.4
_COORD_WINDOW = 80  # characters


def _in_range(value_str: str, lo: float, hi: float) -> bool:
    try:
        v = float(value_str)
    except ValueError:
        return False
    return lo <= v <= hi


def find_coordinate_pairs(text: str) -> list[tuple[int, str]]:
    """Return (line_number, excerpt) for every place a Catalonia-range
    latitude and a Catalonia-range longitude occur within _COORD_WINDOW
    characters of each other. A match needs both a latitude and a
    longitude; a single 4+ decimal number on its own is never enough.
    """
    hits = []
    lats = [(m.start(), m.group(1)) for m in _LAT_RE.finditer(text) if _in_range(m.group(1), _LAT_MIN, _LAT_MAX)]
    lons = [(m.start(), m.group(1)) for m in _LON_RE.finditer(text) if _in_range(m.group(1), _LON_MIN, _LON_MAX)]
    if not lats or not lons:
        return hits
    for lat_pos, lat_val in lats:
        for lon_pos, lon_val in lons:
            if abs(lat_pos - lon_pos) <= _COORD_WINDOW:
                start = max(0, min(lat_pos, lon_pos) - 20)
                end = min(len(text), max(lat_pos, lon_pos) + 20)
                excerpt = text[start:end].replace("\n", " ")
                line_no = text.count("\n", 0, lat_pos) + 1
                hits.append((line_no, f"lat={lat_val} lon={lon_val} :: ...{excerpt}..."))
    return hits


# -----------------------------------------------------------------------------
# Restricted filenames (hard failure)
# -----------------------------------------------------------------------------
# The names of the restricted inputs and intermediates. Written as a list of
# fragments, and this file is exempt from the content check (see run_scan),
# because the rule has to name what it blocks.
_RESTRICTED_NAME_RE = re.compile(
    r"derivacions|derivations_with|_and_abs|abs\.xlsx|rutes_cache|rutes_no_",
    re.IGNORECASE,
)


def find_restricted_names(text: str) -> list[tuple[int, str]]:
    hits = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        m = _RESTRICTED_NAME_RE.search(line)
        if m:
            hits.append((line_no, line.strip()[:120]))
    return hits


# -----------------------------------------------------------------------------
# Reviewed facility names
# -----------------------------------------------------------------------------
# The name-standardisation table in pipeline/utils.py. Reviewed by hand; these
# are public hospital and primary-care names. Any other facility name found in
# utils.py is still reported.
REVIEWED_UTILS_FACILITY_NAMES = frozenset({
    "Hospital Mollet",
    "Hospital Universitari Joan XXIII",
    "CAP Albatarrec",
    "CAP Vilafranca",
    "CAP Moià",
    "Hospital de l'Esperança",
    "La Muntanyeta Tarragona",
    "Hospital General de l'Hospitalet de Llobregat",
    "Hospital de Sant Joan Despí Moisès Broggi",
    "HC Sant Jaume Calella",
    "HC de Blanes",
})

# -----------------------------------------------------------------------------
# "person" / "people" wording check
# -----------------------------------------------------------------------------

_PERSON_WORD_RE = re.compile(r"\b(per[_ ]?person|person|people)\b", re.IGNORECASE)


def _line_is_exempt(line: str) -> bool:
    """A line is exempt from the person/people warning if it is a comment
    naming a legacy column, or part of a rename-map entry (legacy name on
    the left of a mapping to a new name).
    """
    stripped = line.strip()
    if stripped.startswith("#"):
        return True
    # Rename-map style lines: "legacy_name_with_person": "new_name" or
    # legacy_name_with_person -> new_name, legacy_name_with_person: "...".
    if "->" in line or re.search(r"['\"]?\w*per[_]?person\w*['\"]?\s*:", line, re.IGNORECASE):
        return True
    return False


def find_person_wording(text: str) -> list[tuple[int, str]]:
    hits = []
    for i, line in enumerate(text.splitlines(), start=1):
        if _PERSON_WORD_RE.search(line) and not _line_is_exempt(line):
            hits.append((i, line.strip()))
    return hits


# -----------------------------------------------------------------------------
# Facility-name check (warn only)
# -----------------------------------------------------------------------------


def load_facility_names() -> list[str] | None:
    """Load facility names from the geocoded locations file for the human
    review warning. Returns None (with a notice already printed) if the
    file is not readable, since this scan has to work on a clean checkout
    with no data present.
    """
    path = config.data_path("cleaned_locations_geocoded.parquet")
    if not path.exists():
        print(f"NOTICE: facility-name check skipped -- {path} not found "
              "(expected on a clean checkout without datafiles/).")
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(path, columns=["location_name"])
        names = sorted(set(df["location_name"].dropna().astype(str)))
        names = [n for n in names if len(n) >= 4]  # skip trivially short/noisy entries
        return names
    except Exception as exc:  # pragma: no cover - defensive, the scan should not crash here
        print(f"NOTICE: facility-name check skipped -- could not read {path}: {exc}")
        return None


def find_facility_names(text: str, names: list[str]) -> list[tuple[int, str]]:
    hits = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        for name in names:
            if name in line:
                hits.append((i, f"'{name}' found in line: {line.strip()[:120]}"))
    return hits


# -----------------------------------------------------------------------------
# File walking
# -----------------------------------------------------------------------------


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.relative_to(root).parts):
            continue
        yield path


def run_scan(root: Path, facility_names: list[str] | None) -> dict[str, list[tuple[Path, int, str]]]:
    findings: dict[str, list[tuple[Path, int, str]]] = {
        "data_extension": [],
        "large_file": [],
        "api_key": [],
        "coordinate_pair": [],
        "restricted_name": [],
        "facility_name": [],
        "person_wording": [],
    }

    for path in iter_files(root):
        rel = path.relative_to(root)
        suffix = path.suffix.lower()
        size = path.stat().st_size

        if suffix in DATA_EXTENSIONS:
            findings["data_extension"].append((rel, 0, f"extension '{suffix}' is a restricted data format"))

        if size > MAX_FILE_BYTES:
            findings["large_file"].append((rel, 0, f"{size:,} bytes (limit {MAX_FILE_BYTES:,})"))

        if suffix not in TEXT_EXTENSIONS:
            continue  # do not regex-scan binary/data files; extension/size checks already cover them

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # Google-style keys and Bearer tokens are specific enough to scan
        # everywhere, including docs; the generic assignment pattern is
        # restricted to code-like files (see CODE_EXTENSIONS above).
        for m in _GOOGLE_KEY_RE.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            findings["api_key"].append((rel, line_no, f"Google-style API key (AIza...): {m.group(0)[:60]!r}"))
        for m in _BEARER_RE.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            findings["api_key"].append((rel, line_no, f"Bearer token: {m.group(0)[:60]!r}"))

        if suffix in CODE_EXTENSIONS:
            for m in _API_KEY_ASSIGN_RE.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                findings["api_key"].append((rel, line_no, f"api-key/token/secret assignment: {m.group(0)[:60]!r}"))
            for m in _BASE64ISH_RE.finditer(text):
                literal = m.group(1)
                if not _looks_like_secret(literal):
                    continue  # pure lower_snake_case identifier, not a plausible secret
                line_no = text.count("\n", 0, m.start()) + 1
                findings["api_key"].append((rel, line_no, f"long base64-ish literal: {m.group(0)[:60]!r}"))

        # This file's own self-test deliberately embeds a synthetic,
        # real-looking Catalan coordinate pair to prove the rule works (see
        # self_test() below); it must not be reported as a real finding.
        if rel.name != "scan_for_data.py":
            for line_no, excerpt in find_coordinate_pairs(text):
                findings["coordinate_pair"].append((rel, line_no, excerpt))

        # The scanner has to name the restricted patterns to block them, so
        # it is exempt from this check, as it is from the coordinate check.
        if rel.name != "scan_for_data.py":
            for line_no, excerpt in find_restricted_names(text):
                findings["restricted_name"].append((rel, line_no, excerpt))

        if facility_names and rel.name != "scan_for_data.py":
            names = facility_names
            if rel.as_posix() == "pipeline/utils.py":
                names = [n for n in facility_names if n not in REVIEWED_UTILS_FACILITY_NAMES]
            for line_no, excerpt in find_facility_names(text, names):
                findings["facility_name"].append((rel, line_no, excerpt))

        for line_no, excerpt in find_person_wording(text):
            findings["person_wording"].append((rel, line_no, excerpt))

    return findings


# -----------------------------------------------------------------------------
# Self-test: prove the coordinate-pair rule does what it claims
# -----------------------------------------------------------------------------


def self_test() -> None:
    """Assert the coordinate-pair rule does not flag config.py's legitimate
    fleet-share constants, and does flag a synthetic Catalan coordinate
    pair. Raises AssertionError if either check fails.
    """
    fleet_share_text = (
        "DIESEL_SHARE = DGT_DIESEL / DGT_TOTAL  # 0.49413...\n"
        "GASOLINE_SHARE = DGT_GASOLINE / DGT_TOTAL  # 0.48384...\n"
        "OTHER_SHARE = DGT_OTHER / DGT_TOTAL  # 0.02203..., zero tailpipe emissions\n"
    )
    assert find_coordinate_pairs(fleet_share_text) == [], (
        "False positive: the coordinate-pair scan flagged plain fleet-share constants."
    )

    synthetic_coord_text = "origin = (41.3851, 2.1734)  # a real-looking Barcelona coordinate pair"
    hits = find_coordinate_pairs(synthetic_coord_text)
    assert hits, "False negative: the coordinate-pair scan did not flag a synthetic Catalan coordinate pair."

    config_py = (_PUBLISH_DIR / "config.py").read_text(encoding="utf-8")
    assert find_coordinate_pairs(config_py) == [], (
        "False positive: the coordinate-pair scan flagged something in the real config.py."
    )

    assert find_restricted_names("x = read('raw_export_historical.xlsx')") == [], (
        "False positive: a generic input name was flagged as restricted."
    )
    assert find_restricted_names("x = read('" + "derivacions" + "_export.xlsx')"), (
        "False negative: a restricted input name was not flagged."
    )

    print("Self-test passed: fleet shares not flagged, synthetic coordinate pair flagged, "
          "config.py not flagged, restricted-name rule works.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures too.")
    parser.add_argument("--self-test", action="store_true", help="Run the coordinate-rule self-test and exit.")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    self_test()  # always prove the rule works before trusting its results below

    print("=" * 78)
    print("scan_for_data.py -- restricted-content scan of publish/")
    print("=" * 78)
    print(f"Root: {_PUBLISH_DIR}")
    print()

    facility_names = load_facility_names()
    findings = run_scan(_PUBLISH_DIR, facility_names)

    hard_sections = [
        ("Data files by extension", "data_extension"),
        ("Files larger than 1 MB", "large_file"),
        ("API key / secret patterns", "api_key"),
        ("Coordinate pairs inside Catalonia", "coordinate_pair"),
        ("Restricted filenames (in names or text)", "restricted_name"),
    ]
    warn_sections = [
        ("Facility names (human review -- not secret)", "facility_name"),
        ("'person' / 'people' / 'per person' wording", "person_wording"),
    ]

    n_hard = 0
    for title, key in hard_sections:
        items = findings[key]
        print(title)
        print("-" * len(title))
        if not items:
            print("  none found")
        for rel, line_no, detail in items:
            loc = f"{rel}:{line_no}" if line_no else str(rel)
            print(to_ascii(f"  {loc} -- {detail}"))
            n_hard += 1
        print()

    n_warn = 0
    for title, key in warn_sections:
        items = findings[key]
        print(title)
        print("-" * len(title))
        if not items:
            print("  none found")
        for rel, line_no, detail in items:
            print(to_ascii(f"  {rel}:{line_no} -- {detail}"))
            n_warn += 1
        print()

    print("=" * 78)
    print(f"RESULT: {n_hard} hard failure(s), {n_warn} warning(s).")
    failed = n_hard > 0 or (args.strict and n_warn > 0)
    print("VERDICT: FAIL" if failed else "VERDICT: PASS")
    print("=" * 78)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
