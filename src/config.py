"""Project-wide configuration: paths, constants, mappings."""
from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_CSV: Path = ROOT_DIR / "Crime_Data_from_2020_to_Present.csv"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FEATURES_DIR: Path = DATA_DIR / "features"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODELS_DIR: Path = REPORTS_DIR / "models"

CLEAN_PARQUET: Path = PROCESSED_DIR / "crimes_clean.parquet"
DAILY_AREA_PARQUET: Path = FEATURES_DIR / "daily_area_counts.parquet"
WEEKLY_AREA_PARQUET: Path = FEATURES_DIR / "weekly_area_counts.parquet"

for _p in (PROCESSED_DIR, FEATURES_DIR, FIGURES_DIR, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Schema knowledge
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "DR_NO", "Date Rptd", "DATE OCC", "TIME OCC", "AREA", "AREA NAME",
    "Rpt Dist No", "Part 1-2", "Crm Cd", "Crm Cd Desc", "Mocodes",
    "Vict Age", "Vict Sex", "Vict Descent", "Premis Cd", "Premis Desc",
    "Weapon Used Cd", "Weapon Desc", "Status", "Status Desc",
    "Crm Cd 1", "Crm Cd 2", "Crm Cd 3", "Crm Cd 4",
    "LOCATION", "Cross Street", "LAT", "LON",
]

# Memory-efficient dtypes for the raw CSV.
RAW_DTYPES = {
    "DR_NO": "int64",
    "TIME OCC": "int32",
    "AREA": "int8",
    "AREA NAME": "category",
    "Rpt Dist No": "string",
    "Part 1-2": "int8",
    "Crm Cd": "Int32",
    "Crm Cd Desc": "category",
    "Mocodes": "string",
    "Vict Age": "Int16",
    "Vict Sex": "category",
    "Vict Descent": "category",
    "Premis Cd": "Int32",
    "Premis Desc": "category",
    "Weapon Used Cd": "Int32",
    "Weapon Desc": "category",
    "Status": "category",
    "Status Desc": "category",
    "Crm Cd 1": "Int32",
    "Crm Cd 2": "Int32",
    "Crm Cd 3": "Int32",
    "Crm Cd 4": "Int32",
    "LOCATION": "string",
    "Cross Street": "string",
    "LAT": "float32",
    "LON": "float32",
}

DATE_COLUMNS = ["Date Rptd", "DATE OCC"]

# Renaming raw columns -> snake_case domain names.
COLUMN_RENAME = {
    "DR_NO": "dr_no",
    "Date Rptd": "date_reported",
    "DATE OCC": "date_occurred",
    "TIME OCC": "time_occurred",
    "AREA": "area_id",
    "AREA NAME": "area_name",
    "Rpt Dist No": "report_district",
    "Part 1-2": "part_class",
    "Crm Cd": "crime_code",
    "Crm Cd Desc": "crime_desc",
    "Mocodes": "mo_codes",
    "Vict Age": "victim_age",
    "Vict Sex": "victim_sex",
    "Vict Descent": "victim_descent",
    "Premis Cd": "premise_code",
    "Premis Desc": "premise_desc",
    "Weapon Used Cd": "weapon_code",
    "Weapon Desc": "weapon_desc",
    "Status": "status_code",
    "Status Desc": "status_desc",
    "Crm Cd 1": "crime_code_1",
    "Crm Cd 2": "crime_code_2",
    "Crm Cd 3": "crime_code_3",
    "Crm Cd 4": "crime_code_4",
    "LOCATION": "address",
    "Cross Street": "cross_street",
    "LAT": "lat",
    "LON": "lon",
}

# ---------------------------------------------------------------------------
# Domain knowledge
# ---------------------------------------------------------------------------
# LAPD Part 1 *violent* crime codes (FBI UCR definition):
#   - Homicide / manslaughter
#   - Forcible rape and sexual assault
#   - Robbery
#   - Aggravated assault / ADW
# Plus closely related codes (kidnapping, stalking, weapons / shots fired, criminal threats).
# Property-only Part 1 crimes (burglary, larceny, vehicle theft, arson) are excluded.
VIOLENT_CRIME_CODES = {
    110, 113,                                       # Homicide / Manslaughter
    121, 122, 815, 820, 821, 860,                   # Rape / forcible sex offenses
    210, 220,                                       # Robbery / Att. Robbery
    230, 231, 235, 236, 250, 251,                   # Aggravated assault / ADW
    237,                                            # Child abuse (severe)
    761, 762, 763,                                  # Weapons / shots fired
    627, 647, 928, 930,                             # Stalking / kidnapping / criminal threats
}

# Mapping for the LAPD `Status` codes.
STATUS_MAP = {
    "AA": "Adult Arrest",
    "AO": "Adult Other",
    "JA": "Juvenile Arrest",
    "JO": "Juvenile Other",
    "IC": "Investigation Continued",
    "CC": "Closed",
}
ARREST_STATUSES = {"AA", "JA"}

# LA bounding box (used to filter (0,0) and outlier coordinates).
LA_BBOX = {
    "lat_min": 33.60,
    "lat_max": 34.40,
    "lon_min": -118.80,
    "lon_max": -117.95,
}

# ---------------------------------------------------------------------------
# Modelling defaults
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET_VIOLENT = "is_violent"
TARGET_ARREST = "is_arrest"

# Crime risk score weights (used in src/risk_scoring.py).
RISK_WEIGHTS = {
    "intensity": 0.35,    # Recent crime volume
    "trend": 0.20,        # Short-term trend
    "violence": 0.20,     # Share of violent crimes
    "forecast": 0.25,     # Forecasted next-period intensity
}
