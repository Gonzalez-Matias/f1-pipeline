"""
Configuracion central del pipeline F1.

Estructura tipo medallon (patron FIFA):
  include/output/
    bronze/ergast/     ← JSON crudo Archive (persistente, append-only)
    bronze/fastf1/     ← metadata + tel FastF1 (persistente, append-only)
    silver/_parciales/ ← parquets por GP (temporales, se borran tras consolidar)
    f1_all_results.parquet  ← output final
    f1_all_full.parquet     ← output final (si mode=full)
"""
from pathlib import Path
import os

BASE_DIR = Path(os.getenv("F1_BASE_DIR", "/opt/airflow"))
OUTPUT_DIR = Path(os.getenv("F1_OUTPUT_DIR", str(BASE_DIR / "include" / "output")))

BRONZE_DIR = OUTPUT_DIR / "bronze"
SILVER_DIR = OUTPUT_DIR / "silver"
PARTIALS_DIR = SILVER_DIR / "_parciales"

BRONZE_ERGAST = BRONZE_DIR / "ergast"
BRONZE_FASTF1 = BRONZE_DIR / "fastf1"

ARCHIVE_REPO = "TracingInsights-Archive/Stats"
TRACING_REPO_TEMPLATE = "TracingInsights/{year}"
RAW_BASE = "https://raw.githubusercontent.com"

ARCHIVE_FILES = [
    "quali_results.json",
    "results.json",
    "laptimes.json",
    "pitstops.json",
    "driverPoints.json",
    "teamPoints.json",
    "event_info.json",
]

FASTF1_SESSIONS = ["Practice 1", "Practice 2", "Practice 3"]
FASTF1_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
RETRIES = 3
TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.1
MAX_DOWNLOAD_WORKERS = 8

YEAR_START = 2000
YEAR_END = 2026
