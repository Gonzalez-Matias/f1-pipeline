"""
Validacion de completitud del dataset F1.

Compara el calendario oficial (API Ergast) contra lo que ya existe
en los consolidados f1_all_*.parquet y en bronce.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from f1.config import OUTPUT_DIR
from f1.download import get_schedule, slugify

log = logging.getLogger(__name__)


def check_year_completeness(year: int, mode: str = "full", force: bool = False) -> dict:
    """
    Compara el calendario oficial contra lo que ya existe en Plata.

    Returns:
        {
            "year": int,
            "mode": str,
            "force": bool,
            "status": "complete" | "partial" | "missing" | "empty",
            "expected_count": int,
            "existing_count": int,
            "expected_gps": list[dict],
            "missing_gps": list[dict],   # en calendario pero no en all
            "extra_gps": list[int],      # en all pero no en calendario (rounds)
        }
    """
    # 1. Calendario oficial
    schedule = get_schedule(year)
    expected = [
        {"round": gp["round"], "slug": slugify(gp["race_name"]), "name": gp["race_name"]}
        for gp in schedule
    ]
    expected_count = len(expected)
    expected_rounds = {gp["round"] for gp in expected}

    # 2. Lo que ya existe en f1_all_results.parquet
    existing_rounds = set()
    all_results = OUTPUT_DIR / "f1_all_results.parquet"
    if all_results.exists():
        try:
            df = pd.read_parquet(all_results)
            year_df = df[df["Year"] == year]
            existing_rounds = set(year_df["RoundNumber"].dropna().astype(int).unique())
        except Exception as e:
            log.warning("Error leyendo %s: %s", all_results, e)

    existing_count = len(existing_rounds)

    # 3. Determinar estado
    missing = [gp for gp in expected if gp["round"] not in existing_rounds]
    extra = [r for r in existing_rounds if r not in expected_rounds]

    if expected_count == 0:
        status = "empty"
    elif existing_count == 0:
        status = "missing"
    elif existing_count == expected_count and not extra:
        status = "complete"
    else:
        status = "partial"

    result = {
        "year": year,
        "mode": mode,
        "force": force,
        "status": status,
        "expected_count": expected_count,
        "existing_count": existing_count,
        "expected_gps": expected,
        "missing_gps": missing,
        "extra_rounds": extra,
    }

    log.info(
        "Validacion %s: status=%s, expected=%s, existing=%s, missing=%s, extra=%s",
        year, status, expected_count, existing_count, len(missing), len(extra)
    )
    return result


def should_download(year: int, mode: str = "full", force: bool = False) -> dict:
    """
    Wrapper que decide si hay que descargar o no.
    Si force=True, siempre descarga.
    """
    check = check_year_completeness(year, mode, force)

    if force:
        log.info("force=True: se descarga %s aunque este completo", year)
        check["should_download"] = True
        check["reason"] = "force"
    elif check["status"] == "complete":
        log.info("%s esta completo (%s GPs). Se salta.", year, check["expected_count"])
        check["should_download"] = False
        check["reason"] = "complete"
    else:
        log.info("%s necesita descarga: %s GPs faltantes", year, len(check["missing_gps"]))
        check["should_download"] = True
        check["reason"] = check["status"]

    return check
