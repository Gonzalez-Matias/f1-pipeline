"""
Descarga datos crudos de los repos de GitHub a la capa bronce.

Solo descarga lo que falta (idempotente). Para FastF1 baja metadata y la
telemetria selectiva necesaria para Plata: la vuelta mas rapida por piloto y
todas las vueltas MEDIUM que pasan los filtros.

Fuentes:
  - TracingInsights-Archive/Stats : 2000-2026, JSON estilo Ergast
  - TracingInsights/{year}        : 2018-2026, sesiones y telemetria FastF1
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

from f1.config import (
    ARCHIVE_FILES,
    ARCHIVE_REPO,
    BRONZE_ERGAST,
    BRONZE_FASTF1,
    FASTF1_SESSIONS,
    HEADERS,
    MAX_DOWNLOAD_WORKERS,
    RAW_BASE,
    RETRIES,
    SLEEP_BETWEEN_REQUESTS,
    TIMEOUT,
    TRACING_REPO_TEMPLATE,
    YEAR_END,
    YEAR_START,
)

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ utilidades


def slugify(name: str) -> str:
    return name.lower().strip().replace(" ", "-")


def _fetch(url: str, retries: int = RETRIES) -> bytes | None:
    for intento in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            log.warning("HTTP %s para %s (intento %s/%s)", e.code, url, intento + 1, retries)
        except Exception as e:
            log.warning("Error descargando %s: %s (intento %s/%s)", url, e, intento + 1, retries)
        if intento < retries - 1:
            time.sleep(2 * (intento + 1))
    return None


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    if dest.exists() and not force:
        return True

    data = _fetch(url)
    if data is None:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)

    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return True


# ------------------------------------------------------------------- calendario


def get_schedule(year: int) -> list[dict]:
    url = f"https://api.jolpi.ca/ergast/f1/{year}.json"
    data = _fetch(url)
    if data is None:
        log.error("No se pudo obtener el calendario de %s", year)
        return []

    try:
        payload = json.loads(data)
        races = payload["MRData"]["RaceTable"]["Races"]
    except (KeyError, json.JSONDecodeError) as e:
        log.error("Parseo fallido del calendario %s: %s", year, e)
        return []

    return [
        {
            "round": int(r["round"]),
            "race_name": r["raceName"],
            "circuit_id": r["Circuit"]["circuitId"],
            "date": r["date"],
        }
        for r in races
    ]


# --------------------------------------------------------------- archive repo


def download_gp_archive(year: int, race_name: str, slug: str, force: bool = False) -> dict:
    """Descarga los 7 JSON de Archive para un GP."""
    encoded_slug = quote(slug, safe="")
    base_url = f"{RAW_BASE}/{ARCHIVE_REPO}/main/{year}/{encoded_slug}"
    dest_dir = BRONZE_ERGAST / str(year) / slug
    dest_dir.mkdir(parents=True, exist_ok=True)

    ok_count, missing_count = 0, 0
    for fname in ARCHIVE_FILES:
        url = f"{base_url}/{fname}"
        dest = dest_dir / fname
        if download_file(url, dest, force=force):
            ok_count += 1
        else:
            missing_count += 1
            log.debug("No se encontro %s/%s/%s", year, slug, fname)

    return {
        "dest_dir": str(dest_dir),
        "downloaded": ok_count,
        "missing": missing_count,
    }


# --------------------------------------------------------------- fastf1 repo


def _fastf1_session_url(year: int, race_name: str, session: str, path: str) -> str:
    repo = TRACING_REPO_TEMPLATE.format(year=year)
    return (
        f"{RAW_BASE}/{repo}/main/"
        f"{quote(race_name, safe='')}/"
        f"{quote(session, safe='')}/"
        f"{path}"
    )


def filter_session_laps(session_path: Path) -> tuple[dict[str, dict], list[dict]]:
    """Lee session_laptimes.json y filtra vueltas validas.

    Devuelve:
      - vueltas_abs: dict {driver: vuelta_mas_rapida} por piloto (cualquier compuesto)
      - vueltas_medium: lista de dicts con todas las vueltas MEDIUM filtradas
    """
    from f1.config import FASTF1_COMPOUNDS

    if not session_path.exists():
        return {}, []

    try:
        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}, []

    if not isinstance(data, dict):
        return {}, []

    n = len(data.get("lap", []))
    if n == 0:
        return {}, []

    def val(key, i):
        v = data.get(key, [])
        return v[i] if i < len(v) else None

    # Fase 1: filtro base
    candidates = []
    for i in range(n):
        compound = val("compound", i)
        if compound not in FASTF1_COMPOUNDS:
            continue

        pin = val("pin", i)
        pout = val("pout", i)
        if pin != "None" or pout != "None":
            continue

        status = val("status", i)
        if status != "1":
            continue

        iacc = val("iacc", i)
        if iacc is not True:
            continue

        vfl = val("vfl", i)
        if vfl == "None" or vfl is None:
            continue
        try:
            if float(vfl) < 100:
                continue
        except (ValueError, TypeError):
            continue

        lap_time = val("time", i)
        if lap_time == "None" or lap_time is None:
            continue

        deleted = val("del", i)
        if deleted is True:
            continue

        candidates.append({
            "driver": val("drv", i),
            "lap": int(val("lap", i)) if val("lap", i) is not None else None,
            "compound": compound,
            "time": float(lap_time) if lap_time != "None" else None,
            "session_dir": str(session_path.parent),
        })

    # Fase 2: outlier 105% por piloto + compuesto
    from collections import defaultdict
    grouped = defaultdict(list)
    for c in candidates:
        if c["time"] is not None:
            grouped[(c["driver"], c["compound"])].append(c)

    filtered = []
    for key, laps in grouped.items():
        min_time = min(l["time"] for l in laps)
        threshold = min_time * 1.05
        for l in laps:
            if l["time"] <= threshold:
                filtered.append(l)

    # Identificar vuelta absoluta mas rapida POR PILOTO
    vueltas_abs = {}
    by_driver = defaultdict(list)
    for l in filtered:
        by_driver[l["driver"]].append(l)
    
    for driver, laps in by_driver.items():
        vueltas_abs[driver] = min(laps, key=lambda x: x["time"])

    # Filtrar solo MEDIUM
    vueltas_medium = [l for l in filtered if l["compound"] == "MEDIUM"]

    return vueltas_abs, vueltas_medium


def download_telemetry_laps(
    year: int, race_name: str, session: str, slug: str,
    laps_to_download: list[dict], force: bool = False
) -> dict:
    """Descarga tel.json para una lista de vueltas."""
    downloaded, skipped, missing = 0, 0, 0

    for lap_info in laps_to_download:
        driver = lap_info["driver"]
        lap_num = lap_info["lap"]
        if driver is None or lap_num is None:
            continue

        session_dir = BRONZE_FASTF1 / str(year) / slug / session
        driver_dir = session_dir / driver
        driver_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{lap_num}_tel.json"
        dest = driver_dir / fname

        if dest.exists() and not force:
            skipped += 1
            continue

        url = _fastf1_session_url(year, race_name, session, f"{driver}/{fname}")
        if download_file(url, dest, force=force):
            downloaded += 1
        else:
            missing += 1

    return {"downloaded": downloaded, "skipped": skipped, "missing": missing}


def download_gp_fastf1(year: int, race_name: str, slug: str, force: bool = False) -> dict:
    """Descarga metadata + tel.json selectivos (absoluta + MEDIUM)."""
    files_downloaded, files_missing = 0, 0
    sessions_ok = 0
    tel_downloaded, tel_skipped, tel_missing = 0, 0, 0

    for session in FASTF1_SESSIONS:
        session_dir = BRONZE_FASTF1 / str(year) / slug / session
        session_dir.mkdir(parents=True, exist_ok=True)

        # Solo estos dos archivos son necesarios para procesar vueltas.
        # corners y rcm son metadata opcional.
        session_ok = True
        for fname in ("session_laptimes.json", "drivers.json", "corners.json", "rcm.json"):
            url = _fastf1_session_url(year, race_name, session, fname)
            dest = session_dir / fname
            if download_file(url, dest, force=force):
                files_downloaded += 1
            else:
                files_missing += 1
                if fname in ("session_laptimes.json", "drivers.json"):
                    session_ok = False

        if not session_ok:
            continue

        sessions_ok += 1

        # Filtrar vueltas y descargar tel.json necesarios
        laptimes_path = session_dir / "session_laptimes.json"
        vueltas_abs, vueltas_medium = filter_session_laps(laptimes_path)

        laps_to_download = list(vueltas_abs.values())
        laps_to_download.extend(vueltas_medium)

        if laps_to_download:
            tel_res = download_telemetry_laps(
                year, race_name, session, slug, laps_to_download, force=force
            )
            tel_downloaded += tel_res["downloaded"]
            tel_skipped += tel_res["skipped"]
            tel_missing += tel_res["missing"]

    return {
        "sessions": sessions_ok,
        "files_downloaded": files_downloaded,
        "files_missing": files_missing,
        "tel_downloaded": tel_downloaded,
        "tel_skipped": tel_skipped,
        "tel_missing": tel_missing,
    }


# --------------------------------------------------------------------- manifest


def write_manifest(
    dest_dir: Path,
    year: int,
    round_num: int,
    gp_slug: str,
    archive_res: dict,
    fastf1_res: dict,
) -> None:
    manifest = {
        "year": year,
        "round": round_num,
        "gp_slug": gp_slug,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "archive": archive_res,
        "fastf1": fastf1_res,
    }
    manifest_path = dest_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------- single gp


def process_single_gp(
    year: int, round_num: int, race_name: str, force: bool = False
) -> dict:
    """Procesa UN GP completo. Funcion pura, lista para convertir en @task."""
    slug = slugify(race_name)
    log.info("GP %s (%s) - %s", round_num, slug, race_name)

    # Archive (todos los anos)
    archive_res = download_gp_archive(year, race_name, slug, force=force)
    log.info("  Archive: %s descargados, %s faltantes", archive_res["downloaded"], archive_res["missing"])

    # FastF1 metadata + telemetry (2018+)
    fastf1_res = {"sessions": 0, "files_downloaded": 0, "files_missing": 0, "tel_downloaded": 0, "tel_skipped": 0, "tel_missing": 0}
    if year >= 2018:
        fastf1_res = download_gp_fastf1(year, race_name, slug, force=force)
        log.info(
            "  FastF1: %s sesiones, %s metadata descargados (%s faltantes), %s tel descargados (%s skip, %s miss)",
            fastf1_res["sessions"],
            fastf1_res["files_downloaded"],
            fastf1_res["files_missing"],
            fastf1_res["tel_downloaded"],
            fastf1_res["tel_skipped"],
            fastf1_res["tel_missing"],
        )

    # Manifest
    dest_dir = BRONZE_FASTF1 / str(year) / slug if year >= 2018 else BRONZE_ERGAST / str(year) / slug
    write_manifest(dest_dir, year, round_num, slug, archive_res, fastf1_res)

    return {
        "year": year,
        "round": round_num,
        "slug": slug,
        "archive": archive_res,
        "fastf1": fastf1_res,
    }


# --------------------------------------------------------------------- main


def main(years: list[int] | None = None, force: bool = False) -> None:
    if years is None:
        years = list(range(YEAR_START, YEAR_END + 1))

    # Recolectar todos los GPs
    gps_to_process = []
    for year in years:
        log.info("=" * 50)
        log.info("Descubriendo calendario %s", year)
        schedule = get_schedule(year)
        if not schedule:
            log.warning("Sin calendario para %s, se salta", year)
            continue
        log.info("%s GPs en %s", len(schedule), year)
        for gp in schedule:
            gps_to_process.append((year, gp["round"], gp["race_name"]))

    total = len(gps_to_process)
    log.info("=" * 50)
    log.info("Total de GPs a procesar: %s", total)
    log.info("Workers: %s", MAX_DOWNLOAD_WORKERS)
    log.info("=" * 50)

    start_time = time.time()
    completed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
        future_to_gp = {
            executor.submit(process_single_gp, year, rnd, name, force): (year, rnd, name)
            for year, rnd, name in gps_to_process
        }

        for future in as_completed(future_to_gp):
            year, rnd, name = future_to_gp[future]
            try:
                future.result()
                completed += 1
            except Exception as e:
                errors += 1
                log.error("Error en GP %s/%s (%s): %s", year, rnd, name, e)

            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                log.info("Progreso: %s/%s GPs (%s errores) - %.2f GPs/s", completed, total, errors, rate)

    elapsed = time.time() - start_time
    log.info("=" * 50)
    log.info("Fin. %s/%s GPs procesados en %.1fs (%.2f GPs/s). Errores: %s", completed, total, elapsed, completed / elapsed if elapsed > 0 else 0, errors)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Descarga capa bronce F1")
    parser.add_argument("--year", type=int, action="append", help="Ano especifico (se puede repetir)")
    parser.add_argument("--force", action="store_true", help="Forzar re-descarga de todo")
    args = parser.parse_args()

    years = args.year if args.year else None
    main(years=years, force=args.force)
