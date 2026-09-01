"""
Construye la capa Plata a partir de la capa Bronce.

Soporta dos modos:
  - results_only: solo columnas Archive (Ergast)
  - full: Archive + FastF1 (telemetria de practicas)

Para cada GP genera:
  - {gp_slug}_results.parquet  (siempre)
  - {gp_slug}_full.parquet     (solo si mode='full')

Consolidados:
  - f1_{year}_results.parquet  (siempre)
  - f1_{year}_full.parquet     (solo si mode='full')
  - f1_all_results.parquet     (siempre)
  - f1_all_full.parquet        (solo si mode='full')
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from f1.config import (
    BRONZE_ERGAST,
    BRONZE_FASTF1,
    FASTF1_COMPOUNDS,
    FASTF1_SESSIONS,
    OUTPUT_DIR,
    PARTIALS_DIR,
    SILVER_DIR,
)

log = logging.getLogger(__name__)


# ------------------------------------------------------------------- parseo archive


def parse_archive_gp(year: int, gp_slug: str) -> pd.DataFrame:
    """Lee los 7 JSON de Archive y devuelve DataFrame base."""
    base_dir = BRONZE_ERGAST / str(year) / gp_slug

    # --- results.json ---
    results_path = base_dir / "results.json"
    df_results = _parse_results(results_path)

    # Si no hay resultados de carrera (GP futuro), no se puede construir el registro
    if df_results.empty:
        return pd.DataFrame()

    # --- quali_results.json ---
    quali_path = base_dir / "quali_results.json"
    df_quali = _parse_quali(quali_path)

    # Los archivos de standings del repo representan el campeonato DESPUES
    # del GP indicado. Para features pre-carrera se usa el GP anterior.
    previous_slug = _get_previous_gp_slug(year, gp_slug)
    standings_dir = BRONZE_ERGAST / str(year) / previous_slug if previous_slug else None

    # --- driverPoints.json del GP anterior ---
    driver_points_path = standings_dir / "driverPoints.json" if standings_dir else None
    df_driver_pts = _parse_driver_points(driver_points_path)

    # --- teamPoints.json del GP anterior ---
    team_points_path = standings_dir / "teamPoints.json" if standings_dir else None
    df_team_pts = _parse_team_points(team_points_path)

    # Mergear todo
    df = df_results
    if not df_quali.empty:
        df = df.merge(df_quali, on="DriverNumber", how="left")
    if not df_driver_pts.empty:
        df = df.merge(df_driver_pts, on="DriverId", how="left")
    if not df_team_pts.empty:
        df = df.merge(df_team_pts, on="TeamId", how="left")

    # En la primera carrera no existe standings anterior.
    for column in ("DriverPoints_Before", "DriverWins_Before", "TeamPoints_Before", "TeamWins_Before"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    # Mantener el mismo esquema incluso en la primera carrera, donde no hay
    # standings previos.
    for column in (
        "DriverPosition_Before", "DriverPoints_Before", "DriverWins_Before",
        "TeamPosition_Before", "TeamPoints_Before", "TeamWins_Before",
    ):
        if column not in df.columns:
            df[column] = 0

    for column in (
        "GridPosition", "QualyPosition", "Race_Position",
        "Race_Points", "Race_Laps", "DriverPosition_Before",
        "TeamPosition_Before",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Agregar metadata del GP
    df["Year"] = year
    df["RoundNumber"] = _get_round_from_slug(year, gp_slug)

    return df


def _parse_results(path: Path) -> pd.DataFrame:
    """Parsea results.json -> DataFrame con resultados de carrera."""
    if path is None or not path.exists():
        return pd.DataFrame()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return pd.DataFrame()

    results = races[0].get("Results", [])
    rows = []
    for r in results:
        rows.append({
            "DriverNumber": r.get("number"),
            "DriverId": r.get("Driver", {}).get("driverId"),
            "TeamId": r.get("Constructor", {}).get("constructorId"),
            "Race_Position": r.get("position"),
            "Race_ClassifiedPosition": r.get("positionText"),
            "Race_Status": r.get("status"),
            "Race_Points": r.get("points"),
            "Race_Laps": r.get("laps"),
            "GridPosition": r.get("grid"),
        })

    return pd.DataFrame(rows)


def _parse_quali(path: Path) -> pd.DataFrame:
    """Parsea quali_results.json -> DataFrame con grid y tiempos de qualy."""
    if path is None or not path.exists():
        return pd.DataFrame()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return pd.DataFrame()

    quali = races[0].get("QualifyingResults", [])
    rows = []
    for q in quali:
        rows.append({
            "DriverNumber": q.get("number"),
            "QualyPosition": q.get("position"),
            "QualyQ1": q.get("Q1"),
            "QualyQ2": q.get("Q2"),
            "QualyQ3": q.get("Q3"),
        })

    return pd.DataFrame(rows)


def _parse_driver_points(path: Path) -> pd.DataFrame:
    """Parsea driverPoints.json -> standings de pilotos antes del GP."""
    if path is None or not path.exists():
        return pd.DataFrame()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    standings = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    if not standings:
        return pd.DataFrame()

    drivers = standings[0].get("DriverStandings", [])
    rows = []
    for d in drivers:
        rows.append({
            "DriverId": d.get("Driver", {}).get("driverId"),
            "DriverPosition_Before": d.get("position"),
            "DriverPoints_Before": d.get("points"),
            "DriverWins_Before": d.get("wins"),
        })

    return pd.DataFrame(rows)


def _parse_team_points(path: Path) -> pd.DataFrame:
    """Parsea teamPoints.json -> standings de constructores antes del GP."""
    if path is None or not path.exists():
        return pd.DataFrame()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    standings = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    if not standings:
        return pd.DataFrame()

    constructors = standings[0].get("ConstructorStandings", [])
    rows = []
    for c in constructors:
        rows.append({
            "TeamId": c.get("Constructor", {}).get("constructorId"),
            "TeamPosition_Before": c.get("position"),
            "TeamPoints_Before": c.get("points"),
            "TeamWins_Before": c.get("wins"),
        })

    return pd.DataFrame(rows)


def _get_round_from_slug(year: int, gp_slug: str) -> int:
    """Intenta obtener el round number del manifest.json."""
    for base in [BRONZE_ERGAST, BRONZE_FASTF1]:
        manifest_path = base / str(year) / gp_slug / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    m = json.load(f)
                return m.get("round", 0)
            except:
                continue
    return 0


def _get_previous_gp_slug(year: int, gp_slug: str) -> str | None:
    """Busca el GP anterior usando el round guardado en sus manifests."""
    current_round = _get_round_from_slug(year, gp_slug)
    if current_round <= 1:
        return None

    year_dir = BRONZE_ERGAST / str(year)
    if not year_dir.exists():
        return None

    for candidate in year_dir.iterdir():
        if not candidate.is_dir() or candidate.name == gp_slug:
            continue
        if _get_round_from_slug(year, candidate.name) == current_round - 1:
            return candidate.name
    return None


# ------------------------------------------------------------------- parseo fastf1


def parse_fastf1_gp(year: int, gp_slug: str) -> pd.DataFrame:
    """Parsea datos de practicas FastF1 y devuelve DataFrame con features."""
    if year < 2018:
        return pd.DataFrame()

    base_dir = BRONZE_FASTF1 / str(year) / gp_slug
    if not base_dir.exists():
        return pd.DataFrame()

    all_features = []
    for session in FASTF1_SESSIONS:
        session_dir = base_dir / session
        if not session_dir.exists():
            continue

        laptimes_path = session_dir / "session_laptimes.json"
        if not laptimes_path.exists():
            continue

        df_session = _parse_practice_session(laptimes_path, session, year, gp_slug)
        if not df_session.empty:
            all_features.append(df_session)

    if not all_features:
        return pd.DataFrame()

    df_final = all_features[0]
    for df in all_features[1:]:
        df_final = df_final.merge(df, on="DriverNumber", how="outer")

    return df_final


def _load_driver_mapping(session_dir: Path) -> dict[str, str]:
    """Lee drivers.json y devuelve mapeo {abreviatura: numero}."""
    drivers_path = session_dir / "drivers.json"
    if not drivers_path.exists():
        return {}

    try:
        with open(drivers_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapping = {}
        for d in data.get("drivers", []):
            abbr = d.get("driver")
            num = d.get("dn")
            if abbr and num:
                mapping[abbr] = str(num)
        return mapping
    except Exception:
        return {}


def _parse_practice_session(
    laptimes_path: Path, session_name: str, year: int, gp_slug: str
) -> pd.DataFrame:
    """Parsea una sesion de practica y calcula features."""
    session_dir = laptimes_path.parent
    driver_mapping = _load_driver_mapping(session_dir)

    with open(laptimes_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return pd.DataFrame()

    n = len(data.get("lap", []))
    if n == 0:
        return pd.DataFrame()

    def val(key, i):
        v = data.get(key, [])
        return v[i] if i < len(v) else None

    laps = []
    for i in range(n):
        compound = val("compound", i)
        if compound not in FASTF1_COMPOUNDS:
            continue

        lap_time = val("time", i)
        if lap_time == "None" or lap_time is None:
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
        except:
            continue

        deleted = val("del", i)
        if deleted is True:
            continue

        laps.append({
            "driver": val("drv", i),
            "lap": int(val("lap", i)) if val("lap", i) is not None else None,
            "compound": compound,
            "time": float(lap_time),
            "session_dir": str(laptimes_path.parent),
        })

    grouped = defaultdict(list)
    for l in laps:
        grouped[(l["driver"], l["compound"])].append(l)

    filtered = []
    for key, group_laps in grouped.items():
        min_time = min(l["time"] for l in group_laps)
        threshold = min_time * 1.05
        for l in group_laps:
            if l["time"] <= threshold:
                filtered.append(l)

    if not filtered:
        return pd.DataFrame()

    session_prefix = session_name.replace("Practice ", "FP")
    features = []

    by_driver = defaultdict(list)
    for l in filtered:
        by_driver[l["driver"]].append(l)

    for driver_abbr, driver_laps in by_driver.items():
        driver_num = driver_mapping.get(driver_abbr, driver_abbr)
        row = {"DriverNumber": driver_num}

        # Tiempos por compuesto (solo SOFT y MEDIUM; HARD no se usa en practicas)
        for compound in ("SOFT", "MEDIUM"):
            compound_laps = [l for l in driver_laps if l["compound"] == compound]
            if compound_laps:
                times = [l["time"] for l in compound_laps]
                row[f"{session_prefix}_LapTime_min_{compound}"] = min(times)
                row[f"{session_prefix}_LapTime_mean_{compound}"] = sum(times) / len(times)

        abs_lap = min(driver_laps, key=lambda x: x["time"])
        tel_abs = _read_telemetry(abs_lap["session_dir"], driver_abbr, abs_lap["lap"])
        if tel_abs:
            row[f"{session_prefix}_Throttle_abs"] = tel_abs["throttle"]
            row[f"{session_prefix}_Speed_abs"] = tel_abs["speed"]
            row[f"{session_prefix}_RPM_abs"] = tel_abs["rpm"]
            row[f"{session_prefix}_Brake_abs"] = tel_abs["brake"]

        medium_laps = [l for l in driver_laps if l["compound"] == "MEDIUM"]
        if medium_laps:
            tel_values = {"throttle": [], "speed": [], "rpm": [], "brake": []}
            for l in medium_laps:
                tel = _read_telemetry(l["session_dir"], driver_abbr, l["lap"])
                if tel:
                    for k in tel_values:
                        if tel[k] is not None:
                            tel_values[k].append(tel[k])

            if tel_values["throttle"]:
                row[f"{session_prefix}_Throttle_mean_MEDIUM"] = sum(tel_values["throttle"]) / len(tel_values["throttle"])
            if tel_values["speed"]:
                row[f"{session_prefix}_Speed_mean_MEDIUM"] = sum(tel_values["speed"]) / len(tel_values["speed"])
            if tel_values["rpm"]:
                row[f"{session_prefix}_RPM_mean_MEDIUM"] = sum(tel_values["rpm"]) / len(tel_values["rpm"])
            if tel_values["brake"]:
                row[f"{session_prefix}_Brake_mean_MEDIUM"] = sum(tel_values["brake"]) / len(tel_values["brake"])

        features.append(row)

    return pd.DataFrame(features)


def _read_telemetry(session_dir: str, driver: str, lap_num: int) -> dict | None:
    """Lee tel.json de bronce y calcula promedios."""
    path = Path(session_dir) / driver / f"{lap_num}_tel.json"
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tel = data.get("tel", {})
        if not tel:
            return None

        throttle = tel.get("throttle", [])
        speed = tel.get("speed", [])
        rpm = tel.get("rpm", [])
        brake = tel.get("brake", [])

        throttle_valid = [t for t in throttle if 0 <= t <= 100]

        return {
            "throttle": sum(throttle_valid) / len(throttle_valid) if throttle_valid else None,
            "speed": sum(speed) / len(speed) if speed else None,
            "rpm": sum(rpm) / len(rpm) if rpm else None,
            "brake": sum(brake) / len(brake) * 100 if brake else None,
        }
    except Exception:
        return None


# ------------------------------------------------------------------- build gp


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reordena columnas en orden logico."""
    id_cols = ["DriverId", "DriverNumber", "TeamId", "Year", "RoundNumber"]
    qualy_cols = [c for c in df.columns if c.startswith("Qualy")]
    standings_cols = [c for c in df.columns if c.endswith("_Before")]
    fp_cols = [c for c in df.columns if c.startswith(("FP1_", "FP2_", "FP3_"))]
    race_cols = [c for c in df.columns if c.startswith("Race_")]

    ordered = (
        [c for c in id_cols if c in df.columns]
        + qualy_cols
        + standings_cols
        + fp_cols
        + race_cols
    )
    remaining = [c for c in df.columns if c not in ordered]
    final_cols = ordered + remaining
    return df[[c for c in final_cols if c in df.columns]]


def build_gp_silver(year: int, round_num: int, gp_slug: str, mode: str = "full") -> dict[str, str]:
    """
    Construye el DataFrame de plata para un GP.
    
    Args:
        mode: 'results_only' o 'full'
        
    Returns:
        Dict con rutas de los archivos generados.
    """
    log.info("Construyendo Plata para %s/%s (mode=%s)", year, gp_slug, mode)
    
    result_paths = {}

    # 1. Datos de Archive (siempre)
    df_archive = parse_archive_gp(year, gp_slug)
    if df_archive.empty:
        log.warning("No hay datos de Archive para %s/%s", year, gp_slug)
        return result_paths

    # Guardar _results (siempre)
    df_results = _reorder_columns(df_archive)
    results_path = _save_gp_silver(year, gp_slug, df_results, suffix="results")
    result_paths["results"] = results_path

    # 2. Datos de FastF1 (solo si mode='full')
    if mode == "full":
        df_fastf1 = parse_fastf1_gp(year, gp_slug)
        if not df_fastf1.empty:
            df_full = df_archive.merge(df_fastf1, on="DriverNumber", how="left")
        else:
            # Si no hay FastF1 (pre-2018 o sin datos), usar archive como full
            # para mantener consistencia de filas en el consolidado
            df_full = df_archive.copy()
        df_full = _reorder_columns(df_full)
        full_path = _save_gp_silver(year, gp_slug, df_full, suffix="full")
        result_paths["full"] = full_path

    return result_paths


def _save_gp_silver(year: int, gp_slug: str, df: pd.DataFrame, suffix: str) -> str:
    """Guarda el DataFrame de un GP en silver/_parciales/ (temporal)."""
    dest_dir = PARTIALS_DIR / str(year)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{gp_slug}_{suffix}.parquet"
    df.to_parquet(dest, index=False)
    return str(dest)


# ------------------------------------------------------------------- consolidate


def consolidate_year(year: int, mode: str = "full") -> dict[str, str]:
    """
    Une todos los GPs de un ano en uno o dos parquet segun mode.
    
    Returns:
        Dict con rutas de consolidados generados.
    """
    result_paths = {}
    partials_dir = PARTIALS_DIR / str(year)

    if not partials_dir.exists():
        log.warning("No hay datos en Plata para %s", year)
        return result_paths

    # Consolidar _results (siempre)
    result_files = sorted(partials_dir.glob("*_results.parquet"))
    if result_files:
        dfs = [pd.read_parquet(f) for f in result_files]
        df = pd.concat(dfs, ignore_index=True)
        dest = SILVER_DIR / f"f1_{year}_results.parquet"
        df.to_parquet(dest, index=False)
        log.info("Consolidado %s results: %s filas x %s columnas -> %s", year, len(df), len(df.columns), dest)
        result_paths["results"] = str(dest)
    else:
        log.warning("No hay archivos _results para %s", year)

    # Consolidar _full (solo si mode='full')
    if mode == "full":
        full_files = sorted(partials_dir.glob("*_full.parquet"))
        if full_files:
            dfs = [pd.read_parquet(f) for f in full_files]
            df = pd.concat(dfs, ignore_index=True)
            dest = SILVER_DIR / f"f1_{year}_full.parquet"
            df.to_parquet(dest, index=False)
            log.info("Consolidado %s full: %s filas x %s columnas -> %s", year, len(df), len(df.columns), dest)
            result_paths["full"] = str(dest)
        else:
            log.info("No hay archivos _full para %s", year)

    return result_paths


def _upsert_parquet(existing_path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    """Combina new_df con el parquet existente.

    Reemplaza las filas de los GPs presentes en new_df (identificados por
    Year + RoundNumber) y conserva el resto del historico. Evita la perdida
    de datos al consolidar rangos parciales o un unico GP.
    """
    if new_df.empty:
        if existing_path.exists():
            return pd.read_parquet(existing_path)
        return new_df

    if not existing_path.exists():
        return new_df

    if "Year" not in new_df.columns or "RoundNumber" not in new_df.columns:
        return pd.concat([pd.read_parquet(existing_path), new_df], ignore_index=True)

    existing = pd.read_parquet(existing_path)

    new_keys = pd.MultiIndex.from_frame(
        new_df[["Year", "RoundNumber"]].astype(str)
    )
    existing_keys = pd.MultiIndex.from_frame(
        existing[["Year", "RoundNumber"]].astype(str)
    )
    mask = ~existing_keys.isin(new_keys)

    kept = existing[mask]
    return pd.concat([kept, new_df], ignore_index=True)


def consolidate_all(mode: str = "full", cleanup: bool = True) -> dict[str, str]:
    """
    Une TODOS los GPs de TODOS los anos en uno o dos parquet.
    Lee de silver/_parciales/, escribe en output/, y limpia parciales.

    Hace upsert sobre el consolidado existente para no perder anos que no
    estan en los parciales actuales (p.ej. al correr un rango parcial).

    Returns:
        Dict con rutas de consolidados generados.
    """
    result_paths = {}

    # Consolidar _results (siempre)
    all_result_files = sorted(PARTIALS_DIR.glob("*/*_results.parquet"))
    if all_result_files:
        new_df = pd.concat(
            [pd.read_parquet(f) for f in all_result_files], ignore_index=True
        )
        dest = OUTPUT_DIR / "f1_all_results.parquet"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df = _upsert_parquet(dest, new_df)
        df.to_parquet(dest, index=False)
        log.info("Consolidado all results: %s GPs nuevos, %s filas x %s columnas -> %s",
                 len(all_result_files), len(df), len(df.columns), dest)
        result_paths["results"] = str(dest)
    else:
        log.warning("No hay archivos _results para consolidar")

    # Consolidar _full (solo si mode='full')
    if mode == "full":
        all_full_files = sorted(PARTIALS_DIR.glob("*/*_full.parquet"))
        if all_full_files:
            new_df = pd.concat(
                [pd.read_parquet(f) for f in all_full_files], ignore_index=True
            )
            dest = OUTPUT_DIR / "f1_all_full.parquet"
            df = _upsert_parquet(dest, new_df)
            df.to_parquet(dest, index=False)
            log.info("Consolidado all full: %s GPs nuevos, %s filas x %s columnas -> %s",
                     len(all_full_files), len(df), len(df.columns), dest)
            result_paths["full"] = str(dest)
        else:
            log.info("No hay archivos _full para consolidar")

    # Limpiar parciales
    if cleanup and PARTIALS_DIR.exists():
        import shutil
        shutil.rmtree(PARTIALS_DIR)
        log.info("Parciales limpiados: %s", PARTIALS_DIR)

    return result_paths


# ------------------------------------------------------------------- main (CLI backward compat)


def main(years: list[int] | None = None, mode: str = "full") -> None:
    if years is None:
        years = list(range(2000, 2027))

    for year in years:
        log.info("=" * 50)
        log.info("Procesando ano %s (mode=%s)", year, mode)

        if year >= 2018:
            base_dir = BRONZE_FASTF1 / str(year)
        else:
            base_dir = BRONZE_ERGAST / str(year)

        if not base_dir.exists():
            log.warning("No hay datos en Bronce para %s", year)
            continue

        gp_slugs = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
        log.info("%s GPs en bronce para %s", len(gp_slugs), year)

        for gp_slug in gp_slugs:
            round_num = _get_round_from_slug(year, gp_slug)
            build_gp_silver(year, round_num, gp_slug, mode=mode)

        consolidate_year(year, mode=mode)

    consolidate_all(mode=mode)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Construye capa plata F1")
    parser.add_argument("--year", type=int, action="append", help="Ano especifico")
    parser.add_argument("--mode", choices=["results_only", "full"], default="full", help="Modo de construccion")
    args = parser.parse_args()

    years = args.year if args.year else None
    main(years=years, mode=args.mode)
