"""
Construye la capa Plata a partir de la capa Bronce.

Para cada GP genera (parciales, formato interno en parquet):
  - {gp_slug}_full.parquet

Consolidado (formato final en Parquet):
  - f1_all_full.parquet
  - f1_all_postEDA.parquet (ver f1/posteda.py)
"""
from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

from f1.config import (
    BRONZE_ERGAST,
    BRONZE_FASTF1,
    FASTF1_SESSIONS,
    OUTPUT_DIR,
    PARTIALS_DIR,
)
from f1.laps import filter_outliers, valid_laps
from f1.posteda import build_post_eda

log = logging.getLogger(__name__)


# ------------------------------------------------------------------- parseo archive


def parse_archive_gp(year: int, gp_slug: str) -> pd.DataFrame:
    """Lee los JSON de Archive y devuelve DataFrame base."""
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

    # Standings previos: en la primera carrera no existen; se completan a 0.
    for column in (
        "DriverPosition_Before", "DriverPoints_Before",
        "TeamPosition_Before", "TeamPoints_Before",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = 0

    # Puntos: completar NaN con 0.
    for column in ("DriverPoints_Before", "TeamPoints_Before"):
        df[column] = df[column].fillna(0)

    # Posiciones de grilla, qualy y carrera.
    for column in ("Q_GridPosition", "Q_Position", "RACE_Position"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = _fix_grid(df, year, gp_slug)

    # Qualy: convertir Q1/Q2/Q3 a segundos y calcular el mejor tiempo.
    for col in ("Q1", "Q2", "Q3"):
        if col in df.columns:
            df[f"{col}_seconds"] = df[col].apply(_time_to_seconds)

    qualy_seconds_cols = [
        c for c in ("Q1_seconds", "Q2_seconds", "Q3_seconds")
        if c in df.columns
    ]
    if qualy_seconds_cols:
        df["Q_Best_seconds"] = df[qualy_seconds_cols].min(axis=1)

    df = df.drop(columns=[c for c in ("Q1", "Q2", "Q3") if c in df.columns])

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
    max_laps = max((int(r.get("laps") or 0) for r in results), default=0)
    rows = []
    for r in results:
        raw_status = r.get("status") or ""
        laps = int(r.get("laps") or 0)
        laps_down = max_laps - laps
        status = _normalize_status(raw_status, laps, laps_down, r.get("positionText"))
        finished = status in ("Finished", "Lapped") or (
            status == "Disqualified" and laps > 0
        )
        rows.append({
            "DriverNumber": r.get("number"),
            "DriverId": r.get("Driver", {}).get("driverId"),
            "TeamId": r.get("Constructor", {}).get("constructorId"),
            "RACE_Position": r.get("position"),
            "RACE_Finished": int(finished),
            "RACE_Status": status,
            "RACE_Laps_Down": laps_down,
            "RACE_Lapped": int(finished and laps_down > 0),
            "RACE_DNF_Type": None if finished else _dnf_type(raw_status, status),
            # Transitoria: se usa para calcular la tasa de carreras finalizadas
            # y se descarta en el consolidado.
            "RACE_ClassifiedPosition": r.get("positionText"),
            "Q_GridPosition": r.get("grid"),
            # Transitoria: la usa _fix_grid y se descarta ahi.
            "_laps": laps,
        })

    return pd.DataFrame(rows)


# Hasta 2024 la fuente marca la largada desde boxes con grid = 0. Desde 2025
# (Jolpica) le asigna un lugar de grilla y ya no se puede distinguir.
PITLANE_GRID_ZERO_LAST_YEAR = 2024


def _fix_grid(df: pd.DataFrame, year: int, gp_slug: str) -> pd.DataFrame:
    """Reemplaza grid = 0 y agrega Q_PitLaneStart.

    - Desde boxes (grid 0, vueltas > 0): ultimo lugar entre los que largaron
      (empatados si son varios) y Q_PitLaneStart = 1.
    - DNS (grid 0): su Q_Position.
    - Otro caso: vacio + warning.
    Q_PitLaneStart queda vacio desde 2025 (la fuente no lo informa).
    """
    df = df.copy()
    grid0 = df["Q_GridPosition"] == 0
    dns = df["RACE_Status"] == "DNS"
    pit = grid0 & (df["_laps"] > 0)
    dns0 = grid0 & dns
    other = grid0 & ~pit & ~dns0

    starters = int((~dns).sum())
    df.loc[pit, "Q_GridPosition"] = starters

    q_pos = df["Q_Position"] if "Q_Position" in df.columns else pd.Series(pd.NA, index=df.index)
    df.loc[dns0, "Q_GridPosition"] = q_pos[dns0]
    for _, r in df[dns0 & q_pos.isna()].iterrows():
        log.warning("Grid 0 DNS sin Q_Position, queda vacio: %s/%s %s", year, gp_slug, r["DriverId"])

    for _, r in df[other].iterrows():
        log.warning("Grid 0 sin caso conocido, queda vacio: %s/%s %s (status=%s, laps=%s)",
                    year, gp_slug, r["DriverId"], r["RACE_Status"], r["_laps"])
    df.loc[other, "Q_GridPosition"] = pd.NA
    # Mantener entero si no quedaron vacios (como venia de la fuente).
    if df["Q_GridPosition"].notna().all():
        df["Q_GridPosition"] = df["Q_GridPosition"].astype("int64")

    if year <= PITLANE_GRID_ZERO_LAST_YEAR:
        df["Q_PitLaneStart"] = pit.astype(int).where(~other)
    else:
        df["Q_PitLaneStart"] = pd.NA

    return df.drop(columns="_laps")


# Estados de Ergast/Jolpica. Hasta 2022 los doblados vienen como "+N Laps" y
# los abandonos con su causa; desde 2023 como "Lapped" y casi siempre
# "Retired" (causa desconocida).
DNS_STATUSES = {"Did not start", "Withdrew", "Did not qualify", "Did not prequalify"}
ACCIDENT_STATUSES = {
    "Accident", "Collision", "Collision damage", "Spun off", "Damage", "Debris",
}
# "Withdrew" con vueltas > 0 es un abandono (se retiro durante la carrera).
OTHER_DNF_STATUSES = {"Illness", "Out of fuel", "Puncture", "Tyre", "Injury", "Withdrew"}


def _normalize_status(raw: str, laps: int, laps_down: int, pos_text: str | None) -> str:
    """Estado normalizado: Finished / Lapped / Retired / Disqualified / DNS.

    "Finished" con vueltas de desventaja pasa a Lapped (laps_down manda).
    DNS = 0 vueltas y (positionText "W" o status de no largada): hasta 2022
    la fuente a veces pone la causa ("Brakes") en vez de "Did not start".
    """
    if raw == "Finished":
        return "Lapped" if laps_down > 0 else "Finished"
    if raw == "Lapped" or raw.startswith("+"):
        return "Lapped"
    if raw == "Disqualified":
        return "Disqualified"
    if laps == 0 and (pos_text == "W" or raw in DNS_STATUSES):
        return "DNS"
    return "Retired"


def _dnf_type(raw: str, status: str) -> str:
    """Tipo de no-finalizacion: accidente / mecanica / descalificacion /
    no_largo / otro / desconocido ("Retired" sin causa, desde 2023)."""
    if status == "Disqualified":
        return "descalificacion"
    if status == "DNS":
        return "no_largo"
    if raw in ("Retired", ""):
        return "desconocido"
    if raw in ACCIDENT_STATUSES:
        return "accidente"
    if raw in OTHER_DNF_STATUSES:
        return "otro"
    # El resto de las causas de Ergast son fallas del auto (Engine, Gearbox...).
    return "mecanica"


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
            "Q_Position": q.get("position"),
            "Q1": q.get("Q1"),
            "Q2": q.get("Q2"),
            "Q3": q.get("Q3"),
        })

    return pd.DataFrame(rows)


def _time_to_seconds(value) -> float | None:
    """Convierte un tiempo 'mm:ss.mmm' (o None/vacío) a segundos."""
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        if ":" in s:
            minutes, seconds = s.split(":")
            return float(minutes) * 60 + float(seconds)
        return float(s)
    except (ValueError, TypeError):
        return None


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
            except Exception:
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
    """Parsea datos de practicas FastF1 y devuelve features unificados.

    Recolecta las vueltas validas de las 3 sesiones (FP1/FP2/FP3) por piloto
    y genera una unica fila por piloto con columnas prefijadas 'FP_' (sin
    division por sesion).
    """
    base_dir = BRONZE_FASTF1 / str(year) / gp_slug
    if not base_dir.exists():
        return pd.DataFrame()

    # by_driver[abbr] = [(session_dir, lap_dict), ...]
    by_driver = defaultdict(list)
    driver_mapping: dict[str, str] = {}

    for session in FASTF1_SESSIONS:
        session_dir = base_dir / session
        laptimes_path = session_dir / "session_laptimes.json"
        if not session_dir.exists() or not laptimes_path.exists():
            continue

        driver_mapping.update(_load_driver_mapping(session_dir))

        with open(laptimes_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        laps = filter_outliers(valid_laps(data))
        for lap in laps:
            by_driver[lap["driver"]].append((session_dir, lap))

    features = []
    for driver_abbr, entries in by_driver.items():
        driver_num = driver_mapping.get(driver_abbr, driver_abbr)
        row = {"DriverNumber": driver_num}
        laps = [lap for _, lap in entries]

        # Tiempos por compuesto (unificados entre las 3 sesiones)
        compound_labels = {"MEDIUM": "MED", "INTERMEDIATE": "INTER"}
        unique_compounds = {l["compound"] for l in laps}
        for compound in unique_compounds:
            times = [l["time"] for l in laps if l["compound"] == compound]
            label = compound_labels.get(compound, compound)
            row[f"FP_LapTime_min_{label}"] = min(times)
            if compound in ("SOFT", "MEDIUM", "HARD"):
                row[f"FP_LapTime_mean_{label}"] = sum(times) / len(times)

        # Telemetria de la unica vuelta mas rapida del fin de semana
        best_session_dir, best_lap = min(entries, key=lambda e: e[1]["time"])
        tel_abs = _read_telemetry(best_session_dir, driver_abbr, best_lap["lap"])
        if tel_abs:
            row["FP_Throttle_fastlap"] = tel_abs["throttle"]
            row["FP_Speed_fastlap"] = tel_abs["speed"]
            row["FP_Speed_max"] = tel_abs["speed_max"]
            row["FP_RPM_fastlap"] = tel_abs["rpm"]
            row["FP_Brake_fastlap"] = tel_abs["brake"]

        # Telemetria media de todas las vueltas MEDIUM del fin de semana
        medium_entries = [e for e in entries if e[1]["compound"] == "MEDIUM"]
        if medium_entries:
            tel_values = {"throttle": [], "speed": [], "rpm": [], "brake": []}
            for session_dir, lap in medium_entries:
                tel = _read_telemetry(session_dir, driver_abbr, lap["lap"])
                if tel:
                    for k in tel_values:
                        if tel[k] is not None:
                            tel_values[k].append(tel[k])

            if tel_values["throttle"]:
                row["FP_Throttle_mean_MED"] = sum(tel_values["throttle"]) / len(tel_values["throttle"])
            if tel_values["speed"]:
                row["FP_Speed_mean_MED"] = sum(tel_values["speed"]) / len(tel_values["speed"])
            if tel_values["rpm"]:
                row["FP_RPM_mean_MED"] = sum(tel_values["rpm"]) / len(tel_values["rpm"])
            if tel_values["brake"]:
                row["FP_Brake_mean_MED"] = sum(tel_values["brake"]) / len(tel_values["brake"])

        features.append(row)

    return pd.DataFrame(features)


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


def _read_telemetry(session_dir: Path, driver: str, lap_num: int) -> dict | None:
    """Lee tel.json de bronce y calcula promedios."""
    path = session_dir / driver / f"{lap_num}_tel.json"
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

        # Filtra valores de acelerador fuera del rango fisico [0, 100].
        throttle_valid = [t for t in throttle if 0 <= t <= 100]

        return {
            "throttle": sum(throttle_valid) / len(throttle_valid) if throttle_valid else None,
            "speed": sum(speed) / len(speed) if speed else None,
            "speed_max": max(speed) if speed else None,
            "rpm": sum(rpm) / len(rpm) if rpm else None,
            # brake viene en [0, 1]; se escala a porcentaje.
            "brake": sum(brake) / len(brake) * 100 if brake else None,
        }
    except Exception:
        return None


# ------------------------------------------------------------------- build gp


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reordena columnas en orden logico.

    Identificacion -> Standings -> FP ABS -> FP DBT -> Q ABS -> Q DBT -> RACE.
    """
    id_cols = ["DriverId", "DriverNumber", "TeamId", "Year", "RoundNumber"]
    standings_cols = [
        "DriverPosition_Before", "DriverPoints_Before",
        "DriverPoints_Before_DBT_%", "TeamPosition_Before", "TeamPoints_Before",
    ]
    q_pos_cols = ["Q_Position", "Q_GridPosition"]
    fp_abs_cols = [c for c in df.columns if c.startswith("FP_") and c.endswith("_ABS")]
    fp_dbt_cols = [c for c in df.columns if c.startswith("FP_") and c.endswith("_DBT_%")]
    q_abs_cols = [c for c in df.columns if c.startswith("Q") and c.endswith("_ABS")]
    q_dbt_cols = [c for c in df.columns if c.startswith("Q") and c.endswith("_DBT_%")]
    race_cols = [c for c in df.columns if c.startswith("RACE")]

    ordered = (
        [c for c in id_cols if c in df.columns]
        + [c for c in standings_cols if c in df.columns]
        + fp_abs_cols
        + fp_dbt_cols
        + [c for c in q_pos_cols if c in df.columns]
        + q_abs_cols
        + q_dbt_cols
        + race_cols
    )
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


HIGHER_IS_BETTER = ("FP_Speed_", "FP_RPM_", "FP_Throttle_")


def _add_abs_dbt(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega los sufijos _ABS (valor absoluto) y _DBT_% (distancia porcentual
    al mejor de la carrera, 0% = mejor) a cada columna numerica por-carrera.

    El mejor de la carrera es el minimo cuando menor es mejor (tiempos de
    vuelta, frenada y qualy) y el maximo cuando mayor es mejor (velocidad,
    RPM y acelerador). En ambos casos DBT_% = |valor - mejor| / mejor * 100.
    Las columnas FP_* y de qualy se renombran a _ABS y agregan su _DBT_%;
    DriverPoints_Before mantiene su nombre y solo agrega su _DBT_% (medido
    respecto al maximo, el lider del campeonato).
    """
    rename_cols = [c for c in df.columns if c.startswith("FP_")]
    rename_cols += [
        c for c in ("Q1_seconds", "Q2_seconds", "Q3_seconds", "Q_Best_seconds")
        if c in df.columns
    ]

    rename = {}
    for col in rename_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if col.startswith(HIGHER_IS_BETTER):
            best = df[col].max()
            df[f"{col}_DBT_%"] = (best - df[col]) / best * 100 if best > 0 else pd.NA
        else:
            best = df[col].min()
            df[f"{col}_DBT_%"] = (df[col] - best) / best * 100 if best > 0 else pd.NA
        rename[col] = f"{col}_ABS"

    if rename:
        df = df.rename(columns=rename)

    if "DriverPoints_Before" in df.columns:
        best = df["DriverPoints_Before"].max()
        if best > 0:
            df["DriverPoints_Before_DBT_%"] = (
                (best - df["DriverPoints_Before"]) / best * 100
            )
        else:
            df["DriverPoints_Before_DBT_%"] = pd.NA

    return df


def build_gp_silver(year: int, gp_slug: str) -> dict[str, str]:
    """
    Construye el DataFrame de plata para un GP.

    Returns:
        Dict con rutas de los archivos generados.
    """
    log.info("Construyendo Plata para %s/%s", year, gp_slug)

    # 1. Datos de Archive (resultados, quali y standings)
    df_archive = parse_archive_gp(year, gp_slug)
    if df_archive.empty:
        log.warning("No hay datos de Archive para %s/%s", year, gp_slug)
        return {}

    # 2. Datos de FastF1 (telemetria de practicas)
    df_fastf1 = parse_fastf1_gp(year, gp_slug)
    if not df_fastf1.empty:
        df_full = df_archive.merge(df_fastf1, on="DriverNumber", how="left")
    else:
        # Si no hay FastF1 (pre-2018 o sin datos), usar archive como full
        # para mantener consistencia de filas en el consolidado
        df_full = df_archive.copy()

    df_full = _add_abs_dbt(df_full)
    df_full = _reorder_columns(df_full)
    full_path = _save_gp_silver(year, gp_slug, df_full, suffix="full")

    return {"full": full_path}


def _save_gp_silver(year: int, gp_slug: str, df: pd.DataFrame, suffix: str) -> str:
    """Guarda el DataFrame de un GP en silver/_parciales/ (temporal)."""
    dest_dir = PARTIALS_DIR / str(year)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{gp_slug}_{suffix}.parquet"
    df.to_parquet(dest, index=False)
    return str(dest)


# ------------------------------------------------------------------- consolidate


def _upsert_table(existing_path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    """Combina new_df con el consolidado Parquet existente.

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

    # Clave (Year, RoundNumber): conserva del historico las filas cuyo GP
    # no esta en new_df, y las reemplaza por las nuevas cuando coincide.
    new_keys = pd.MultiIndex.from_frame(
        new_df[["Year", "RoundNumber"]].astype(str)
    )
    existing_keys = pd.MultiIndex.from_frame(
        existing[["Year", "RoundNumber"]].astype(str)
    )
    mask = ~existing_keys.isin(new_keys)

    kept = existing[mask]
    return pd.concat([kept, new_df], ignore_index=True)


def _add_pre_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features acumuladas por temporada (pre-carrera, sin la actual).

    - Races_Before: carreras previas del piloto en la temporada.
    - FinishRate_Before: finalizadas previas / Races_Before.
    - DriverPoints_Rate_Before: DriverPoints_Before / (Races_Before * 25).
    - Avg_Position_Gain_Before: media de (Q_GridPosition - RACE_Position) en
      las carreras previas de la temporada.
    Luego descarta la columna transitoria RACE_ClassifiedPosition.
    """
    if "RACE_ClassifiedPosition" not in df.columns:
        return df

    df = df.copy()
    df["_finished"] = pd.to_numeric(df["RACE_ClassifiedPosition"], errors="coerce").notna()

    df = df.sort_values(["DriverId", "Year", "RoundNumber"]).reset_index(drop=True)

    grp = df.groupby(["DriverId", "Year"], sort=False)
    df["Races_Before"] = grp.cumcount()
    finished_cum = grp["_finished"].cumsum()
    df["FinishRate_Before"] = (finished_cum - df["_finished"]) / df["Races_Before"]

    # Tasa de dominancia: puntos del campeonato normalizados por el máximo
    # teórico (carreras previas * 25 pts por victoria). NaN sin carreras previas.
    denom = (df["Races_Before"] * 25).where(df["Races_Before"] != 0)
    df["DriverPoints_Rate_Before"] = df["DriverPoints_Before"] / denom

    # Ganancia media de posiciones (grid - carrera) en carreras previas.
    gain = df["Q_GridPosition"] - df["RACE_Position"]
    gain_cum = grp["Q_GridPosition"].cumsum() - grp["RACE_Position"].cumsum()
    df["Avg_Position_Gain_Before"] = (gain_cum - gain) / df["Races_Before"]

    return df.drop(columns=["_finished", "RACE_ClassifiedPosition"])


def consolidate_all(cleanup: bool = True) -> dict[str, str]:
    """
    Une TODOS los GPs de TODOS los anos en un Parquet consolidado.
    Lee de silver/_parciales/ (parquet, formato interno temporal), escribe
    en output/ como Parquet, y limpia parciales.

    Hace upsert sobre el consolidado existente para no perder anos que no
    estan en los parciales actuales (p.ej. al correr un rango parcial).

    Returns:
        Dict con rutas de consolidados generados.
    """
    result_paths = {}

    # Consolidar _full (unico output)
    all_full_files = sorted(PARTIALS_DIR.glob("*/*_full.parquet"))
    if all_full_files:
        new_df = pd.concat(
            [pd.read_parquet(f) for f in all_full_files], ignore_index=True
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = OUTPUT_DIR / "f1_all_full.parquet"
        df = _upsert_table(dest, new_df)
        df = _add_pre_race_features(df)
        df.to_parquet(dest, index=False)
        log.info("Consolidado all full: %s GPs nuevos, %s filas x %s columnas -> %s",
                 len(all_full_files), len(df), len(df.columns), dest)
        result_paths["full"] = str(dest)
        result_paths["postEDA"] = str(
            build_post_eda(dest, OUTPUT_DIR / "f1_all_postEDA.parquet")
        )
    else:
        log.info("No hay archivos _full para consolidar")

    # Limpiar parciales
    if cleanup and PARTIALS_DIR.exists():
        shutil.rmtree(PARTIALS_DIR)
        log.info("Parciales limpiados: %s", PARTIALS_DIR)

    return result_paths

