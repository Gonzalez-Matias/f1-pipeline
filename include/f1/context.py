"""
Contexto de cada GP a partir de bronce FastF1 (clima, direccion de carrera y
curvas) y del circuito (Ergast).

Columnas por GP (iguales para todos los pilotos de la carrera):
  Pre-carrera (se conocen antes de la largada):
    CIRCUIT_Id, CIRCUIT_Street, CIRCUIT_Corners,
    FP_Rain, FP_TrackTemp_mean, Q_Rain, Q_TrackTemp_mean, Q_AirTemp_mean,
    RACE_Rain_Forecast_% (pronostico de lluvia publicado por direccion de
    carrera antes de la largada).
  Resultado de la carrera (no son features):
    RACE_Rain, RACE_SC_Count, RACE_VSC_Count, RACE_RedFlag.

Columna por piloto:
    RACE_DNF_Type_Inferred: accidente / mecanica, inferido de los mensajes de
    direccion de carrera para los abandonos (RACE_Status = Retired). Acierta
    ~73% contra la causa real de 2018-2022 (validado por temporada).

En el consolidado se agrega CIRCUIT_SC_Before (ver add_circuit_history).

NO esta conectado a silver/postEDA: las salidas no incluyen estas columnas
(en las pruebas no aportaron senal para predecir abandonos). Los JSON de
contexto se siguen descargando (download_gp_context) por si se retoma.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Circuitos callejeros o semipermanentes sobre calles publicas.
STREET_CIRCUITS = {
    "albert_park", "baku", "jeddah", "madring", "marina_bay", "miami",
    "monaco", "sochi", "vegas", "villeneuve",
}

PRACTICE_SESSIONS = ["Practice 1", "Practice 2", "Practice 3"]
RAIN_FORECAST = re.compile(r"RISK OF RAIN FOR (?:THE )?(?:F1 RACE|THIS SESSION) IS (\d+) ?%")
CARS = re.compile(r"CARS? ((?:\d+ \([A-Z]{3}\)(?:,| AND|, AND)? ?)+)")
# Mensajes que mencionan autos pero no tienen que ver con abandonos.
IGNORED_MSGS = ("BLUE FLAG", "TRACK LIMITS", "PIT LANE")


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        log.warning("JSON invalido: %s", path)
        return None


def _num(values) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _weather(data: dict | None) -> dict:
    """Lluvia (0/1) y medias de temperatura de pista y aire."""
    if not data or not data.get("wR"):
        return {"rain": np.nan, "track": np.nan, "air": np.nan}
    return {
        "rain": int(any(v is True or v == "True" for v in data["wR"])),
        "track": _num(data.get("wTT", [])).mean(),
        "air": _num(data.get("wAT", [])).mean(),
    }


def _laps(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rcm_counts(rcm: dict | None) -> dict:
    """Autos de seguridad, VSC, bandera roja y pronostico de lluvia."""
    out = {"sc": np.nan, "vsc": np.nan, "red": np.nan, "forecast": np.nan}
    if not rcm:
        return out
    sc = vsc = red = 0
    for msg, flag in zip(rcm["msg"], rcm["flag"]):
        msg = str(msg)
        if "VIRTUAL SAFETY CAR DEPLOYED" in msg or "VSC DEPLOYED" in msg:
            vsc += 1
        elif "SAFETY CAR DEPLOYED" in msg:
            sc += 1
        if flag == "RED":
            red = 1
        m = RAIN_FORECAST.search(msg)
        if m and np.isnan(out["forecast"]):
            out["forecast"] = float(m.group(1))
    out.update(sc=sc, vsc=vsc, red=red)
    return out


def _last_laps(laptimes: dict | None) -> dict[str, int]:
    """Ultima vuelta registrada por numero de auto en la carrera."""
    if not laptimes:
        return {}
    df = pd.DataFrame({"dNum": laptimes["dNum"], "lap": laptimes["lap"]})
    df["lap"] = pd.to_numeric(df["lap"], errors="coerce")
    return df.groupby(df["dNum"].astype(str))["lap"].max().dropna().astype(int).to_dict()


def _infer_dnf(rcm: dict, num: str, last: int) -> str:
    """accidente si hay un incidente con otros autos cerca del abandono, o si
    se neutralizo la carrera justo despues (salvo un "STOPPED" sin vuelta 1).
    Regla sacada de un arbol de decision validado en 2018-2022."""
    multi = stopped = neutral = False
    for msg, lap, flag in zip(rcm["msg"], rcm["lap"], rcm["flag"]):
        lap = _laps(lap)
        if lap is None:
            continue
        msg = str(msg)
        if last <= lap <= last + 1 and (
            ("SAFETY CAR DEPLOYED" in msg) or ("VSC DEPLOYED" in msg) or flag == "RED"
        ):
            neutral = True
        if not (last - 1 <= lap <= last + 2) or any(s in msg for s in IGNORED_MSGS):
            continue
        m = CARS.search(msg)
        cars = re.findall(r"(\d+) \(", m.group(1)) if m else []
        if num not in cars:
            continue
        if len(cars) >= 2 and "INCIDENT" in msg:
            multi = True
        if "STOPPED" in msg:
            stopped = True
    if multi or (neutral and (last <= 1 or not stopped)):
        return "accidente"
    return "mecanica"


def parse_context_gp(
    year: int, gp_slug: str, fastf1_dir: Path, ergast_dir: Path, df: pd.DataFrame
) -> pd.DataFrame:
    """Agrega a df (resultados del GP) las columnas de contexto."""
    df = df.copy()
    gp_dir = fastf1_dir / str(year) / gp_slug

    results = _load(ergast_dir / str(year) / gp_slug / "results.json") or {}
    races = results.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    circuit = races[0].get("Circuit", {}).get("circuitId") if races else None

    corners = None
    for session in ("Race", "Qualifying"):
        data = _load(gp_dir / session / "corners.json")
        if data and data.get("CornerNumber"):
            corners = len(data["CornerNumber"])
            break

    fp_rain, fp_track = [], []
    for session in PRACTICE_SESSIONS:
        # weather.json de la sesion; si falta, el clima por vuelta del laptimes.
        w = _weather(_load(gp_dir / session / "weather.json"))
        if np.isnan(w["rain"]):
            w = _weather(_load(gp_dir / session / "session_laptimes.json"))
        if not np.isnan(w["rain"]):
            fp_rain.append(w["rain"])
            fp_track.append(w["track"])
    q = _weather(_load(gp_dir / "Qualifying" / "weather.json"))
    race_w = _weather(_load(gp_dir / "Race" / "weather.json"))
    rcm = _load(gp_dir / "Race" / "rcm.json")
    counts = _rcm_counts(rcm)
    if rcm is None:
        log.warning("Sin rcm de carrera: %s/%s", year, gp_slug)

    df["CIRCUIT_Id"] = circuit
    df["CIRCUIT_Street"] = int(circuit in STREET_CIRCUITS) if circuit else np.nan
    df["CIRCUIT_Corners"] = corners if corners is not None else np.nan
    df["FP_Rain"] = max(fp_rain) if fp_rain else np.nan
    df["FP_TrackTemp_mean"] = np.nanmean(fp_track) if fp_track else np.nan
    df["Q_Rain"] = q["rain"]
    df["Q_TrackTemp_mean"] = q["track"]
    df["Q_AirTemp_mean"] = q["air"]
    df["RACE_Rain_Forecast_%"] = counts["forecast"]
    df["RACE_Rain"] = race_w["rain"]
    df["RACE_SC_Count"] = counts["sc"]
    df["RACE_VSC_Count"] = counts["vsc"]
    df["RACE_RedFlag"] = counts["red"]

    inferred = pd.Series(None, index=df.index, dtype=object)
    if rcm is not None:
        last_laps = _last_laps(_load(gp_dir / "Race" / "session_laptimes.json"))
        for i, r in df[df["RACE_Status"] == "Retired"].iterrows():
            num = str(r["DriverNumber"])
            if num in last_laps:
                inferred[i] = _infer_dnf(rcm, num, last_laps[num])
    df["RACE_DNF_Type_Inferred"] = inferred
    return df


def add_circuit_history(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega CIRCUIT_SC_Before (media de autos de seguridad en ediciones
    previas del circuito; NaN si no hay) y completa CIRCUIT_Corners faltantes
    con el valor del mismo circuito en otros anos. Usa solo columnas que se
    guardan en el consolidado (sirve tambien en corridas incrementales)."""
    if "CIRCUIT_Id" not in df.columns:
        return df
    gp = (
        df[["Year", "RoundNumber", "CIRCUIT_Id", "RACE_SC_Count"]]
        .drop_duplicates(["Year", "RoundNumber"])
        .sort_values(["Year", "RoundNumber"])
    )
    value = gp["RACE_SC_Count"].fillna(0)
    has = gp["RACE_SC_Count"].notna().astype(int)
    prev_sum = value.groupby(gp["CIRCUIT_Id"]).cumsum() - value
    prev_n = has.groupby(gp["CIRCUIT_Id"]).cumsum() - has
    gp["CIRCUIT_SC_Before"] = prev_sum / prev_n.where(prev_n > 0)
    df = df.drop(columns=["CIRCUIT_SC_Before"], errors="ignore").merge(
        gp[["Year", "RoundNumber", "CIRCUIT_SC_Before"]], on=["Year", "RoundNumber"], how="left"
    )
    df["CIRCUIT_Corners"] = df["CIRCUIT_Corners"].fillna(
        df.groupby("CIRCUIT_Id")["CIRCUIT_Corners"].transform("median")
    )
    return df
