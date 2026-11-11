"""
Sanciones de carrera (no de grilla) a partir de los mensajes de direccion de
carrera (Race/rcm.json) y las vueltas de carrera (Race/session_laptimes.json)
de TracingInsights.

  RACE_Penalty_Seconds: segundos de sancion del piloto en la carrera.
    - N SECOND TIME PENALTY: N.
    - DRIVE THROUGH: parada del GP - 6 s.
    - N SECOND STOP/GO: parada del GP + N - 2,5 s.
    "Parada del GP" = mediana de lo que se pierde en una parada normal con
    bandera verde: (vuelta de entrada + vuelta de salida) - 2 x vuelta normal.
    Las constantes salen de medir 12 drive-through y 11 stop & go reales de
    2018-2026.
  RACE_Penalty_Positions_Lost: cuantos pilotos que terminaron delante (con las
    mismas vueltas) quedarian detras si al piloto se le restan sus segundos de
    sancion. Cuenta tanto las sanciones cumplidas en pista (si el piloto entra
    a boxes despues del mensaje) como las sumadas al tiempo final. Solo para Finished/Lapped; NaN si no termino.

Ambas quedan NaN si falta rcm.json (o las vueltas, para los puestos). No se
cuentan sanciones aplicadas despues de la carrera que no figuran en los
mensajes.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TIME_PEN = re.compile(r"(\d+) SECOND TIME PENALTY FOR CAR (\d+)")
DRIVE_THROUGH = re.compile(r"DRIVE[ -]THROUGH PENALTY FOR CAR (\d+)")
STOP_GO = re.compile(r"(?:(\d+) SECOND )?STOP(?:/GO|-AND-GO| AND GO) PENALTY FOR CAR (\d+)")

DEFAULT_PIT_LOSS = 22.0  # mediana 2018-2026, si el GP no tiene paradas medibles
DRIVE_THROUGH_OFFSET = -6.0
STOP_GO_OFFSET = -2.5
STOP_GO_DEFAULT = 10


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        log.warning("JSON invalido: %s", path)
        return None


def _laps_frame(data: dict) -> pd.DataFrame:
    df = pd.DataFrame({k: data[k] for k in ("lap", "time", "sesT", "life", "dNum", "pin", "pout", "status")})
    for col in ("lap", "time", "sesT", "life"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dNum"] = df["dNum"].astype(str)
    df["pin"] = df["pin"].astype(str) != "None"
    df["pout"] = df["pout"].astype(str) != "None"
    df["green"] = df["status"].astype(str) == "1"
    return df.sort_values(["dNum", "lap"]).reset_index(drop=True)


def _pit_loss(laps: pd.DataFrame) -> float:
    """Mediana de lo que cuesta una parada normal (con cambio de gomas)."""
    losses = []
    for _, g in laps.groupby("dNum"):
        g = g.reset_index(drop=True)
        normal = g[~g["pin"] & ~g["pout"] & g["green"] & g["time"].notna()]
        for i in g.index[g["pin"]]:
            if i + 1 >= len(g) or not g.loc[i + 1, "pout"] or g.loc[i, "lap"] <= 1:
                continue
            if not (g.loc[i, "green"] and g.loc[i + 1, "green"]):
                continue
            # Misma goma (la edad sigue sumando): drive-through o stop & go.
            if g.loc[i + 1, "life"] == g.loc[i, "life"] + 1:
                continue
            ref = normal.loc[(normal["lap"] - g.loc[i, "lap"]).abs() <= 6, "time"]
            if len(ref) < 4:
                continue
            loss = g.loc[i, "time"] + g.loc[i + 1, "time"] - 2 * ref.median()
            if 10 <= loss <= 45:
                losses.append(loss)
    return float(np.median(losses)) if len(losses) >= 3 else DEFAULT_PIT_LOSS


def _penalties(rcm: dict, pit_loss: float) -> list[tuple[str, int, float]]:
    """(auto, vuelta del mensaje, segundos) por cada sancion."""
    out = []
    for msg, lap in zip(rcm["msg"], rcm["lap"]):
        msg = str(msg)
        if "PENALTY SERVED" in msg:
            continue
        lap = pd.to_numeric(lap, errors="coerce")
        lap = 0 if pd.isna(lap) else int(lap)
        if m := TIME_PEN.search(msg):
            out.append((m.group(2), lap, float(m.group(1))))
        elif m := DRIVE_THROUGH.search(msg):
            out.append((m.group(1), lap, pit_loss + DRIVE_THROUGH_OFFSET))
        elif m := STOP_GO.search(msg):
            secs = int(m.group(1) or STOP_GO_DEFAULT)
            out.append((m.group(2), lap, pit_loss + secs + STOP_GO_OFFSET))
    return out


def parse_penalties_gp(year: int, gp_slug: str, fastf1_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Agrega RACE_Penalty_Seconds y RACE_Penalty_Positions_Lost a df."""
    df = df.copy()
    race_dir = fastf1_dir / str(year) / gp_slug / "Race"
    rcm = _load(race_dir / "rcm.json")
    if rcm is None:
        log.warning("Sin rcm de carrera, sanciones vacias: %s/%s", year, gp_slug)
        df["RACE_Penalty_Seconds"] = np.nan
        df["RACE_Penalty_Positions_Lost"] = np.nan
        return df

    laptimes = _load(race_dir / "session_laptimes.json")
    laps = _laps_frame(laptimes) if laptimes else None
    pit_loss = _pit_loss(laps) if laps is not None else DEFAULT_PIT_LOSS

    num = df["DriverNumber"].astype(str)
    total = pd.Series(0.0, index=df.index)
    served = pd.Series(0.0, index=df.index)
    for car, lap, secs in _penalties(rcm, pit_loss):
        mask = num == car
        if not mask.any():
            log.warning("Sancion para auto %s sin piloto: %s/%s", car, year, gp_slug)
            continue
        total[mask] += secs
        # Se cumple en pista si el piloto entra a boxes despues del mensaje;
        # si no, se suma al tiempo final (tambien drive-through y stop & go
        # dados en las ultimas vueltas).
        if laps is not None and (laps["dNum"].eq(car) & laps["pin"] & (laps["lap"] > lap)).any():
            served[mask] += secs
    df["RACE_Penalty_Seconds"] = total

    lost = pd.Series(np.nan, index=df.index)
    if laps is not None:
        line = laps.dropna(subset=["sesT"]).groupby("dNum")["sesT"].last()
        finished = df["RACE_Status"].isin(["Finished", "Lapped"])
        # Tiempo oficial aproximado: llegada + sanciones no cumplidas en pista.
        official = num.map(line) + (total - served)
        lost[finished] = 0.0
        for i in df.index[finished & (total > 0) & official.notna()]:
            ahead = (
                finished & official.notna()
                & (df["RACE_Laps_Down"] == df.at[i, "RACE_Laps_Down"])
                & (df["RACE_Position"] < df.at[i, "RACE_Position"])
            )
            lost[i] = float((official[ahead] > official[i] - total[i]).sum())
        lost[finished & official.isna() & (total > 0)] = np.nan
    df["RACE_Penalty_Positions_Lost"] = lost
    return df
