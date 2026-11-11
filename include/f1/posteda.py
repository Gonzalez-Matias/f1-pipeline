"""
Salida extra post-EDA: f1_all_postEDA.parquet.

Se genera a partir de f1_all_full.parquet sin modificarlo:
  - Descarta ambas filas de las claves (Year, RoundNumber, DriverId) repetidas.
  - Pasa a nulo los ceros que no son datos (Q_GridPosition y posiciones de
    campeonato en la primera fecha).
  - Recalcula FinishRate_Before con RACE_Finished (si la columna existe).
  - Agrega FP_Desgaste_{SOFT,MED}.
  - Exporta solo las columnas finales del EDA (ver FINAL_COLUMNS).

RACE_Finished, RACE_Status/Laps_Down/Lapped/DNF_Type y RACE_Penalty_* se exportan para
filtrar o analizar (no son features: son resultado de la carrera) y
Year/RoundNumber para separar entrenamiento y prueba.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

KEYS = ["Year", "RoundNumber", "DriverId"]
ZERO_AS_NULL = [
    "Q_GridPosition", "DriverPosition_Before", "TeamPosition_Before",
]
IDENTIFIERS = ["DriverId", "TeamId", "Year", "RoundNumber"]
FEATURES = [
    "Q_GridPosition", "Q_Best_seconds_DBT_%", "DriverPoints_Before_DBT_%",
    "FP_LapTime_min_SOFT_DBT_%", "FP_LapTime_min_MED_DBT_%",
    "FP_LapTime_min_HARD_DBT_%", "FP_Desgaste_SOFT", "FP_Desgaste_MED",
    "FinishRate_Before",
]
TARGET = "RACE_Position"
FILTER = "RACE_Finished"
# Resultado de carrera (no son features): para filtrar o analizar.
RACE_OUTCOME = [
    "RACE_Status", "RACE_Laps_Down", "RACE_Lapped", "RACE_DNF_Type",
    "RACE_Penalty_Seconds", "RACE_Penalty_Positions_Lost",
]
# Informativa, no feature: vacia desde 2025 (la fuente no informa boxes).
INFO = ["Q_PitLaneStart"]
FINAL_COLUMNS = IDENTIFIERS + FEATURES + INFO + [FILTER] + RACE_OUTCOME + [TARGET]


def _add_finish_rate(df: pd.DataFrame) -> pd.DataFrame:
    """FinishRate_Before = finalizadas previas / Races_Before (usa RACE_Finished)."""
    if "RACE_Finished" not in df.columns:
        log.warning("postEDA: sin RACE_Finished, se conserva FinishRate_Before original")
        return df
    df = df.sort_values(["DriverId", "Year", "RoundNumber"]).reset_index(drop=True)
    finished = df["RACE_Finished"].astype(int)
    grp = finished.groupby([df["DriverId"], df["Year"]], sort=False)
    prev = grp.cumsum() - finished
    # Races_Before del consolidado cuenta doble las filas duplicadas; se
    # recalcula aca sobre filas unicas.
    races_before = grp.cumcount()
    df["FinishRate_Before"] = prev / races_before.where(races_before != 0)
    return df


def build_post_eda(src: Path, dest: Path) -> Path:
    df = pd.read_parquet(src)
    n0 = len(df)

    # FinishRate se calcula con una fila por clave para no contar doble.
    df = _add_finish_rate(df.drop_duplicates(KEYS, keep="first")).merge(
        df[df.duplicated(KEYS, keep=False)][KEYS].drop_duplicates().assign(_dup=True),
        on=KEYS, how="left",
    )
    df = df[df["_dup"].isna()].drop(columns="_dup")

    for col in ZERO_AS_NULL:
        if col in df.columns:
            df[col] = df[col].mask(df[col] == 0)

    for tyre in ("SOFT", "MED"):
        mean_c, min_c = f"FP_LapTime_mean_{tyre}_ABS", f"FP_LapTime_min_{tyre}_ABS"
        if mean_c in df.columns and min_c in df.columns:
            df[f"FP_Desgaste_{tyre}"] = (df[mean_c] - df[min_c]) / df[min_c] * 100

    df = df[[c for c in FINAL_COLUMNS if c in df.columns]]
    df = df.sort_values(KEYS).reset_index(drop=True)

    df.to_parquet(dest, index=False)
    log.info("postEDA: %s -> %s filas x %s columnas -> %s", n0, len(df), len(df.columns), dest)
    return dest
