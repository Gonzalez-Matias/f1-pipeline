"""
Utilidades compartidas para el filtrado de vueltas FastF1.

Reune la logica duplicada entre la descarga (que decide que tel.json bajar)
y el procesamiento en Plata (que calcula features), ademas de utilidades
genericas como slugify.
"""
from __future__ import annotations

from collections import defaultdict

from f1.config import FASTF1_COMPOUNDS


def slugify(name: str) -> str:
    """Normaliza el nombre de una carrera a un slug valido para rutas/URLs."""
    return name.lower().strip().replace(" ", "-")


def valid_laps(data: dict) -> list[dict]:
    """Aplica el filtro base a los datos de session_laptimes.json.

    Devuelve una lista de dicts {driver, lap, compound, time} con las
    vueltas que pasan los criterios de validez: compuesto soportado, sin
    pit, estado ok, aceleracion consistente, vfl >= 100, con tiempo y no
    borradas.
    """
    if not isinstance(data, dict):
        return []

    n = len(data.get("lap", []))
    if n == 0:
        return []

    def val(key, i):
        v = data.get(key, [])
        return v[i] if i < len(v) else None

    laps = []
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

        laps.append({
            "driver": val("drv", i),
            "lap": int(val("lap", i)) if val("lap", i) is not None else None,
            "compound": compound,
            "time": float(lap_time),
        })

    return laps


def filter_outliers(laps: list[dict]) -> list[dict]:
    """Conserva las vueltas dentro del 105% del mejor tiempo de cada
    piloto + compuesto (descarta vueltas anormalmente lentas)."""
    grouped = defaultdict(list)
    for lap in laps:
        if lap["time"] is not None:
            grouped[(lap["driver"], lap["compound"])].append(lap)

    filtered = []
    for group_laps in grouped.values():
        min_time = min(l["time"] for l in group_laps)
        threshold = min_time * 1.05
        for l in group_laps:
            if l["time"] <= threshold:
                filtered.append(l)

    return filtered
