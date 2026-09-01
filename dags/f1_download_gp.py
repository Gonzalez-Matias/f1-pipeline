"""
### Pipeline F1 — Descarga de un solo GP

Descarga un Gran Premio específico y reconstruye los consolidados.

**Parametros:**
- `year` (int)
- `round` (int)
- `mode` (str): `"results_only"` | `"full"`
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models.param import Param

from f1.download import get_schedule, process_single_gp
from f1.silver import build_gp_silver, consolidate_all

log = logging.getLogger(__name__)


default_args = {
    "owner": "f1-pipeline",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


@dag(
    dag_id="f1_download_gp",
    default_args=default_args,
    description="Descarga un solo GP y reconstruye consolidados",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["f1", "bronze", "silver", "gp"],
    params={
        "year": Param(2024, type="integer"),
        "round": Param(1, type="integer"),
        "mode": Param("full", type="string", enum=["results_only", "full"]),
    },
)
def f1_download_gp():

    @task
    def download(**context) -> dict:
        year = context["params"]["year"]
        round_num = context["params"]["round"]
        mode = context["params"]["mode"]

        schedule = get_schedule(year)
        race_name = None
        for gp in schedule:
            if gp["round"] == round_num:
                race_name = gp["race_name"]
                break
        if not race_name:
            raise ValueError(f"GP {round_num} no encontrado en calendario {year}")

        log.info("[Download] %s/%s %s (mode=%s)", year, round_num, race_name, mode)
        result = process_single_gp(year, round_num, race_name, force=False, mode=mode)
        return {
            "year": year,
            "round": round_num,
            "slug": result["slug"],
            "mode": mode,
        }

    @task
    def build(bronze_result: dict, **context) -> dict:
        year = bronze_result["year"]
        round_num = bronze_result["round"]
        slug = bronze_result["slug"]
        mode = bronze_result["mode"]

        log.info("[Silver] %s/%s (mode=%s)", year, slug, mode)
        paths = build_gp_silver(year, round_num, slug, mode=mode)
        return {
            "year": year,
            "round": round_num,
            "slug": slug,
            "mode": mode,
            "paths": paths,
        }

    @task
    def consolidate(silver_result: dict, **context) -> dict:
        mode = silver_result["mode"]
        log.info("Reconstruyendo consolidados (mode=%s)", mode)
        paths = consolidate_all(mode=mode)
        return {"mode": mode, "paths": paths}

    bronze = download()
    silver = build(bronze)
    all_cons = consolidate(silver)

    bronze >> silver >> all_cons


f1_download_gp()
