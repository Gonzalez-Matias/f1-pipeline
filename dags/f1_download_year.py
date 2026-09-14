"""
### Pipeline F1 — Descarga por rango de años

Descarga todos los GPs de un rango de temporadas, validando primero
si ya estan completos en el consolidado historico.

**Parametros:**
- `year_start` (int): primera temporada (default 2018)
- `year_end`   (int): ultima temporada  (default 2026)
- `force`      (bool): re-descargar aunque este completo

**Salida:**
- `silver/f1_all_full.csv`
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models.param import Param

from f1.download import get_schedule, process_single_gp
from f1.silver import build_gp_silver, consolidate_all
from f1.validate import should_download

log = logging.getLogger(__name__)


default_args = {
    "owner": "f1-pipeline",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


@dag(
    dag_id="f1_download_year",
    default_args=default_args,
    description="Descarga rango de años F1",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_tasks=8,
    tags=["f1", "bronze", "silver"],
    params={
        "year_start": Param(2018, type="integer", description="Primer año (inclusive)"),
        "year_end":   Param(2026, type="integer", description="Ultimo año (inclusive)"),
        "force":      Param(False, type="boolean", description="Re-descargar aunque este completo"),
    },
)
def f1_download_year():

    @task
    def discover_range(**context) -> list[dict]:
        """Devuelve lista plana de todos los GPs a descargar en el rango."""
        year_start = context["params"]["year_start"]
        year_end   = context["params"]["year_end"]
        force      = context["params"]["force"]

        all_gps = []
        for year in range(year_start, year_end + 1):
            check = should_download(year, force)
            if not check["should_download"]:
                continue

            schedule = get_schedule(year)
            if check["missing_rounds"]:
                missing_rounds = set(check["missing_rounds"])
                to_download = [gp for gp in schedule if gp["round"] in missing_rounds]
            else:
                to_download = schedule

            for gp in to_download:
                all_gps.append({
                    "year": year,
                    "round": gp["round"],
                    "race_name": gp["race_name"],
                })

        log.info("Rango %s-%s: %s GPs a descargar", year_start, year_end, len(all_gps))
        return all_gps

    @task(retries=3, retry_delay=timedelta(seconds=30))
    def download(gp_info: dict, **context) -> dict:
        year = gp_info["year"]
        round_num = gp_info["round"]
        race_name = gp_info["race_name"]
        force = context["params"]["force"]

        log.info("[Download] %s/%s %s (force=%s)", year, round_num, race_name, force)
        result = process_single_gp(year, round_num, race_name, force=force)
        return {
            "year": year,
            "slug": result["slug"],
        }

    @task
    def build(gp_info: dict, **context) -> dict:
        year = gp_info["year"]
        slug = gp_info["slug"]

        log.info("[Silver] %s/%s", year, slug)
        build_gp_silver(year, slug)
        return {"year": year, "slug": slug}

    @task(trigger_rule="none_failed_min_one_success")
    def consolidate(gp_results: list[dict], **context) -> dict:
        log.info("Consolidando historico")
        paths = consolidate_all(cleanup=True)
        return {"paths": paths}

    # Flujo
    gps = discover_range()

    downloaded = download.expand(gp_info=gps)
    built = build.expand(gp_info=downloaded)
    consolidated = consolidate(built)

    downloaded >> built >> consolidated


f1_download_year()
