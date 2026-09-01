"""
### Pipeline F1 — Descarga por rango de años

Descarga todos los GPs de un rango de temporadas, validando primero
si ya estan completos en el consolidado historico.

**Parametros:**
- `year_start` (int): primera temporada (default 2000)
- `year_end`   (int): ultima temporada  (default 2026)
- `mode`       (str): "results_only" | "full" (default "full")
- `force`      (bool): re-descargar aunque este completo

**Salida:**
- `silver/f1_all_results.parquet`
- `silver/f1_all_full.parquet` (si mode="full")
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
    description="Descarga rango de anos F1",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_tasks=8,
    tags=["f1", "bronze", "silver"],
    params={
        "year_start": Param(2000, type="integer", description="Primer ano (inclusive)"),
        "year_end":   Param(2026, type="integer", description="Ultimo ano (inclusive)"),
        "mode":       Param("full", type="string", enum=["results_only", "full"]),
        "force":      Param(False, type="boolean", description="Re-descargar aunque este completo"),
    },
)
def f1_download_year():

    @task
    def discover_range(**context) -> list[dict]:
        """Devuelve lista plana de todos los GPs a descargar en el rango."""
        year_start = context["params"]["year_start"]
        year_end   = context["params"]["year_end"]
        mode       = context["params"]["mode"]
        force      = context["params"]["force"]

        all_gps = []
        for year in range(year_start, year_end + 1):
            check = should_download(year, mode, force)
            if not check["should_download"]:
                continue

            schedule = get_schedule(year)
            if check["missing_gps"]:
                missing_rounds = {gp["round"] for gp in check["missing_gps"]}
                to_download = [gp for gp in schedule if gp["round"] in missing_rounds]
            else:
                to_download = schedule

            for gp in to_download:
                all_gps.append({
                    "year": year,
                    "round": gp["round"],
                    "race_name": gp["race_name"],
                    "mode": mode,
                })

        log.info("Rango %s-%s: %s GPs a descargar", year_start, year_end, len(all_gps))
        return all_gps

    @task(retries=3, retry_delay=timedelta(seconds=30))
    def download(gp_info: dict, **context) -> dict:
        year = gp_info["year"]
        round_num = gp_info["round"]
        race_name = gp_info["race_name"]
        mode = gp_info["mode"]
        force = context["params"]["force"]

        log.info("[Download] %s/%s %s (force=%s, mode=%s)", year, round_num, race_name, force, mode)
        result = process_single_gp(year, round_num, race_name, force=force, mode=mode)
        return {
            "year": year,
            "round": round_num,
            "slug": result["slug"],
            "mode": mode,
        }

    @task
    def build(gp_info: dict, **context) -> dict:
        year = gp_info["year"]
        round_num = gp_info["round"]
        slug = gp_info["slug"]
        mode = gp_info["mode"]

        log.info("[Silver] %s/%s", year, slug)
        paths = build_gp_silver(year, round_num, slug, mode=mode)
        return {
            "year": year,
            "round": round_num,
            "slug": slug,
            "mode": mode,
            "paths": paths,
        }

    @task(trigger_rule="none_failed_min_one_success")
    def consolidate(gp_results: list[dict], **context) -> dict:
        mode = context["params"]["mode"]
        log.info("Consolidando historico (mode=%s)", mode)
        paths = consolidate_all(mode=mode, cleanup=True)
        return {"mode": mode, "paths": paths}

    @task(trigger_rule="none_failed")
    def skip_notice(gps: list[dict], **context) -> dict:
        """No-op cuando no hay nada que descargar."""
        log.info("Nada que descargar en el rango.")
        return {}

    # Flujo
    gps = discover_range()

    # Branch: si hay GPs, descarga; si no, skip
    downloaded = download.expand(gp_info=gps)
    built = build.expand(gp_info=downloaded)
    consolidated = consolidate(built)
    skipped = skip_notice(gps)

    gps >> [downloaded, skipped]
    downloaded >> built >> consolidated


f1_download_year()
