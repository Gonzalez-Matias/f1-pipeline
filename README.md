# F1 Data Pipeline

Pipeline de datos para Formula 1 usando Apache Airflow. Descarga datos históricos de carreras (2018-2026) desde repositorios públicos de GitHub (TracingInsights), los procesa en capas tipo medallón (bronze → silver) y genera un dataset consolidado en CSV.

## Requisitos

- Docker + Docker Compose
- ~2GB de espacio libre (para datos descargados)

## Levantar el proyecto

```bash
cd f1-pipeline
./setup.sh
```

`setup.sh` genera el `.env` con tu UID y levanta la pila (`docker-compose up -d --build`).

Alternativa manual (equivale a lo que hace `setup.sh`):

```bash
cd f1-pipeline
echo -e "AIRFLOW_UID=$(id -u)" > .env   # que Airflow corra con tu usuario (evita errores de permisos)
docker-compose up -d --build
```

> En Linux es necesario generar `.env` con `AIRFLOW_UID=$(id -u)` antes de
> levantar. Sin eso, `logs/` y `plugins/` se crean como `root` y los
> contenedores (que corren con un usuario no-root) no pueden escribir, lo que
> rompe el procesador de DAGs.

Esperar a que los servicios estén healthy (unos 30 segundos).

## Acceso

- **Airflow UI:** http://localhost:8080
- **Usuario:** `admin` / **Password:** `admin`

> Los DAGs se crean **pausados** (`DAGS_ARE_PAUSED_AT_CREATION=true`).
> Activá el toggle en la UI o disparalos manualmente con el botón play.

## DAGs disponibles

### `f1_download_year`
Descarga y procesa un rango de temporadas.

**Parámetros:**
- `year_start` (int): primera temporada (default: 2018)
- `year_end` (int): última temporada (default: 2026)
- `force` (bool): reprocesar todo, incluso si ya está completo (default: `false`)

**Outputs:**
- `include/output/f1_all_full.csv`

### `f1_download_gp`
Descarga y procesa un único Gran Premio.

**Parámetros:**
- `year` (int): temporada
- `round` (int): número de ronda del GP (según calendario)

## Estructura

```
f1-pipeline/
├── dags/                 # DAGs de Airflow
│   ├── f1_download_year.py
│   └── f1_download_gp.py
├── include/
│   ├── f1/              # Lógica del pipeline
│   │   ├── __init__.py
│   │   ├── config.py    # Configuración central
│   │   ├── download.py  # Descarga de datos crudos (bronze)
│   │   ├── laps.py      # Filtrado de vueltas FastF1 (compartido)
│   │   ├── silver.py    # Procesamiento y features (silver)
│   │   └── validate.py  # Validación de completitud
│   └── output/          # Datos generados (no se versiona)
│       ├── bronze/      # JSON crudo (ergast/ y fastf1/)
│       ├── silver/_parciales/  # Parquets por GP (temporales, se borran)
│       └── f1_all_full.csv     # output final
├── notebooks/
│   └── explorar_parquet_f1.ipynb
├── docker-compose.yml
├── Dockerfile
├── setup.sh
└── requirements.txt
```

## Notas

- Los datos de prácticas (FastF1) solo están disponibles desde **2018** en adelante; el pipeline cubre 2018-2026.
- GPs futuros (sin `results.json`) se saltan automáticamente.
- Se descargan todas las vueltas MEDIUM + la vuelta más rápida absoluta (cualquier compuesto: SOFT, MEDIUM, HARD, INTERMEDIATE, WET).
- En el dataset: `LapTime_min_*` existe para todos los compuestos; `LapTime_mean_*` para SOFT, MEDIUM y HARD (que tienen múltiples vueltas).
- Telemetría (`Throttle_mean_*`, `Speed_mean_*`, etc.) se calcula solo para MEDIUM.
- Las columnas `FP_*` (prácticas) se unifican entre FP1/FP2/FP3. El diccionario de columnas completo (convención `_ABS`/`_DBT_%`) está en `include/output/columnas.md`.

## Limpiar y regenerar todo

1. Borrar outputs:
   ```bash
   rm -rf include/output/*
   ```

2. Trigger del DAG con **`force=true`** desde la UI de Airflow.

   Con `force=true` se **re-descargan** todos los datos desde los repos (no solo se reprocesan).

---

*Proyecto académico — UTN*
