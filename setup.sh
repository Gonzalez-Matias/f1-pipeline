#!/usr/bin/env bash
# Levanta el pipeline F1: genera .env con el UID del host y arranca la pila.
set -euo pipefail

cd "$(dirname "$0")"

echo -e "AIRFLOW_UID=$(id -u)" > .env

docker-compose up -d --build
