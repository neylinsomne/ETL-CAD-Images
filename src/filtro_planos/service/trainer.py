"""Trainer: consume la cola de entrenamiento (aparte de la de ingesta, para no frenarla).

    ajuste_parametros  partición 30/70 e iteraciones -> nueva versión de parámetros del filtro
    dataset_dfine      planos anotados coloreados + COCO -> versión de dataset para afinar D-FINE
    segmentador_trazos flujo B, pendiente de elegir modelo base (docs/INVESTIGACION.md)

Cada resultado queda versionado en el registro de modelos; se activa con
POST /v1/modelos/{id}/activar (o "activar": true al lanzar el entrenamiento).

Uso: python -m filtro_planos.service.trainer   (con QUEUE_BACKEND=redis)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime

from filtro_planos.infra import Job, build_queue
from filtro_planos.infra.queue import RedisStreamQueue
from filtro_planos.service.worker import PARAMS_KIND, Deps, build_deps, current_params

log = logging.getLogger("filtro_planos.trainer")
DATASET_KIND = "dataset_dfine"
Trainer = Callable[[Job, Deps], tuple[dict, dict]]  # devuelve (métricas, artefactos)


def _version() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def tune_params(job: Job, deps: Deps) -> tuple[dict, dict]:
    from filtro_planos.training.tuning import run_tuning

    p = job.payload["parametros"]
    result = run_tuning(
        current_params(deps),
        iterations=p.get("iteraciones", 24),
        per_style=p.get("planos_por_estilo", 80),
        seed=p.get("semilla", 0),
        measure_per_style=p.get("medicion_por_estilo"),
        progress=False,
    )
    version = _version()
    uri = deps.storage.put(
        f"modelos/{PARAMS_KIND}/{version}.json", json.dumps(asdict(result.best)).encode(), "application/json"
    )
    metrics = result.metrics()
    model_id = deps.repo.register_model(PARAMS_KIND, version, uri, metrics, active=job.payload.get("activar", False))
    return metrics, {"modelo_id": model_id, "version": version, "uri": uri}


def export_dataset(job: Job, deps: Deps) -> tuple[dict, dict]:
    from filtro_planos.training.dfine_dataset import export_dfine_dataset

    p = job.payload["parametros"]
    version = _version()
    stats = export_dfine_dataset(
        deps.storage,
        current_params(deps),
        version,
        per_style=p.get("planos_por_estilo"),
        seed=p.get("semilla", 0),
        progress=False,
    )
    model_id = deps.repo.register_model(DATASET_KIND, version, stats["prefijo"], stats)
    return stats, {"modelo_id": model_id, "version": version, "prefijo": stats["prefijo"]}


TRAINERS: dict[str, Trainer] = {"ajuste_parametros": tune_params, "dataset_dfine": export_dataset}
PENDING = {"segmentador_trazos": "Flujo B pendiente: falta decidir el modelo base (docs/INVESTIGACION.md)"}


def handle_training(job: Job, deps: Deps) -> None:
    training_id = job.payload["training_id"]
    deps.repo.update_training(training_id, "corriendo")
    try:
        metrics, artifacts = TRAINERS[job.kind](job, deps)
        deps.repo.update_training(training_id, "listo", metrics, artifacts)
    except Exception as exc:
        log.exception("entrenamiento %s falló", training_id)
        deps.repo.update_training(training_id, "error", error=repr(exc))
        raise


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    deps = build_deps()
    queue = build_queue(deps.settings, deps.settings.training_stream)
    if not isinstance(queue, RedisStreamQueue):
        raise SystemExit("El trainer necesita QUEUE_BACKEND=redis (con inline, la API entrena al recibir)")
    log.info("trainer escuchando %s", queue.stream)
    while True:
        queue.consume(lambda job: handle_training(job, deps), block_ms=30000)


if __name__ == "__main__":
    main()
