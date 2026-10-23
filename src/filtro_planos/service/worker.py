"""Worker: consume trabajos de la cola y ejecuta el paso del filtro sobre cada página.

    procesar_plano: original en storage -> páginas (ingest) -> process_page -> salidas a storage
                    -> métricas al repositorio

Uso: python -m filtro_planos.service.worker   (con QUEUE_BACKEND=redis)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from filtro_planos.config import Settings
from filtro_planos.detector import Detector, build_detector
from filtro_planos.envelope import EnvelopeParams
from filtro_planos.infra import Job, build_queue, build_repository, build_storage
from filtro_planos.infra.queue import RedisStreamQueue
from filtro_planos.infra.repository import Repository
from filtro_planos.infra.storage import Storage
from filtro_planos.ingest import load_pages
from filtro_planos.pipeline import load_params, page_outputs, page_record, process_page

log = logging.getLogger("filtro_planos.worker")


PARAMS_KIND = "parametros_filtro"


@dataclass
class Deps:
    settings: Settings
    storage: Storage
    repo: Repository
    params: EnvelopeParams  # los del archivo de config; se reemplazan por la versión activa del registro
    detector: Detector
    params_version: str | None = None


def current_params(deps: Deps) -> EnvelopeParams:
    """Versión activa del registro de modelos si existe; si no, la del archivo de config.

    Activar una versión (POST /v1/modelos/{id}/activar) la aplica sin reiniciar workers.
    """
    active = deps.repo.active_model(PARAMS_KIND)
    if active is not None and active["version"] != deps.params_version:
        deps.params = EnvelopeParams.from_dict(json.loads(deps.storage.get(active["uri"])))
        deps.params_version = active["version"]
    return deps.params


def output_key(plan_id: str, page: int, kind: str) -> str:
    return f"procesado/{plan_id}/p{page}/{kind}.png"


def process_plan(job: Job, deps: Deps) -> None:
    plan_id = job.payload["plan_id"]
    deps.repo.set_status(plan_id, "procesando")
    try:
        params = current_params(deps)
        pages = load_pages(deps.storage.get(job.payload["raw_key"]), deps.settings.pdf_dpi, deps.settings.max_pixels)
        for page in pages:
            result = process_page(page.gray, params, deps.detector)
            keys = {
                kind: deps.storage.put(output_key(plan_id, page.number, kind), data, "image/png")
                for kind, data in page_outputs(result).items()
            }
            deps.repo.add_page(plan_id, page.number, result.status, keys, page_record(result))
        deps.repo.set_status(plan_id, "listo")
    except Exception as exc:
        log.exception("plano %s falló", plan_id)
        deps.repo.set_status(plan_id, "error", repr(exc))
        raise


HANDLERS = {"procesar_plano": process_plan}


def handle(job: Job, deps: Deps) -> None:
    handler = HANDLERS.get(job.kind)
    if handler is None:
        raise ValueError(f"Tipo de trabajo desconocido: {job.kind!r}")
    handler(job, deps)


def build_deps(settings: Settings | None = None) -> Deps:
    settings = settings or Settings.from_env()
    return Deps(
        settings=settings,
        storage=build_storage(settings),
        repo=build_repository(settings),
        params=load_params(settings.envelope_params),
        detector=build_detector(settings),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    deps = build_deps()
    queue = build_queue(deps.settings)
    if not isinstance(queue, RedisStreamQueue):
        raise SystemExit("El worker necesita QUEUE_BACKEND=redis (con inline, la API procesa al recibir)")
    log.info("worker escuchando %s", queue.stream)
    while True:
        queue.consume(lambda job: handle(job, deps))


if __name__ == "__main__":
    main()
