"""API del filtro: ingesta de planos y entrenamiento de los modelos de visión.

Ingesta
    POST /v1/planos                              sube un archivo y lo encola para el filtro
    GET  /v1/planos/{id}                         estado, páginas y métricas
    GET  /v1/planos/{id}/paginas/{n}/{salida}    PNG: color | etiquetas | hints
    GET  /v1/parametros                          parámetros activos del filtro

Entrenamiento
    POST /v1/entrenamientos                      lanza ajuste_parametros o dataset_dfine
    GET  /v1/entrenamientos/{id}                 estado, métricas y artefactos
    GET  /v1/modelos                             versiones registradas (filtro opcional ?tipo=)
    POST /v1/modelos/{id}/activar                activa una versión; los workers la toman solos

Operación
    GET  /health                                 el proceso responde
    GET  /ready                                  storage, colas y repositorio responden

Uso: uvicorn filtro_planos.service.api:create_app --factory --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict
from pathlib import PurePath

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from filtro_planos import __version__
from filtro_planos.infra import Job, build_queue
from filtro_planos.infra.queue import InlineQueue, Queue
from filtro_planos.ingest import UnsupportedFormat, sniff_format
from filtro_planos.service import trainer
from filtro_planos.service.worker import Deps, build_deps, current_params, handle

log = logging.getLogger("filtro_planos.api")
OUTPUTS = ("color", "etiquetas", "hints")


def _status(record: dict | None) -> str | None:
    return record["estado"] if record else None


class TrainingRequest(BaseModel):
    tipo: str = Field(description="ajuste_parametros | dataset_dfine | segmentador_trazos")
    parametros: dict = Field(default_factory=dict)
    activar: bool = Field(default=False, description="activar la versión resultante al terminar")


def create_app(deps: Deps | None = None, queue: Queue | None = None, training_queue: Queue | None = None) -> FastAPI:
    deps = deps or build_deps()
    queue = queue or build_queue(deps.settings)
    training_queue = training_queue or build_queue(deps.settings, deps.settings.training_stream)
    if isinstance(queue, InlineQueue) and queue.handler is None:
        queue.handler = lambda job: handle(job, deps)
    if isinstance(training_queue, InlineQueue) and training_queue.handler is None:
        training_queue.handler = lambda job: trainer.handle_training(job, deps)
    max_bytes = deps.settings.max_upload_mb * 1024 * 1024

    app = FastAPI(
        title="Filtro de planos",
        version=__version__,
        description="Ingesta y entrenamiento para modelos de visión sobre planos arquitectónicos.",
    )

    # Ingesta

    @app.post("/v1/planos", status_code=202, tags=["ingesta"])
    def upload_plan(response: Response, file: UploadFile = File(...)) -> dict:
        data = file.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"El archivo supera {deps.settings.max_upload_mb} MB")
        try:
            fmt = sniff_format(data)
        except UnsupportedFormat as exc:
            raise HTTPException(415, str(exc)) from exc

        sha256 = hashlib.sha256(data).hexdigest()
        existing = deps.repo.find_by_sha(sha256)
        if existing:
            response.status_code = 200
            return {"id": existing, "estado": _status(deps.repo.get_plan(existing)), "duplicado": True}

        name = PurePath(file.filename or f"plano.{fmt}").name
        raw_key = deps.storage.put(f"originales/{sha256}/{name}", data, file.content_type or "application/octet-stream")
        plan_id = deps.repo.create_plan(sha256, name, fmt, len(data), raw_key)
        try:
            queue.enqueue(Job("procesar_plano", {"plan_id": plan_id, "raw_key": raw_key}))
        except Exception:  # en línea, el error ya quedó en el estado del plano
            log.exception("no se pudo procesar el plano %s", plan_id)
        return {"id": plan_id, "estado": _status(deps.repo.get_plan(plan_id)), "duplicado": False}

    @app.get("/v1/planos/{plan_id}", tags=["ingesta"])
    def get_plan(plan_id: str) -> dict:
        plan = deps.repo.get_plan(plan_id)
        if plan is None:
            raise HTTPException(404, "Plano no encontrado")
        return plan

    @app.get("/v1/planos/{plan_id}/paginas/{number}/{output}", tags=["ingesta"])
    def get_output(plan_id: str, number: int, output: str) -> Response:
        if output not in OUTPUTS:
            raise HTTPException(404, f"Salida desconocida; opciones: {', '.join(OUTPUTS)}")
        plan = deps.repo.get_plan(plan_id)
        page = next((p for p in (plan or {}).get("paginas", []) if p["numero"] == number), None)
        if page is None:
            raise HTTPException(404, "Página no encontrada")
        return Response(deps.storage.get(page["salidas"][output]), media_type="image/png")

    @app.get("/v1/parametros", tags=["ingesta"])
    def get_params() -> dict:
        params = current_params(deps)  # primero: puede cambiar la versión activa
        return {"version": deps.params_version or "config", "parametros": asdict(params)}

    # Entrenamiento

    @app.post("/v1/entrenamientos", status_code=202, tags=["entrenamiento"])
    def start_training(request: TrainingRequest) -> dict:
        if request.tipo in trainer.PENDING:
            raise HTTPException(501, trainer.PENDING[request.tipo])
        if request.tipo not in trainer.TRAINERS:
            options = ", ".join([*trainer.TRAINERS, *trainer.PENDING])
            raise HTTPException(422, f"tipo desconocido; opciones: {options}")
        training_id = deps.repo.create_training(request.tipo, request.parametros)
        payload = {"training_id": training_id, "parametros": request.parametros, "activar": request.activar}
        job = Job(request.tipo, payload)
        try:
            training_queue.enqueue(job)
        except Exception:  # en línea, el error ya quedó en el estado del entrenamiento
            log.exception("entrenamiento %s falló", training_id)
        return {"id": training_id, "tipo": request.tipo, "estado": _status(deps.repo.get_training(training_id))}

    @app.get("/v1/entrenamientos/{training_id}", tags=["entrenamiento"])
    def get_training(training_id: str) -> dict:
        training = deps.repo.get_training(training_id)
        if training is None:
            raise HTTPException(404, "Entrenamiento no encontrado")
        return training

    @app.get("/v1/modelos", tags=["entrenamiento"])
    def list_models(tipo: str | None = None) -> list[dict]:
        return deps.repo.list_models(tipo)

    @app.post("/v1/modelos/{model_id}/activar", tags=["entrenamiento"])
    def activate_model(model_id: str) -> dict:
        model = deps.repo.activate_model(model_id)
        if model is None:
            raise HTTPException(404, "Modelo no encontrado")
        return model

    # Operación

    @app.get("/health", tags=["operacion"])
    def health() -> dict:
        return {"estado": "ok", "version": __version__}

    @app.get("/ready", tags=["operacion"])
    def ready(response: Response) -> dict:
        checks = {}
        components = (
            ("storage", deps.storage),
            ("cola", queue),
            ("cola_entrenamiento", training_queue),
            ("repositorio", deps.repo),
        )
        for name, component in components:
            try:
                checks[name] = bool(component.ping())
            except Exception as exc:
                checks[name] = False
                log.warning("%s no responde: %r", name, exc)
        if not all(checks.values()):
            response.status_code = 503
        return checks

    return app
