"""Metadatos de planos y páginas: en memoria o en Postgres (esquema en sql/001_init.sql).

En producción DATABASE_URL apunta a PgBouncer. Con pool por transacción, PgBouncer no conserva
sentencias preparadas del servidor, así que se desactivan (prepare_threshold=None).
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from typing import Protocol

from filtro_planos.config import Settings


class Repository(Protocol):
    # Planos (ingesta)
    def create_plan(self, sha256: str, name: str, fmt: str, size: int, raw_key: str) -> str: ...
    def find_by_sha(self, sha256: str) -> str | None: ...
    def set_status(self, plan_id: str, status: str, error: str | None = None) -> None: ...
    def add_page(self, plan_id: str, number: int, status: str, outputs: dict, record: dict) -> None: ...
    def get_plan(self, plan_id: str) -> dict | None: ...
    # Entrenamientos y registro de modelos
    def create_training(self, kind: str, params: dict) -> str: ...
    def update_training(self, training_id: str, status: str, metrics: dict | None = None,
                        artifacts: dict | None = None, error: str | None = None) -> None: ...
    def get_training(self, training_id: str) -> dict | None: ...
    def register_model(self, kind: str, version: str, uri: str, metrics: dict, active: bool = False) -> str: ...
    def list_models(self, kind: str | None = None) -> list[dict]: ...
    def activate_model(self, model_id: str) -> dict | None: ...
    def active_model(self, kind: str) -> dict | None: ...
    def ping(self) -> bool: ...


def _now() -> str:
    return datetime.now(UTC).isoformat()


class MemoryRepository:
    def __init__(self):
        self._plans: dict[str, dict] = {}
        self._trainings: dict[str, dict] = {}
        self._models: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create_training(self, kind: str, params: dict) -> str:
        with self._lock:
            training_id = uuid.uuid4().hex
            self._trainings[training_id] = {
                "id": training_id,
                "tipo": kind,
                "parametros": params,
                "estado": "en_cola",
                "metricas": None,
                "artefactos": None,
                "error": None,
                "creado": _now(),
            }
            return training_id

    def update_training(self, training_id: str, status: str, metrics: dict | None = None,
                        artifacts: dict | None = None, error: str | None = None) -> None:
        with self._lock:
            self._trainings[training_id].update(
                estado=status, error=error, actualizado=_now(),
                **({"metricas": metrics} if metrics is not None else {}),
                **({"artefactos": artifacts} if artifacts is not None else {}),
            )

    def get_training(self, training_id: str) -> dict | None:
        training = self._trainings.get(training_id)
        return json.loads(json.dumps(training)) if training else None

    def register_model(self, kind: str, version: str, uri: str, metrics: dict, active: bool = False) -> str:
        with self._lock:
            model_id = uuid.uuid4().hex
            self._models[model_id] = {
                "id": model_id, "tipo": kind, "version": version, "uri": uri,
                "metricas": metrics, "activo": False, "creado": _now(),
            }
        if active:
            self.activate_model(model_id)
        return model_id

    def list_models(self, kind: str | None = None) -> list[dict]:
        models = [m for m in self._models.values() if kind is None or m["tipo"] == kind]
        return json.loads(json.dumps(sorted(models, key=lambda m: m["creado"], reverse=True)))

    def activate_model(self, model_id: str) -> dict | None:
        with self._lock:
            model = self._models.get(model_id)
            if model is None:
                return None
            for other in self._models.values():
                if other["tipo"] == model["tipo"]:
                    other["activo"] = other["id"] == model_id
            return json.loads(json.dumps(model))

    def active_model(self, kind: str) -> dict | None:
        active = (m for m in self._models.values() if m["tipo"] == kind and m["activo"])
        return next((json.loads(json.dumps(m)) for m in active), None)

    def create_plan(self, sha256: str, name: str, fmt: str, size: int, raw_key: str) -> str:
        with self._lock:
            plan_id = uuid.uuid4().hex
            self._plans[plan_id] = {
                "id": plan_id,
                "sha256": sha256,
                "nombre": name,
                "formato": fmt,
                "bytes": size,
                "clave_original": raw_key,
                "estado": "en_cola",
                "error": None,
                "creado": _now(),
                "paginas": [],
            }
            return plan_id

    def find_by_sha(self, sha256: str) -> str | None:
        return next((p["id"] for p in self._plans.values() if p["sha256"] == sha256), None)

    def set_status(self, plan_id: str, status: str, error: str | None = None) -> None:
        with self._lock:
            self._plans[plan_id].update(estado=status, error=error)

    def add_page(self, plan_id: str, number: int, status: str, outputs: dict, record: dict) -> None:
        with self._lock:
            pages = [p for p in self._plans[plan_id]["paginas"] if p["numero"] != number]
            pages.append({"numero": number, "estado": status, "salidas": outputs, "metricas": record})
            self._plans[plan_id]["paginas"] = sorted(pages, key=lambda p: p["numero"])

    def get_plan(self, plan_id: str) -> dict | None:
        plan = self._plans.get(plan_id)
        return json.loads(json.dumps(plan)) if plan else None  # copia

    def ping(self) -> bool:
        return True


class PostgresRepository:
    """No probado contra un Postgres real en este repositorio; el SQL sigue sql/001_init.sql."""

    def __init__(self, dsn: str):
        self.dsn = dsn

    def _connect(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self.dsn, autocommit=True, prepare_threshold=None, row_factory=dict_row)

    def create_plan(self, sha256: str, name: str, fmt: str, size: int, raw_key: str) -> str:
        plan_id = uuid.uuid4().hex
        with self._connect() as conn:
            row = conn.execute(
                """
                INSERT INTO planos (id, sha256, nombre, formato, bytes, clave_original, estado)
                VALUES (%s, %s, %s, %s, %s, %s, 'en_cola')
                ON CONFLICT (sha256) DO UPDATE SET sha256 = EXCLUDED.sha256
                RETURNING id
                """,
                (plan_id, sha256, name, fmt, size, raw_key),
            ).fetchone()
        return str(row["id"]).replace("-", "")

    def find_by_sha(self, sha256: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM planos WHERE sha256 = %s", (sha256,)).fetchone()
        return str(row["id"]).replace("-", "") if row else None

    def set_status(self, plan_id: str, status: str, error: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE planos SET estado = %s, error = %s, actualizado = now() WHERE id = %s",
                (status, error, plan_id),
            )

    def add_page(self, plan_id: str, number: int, status: str, outputs: dict, record: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO paginas (plano_id, numero, estado, salidas, metricas)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (plano_id, numero) DO UPDATE
                SET estado = EXCLUDED.estado, salidas = EXCLUDED.salidas, metricas = EXCLUDED.metricas
                """,
                (plan_id, number, status, json.dumps(outputs), json.dumps(record)),
            )

    def get_plan(self, plan_id: str) -> dict | None:
        with self._connect() as conn:
            plan = conn.execute("SELECT * FROM planos WHERE id = %s", (plan_id,)).fetchone()
            if plan is None:
                return None
            pages = conn.execute(
                "SELECT numero, estado, salidas, metricas FROM paginas WHERE plano_id = %s ORDER BY numero",
                (plan_id,),
            ).fetchall()
        plan = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in plan.items()}
        plan["id"] = str(plan["id"]).replace("-", "")
        plan["paginas"] = pages
        return plan

    def create_training(self, kind: str, params: dict) -> str:
        training_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO entrenamientos (id, tipo, parametros, estado) VALUES (%s, %s, %s, 'en_cola')",
                (training_id, kind, json.dumps(params)),
            )
        return training_id

    def update_training(self, training_id: str, status: str, metrics: dict | None = None,
                        artifacts: dict | None = None, error: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE entrenamientos
                SET estado = %s, error = %s, actualizado = now(),
                    metricas = COALESCE(%s, metricas), artefactos = COALESCE(%s, artefactos)
                WHERE id = %s
                """,
                (status, error, json.dumps(metrics) if metrics is not None else None,
                 json.dumps(artifacts) if artifacts is not None else None, training_id),
            )

    def get_training(self, training_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM entrenamientos WHERE id = %s", (training_id,)).fetchone()
        return _clean(row) if row else None

    def register_model(self, kind: str, version: str, uri: str, metrics: dict, active: bool = False) -> str:
        model_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO modelos (id, tipo, version, uri, metricas) VALUES (%s, %s, %s, %s, %s)",
                (model_id, kind, version, uri, json.dumps(metrics)),
            )
        if active:
            self.activate_model(model_id)
        return model_id

    def list_models(self, kind: str | None = None) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM modelos WHERE %s::text IS NULL OR tipo = %s ORDER BY creado DESC", (kind, kind)
            ).fetchall()
        return [_clean(r) for r in rows]

    def activate_model(self, model_id: str) -> dict | None:
        with self._connect() as conn, conn.transaction():
            row = conn.execute("SELECT tipo FROM modelos WHERE id = %s", (model_id,)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE modelos SET activo = false WHERE tipo = %s AND activo", (row["tipo"],))
            model = conn.execute("UPDATE modelos SET activo = true WHERE id = %s RETURNING *", (model_id,)).fetchone()
        return _clean(model)

    def active_model(self, kind: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM modelos WHERE tipo = %s AND activo", (kind,)).fetchone()
        return _clean(row) if row else None

    def ping(self) -> bool:
        with self._connect() as conn:
            conn.execute("SELECT 1")
        return True


def _clean(row: dict) -> dict:
    """uuid sin guiones y fechas en ISO, igual que el repositorio en memoria."""
    out = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()}
    out["id"] = str(out["id"]).replace("-", "")
    return out


def build_repository(settings: Settings) -> Repository:
    if settings.database_backend == "memory":
        return MemoryRepository()
    if settings.database_backend == "postgres":
        return PostgresRepository(settings.database_url)
    raise ValueError(f"DATABASE_BACKEND desconocido: {settings.database_backend!r} (memory | postgres)")
