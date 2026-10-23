"""Configuración por variables de entorno.

Cada servicio externo (S3, Redis, Postgres vía PgBouncer) tiene un reemplazo local para
desarrollo y pruebas: disco, cola en línea y repositorio en memoria. En producción solo se
cambian las variables (ver .env.example).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # Paso del filtro
    envelope_params: Path  # parámetros ajustados (scripts/tune.py)
    detector: str  # "none" | "dfine"
    dfine_config: Path
    dfine_weights: Path
    dfine_score_threshold: float
    # Ingesta
    pdf_dpi: int
    max_upload_mb: int
    max_pixels: int
    # Almacenamiento de archivos: "local" | "s3"
    storage_backend: str
    storage_root: Path
    s3_bucket: str
    s3_endpoint_url: str | None
    # Cola de trabajos: "inline" | "redis"
    queue_backend: str
    redis_url: str
    queue_stream: str  # ingesta (worker)
    training_stream: str  # entrenamiento (trainer), aparte para no frenar la ingesta
    # Metadatos: "memory" | "postgres" (DATABASE_URL apunta a PgBouncer)
    database_backend: str
    database_url: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            envelope_params=Path(_env("ENVELOPE_PARAMS", str(PROJECT_ROOT / "config" / "envelope_params.json"))),
            detector=_env("DETECTOR", "none").lower(),
            dfine_config=Path(_env("DFINE_CONFIG", "/models/dfine/config.yml")),
            dfine_weights=Path(_env("DFINE_WEIGHTS", "/models/dfine/model.pth")),
            dfine_score_threshold=float(_env("DFINE_SCORE_THRESHOLD", "0.5")),
            pdf_dpi=int(_env("PDF_DPI", "200")),
            max_upload_mb=int(_env("MAX_UPLOAD_MB", "100")),
            max_pixels=int(_env("MAX_PIXELS", "150000000")),
            storage_backend=_env("STORAGE_BACKEND", "local").lower(),
            storage_root=Path(_env("STORAGE_ROOT", str(PROJECT_ROOT / "data" / "store"))),
            s3_bucket=_env("S3_BUCKET", "filtro-planos"),
            s3_endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
            queue_backend=_env("QUEUE_BACKEND", "inline").lower(),
            redis_url=_env("REDIS_URL", "redis://localhost:6379/0"),
            queue_stream=_env("QUEUE_STREAM", "filtro-planos:trabajos"),
            training_stream=_env("TRAINING_STREAM", "filtro-planos:entrenamiento"),
            database_backend=_env("DATABASE_BACKEND", "memory").lower(),
            database_url=_env("DATABASE_URL", "postgresql://planos:planos@localhost:6432/planos"),
        )
