"""Rutas de los datasets descargados en data/raw y su carga uniforme."""

from __future__ import annotations

from pathlib import Path

from filtro_planos.annotations import (
    Annotation,
    cubicasa_image_id,
    cubicasa_index,
    cvc_image_id,
    cvc_index,
    load_cubicasa,
    load_cvc,
)
from filtro_planos.config import PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SOURCES = {
    "cubicasa5k": (RAW_DIR / "cubicasa5k", cubicasa_index, load_cubicasa, cubicasa_image_id),
    "cvc_fp": (RAW_DIR / "cvc_fp" / "ImagesGT", cvc_index, load_cvc, cvc_image_id),
}


def index(dataset: str) -> list[tuple[Path, str]]:
    root, index_fn, _, _ = SOURCES[dataset]
    return index_fn(root)


def load(dataset: str, path: Path) -> Annotation:
    return SOURCES[dataset][2](path)


def image_id(dataset: str, path: Path) -> str:
    return SOURCES[dataset][3](path)


def style(image_id: str) -> str:
    """Estilo de dibujo para balancear: colorful / high_quality / high_quality_architectural / cvc_fp."""
    dataset, *rest = image_id.split("/")
    return dataset if dataset == "cvc_fp" else rest[0]
