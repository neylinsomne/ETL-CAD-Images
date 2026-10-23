"""Paso 2: detección de elementos compactos (ventanas, puertas, muros cortina, columnas).

Las cajas sirven para elementos compactos; el muro exterior NO se detecta con cajas
(su caja sería el edificio entero), se obtiene en `envelope.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from filtro_planos.config import Settings

# Clases que cierran vanos del muro exterior al rellenar desde el borde.
OPENING_LABELS = ("window", "door", "curtain_wall")


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, ...]  # x1, y1, x2, y2 en píxeles


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


class NoDetector:
    """Solo el paso clásico: sin modelo."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        return []


class StaticDetector:
    """Devuelve detecciones fijas (p. ej. las del ground truth, para evaluar el techo del método)."""

    def __init__(self, detections: list[Detection]):
        self._detections = detections

    def detect(self, image: np.ndarray) -> list[Detection]:
        return list(self._detections)


class DFineDetector:
    """D-FINE (Peng et al., ICLR 2025) afinado en ventanas/puertas.

    La carga del modelo se asume: aquí solo se valida que las rutas montadas existan.
    """

    def __init__(self, config_path: Path, weights_path: Path, score_threshold: float = 0.5):
        for path in (config_path, weights_path):
            if not path.exists():
                raise FileNotFoundError(f"No existe {path}: monta el modelo en /models (ver docker-compose.yml)")
        self.score_threshold = score_threshold
        self.model = self._load(config_path, weights_path)

    def _load(self, config_path: Path, weights_path: Path):
        # Se asume: YAMLConfig(config_path).model + checkpoint de weights_path, en eval().
        raise NotImplementedError("Carga de D-FINE pendiente (se asume)")

    def detect(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError("Inferencia de D-FINE pendiente (se asume)")


def build_detector(settings: Settings) -> Detector:
    if settings.detector == "none":
        return NoDetector()
    if settings.detector == "dfine":
        return DFineDetector(settings.dfine_config, settings.dfine_weights, settings.dfine_score_threshold)
    raise ValueError(f"DETECTOR desconocido: {settings.detector!r} (usa 'none' o 'dfine')")
