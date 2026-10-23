"""Salidas visuales: trazos coloreados (rosado muro, azul vidrio) y canales de pista."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from filtro_planos.detector import Detection
from filtro_planos.strokes import GLASS, WALL

PINK_BGR = (160, 64, 255)  # muro exterior
BLUE_BGR = (255, 110, 0)  # vidrio en el límite: ventanas, muro cortina
DETECTION_BGR = (0, 170, 255)


def color_strokes(gray: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Pinta solo los píxeles de trazo etiquetados; el grosor del dibujo no cambia."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgr[labels == WALL] = PINK_BGR
    bgr[labels == GLASS] = BLUE_BGR
    return bgr


def overlay(
    gray: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = PINK_BGR, alpha: float = 0.6
) -> np.ndarray:
    """Pinta una máscara (banda) sobre el plano, semitransparente. Para revisar el límite a ojo."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tinted = bgr.copy()
    tinted[mask] = color
    return cv2.addWeighted(tinted, alpha, bgr, 1 - alpha, 0)


def draw_detections(bgr: np.ndarray, detections: Iterable[Detection], color=DETECTION_BGR) -> np.ndarray:
    out = bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out


def hint_channels(gray: np.ndarray, wall: np.ndarray, glass: np.ndarray) -> np.ndarray:
    """Para un modelo: B = plano original, G = trazos de muro, R = trazos de vidrio.

    El plano es blanco y negro, así que RGB repite el mismo canal tres veces; aquí los otros dos
    llevan las clases sin tapar ningún trazo.
    """
    return np.dstack([gray, wall.astype(np.uint8) * 255, glass.astype(np.uint8) * 255])
