"""Coloreado de trazos del límite en dos clases, a resolución completa.

No se pinta una banda: se pintan los mismos trazos del dibujo (finos o gruesos, como estén),
para que el siguiente modelo vea el vidrio como línea azul en vez de una línea gris de 1 px.

El límite se busca a escala reducida (envelope.py), pero los trazos se clasifican sobre la
imagen original: al reducir, el vidrio se aclara y desaparece.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from filtro_planos.envelope import EnvelopeParams, EnvelopeResult
from filtro_planos.preprocess import faint_lines, ink_mask

BACKGROUND, WALL, GLASS = 0, 1, 2


def _upscale(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def classify_strokes(gray: np.ndarray, result: EnvelopeResult, scale: float, params: EnvelopeParams) -> np.ndarray:
    """Etiqueta cada píxel de trazo del límite: WALL (grueso y oscuro) o GLASS (fino o claro)."""
    labels = np.zeros(gray.shape, np.uint8)
    if not result.exterior_wall.any():
        return labels
    reach = int(math.ceil(1 / scale)) + 1  # un píxel reducido ocupa ~1/scale píxeles originales
    band = _upscale(result.exterior_wall, gray.shape).astype(np.uint8)
    zone = cv2.dilate(band, np.ones((2 * reach + 1,) * 2, np.uint8)) > 0
    thick = _upscale(result.thick, gray.shape)
    strokes = ink_mask(gray, params.max_ink_level) | faint_lines(gray, params.stroke_faint_contrast)
    in_zone = strokes & zone
    wall = in_zone & thick & (gray < params.glass_min_gray)
    labels[in_zone] = GLASS
    labels[wall] = WALL
    return labels
