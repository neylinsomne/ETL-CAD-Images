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


def sparse_segments(strokes: np.ndarray, zone: np.ndarray, window: int, ratio: float) -> np.ndarray:
    """Tramos del límite con mucha menos tinta que lo típico del mismo plano: vanos y ventanas.

    Densidad = fracción de tinta dentro de la zona, en una ventana del orden del grosor del muro
    (normalizada por la zona, así el borde del muro no cuenta como vacío). Un muro macizo está
    cerca de 1; una ventana dibujada como rectángulo con líneas finas, muy por debajo. Es relativa
    a la mediana del plano, así que se adapta al estilo: con muros a doble línea, muro y ventana
    se parecen y no se marca nada.
    """
    k = (window, window)
    zone_f = zone.astype(np.float32)
    inked = cv2.blur((strokes & zone).astype(np.float32), k)
    covered = cv2.blur(zone_f, k)
    density = np.divide(inked, covered, out=np.zeros_like(inked), where=covered > 0)
    typical = float(np.median(density[zone])) if zone.any() else 0.0
    return zone & (density < ratio * typical)


def classify_strokes(gray: np.ndarray, result: EnvelopeResult, scale: float, params: EnvelopeParams) -> np.ndarray:
    """Etiqueta cada píxel de trazo del límite: WALL o GLASS.

    Es vidrio si el trazo es claro (en CubiCasa5K/CVC-FP el vidrio tiene gris mediano 122-160 y el
    muro 0-105) o si está en un tramo con poca tinta (ventana dibujada en negro, como en CVC-FP).
    """
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
    if params.glass_density_ratio > 0:
        window = max(3, int(round(params.glass_density_window * result.wall_thickness_px / scale)))
        wall &= ~sparse_segments(strokes, zone, window, params.glass_density_ratio)
    labels[in_zone] = GLASS
    labels[wall] = WALL
    return labels
