"""Paso 1: limpieza clásica del plano.

Binariza, quita texto y marcos de lámina, y rellena muros dibujados a doble línea
o con achurado para que se comporten como muros sólidos.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """cv2.imread falla con rutas no ASCII en Windows; imdecode no."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(Path(path).suffix or ".png", image)
    if not ok:
        raise ValueError(f"No se pudo codificar {path}")
    encoded.tofile(str(path))


def to_gray(image: np.ndarray) -> np.ndarray:
    """BGR/BGRA/gris a gris uint8; la transparencia se compone sobre blanco."""
    if image.dtype != np.uint8:
        span = float(image.max() - image.min()) or 1.0
        image = ((image - image.min()) * (255.0 / span)).astype(np.uint8)
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        image = (image[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def downscale(gray: np.ndarray, max_side: int, keep_lines: bool = False) -> tuple[np.ndarray, float]:
    """Reduce el plano a max_side px de lado mayor; devuelve (imagen, escala).

    Promediar (INTER_AREA) aclara las líneas de 1 px: a escala 0,46 una ventana queda en grises
    de 155-238 y no pasa como tinta. keep_lines aplica antes un filtro de mínimo que las conserva
    oscuras, pero engrosa todo: subió F1 sin detector en CVC-FP (0,79 -> 0,85) y lo bajó con
    detecciones (0,93 -> 0,85), así que queda apagado por defecto.
    """
    scale = min(1.0, max_side / max(gray.shape))
    if scale >= 1.0:
        return gray, 1.0
    k = int(math.ceil(1 / scale))
    if keep_lines and k >= 2:
        gray = cv2.erode(gray, np.ones((k, k), np.uint8))
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def ink_mask(gray: np.ndarray, max_ink_level: int) -> np.ndarray:
    """Tinta = True. Otsu, con tope para que los rellenos claros de color no cuenten como trazo."""
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray <= min(otsu, max_ink_level)


def faint_lines(gray: np.ndarray, min_contrast: int, size: int = 9) -> np.ndarray:
    """Trazos finos más oscuros que su entorno por >= min_contrast, aunque sean grises claros.

    Black-hat (cierre - original): una vidriera dibujada en gris 221 sobre fondo 254 no pasa
    como tinta, pero sí como línea. Un relleno claro de color es un área, no una línea, y no sale.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel) >= min_contrast


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Quita texto, cotas sueltas y ruido de escaneo."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels]


def remove_frame_lines(mask: np.ndarray, span: float = 0.9, max_fill: float = 0.03) -> np.ndarray:
    """Quita el marco de la lámina: si no, el relleno desde el borde no entra nunca al dibujo."""
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    bw, bh, area = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
    frame = (bw >= span * w) & (bh >= span * h) & (area < max_fill * bw * bh)
    frame[0] = False
    return mask & ~frame[labels]


def distance_to_background(mask: np.ndarray) -> np.ndarray:
    """distanceTransform acotado: sin píxeles de fondo devuelve ~FLT_MAX (y luego infinito)."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.minimum(dist, float(max(mask.shape)))


def stroke_width(mask: np.ndarray, percentile: float = 90) -> float:
    """Ancho de los trazos más gruesos: percentil del ancho (2 x distancia) en el eje medio.

    Los muros son los trazos más gruesos del plano; el p90 se acerca a su grosor
    (mediana 1.1x el anotado en CubiCasa5K val), sin necesitar el relleno.
    """
    dist = distance_to_background(mask)
    ridge = (dist >= 1) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))
    return float(2 * np.percentile(dist[ridge], percentile)) if ridge.any() else 1.0


def remove_thin_strokes(mask: np.ndarray, min_width: float) -> np.ndarray:
    """Apertura morfológica: borra trazos más delgados que min_width (cotas, ejes, mobiliario)."""
    k = int(round(min_width))
    if k < 2:
        return mask
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, disk) > 0


def bridge_wall_gaps(mask: np.ndarray, min_run: float, max_gap: float) -> np.ndarray:
    """Cierra vanos (ventanas, puertas) entre tramos alineados de un mismo muro.

    Solo une tramos largos (>= min_run) en la dirección del muro: una ventana queda entre dos
    tramos de la misma fila; un muro perpendicular o un símbolo cercano no forma tramo y no se une.
    """
    run, gap = int(round(min_run)), int(round(max_gap))
    if run < 2 or gap < 3:
        return mask
    m = mask.astype(np.uint8)
    out = mask.copy()
    for run_shape, gap_shape in (((run, 1), (gap, 1)), ((1, run), (1, gap))):
        runs = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, run_shape))
        # Fuera de la imagen = vacío: con el borde por defecto, el cierre estira hasta el borde
        # todo tramo que quede a menos de max_gap/2 de él.
        closed = cv2.morphologyEx(
            runs,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, gap_shape),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out |= closed > 0
    return out


def close_glazing(solid: np.ndarray, thin: np.ndarray, wall_px: float, max_len: float) -> np.ndarray:
    """Vidrieras: trazo fino recto que va de un sólido a otro (muro, jamba o poste).

    Regla clásica de ventana en planos: el vidrio une las dos jambas. En un porche acristalado
    va de poste a poste. Devuelve cada tramo engrosado al ancho de lo que une.
    """
    run = max(3, int(round(2 * wall_px)))
    reach = max(2, int(round(wall_px / 2)))
    out = np.zeros(solid.shape, bool)
    for transpose in (False, True):  # horizontal y vertical con el mismo código
        s = solid.T if transpose else solid
        t = thin.T if transpose else thin
        o = out.T if transpose else out
        line = cv2.getStructuringElement(cv2.MORPH_RECT, (run, 1))
        glass = cv2.morphologyEx(t.astype(np.uint8), cv2.MORPH_OPEN, line)
        _, _, stats, _ = cv2.connectedComponentsWithStats(glass, connectivity=8)
        for x, y, w, h, _ in stats[1:]:
            if w > max_len:
                continue
            rows = slice(max(0, y - reach), y + h + reach)
            left = s[rows, max(0, x - reach) : x + 1].any(axis=1)
            right = s[rows, x + w - 1 : x + w + reach].any(axis=1)
            if left.any() and right.any():
                jamb_rows = np.flatnonzero(left | right) + rows.start
                o[jamb_rows.min() : jamb_rows.max() + 1, x : x + w] = True
    return out


def fill_thin_cavities(ink: np.ndarray, max_radius: float) -> np.ndarray:
    """Rellena huecos blancos cerrados y delgados: el interior de muros a doble línea,
    el achurado y el espacio entre las líneas de una ventana.

    Un ambiente tiene un radio inscrito grande; un muro hueco, uno del orden de medio grosor.
    """
    free = (~ink).astype(np.uint8)
    n, labels = cv2.connectedComponents(free, connectivity=4)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 3)
    max_dist = np.zeros(n, np.float32)
    np.maximum.at(max_dist, labels.ravel().astype(np.intp), dist.ravel())
    touches_border = np.zeros(n, bool)
    touches_border[np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])] = True
    cavity = (max_dist <= max_radius) & ~touches_border
    cavity[0] = False
    return ink | cavity[labels]
