"""Pasos 3-5: cerrar vanos, rellenar desde el borde y extraer el muro exterior.

1. Limpieza (preprocess) -> máscara de muros "sólidos". Se borran los trazos más delgados que
   ~medio muro: cotas, ejes y mobiliario no deben frenar el relleno.
2. Las cajas de ventanas/puertas detectadas se pintan como sólidas (cierran vanos).
3. Vanos sin detectar: se cierran los huecos entre tramos alineados de un mismo muro (ventanas,
   puertas) de hasta ~bridge_factor x grosor, y el relleno usa una sonda gruesa que no cabe por
   los vanos restantes de menos de ~gap_factor x grosor.
   (Un cierre con disco no sirve: en muros delgados el disco entra al vano por ambos lados. Un
   cierre en línea sin exigir tramos de muro tampoco: une el muro con lo que haya en la fila.)
   Ventanas más anchas que el puente sí dejan escapar el relleno: eso lo resuelve D-FINE.
4. Huella del edificio = lo que no es afuera.
5. Muro exterior = parte sólida alcanzable desde afuera hasta ~1.5 grosores de profundidad.
   Va por contornos, no por rectas (Hough), así que las esquinas curvas salen solas.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from filtro_planos.detector import OPENING_LABELS, Detection
from filtro_planos.geometry import valid_polygons
from filtro_planos.preprocess import (
    bridge_wall_gaps,
    close_glazing,
    distance_to_background,
    faint_lines,
    fill_thin_cavities,
    ink_mask,
    remove_frame_lines,
    remove_small_components,
    remove_thin_strokes,
    stroke_width,
    to_gray,
)

_KERNEL_3 = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class EnvelopeParams:
    max_side: int = 1600  # lado mayor al que se reduce el plano
    keep_thin_lines: bool = False  # reducir con filtro de mínimo (ver preprocess.downscale)
    faint_line_contrast: int = 0  # > 0: también cuentan como tinta las líneas tenues con ese contraste
    max_ink_level: int = 160
    max_ink_frac: float = 0.35  # más tinta que esto = fondo oscuro o rellenos fuertes: no se procesa
    max_wall_frac: float = 0.05  # un muro no es más grueso que 5% del lado mayor
    min_component_area_frac: float = 2e-5  # texto y ruido, relativo al área de la imagen
    cavity_radius_frac: float = 0.01  # radio máx. de un muro hueco, relativo al lado mayor
    thin_stroke_factor: float = 0.5  # se borran trazos < thin_stroke_factor x ancho p90
    bridge_factor: float = 8.0  # vanos entre tramos alineados de hasta bridge_factor x ancho p90 (0 = no)
    glazing_factor: float = 0.0  # vidrieras (trazo fino entre sólidos) de hasta glazing_factor x ancho (0 = no)
    cropped_factor: float = 1.5  # sólido en el borde > 1,5 muros (entra y sale un muro) = plano recortado
    gap_factor: float = 2.0  # la sonda no pasa por vanos de menos de gap_factor x grosor
    depth_factor: float = 1.5  # profundidad de la banda exterior ~ depth_factor x grosor
    corner_fill: bool = True  # incluir sólido cercano al exterior aunque no esté conectado al muro
    # Coloreado de trazos (strokes.py): dentro del límite, trazo grueso y oscuro = muro; el resto = vidrio.
    # En CubiCasa5K/CVC-FP el gris mediano del vidrio es 122-160 y el del muro 0-105.
    glass_min_gray: int = 115
    stroke_faint_contrast: int = 30  # el vidrio suele ser tenue: siempre se buscan líneas tenues al colorear
    min_footprint_frac: float = 0.02
    max_footprint_frac: float = 0.95
    min_component_frac: float = 0.1  # huellas secundarias >= 10% de la mayor
    simplify_tolerance_px: float = 1.5
    opening_labels: tuple[str, ...] = OPENING_LABELS

    @classmethod
    def from_dict(cls, data: dict) -> EnvelopeParams:
        """Los campos que falten quedan por defecto; los desconocidos se ignoran."""
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        if "opening_labels" in data:
            data["opening_labels"] = tuple(data["opening_labels"])
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> EnvelopeParams:
        """Lee los parámetros ajustados (scripts/tune.py o el registro de modelos)."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class EnvelopeResult:
    status: str  # "ok" | "cropped" | "no_footprint" | "unbounded" | "too_dark"
    wall_thickness_px: float
    footprint_frac: float
    thick: np.ndarray  # trazos gruesos tal como están dibujados (antes de cerrar vanos)
    solid: np.ndarray
    outside: np.ndarray
    footprint: np.ndarray
    exterior_wall: np.ndarray
    polygons: list[Polygon] = field(default_factory=list)


def boxes_mask(detections: Iterable[Detection], shape: tuple[int, int], labels: Sequence[str]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for det in detections:
        if det.label in labels:
            x1, y1, x2, y2 = (int(round(v)) for v in det.box)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)
    return mask > 0


def _border_connected(free: np.ndarray) -> np.ndarray:
    n, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    is_outside = np.zeros(n, bool)
    is_outside[np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])] = True
    is_outside[0] = False
    return is_outside[labels]


def outside_region(barrier: np.ndarray, probe_radius: int = 0) -> np.ndarray:
    """Relleno desde el borde de la imagen.

    Con probe_radius > 0 rellena con un disco de ese radio: no pasa por vanos de menos de
    2 x probe_radius, y el área que barre el disco es el afuera.
    """
    if probe_radius <= 0:
        return _border_connected(~barrier)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * probe_radius + 1,) * 2)
    centers = _border_connected(cv2.dilate(barrier.astype(np.uint8), disk) == 0)
    return (cv2.dilate(centers.astype(np.uint8), disk) > 0) & ~barrier


def main_components(mask: np.ndarray, min_component_frac: float) -> np.ndarray:
    """La componente mayor y las que tengan al menos min_component_frac de su área."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    areas[0] = 0
    keep = areas >= min_component_frac * areas.max()
    keep[0] = False
    return keep[labels]


def estimate_wall_thickness(solid: np.ndarray, outside: np.ndarray, footprint: np.ndarray) -> float:
    """Grosor del trazo que da al exterior: 2 x distancia en el eje medio de esos trazos."""
    dist_in = distance_to_background(solid)
    dist_out = distance_to_background(~outside)
    ridge = (dist_in >= 1) & (dist_in >= cv2.dilate(dist_in, _KERNEL_3))
    medial = ridge & footprint & (np.abs(dist_out - dist_in) <= 1.5)
    values = dist_in[medial] if medial.sum() >= 20 else dist_in[ridge & solid]
    if values.size == 0:
        return 1.0
    return float(max(1.0, 2.0 * np.median(values)))


def geodesic_band(region: np.ndarray, seed: np.ndarray, depth: int) -> np.ndarray:
    """Crece `seed` dentro de `region` hasta `depth` píxeles (dilatación geodésica)."""
    band = (seed & region).astype(np.uint8)
    region_u8 = region.astype(np.uint8)
    for _ in range(depth):
        grown = np.bitwise_and(cv2.dilate(band, _KERNEL_3), region_u8)
        if np.array_equal(grown, band):
            break
        band = grown
    return band > 0


def footprint_polygons(footprint: np.ndarray, tolerance: float) -> list[Polygon]:
    """Contorno exterior de cada huella. approxPolyDP con tolerancia baja conserva las curvas."""
    contours, _ = cv2.findContours(footprint.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons: list[Polygon] = []
    for contour in contours:
        polygons.extend(valid_polygons(cv2.approxPolyDP(contour, tolerance, True)[:, 0, :]))
    largest = max((p.area for p in polygons), default=0.0)
    return [p for p in polygons if p.area >= 0.01 * largest]  # sin lóbulos sueltos de make_valid


def find_envelope(
    image: np.ndarray,
    detections: Iterable[Detection] = (),
    params: EnvelopeParams | None = None,
) -> EnvelopeResult:
    params = params or EnvelopeParams()
    gray = to_gray(image)
    h, w = gray.shape
    ink = ink_mask(gray, params.max_ink_level)
    if ink.mean() > params.max_ink_frac:
        empty = np.zeros((h, w), bool)
        return EnvelopeResult("too_dark", 0.0, 0.0, empty, empty, empty, empty, empty)
    if params.faint_line_contrast > 0:
        ink |= faint_lines(gray, params.faint_line_contrast)
    max_wall = params.max_wall_frac * max(h, w)
    ink = remove_small_components(ink, max(4, int(params.min_component_area_frac * h * w)))
    ink = remove_frame_lines(ink)
    filled = fill_thin_cavities(ink, params.cavity_radius_frac * max(h, w))
    stroke = min(stroke_width(filled), max_wall)
    solid = remove_thin_strokes(filled, params.thin_stroke_factor * stroke)
    thick = solid.copy()
    if params.glazing_factor > 0:
        solid |= close_glazing(solid, filled & ~solid, stroke, params.glazing_factor * stroke)
    solid |= boxes_mask(detections, (h, w), params.opening_labels)
    if params.bridge_factor > 0:
        solid = bridge_wall_gaps(solid, 2 * stroke, params.bridge_factor * stroke)

    # Pasada 1: huella aproximada, solo para medir el grosor de muro.
    outside = outside_region(solid)
    footprint = main_components(~outside, params.min_component_frac)
    thickness = min(estimate_wall_thickness(solid, outside, footprint), max_wall)

    # Pasada 2: sonda gruesa que no entra por vanos del orden de una puerta.
    probe_radius = max(1, int(round(params.gap_factor * thickness / 2)))
    outside = outside_region(solid, probe_radius)
    footprint = main_components(~outside, params.min_component_frac)
    footprint_frac = float(footprint.mean())

    on_border = solid[0].sum() + solid[-1].sum() + solid[:, 0].sum() + solid[:, -1].sum()
    if footprint_frac < params.min_footprint_frac:
        status = "no_footprint"
    elif footprint_frac > params.max_footprint_frac:
        status = "unbounded"
    elif params.cropped_factor > 0 and on_border > params.cropped_factor * stroke:
        status = "cropped"  # el edificio sigue fuera de la imagen: el relleno entra por el corte
    else:
        status = "ok"

    dist_out = distance_to_background(~outside)
    region = solid & footprint
    depth = int(math.ceil(params.depth_factor * thickness))
    # Geodésica: sigue el muro. Distancia directa: pilares de esquina sueltos que la sonda no rozó.
    exterior = geodesic_band(region, region & (dist_out <= 1.5), depth)
    if params.corner_fill:
        exterior |= region & (dist_out <= depth)

    return EnvelopeResult(
        status=status,
        wall_thickness_px=thickness,
        footprint_frac=footprint_frac,
        thick=thick,
        solid=solid,
        outside=outside,
        footprint=footprint,
        exterior_wall=exterior,
        polygons=footprint_polygons(footprint, params.simplify_tolerance_px) if status in ("ok", "cropped") else [],
    )
