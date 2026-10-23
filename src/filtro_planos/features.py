"""Features por elemento para el dataframe: geometría + contexto (vecinos).

El contexto es la parte de "si hay X al lado, seguro es Y" (razonamiento contextual):
un muro exterior suele tener ventanas, tocar un solo ambiente y estar sobre la envolvente
convexa del piso; uno interior separa dos ambientes.
"""

from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np
from shapely import STRtree
from shapely.geometry import Polygon

from filtro_planos.annotations import Annotation, Element

OPENINGS = ("window", "door")
TOUCH_PX = 3.0  # holgura para "toca a"


def geometry_features(polygon: Polygon) -> dict:
    rect = polygon.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    sides = [math.dist((xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(2)]
    long_i = int(np.argmax(sides))
    length = max(sides)
    angle = math.degrees(math.atan2(ys[long_i + 1] - ys[long_i], xs[long_i + 1] - xs[long_i])) % 180
    x1, y1, x2, y2 = polygon.bounds
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "area": polygon.area,
        "length": length,
        "thickness": polygon.area / length if length > 0 else 0.0,
        "angle": angle,
        "axis_aligned": min(angle % 90, 90 - angle % 90) < 2.0,
        "n_vertices": len(polygon.exterior.coords) - 1,
        "rectangularity": polygon.area / rect.area if rect.area > 0 else 0.0,
    }


def element_rows(ann: Annotation) -> list[dict]:
    """Una fila por elemento con geometría y contexto."""
    by_floor: dict[int, list[Element]] = defaultdict(list)
    for el in ann.elements:
        by_floor[el.floor].append(el)

    rows = []
    for floor_elements in by_floor.values():
        walls = [e for e in floor_elements if e.cls == "wall"]
        openings = [e for e in floor_elements if e.cls in OPENINGS]
        rooms = [e for e in floor_elements if e.cls == "room"]
        hull = _floor_hull(walls)
        hull_scale = math.sqrt(hull.area) if hull is not None and hull.area > 0 else 1.0
        opening_tree = STRtree([e.polygon for e in openings]) if openings else None
        room_tree = STRtree([e.polygon for e in rooms]) if rooms else None

        for el in floor_elements:
            row = {
                "image_id": ann.image_id,
                "element_id": el.element_id,
                "dataset": ann.dataset,
                "cls": el.cls,
                "subtype": el.subtype,
                "floor": el.floor,
                "parent_id": el.parent_id,
                "is_exterior": el.is_exterior,
                "geometry_wkt": el.polygon.wkt,
                **geometry_features(el.polygon),
            }
            if el.cls in ("wall", "railing", *OPENINGS):
                probe = el.polygon.buffer(TOUCH_PX)
                near_open = [openings[i] for i in opening_tree.query(probe, "intersects")] if opening_tree else []
                near_rooms = [rooms[i] for i in room_tree.query(probe, "intersects")] if room_tree else []
                indoor = [r for r in near_rooms if "Outdoor" not in r.subtype]
                row.update(
                    n_windows=sum(o.cls == "window" for o in near_open if o is not el),
                    n_doors=sum(o.cls == "door" for o in near_open if o is not el),
                    n_rooms_adjacent=len(indoor),
                    touches_outdoor=len(indoor) != len(near_rooms),
                    dist_to_hull_norm=(el.polygon.distance(hull.exterior) / hull_scale) if hull is not None else None,
                )
            rows.append(row)
    return rows


def _floor_hull(walls: list[Element]) -> Polygon | None:
    if not walls:
        return None
    points = np.concatenate([np.asarray(w.polygon.exterior.coords) for w in walls])
    hull = Polygon(points).convex_hull if len(points) >= 3 else None
    return hull if isinstance(hull, Polygon) and hull.area > 0 else None


def rasterize(polygons: list[Polygon], shape: tuple[int, int], scale: float = 1.0) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for p in polygons:
        # De a uno: fillPoly con varios polígonos superpuestos hace XOR (ventana dentro de muro = hueco).
        cv2.fillPoly(mask, [np.round(np.asarray(p.exterior.coords) * scale).astype(np.int32)], 1)
    return mask > 0


def envelope_ground_truth(ann: Annotation, shape: tuple[int, int], scale: float = 1.0) -> np.ndarray:
    """Máscara del muro exterior anotado (muros externos + sus ventanas y puertas)."""
    polygons = [e.polygon for e in ann.elements if e.cls in ("wall", *OPENINGS) and e.is_exterior]
    return rasterize(polygons, shape, scale)
