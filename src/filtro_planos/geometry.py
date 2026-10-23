"""Utilidades de geometría compartidas."""

from __future__ import annotations

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid


def polygon_parts(geom: BaseGeometry) -> list[Polygon]:
    """Partes poligonales de cualquier geometría.

    make_valid puede devolver una GeometryCollection (polígonos + líneas sueltas) cuando
    el contorno se toca a sí mismo; aquí se rescatan los polígonos de cualquier anidación.
    """
    if isinstance(geom, Polygon):
        return [] if geom.is_empty or geom.area <= 0 else [geom]
    return [p for g in getattr(geom, "geoms", ()) for p in polygon_parts(g)]


def valid_polygons(coords) -> list[Polygon]:
    if len(coords) < 3:
        return []
    return polygon_parts(make_valid(Polygon(coords)))


def largest_valid_polygon(coords) -> Polygon | None:
    parts = valid_polygons(coords)
    return max(parts, key=lambda p: p.area) if parts else None
