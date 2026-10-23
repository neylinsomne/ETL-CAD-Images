from filtro_planos.annotations import parse_points
from filtro_planos.geometry import valid_polygons


def test_bowtie_keeps_both_lobes():
    # Contorno que se cruza a sí mismo: make_valid lo parte en dos triángulos.
    parts = valid_polygons([(0, 0), (10, 10), (10, 0), (0, 10)])
    assert len(parts) == 2
    assert abs(sum(p.area for p in parts) - 50) < 1e-6


def test_polygon_with_dangling_spike_is_rescued():
    # Un "pelo" de ida y vuelta: make_valid devuelve GeometryCollection(Polygon, LineString).
    parts = valid_polygons([(0, 0), (10, 0), (10, 10), (15, 10), (10, 10), (0, 10)])
    assert len(parts) == 1
    assert abs(parts[0].area - 100) < 1e-6


def test_parse_points_accepts_cvc_and_cubicasa_formats():
    assert parse_points("485.5,195.5 510,200 506.5,692.5 482.5,694.5 ").area > 0
    assert parse_points("2239.97,1572.27 2754.68,1572.27 2791.68,1535.27 2202.97,1535.27").area > 0
    assert parse_points("1,1 2,2") is None
