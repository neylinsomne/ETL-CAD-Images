from dataclasses import replace

import cv2
import numpy as np

from filtro_planos.detector import Detection
from filtro_planos.envelope import EnvelopeParams, find_envelope
from filtro_planos.render import hint_channels

BLACK = 0
WALL = 12
X1, Y1, X2, Y2 = 100, 100, 500, 450  # caja exterior del edificio
BUILDING_AREA = (X2 - X1) * (Y2 - Y1)


def blank(size=600):
    return np.full((size, size), 255, np.uint8)


def solid_building(img, gap=None):
    """Muros macizos de 12 px; `gap` = (x_ini, x_fin) abre un vano en el muro inferior."""
    cv2.rectangle(img, (X1, Y1), (X2 - 1, Y1 + WALL - 1), BLACK, -1)
    cv2.rectangle(img, (X1, Y2 - WALL), (X2 - 1, Y2 - 1), BLACK, -1)
    cv2.rectangle(img, (X1, Y1), (X1 + WALL - 1, Y2 - 1), BLACK, -1)
    cv2.rectangle(img, (X2 - WALL, Y1), (X2 - 1, Y2 - 1), BLACK, -1)
    cv2.rectangle(img, (297, Y1), (302, Y2 - 1), BLACK, -1)  # muro interior
    if gap:
        img[Y2 - WALL : Y2, gap[0] : gap[1]] = 255
    return img


def test_door_gap_is_closed_and_interior_wall_excluded():
    result = find_envelope(solid_building(blank(), gap=(200, 230)))

    assert result.status == "ok"
    assert abs(result.wall_thickness_px - WALL) <= 3
    assert abs(result.footprint.sum() - BUILDING_AREA) / BUILDING_AREA < 0.03
    assert result.exterior_wall[Y1 + 5, 250]  # muro superior
    assert not result.outside[275, 200]  # el relleno no se escapó al cuarto
    assert not result.exterior_wall[275, 299]  # muro interior, lejos del borde
    assert len(result.polygons) == 1


def test_curved_corner_is_kept_in_polygon_and_band():
    img = blank()
    r = 120
    cv2.line(img, (X1, Y1), (X2 - r, Y1), BLACK, WALL)
    cv2.line(img, (X2, Y1 + r), (X2, Y2), BLACK, WALL)
    cv2.line(img, (X2, Y2), (X1, Y2), BLACK, WALL)
    cv2.line(img, (X1, Y2), (X1, Y1), BLACK, WALL)
    cv2.ellipse(img, (X2 - r, Y1 + r), (r, r), 0, 270, 360, BLACK, WALL)

    result = find_envelope(img)

    assert result.status == "ok"
    arc_x = int(X2 - r + r * np.cos(np.radians(315)))
    arc_y = int(Y1 + r + r * np.sin(np.radians(315)))
    assert result.exterior_wall[arc_y, arc_x]
    poly = result.polygons[0]
    assert len(poly.exterior.coords) > 8  # la curva no se colapsa a un rectángulo
    square_corner_area = r * r - np.pi * r * r / 4
    expected = (X2 - X1 + WALL) * (Y2 - Y1 + WALL) - square_corner_area
    assert abs(poly.area - expected) / expected < 0.05


def test_double_line_walls_fill_the_cavity():
    img = blank()
    cv2.rectangle(img, (X1, Y1), (X2, Y2), BLACK, 2)
    cv2.rectangle(img, (X1 + WALL, Y1 + WALL), (X2 - WALL, Y2 - WALL), BLACK, 2)
    cv2.line(img, (300, Y1 + WALL), (300, Y2 - WALL), BLACK, 2)

    result = find_envelope(img)

    assert result.status == "ok"
    assert result.exterior_wall[Y1 + WALL // 2, 250]  # entre las dos líneas
    assert result.wall_thickness_px >= WALL - 2


def test_wide_window_only_closes_with_detection():
    img = solid_building(blank(), gap=(150, 290))  # vano de ~12 grosores: el cierre no alcanza

    without = find_envelope(img)
    window = Detection("window", 0.9, (150, Y2 - WALL, 290, Y2 - 1))
    with_det = find_envelope(img, [window])

    assert without.outside[275, 200]  # sin detección, el relleno entra al cuarto izquierdo
    assert not with_det.outside[275, 200]
    assert with_det.status == "ok"
    assert abs(with_det.footprint.sum() - BUILDING_AREA) / BUILDING_AREA < 0.03
    assert with_det.exterior_wall[Y2 - 5, 220]


def test_window_gap_is_bridged_and_highlighted():
    img = solid_building(blank(), gap=(150, 210))  # ventana de 5 grosores, sin detección
    cv2.line(img, (150, Y2 - WALL // 2), (209, Y2 - WALL // 2), BLACK, 1)  # vidrio: trazo fino

    result = find_envelope(img)

    assert not result.outside[275, 200]  # el relleno no entra por la ventana
    assert result.exterior_wall[Y2 - WALL // 2, 180]  # el resaltado sigue de corrido sobre la ventana


def test_dimension_line_does_not_block_the_flood_fill():
    img = solid_building(blank())
    cv2.line(img, (X1, Y2 + 40), (X2, Y2 + 40), BLACK, 1)  # cota paralela al muro, por fuera
    for x in (X1, X2):
        cv2.line(img, (x, Y2 + 30), (x, Y2 + 50), BLACK, 1)

    result = find_envelope(img)

    assert result.outside[Y2 + 20, 300]  # la franja entre muro y cota sigue siendo afuera


def test_cropped_plan_is_flagged():
    img = blank()
    cv2.rectangle(img, (-50, 100), (500, 450), BLACK, WALL)  # el edificio sigue fuera de la imagen

    result = find_envelope(img)

    assert result.status == "cropped"


def test_bridging_does_not_extend_walls_to_the_image_border():
    img = blank()
    cv2.rectangle(img, (X1, Y1), (X2, Y2), BLACK, WALL)
    cv2.rectangle(img, (X1 - 60, 300), (X1 - 60 + 3 * WALL, 300 + WALL), BLACK, -1)  # tramo suelto afuera

    result = find_envelope(img)

    assert not result.solid[300 + WALL // 2, 2]  # nada estirado hasta el borde izquierdo


def test_corner_pillar_is_part_of_the_envelope():
    img = solid_building(blank())
    cv2.rectangle(img, (X2, Y2), (X2 + WALL, Y2 + WALL), BLACK, -1)  # pilar suelto en la esquina
    img[Y2 : Y2 + 2, X2 : X2 + WALL + 1] = 255  # separado del muro por una raya blanca

    result = find_envelope(img)

    assert result.exterior_wall[Y2 + WALL // 2 + 1, X2 + WALL // 2]


def test_sheet_frame_does_not_block_the_flood_fill():
    img = solid_building(blank())
    cv2.rectangle(img, (5, 5), (594, 594), BLACK, 2)

    result = find_envelope(img)

    assert result.status == "ok"
    assert abs(result.footprint.sum() - BUILDING_AREA) / BUILDING_AREA < 0.03


def test_dark_plan_is_flagged_instead_of_processed():
    img = np.full((600, 600), 40, np.uint8)  # fondo oscuro: casi todo es "tinta"
    img[200:400, 200:400] = 255

    result = find_envelope(img)

    assert result.status == "too_dark"
    assert not result.exterior_wall.any()


def test_light_gray_glazing_closes_only_with_faint_line_detection():
    img = solid_building(blank())
    img[Y1 + WALL : Y2 - WALL, X2 - WALL : X2] = 255  # muro derecho -> vidriera
    for x in (X2 - WALL, X2 - 1):
        img[Y1 + WALL : Y2 - WALL, x] = 221  # dos líneas gris claro, como el KUISTI

    without = find_envelope(img)
    with_faint = find_envelope(img, params=replace(EnvelopeParams(), faint_line_contrast=25))

    assert without.outside[275, 400]  # sin detectar líneas tenues, el relleno entra
    assert not with_faint.outside[275, 400]
    assert with_faint.exterior_wall[275, X2 - WALL // 2]  # la vidriera queda resaltada


def test_porch_glazing_between_posts_closes_with_glazing_rule():
    img = solid_building(blank())
    img[Y1 + WALL : Y2 - WALL, X2 - WALL : X2] = 255  # muro derecho -> porche acristalado
    for y in range(Y1 + WALL, Y2 - WALL, 80):
        img[y : y + WALL, X2 - WALL : X2] = BLACK  # postes
    img[Y1 + WALL : Y2 - WALL, X2 - WALL // 2] = BLACK  # vidrio: una línea fina de poste a poste

    without = find_envelope(img)
    with_rule = find_envelope(img, params=replace(EnvelopeParams(), glazing_factor=15))

    assert without.outside[275, 400]  # la línea fina se borra y el relleno entra
    assert not with_rule.outside[275, 400]
    assert with_rule.exterior_wall[Y1 + WALL + 40, X2 - WALL // 2]  # el vidrio entre postes queda resaltado


def test_params_round_trip_through_json(tmp_path):
    params = replace(EnvelopeParams(), faint_line_contrast=25, keep_thin_lines=True, bridge_factor=6.0)

    params.to_json(tmp_path / "p.json")

    assert EnvelopeParams.from_json(tmp_path / "p.json") == params


def test_hint_channels_keep_original_plan_intact():
    img = solid_building(blank())
    result = find_envelope(img)

    stacked = hint_channels(img, result.exterior_wall, np.zeros_like(img, bool))

    assert stacked.shape == (600, 600, 3)
    assert np.array_equal(stacked[:, :, 0], img)
