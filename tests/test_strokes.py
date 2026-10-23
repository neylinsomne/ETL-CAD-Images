import cv2
from conftest import WALL, WINDOW, X1, Y1, Y2, synthetic_plan

from filtro_planos.envelope import EnvelopeParams
from filtro_planos.pipeline import page_outputs, page_record, process_page
from filtro_planos.render import BLUE_BGR, PINK_BGR
from filtro_planos.strokes import GLASS
from filtro_planos.strokes import WALL as WALL_LABEL


def test_boundary_strokes_get_two_classes_and_interior_stays_uncolored(plan_gray):
    result = process_page(plan_gray, EnvelopeParams())

    assert result.labels[Y1 + WALL // 2, 250] == WALL_LABEL  # muro exterior
    assert result.labels[Y2 - WALL // 2, (WINDOW[0] + WINDOW[1]) // 2] == GLASS  # línea de vidrio
    assert result.labels[275, 299] == 0  # muro interior, lejos del límite
    assert result.labels[275, 200] == 0  # interior vacío


def test_window_drawn_in_black_is_glass_by_low_ink_density():
    """Estilo CVC-FP: muro macizo negro y ventana como rectángulo de líneas negras finas."""
    img = synthetic_plan()
    x0, x1 = 360, 440
    img[Y1 : Y1 + WALL, x0:x1] = 255  # vano en el muro superior
    cv2.rectangle(img, (x0, Y1), (x1 - 1, Y1 + WALL - 1), 0, 1)  # marco de la ventana
    img[Y1 + WALL // 2, x0:x1] = 0  # vidrio, también negro

    result = process_page(img, EnvelopeParams())

    assert result.labels[Y1 + WALL // 2, 400] == GLASS
    assert result.labels[Y1 + WALL // 2, 250] == WALL_LABEL  # el muro macizo sigue rosado


def test_only_stroke_pixels_are_painted_so_line_width_is_kept(plan_gray):
    result = process_page(plan_gray, EnvelopeParams())
    y, x = Y2 - WALL // 2, (WINDOW[0] + WINDOW[1]) // 2

    assert tuple(result.colored[y, x]) == BLUE_BGR
    assert tuple(result.colored[y - 2, x]) == (255, 255, 255)  # el blanco junto al vidrio sigue blanco
    assert tuple(result.colored[Y1 + WALL // 2, X1 + 50]) == PINK_BGR


def test_outputs_and_record_are_ready_to_store(plan_gray):
    result = process_page(plan_gray, EnvelopeParams())

    outputs = page_outputs(result)
    record = page_record(result)

    assert set(outputs) == {"color", "etiquetas", "hints"}
    assert all(data.startswith(b"\x89PNG") for data in outputs.values())
    assert record["status"] == "ok" and record["wall_px"] > 0 and record["glass_px"] > 0
