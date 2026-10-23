import io

import cv2
import numpy as np
import pytest
from PIL import Image

WALL = 12
X1, Y1, X2, Y2 = 100, 100, 500, 450
WINDOW = (150, 210)  # vano en el muro inferior, con vidrio de línea fina gris claro
GLASS_GRAY = 200


def synthetic_plan() -> np.ndarray:
    img = np.full((600, 600), 255, np.uint8)
    cv2.rectangle(img, (X1, Y1), (X2 - 1, Y2 - 1), 0, WALL * 2)  # muros exteriores
    img[Y1 + WALL : Y2 - WALL, X1 + WALL : X2 - WALL] = 255
    cv2.rectangle(img, (297, Y1), (302, Y2 - 1), 0, -1)  # muro interior
    img[Y2 - WALL : Y2, WINDOW[0] : WINDOW[1]] = 255
    img[Y2 - WALL // 2, WINDOW[0] : WINDOW[1]] = GLASS_GRAY  # vidrio
    return img


@pytest.fixture
def plan_gray() -> np.ndarray:
    return synthetic_plan()


def encode(image: np.ndarray | list[np.ndarray], fmt: str) -> bytes:
    """Codifica una o varias páginas (TIFF y PDF admiten varias)."""
    pages = image if isinstance(image, list) else [image]
    pil = [Image.fromarray(p) for p in pages]
    buf = io.BytesIO()
    pil[0].save(buf, format=fmt, save_all=len(pil) > 1, append_images=pil[1:])
    return buf.getvalue()


@pytest.fixture
def plan_png(plan_gray) -> bytes:
    return encode(plan_gray, "PNG")
