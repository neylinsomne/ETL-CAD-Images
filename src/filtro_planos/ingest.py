"""Ingesta: cualquier archivo que llegue se convierte en páginas en escala de grises.

El formato se detecta por el contenido (firma de bytes), no por la extensión. Cada formato
tiene su adecuación:
- PDF: cada página se rasteriza a pdf_dpi (pypdfium2, licencia Apache/BSD; PyMuPDF es AGPL).
- TIFF: puede traer varias páginas; se toman todas.
- JPEG: se respeta la orientación EXIF (fotos o escaneos girados).
- PNG/WebP con transparencia: se compone sobre blanco (si no, el fondo transparente sale negro).
- CMYK, paleta, 16 bits, 1 bit: se normalizan a gris de 8 bits.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, ImageSequence


class UnsupportedFormat(ValueError):
    pass


class ImageTooLarge(ValueError):
    pass


_SIGNATURES = (
    (b"%PDF", "pdf"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
    (b"BM", "bmp"),
)


@dataclass
class Page:
    number: int  # desde 1
    gray: np.ndarray  # uint8, alto x ancho
    dpi: float | None


def sniff_format(data: bytes) -> str:
    head = data[:16]
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    for signature, fmt in _SIGNATURES:
        if head.startswith(signature):
            return fmt
    raise UnsupportedFormat("Formato no soportado: se aceptan PDF, PNG, JPEG, TIFF, BMP y WebP")


def load_pages(data: bytes, pdf_dpi: int = 200, max_pixels: int = 150_000_000) -> list[Page]:
    fmt = sniff_format(data)
    pages = _pdf_pages(data, pdf_dpi) if fmt == "pdf" else _raster_pages(data, max_pixels)
    for page in pages:
        if page.gray.size > max_pixels:
            raise ImageTooLarge(f"Página {page.number}: {page.gray.size} px supera el máximo de {max_pixels}")
    return pages


def _pdf_pages(data: bytes, dpi: int) -> list[Page]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        pages = []
        for i in range(len(pdf)):
            bitmap = pdf[i].render(scale=dpi / 72, grayscale=True)
            pages.append(Page(i + 1, np.array(bitmap.to_pil().convert("L")), float(dpi)))
        return pages
    finally:
        pdf.close()


def _raster_pages(data: bytes, max_pixels: int) -> list[Page]:
    Image.MAX_IMAGE_PIXELS = max_pixels  # protección contra imágenes bomba
    with Image.open(io.BytesIO(data)) as img:
        dpi = img.info.get("dpi", (None,))[0]
        return [
            Page(i + 1, to_gray_array(ImageOps.exif_transpose(frame)), float(dpi) if dpi else None)
            for i, frame in enumerate(ImageSequence.Iterator(img))
        ]


def to_gray_array(img: Image.Image) -> np.ndarray:
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img, dtype=np.float64)
        span = arr.max() - arr.min()
        arr = (arr - arr.min()) / span * 255 if span else np.zeros_like(arr)
        return arr.astype(np.uint8)
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img.convert("RGBA"))
    return np.array(img.convert("L"))
