import io

import numpy as np
import pytest
from conftest import encode
from PIL import Image

from filtro_planos.ingest import ImageTooLarge, UnsupportedFormat, load_pages, sniff_format


def test_format_is_detected_by_content_not_extension(plan_gray):
    assert sniff_format(encode(plan_gray, "PNG")) == "png"
    assert sniff_format(encode(plan_gray, "JPEG")) == "jpeg"
    assert sniff_format(encode(plan_gray, "TIFF")) == "tiff"
    assert sniff_format(encode(plan_gray, "BMP")) == "bmp"
    assert sniff_format(encode(plan_gray, "WEBP")) == "webp"
    assert sniff_format(encode(plan_gray, "PDF")) == "pdf"


def test_unsupported_bytes_are_rejected():
    with pytest.raises(UnsupportedFormat):
        load_pages(b"esto no es un plano")


def test_pdf_pages_are_rasterized_at_requested_dpi(plan_gray):
    small = plan_gray[::6, ::6]  # 100 x 100 px -> página de 100 x 100 pt (PIL guarda a 72 dpi)
    pages = load_pages(encode([small, small], "PDF"), pdf_dpi=144)

    assert [p.number for p in pages] == [1, 2]
    assert pages[0].gray.shape == (200, 200)
    assert pages[0].dpi == 144


def test_multipage_tiff_yields_every_page(plan_gray):
    pages = load_pages(encode([plan_gray, 255 - plan_gray], "TIFF"))

    assert len(pages) == 2
    assert pages[1].gray.mean() < pages[0].gray.mean()


def test_jpeg_exif_rotation_is_applied():
    img = Image.fromarray(np.full((20, 40), 255, np.uint8))  # 40 de ancho, 20 de alto
    exif = img.getexif()
    exif[0x0112] = 6  # girada 90 grados
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)

    (page,) = load_pages(buf.getvalue())

    assert page.gray.shape == (40, 20)


def test_transparent_background_becomes_white_not_black():
    rgba = np.zeros((30, 30, 4), np.uint8)  # todo transparente
    rgba[10:20, 10:20] = (0, 0, 0, 255)  # un cuadro negro opaco
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")

    (page,) = load_pages(buf.getvalue())

    assert page.gray[0, 0] == 255
    assert page.gray[15, 15] == 0


def test_16_bit_scan_is_normalized_to_8_bit():
    arr = np.linspace(0, 65535, 64 * 64).reshape(64, 64).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")

    (page,) = load_pages(buf.getvalue())

    assert page.gray.dtype == np.uint8
    assert page.gray.min() == 0 and page.gray.max() == 255


def test_oversized_image_is_rejected(plan_gray):
    with pytest.raises((ImageTooLarge, Image.DecompressionBombError, Image.DecompressionBombWarning)):
        load_pages(encode(plan_gray, "PNG"), max_pixels=10_000)
