"""Paso del filtro sobre una página: límite exterior y trazos coloreados.

Funciones encadenadas que usan tanto la CLI como el worker:
    load_pages (ingest) -> process_page -> page_outputs / page_record

Uso por CLI (cualquier formato soportado, incluido PDF de varias páginas):
    python -m filtro_planos.pipeline --input data/input --output data/output

Por página escribe <nombre>_p<n>_color.png (rosado muro, azul vidrio), _etiquetas.png
(0 fondo, 1 muro, 2 vidrio, para entrenar) y _hints.png, más una fila en planos.parquet.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from shapely.affinity import scale as scale_geom
from shapely.geometry import MultiPolygon, Polygon

from filtro_planos.config import Settings
from filtro_planos.detector import Detection, Detector, NoDetector, build_detector
from filtro_planos.envelope import EnvelopeParams, find_envelope
from filtro_planos.ingest import load_pages
from filtro_planos.preprocess import downscale
from filtro_planos.render import color_strokes, hint_channels
from filtro_planos.strokes import GLASS, WALL, classify_strokes

INPUT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class PageResult:
    status: str
    labels: np.ndarray  # resolución original: 0 fondo, 1 muro, 2 vidrio
    colored: np.ndarray  # BGR, resolución original
    hints: np.ndarray
    band: np.ndarray  # banda del muro exterior a escala reducida (para evaluar)
    scale: float
    wall_thickness_px: float
    footprint_frac: float
    n_detections: int
    seconds: float
    polygons: list[Polygon] = field(default_factory=list)  # huella, en px originales


def load_params(path: Path) -> EnvelopeParams:
    """Parámetros ajustados si existen (scripts/tune.py); si no, los por defecto."""
    return EnvelopeParams.from_json(path) if path.exists() else EnvelopeParams()


def process_page(gray_full: np.ndarray, params: EnvelopeParams, detector: Detector | None = None) -> PageResult:
    started = time.perf_counter()
    detector = detector or NoDetector()
    gray, scale = downscale(gray_full, params.max_side, params.keep_thin_lines)
    detections = [Detection(d.label, d.score, tuple(v * scale for v in d.box)) for d in detector.detect(gray_full)]
    result = find_envelope(gray, detections, params)
    labels = classify_strokes(gray_full, result, scale, params)
    return PageResult(
        status=result.status,
        labels=labels,
        colored=color_strokes(gray_full, labels),
        hints=hint_channels(gray_full, labels == WALL, labels == GLASS),
        band=result.exterior_wall,
        scale=scale,
        wall_thickness_px=result.wall_thickness_px / scale,
        footprint_frac=result.footprint_frac,
        n_detections=len(detections),
        seconds=round(time.perf_counter() - started, 3),
        polygons=[scale_geom(p, 1 / scale, 1 / scale, origin=(0, 0)) for p in result.polygons],
    )


def page_outputs(result: PageResult) -> dict[str, bytes]:
    """Salidas de la página codificadas en PNG, por nombre."""
    images = {"color": result.colored, "etiquetas": result.labels, "hints": result.hints}
    out = {}
    for name, image in images.items():
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError(f"No se pudo codificar {name}")
        out[name] = encoded.tobytes()
    return out


def page_record(result: PageResult) -> dict:
    """Métricas y geometría de la página, listas para guardar."""
    return {
        "status": result.status,
        "wall_thickness_px": round(result.wall_thickness_px, 2),
        "footprint_frac": round(result.footprint_frac, 4),
        "wall_px": int((result.labels == WALL).sum()),
        "glass_px": int((result.labels == GLASS).sum()),
        "n_buildings": len(result.polygons),
        "footprint_wkt": MultiPolygon(result.polygons).wkt if result.polygons else None,
        "n_detections": result.n_detections,
        "processing_scale": result.scale,
        "seconds": result.seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="carpeta con planos o un archivo")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    settings = Settings.from_env()
    detector = build_detector(settings)
    params = load_params(settings.envelope_params)
    paths = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.rglob("*") if p.suffix.lower() in INPUT_SUFFIXES
    )
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for path in paths:
        # Carpeta por plano (p. ej. CubiCasa: <id>/F1_scaled.png): el nombre sale de la ruta relativa.
        name = path.stem if args.input.is_file() else "_".join(path.relative_to(args.input).with_suffix("").parts)
        try:
            pages = load_pages(path.read_bytes(), settings.pdf_dpi, settings.max_pixels)
        except Exception as exc:  # un archivo roto no detiene el funnel
            rows.append({"file": str(path), "page": None, "status": f"error: {exc!r}"})
            print(f"{rows[-1]['status']:>14}  {path.name}")
            continue
        for page in pages:
            result = process_page(page.gray, params, detector)
            for kind, data in page_outputs(result).items():
                (args.output / f"{name}_p{page.number}_{kind}.png").write_bytes(data)
            rows.append({"file": str(path), "page": page.number, **page_record(result)})
            print(f"{result.status:>14}  {path.name} p{page.number}")

    table = pd.DataFrame(rows)
    table.to_parquet(args.output / "planos.parquet", index=False)
    print(f"\n{len(rows)} páginas -> {args.output / 'planos.parquet'}")
    print(table["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
