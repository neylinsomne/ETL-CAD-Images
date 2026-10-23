"""Muestra del filtro: corre el paso sobre planos al azar de evaluación y arma un mosaico.

Elige --per-style planos al azar de cada estilo (CubiCasa5K val por estilo de dibujo y CVC-FP),
nunca de entrenamiento, y escribe en --output las salidas del paso y muestra_mosaico.jpg con los
trazos coloreados (rosado muro, azul vidrio). Cada cuadro lleva el F1 de la banda del límite
contra el muro exterior anotado a mano.

Uso: python scripts/make_sample.py [--per-style 2] [--seed 7] [--output data/output/muestra]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from filtro_planos.config import Settings
from filtro_planos.datasets import PROCESSED_DIR, PROJECT_ROOT, load
from filtro_planos.features import envelope_ground_truth
from filtro_planos.metrics import tolerant_prf
from filtro_planos.pipeline import PageResult, load_params, page_outputs, page_record, process_page
from filtro_planos.preprocess import read_image, to_gray, write_image

TILE_W, TILE_H, TITLE_H = 800, 600, 44


def pick(per_style: int, seed: int) -> pd.DataFrame:
    images = pd.read_parquet(PROCESSED_DIR / "images.parquet")
    pool = images[(images["dataset"] == "cvc_fp") | (images["split"] == "val")]
    pool = pool.assign(style=np.where(pool["dataset"] == "cvc_fp", "cvc_fp", pool["subset"]))
    n = min(per_style, int(pool["style"].value_counts().min()))
    return pool.groupby("style").sample(n=n, random_state=seed).reset_index(drop=True)


def sample_f1(dataset: str, image_path: Path, result: PageResult) -> float:
    """F1 de la banda del límite contra el muro exterior anotado (misma holgura que evaluate.py)."""
    ann = load(dataset, image_path.parent if dataset == "cubicasa5k" else svg_for(image_path))
    gt = envelope_ground_truth(ann, result.band.shape, result.scale)
    return tolerant_prf(result.band, gt, 3)[2]


def svg_for(image_path: Path) -> Path:
    return next(image_path.parent.glob(f"{image_path.stem}_gt_*.svg"))


def tile(image: np.ndarray, title: str) -> np.ndarray:
    scale = min(TILE_W / image.shape[1], TILE_H / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((TILE_H + TITLE_H, TILE_W, 3), 255, np.uint8)
    y0, x0 = TITLE_H + (TILE_H - resized.shape[0]) // 2, (TILE_W - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    canvas[:TITLE_H] = (40, 40, 40)
    cv2.putText(canvas, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (TILE_W - 1, TILE_H + TITLE_H - 1), (200, 200, 200), 1)
    return canvas


def before_after(gray: np.ndarray, colored: np.ndarray, title: str, width: int = 1800) -> np.ndarray:
    """Original a la izquierda, trazos coloreados a la derecha, con títulos."""
    left = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    pair = np.hstack([left, np.full((gray.shape[0], max(8, gray.shape[1] // 100), 3), 255, np.uint8), colored])
    scale = width / pair.shape[1]
    pair = cv2.resize(pair, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    bar = np.full((48, pair.shape[1], 3), (40, 40, 40), np.uint8)
    half = pair.shape[1] // 2
    for x, text in ((12, f"ANTES  {title}"), (half + 12, "DESPUES  rosado muro, azul vidrio")):
        cv2.putText(bar, text, (x, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, pair])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-style", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "output" / "muestra")
    args = parser.parse_args()
    (args.output / "antes_despues").mkdir(parents=True, exist_ok=True)

    sample = pick(args.per_style, args.seed)
    params = load_params(Settings.from_env().envelope_params)
    tiles, rows = [], []
    for img in sample.itertuples():
        name = img.image_id.replace("/", "_")
        image_path = PROJECT_ROOT / img.image_path
        gray = to_gray(read_image(image_path))
        result = process_page(gray, params)
        for kind, data in page_outputs(result).items():
            (args.output / f"{name}_{kind}.png").write_bytes(data)
        f1 = sample_f1(img.dataset, image_path, result)
        rows.append({**page_record(result), "image_id": img.image_id, "style": img.style, "f1": f1})
        f1_text = f"F1 {f1:.2f}"
        title = f"{img.style}  {img.image_id.split('/')[-1]}  {f1_text}"
        tiles.append(tile(result.colored, title))
        write_image(args.output / "antes_despues" / f"{name}.jpg", before_after(gray, result.colored, title))
        print(f"{result.status:>12}  {f1_text}  {img.image_id}")

    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    mosaic = np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)])
    write_image(args.output / "muestra_mosaico.jpg", mosaic)
    pd.DataFrame(rows).to_csv(args.output / "muestra.csv", index=False)
    print(f"\n-> {args.output / 'muestra_mosaico.jpg'}")


if __name__ == "__main__":
    main()
