"""Evalúa el paso del muro exterior contra la anotación manual.

Dos modos por plano:
- none:   solo el paso clásico (sin detector)
- oracle: con las ventanas/puertas anotadas como si fueran detecciones perfectas de D-FINE
          (techo de lo que aporta el detector)

Usa los parámetros de config/envelope_params.json (o ENVELOPE_PARAMS) si existen.
Salida: data/processed/eval/<dataset>_<split>.parquet y comparaciones en eval/vis/.

Uso: python scripts/evaluate.py --dataset cubicasa5k --split val [--limit 100]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from filtro_planos.config import Settings
from filtro_planos.datasets import PROCESSED_DIR, SOURCES, index, load
from filtro_planos.detector import Detection
from filtro_planos.envelope import EnvelopeParams, find_envelope, main_components, outside_region
from filtro_planos.features import envelope_ground_truth
from filtro_planos.metrics import comparison_image, iou, tolerant_prf
from filtro_planos.pipeline import load_params
from filtro_planos.preprocess import downscale, read_image, to_gray, write_image

OUT_DIR = PROCESSED_DIR / "eval"


def oracle_detections(ann, scale: float) -> list[Detection]:
    dets = []
    for el in ann.elements:
        if el.cls in ("window", "door"):
            x1, y1, x2, y2 = (v * scale for v in el.polygon.bounds)
            dets.append(Detection(el.cls, 1.0, (x1, y1, x2, y2)))
    return dets


def evaluate_one(
    task: tuple[str, Path, str], params: EnvelopeParams, tolerance_px: int, vis_dir: Path | None
) -> list[dict]:
    dataset, path, split = task
    ann = load(dataset, path)
    gray, scale = downscale(to_gray(read_image(ann.image_path)), params.max_side, params.keep_thin_lines)
    gt = envelope_ground_truth(ann, gray.shape, scale)
    gt_footprint = main_components(~outside_region(gt), 0.1)

    rows = []
    for mode, dets in (("none", []), ("oracle", oracle_detections(ann, scale))):
        result = find_envelope(gray, dets, params)
        precision, recall, f1 = tolerant_prf(result.exterior_wall, gt, tolerance_px)
        rows.append(
            {
                "image_id": ann.image_id,
                "dataset": dataset,
                "split": split,
                "mode": mode,
                "status": result.status,
                "wall_thickness_px": result.wall_thickness_px,
                "footprint_frac": result.footprint_frac,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou(result.exterior_wall, gt),
                "footprint_iou": iou(result.footprint, gt_footprint),
                "scale": scale,
            }
        )
        if vis_dir is not None:
            name = ann.image_id.replace("/", "_")
            write_image(vis_dir / f"{name}_{mode}.jpg", comparison_image(gray, result.exterior_wall, gt))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="cubicasa5k", choices=list(SOURCES))
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tolerance", type=int, default=3, help="holgura en px para precisión/recall")
    parser.add_argument("--vis", type=int, default=12, help="cuántos planos guardar como imagen")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    tasks = [(args.dataset, p, s) for p, s in index(args.dataset) if s == args.split][: args.limit]
    params = load_params(Settings.from_env().envelope_params)
    vis_dir = OUT_DIR / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(evaluate_one, task, params, args.tolerance, vis_dir if i < args.vis else None)
            for i, task in enumerate(tasks)
        ]
        for fut in tqdm(futures, desc=f"{args.dataset}/{args.split}"):
            rows.extend(fut.result())

    df = pd.DataFrame(rows)
    out = OUT_DIR / f"{args.dataset}_{args.split}.parquet"
    df.to_parquet(out, index=False)
    summary = df.groupby("mode").agg(
        planos=("image_id", "size"),
        status_ok=("status", lambda s: (s == "ok").mean()),
        precision=("precision", "median"),
        recall=("recall", "median"),
        f1_mediana=("f1", "median"),
        f1_media=("f1", "mean"),
        f1_mayor_0_8=("f1", lambda s: (s > 0.8).mean()),
        footprint_iou=("footprint_iou", "median"),
    )
    print(summary.round(3).to_string())
    if args.dataset == "cubicasa5k":  # colorful / high_quality / high_quality_architectural
        estilo = df["image_id"].str.split("/").str[1]
        by_style = df.assign(estilo=estilo).groupby(["estilo", "mode"])["f1"].median().unstack()
        print("\nF1 mediana por estilo de dibujo:")
        print(by_style.round(3).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
