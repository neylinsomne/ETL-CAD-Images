"""Dataset para afinar D-FINE con planos coloreados por el filtro.

Toma los planos anotados (CubiCasa5K y CVC-FP), los pasa por el filtro con los parámetros
activos (las mismas imágenes coloreadas que D-FINE verá en uso) y escribe un COCO con las cajas
anotadas de ventanas y puertas. La partición sigue la del ajuste: 30 % "ajuste" = train,
70 % "medicion" = val.

Claves en el almacenamiento:
    datasets/dfine/<version>/images/<train|val>/<plano>.png
    datasets/dfine/<version>/annotations/<train|val>.json
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import pandas as pd
from tqdm import tqdm

from filtro_planos.datasets import PROCESSED_DIR, PROJECT_ROOT
from filtro_planos.envelope import EnvelopeParams
from filtro_planos.infra.storage import Storage
from filtro_planos.pipeline import process_page
from filtro_planos.preprocess import read_image, to_gray
from filtro_planos.training.tuning import load_splits

CATEGORIES = [{"id": 1, "name": "window"}, {"id": 2, "name": "door"}]
CATEGORY_IDS = {c["name"]: c["id"] for c in CATEGORIES}
SPLIT_NAMES = {"ajuste": "train", "medicion": "val"}


def _color_plan(image_path: str, params: EnvelopeParams) -> tuple[bytes, int, int, str]:
    gray = to_gray(read_image(PROJECT_ROOT / image_path))
    result = process_page(gray, params)
    ok, encoded = cv2.imencode(".png", result.colored)
    if not ok:
        raise ValueError(f"No se pudo codificar {image_path}")
    return encoded.tobytes(), gray.shape[1], gray.shape[0], result.status


def export_dfine_dataset(
    storage: Storage,
    params: EnvelopeParams,
    version: str,
    per_style: int | None = None,
    seed: int = 0,
    workers: int | None = None,
    progress: bool = True,
) -> dict:
    """Escribe imágenes coloreadas y anotaciones COCO; devuelve conteos y el prefijo de claves."""
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    images = pd.read_parquet(PROCESSED_DIR / "images.parquet")[["image_id", "image_path"]]
    plans = load_splits(seed).merge(images, on="image_id")
    if per_style:
        plans = plans.sample(frac=1.0, random_state=seed).groupby(["split", "style"]).head(per_style)
    elements = pd.read_parquet(PROCESSED_DIR / "elements.parquet", columns=["image_id", "cls", "x1", "y1", "x2", "y2"])
    boxes = {k: g for k, g in elements[elements["cls"].isin(CATEGORY_IDS)].groupby("image_id")}

    prefix = f"datasets/dfine/{version}"
    coco = {name: {"images": [], "annotations": [], "categories": CATEGORIES} for name in SPLIT_NAMES.values()}
    statuses: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_color_plan, row.image_path, params): row for row in plans.itertuples()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="dataset D-FINE", disable=not progress):
            row = futures[fut]
            png, width, height, status = fut.result()
            statuses[status] = statuses.get(status, 0) + 1
            split = SPLIT_NAMES[row.split]
            file_name = f"images/{split}/{row.image_id.replace('/', '_')}.png"
            storage.put(f"{prefix}/{file_name}", png, "image/png")
            data = coco[split]
            image_num = len(data["images"]) + 1
            data["images"].append({"id": image_num, "file_name": file_name, "width": width, "height": height})
            for box in boxes.get(row.image_id, pd.DataFrame()).itertuples():
                data["annotations"].append(
                    {
                        "id": len(data["annotations"]) + 1,
                        "image_id": image_num,
                        "category_id": CATEGORY_IDS[box.cls],
                        "bbox": [box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1],
                        "area": float((box.x2 - box.x1) * (box.y2 - box.y1)),
                        "iscrowd": 0,
                    }
                )

    for split, data in coco.items():
        storage.put(f"{prefix}/annotations/{split}.json", json.dumps(data).encode(), "application/json")
    return {
        "prefijo": prefix,
        "imagenes": {split: len(d["images"]) for split, d in coco.items()},
        "cajas": {split: len(d["annotations"]) for split, d in coco.items()},
        "estados_filtro": statuses,
    }
