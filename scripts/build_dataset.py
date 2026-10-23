"""Construye la base de planos en data/processed/.

- images.parquet     una fila por plano: dataset, split, tamaño, conteos
- elements.parquet   una fila por muro/vano/ambiente: geometría, contexto y si es exterior
- relations.parquet  relaciones anotadas de CVC-FP (incident, neighbour, surround, access, outerP)
- coco/<dataset>_<split>.json  cajas de ventanas y puertas para afinar D-FINE

Uso: python scripts/build_dataset.py [--datasets cubicasa5k cvc_fp] [--workers 8] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from filtro_planos.datasets import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, SOURCES, index, load
from filtro_planos.features import element_rows

COCO_CATEGORIES = [{"id": 1, "name": "window"}, {"id": 2, "name": "door"}]
COCO_IDS = {c["name"]: c["id"] for c in COCO_CATEGORIES}


def process(task: tuple[str, Path, str]) -> dict:
    dataset, path, split = task
    try:
        ann = load(dataset, path)
        with Image.open(ann.image_path) as im:
            width, height = im.size
        rows = element_rows(ann)
    except Exception as exc:  # un SVG roto no debe tumbar toda la corrida
        return {"error": {"dataset": dataset, "path": str(path), "error": repr(exc)}}

    classes = defaultdict(int)
    for el in ann.elements:
        classes[el.cls] += 1
    image_row = {
        "image_id": ann.image_id,
        "dataset": dataset,
        "subset": ann.subset,
        "split": split,
        "image_path": ann.image_path.relative_to(PROJECT_ROOT).as_posix(),
        "width": width,
        "height": height,
        "n_floors": len({el.floor for el in ann.elements}) or 1,
        "n_walls": classes["wall"],
        "n_exterior_walls": sum(1 for el in ann.elements if el.cls == "wall" and el.is_exterior),
        "n_windows": classes["window"],
        "n_doors": classes["door"],
        "n_rooms": classes["room"],
    }
    relations = [
        {"image_id": ann.image_id, "kind": r.kind, "objects": r.objects, "n_objects": len(r.objects)}
        for r in ann.relations
    ]
    return {"image": image_row, "elements": rows, "relations": relations}


def write_coco(images: pd.DataFrame, elements: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    openings = elements[elements["cls"].isin(COCO_IDS)]
    by_image = {k: g for k, g in openings.groupby("image_id")}
    for (dataset, split), group in images.groupby(["dataset", "split"]):
        coco = {"images": [], "annotations": [], "categories": COCO_CATEGORIES}
        for image_num, img in enumerate(group.itertuples(), start=1):
            coco["images"].append(
                {
                    "id": image_num,
                    "file_name": (PROJECT_ROOT / img.image_path).relative_to(RAW_DIR).as_posix(),
                    "width": int(img.width),
                    "height": int(img.height),
                }
            )
            for el in by_image.get(img.image_id, pd.DataFrame()).itertuples():
                coco["annotations"].append(
                    {
                        "id": len(coco["annotations"]) + 1,
                        "image_id": image_num,
                        "category_id": COCO_IDS[el.cls],
                        "bbox": [el.x1, el.y1, el.x2 - el.x1, el.y2 - el.y1],
                        "area": float(el.area),
                        "iscrowd": 0,
                        "is_exterior": None if pd.isna(el.is_exterior) else bool(el.is_exterior),
                    }
                )
        path = out_dir / f"{dataset}_{split}.json"
        path.write_text(json.dumps(coco))
        print(f"  {path.relative_to(PROJECT_ROOT)}: {len(coco['images'])} imágenes, {len(coco['annotations'])} cajas")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--limit", type=int, default=None, help="máx. planos por dataset (pruebas rápidas)")
    args = parser.parse_args()

    tasks = [(ds, path, split) for ds in args.datasets for path, split in index(ds)[: args.limit]]
    images, elements, relations, errors = [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in tqdm(pool.map(process, tasks, chunksize=8), total=len(tasks), desc="planos"):
            if "error" in result:
                errors.append(result["error"])
                continue
            images.append(result["image"])
            elements.extend(result["elements"])
            relations.extend(result["relations"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    images_df = pd.DataFrame(images)
    elements_df = pd.DataFrame(elements)
    images_df.to_parquet(PROCESSED_DIR / "images.parquet", index=False)
    elements_df.to_parquet(PROCESSED_DIR / "elements.parquet", index=False)
    pd.DataFrame(relations).to_parquet(PROCESSED_DIR / "relations.parquet", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(PROCESSED_DIR / "build_errors.csv", index=False)

    print(
        f"\n{len(images_df)} planos, {len(elements_df)} elementos, "
        f"{len(relations)} relaciones, {len(errors)} errores"
    )
    print(images_df.groupby(["dataset", "split"]).size().to_string())
    walls = elements_df[elements_df["cls"] == "wall"]
    print("\nmuros por dataset (exterior / total):")
    print(walls.groupby("dataset")["is_exterior"].agg(["sum", "count"]).to_string())
    write_coco(images_df, elements_df, PROCESSED_DIR / "coco")


if __name__ == "__main__":
    main()
