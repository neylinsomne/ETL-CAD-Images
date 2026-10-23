import json

import cv2
import pandas as pd
from conftest import WINDOW, Y2

from filtro_planos.envelope import EnvelopeParams
from filtro_planos.infra.storage import LocalStorage
from filtro_planos.training import dfine_dataset


def test_export_writes_colored_images_and_coco_boxes(tmp_path, monkeypatch, plan_gray):
    for name in ("a", "b"):
        cv2.imwrite(str(tmp_path / f"{name}.png"), plan_gray)
    pd.DataFrame(
        {"image_id": ["cvc_fp/a", "cvc_fp/b"], "image_path": [str(tmp_path / "a.png"), str(tmp_path / "b.png")]}
    ).to_parquet(tmp_path / "images.parquet")
    pd.DataFrame(
        {"image_id": ["cvc_fp/a", "cvc_fp/b"], "cls": ["window", "door"],
         "x1": [WINDOW[0], 300.0], "y1": [Y2 - 12, 200.0], "x2": [WINDOW[1], 330.0], "y2": [Y2, 230.0]}
    ).to_parquet(tmp_path / "elements.parquet")
    splits = pd.DataFrame(
        {"image_id": ["cvc_fp/a", "cvc_fp/b"], "style": ["cvc_fp"] * 2, "split": ["ajuste", "medicion"]}
    )
    monkeypatch.setattr(dfine_dataset, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(dfine_dataset, "load_splits", lambda seed: splits)
    storage = LocalStorage(tmp_path / "store")

    stats = dfine_dataset.export_dfine_dataset(storage, EnvelopeParams(), "v1", workers=1, progress=False)

    assert stats["imagenes"] == {"train": 1, "val": 1}
    assert stats["cajas"] == {"train": 1, "val": 1}
    train = json.loads(storage.get("datasets/dfine/v1/annotations/train.json"))
    assert train["categories"] == dfine_dataset.CATEGORIES
    assert train["annotations"][0]["bbox"] == [WINDOW[0], Y2 - 12, WINDOW[1] - WINDOW[0], 12]
    assert storage.exists(f"datasets/dfine/v1/{train['images'][0]['file_name']}")
