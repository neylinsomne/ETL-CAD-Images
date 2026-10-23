"""Ajuste del filtro con partición balanceada: 30 % para iterar parámetros, 70 % para medir.

1. Partición (data/processed/splits.parquet): todos los planos anotados (el filtro no aprende,
   así que también entran los de train de CubiCasa5K), estratificados por estilo de dibujo
   (colorful / high_quality / high_quality_architectural / cvc_fp): 30 % "ajuste", 70 % "medicion".
   Se crea una vez y se reutiliza, para que las corridas sean comparables.
2. Iteraciones sobre el 30 %: la 0 son los parámetros actuales; las demás, combinaciones al azar
   del espacio de búsqueda. Objetivo balanceado: promedio del F1 mediano de cada estilo.
   Cada iteración usa una muestra balanceada del 30 % (per_style planos por estilo).
3. Medición en el 70 %: parámetros actuales vs. los mejores.
"""

from __future__ import annotations

import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from filtro_planos.datasets import PROCESSED_DIR, SOURCES, image_id, index, load, style
from filtro_planos.envelope import EnvelopeParams, find_envelope
from filtro_planos.features import envelope_ground_truth
from filtro_planos.metrics import tolerant_prf
from filtro_planos.preprocess import downscale, read_image, to_gray

SPLITS = PROCESSED_DIR / "splits.parquet"
TUNE_FRAC = 0.3
TOLERANCE_PX = 3

SPACE: dict[str, list[Any]] = {
    "keep_thin_lines": [False, True],
    "faint_line_contrast": [0, 20, 30, 45],
    "max_ink_level": [140, 160, 180],
    "cavity_radius_frac": [0.006, 0.01, 0.014],
    "thin_stroke_factor": [0.35, 0.5, 0.65],
    "bridge_factor": [6.0, 8.0, 10.0],
    "glazing_factor": [0.0, 15.0, 30.0],
    "gap_factor": [1.5, 2.0, 3.0],
    "depth_factor": [1.25, 1.5, 2.0],
}


@dataclass
class TuningResult:
    best: EnvelopeParams
    best_iteration: int
    history: pd.DataFrame  # una fila por iteración: F1 por estilo, balanceado, global y parámetros
    measurement: pd.DataFrame  # actuales vs mejores en el 70 %, por estilo
    measurement_scores: pd.DataFrame  # una fila por plano del 70 % y configuración

    def metrics(self) -> dict:
        """Resumen para guardar junto a la versión de parámetros."""
        m = self.measurement.round(4)
        return {
            "mejor_iteracion": self.best_iteration,
            "iteraciones": len(self.history) - 1,
            "f1_balanceado_30": round(float(self.history.loc[self.best_iteration, "f1_balanceado"]), 4),
            "medicion_70": {cfg: {k: float(v) for k, v in row.items()} for cfg, row in m.iterrows()},
        }


def make_splits(seed: int = 0) -> pd.DataFrame:
    rows = [
        {"image_id": image_id(ds, path), "dataset": ds, "path": str(path)}
        for ds in SOURCES
        for path, _ in index(ds)
    ]
    df = pd.DataFrame(rows)
    df["style"] = df["image_id"].map(style)
    parts = []
    for _, group in df.groupby("style"):
        group = group.sample(frac=1.0, random_state=seed)
        n_tune = math.ceil(TUNE_FRAC * len(group))
        parts.append(group.assign(split=["ajuste"] * n_tune + ["medicion"] * (len(group) - n_tune)))
    return pd.concat(parts, ignore_index=True)


def load_splits(seed: int = 0, resplit: bool = False) -> pd.DataFrame:
    if resplit or not SPLITS.exists():
        SPLITS.parent.mkdir(parents=True, exist_ok=True)
        make_splits(seed).to_parquet(SPLITS, index=False)
    return pd.read_parquet(SPLITS)


def candidate_configs(base: EnvelopeParams, iterations: int, seed: int) -> list[EnvelopeParams]:
    rng = random.Random(seed)
    configs, seen = [base], {json.dumps(asdict(base), sort_keys=True)}
    while len(configs) < iterations + 1:
        params = replace(base, **{k: rng.choice(v) for k, v in SPACE.items()})
        key = json.dumps(asdict(params), sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(params)
    return configs


def score_plan(task: tuple[str, str], configs: list[EnvelopeParams]) -> list[dict]:
    """Corre todas las configuraciones sobre un plano (se lee y reduce una sola vez)."""
    dataset, path = task
    ann = load(dataset, Path(path))
    full = to_gray(read_image(ann.image_path))
    reduced = {}
    rows = []
    for i, params in enumerate(configs):
        key = (params.max_side, params.keep_thin_lines)
        if key not in reduced:
            gray, scale = downscale(full, *key)
            reduced[key] = (gray, envelope_ground_truth(ann, gray.shape, scale))
        gray, gt = reduced[key]
        result = find_envelope(gray, (), params)
        precision, recall, f1 = tolerant_prf(result.exterior_wall, gt, TOLERANCE_PX)
        rows.append(
            {
                "iteration": i,
                "image_id": ann.image_id,
                "style": style(ann.image_id),
                "status": result.status,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def score_many(
    tasks: list[tuple[str, str]], configs: list[EnvelopeParams], workers: int, desc: str, progress: bool = True
) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(score_plan, task, configs) for task in tasks]
        for fut in tqdm(futures, desc=desc, disable=not progress):
            rows.extend(fut.result())
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """F1 mediano por estilo, promedio balanceado (cada estilo pesa igual) y F1 mediano global."""
    by_style = scores.pivot_table(index="iteration", columns="style", values="f1", aggfunc="median")
    by_style["f1_balanceado"] = by_style.mean(axis=1)
    by_style["f1_global"] = scores.groupby("iteration")["f1"].median()
    return by_style


def run_tuning(
    current: EnvelopeParams,
    iterations: int = 24,
    per_style: int = 80,
    seed: int = 0,
    measure_per_style: int | None = None,
    workers: int | None = None,
    resplit: bool = False,
    progress: bool = True,
) -> TuningResult:
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    splits = load_splits(seed, resplit)
    configs = candidate_configs(current, iterations, seed)

    tune = splits[splits["split"] == "ajuste"].sample(frac=1.0, random_state=seed).groupby("style").head(per_style)
    tune_tasks = list(zip(tune["dataset"], tune["path"], strict=True))
    history = summarize(score_many(tune_tasks, configs, workers, "ajuste (30 %)", progress))
    history["mejor_hasta_aqui"] = history["f1_balanceado"].cummax()
    history = history.join(pd.DataFrame([{k: getattr(c, k) for k in SPACE} for c in configs]))
    best_i = int(history["f1_balanceado"].idxmax())

    measure = splits[splits["split"] == "medicion"]
    if measure_per_style:
        measure = measure.sample(frac=1.0, random_state=seed).groupby("style").head(measure_per_style)
    measure_tasks = list(zip(measure["dataset"], measure["path"], strict=True))
    final = score_many(measure_tasks, [current, configs[best_i]], workers, "medición (70 %)", progress)
    final["config"] = final["iteration"].map({0: "actuales", 1: "mejores"})
    measurement = summarize(final).rename(index={0: "actuales", 1: "mejores"})
    measurement["planos"] = final.groupby("iteration")["image_id"].size().values
    return TuningResult(configs[best_i], best_i, history, measurement, final)
