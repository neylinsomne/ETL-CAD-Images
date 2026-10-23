"""¿Cuánto aporta el contexto? Clasificador muro exterior vs interior sobre elements.parquet.

Compara tres grupos de features con un gradient boosting:
- geometria: solo la forma del muro (largo, grosor relativo, ángulo, rectangularidad)
- contexto:  solo los vecinos (ventanas, puertas, ambientes que toca, distancia a la envolvente)
- ambos

Entrena en CubiCasa5K train; evalúa en CubiCasa5K test y en CVC-FP (otro estilo de dibujo).
Las features salen de la anotación, así que esto mide el techo del método: en producción
vendrían de lo que detecte/segmente el modelo.

Uso: python scripts/context_baseline.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score

from filtro_planos.datasets import PROCESSED_DIR

GEOMETRY = ["thick_rel", "length_rel", "angle", "axis_aligned", "rectangularity", "n_vertices"]
CONTEXT = ["n_windows", "n_doors", "n_rooms_adjacent", "dist_to_hull_norm"]


def load_walls() -> pd.DataFrame:
    elements = pd.read_parquet(PROCESSED_DIR / "elements.parquet")
    images = pd.read_parquet(PROCESSED_DIR / "images.parquet")[["image_id", "split"]]
    walls = elements[elements["cls"] == "wall"].merge(images, on="image_id")
    walls = walls[walls["is_exterior"].notna()].copy()
    # Relativo al plano: la escala en px cambia de un plano a otro.
    walls["thick_rel"] = walls["thickness"] / walls.groupby("image_id")["thickness"].transform("median")
    walls["length_rel"] = walls["length"] / walls.groupby("image_id")["length"].transform("median")
    walls["axis_aligned"] = walls["axis_aligned"].astype(int)
    walls["is_exterior"] = walls["is_exterior"].astype(bool)
    return walls


def main() -> None:
    walls = load_walls()
    train = walls[(walls["dataset"] == "cubicasa5k") & (walls["split"] == "train")]
    evals = {
        "cubicasa5k/test": walls[(walls["dataset"] == "cubicasa5k") & (walls["split"] == "test")],
        "cvc_fp (otro estilo)": walls[walls["dataset"] == "cvc_fp"],
    }
    rows = []
    for name, cols in (("geometria", GEOMETRY), ("contexto", CONTEXT), ("ambos", GEOMETRY + CONTEXT)):
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)
        model.fit(train[cols], train["is_exterior"])
        for split_name, df in evals.items():
            proba = model.predict_proba(df[cols])[:, 1]
            rows.append(
                {
                    "features": name,
                    "evaluacion": split_name,
                    "muros": len(df),
                    "f1": f1_score(df["is_exterior"], proba >= 0.5),
                    "average_precision": average_precision_score(df["is_exterior"], proba),
                }
            )
    table = pd.DataFrame(rows).pivot(index="features", columns="evaluacion", values=["f1", "average_precision"])
    print(table.round(3).to_string())
    out = PROCESSED_DIR / "eval" / "context_baseline.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
