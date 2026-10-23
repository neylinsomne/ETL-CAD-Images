"""Ajuste del filtro con la partición 30/70 balanceada (ver filtro_planos.training.tuning).

Salida: data/processed/tuning/{historial.csv, medicion_70.csv} y config/envelope_params.json.
El mismo ajuste se puede lanzar como servicio: POST /v1/entrenamientos {"tipo": "ajuste_parametros"}.

Uso: python scripts/tune.py [--iterations 24] [--per-style 80] [--seed 0] [--resplit] [--dry-run]
"""

from __future__ import annotations

import argparse
import json

from filtro_planos.config import Settings
from filtro_planos.datasets import PROCESSED_DIR
from filtro_planos.pipeline import load_params
from filtro_planos.training.tuning import SPACE, load_splits, run_tuning

TUNING_DIR = PROCESSED_DIR / "tuning"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=24, help="combinaciones al azar, además de la actual")
    parser.add_argument("--per-style", type=int, default=80, help="planos por estilo en cada iteración")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resplit", action="store_true", help="rehacer la partición 30/70")
    parser.add_argument("--measure-per-style", type=int, default=None, help="limitar la medición (pruebas)")
    parser.add_argument("--dry-run", action="store_true", help="no escribir config/envelope_params.json")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    splits = load_splits(args.seed, args.resplit)
    print(splits.groupby(["style", "split"]).size().unstack().to_string(), "\n")

    settings = Settings.from_env()
    result = run_tuning(
        load_params(settings.envelope_params),
        iterations=args.iterations,
        per_style=args.per_style,
        seed=args.seed,
        measure_per_style=args.measure_per_style,
        workers=args.workers,
    )

    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    result.history.to_csv(TUNING_DIR / "historial.csv", index_label="iteracion")
    result.measurement_scores.to_csv(TUNING_DIR / "medicion_70.csv", index=False)
    print("\nIteraciones (F1 mediano por estilo; 0 = parámetros actuales):")
    print(result.history.drop(columns=list(SPACE)).round(3).to_string())
    print(f"\nMejor: iteración {result.best_iteration} -> {json.dumps({k: getattr(result.best, k) for k in SPACE})}")
    print("\nMedición en el 70 % (no usado para elegir):")
    print(result.measurement.round(3).to_string())

    if not args.dry_run:
        result.best.to_json(settings.envelope_params)
        print(f"\n-> {settings.envelope_params}")


if __name__ == "__main__":
    main()
