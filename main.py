from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src.convert import build_dataset_splits
from src.evaluate import run_full_evaluation
from src.features import build_all_features
from src.train_cnn import train_cnn
from src.train_ml import train_all_ml
from src.visualize import generate_all_plots


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Malware detection full pipeline runner")
    parser.add_argument("--skip-convert", action="store_true", help="Skip CSV split generation")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature extraction/cache build")
    parser.add_argument("--skip-cnn", action="store_true", help="Skip CNN training")
    parser.add_argument("--skip-ml", action="store_true", help="Skip traditional ML training")
    parser.add_argument("--skip-plots", action="store_true", help="Skip visualization generation")
    return parser


def _read_best_model_from_results(results_csv: Path):
    if not results_csv.exists():
        return None, None

    df = pd.read_csv(results_csv)
    if df.empty or "Test Accuracy" not in df.columns:
        return None, None

    best_idx = df["Test Accuracy"].astype(float).idxmax()
    row = df.loc[best_idx]
    return str(row["Model"]), float(row["Test Accuracy"])


def main() -> None:
    start_time = time.time()
    args = _build_arg_parser().parse_args()

    try:
        if not args.skip_convert:
            build_dataset_splits("data/raw", "data/splits")
            print("[1/5] Dataset splits created.")
        else:
            print("[1/5] Dataset split generation skipped.")

        if not args.skip_features:
            build_all_features("data/splits")
            print("[0/5] Feature caches prepared.")
        else:
            print("[0/5] Feature extraction skipped.")

        if not args.skip_cnn:
            train_cnn()
            print("[2/5] CNN training complete.")
        else:
            print("[2/5] CNN training skipped.")

        if not args.skip_ml:
            train_all_ml()
            print("[3/5] ML models trained.")
        else:
            print("[3/5] ML model training skipped.")

        eval_results = run_full_evaluation()
        print("[4/5] Evaluation complete.")

        if not args.skip_plots:
            generate_all_plots(
                results_dict=eval_results.get("results", {}) if isinstance(eval_results, dict) else {},
                splits_dir="data/splits",
                log_path="outputs/models/cnn_training_log.json",
                raw_dir="data/raw",
            )
            print("[5/5] Plots generated.")
        else:
            print("[5/5] Plot generation skipped.")

        best_model, best_acc = _read_best_model_from_results(Path("outputs/results.csv"))

        print("=" * 48)
        print("RESULTS SUMMARY")
        if best_model is not None and best_acc is not None:
            print(f"Best model: {best_model} with test accuracy {best_acc:.4f}")
        else:
            print("Best model: N/A (run full training + evaluation first)")
        print("Full results: outputs/results.csv")
        print("Plots: outputs/plots/")
        print("=" * 48)

        elapsed = time.time() - start_time
        print(f"Total elapsed time: {elapsed:.2f} seconds")

    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
