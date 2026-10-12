from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FIT_KEY = ["repeat", "fold", "method", "seed_label"]
PREDICTION_KEY = FIT_KEY + ["node_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge non-overlapping/resumed uniform deep-model shards into one target directory.")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-fits", type=int, default=45)
    args = parser.parse_args()
    directories = [args.target, *args.sources]
    audits = pd.concat([pd.read_csv(path / "additional_seed_training_audit.csv") for path in directories], ignore_index=True)
    predictions = pd.concat([pd.read_csv(path / "additional_seed_oof_predictions.csv.gz") for path in directories], ignore_index=True)

    duplicate_mask = predictions.duplicated(PREDICTION_KEY, keep=False)
    if duplicate_mask.any():
        duplicate_scores = predictions.loc[duplicate_mask].groupby(PREDICTION_KEY)["score"].agg(
            lambda values: float(np.ptp(values.to_numpy(dtype=float)))
        )
        if duplicate_scores.max() > 1e-7:
            raise ValueError(f"Overlapping shards contain inconsistent predictions (maximum score range {duplicate_scores.max():.3g}).")
    audits = audits.sort_values(FIT_KEY).drop_duplicates(FIT_KEY, keep="first")
    predictions = predictions.sort_values(PREDICTION_KEY).drop_duplicates(PREDICTION_KEY, keep="first")
    if len(audits) != args.expected_fits:
        raise ValueError(f"Expected {args.expected_fits} unique fits after merging, observed {len(audits)}.")
    observed_fits = predictions[FIT_KEY].drop_duplicates()
    if len(observed_fits) != args.expected_fits:
        raise ValueError(f"Predictions cover {len(observed_fits)} fits, expected {args.expected_fits}.")
    if audits[FIT_KEY].duplicated().any() or predictions[PREDICTION_KEY].duplicated().any():
        raise ValueError("Duplicate identifiers remain after shard merging.")

    args.target.mkdir(parents=True, exist_ok=True)
    audits.to_csv(args.target / "additional_seed_training_audit.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(args.target / "additional_seed_oof_predictions.csv.gz", index=False, compression="gzip")
    print(f"Merged {len(audits)} unique fits and {len(predictions)} prediction rows into {args.target}.")


if __name__ == "__main__":
    main()
