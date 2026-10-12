from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260926


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("benchmark_degree", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Current-26.06 network-degree logistic baseline on shared folds.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--current-oof", type=Path, required=True)
    parser.add_argument("--benchmark-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    module = load_module(args.benchmark_code)
    degree, names = module.network_degree_features(data)
    # The baseline is deliberately restricted to the six degree/weighted-degree
    # covariates and receives no RWR or expression features.
    source = pd.read_csv(args.current_oof)
    source = source[source["method"] == "RWR"][["repeat", "fold", "node_id", "label"]]
    labels = source.groupby("node_id")["label"].first().sort_index().to_numpy(dtype=int)
    rows = []
    audit = []
    for repeat in range(3):
        for fold in range(5):
            split_number = repeat * 5 + fold
            block = source[(source["repeat"] == repeat) & (source["fold"] == fold)]
            outer_test = np.sort(block["node_id"].to_numpy(dtype=int))
            mask = np.ones(len(labels), dtype=bool)
            mask[outer_test] = False
            outer_train = np.flatnonzero(mask)
            core_train, inner_val = train_test_split(
                outer_train,
                test_size=0.15,
                stratify=labels[outer_train],
                random_state=SEED + split_number,
            )
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=SEED + split_number,
                ),
            )
            model.fit(degree[core_train], labels[core_train])
            score = model.predict_proba(degree[outer_test])[:, 1]
            rows.extend(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "node_id": int(node),
                    "label": int(labels[node]),
                    "method": "Network-degree logistic",
                    "score": float(value),
                }
                for node, value in zip(outer_test, score)
            )
            audit.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "core_train_n": len(core_train),
                    "core_train_positives": int(labels[core_train].sum()),
                    "held_inner_validation_n": len(inner_val),
                    "outer_test_n": len(outer_test),
                    "outer_test_positives": int(labels[outer_test].sum()),
                    "outer_test_labels_used_for_fitting": False,
                }
            )
    pd.DataFrame(rows).to_csv(args.output / "current26_degree_oof_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(audit).to_csv(args.output / "current26_degree_training_audit.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "endpoint": "Open Targets 26.06 disease-specific LUAD clinical-precedence evidence",
        "features": names,
        "model": "class-weighted logistic regression with fold-local standardization",
        "split_policy": "same outer folds and same core/inner partition as the current-endpoint stack and deep models",
        "outer_test_labels_used_for_fitting": False,
    }
    (args.output / "current26_degree_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
