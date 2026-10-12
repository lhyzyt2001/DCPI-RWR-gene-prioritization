from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 20260926
INNER_VAL_FRACTION = 0.15
DISPLAY = {"appnp": "APPNP", "gcnii": "GCNII", "gatv2": "GATv2"}
BASE_OFFSET = {"appnp": 1000, "gcnii": 2000, "gatv2": 3000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uniform optimization-seed sensitivity on the disease-specific Open Targets 26.06 endpoint."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--current-oof", type=Path, required=True)
    parser.add_argument("--strong-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=list(DISPLAY), required=True)
    parser.add_argument("--seed-offsets", nargs="+", type=int, default=[0, 10000, 20000])
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--betweenness-samples", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("strong_gnn_current26", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_labels(data, mapping_path: Path, label_path: Path) -> tuple[np.ndarray, int]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    symbol_to_node = {str(symbol).upper(): int(node) for symbol, node in mapping.items()}
    positive_symbols = set(pd.read_csv(label_path)["symbol"].astype(str).str.upper())
    positive_nodes = sorted(symbol_to_node[symbol] for symbol in positive_symbols if symbol in symbol_to_node)
    labels = np.zeros(data.num_nodes, dtype=np.int64)
    labels[positive_nodes] = 1
    if len(positive_nodes) != len(positive_symbols):
        raise RuntimeError(
            f"Mapped {len(positive_nodes)} of {len(positive_symbols)} positive symbols; endpoint universe changed."
        )
    return labels, len(positive_nodes)


def save_partial(audit_rows: list[dict], prediction_rows: list[dict], output: Path) -> None:
    pd.DataFrame(audit_rows).to_csv(output / "current26_deep_training_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(
        output / "current26_deep_oof_predictions.csv.gz", index=False, compression="gzip"
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))
    module = load_module(args.strong_code)
    started = time.time()

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    labels, positive_count = build_labels(data, args.mapping, args.labels)
    y_tensor = torch.as_tensor(labels, dtype=torch.long)
    raw_expression = data.x.cpu().numpy().astype(np.float32)
    topology, _ = module.prepare_static_features(data, args.betweenness_samples)
    transition = module.prepare_transition(data)
    edge_index = data.edge_index_ppi.cpu().long().contiguous()
    edge_weight = data.edge_weight_ppi.cpu().float().contiguous()

    classical = pd.read_csv(args.current_oof)
    split_source = classical[classical["method"] == "RWR"][["repeat", "fold", "node_id", "label"]].copy()
    if len(split_source) != len(labels) * 3:
        raise RuntimeError("Current-endpoint RWR OOF must contain one row per gene and repeat.")
    observed = split_source.groupby("node_id")["label"].first().sort_index().to_numpy(dtype=int)
    if not np.array_equal(observed, labels):
        raise RuntimeError("The supplied 26.06 labels do not match the current-endpoint OOF labels.")

    audit_rows: list[dict] = []
    prediction_rows: list[dict] = []
    completed_keys: set[tuple[int, int, str, str]] = set()
    if args.resume:
        audit_path = args.output / "current26_deep_training_audit.csv"
        prediction_path = args.output / "current26_deep_oof_predictions.csv.gz"
        if audit_path.exists() and prediction_path.exists():
            previous_audit = pd.read_csv(audit_path)
            previous_predictions = pd.read_csv(prediction_path)
            audit_rows = previous_audit.to_dict("records")
            prediction_rows = previous_predictions.to_dict("records")
            completed_keys = {
                (int(row.repeat), int(row.fold), str(row.method), str(row.seed_label))
                for row in previous_audit.itertuples(index=False)
            }
            print(f"Resuming from {len(completed_keys)} completed fits", flush=True)

    total = 3 * 5 * len(args.models) * len(args.seed_offsets)
    completed = len(completed_keys)
    for repeat in range(3):
        for fold in range(5):
            split_number = repeat * 5 + fold
            split_seed = SEED + split_number
            block = split_source[(split_source["repeat"] == repeat) & (split_source["fold"] == fold)]
            outer_test = np.sort(block["node_id"].to_numpy(dtype=int))
            outer_mask = np.ones(len(labels), dtype=bool)
            outer_mask[outer_test] = False
            outer_train = np.flatnonzero(outer_mask)
            core_train, inner_val = train_test_split(
                outer_train,
                test_size=INNER_VAL_FRACTION,
                stratify=labels[outer_train],
                random_state=split_seed,
            )

            expression_scaler = StandardScaler().fit(raw_expression[core_train])
            expression_scaled = expression_scaler.transform(raw_expression)
            pca = PCA(n_components=5, random_state=SEED).fit(expression_scaled[core_train])
            feature_all = np.column_stack(
                [topology, expression_scaled, pca.transform(expression_scaled)]
            ).astype(np.float32)
            core_rwr_oof = module.crossfit_rwr(transition, labels, core_train, split_seed)
            seed_nodes = core_train[labels[core_train] == 1]
            all_rwr = module.run_rwr(transition, len(labels), seed_nodes)
            calibrator = module.fit_rwr_calibrator(core_rwr_oof, labels[core_train])
            rwr_logit = np.zeros(len(labels), dtype=np.float32)
            rwr_logit[core_train] = module.calibrated_logit(calibrator, core_rwr_oof)
            rwr_logit[inner_val] = module.calibrated_logit(calibrator, all_rwr[inner_val])
            rwr_logit[outer_test] = module.calibrated_logit(calibrator, all_rwr[outer_test])
            direct = np.column_stack([rwr_logit, feature_all]).astype(np.float32)
            neural_scaler = StandardScaler().fit(direct[core_train])
            direct_scaled = np.nan_to_num(
                neural_scaler.transform(direct), nan=0.0, posinf=10.0, neginf=-10.0
            )
            x_tensor = torch.as_tensor(direct_scaled, dtype=torch.float32)

            for model_name in args.models:
                method = DISPLAY[model_name]
                for seed_index, seed_offset in enumerate(args.seed_offsets, start=1):
                    seed_label = f"uniform_seed_{seed_index}"
                    fit_key = (repeat, fold, method, seed_label)
                    if fit_key in completed_keys:
                        continue
                    seed = split_seed + BASE_OFFSET[model_name] + seed_offset
                    completed += 1
                    print(
                        f"current26 fit {completed}/{total} | repeat {repeat + 1}/3 | "
                        f"fold {fold + 1}/5 | {model_name} | seed {seed_index}",
                        flush=True,
                    )
                    _, test_score, fit_audit = module.train_model(
                        model_name,
                        x_tensor,
                        y_tensor,
                        edge_index,
                        edge_weight,
                        core_train,
                        inner_val,
                        outer_test,
                        args.max_epochs,
                        args.min_epochs,
                        args.patience,
                        seed,
                    )
                    audit_rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "method": method,
                            "seed_label": seed_label,
                            "seed": seed,
                            "endpoint_release": "26.06",
                            "endpoint_positive_count": positive_count,
                            "core_train_n": len(core_train),
                            "core_train_positives": int(labels[core_train].sum()),
                            "inner_val_n": len(inner_val),
                            "inner_val_positives": int(labels[inner_val].sum()),
                            "outer_test_n": len(outer_test),
                            "outer_test_positives": int(labels[outer_test].sum()),
                            "outer_test_labels_used_for_training": False,
                            "outer_test_nodes_used_as_rwr_seeds": False,
                            **fit_audit,
                        }
                    )
                    prediction_rows.extend(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "node_id": int(node),
                            "label": int(labels[node]),
                            "method": method,
                            "seed_label": seed_label,
                            "seed": seed,
                            "score": float(score),
                        }
                        for node, score in zip(outer_test, test_score)
                    )
                    save_partial(audit_rows, prediction_rows, args.output)

    audit = pd.DataFrame(audit_rows)
    predictions = pd.DataFrame(prediction_rows)
    expected = 15 * len(args.models) * len(args.seed_offsets)
    if len(audit) != expected or audit[["repeat", "fold", "method", "seed_label"]].duplicated().any():
        raise RuntimeError(f"Expected {expected} unique fits, found {len(audit)}")
    manifest = {
        "endpoint": "Open Targets 26.06 disease-specific LUAD clinical-precedence evidence",
        "positive_count": positive_count,
        "total_genes": int(data.num_nodes),
        "outer_split_source": str(args.current_oof),
        "models": [DISPLAY[name] for name in args.models],
        "optimization_seeds_per_outer_fold": len(args.seed_offsets),
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "hyperparameter_policy": "Architecture hyperparameters fixed before evaluation; epoch selected by inner-validation AUPRC.",
        "outer_test_labels_used_for_training_or_seed_selection": False,
        "runtime_seconds": time.time() - started,
    }
    (args.output / "current26_deep_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
