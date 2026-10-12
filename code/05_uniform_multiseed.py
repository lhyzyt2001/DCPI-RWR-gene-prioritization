from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 42
INNER_VAL_FRACTION = 0.15
MODEL_NAMES = ["appnp", "gcnii", "gatv2"]
DISPLAY = {
    "appnp": "RWR-feature APPNP",
    "gcnii": "RWR-feature GCNII",
    "gatv2": "RWR-feature GATv2",
}
BASE_OFFSET = {"appnp": 1000, "gcnii": 2000, "gatv2": 3000}
COLORS = {
    "RWR-feature APPNP": "#CC79A7",
    "RWR-feature GCNII": "#D55E00",
    "RWR-feature GATv2": "#56B4E9",
    "RWR (STRING graph)": "#0072B2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uniform-protocol optimization-seed sensitivity across all 15 outer folds."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--strong-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seed-offsets", nargs="+", type=int, default=[0, 10000, 20000])
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--betweenness-samples", type=int, default=128)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--resume", action="store_true", help="Resume from the two partial CSV files in --output.")
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("strong_gnn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_partial(rows: list[dict], prediction_rows: list[dict], output: Path) -> None:
    pd.DataFrame(rows).to_csv(output / "additional_seed_training_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(
        output / "additional_seed_oof_predictions.csv.gz", index=False, compression="gzip"
    )


def evaluate_predictions(block: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, seed_label, repeat, fold), group in block.groupby(["method", "seed_label", "repeat", "fold"]):
        nodes = group["node_id"].to_numpy(dtype=int)
        score = group["score"].to_numpy(dtype=float)
        y = labels[nodes]
        rows.append(
            {
                "method": model,
                "seed_label": seed_label,
                "repeat": int(repeat),
                "fold": int(fold),
                "split_id": int(repeat) * 5 + int(fold),
                "n": len(group),
                "positives": int(y.sum()),
                "roc_auc": roc_auc_score(y, score),
                "auprc": average_precision_score(y, score),
            }
        )
    return pd.DataFrame(rows)


def prediction_correlations(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (method, repeat, fold), group in predictions.groupby(["method", "repeat", "fold"]):
        wide = group.pivot(index="node_id", columns="seed_label", values="score")
        columns = list(wide.columns)
        for left_idx in range(len(columns)):
            for right_idx in range(left_idx + 1, len(columns)):
                rho = spearmanr(wide[columns[left_idx]], wide[columns[right_idx]]).statistic
                rows.append(
                    {
                        "method": method,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "split_id": int(repeat) * 5 + int(fold),
                        "seed_a": columns[left_idx],
                        "seed_b": columns[right_idx],
                        "spearman_rho": float(rho),
                    }
                )
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame, correlations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = (
        metrics.groupby(["method", "seed_label"], as_index=False)
        .agg(
            outer_folds=("fold", "size"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd_across_folds=("roc_auc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_sd_across_folds=("auprc", "std"),
        )
    )
    fold_variability = (
        metrics.groupby(["method", "split_id"], as_index=False)
        .agg(
            seed_count=("seed_label", "size"),
            auprc_mean_across_seeds=("auprc", "mean"),
            auprc_sd_across_seeds=("auprc", "std"),
            auprc_min=("auprc", "min"),
            auprc_max=("auprc", "max"),
        )
    )
    model_summary = (
        fold_variability.groupby("method", as_index=False)
        .agg(
            folds=("split_id", "size"),
            mean_fold_auprc=("auprc_mean_across_seeds", "mean"),
            mean_within_fold_seed_sd=("auprc_sd_across_seeds", "mean"),
            max_within_fold_seed_range=("auprc_max", lambda x: 0.0),
        )
    )
    ranges = fold_variability.assign(seed_range=fold_variability["auprc_max"] - fold_variability["auprc_min"])
    model_summary["max_within_fold_seed_range"] = model_summary["method"].map(
        ranges.groupby("method")["seed_range"].max()
    )
    model_summary["mean_pairwise_prediction_spearman"] = model_summary["method"].map(
        correlations.groupby("method")["spearman_rho"].mean()
    )
    return per_seed, model_summary


def plot_results(metrics: pd.DataFrame, correlations: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    methods = list(metrics["method"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ax = axes[0]
    offsets = np.linspace(-0.18, 0.18, 3)
    seed_labels = sorted(metrics["seed_label"].unique())
    markers = ["o", "s", "^"]
    for mi, method in enumerate(methods):
        block = metrics[metrics["method"] == method]
        for si, seed_label in enumerate(seed_labels):
            values = block[block["seed_label"] == seed_label].sort_values("fold")
            ax.scatter(
                np.full(len(values), mi + offsets[si]),
                values["auprc"],
                marker=markers[si],
                s=24,
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.9,
                label=seed_label.replace("_", " ") if mi == 0 else None,
            )
        ax.hlines(block["auprc"].mean(), mi - 0.28, mi + 0.28, color="#222222", linewidth=1.2)
    ax.set_xticks(range(len(methods)), [m.replace("RWR-feature ", "") for m in methods])
    ax.set_ylabel("Outer-test AUPRC")
    ax.set_title("a  Fixed folds, three optimization seeds", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)

    ax = axes[1]
    summary = correlations.groupby("method")["spearman_rho"].agg(["mean", "std"]).reindex(methods)
    ax.bar(
        np.arange(len(methods)), summary["mean"], yerr=summary["std"],
        color=[COLORS[m] for m in methods], capsize=3, width=0.68,
    )
    ax.set_xticks(range(len(methods)), [m.replace("RWR-feature ", "") for m in methods])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pairwise Spearman correlation")
    ax.set_title("b  Prediction stability across seeds", loc="left")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "deep_multiseed_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(output / "deep_multiseed_sensitivity.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))
    module = load_module(args.strong_code)
    started = time.time()
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    labels = data.y.cpu().numpy().astype(int)
    raw_expression = data.x.cpu().numpy().astype(np.float32)
    topology, _ = module.prepare_static_features(data, args.betweenness_samples)
    transition = module.prepare_transition(data)
    edge_index = data.edge_index_ppi.cpu().long().contiguous()
    edge_weight = data.edge_weight_ppi.cpu().float().contiguous()
    y_tensor = data.y.cpu().long()
    archived = pd.read_csv(args.oof)
    prediction_rows: list[dict] = []
    audit_rows: list[dict] = []
    completed_keys: set[tuple[int, int, str, str]] = set()
    if args.resume:
        audit_path = args.output / "additional_seed_training_audit.csv"
        prediction_path = args.output / "additional_seed_oof_predictions.csv.gz"
        if audit_path.exists() and prediction_path.exists():
            previous_audit = pd.read_csv(audit_path)
            previous_predictions = pd.read_csv(prediction_path)
            audit_rows = previous_audit.to_dict("records")
            prediction_rows = previous_predictions.to_dict("records")
            completed_keys = {
                (int(row.repeat), int(row.fold), str(row.method), str(row.seed_label))
                for row in previous_audit.itertuples(index=False)
            }
            print(f"resuming from {len(completed_keys)} completed fits", flush=True)
    completed = len(completed_keys)
    total = len(args.repeats) * 5 * len(args.models) * len(args.seed_offsets)
    for repeat in args.repeats:
        split_source = archived[
            (archived["method"] == "RWR (STRING graph)") & (archived["repeat"] == repeat)
        ][["fold", "node_id"]]
        for fold in range(5):
            split_seed = SEED + repeat * 5 + fold
            outer_test = np.sort(split_source[split_source["fold"] == fold]["node_id"].to_numpy(dtype=int))
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
            expression_pca = pca.transform(expression_scaled)
            feature_all = np.column_stack([topology, expression_scaled, expression_pca]).astype(np.float32)
            core_rwr_oof = module.crossfit_rwr(transition, labels, core_train, split_seed)
            seed_nodes = core_train[labels[core_train] == 1]
            all_outer_rwr = module.run_rwr(transition, len(labels), seed_nodes)
            calibrator = module.fit_rwr_calibrator(core_rwr_oof, labels[core_train])
            rwr_logit = np.zeros(len(labels), dtype=np.float32)
            rwr_logit[core_train] = module.calibrated_logit(calibrator, core_rwr_oof)
            rwr_logit[inner_val] = module.calibrated_logit(calibrator, all_outer_rwr[inner_val])
            rwr_logit[outer_test] = module.calibrated_logit(calibrator, all_outer_rwr[outer_test])
            direct = np.column_stack([rwr_logit, feature_all]).astype(np.float32)
            neural_scaler = StandardScaler().fit(direct[core_train])
            direct_scaled = np.nan_to_num(neural_scaler.transform(direct), nan=0.0, posinf=10.0, neginf=-10.0)
            x_tensor = torch.as_tensor(direct_scaled, dtype=torch.float32)

            for model_name in args.models:
                for seed_index, seed_offset in enumerate(args.seed_offsets, start=1):
                    seed = split_seed + BASE_OFFSET[model_name] + seed_offset
                    seed_label = f"uniform_seed_{seed_index}"
                    fit_key = (repeat, fold, DISPLAY[model_name], seed_label)
                    if fit_key in completed_keys:
                        continue
                    completed += 1
                    print(
                        f"uniform fit {completed}/{total} | repeat {repeat + 1}/3 | fold {fold + 1}/5 | {model_name} | seed {seed_index}",
                        flush=True,
                    )
                    _, test_score, audit = module.train_model(
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
                            "method": DISPLAY[model_name],
                            "seed_label": seed_label,
                            "seed": seed,
                            "core_train_n": len(core_train),
                            "inner_val_n": len(inner_val),
                            "outer_test_n": len(outer_test),
                            "outer_test_labels_used_for_training": False,
                            "outer_test_nodes_used_as_rwr_seeds": False,
                            **audit,
                        }
                    )
                    prediction_rows.extend(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "node_id": int(node),
                            "method": DISPLAY[model_name],
                            "seed_label": seed_label,
                            "seed": seed,
                            "score": float(score),
                        }
                        for node, score in zip(outer_test, test_score)
                    )
                    save_partial(audit_rows, prediction_rows, args.output)

    predictions = pd.DataFrame(prediction_rows)
    metrics = evaluate_predictions(predictions, labels)
    correlations = prediction_correlations(predictions)
    per_seed, model_summary = summarize(metrics, correlations)
    metrics.to_csv(args.output / "deep_multiseed_fold_metrics.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(args.output / "deep_multiseed_prediction_correlations.csv", index=False, encoding="utf-8-sig")
    per_seed.to_csv(args.output / "deep_multiseed_per_seed_summary.csv", index=False, encoding="utf-8-sig")
    model_summary.to_csv(args.output / "deep_multiseed_model_summary.csv", index=False, encoding="utf-8-sig")
    plot_results(metrics, correlations, args.output)
    manifest = {
        "scope": "all 15 outer folds; fixed splits, preprocessing and identical training protocol; three optimization seeds per model",
        "repeats": args.repeats,
        "models": [DISPLAY[name] for name in args.models],
        "seed_labels": [f"uniform_seed_{i}" for i in range(1, len(args.seed_offsets) + 1)],
        "seed_offsets": args.seed_offsets,
        "max_epochs_for_all_seeds": args.max_epochs,
        "min_epochs_for_all_seeds": args.min_epochs,
        "patience_for_all_seeds": args.patience,
        "early_stopping_metric": "inner-validation AUPRC",
        "outer_test_labels_used_for_training_or_seed_selection": False,
        "runtime_seconds": time.time() - started,
        "limitation": "Optimization-seed sensitivity is not an independent validation cohort.",
    }
    (args.output / "deep_multiseed_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(model_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
