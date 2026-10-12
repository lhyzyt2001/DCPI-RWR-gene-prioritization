from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score


SEED = 20260812
COLORS = {
    "RWR (STRING graph)": "#4C78A8",
    "RWR-feature APPNP": "#F58518",
    "RWR-feature GCNII": "#54A24B",
    "RWR-feature GATv2": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine uniformly trained deep-model seed analyses.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archived-oof", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def gene_cluster_bootstrap(frame: pd.DataFrame, method: str, n_boot: int) -> dict[str, float | str | int]:
    methods = frame[frame["method"].isin([method, "RWR (STRING graph)"])].copy()
    nodes = np.sort(methods["node_id"].unique().astype(int))
    label_map = methods.groupby("node_id")["label"].first()
    labels = label_map.loc[nodes].to_numpy(dtype=int)
    node_position = {node: i for i, node in enumerate(nodes)}
    split_arrays: list[dict[str, np.ndarray]] = []
    for (_, _), block in methods.groupby(["repeat", "fold"]):
        arrays = {name: np.full(len(nodes), np.nan, dtype=float) for name in [method, "RWR (STRING graph)"]}
        for name, part in block.groupby("method"):
            positions = np.asarray([node_position[int(node)] for node in part["node_id"]], dtype=int)
            arrays[name][positions] = part["score"].to_numpy(dtype=float)
        split_arrays.append(arrays)
    rng = np.random.default_rng(SEED)
    deltas: list[float] = []
    for _ in range(n_boot):
        sampled = rng.integers(0, len(nodes), size=len(nodes))
        y = labels[sampled]
        if np.unique(y).size < 2:
            continue
        method_aps, rwr_aps = [], []
        for arrays in split_arrays:
            method_score = arrays[method][sampled]
            rwr_score = arrays["RWR (STRING graph)"][sampled]
            keep = np.isfinite(method_score) & np.isfinite(rwr_score)
            if keep.sum() and np.unique(y[keep]).size == 2:
                method_aps.append(average_precision_score(y[keep], method_score[keep]))
                rwr_aps.append(average_precision_score(y[keep], rwr_score[keep]))
        method_ap = float(np.mean(method_aps)); rwr_ap = float(np.mean(rwr_aps))
        deltas.append(method_ap - rwr_ap)
    values = np.asarray(deltas)
    observed_method, observed_rwr = [], []
    for arrays in split_arrays:
        keep = np.isfinite(arrays[method]) & np.isfinite(arrays["RWR (STRING graph)"])
        observed_method.append(average_precision_score(labels[keep], arrays[method][keep]))
        observed_rwr.append(average_precision_score(labels[keep], arrays["RWR (STRING graph)"][keep]))
    observed_method = float(np.mean(observed_method)); observed_rwr = float(np.mean(observed_rwr))
    return {
        "method": method,
        "reference": "RWR (STRING graph)",
        "estimand": "AUPRC difference using the mean of three uniformly trained optimization seeds",
        "observed_delta": float(observed_method - observed_rwr),
        "bootstrap_median": float(np.median(values)),
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "two_sided_p": float(2 * min((np.sum(values <= 0) + 1) / (len(values) + 1), (np.sum(values >= 0) + 1) / (len(values) + 1))),
        "bootstrap_replicates": int(len(values)),
        "aggregation": "AUPRC calculated within each outer fold and averaged across the 15 folds",
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model_dirs = [args.root / name for name in ["APPNP", "GCNII", "GATv2"]]
    expected = [
        "deep_multiseed_fold_metrics.csv",
        "deep_multiseed_prediction_correlations.csv",
        "deep_multiseed_per_seed_summary.csv",
        "deep_multiseed_model_summary.csv",
        "additional_seed_oof_predictions.csv.gz",
        "additional_seed_training_audit.csv",
        "deep_multiseed_manifest.json",
    ]
    missing = [str(d / f) for d in model_dirs for f in expected if not (d / f).exists()]
    if missing:
        raise FileNotFoundError("Uniform multiseed runs are incomplete:\n" + "\n".join(missing))

    metrics = pd.concat([pd.read_csv(d / "deep_multiseed_fold_metrics.csv") for d in model_dirs], ignore_index=True)
    correlations = pd.concat([pd.read_csv(d / "deep_multiseed_prediction_correlations.csv") for d in model_dirs], ignore_index=True)
    per_seed = pd.concat([pd.read_csv(d / "deep_multiseed_per_seed_summary.csv") for d in model_dirs], ignore_index=True)
    summaries = pd.concat([pd.read_csv(d / "deep_multiseed_model_summary.csv") for d in model_dirs], ignore_index=True)
    audits = pd.concat([pd.read_csv(d / "additional_seed_training_audit.csv") for d in model_dirs], ignore_index=True)
    predictions = pd.concat([pd.read_csv(d / "additional_seed_oof_predictions.csv.gz") for d in model_dirs], ignore_index=True)

    if len(audits) != 135 or len(metrics) != 135:
        raise ValueError(f"Expected 135 fits/metric rows, observed audits={len(audits)}, metrics={len(metrics)}")
    if audits[["repeat", "fold", "method", "seed_label"]].duplicated().any():
        raise ValueError("Duplicate deep-model fit identifiers detected.")

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    labels = data.y.cpu().numpy().astype(int)
    predictions["label"] = labels[predictions["node_id"].to_numpy(dtype=int)]
    deep_mean = predictions.groupby(["repeat", "fold", "node_id", "method"], as_index=False).agg(score=("score", "mean"), label=("label", "first"))
    archived = pd.read_csv(args.archived_oof)
    rwr = archived[archived["method"].eq("RWR (STRING graph)")].copy()
    rwr["label"] = labels[rwr["node_id"].to_numpy(dtype=int)]
    combined = pd.concat([deep_mean, rwr[["repeat", "fold", "node_id", "method", "score", "label"]]], ignore_index=True)
    combined.to_csv(args.out / "uniform_seed_mean_and_rwr_oof.csv.gz", index=False, compression="gzip")

    method_metrics = []
    for method, block in combined.groupby("method"):
        for (repeat, fold), split in block.groupby(["repeat", "fold"]):
            method_metrics.append({"method": method, "repeat": repeat, "fold": fold, "AUPRC": average_precision_score(split["label"], split["score"])})
    method_metrics = pd.DataFrame(method_metrics)
    performance = method_metrics.groupby("method", as_index=False).agg(folds=("AUPRC", "size"), mean_AUPRC=("AUPRC", "mean"), sd_AUPRC=("AUPRC", "std"))
    performance.to_csv(args.out / "uniform_deep_vs_rwr_summary.csv", index=False, encoding="utf-8-sig")
    bootstrap = pd.DataFrame([gene_cluster_bootstrap(combined, method, args.bootstrap) for method in sorted(deep_mean["method"].unique())])
    bootstrap.to_csv(args.out / "uniform_deep_vs_rwr_gene_cluster_bootstrap.csv", index=False, encoding="utf-8-sig")

    metrics.to_csv(args.out / "uniform_multiseed_fold_metrics.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(args.out / "uniform_multiseed_prediction_correlations.csv", index=False, encoding="utf-8-sig")
    per_seed.to_csv(args.out / "uniform_multiseed_per_seed_summary.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(args.out / "uniform_multiseed_model_summary.csv", index=False, encoding="utf-8-sig")
    audits.to_csv(args.out / "uniform_multiseed_training_audit.csv", index=False, encoding="utf-8-sig")

    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    order = ["RWR (STRING graph)", "RWR-feature APPNP", "RWR-feature GCNII", "RWR-feature GATv2"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for i, method in enumerate(order):
        values = method_metrics.loc[method_metrics["method"].eq(method), "AUPRC"].to_numpy()
        x = np.linspace(i - 0.12, i + 0.12, len(values))
        axes[0].scatter(x, values, s=15, color=COLORS[method], alpha=0.75, edgecolor="white", linewidth=0.3)
        axes[0].hlines(values.mean(), i - 0.25, i + 0.25, color="#222222", linewidth=1.2)
    axes[0].set_xticks(range(4), ["RWR", "APPNP", "GCNII", "GATv2"])
    axes[0].set_ylabel("Outer-test AUPRC")
    axes[0].set_title("a  Uniform-seed mean versus RWR", loc="left")
    axes[0].grid(axis="y", color="#E6E6E6", linewidth=0.6)
    corr = correlations.groupby("method")["spearman_rho"].agg(["mean", "std"]).reindex(order[1:])
    axes[1].bar(range(3), corr["mean"], yerr=corr["std"], capsize=3, color=[COLORS[m] for m in order[1:]])
    axes[1].set_xticks(range(3), ["APPNP", "GCNII", "GATv2"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Pairwise Spearman correlation")
    axes[1].set_title("b  Optimization-seed stability", loc="left")
    axes[1].grid(axis="y", color="#E6E6E6", linewidth=0.6)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "uniform_multiseed_deep_benchmark.pdf", bbox_inches="tight")
    fig.savefig(args.out / "uniform_multiseed_deep_benchmark.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "models": order[1:],
        "training_protocol": "Three optimization seeds for every one of 15 fixed outer folds; identical maximum/minimum epochs and patience for every model and seed.",
        "fits": int(len(audits)),
        "outer_test_labels_used_for_training": bool(audits["outer_test_labels_used_for_training"].any()),
        "comparison": "Mean prediction across three uniformly trained optimization seeds compared with leakage-controlled RWR on the same splits.",
        "bootstrap_unit": "gene cluster, retaining all available repeat-level predictions",
        "bootstrap_replicates": args.bootstrap,
        "seed": SEED,
    }
    (args.out / "uniform_multiseed_combined_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(performance.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
