from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 20260930
METHOD_ORDER = [
    "RWR",
    "Network-degree logistic",
    "XGBoost + RWR/features",
    "LightGBM + RWR/features",
    "APPNP",
    "GCNII",
    "GATv2",
]
COLORS = {
    "RWR": "#0072B2",
    "Network-degree logistic": "#777777",
    "XGBoost + RWR/features": "#E69F00",
    "LightGBM + RWR/features": "#009E73",
    "APPNP": "#CC79A7",
    "GCNII": "#D55E00",
    "GATv2": "#56B4E9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine every model on the shared Open Targets 26.06 endpoint.")
    parser.add_argument("--classical-oof", type=Path, required=True)
    parser.add_argument("--degree-oof", type=Path)
    parser.add_argument("--deep-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def holm_adjust(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    total = len(p)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def load_deep(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    audits = []
    for model in ["APPNP", "GCNII", "GATv2"]:
        model_dir = root / model
        pred = pd.read_csv(model_dir / "current26_deep_oof_predictions.csv.gz")
        audit = pd.read_csv(model_dir / "current26_deep_training_audit.csv")
        predictions.append(pred)
        audits.append(audit)
    return pd.concat(predictions, ignore_index=True), pd.concat(audits, ignore_index=True)


def fold_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (repeat, fold, method), block in oof.groupby(["repeat", "fold", "method"]):
        rows.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "method": method,
                "n": len(block),
                "positives": int(block["label"].sum()),
                "prevalence": float(block["label"].mean()),
                "ROC_AUC": float(roc_auc_score(block["label"], block["score"])),
                "AUPRC": float(average_precision_score(block["label"], block["score"])),
            }
        )
    return pd.DataFrame(rows)


def clustered_bootstrap(oof: pd.DataFrame, replicates: int) -> pd.DataFrame:
    methods = [method for method in METHOD_ORDER if method in set(oof["method"])]
    nodes = np.sort(oof["node_id"].unique().astype(int))
    label_map = oof.groupby("node_id")["label"].first()
    labels = label_map.loc[nodes].to_numpy(dtype=int)
    positives = np.flatnonzero(labels == 1)
    unlabelled = np.flatnonzero(labels == 0)
    node_position = {node: index for index, node in enumerate(nodes)}
    split_arrays: list[dict[str, np.ndarray]] = []
    for (_, _), block in oof.groupby(["repeat", "fold"]):
        arrays = {method: np.full(len(nodes), np.nan, dtype=float) for method in methods}
        for method, part in block.groupby("method"):
            positions = np.asarray([node_position[int(node)] for node in part["node_id"]], dtype=int)
            arrays[method][positions] = part["score"].to_numpy(dtype=float)
        split_arrays.append(arrays)

    comparisons = [(method, "RWR") for method in methods if method != "RWR"]
    distributions = {pair: np.empty(replicates, dtype=float) for pair in comparisons}
    rng = np.random.default_rng(SEED)
    for iteration in range(replicates):
        sampled = np.concatenate(
            [
                rng.choice(positives, len(positives), replace=True),
                rng.choice(unlabelled, len(unlabelled), replace=True),
            ]
        )
        sampled_y = labels[sampled]
        aps: dict[str, float] = {}
        for method in methods:
            per_fold = []
            for arrays in split_arrays:
                score = arrays[method][sampled]
                keep = np.isfinite(score)
                if keep.sum() and np.unique(sampled_y[keep]).size == 2:
                    per_fold.append(average_precision_score(sampled_y[keep], score[keep]))
            aps[method] = float(np.mean(per_fold))
        for pair in comparisons:
            distributions[pair][iteration] = aps[pair[0]] - aps[pair[1]]

    rows = []
    for (method, reference), values in distributions.items():
        p_value = 2 * min(
            (np.sum(values <= 0) + 1) / (replicates + 1),
            (np.sum(values >= 0) + 1) / (replicates + 1),
        )
        rows.append(
            {
                "method": method,
                "reference": reference,
                "median_delta_AUPRC": float(np.median(values)),
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "p_two_sided": float(min(1.0, p_value)),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "label-stratified gene cluster retained across repeats; AUPRC computed within each outer fold",
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_two_sided"])
    return result


def topk_recovery(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, repeat), block in oof.groupby(["method", "repeat"]):
        positives = int(block["label"].sum())
        ranked = block.sort_values("score", ascending=False)
        for k in [20, 50, 100, 200, 500, 1000]:
            hits = int(ranked.head(k)["label"].sum())
            rows.append(
                {
                    "method": method,
                    "repeat": int(repeat),
                    "K": k,
                    "positive_hits": hits,
                    "precision_at_K": hits / k,
                    "recall_at_K": hits / positives,
                }
            )
    return pd.DataFrame(rows)


def plot_results(metrics: pd.DataFrame, bootstrap: pd.DataFrame, prevalence: float, output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    order = [method for method in METHOD_ORDER if method in set(metrics["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"wspace": 0.48})
    ax = axes[0]
    for index, method in enumerate(order):
        values = metrics.loc[metrics["method"] == method, "AUPRC"].to_numpy()
        jitter = np.linspace(-0.075, 0.075, len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=12,
            facecolor="white",
            edgecolor=COLORS[method],
            linewidth=0.75,
            zorder=2,
        )
        ax.errorbar(
            index,
            values.mean(),
            yerr=values.std(ddof=1),
            fmt="o",
            color=COLORS[method],
            capsize=3,
            markersize=5,
            linewidth=1.2,
            zorder=3,
        )
    short = {
        "RWR": "RWR",
        "Network-degree logistic": "Degree",
        "XGBoost + RWR/features": "XGBoost",
        "LightGBM + RWR/features": "LightGBM",
        "APPNP": "APPNP",
        "GCNII": "GCNII",
        "GATv2": "GATv2",
    }
    ax.axhline(prevalence, color="#666666", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(order)), [short[m] for m in order], rotation=35, ha="right")
    ax.set_ylabel("Outer-test AUPRC")
    ax.set_title("a  Shared 26.06 endpoint and outer folds", loc="left")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)

    ax = axes[1]
    selected = bootstrap.set_index("method").loc[[m for m in order if m != "RWR"]].reset_index()
    selected = selected.iloc[::-1].reset_index(drop=True)
    for index, row in selected.iterrows():
        estimate = row["median_delta_AUPRC"]
        ax.errorbar(
            estimate,
            index,
            xerr=[[estimate - row["ci_lower"]], [row["ci_upper"] - estimate]],
            fmt="o",
            color=COLORS[row["method"]],
            capsize=3,
            linewidth=1.3,
            markersize=5,
        )
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(selected)), [short[m] + " − RWR" for m in selected["method"]])
    ax.set_xlabel("Median AUPRC difference (95% CI)")
    ax.set_title("b  Gene-cluster bootstrap", loc="left")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.28, top=0.86)
    fig.savefig(output / "current26_unified_benchmark.pdf", bbox_inches="tight")
    fig.savefig(output / "current26_unified_benchmark.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    classical = pd.read_csv(args.classical_oof)
    classical = classical[classical["method"].isin(METHOD_ORDER)].copy()
    if args.degree_oof and args.degree_oof.exists():
        degree = pd.read_csv(args.degree_oof)
        degree = degree[degree["method"] == "Network-degree logistic"].copy()
        if not degree.empty:
            classical = pd.concat([classical, degree[classical.columns]], ignore_index=True)

    deep, audit = load_deep(args.deep_root)
    if len(audit) != 135:
        raise RuntimeError(f"Expected 135 current-endpoint deep fits, found {len(audit)}")
    if audit[["repeat", "fold", "method", "seed_label"]].duplicated().any():
        raise RuntimeError("Duplicate deep fit identifiers detected.")
    if audit["outer_test_labels_used_for_training"].astype(str).str.lower().isin({"true", "1"}).any():
        raise RuntimeError("An outer-test label leakage flag is set.")
    deep_mean = (
        deep.groupby(["repeat", "fold", "node_id", "label", "method"], as_index=False)["score"].mean()
    )
    combined = pd.concat([classical, deep_mean], ignore_index=True)
    expected_methods = [method for method in METHOD_ORDER if method in set(combined["method"])]
    counts = combined.groupby("method").size()
    expected_rows = combined["node_id"].nunique() * 3
    if not (counts.loc[expected_methods] == expected_rows).all():
        raise RuntimeError(f"Incomplete OOF rows by method: {counts.to_dict()}")

    metrics = fold_metrics(combined)
    summary = (
        metrics.groupby("method", as_index=False)
        .agg(
            folds=("AUPRC", "size"),
            mean_ROC_AUC=("ROC_AUC", "mean"),
            sd_ROC_AUC=("ROC_AUC", "std"),
            mean_AUPRC=("AUPRC", "mean"),
            sd_AUPRC=("AUPRC", "std"),
        )
    )
    summary["endpoint"] = "Open Targets 26.06 disease-specific LUAD clinical-precedence"
    bootstrap = clustered_bootstrap(combined, args.bootstrap)
    topk = topk_recovery(combined)
    topk_summary = (
        topk.groupby(["method", "K"], as_index=False)
        .agg(
            positive_hits_mean=("positive_hits", "mean"),
            positive_hits_sd=("positive_hits", "std"),
            precision_at_K_mean=("precision_at_K", "mean"),
            recall_at_K_mean=("recall_at_K", "mean"),
        )
    )

    wide = metrics.pivot(index=["repeat", "fold"], columns="method", values="AUPRC")
    tests = []
    for method in expected_methods:
        if method == "RWR":
            continue
        delta = wide[method] - wide["RWR"]
        test = wilcoxon(delta, alternative="two-sided", zero_method="pratt")
        tests.append(
            {
                "method": method,
                "reference": "RWR",
                "mean_fold_delta_AUPRC": float(delta.mean()),
                "folds_positive": int((delta > 0).sum()),
                "folds": len(delta),
                "wilcoxon_p_two_sided": float(test.pvalue),
                "note": "Secondary only; repeated-CV folds are correlated.",
            }
        )
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm_adjust(tests["wilcoxon_p_two_sided"])

    mapping = json.loads(args.mapping.read_text(encoding="utf-8-sig"))
    node_to_symbol = {int(node): str(symbol) for symbol, node in mapping.items()}
    assignments = (
        combined[combined["method"] == "RWR"][["repeat", "fold", "node_id", "label"]]
        .drop_duplicates()
        .sort_values(["repeat", "fold", "node_id"])
    )
    assignments.insert(3, "gene", assignments["node_id"].map(node_to_symbol))

    combined.to_csv(args.output / "current26_all_model_oof_predictions.csv.gz", index=False, compression="gzip")
    audit.to_csv(args.output / "current26_all_deep_training_audit.csv", index=False, encoding="utf-8-sig")
    assignments.to_csv(args.output / "current26_fold_assignments.csv.gz", index=False, compression="gzip")
    metrics.to_csv(args.output / "current26_all_model_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output / "current26_all_model_summary.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(args.output / "current26_all_model_gene_cluster_bootstrap.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(args.output / "current26_all_model_paired_fold_tests.csv", index=False, encoding="utf-8-sig")
    topk.to_csv(args.output / "current26_all_model_topk_by_repeat.csv", index=False, encoding="utf-8-sig")
    topk_summary.to_csv(args.output / "current26_all_model_topk_summary.csv", index=False, encoding="utf-8-sig")
    prevalence = float(combined.groupby("node_id")["label"].first().mean())
    plot_results(metrics, bootstrap, prevalence, args.output)
    manifest = {
        "primary_endpoint": "Open Targets 26.06 disease-specific LUAD clinical-precedence evidence",
        "positive_count": int(combined.groupby("node_id")["label"].first().sum()),
        "gene_count": int(combined["node_id"].nunique()),
        "prevalence": prevalence,
        "outer_design": "three-repeat five-fold shared transductive split",
        "deep_seed_policy": "three prespecified optimization seeds per model and outer fold; predictions averaged before model comparison",
        "inference": "label-stratified gene-cluster bootstrap retaining the gene across all repeats; Holm correction across comparisons versus RWR",
        "important_boundary": "Methods are compared only within this endpoint; AUPRC is not compared numerically with historical, matched or temporal endpoints.",
    }
    (args.output / "current26_unified_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.sort_values("mean_AUPRC", ascending=False).to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
