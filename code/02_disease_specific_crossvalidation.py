from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 20260926
METHODS = ["RWR", "XGBoost + RWR/features", "LightGBM + RWR/features"]
COLORS = {"RWR": "#0072B2", "XGBoost + RWR/features": "#E69F00", "LightGBM + RWR/features": "#009E73"}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def clustered_bootstrap(frame: pd.DataFrame, replicates: int) -> pd.DataFrame:
    nodes = np.sort(frame["node_id"].unique().astype(int))
    label_map = frame.groupby("node_id")["label"].first()
    labels = label_map.loc[nodes].to_numpy(dtype=int)
    node_position = {node: i for i, node in enumerate(nodes)}
    split_arrays: list[dict[str, np.ndarray]] = []
    for (_, _), block in frame.groupby(["repeat", "fold"]):
        arrays = {method: np.full(len(nodes), np.nan, dtype=float) for method in METHODS}
        for method, part in block.groupby("method"):
            positions = np.asarray([node_position[int(node)] for node in part["node_id"]], dtype=int)
            arrays[method][positions] = part["score"].to_numpy(dtype=float)
        split_arrays.append(arrays)
    comparisons = [(METHODS[1], METHODS[0]), (METHODS[2], METHODS[0]), (METHODS[2], METHODS[1])]
    rng = np.random.default_rng(SEED)
    distributions = {pair: np.empty(replicates) for pair in comparisons}
    for b in range(replicates):
        sampled_positions = rng.integers(0, len(nodes), len(nodes))
        y = labels[sampled_positions]
        aps = {}
        for method in METHODS:
            split_ap = []
            for arrays in split_arrays:
                scores = arrays[method][sampled_positions]
                keep = np.isfinite(scores)
                if keep.sum() and np.unique(y[keep]).size == 2:
                    split_ap.append(average_precision_score(y[keep], scores[keep]))
            aps[method] = float(np.mean(split_ap))
        for pair in comparisons:
            distributions[pair][b] = aps[pair[0]] - aps[pair[1]]
    result = []
    for (method, reference), values in distributions.items():
        p = 2 * min((np.sum(values <= 0) + 1) / (replicates + 1), (np.sum(values >= 0) + 1) / (replicates + 1))
        result.append({
            "method": method, "reference": reference,
            "median_delta_AUPRC": float(np.median(values)),
            "ci_lower": float(np.quantile(values, 0.025)), "ci_upper": float(np.quantile(values, 0.975)),
            "p_two_sided": float(min(1, p)), "bootstrap_replicates": replicates,
            "bootstrap_unit": "gene cluster retained across repeats; AUPRC calculated within each outer fold and averaged across the 15 folds",
        })
    return pd.DataFrame(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated CV on a disease-specific Open Targets clinical-precedence endpoint.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--benchmark-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", default="26.06")
    parser.add_argument("--disease", default="LUAD")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = load_module(args.benchmark_code)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8-sig"))
    symbol_to_node = {str(g).upper(): int(i) for g, i in mapping.items()}
    symbols = set(pd.read_csv(args.labels)["symbol"].astype(str).str.upper())
    positives = sorted(symbol_to_node[g] for g in symbols if g in symbol_to_node)
    labels = np.zeros(data.num_nodes, dtype=int)
    labels[positives] = 1
    if len(positives) < 20:
        raise ValueError(f"Only {len(positives)} positives mapped; endpoint input is likely wrong.")

    raw_expression = data.x.cpu().numpy().astype(np.float32)
    degree, degree_names = base.network_degree_features(data)
    centrality, centrality_names = base.string_centralities(data, 128)
    static = np.column_stack([degree, centrality]).astype(np.float32)
    transition = base.prepare_transition(data)
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    predictions: list[dict] = []
    audits: list[dict] = []
    for split_number, (outer_train, outer_test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        repeat, fold = divmod(split_number, 5)
        split_seed = SEED + split_number
        core, validation = train_test_split(outer_train, test_size=0.15, stratify=labels[outer_train], random_state=split_seed)
        scaler = StandardScaler().fit(raw_expression[core])
        expression = scaler.transform(raw_expression)
        pca = PCA(n_components=5, random_state=SEED).fit(expression[core])
        features = np.column_stack([static, expression, pca.transform(expression)]).astype(np.float32)
        core_rwr = base.crossfit_rwr(transition, labels, core, split_seed)
        all_rwr = base.run_rwr(transition, len(labels), core[labels[core] == 1])
        calibrator = base.fit_rwr_calibrator(core_rwr, labels[core])
        x_core = np.column_stack([base.calibrated_logit(calibrator, core_rwr), features[core]])
        x_val = np.column_stack([base.calibrated_logit(calibrator, all_rwr[validation]), features[validation]])
        x_test = np.column_stack([base.calibrated_logit(calibrator, all_rwr[outer_test]), features[outer_test]])
        xgb, _, xgb_score = base.fit_xgb_direct(x_core, labels[core], x_val, labels[validation], x_test, split_seed)
        lgb, _, lgb_score = base.fit_lgb_direct(x_core, labels[core], x_val, labels[validation], x_test, split_seed)
        scores = {METHODS[0]: all_rwr[outer_test], METHODS[1]: xgb_score, METHODS[2]: lgb_score}
        for method, values in scores.items():
            predictions.extend({
                "repeat": repeat, "fold": fold, "node_id": int(node), "label": int(labels[node]),
                "method": method, "score": float(score),
            } for node, score in zip(outer_test, values))
        audits.append({
            "repeat": repeat, "fold": fold, "core_n": len(core), "core_positives": int(labels[core].sum()),
            "validation_n": len(validation), "validation_positives": int(labels[validation].sum()),
            "outer_test_n": len(outer_test), "outer_test_positives": int(labels[outer_test].sum()),
            "outer_test_labels_used_for_fitting": False, "outer_test_nodes_used_as_RWR_seeds": False,
            "xgb_best_iteration": int(getattr(xgb, "best_iteration", 0) or 0),
            "lgb_best_iteration": int(lgb.best_iteration_ or 0),
        })
        print(f"disease-specific CV fold {split_number + 1}/15", flush=True)
    oof = pd.DataFrame(predictions)
    audit = pd.DataFrame(audits)
    metrics = []
    for (repeat, fold, method), part in oof.groupby(["repeat", "fold", "method"]):
        metrics.append({
            "repeat": repeat, "fold": fold, "method": method, "n": len(part), "positives": int(part["label"].sum()),
            "prevalence": float(part["label"].mean()), "ROC_AUC": roc_auc_score(part["label"], part["score"]),
            "AUPRC": average_precision_score(part["label"], part["score"]),
        })
    metrics = pd.DataFrame(metrics)
    summary = metrics.groupby("method", as_index=False).agg(folds=("AUPRC", "size"), mean_ROC_AUC=("ROC_AUC", "mean"), sd_ROC_AUC=("ROC_AUC", "std"), mean_AUPRC=("AUPRC", "mean"), sd_AUPRC=("AUPRC", "std"))
    boot = clustered_bootstrap(oof, args.bootstrap)
    oof.to_csv(args.output / "disease_specific_oof_predictions.csv.gz", index=False, compression="gzip")
    audit.to_csv(args.output / "disease_specific_training_audit.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output / "disease_specific_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output / "disease_specific_summary.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(args.output / "disease_specific_gene_cluster_bootstrap.csv", index=False, encoding="utf-8-sig")

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "pdf.fonttype": 42})
    order = METHODS
    means = summary.set_index("method").loc[order]
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ax.bar(range(3), means["mean_AUPRC"], yerr=means["sd_AUPRC"], capsize=3, color=[COLORS[m] for m in order])
    ax.axhline(labels.mean(), color="#666666", linestyle="--", linewidth=1, label=f"Prevalence ({labels.mean():.3f})")
    ax.set_xticks(range(3), ["RWR", "XGBoost + RWR/features", "LightGBM + RWR/features"], rotation=20, ha="right")
    ax.set_ylabel("Outer-test AUPRC")
    ax.set_title(f"{args.disease} disease-specific {args.release} endpoint")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(args.output / "disease_specific_crossvalidation.pdf", bbox_inches="tight")
    fig.savefig(args.output / "disease_specific_crossvalidation.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    manifest = {
        "disease": args.disease, "release": args.release,
        "endpoint": "disease-specific clinical-precedence evidence mapped to the graph",
        "positive_count": len(positives), "total_genes": int(data.num_nodes),
        "unlabelled_note": "Zero denotes absence of this evidence in the specified release, not a confirmed biological negative.",
        "design": "three-repeat five-fold transductive CV; fold-local scaling/PCA; RWR seeds restricted to positive core-training nodes",
        "bootstrap": "gene-cluster bootstrap retaining repeated predictions; AUPRC calculated within each outer fold and averaged across 15 folds",
        "interpretation_boundary": "This is internal endpoint prediction, not causal-gene discovery or prospective validation.",
    }
    (args.output / "disease_specific_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
