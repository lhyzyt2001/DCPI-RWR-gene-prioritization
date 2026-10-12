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


SEED = 42
N_SPLITS = 5
N_REPEATS = 3
RWR = "RWR (STRING graph)"
STANDARD = "LightGBM direct stack"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("benchmark_core", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fit_pu_bagging(base, x_train, y_train, x_val, y_val, x_test, seed: int, bags: int, ratio: int):
    positives = np.flatnonzero(y_train == 1)
    unlabelled = np.flatnonzero(y_train == 0)
    val_scores, test_scores = [], []
    for bag in range(bags):
        rng = np.random.default_rng(seed + 1009 * (bag + 1) + 7919 * ratio)
        n_unlabelled = min(len(unlabelled), ratio * len(positives))
        sampled_unlabelled = rng.choice(unlabelled, size=n_unlabelled, replace=False)
        selected = np.concatenate([positives, sampled_unlabelled])
        rng.shuffle(selected)
        model = base.lgb_model(seed + bag + ratio * 100)
        model.fit(
            x_train[selected],
            y_train[selected],
            sample_weight=np.ones(len(selected), dtype=float),
            eval_set=[(x_val, y_val)],
            eval_metric="average_precision",
            callbacks=[base.lgb.early_stopping(100, verbose=False), base.lgb.log_evaluation(0)],
        )
        val_scores.append(model.predict_proba(x_val, num_iteration=model.best_iteration_)[:, 1])
        test_scores.append(model.predict_proba(x_test, num_iteration=model.best_iteration_)[:, 1])
    return np.mean(val_scores, axis=0), np.mean(test_scores, axis=0)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def cluster_bootstrap(oof: pd.DataFrame, labels: np.ndarray, methods: list[str], replicates: int) -> pd.DataFrame:
    pivots = {int(r): x.pivot(index="node_id", columns="method", values="score") for r, x in oof.groupby("repeat")}
    positive = np.flatnonzero(labels == 1)
    unlabelled = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(SEED)
    comparisons = [(m, STANDARD) for m in methods if m not in {STANDARD, RWR}] + [(m, RWR) for m in methods if m != RWR]
    comparisons = list(dict.fromkeys(comparisons))
    distributions = {pair: np.empty(replicates) for pair in comparisons}
    for b in range(replicates):
        sampled = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(unlabelled, len(unlabelled), replace=True),
            ]
        )
        y = labels[sampled]
        ap = {
            method: np.mean(
                [average_precision_score(y, pivots[repeat].loc[sampled, method].to_numpy()) for repeat in pivots]
            )
            for method in methods
        }
        for pair in comparisons:
            distributions[pair][b] = ap[pair[0]] - ap[pair[1]]
    rows = []
    for (method, reference), dist in distributions.items():
        p = min(
            1.0,
            2
            * min(
                (np.sum(dist <= 0) + 1) / (replicates + 1),
                (np.sum(dist >= 0) + 1) / (replicates + 1),
            ),
        )
        rows.append(
            {
                "method": method,
                "reference": reference,
                "median_delta_AUPRC": float(np.median(dist)),
                "ci_lower": float(np.quantile(dist, 0.025)),
                "ci_upper": float(np.quantile(dist, 0.975)),
                "p_two_sided": float(p),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "gene retained across all three repeat-level OOF predictions",
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_two_sided"].to_numpy())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Bagging positive-unlabelled sensitivity analysis.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--archived-oof", type=Path, required=True)
    parser.add_argument("--benchmark-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratios", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--bags", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = load_module(args.benchmark_code)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    labels = data.y.cpu().numpy().astype(int)
    raw_expression = data.x.cpu().numpy().astype(np.float32)
    degree, _ = base.network_degree_features(data)
    centrality, _ = base.string_centralities(data, 128)
    topology = np.column_stack([degree, centrality]).astype(np.float32)
    transition = base.prepare_transition(data)
    archive = pd.read_csv(args.archived_oof)
    archive = archive[archive["method"].isin([RWR, STANDARD])][["repeat", "fold", "node_id", "method", "score"]]

    rows: list[dict] = []
    metrics: list[dict] = []
    outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    for split_number, (outer_train, outer_test) in enumerate(outer.split(np.zeros(len(labels)), labels)):
        repeat, fold = divmod(split_number, N_SPLITS)
        split_seed = SEED + split_number
        core, val = train_test_split(
            outer_train, test_size=0.15, stratify=labels[outer_train], random_state=split_seed
        )
        scaler = StandardScaler().fit(raw_expression[core])
        expression = scaler.transform(raw_expression)
        pca = PCA(n_components=5, random_state=SEED).fit(expression[core])
        feature = np.column_stack([topology, expression, pca.transform(expression)]).astype(np.float32)
        core_rwr = base.crossfit_rwr(transition, labels, core, split_seed)
        all_rwr = base.run_rwr(transition, len(labels), core[labels[core] == 1])
        calibrator = base.fit_rwr_calibrator(core_rwr, labels[core])
        x_core = np.column_stack([base.calibrated_logit(calibrator, core_rwr), feature[core]])
        x_val = np.column_stack([base.calibrated_logit(calibrator, all_rwr[val]), feature[val]])
        x_test = np.column_stack([base.calibrated_logit(calibrator, all_rwr[outer_test]), feature[outer_test]])
        for ratio in args.ratios:
            method = f"BaggingPU LightGBM {ratio}:1 unlabelled:positive"
            _, score = fit_pu_bagging(
                base, x_core, labels[core], x_val, labels[val], x_test, split_seed, args.bags, ratio
            )
            y_test = labels[outer_test]
            metrics.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "method": method,
                    "ROC_AUC": roc_auc_score(y_test, score),
                    "AUPRC": average_precision_score(y_test, score),
                }
            )
            rows.extend(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "node_id": int(node),
                    "method": method,
                    "score": float(value),
                }
                for node, value in zip(outer_test, score)
            )
        print(f"PU fold {split_number + 1}/15 complete", flush=True)

    pu_oof = pd.DataFrame(rows)
    combined = pd.concat([archive, pu_oof], ignore_index=True)
    archive_metrics = []
    for (repeat, fold, method), block in archive.groupby(["repeat", "fold", "method"]):
        nodes = block["node_id"].to_numpy(dtype=int)
        score = block["score"].to_numpy(dtype=float)
        archive_metrics.append(
            {
                "repeat": repeat,
                "fold": fold,
                "method": method,
                "ROC_AUC": roc_auc_score(labels[nodes], score),
                "AUPRC": average_precision_score(labels[nodes], score),
            }
        )
    fold_metrics = pd.concat([pd.DataFrame(archive_metrics), pd.DataFrame(metrics)], ignore_index=True)
    summary = (
        fold_metrics.groupby("method")[["ROC_AUC", "AUPRC"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["method", "ROC_AUC_mean", "ROC_AUC_sd", "AUPRC_mean", "AUPRC_sd"]
    summary = summary.sort_values("AUPRC_mean", ascending=False)
    methods = summary["method"].tolist()
    bootstrap = cluster_bootstrap(combined, labels, methods, args.bootstrap_replicates)

    ranks = []
    for (repeat, method), block in combined.groupby(["repeat", "method"]):
        block = block.groupby("node_id", as_index=False)["score"].mean()
        block = block.loc[labels[block["node_id"].to_numpy(dtype=int)] == 0].sort_values("score", ascending=False)
        for k in (20, 100, 500):
            top = set(block.head(k)["node_id"].astype(int))
            standard_block = combined[(combined["repeat"].eq(repeat)) & (combined["method"].eq(STANDARD))]
            standard_block = standard_block.loc[
                labels[standard_block["node_id"].to_numpy(dtype=int)] == 0
            ].sort_values("score", ascending=False)
            standard_top = set(standard_block.head(k)["node_id"].astype(int))
            ranks.append(
                {
                    "repeat": repeat,
                    "method": method,
                    "k": k,
                    "overlap_with_standard_lightgbm": len(top & standard_top),
                    "jaccard_with_standard_lightgbm": len(top & standard_top) / len(top | standard_top),
                }
            )

    fold_metrics.to_csv(args.output / "pu_fold_metrics.csv", index=False)
    combined.to_csv(args.output / "pu_and_reference_oof_predictions.csv.gz", index=False, compression="gzip")
    summary.to_csv(args.output / "pu_summary.csv", index=False)
    bootstrap.to_csv(args.output / "pu_gene_cluster_bootstrap.csv", index=False)
    pd.DataFrame(ranks).to_csv(args.output / "pu_candidate_rank_stability.csv", index=False)

    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.45})
    ordered = summary.sort_values("AUPRC_mean")
    axes[0].barh(ordered["method"], ordered["AUPRC_mean"], xerr=ordered["AUPRC_sd"], color="#4477AA")
    axes[0].set(xlabel="Outer-fold AUPRC (mean ± SD)", title="PU sensitivity")
    rank_table = pd.DataFrame(ranks)
    rank_table = rank_table[(rank_table["method"] != STANDARD) & (rank_table["k"] == 100)]
    rank_mean = rank_table.groupby("method", as_index=False)["jaccard_with_standard_lightgbm"].mean()
    axes[1].barh(rank_mean["method"], rank_mean["jaccard_with_standard_lightgbm"], color="#228833")
    axes[1].set(xlabel="Mean Top-100 Jaccard", title="Candidate-list sensitivity", xlim=(0, 1))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#E5E5E5", lw=0.6)
    fig.savefig(args.output / "pu_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(args.output / "pu_sensitivity.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    manifest = {
        "design": "bagging positive-unlabelled sensitivity",
        "temporary_negative_sampling_ratios": args.ratios,
        "bags_per_fold": args.bags,
        "outer_validation": "5 folds x 3 repeats, identical split generator to primary benchmark",
        "outer_test_labels_used_for_fitting": False,
        "important_boundary": "Label-zero genes are unlabelled, not confirmed negatives; this is a sensitivity analysis rather than identification of true negatives.",
    }
    (args.output / "pu_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
