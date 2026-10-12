from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 20260926


def degree_pairs(labels: np.ndarray, degree: np.ndarray, eligible_nodes: np.ndarray) -> pd.DataFrame:
    eligible_nodes = np.asarray(eligible_nodes, dtype=int)
    positives = eligible_nodes[labels[eligible_nodes] == 1]
    pool = set(eligible_nodes[labels[eligible_nodes] == 0].tolist())
    by_degree: dict[int, list[int]] = {}
    for node in pool:
        by_degree.setdefault(int(degree[node]), []).append(int(node))
    rng = np.random.default_rng(SEED)
    for nodes in by_degree.values():
        rng.shuffle(nodes)
    rows, unmatched = [], []
    for positive in sorted(positives, key=lambda n: int(degree[n]), reverse=True):
        bucket = by_degree.get(int(degree[positive]), [])
        while bucket and bucket[-1] not in pool:
            bucket.pop()
        if bucket:
            negative = bucket.pop(); pool.remove(negative)
            rows.append((positive, negative, "exact", 0))
        else:
            unmatched.append(positive)
    for positive in unmatched:
        negative = min(pool, key=lambda n: (abs(int(degree[n]) - int(degree[positive])), n))
        pool.remove(negative)
        rows.append((positive, negative, "nearest", abs(int(degree[positive]) - int(degree[negative]))))
    return pd.DataFrame(rows, columns=["positive_node", "unlabelled_node", "match_type", "absolute_degree_difference"])


def bootstrap(oof: pd.DataFrame, pairs: pd.DataFrame, replicates: int) -> pd.DataFrame:
    methods = list(oof["method"].drop_duplicates())
    reference = "RWR"
    split_data = []
    for (repeat, fold), pair_block in pairs.groupby(["repeat", "fold"]):
        matrix = oof[(oof["repeat"].eq(repeat)) & (oof["fold"].eq(fold))].pivot(index="node_id", columns="method", values="score")
        split_data.append((pair_block[["positive_node", "unlabelled_node"]].to_numpy(dtype=int), matrix))
    rng = np.random.default_rng(SEED)
    dist = {method: np.empty(replicates) for method in methods if method != reference}
    for b in range(replicates):
        split_aps = {method: [] for method in methods}
        for pair_nodes, matrix in split_data:
            sampled_pairs = pair_nodes[rng.integers(0, len(pair_nodes), len(pair_nodes))]
            nodes = sampled_pairs.reshape(-1)
            y = np.tile([1, 0], len(sampled_pairs))
            for method in methods:
                split_aps[method].append(average_precision_score(y, matrix.loc[nodes, method].to_numpy()))
        aps = {method: float(np.mean(values)) for method, values in split_aps.items()}
        for method in dist:
            dist[method][b] = aps[method] - aps[reference]
    rows = []
    for method, values in dist.items():
        p = min(1.0, 2 * min((np.sum(values <= 0) + 1) / (replicates + 1), (np.sum(values >= 0) + 1) / (replicates + 1)))
        rows.append({"method": method, "reference": reference, "median_delta_AUPRC": np.median(values), "ci_lower": np.quantile(values, .025), "ci_upper": np.quantile(values, .975), "p_two_sided": p, "bootstrap_replicates": replicates, "bootstrap_unit": "degree-matched pair within each outer test fold; fold AUPRCs averaged across 15 splits"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Network-degree audit for the current disease-specific endpoint.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--reuse-bootstrap", action="store_true", help="Reuse completed matched-bootstrap tables when regenerating descriptive audits/figures.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    edge = data.edge_index_ppi.cpu().numpy()
    degree = np.bincount(edge[0], minlength=data.num_nodes)
    oof = pd.read_csv(args.oof)
    labels = oof.groupby("node_id")["label"].first().reindex(range(data.num_nodes)).to_numpy(dtype=int)
    pair_blocks = []
    for (repeat, fold), block in oof.groupby(["repeat", "fold"]):
        nodes = block["node_id"].unique().astype(int)
        current = degree_pairs(labels, degree, nodes)
        current.insert(0, "fold", int(fold)); current.insert(0, "repeat", int(repeat))
        pair_blocks.append(current)
    pairs = pd.concat(pair_blocks, ignore_index=True)
    exact = pairs[pairs["match_type"].eq("exact")]
    all_boot_path = args.output / "current_endpoint_degree_matched_bootstrap.csv"
    exact_boot_path = args.output / "current_endpoint_exact_degree_only_bootstrap.csv"
    if args.reuse_bootstrap and all_boot_path.exists() and exact_boot_path.exists():
        all_boot = pd.read_csv(all_boot_path)
        exact_boot = pd.read_csv(exact_boot_path)
    else:
        all_boot = bootstrap(oof, pairs, args.bootstrap)
        exact_boot = bootstrap(oof, exact, args.bootstrap)

    correlations = []
    removal = []
    quantiles = pd.qcut(pd.Series(degree).rank(method="first"), q=6, labels=False).to_numpy()
    strata = []
    percentile = pd.Series(degree).rank(pct=True).to_numpy()
    for (repeat, method), block in oof.groupby(["repeat", "method"]):
        block = block.set_index("node_id").sort_index()
        nodes = block.index.to_numpy(dtype=int); scores = block["score"].to_numpy(); y = block["label"].to_numpy(dtype=int)
        correlations.append({"repeat": repeat, "method": method, "spearman_score_vs_log_degree": spearmanr(scores, np.log1p(degree[nodes])).statistic})
        for stratum in range(6):
            keep = quantiles[nodes] == stratum
            y_keep = y[keep]
            estimable = np.unique(y_keep).size == 2
            strata.append({"repeat": repeat, "method": method, "degree_stratum": stratum + 1, "n": int(keep.sum()), "positives": int(y_keep.sum()), "prevalence": float(y_keep.mean()), "estimable": estimable, "ROC_AUC": roc_auc_score(y_keep, scores[keep]) if estimable else np.nan, "AUPRC": average_precision_score(y_keep, scores[keep]) if estimable else np.nan})
        for fraction in [0, .01, .05, .10]:
            keep = percentile[nodes] <= 1 - fraction
            y_keep = y[keep]
            estimable = np.unique(y_keep).size == 2
            removal.append({"repeat": repeat, "method": method, "removed_top_degree_fraction": fraction, "remaining_n": int(keep.sum()), "remaining_positives": int(y_keep.sum()), "prevalence": float(y_keep.mean()), "estimable": estimable, "ROC_AUC": roc_auc_score(y_keep, scores[keep]) if estimable else np.nan, "AUPRC": average_precision_score(y_keep, scores[keep]) if estimable else np.nan})
    correlations = pd.DataFrame(correlations); removal = pd.DataFrame(removal); strata = pd.DataFrame(strata)
    pairs.to_csv(args.output / "current_endpoint_degree_pairs.csv", index=False, encoding="utf-8-sig")
    all_boot.to_csv(all_boot_path, index=False, encoding="utf-8-sig")
    exact_boot.to_csv(exact_boot_path, index=False, encoding="utf-8-sig")
    correlations.to_csv(args.output / "current_endpoint_score_degree_correlations.csv", index=False, encoding="utf-8-sig")
    removal.to_csv(args.output / "current_endpoint_hub_removal.csv", index=False, encoding="utf-8-sig")
    strata.to_csv(args.output / "current_endpoint_degree_stratified_performance.csv", index=False, encoding="utf-8-sig")
    logd = np.log1p(degree); p = pairs.positive_node.to_numpy(); n = pairs.unlabelled_node.to_numpy()
    balance = {"pairs": len(pairs), "exact_pairs": len(exact), "nearest_pairs": int((pairs.match_type == "nearest").sum()), "max_absolute_degree_difference": int(pairs.absolute_degree_difference.max()), "log_degree_SMD": float((logd[p].mean()-logd[n].mean()) / np.sqrt((logd[p].var(ddof=1)+logd[n].var(ddof=1))/2)), "KS_p": float(ks_2samp(logd[p],logd[n]).pvalue)}
    (args.output / "current_endpoint_degree_balance.json").write_text(json.dumps(balance, indent=2), encoding="utf-8")

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for method, block in strata.groupby("method"):
        summary = block.groupby("degree_stratum")["AUPRC"].mean()
        axes[0].plot(summary.index, summary.values, marker="o", label=method)
    axes[0].set(xlabel="STRING degree stratum", ylabel="Within-stratum AUPRC", title="a  Degree-stratified performance")
    axes[0].set_xticks(range(1, 7)); axes[0].set_xlim(.8, 6.2)
    axes[0].text(.02, .02, "Strata without both classes are not estimable", transform=axes[0].transAxes, fontsize=6, color="#444444")
    for method, block in removal.groupby("method"):
        summary = block.groupby("removed_top_degree_fraction")["AUPRC"].mean()
        axes[1].plot(100*summary.index, summary.values, marker="o", label=method)
    axes[1].set(xlabel="Highest-degree genes removed (%)", ylabel="AUPRC", title="b  Hub-removal sensitivity")
    axes[0].legend(frameon=False, fontsize=6); axes[1].legend(frameon=False, fontsize=6)
    for ax in axes: ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y", color="#E5E5E5", linewidth=.6)
    fig.tight_layout(); fig.savefig(args.output / "current_endpoint_centrality_audit.pdf", bbox_inches="tight"); fig.savefig(args.output / "current_endpoint_centrality_audit.png", dpi=600, bbox_inches="tight"); plt.close(fig)
    print(json.dumps(balance, indent=2)); print(all_boot.to_string(index=False)); print(exact_boot.to_string(index=False))


if __name__ == "__main__":
    main()
