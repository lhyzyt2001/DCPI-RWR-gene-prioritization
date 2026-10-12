from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 42
METHODS = [
    "RWR (STRING graph)",
    "XGBoost direct stack",
    "LightGBM direct stack",
    "RWR-feature APPNP",
    "RWR-feature GCNII",
    "RWR-feature GATv2",
]
COLORS = {
    METHODS[0]: "#0072B2",
    METHODS[1]: "#E69F00",
    METHODS[2]: "#009E73",
    METHODS[3]: "#CC79A7",
    METHODS[4]: "#D55E00",
    METHODS[5]: "#56B4E9",
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metrics(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if np.unique(y).size < 2:
        return np.nan, np.nan
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def make_strata(degree: np.ndarray) -> np.ndarray:
    return pd.qcut(pd.Series(np.log1p(degree)), 10, labels=False, duplicates="drop").fillna(0).astype(int).to_numpy()


def fixed_degree_match(labels: np.ndarray, degree: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Match positives to unique unlabelled genes, exact degree first then nearest degree.

    This avoids calling label-zero genes confirmed negatives and avoids broad quantile strata.
    """
    rng = np.random.default_rng(SEED)
    pairs = []
    positives = np.flatnonzero(labels == 1)
    unlabelled = np.flatnonzero(labels == 0)
    by_degree: dict[int, list[int]] = {}
    for node in unlabelled:
        by_degree.setdefault(int(degree[node]), []).append(int(node))
    for nodes in by_degree.values():
        rng.shuffle(nodes)
    unmatched = []
    used: set[int] = set()
    for positive in rng.permutation(positives):
        candidates = by_degree.get(int(degree[positive]), [])
        if candidates:
            unlabelled_node = candidates.pop()
            used.add(unlabelled_node)
            pairs.append(
                {
                    "pair_id": len(pairs),
                    "match_type": "exact_degree",
                    "positive_node": int(positive),
                    "unlabelled_node": unlabelled_node,
                    "positive_degree": int(degree[positive]),
                    "unlabelled_degree": int(degree[unlabelled_node]),
                }
            )
        else:
            unmatched.append(int(positive))
    available = sorted((int(degree[n]), int(n)) for n in unlabelled if int(n) not in used)
    available_degrees = [x[0] for x in available]
    for positive in sorted(unmatched, key=lambda n: int(degree[n]), reverse=True):
        target = int(degree[positive])
        location = bisect.bisect_left(available_degrees, target)
        choices = [i for i in (location - 1, location) if 0 <= i < len(available)]
        best = min(choices, key=lambda i: (abs(available[i][0] - target), available[i][0], available[i][1]))
        unlabelled_degree, unlabelled_node = available.pop(best)
        available_degrees.pop(best)
        pairs.append(
            {
                "pair_id": len(pairs),
                "match_type": "nearest_degree",
                "positive_node": int(positive),
                "unlabelled_node": int(unlabelled_node),
                "positive_degree": target,
                "unlabelled_degree": int(unlabelled_degree),
            }
        )
    frame = pd.DataFrame(pairs)
    frame["absolute_degree_difference"] = (frame["positive_degree"] - frame["unlabelled_degree"]).abs()
    nodes = np.concatenate([frame["positive_node"].to_numpy(), frame["unlabelled_node"].to_numpy()])
    return nodes, frame


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def matched_bootstrap(oof: pd.DataFrame, labels: np.ndarray, pairs: pd.DataFrame, replicates: int) -> pd.DataFrame:
    pivots = {int(r): x.pivot(index="node_id", columns="method", values="score") for r, x in oof.groupby("repeat")}
    rng = np.random.default_rng(SEED)
    comparisons = [(method, METHODS[0]) for method in METHODS if method != METHODS[0]]
    values = {pair: np.empty(replicates) for pair in comparisons}
    pair_nodes = pairs[["positive_node", "unlabelled_node"]].to_numpy(dtype=int)
    for b in range(replicates):
        sampled = pair_nodes[rng.integers(0, len(pair_nodes), len(pair_nodes))].reshape(-1)
        y = labels[sampled]
        ap = {
            method: np.mean([average_precision_score(y, pivots[r].loc[sampled, method].to_numpy()) for r in pivots])
            for method in METHODS
        }
        for pair in comparisons:
            values[pair][b] = ap[pair[0]] - ap[pair[1]]
    rows = []
    for (method, reference), dist in values.items():
        p = min(1.0, 2 * min((np.sum(dist <= 0) + 1) / (replicates + 1), (np.sum(dist >= 0) + 1) / (replicates + 1)))
        rows.append(
            {
                "method": method,
                "reference": reference,
                "median_delta_AUPRC": float(np.median(dist)),
                "ci_lower": float(np.quantile(dist, 0.025)),
                "ci_upper": float(np.quantile(dist, 0.975)),
                "p_two_sided": float(p),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "degree-matched positive-unlabelled pair retained across three repeats",
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_two_sided"].to_numpy())
    return result


def conditional_permutation(
    data,
    labels: np.ndarray,
    oof: pd.DataFrame,
    strata: np.ndarray,
    base,
    permutations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_expression = data.x.cpu().numpy().astype(np.float32)
    degree_features, degree_names = base.network_degree_features(data)
    centrality, centrality_names = base.string_centralities(data, 128)
    topology = np.column_stack([degree_features, centrality]).astype(np.float32)
    transition = base.prepare_transition(data)
    split_source = oof[oof["method"].eq(METHODS[0])][["repeat", "fold", "node_id"]]
    rows, audits = [], []
    rng = np.random.default_rng(SEED)
    for split_number, ((repeat, fold), test_block) in enumerate(split_source.groupby(["repeat", "fold"], sort=True)):
        outer_test = np.sort(test_block["node_id"].to_numpy(dtype=int))
        mask = np.ones(len(labels), dtype=bool)
        mask[outer_test] = False
        outer_train = np.flatnonzero(mask)
        split_seed = SEED + split_number
        core, val = train_test_split(outer_train, test_size=0.15, stratify=labels[outer_train], random_state=split_seed)
        scaler = StandardScaler().fit(raw_expression[core])
        expression = scaler.transform(raw_expression)
        pca = PCA(n_components=5, random_state=SEED).fit(expression[core])
        feature = np.column_stack([topology, expression, pca.transform(expression)]).astype(np.float32)
        core_rwr = base.crossfit_rwr(transition, labels, core, split_seed)
        all_rwr = base.run_rwr(transition, len(labels), core[labels[core] == 1])
        calibrator = base.fit_rwr_calibrator(core_rwr, labels[core])
        core_base = base.calibrated_logit(calibrator, core_rwr)
        val_base = base.calibrated_logit(calibrator, all_rwr[val])
        test_base = base.calibrated_logit(calibrator, all_rwr[outer_test])
        x_core = np.column_stack([core_base, feature[core]])
        x_val = np.column_stack([val_base, feature[val]])
        x_test = np.column_stack([test_base, feature[outer_test]])
        fitted = {}
        fitted[METHODS[1]], _, baseline_xgb = base.fit_xgb_direct(x_core, labels[core], x_val, labels[val], x_test, split_seed)
        fitted[METHODS[2]], _, baseline_lgb = base.fit_lgb_direct(x_core, labels[core], x_val, labels[val], x_test, split_seed)
        baseline = {METHODS[1]: baseline_xgb, METHODS[2]: baseline_lgb}
        archive = oof[(oof["repeat"].eq(repeat)) & (oof["fold"].eq(fold))]
        max_diff = {}
        for method in (METHODS[1], METHODS[2]):
            expected = archive[archive["method"].eq(method)].set_index("node_id").loc[outer_test, "score"].to_numpy()
            max_diff[method] = float(np.max(np.abs(expected - baseline[method])))
        y_test = labels[outer_test]
        baseline_ap = {m: average_precision_score(y_test, s) for m, s in baseline.items()}
        rwr_condition = pd.qcut(pd.Series(test_base), 10, labels=False, duplicates="drop").fillna(0).astype(int).to_numpy()
        degree_condition = strata[outer_test]
        for permutation in range(permutations):
            topology_columns = list(range(1, 1 + topology.shape[1]))
            expression_columns = list(range(1 + topology.shape[1], x_test.shape[1]))
            for block_name, columns, condition in (
                ("RWR feature | STRING-degree strata", [0], degree_condition),
                ("all degree features | RWR strata", list(range(1, 1 + len(degree_names))), rwr_condition),
                ("all topology features | RWR strata", topology_columns, rwr_condition),
                ("expression and PCA features | RWR strata", expression_columns, rwr_condition),
            ):
                permuted = x_test.copy()
                for level in np.unique(condition):
                    local = np.flatnonzero(condition == level)
                    source = rng.permutation(local)
                    permuted[np.ix_(local, columns)] = x_test[np.ix_(source, columns)]
                for method, model in fitted.items():
                    if method == METHODS[1]:
                        score = model.predict_proba(permuted)[:, 1]
                    else:
                        score = model.predict_proba(permuted, num_iteration=model.best_iteration_)[:, 1]
                    rows.append(
                        {
                            "repeat": int(repeat),
                            "fold": int(fold),
                            "method": method,
                            "permuted_block": block_name,
                            "permutation": permutation + 1,
                            "baseline_AUPRC": baseline_ap[method],
                            "permuted_AUPRC": average_precision_score(y_test, score),
                            "AUPRC_drop": baseline_ap[method] - average_precision_score(y_test, score),
                        }
                    )
        audits.append({"repeat": int(repeat), "fold": int(fold), **{f"max_abs_diff_{m}": v for m, v in max_diff.items()}})
        print(f"conditional permutation fold {split_number + 1}/15 complete", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(audits)


def plot_audit(stratified: pd.DataFrame, removed: pd.DataFrame, matched_boot: pd.DataFrame, permutation: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2), gridspec_kw={"hspace": 0.5, "wspace": 0.42})
    mean_strata = stratified.groupby(["method", "degree_stratum"], as_index=False)["AUPRC"].mean()
    for method in METHODS:
        part = mean_strata[mean_strata["method"].eq(method)]
        axes[0, 0].plot(part["degree_stratum"], part["AUPRC"], marker="o", ms=2.5, lw=1, color=COLORS[method], label=method.replace("RWR-feature ", ""))
    axes[0, 0].set(xlabel="Empirical STRING degree stratum", ylabel="Within-stratum AUPRC", title="Degree-stratified performance")
    axes[0, 0].legend(frameon=False, fontsize=5.7, ncol=2)
    mean_removed = removed.groupby(["method", "removed_top_degree_fraction"], as_index=False)["AUPRC"].mean()
    for method in METHODS:
        part = mean_removed[mean_removed["method"].eq(method)]
        axes[0, 1].plot(100 * part["removed_top_degree_fraction"], part["AUPRC"], marker="o", ms=3, lw=1, color=COLORS[method])
    axes[0, 1].set(xlabel="Highest-degree genes removed (%)", ylabel="AUPRC", title="Hub-removal sensitivity")
    boot = matched_boot.iloc[::-1].reset_index(drop=True)
    for i, row in boot.iterrows():
        axes[1, 0].errorbar(row["median_delta_AUPRC"], i, xerr=[[row["median_delta_AUPRC"]-row["ci_lower"]],[row["ci_upper"]-row["median_delta_AUPRC"]]], fmt="o", color=COLORS[row["method"]], capsize=3)
    axes[1, 0].axvline(0, ls="--", lw=1, color="#666666")
    axes[1, 0].set_yticks(range(len(boot)), [x.replace("RWR-feature ", "").replace(" direct stack", "") + " - RWR" for x in boot["method"]])
    axes[1, 0].set(xlabel="Degree-matched ΔAUPRC (95% CI)", title="1:1 degree-matched bootstrap")
    perm = permutation.groupby(["method", "permuted_block"], as_index=False)["AUPRC_drop"].mean()
    labels = list(perm["permuted_block"].unique())
    width = 0.35
    for i, method in enumerate((METHODS[1], METHODS[2])):
        part = perm[perm["method"].eq(method)].set_index("permuted_block").loc[labels]
        axes[1, 1].bar(np.arange(len(labels)) + (i-0.5)*width, part["AUPRC_drop"], width, color=COLORS[method], label=method.split()[0])
    axes[1, 1].axhline(0, color="#666666", lw=0.8)
    short_labels = {
        "RWR feature | STRING-degree strata": "RWR\nwithin degree",
        "all degree features | RWR strata": "Degree\nwithin RWR",
        "all topology features | RWR strata": "Topology\nwithin RWR",
        "expression and PCA features | RWR strata": "Expression\nwithin RWR",
    }
    axes[1, 1].set_xticks(range(len(labels)), [short_labels.get(x, x) for x in labels], rotation=15, ha="right")
    axes[1, 1].set(ylabel="Mean AUPRC decrease", title="Conditional feature permutation")
    axes[1, 1].legend(frameon=False)
    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.18, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E5E5E5", lw=0.6)
    fig.savefig(output / "network_centrality_audit.pdf", bbox_inches="tight")
    fig.savefig(output / "network_centrality_audit.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--benchmark-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    labels = data.y.cpu().numpy().astype(int)
    edge = data.edge_index_ppi.cpu().numpy()
    degree = np.bincount(edge[0], minlength=data.num_nodes)
    strata = make_strata(degree)
    oof = pd.read_csv(args.oof)
    oof = oof[oof["method"].isin(METHODS)].copy()
    if set(oof["method"].unique()) != set(METHODS):
        raise RuntimeError(f"Unexpected methods: {sorted(oof['method'].unique())}")
    strat_rows, calibration_rows, top_rows, corr_rows, removal_rows = [], [], [], [], []
    degree_percentile = pd.Series(degree).rank(pct=True).to_numpy()
    for (repeat, method), part in oof.groupby(["repeat", "method"]):
        part = part.set_index("node_id").sort_index()
        nodes = part.index.to_numpy(dtype=int)
        scores = part["score"].to_numpy()
        corr_rows.append({"repeat": repeat, "method": method, "spearman_score_vs_log_degree": float(spearmanr(scores, np.log1p(degree[nodes])).statistic)})
        for decile in sorted(np.unique(strata)):
            keep = strata[nodes] == decile
            y = labels[nodes[keep]]
            s = scores[keep]
            roc, ap = metrics(y, s)
            strat_rows.append({"repeat": repeat, "method": method, "degree_stratum": int(decile)+1, "n": int(keep.sum()), "positives": int(y.sum()), "prevalence": float(y.mean()), "ROC_AUC": roc, "AUPRC": ap})
            calibration_rows.append({"repeat": repeat, "method": method, "degree_stratum": int(decile)+1, "mean_score": float(s.mean()), "observed_prevalence": float(y.mean()), "brier_score": float(brier_score_loss(y, s)), "n": int(len(y))})
            order = np.argsort(-s)
            k = max(1, int(np.ceil(0.10 * len(order))))
            hits = int(y[order[:k]].sum())
            top_rows.append({"repeat": repeat, "method": method, "degree_stratum": int(decile)+1, "top_fraction": 0.10, "k": k, "hits": hits, "precision": hits/k, "recall_within_stratum": hits/max(int(y.sum()),1)})
        for fraction in (0.0, 0.01, 0.05, 0.10):
            keep = degree_percentile[nodes] <= 1.0 - fraction
            roc, ap = metrics(labels[nodes[keep]], scores[keep])
            remaining_n = int(keep.sum())
            remaining_positives = int(labels[nodes[keep]].sum())
            prevalence = remaining_positives / remaining_n
            removal_rows.append({
                "repeat": repeat,
                "method": method,
                "removed_top_degree_fraction": fraction,
                "remaining_n": remaining_n,
                "remaining_positives": remaining_positives,
                "prevalence": prevalence,
                "ROC_AUC": roc,
                "AUPRC": ap,
                "AUPRC_over_prevalence": ap / prevalence if prevalence > 0 else np.nan,
            })
    matched_nodes, pair_table = fixed_degree_match(labels, degree)
    matched_rows = []
    for (repeat, method), part in oof.groupby(["repeat", "method"]):
        pivot = part.set_index("node_id")
        y = labels[matched_nodes]
        roc, ap = metrics(y, pivot.loc[matched_nodes, "score"].to_numpy())
        matched_rows.append({"repeat": repeat, "method": method, "matched_n": len(y), "matched_positives": int(y.sum()), "prevalence": float(y.mean()), "ROC_AUC": roc, "AUPRC": ap})
    log_degree = np.log1p(degree)
    pos = pair_table["positive_node"].to_numpy()
    neg = pair_table["unlabelled_node"].to_numpy()
    balance = {
        "pairs": len(pair_table),
        "positive_mean_log_degree": float(log_degree[pos].mean()),
        "negative_mean_log_degree": float(log_degree[neg].mean()),
        "standardized_mean_difference_log_degree": float((log_degree[pos].mean()-log_degree[neg].mean()) / np.sqrt((log_degree[pos].var(ddof=1)+log_degree[neg].var(ddof=1))/2)),
        "ks_statistic": float(ks_2samp(log_degree[pos], log_degree[neg]).statistic),
        "ks_p_value": float(ks_2samp(log_degree[pos], log_degree[neg]).pvalue),
        "exact_degree_pairs": int(pair_table["match_type"].eq("exact_degree").sum()),
        "nearest_degree_pairs": int(pair_table["match_type"].eq("nearest_degree").sum()),
        "maximum_absolute_degree_difference": int(pair_table["absolute_degree_difference"].max()),
    }
    matched_boot = matched_bootstrap(oof, labels, pair_table, args.bootstrap_replicates)
    exact_pair_table = pair_table[pair_table["match_type"].eq("exact_degree")].copy()
    exact_matched_boot = matched_bootstrap(oof, labels, exact_pair_table, args.bootstrap_replicates)
    base = load_module(args.benchmark_code)
    permutation, permutation_audit = conditional_permutation(data, labels, oof, strata, base, args.permutations)
    tables = {
        "degree_stratified_performance.csv": pd.DataFrame(strat_rows),
        "degree_stratified_calibration.csv": pd.DataFrame(calibration_rows),
        "degree_stratified_top10pct_recovery.csv": pd.DataFrame(top_rows),
        "score_degree_correlations.csv": pd.DataFrame(corr_rows),
        "high_degree_removal_sensitivity.csv": pd.DataFrame(removal_rows),
        "degree_matched_pairs.csv": pair_table,
        "degree_matched_performance.csv": pd.DataFrame(matched_rows),
        "degree_matched_gene_cluster_bootstrap.csv": matched_boot,
        "exact_degree_only_gene_cluster_bootstrap.csv": exact_matched_boot,
        "conditional_permutation_results.csv.gz": permutation,
        "conditional_permutation_reproduction_audit.csv": permutation_audit,
    }
    for name, frame in tables.items():
        if name.endswith(".gz"):
            frame.to_csv(args.output / name, index=False, compression="gzip")
        else:
            frame.to_csv(args.output / name, index=False, encoding="utf-8-sig")
    (args.output / "degree_matching_balance.json").write_text(json.dumps(balance, indent=2), encoding="utf-8")
    plot_audit(tables["degree_stratified_performance.csv"], tables["high_degree_removal_sensitivity.csv"], matched_boot, permutation, args.output)
    manifest = {
        "methods": METHODS,
        "degree_strata": int(strata.max()+1),
        "matched_pairs": len(pair_table),
        "exact_degree_only_pairs": len(exact_pair_table),
        "permutations_per_fold_block": args.permutations,
        "bootstrap_replicates": args.bootstrap_replicates,
        "balance": balance,
        "matching_note": "The primary 1:1 set uses exact degree when available and nearest unused degree otherwise. An exact-degree-only bootstrap is reported separately to expose sensitivity to the 57 nearest matches.",
    }
    (args.output / "centrality_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
