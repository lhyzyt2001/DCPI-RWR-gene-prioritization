from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


SEED = 42
N_SPLITS = 5
N_REPEATS = 3
INNER_VAL_FRACTION = 0.15
METHOD_RWR = "RWR (STRING graph)"
METHOD_DEGREE = "Network-degree logistic"
METHOD_GNN = "Legacy multiview GCN comparator (local attention + class-weighted CE)"
METHODS = [
    METHOD_RWR,
    METHOD_DEGREE,
    METHOD_GNN,
    "LightGBM RWR-only stack",
    "LightGBM engineered features",
    "LightGBM stack excluding oligodendrocyte and cell-type graph",
    "XGBoost direct stack",
    "LightGBM direct stack",
    "RWR + XGBoost residual",
    "RWR + LightGBM residual",
    "RWR + XGBoost residual (topology only)",
    "RWR + XGBoost residual (expression only)",
]
PLOT_METHODS = [
    "LightGBM direct stack",
    METHOD_RWR,
    METHOD_DEGREE,
    METHOD_GNN,
    "XGBoost direct stack",
    "RWR + LightGBM residual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-controlled nested benchmark of RWR with XGBoost/LightGBM residual learners."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to GNN_Dataset_Raw.pt")
    parser.add_argument("--archived-oof", type=Path, required=True, help="Archived primary oof_predictions.csv.gz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--betweenness-samples", type=int, default=128)
    return parser.parse_args()


def save_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def network_degree_features(data) -> tuple[np.ndarray, list[str]]:
    values: list[np.ndarray] = []
    names: list[str] = []
    for prefix, edge_name, weight_name in (
        ("STRING", "edge_index_ppi", "edge_weight_ppi"),
        ("CellType", "edge_index_coexp", "edge_weight_coexp"),
        ("Pathway", "edge_index_pathway", "edge_weight_pathway"),
    ):
        edge = getattr(data, edge_name).cpu().numpy()
        weight = getattr(data, weight_name).cpu().numpy()
        degree = np.bincount(edge[0], minlength=data.num_nodes)
        weighted = np.bincount(edge[0], weights=weight, minlength=data.num_nodes)
        values.extend([np.log1p(degree), np.log1p(weighted)])
        names.extend([f"{prefix}_log_degree", f"{prefix}_log_weighted_degree"])
    return np.column_stack(values).astype(np.float32), names


def string_centralities(data, samples: int) -> tuple[np.ndarray, list[str]]:
    edge = data.edge_index_ppi.cpu().numpy()
    weight = data.edge_weight_ppi.cpu().numpy()
    keep = edge[0] < edge[1]
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_weighted_edges_from(
        (int(u), int(v), float(w)) for u, v, w in zip(edge[0, keep], edge[1, keep], weight[keep])
    )
    pagerank = nx.pagerank(graph, alpha=0.85, weight="weight", max_iter=200, tol=1e-10)
    clustering = nx.clustering(graph, weight="weight")
    core = nx.core_number(graph)
    betweenness = nx.betweenness_centrality(
        graph,
        k=min(samples, data.num_nodes),
        normalized=True,
        weight=None,
        seed=SEED,
    )
    matrix = np.column_stack(
        [
            np.log1p(np.fromiter((pagerank[i] for i in range(data.num_nodes)), dtype=float)),
            np.fromiter((clustering[i] for i in range(data.num_nodes)), dtype=float),
            np.log1p(np.fromiter((core[i] for i in range(data.num_nodes)), dtype=float)),
            np.log1p(np.fromiter((betweenness[i] for i in range(data.num_nodes)), dtype=float)),
        ]
    ).astype(np.float32)
    return matrix, [
        "STRING_log_PageRank",
        "STRING_weighted_clustering",
        "STRING_log_core_number",
        "STRING_log_approx_betweenness",
    ]


def prepare_transition(data) -> sp.csc_matrix:
    edge = data.edge_index_ppi.cpu().numpy()
    weight = data.edge_weight_ppi.cpu().numpy()
    adjacency = sp.coo_matrix(
        (weight, (edge[0], edge[1])), shape=(data.num_nodes, data.num_nodes)
    ).tocsc()
    degree = np.asarray(adjacency.sum(axis=0)).ravel()
    degree[degree == 0] = 1.0
    return adjacency @ sp.diags(1.0 / degree)


def run_rwr(transition: sp.csc_matrix, num_nodes: int, seed_idx: np.ndarray, restart: float = 0.3) -> np.ndarray:
    initial = np.zeros(num_nodes, dtype=float)
    initial[np.asarray(seed_idx, dtype=int)] = 1.0 / max(len(seed_idx), 1)
    score = initial.copy()
    for _ in range(200):
        updated = (1.0 - restart) * transition.dot(score) + restart * initial
        if np.linalg.norm(updated - score, 1) < 1e-9:
            return np.asarray(updated).ravel()
        score = updated
    return np.asarray(score).ravel()


def crossfit_rwr(
    transition: sp.csc_matrix,
    labels: np.ndarray,
    indices: np.ndarray,
    split_seed: int,
) -> np.ndarray:
    result = np.full(len(indices), np.nan, dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed + 1000)
    local_y = labels[indices]
    for fit_local, held_local in splitter.split(np.zeros(len(indices)), local_y):
        fit_nodes = indices[fit_local]
        held_nodes = indices[held_local]
        seeds = fit_nodes[labels[fit_nodes] == 1]
        all_scores = run_rwr(transition, len(labels), seeds)
        result[held_local] = all_scores[held_nodes]
    if np.isnan(result).any():
        raise RuntimeError("Cross-fitted RWR contains missing values")
    return result


def fit_rwr_calibrator(scores: np.ndarray, labels: np.ndarray):
    x = np.log10(np.maximum(scores, 1e-15)).reshape(-1, 1)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
    model.fit(x, labels)
    return model


def calibrated_logit(model, scores: np.ndarray) -> np.ndarray:
    x = np.log10(np.maximum(scores, 1e-15)).reshape(-1, 1)
    probability = model.predict_proba(x)[:, 1]
    return logit(np.clip(probability, 1e-5, 1 - 1e-5))


def class_weights(y: np.ndarray) -> np.ndarray:
    positive = max(int((y == 1).sum()), 1)
    negative = max(int((y == 0).sum()), 1)
    return np.where(y == 1, negative / positive, 1.0).astype(float)


def xgb_model(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=1400,
        learning_rate=0.025,
        max_depth=3,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5.0,
        gamma=0.0,
        tree_method="hist",
        early_stopping_rounds=60,
        random_state=seed,
        n_jobs=8,
    )


def lgb_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1800,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=4,
        min_child_samples=80,
        min_split_gain=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=8,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def choose_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        value = f1_score(y, scores >= threshold, zero_division=0)
        if value > best_f1:
            best_threshold, best_f1 = float(threshold), float(value)
    return best_threshold


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores >= threshold
    return {
        "ROC_AUC": roc_auc_score(y, scores),
        "AUPRC": average_precision_score(y, scores),
        "Accuracy": accuracy_score(y, predicted),
        "Precision": precision_score(y, predicted, zero_division=0),
        "Recall": recall_score(y, predicted, zero_division=0),
        "F1": f1_score(y, predicted, zero_division=0),
    }


def fit_xgb_direct(x_train, y_train, x_val, y_val, x_test, seed):
    model = xgb_model(seed)
    train_weight = class_weights(y_train)
    val_weight = class_weights(y_val)
    model.fit(
        x_train,
        y_train,
        sample_weight=train_weight,
        eval_set=[(x_val, y_val)],
        sample_weight_eval_set=[val_weight],
        verbose=False,
    )
    return model, model.predict_proba(x_val)[:, 1], model.predict_proba(x_test)[:, 1]


def fit_xgb_residual(x_train, y_train, x_val, y_val, x_test, base_train, base_val, base_test, seed):
    model = xgb_model(seed)
    train_weight = class_weights(y_train)
    val_weight = class_weights(y_val)
    model.fit(
        x_train,
        y_train,
        sample_weight=train_weight,
        base_margin=base_train,
        eval_set=[(x_val, y_val)],
        sample_weight_eval_set=[val_weight],
        base_margin_eval_set=[base_val],
        verbose=False,
    )
    val_score = model.predict_proba(x_val, base_margin=base_val)[:, 1]
    test_score = model.predict_proba(x_test, base_margin=base_test)[:, 1]
    return model, val_score, test_score


def fit_lgb_direct(x_train, y_train, x_val, y_val, x_test, seed):
    model = lgb_model(seed)
    model.fit(
        x_train,
        y_train,
        sample_weight=class_weights(y_train),
        eval_set=[(x_val, y_val)],
        eval_sample_weight=[class_weights(y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    iteration = model.best_iteration_ or model.n_estimators
    return model, model.predict_proba(x_val, num_iteration=iteration)[:, 1], model.predict_proba(x_test, num_iteration=iteration)[:, 1]


def fit_lgb_residual(x_train, y_train, x_val, y_val, x_test, base_train, base_val, base_test, seed):
    model = lgb_model(seed)
    model.fit(
        x_train,
        y_train,
        sample_weight=class_weights(y_train),
        init_score=base_train,
        eval_set=[(x_val, y_val)],
        eval_sample_weight=[class_weights(y_val)],
        eval_init_score=[base_val],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    iteration = model.best_iteration_ or model.n_estimators
    val_residual = model.predict(x_val, raw_score=True, num_iteration=iteration)
    test_residual = model.predict(x_test, raw_score=True, num_iteration=iteration)
    return model, expit(base_val + val_residual), expit(base_test + test_residual)


def bootstrap_cluster_differences(oof: pd.DataFrame, labels: np.ndarray, replicates: int) -> pd.DataFrame:
    wide = oof.pivot(index=["node_id", "repeat"], columns="method", values="score").reset_index()
    repeat_values = sorted(wide["repeat"].unique())
    matrices = {
        int(repeat): wide[wide["repeat"] == repeat].set_index("node_id").sort_index()
        for repeat in repeat_values
    }
    node_ids = matrices[int(repeat_values[0])].index.to_numpy(dtype=int)
    y = labels[node_ids]
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(SEED)
    methods = METHODS
    pairs = [(method, METHOD_RWR) for method in methods if method != METHOD_RWR] + [
        ("LightGBM direct stack", "LightGBM RWR-only stack"),
        ("LightGBM direct stack", "LightGBM engineered features"),
        ("LightGBM direct stack", "RWR + LightGBM residual"),
        ("LightGBM direct stack", "LightGBM stack excluding oligodendrocyte and cell-type graph"),
    ]
    differences = {pair: np.empty(replicates, dtype=float) for pair in pairs}
    for b in range(replicates):
        sampled_local = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        sampled_y = y[sampled_local]
        method_ap = {method: [] for method in methods}
        for repeat in repeat_values:
            matrix = matrices[int(repeat)]
            for method in methods:
                method_ap[method].append(
                    average_precision_score(sampled_y, matrix[method].to_numpy()[sampled_local])
                )
        method_ap = {method: float(np.mean(values)) for method, values in method_ap.items()}
        for pair in pairs:
            differences[pair][b] = method_ap[pair[0]] - method_ap[pair[1]]
    rows = []
    for (method, reference), values in differences.items():
        p_two = min(1.0, 2.0 * min((np.sum(values <= 0) + 1) / (replicates + 1), (np.sum(values >= 0) + 1) / (replicates + 1)))
        rows.append(
            {
                "method": method,
                "reference": reference,
                "median_delta_AUPRC": float(np.median(values)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "p_two_sided": float(p_two),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "gene cluster retained across all three repeats",
            }
        )
    return pd.DataFrame(rows).sort_values("median_delta_AUPRC", ascending=False)


def plot_results(oof: pd.DataFrame, labels: np.ndarray, summary: pd.DataFrame, bootstrap: pd.DataFrame, importance: pd.DataFrame, output_dir: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        METHOD_RWR: "#000000",
        METHOD_DEGREE: "#777777",
        METHOD_GNN: "#CC79A7",
        "LightGBM RWR-only stack": "#999999",
        "LightGBM engineered features": "#F0E442",
        "XGBoost direct stack": "#E69F00",
        "LightGBM direct stack": "#56B4E9",
        "RWR + XGBoost residual": "#009E73",
        "RWR + LightGBM residual": "#0072B2",
        "RWR + XGBoost residual (topology only)": "#D55E00",
        "RWR + XGBoost residual (expression only)": "#CC79A7",
    }
    fig = plt.figure(figsize=(10.8, 7.2))
    grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    ax_roc = fig.add_subplot(grid[0, 0])
    ax_pr = fig.add_subplot(grid[0, 1])
    ax_delta = fig.add_subplot(grid[1, 0])
    ax_imp = fig.add_subplot(grid[1, 1])
    for method in PLOT_METHODS:
        repeat_curves_roc, repeat_curves_pr = [], []
        x_roc = np.linspace(0, 1, 501)
        x_pr = np.linspace(0, 1, 501)
        for repeat in sorted(oof["repeat"].unique()):
            block = oof[(oof["repeat"] == repeat) & (oof["method"] == method)].sort_values("node_id")
            y = labels[block["node_id"].to_numpy(dtype=int)]
            score = block["score"].to_numpy()
            fpr, tpr, _ = roc_curve(y, score)
            precision, recall, _ = precision_recall_curve(y, score)
            repeat_curves_roc.append(np.interp(x_roc, fpr, tpr))
            order = np.argsort(recall)
            repeat_curves_pr.append(np.interp(x_pr, recall[order], precision[order]))
        roc_mean = np.mean(repeat_curves_roc, axis=0)
        pr_mean = np.mean(repeat_curves_pr, axis=0)
        row = summary[summary["Method"] == method].iloc[0]
        label = f"{method} ({row['ROC_AUC_mean']:.3f} | {row['AUPRC_mean']:.3f})"
        ax_roc.plot(x_roc, roc_mean, lw=1.6, color=colors[method], label=label)
        ax_pr.plot(x_pr, pr_mean, lw=1.6, color=colors[method])
    ax_roc.plot([0, 1], [0, 1], "--", color="#888888", lw=1)
    ax_pr.axhline(labels.mean(), ls="--", color="#888888", lw=1)
    ax_roc.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curves")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves")
    ax_roc.legend(frameon=False, loc="lower right", fontsize=6)
    for ax in (ax_roc, ax_pr):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    boot = bootstrap[bootstrap["reference"] == METHOD_RWR].sort_values("median_delta_AUPRC")
    yloc = np.arange(len(boot))
    ax_delta.axvline(0, color="#777777", lw=1, ls="--")
    ax_delta.errorbar(
        boot["median_delta_AUPRC"],
        yloc,
        xerr=[boot["median_delta_AUPRC"] - boot["ci_low"], boot["ci_high"] - boot["median_delta_AUPRC"]],
        fmt="o",
        color="#0072B2",
        ecolor="#0072B2",
        capsize=2.5,
    )
    ax_delta.set_yticks(yloc, [m.replace("RWR + ", "") for m in boot["method"]])
    ax_delta.set_xlabel("AUPRC difference versus RWR")
    ax_delta.set_title("Gene-cluster bootstrap (95% CI)")
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["right"].set_visible(False)

    best_method = next(method for method in summary["Method"] if method in set(importance["method"]))
    imp = importance[importance["method"] == best_method].groupby("feature", as_index=False)["importance"].mean().nlargest(12, "importance").sort_values("importance")
    ax_imp.barh(np.arange(len(imp)), imp["importance"], color="#009E73")
    ax_imp.set_yticks(np.arange(len(imp)), imp["feature"])
    ax_imp.set_xlabel("Mean gain importance")
    ax_imp.set_title(f"{best_method} feature importance")
    ax_imp.spines["top"].set_visible(False)
    ax_imp.spines["right"].set_visible(False)
    for label, ax in zip("abcd", (ax_roc, ax_pr, ax_delta, ax_imp)):
        ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    fig.suptitle("Leakage-controlled RWR-GBDT benchmarking", fontsize=12, y=0.995)
    fig.savefig(output_dir / "RWR_GBDT_nested_benchmark.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "RWR_GBDT_nested_benchmark.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    labels = data.y.cpu().numpy().astype(int)
    raw_expression = data.x.cpu().numpy().astype(np.float32)
    degree, degree_names = network_degree_features(data)
    centrality, centrality_names = string_centralities(data, args.betweenness_samples)
    topology = np.column_stack([degree, centrality]).astype(np.float32)
    topology_names = degree_names + centrality_names
    transition = prepare_transition(data)

    archived = pd.read_csv(args.archived_oof)
    archived_rwr = archived[archived["method"] == METHOD_RWR][["repeat", "fold", "node_id", "score"]].copy()
    if len(archived_rwr) != len(labels) * N_REPEATS:
        raise RuntimeError("Archived RWR OOF file does not contain exactly one score per gene and repeat")

    outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    metric_rows: list[dict] = []
    oof_rows: list[dict] = []
    importance_rows: list[dict] = []
    audit_rows: list[dict] = []
    max_rwr_difference = 0.0
    for split_number, (outer_train, outer_test) in enumerate(outer.split(np.zeros(len(labels)), labels)):
        repeat, fold = divmod(split_number, N_SPLITS)
        split_seed = SEED + split_number
        core_train, inner_val = train_test_split(
            outer_train,
            test_size=INNER_VAL_FRACTION,
            stratify=labels[outer_train],
            random_state=split_seed,
        )
        scaler = StandardScaler().fit(raw_expression[core_train])
        expression_scaled = scaler.transform(raw_expression)
        pca = PCA(n_components=5, random_state=SEED).fit(expression_scaled[core_train])
        expression_pca = pca.transform(expression_scaled)
        feature_all = np.column_stack([topology, expression_scaled, expression_pca]).astype(np.float32)
        cell_type_names = [str(name).replace(" ", "_") for name in data.cell_types]
        feature_names = topology_names + [f"CellType_{name}" for name in cell_type_names] + [f"PCA_{i + 1}" for i in range(5)]
        oligodendrocyte_idx = [i for i, name in enumerate(cell_type_names) if "oligodendro" in name.lower()]
        if len(oligodendrocyte_idx) != 1:
            raise RuntimeError(f"Expected one oligodendrocyte feature, found {oligodendrocyte_idx}")
        retained_cell_idx = np.array([i for i in range(raw_expression.shape[1]) if i not in oligodendrocyte_idx])
        sensitivity_scaler = StandardScaler().fit(raw_expression[core_train][:, retained_cell_idx])
        sensitivity_expression = sensitivity_scaler.transform(raw_expression[:, retained_cell_idx])
        sensitivity_pca = PCA(n_components=5, random_state=SEED).fit(sensitivity_expression[core_train])
        sensitivity_pca_all = sensitivity_pca.transform(sensitivity_expression)
        retained_topology_idx = np.array([i for i, name in enumerate(topology_names) if not name.startswith("CellType_")])
        sensitivity_features = np.column_stack(
            [topology[:, retained_topology_idx], sensitivity_expression, sensitivity_pca_all]
        ).astype(np.float32)
        sensitivity_feature_names = (
            [topology_names[i] for i in retained_topology_idx]
            + [f"CellType_{cell_type_names[i]}" for i in retained_cell_idx]
            + [f"PCA_no_oligodendrocyte_{i + 1}" for i in range(5)]
        )
        topology_idx = np.arange(len(topology_names))
        expression_idx = np.arange(len(topology_names), feature_all.shape[1])

        core_rwr_oof = crossfit_rwr(transition, labels, core_train, split_seed)
        outer_seed_nodes = core_train[labels[core_train] == 1]
        all_outer_rwr = run_rwr(transition, len(labels), outer_seed_nodes)
        val_rwr = all_outer_rwr[inner_val]
        test_rwr = all_outer_rwr[outer_test]
        calibrator = fit_rwr_calibrator(core_rwr_oof, labels[core_train])
        core_base = calibrated_logit(calibrator, core_rwr_oof)
        val_base = calibrated_logit(calibrator, val_rwr)
        test_base = calibrated_logit(calibrator, test_rwr)

        archived_fold = archived_rwr[(archived_rwr["repeat"] == repeat) & (archived_rwr["fold"] == fold)].set_index("node_id")
        archived_scores = archived_fold.loc[outer_test, "score"].to_numpy()
        fold_diff = float(np.max(np.abs(archived_scores - test_rwr)))
        max_rwr_difference = max(max_rwr_difference, fold_diff)

        test_y = labels[outer_test]
        val_y = labels[inner_val]
        rwr_threshold = choose_threshold(val_y, val_rwr)
        rwr_metrics = metrics(test_y, test_rwr, rwr_threshold)
        metric_rows.append({"repeat": repeat, "fold": fold, "method": METHOD_RWR, **rwr_metrics})
        oof_rows.extend(
            {"repeat": repeat, "fold": fold, "node_id": int(node), "method": METHOD_RWR, "score": float(score), "threshold": rwr_threshold}
            for node, score in zip(outer_test, test_rwr)
        )

        x_train_full = feature_all[core_train]
        x_val_full = feature_all[inner_val]
        x_test_full = feature_all[outer_test]
        direct_train = np.column_stack([core_base, x_train_full])
        direct_val = np.column_stack([val_base, x_val_full])
        direct_test = np.column_stack([test_base, x_test_full])

        fitted: list[tuple[str, object, np.ndarray, np.ndarray, list[str]]] = []
        model, val_score, test_score = fit_lgb_direct(
            core_base.reshape(-1, 1), labels[core_train], val_base.reshape(-1, 1), val_y, test_base.reshape(-1, 1), split_seed
        )
        fitted.append(("LightGBM RWR-only stack", model, val_score, test_score, ["RWR_calibrated_logit"]))
        model, val_score, test_score = fit_lgb_direct(x_train_full, labels[core_train], x_val_full, val_y, x_test_full, split_seed)
        fitted.append(("LightGBM engineered features", model, val_score, test_score, feature_names))
        sensitivity_train = np.column_stack([core_base, sensitivity_features[core_train]])
        sensitivity_val = np.column_stack([val_base, sensitivity_features[inner_val]])
        sensitivity_test = np.column_stack([test_base, sensitivity_features[outer_test]])
        model, val_score, test_score = fit_lgb_direct(
            sensitivity_train, labels[core_train], sensitivity_val, val_y, sensitivity_test, split_seed
        )
        fitted.append(
            (
                "LightGBM stack excluding oligodendrocyte and cell-type graph",
                model,
                val_score,
                test_score,
                ["RWR_calibrated_logit"] + sensitivity_feature_names,
            )
        )
        model, val_score, test_score = fit_xgb_direct(direct_train, labels[core_train], direct_val, val_y, direct_test, split_seed)
        fitted.append(("XGBoost direct stack", model, val_score, test_score, ["RWR_calibrated_logit"] + feature_names))
        model, val_score, test_score = fit_lgb_direct(direct_train, labels[core_train], direct_val, val_y, direct_test, split_seed)
        fitted.append(("LightGBM direct stack", model, val_score, test_score, ["RWR_calibrated_logit"] + feature_names))
        model, val_score, test_score = fit_xgb_residual(x_train_full, labels[core_train], x_val_full, val_y, x_test_full, core_base, val_base, test_base, split_seed)
        fitted.append(("RWR + XGBoost residual", model, val_score, test_score, feature_names))
        model, val_score, test_score = fit_lgb_residual(x_train_full, labels[core_train], x_val_full, val_y, x_test_full, core_base, val_base, test_base, split_seed)
        fitted.append(("RWR + LightGBM residual", model, val_score, test_score, feature_names))
        model, val_score, test_score = fit_xgb_residual(
            x_train_full[:, topology_idx], labels[core_train], x_val_full[:, topology_idx], val_y, x_test_full[:, topology_idx], core_base, val_base, test_base, split_seed
        )
        fitted.append(("RWR + XGBoost residual (topology only)", model, val_score, test_score, [feature_names[i] for i in topology_idx]))
        model, val_score, test_score = fit_xgb_residual(
            x_train_full[:, expression_idx], labels[core_train], x_val_full[:, expression_idx], val_y, x_test_full[:, expression_idx], core_base, val_base, test_base, split_seed
        )
        fitted.append(("RWR + XGBoost residual (expression only)", model, val_score, test_score, [feature_names[i] for i in expression_idx]))

        for method, model, val_score, test_score, model_features in fitted:
            threshold = choose_threshold(val_y, val_score)
            row = metrics(test_y, test_score, threshold)
            metric_rows.append({"repeat": repeat, "fold": fold, "method": method, **row})
            oof_rows.extend(
                {"repeat": repeat, "fold": fold, "node_id": int(node), "method": method, "score": float(score), "threshold": threshold}
                for node, score in zip(outer_test, test_score)
            )
            if method in {"LightGBM direct stack", "XGBoost direct stack"}:
                if method.startswith("LightGBM"):
                    values = model.booster_.feature_importance(importance_type="gain")
                else:
                    values = model.feature_importances_
                values = np.asarray(values, dtype=float)
                values = values / values.sum() if values.sum() > 0 else values
                for feature, value in zip(model_features, values):
                    importance_rows.append({"repeat": repeat, "fold": fold, "method": method, "feature": feature, "importance": float(value)})
        audit_rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "core_train_n": len(core_train),
                "inner_val_n": len(inner_val),
                "outer_test_n": len(outer_test),
                "core_train_positives": int(labels[core_train].sum()),
                "inner_val_positives": int(labels[inner_val].sum()),
                "outer_test_positives": int(test_y.sum()),
                "rwr_archive_max_abs_difference": fold_diff,
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            }
        )
        print(f"repeat {repeat + 1}/{N_REPEATS}, fold {fold + 1}/{N_SPLITS} complete", flush=True)

    if max_rwr_difference > 1e-12:
        raise RuntimeError(f"Recomputed RWR does not reproduce the archive: max difference {max_rwr_difference}")
    for method in (METHOD_DEGREE, METHOD_GNN):
        archived_method = archived[archived["method"] == method].copy()
        if len(archived_method) != len(labels) * N_REPEATS:
            raise RuntimeError(f"Archived OOF predictions are incomplete for {method}")
        for (repeat, fold), block in archived_method.groupby(["repeat", "fold"], sort=True):
            node_ids = block["node_id"].to_numpy(dtype=int)
            scores = block["score"].to_numpy(dtype=float)
            threshold = float(block["threshold"].iloc[0])
            metric_rows.append(
                {"repeat": int(repeat), "fold": int(fold), "method": method, **metrics(labels[node_ids], scores, threshold)}
            )
        oof_rows.extend(archived_method[["repeat", "fold", "node_id", "method", "score", "threshold"]].to_dict("records"))
    fold_metrics = pd.DataFrame(metric_rows)
    oof = pd.DataFrame(oof_rows)
    importance = pd.DataFrame(importance_rows)
    audit = pd.DataFrame(audit_rows)
    numeric = ["ROC_AUC", "AUPRC", "Accuracy", "Precision", "Recall", "F1"]
    summary = fold_metrics.groupby("method")[numeric].agg(["mean", "std"]).reset_index()
    summary.columns = ["Method"] + [f"{metric}_{stat}" for metric, stat in summary.columns.tolist()[1:]]
    summary = summary.sort_values("AUPRC_mean", ascending=False)
    bootstrap = bootstrap_cluster_differences(oof, labels, args.bootstrap_replicates)

    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(args.output_dir / "oof_predictions.csv.gz", index=False, compression="gzip")
    summary.to_csv(args.output_dir / "benchmark_summary.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(args.output_dir / "gene_cluster_bootstrap_pairwise.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(args.output_dir / "feature_importance_by_fold.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(args.output_dir / "split_and_leakage_audit.csv", index=False, encoding="utf-8-sig")
    plot_results(oof, labels, summary, bootstrap, importance, args.output_dir)
    manifest = {
        "analysis": "Leakage-controlled RWR-XGBoost/LightGBM benchmark",
        "created": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "dataset": str(args.dataset.resolve()),
        "archived_oof": str(args.archived_oof.resolve()),
        "n_nodes": int(len(labels)),
        "n_positive": int(labels.sum()),
        "outer_validation": "RepeatedStratifiedKFold(5 folds x 3 repeats, random_state=42)",
        "inner_validation": "15% stratified split of each outer-training fold",
        "stacking_leakage_control": "RWR features for residual-learner training were generated by five-fold cross-fitting inside core_train; inner-validation and outer-test genes were never RWR seeds.",
        "rwr_reproduction_max_abs_difference": max_rwr_difference,
        "feature_groups": {
            "topology": topology_names,
            "cell_type_scaled": [f"CellType_{name}" for name in cell_type_names],
            "cell_type_pca": [f"PCA_{i + 1}" for i in range(5)],
        },
        "approx_betweenness_samples": args.betweenness_samples,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_unit": "gene cluster retained across all three repeats",
        "software": {
            "python": __import__("sys").version.split()[0],
            "xgboost": __import__("xgboost").__version__,
            "lightgbm": lgb.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "torch": torch.__version__,
        },
        "elapsed_seconds": time.time() - start,
    }
    save_json(manifest, args.output_dir / "analysis_manifest.json")
    print(summary[["Method", "ROC_AUC_mean", "AUPRC_mean"]].to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
