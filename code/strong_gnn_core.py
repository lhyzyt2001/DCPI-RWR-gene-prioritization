from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.special import logit
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.nn import APPNP, GATv2Conv, GCN2Conv


SEED = 42
N_SPLITS = 5
N_REPEATS = 3
INNER_VAL_FRACTION = 0.15
CLASSICAL_METHODS = ["RWR (STRING graph)", "XGBoost direct stack", "LightGBM direct stack"]
DEEP_METHODS = ["RWR-feature APPNP", "RWR-feature GCNII", "RWR-feature GATv2"]
ALL_METHODS = CLASSICAL_METHODS + DEEP_METHODS
COLORS = {
    "RWR (STRING graph)": "#0072B2",
    "XGBoost direct stack": "#E69F00",
    "LightGBM direct stack": "#009E73",
    "RWR-feature APPNP": "#CC79A7",
    "RWR-feature GCNII": "#D55E00",
    "RWR-feature GATv2": "#56B4E9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-controlled comparison of strong graph neural networks with RWR-guided boosting."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["appnp", "gcnii", "gatv2"], default=["appnp", "gcnii", "gatv2"])
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--betweenness-samples", type=int, default=128)
    parser.add_argument("--limit-splits", type=int, default=0, help="Pilot only; 0 runs all 15 outer folds.")
    parser.add_argument("--split-indices", nargs="*", type=int, default=[], help="Optional zero-based outer split indices.")
    parser.add_argument("--threads", type=int, default=12)
    return parser.parse_args()


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
        graph, k=min(samples, data.num_nodes), normalized=True, weight=None, seed=SEED
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
    adjacency = sp.coo_matrix((weight, (edge[0], edge[1])), shape=(data.num_nodes, data.num_nodes)).tocsc()
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


def crossfit_rwr(transition: sp.csc_matrix, labels: np.ndarray, indices: np.ndarray, split_seed: int) -> np.ndarray:
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
        raise RuntimeError("Cross-fitted RWR contains missing values.")
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class APPNPNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden)
        self.lin2 = nn.Linear(hidden, 2)
        self.propagate = APPNP(K=10, alpha=0.1, dropout=0.0, cached=True)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return self.propagate(x, edge_index, edge_weight)


class GCNIINet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, layers: int = 8, dropout: float = 0.5):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden)
        self.convs = nn.ModuleList(
            [
                GCN2Conv(
                    channels=hidden,
                    alpha=0.1,
                    theta=0.5,
                    layer=i + 1,
                    shared_weights=True,
                    cached=True,
                    normalize=True,
                )
                for i in range(layers)
            ]
        )
        self.output_layer = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x0 = F.relu(self.input_layer(x))
        h = x0
        for conv in self.convs:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(conv(h, x0, edge_index, edge_weight))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.output_layer(h)


class GATv2Net(nn.Module):
    def __init__(self, input_dim: int, hidden_per_head: int = 24, heads: int = 2, dropout: float = 0.4):
        super().__init__()
        self.conv1 = GATv2Conv(
            input_dim,
            hidden_per_head,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=1,
            add_self_loops=True,
            fill_value="mean",
        )
        self.conv2 = GATv2Conv(
            hidden_per_head * heads,
            2,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=1,
            add_self_loops=True,
            fill_value="mean",
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        edge_attr = edge_weight.view(-1, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index, edge_attr)


def make_model(name: str, input_dim: int) -> nn.Module:
    if name == "appnp":
        return APPNPNet(input_dim)
    if name == "gcnii":
        return GCNIINet(input_dim)
    if name == "gatv2":
        return GATv2Net(input_dim)
    raise ValueError(name)


def display_name(name: str) -> str:
    return {"appnp": "RWR-feature APPNP", "gcnii": "RWR-feature GCNII", "gatv2": "RWR-feature GATv2"}[name]


def train_model(
    model_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    core_train: np.ndarray,
    inner_val: np.ndarray,
    outer_test: np.ndarray,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    seed_everything(seed)
    model = make_model(model_name, x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)
    core_idx = torch.as_tensor(core_train, dtype=torch.long)
    val_idx = torch.as_tensor(inner_val, dtype=torch.long)
    test_idx = torch.as_tensor(outer_test, dtype=torch.long)
    train_y = y[core_idx]
    counts = torch.bincount(train_y, minlength=2).float()
    class_weight = counts.sum() / (2.0 * counts.clamp_min(1.0))
    best_state = None
    best_val = -np.inf
    best_epoch = 0
    stale = 0
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edge_index, edge_weight)
        loss = F.cross_entropy(logits[core_idx], train_y, weight=class_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prob = torch.softmax(model(x, edge_index, edge_weight)[val_idx], dim=1)[:, 1].cpu().numpy()
        val_ap = average_precision_score(y[val_idx].cpu().numpy(), val_prob)
        if val_ap > best_val + 1e-5:
            best_val = float(val_ap)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"No valid state for {model_name}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probability = torch.softmax(model(x, edge_index, edge_weight), dim=1)[:, 1]
    val_prob = probability[val_idx].cpu().numpy()
    test_prob = probability[test_idx].cpu().numpy()
    audit = {
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_inner_val_auprc": best_val,
        "training_seconds": time.time() - started,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "positive_class_weight": float(class_weight[1]),
    }
    return val_prob, test_prob, audit


def compute_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    return {"roc_auc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score))}


def prepare_static_features(data, betweenness_samples: int) -> tuple[np.ndarray, list[str]]:
    degree, degree_names = network_degree_features(data)
    centrality, centrality_names = string_centralities(data, betweenness_samples)
    return np.column_stack([degree, centrality]).astype(np.float32), degree_names + centrality_names


def run_nested(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    labels = data.y.cpu().numpy().astype(int)
    raw_expression = data.x.cpu().numpy().astype(np.float32)
    topology, topology_names = prepare_static_features(data, args.betweenness_samples)
    transition = prepare_transition(data)
    edge_index = data.edge_index_ppi.cpu().long().contiguous()
    edge_weight = data.edge_weight_ppi.cpu().float().contiguous()
    y_tensor = data.y.cpu().long()
    archived = pd.read_csv(args.oof)
    split_source = archived[archived["method"] == "RWR (STRING graph)"][
        ["repeat", "fold", "node_id", "score"]
    ].copy()
    if len(split_source) != len(labels) * N_REPEATS:
        raise RuntimeError("RWR OOF table must contain one row per gene and repeat.")

    metrics_rows: list[dict] = []
    oof_rows: list[dict] = []
    audit_rows: list[dict] = []
    split_counter = 0
    selected_splits = set(args.split_indices)
    for repeat in range(N_REPEATS):
        for fold in range(N_SPLITS):
            if args.limit_splits and split_counter >= args.limit_splits:
                break
            if selected_splits and split_counter not in selected_splits:
                split_counter += 1
                continue
            split_seed = SEED + split_counter
            test_block = split_source[(split_source["repeat"] == repeat) & (split_source["fold"] == fold)]
            outer_test = np.sort(test_block["node_id"].to_numpy(dtype=int))
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

            core_rwr_oof = crossfit_rwr(transition, labels, core_train, split_seed)
            seed_nodes = core_train[labels[core_train] == 1]
            all_outer_rwr = run_rwr(transition, len(labels), seed_nodes)
            calibrator = fit_rwr_calibrator(core_rwr_oof, labels[core_train])
            rwr_logit = np.zeros(len(labels), dtype=np.float32)
            rwr_logit[core_train] = calibrated_logit(calibrator, core_rwr_oof)
            rwr_logit[inner_val] = calibrated_logit(calibrator, all_outer_rwr[inner_val])
            rwr_logit[outer_test] = calibrated_logit(calibrator, all_outer_rwr[outer_test])
            direct = np.column_stack([rwr_logit, feature_all]).astype(np.float32)
            neural_scaler = StandardScaler().fit(direct[core_train])
            direct_scaled = np.nan_to_num(neural_scaler.transform(direct), nan=0.0, posinf=10.0, neginf=-10.0)
            x_tensor = torch.as_tensor(direct_scaled, dtype=torch.float32)

            archived_fold = archived[
                (archived["repeat"] == repeat)
                & (archived["fold"] == fold)
                & (archived["method"].isin(CLASSICAL_METHODS))
            ]
            for method in CLASSICAL_METHODS:
                block = archived_fold[archived_fold["method"] == method].sort_values("node_id")
                if not np.array_equal(block["node_id"].to_numpy(dtype=int), outer_test):
                    raise RuntimeError(f"Outer-test nodes are inconsistent for {method}, repeat {repeat}, fold {fold}.")
                score = block["score"].to_numpy(dtype=float)
                row = compute_metrics(labels[outer_test], score)
                metrics_rows.append({"repeat": repeat, "fold": fold, "method": method, **row})
                oof_rows.extend(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "node_id": int(node),
                        "method": method,
                        "score": float(value),
                    }
                    for node, value in zip(outer_test, score)
                )

            for model_name in args.models:
                method = display_name(model_name)
                val_score, test_score, audit = train_model(
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
                    split_seed + {"appnp": 1000, "gcnii": 2000, "gatv2": 3000}[model_name],
                )
                row = compute_metrics(labels[outer_test], test_score)
                metrics_rows.append({"repeat": repeat, "fold": fold, "method": method, **row})
                oof_rows.extend(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "node_id": int(node),
                        "method": method,
                        "score": float(value),
                    }
                    for node, value in zip(outer_test, test_score)
                )
                audit_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "method": method,
                        "core_train_n": len(core_train),
                        "inner_val_n": len(inner_val),
                        "outer_test_n": len(outer_test),
                        "outer_test_labels_used_for_training": False,
                        "outer_test_nodes_used_as_rwr_seeds": False,
                        **audit,
                    }
                )
            split_counter += 1
        if args.limit_splits and split_counter >= args.limit_splits:
            break
    return pd.DataFrame(metrics_rows), pd.DataFrame(oof_rows), pd.DataFrame(audit_rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    order = [method for method in ALL_METHODS if method in set(metrics["method"])]
    return (
        metrics.groupby("method", as_index=False)
        .agg(
            outer_folds=("auprc", "size"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
        )
        .set_index("method")
        .loc[order]
        .reset_index()
    )


def cluster_bootstrap(oof: pd.DataFrame, labels: np.ndarray, replicates: int) -> pd.DataFrame:
    methods = list(oof["method"].drop_duplicates())
    wide = oof.pivot(index=["node_id", "repeat"], columns="method", values="score").reset_index()
    repeats = sorted(wide["repeat"].unique())
    if len(repeats) != N_REPEATS:
        return pd.DataFrame()
    matrices = {int(r): wide[wide["repeat"] == r].set_index("node_id").sort_index() for r in repeats}
    node_ids = matrices[int(repeats[0])].index.to_numpy(dtype=int)
    y = labels[node_ids]
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    pairs = []
    for method in methods:
        if method != "RWR (STRING graph)":
            pairs.append((method, "RWR (STRING graph)"))
        if method in DEEP_METHODS:
            pairs.append((method, "LightGBM direct stack"))
    values = {pair: np.empty(replicates, dtype=float) for pair in pairs}
    rng = np.random.default_rng(SEED)
    arrays = {r: {m: matrices[r][m].to_numpy() for m in methods} for r in matrices}
    for b in range(replicates):
        sampled = np.concatenate(
            [rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        )
        sampled_y = y[sampled]
        aps = {
            method: float(
                np.mean([average_precision_score(sampled_y, arrays[r][method][sampled]) for r in repeats])
            )
            for method in methods
        }
        for pair in pairs:
            values[pair][b] = aps[pair[0]] - aps[pair[1]]
    rows = []
    for (method, reference), distribution in values.items():
        p_value = min(
            1.0,
            2.0
            * min(
                (np.sum(distribution <= 0) + 1) / (replicates + 1),
                (np.sum(distribution >= 0) + 1) / (replicates + 1),
            ),
        )
        rows.append(
            {
                "method": method,
                "reference": reference,
                "median_delta_auprc": float(np.median(distribution)),
                "ci_lower": float(np.quantile(distribution, 0.025)),
                "ci_upper": float(np.quantile(distribution, 0.975)),
                "p_two_sided": float(p_value),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "gene cluster retained across all three repeats",
            }
        )
    return pd.DataFrame(rows)


def paired_fold_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    wide = metrics.pivot(index=["repeat", "fold"], columns="method", values="auprc")
    rows = []
    for method in wide.columns:
        for reference in ["RWR (STRING graph)", "LightGBM direct stack"]:
            if method == reference or reference not in wide.columns:
                continue
            delta = wide[method] - wide[reference]
            result = wilcoxon(delta, alternative="two-sided", zero_method="pratt")
            rows.append(
                {
                    "method": method,
                    "reference": reference,
                    "mean_fold_delta_auprc": float(delta.mean()),
                    "median_fold_delta_auprc": float(delta.median()),
                    "folds_positive": int((delta > 0).sum()),
                    "outer_folds": len(delta),
                    "wilcoxon_p_two_sided": float(result.pvalue),
                    "note": "Secondary test; repeated-CV folds are correlated.",
                }
            )
    return pd.DataFrame(rows)


def plot_summary(metrics: pd.DataFrame, bootstrap: pd.DataFrame, output_dir: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    order = [method for method in ALL_METHODS if method in set(metrics["method"])]
    fig, (ax_fold, ax_boot) = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"wspace": 0.5})
    wide = metrics.pivot(index=["repeat", "fold"], columns="method", values="auprc").loc[:, order]
    x = np.arange(len(order))
    for i, method in enumerate(order):
        values = wide[method].to_numpy()
        jitter = np.linspace(-0.055, 0.055, len(values))
        ax_fold.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=12,
            facecolor="white",
            edgecolor=COLORS[method],
            linewidth=0.75,
            zorder=2,
        )
        ax_fold.errorbar(
            i,
            values.mean(),
            yerr=values.std(ddof=1),
            fmt="o",
            color=COLORS[method],
            capsize=3,
            markersize=5,
            linewidth=1.2,
            zorder=3,
        )
    short = [m.replace(" (STRING graph)", "").replace(" direct stack", "").replace("RWR-feature ", "") for m in order]
    ax_fold.set_xticks(x, short, rotation=35, ha="right")
    ax_fold.set_ylabel("AUPRC")
    ax_fold.set_title("Repeated outer-fold performance")
    ax_fold.text(-0.18, 1.06, "a", transform=ax_fold.transAxes, fontsize=11, fontweight="bold", va="top")
    ax_fold.grid(axis="y", color="#E5E5E5", linewidth=0.7)

    selected = bootstrap[bootstrap["method"].isin(DEEP_METHODS) & (bootstrap["reference"] == "LightGBM direct stack")].copy()
    selected = selected.iloc[::-1].reset_index(drop=True)
    for i, row in selected.iterrows():
        estimate, low, high = row["median_delta_auprc"], row["ci_lower"], row["ci_upper"]
        ax_boot.errorbar(
            estimate,
            i,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=COLORS[row["method"]],
            capsize=3,
            linewidth=1.4,
            markersize=5,
        )
    ax_boot.axvline(0, color="#666666", linestyle="--", linewidth=1)
    ax_boot.set_yticks(np.arange(len(selected)), [m.replace("RWR-feature ", "") + " - LightGBM" for m in selected["method"]])
    ax_boot.set_xlabel("Median difference in AUPRC (95% CI)")
    ax_boot.set_title("Gene-cluster bootstrap")
    ax_boot.text(-0.24, 1.06, "b", transform=ax_boot.transAxes, fontsize=11, fontweight="bold", va="top")
    ax_boot.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    for ax in (ax_fold, ax_boot):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.28, top=0.86)
    fig.savefig(output_dir / "strong_GNN_comparison.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "strong_GNN_comparison.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    metrics, oof, audit = run_nested(args)
    summary = summarize(metrics)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    labels = data.y.cpu().numpy().astype(int)
    partial_run = bool(args.limit_splits or args.split_indices)
    bootstrap = cluster_bootstrap(oof, labels, args.bootstrap_replicates) if not partial_run else pd.DataFrame()
    fold_tests = paired_fold_tests(metrics) if not partial_run else pd.DataFrame()

    metrics.to_csv(args.output_dir / "deep_benchmark_fold_metrics.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(args.output_dir / "deep_benchmark_oof_predictions.csv.gz", index=False, compression="gzip")
    audit.to_csv(args.output_dir / "deep_training_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "deep_benchmark_summary.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(args.output_dir / "deep_gene_cluster_bootstrap.csv", index=False, encoding="utf-8-sig")
    fold_tests.to_csv(args.output_dir / "deep_paired_fold_tests.csv", index=False, encoding="utf-8-sig")
    if not partial_run:
        plot_summary(metrics, bootstrap, args.output_dir)
    manifest = {
        "models": [display_name(model) for model in args.models],
        "excluded_model": "legacy multiview GCN comparator",
        "outer_folds": int(summary["outer_folds"].max()),
        "transductive": True,
        "message_passing_graph": "high-confidence STRING functional-association graph",
        "node_features": "cross-fitted calibrated RWR logit plus train-fitted topology, standardized cell-type expression and PCA features",
        "split_control": "outer-test labels excluded from fitting, early stopping, scaling, PCA and RWR seeds",
        "hyperparameter_policy": "architecture hyperparameters fixed before outer-test evaluation; epoch selected by inner-validation AUPRC",
        "training_random_seeds_per_outer_fold": 1,
        "runtime_seconds": time.time() - started,
        "software": {"torch": torch.__version__},
        "limitations": [
            "One optimization seed per outer fold; stochastic sensitivity should be added before a final high-stakes submission.",
            "Model-family comparison remains internal and requires locked temporal or external validation.",
        ],
    }
    (args.output_dir / "deep_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    if not bootstrap.empty:
        print(bootstrap.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
