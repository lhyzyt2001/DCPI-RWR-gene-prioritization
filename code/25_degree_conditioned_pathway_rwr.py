from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import binomtest, rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


SEED = 20261001
CANCERS = ["LUAD", "BRCA", "CRC", "PRAD", "MEL", "OV"]
VIEW_NAMES = ["STRING", "CellType", "Pathway"]


@dataclass(frozen=True)
class Config:
    restart: float
    blend: float
    weight_string: float
    weight_celltype: float
    weight_pathway: float

    @property
    def config_id(self) -> str:
        return (
            f"r{self.restart:.2f}_b{self.blend:.2f}_"
            f"w{self.weight_string:.2f}-{self.weight_celltype:.2f}-{self.weight_pathway:.2f}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_labels(path: Path, mapping: dict[str, int], size: int) -> np.ndarray:
    frame = pd.read_csv(path)
    labels = np.zeros(size, dtype=np.int8)
    nodes = [mapping[str(symbol).upper()] for symbol in frame["symbol"] if str(symbol).upper() in mapping]
    labels[np.asarray(nodes, dtype=int)] = 1
    return labels


def transition(edge: np.ndarray, weight: np.ndarray, size: int) -> sp.csc_matrix:
    adjacency = sp.coo_matrix((weight, (edge[0], edge[1])), shape=(size, size)).tocsc()
    degree = np.asarray(adjacency.sum(axis=0)).ravel()
    degree[degree == 0] = 1.0
    return adjacency @ sp.diags(1.0 / degree)


def rwr(matrix: sp.csc_matrix, seeds: np.ndarray, size: int, restart: float) -> np.ndarray:
    initial = np.zeros(size, dtype=float)
    initial[np.asarray(seeds, dtype=int)] = 1.0 / max(len(seeds), 1)
    score = initial.copy()
    for _ in range(300):
        updated = (1.0 - restart) * matrix.dot(score) + restart * initial
        if np.linalg.norm(updated - score, 1) < 1e-10:
            return np.asarray(updated).ravel()
        score = updated
    return np.asarray(score).ravel()


def degree_bins(degree: np.ndarray, requested: int = 20) -> np.ndarray:
    log_degree = np.log1p(degree.astype(float))
    try:
        bins = pd.qcut(log_degree, q=requested, labels=False, duplicates="drop")
        return np.asarray(bins, dtype=int)
    except ValueError:
        return np.zeros(len(degree), dtype=int)


def percentile(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def conditioned_percentile(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values), dtype=float)
    for bin_id in np.unique(bins):
        selected = np.flatnonzero(bins == bin_id)
        output[selected] = rankdata(values[selected], method="average") / len(selected)
    return output


def components(
    transitions: list[sp.csc_matrix],
    bins: list[np.ndarray],
    seeds: np.ndarray,
    size: int,
    restart: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = [rwr(matrix, seeds, size, restart) for matrix in transitions]
    raw_string = percentile(raw[0])
    conditioned = [conditioned_percentile(score, view_bins) for score, view_bins in zip(raw, bins)]
    return raw_string, conditioned[0], conditioned[1], conditioned[2]


def score_config(parts: tuple[np.ndarray, ...], config: Config) -> np.ndarray:
    raw_string, string_conditioned, cell_conditioned, pathway_conditioned = parts
    multiview = (
        config.weight_string * string_conditioned
        + config.weight_celltype * cell_conditioned
        + config.weight_pathway * pathway_conditioned
    )
    return (1.0 - config.blend) * raw_string + config.blend * multiview


def configs() -> list[Config]:
    weights = []
    for string_steps in range(5):
        for cell_steps in range(5 - string_steps):
            pathway_steps = 4 - string_steps - cell_steps
            weights.append((string_steps / 4, cell_steps / 4, pathway_steps / 4))
    result = []
    for restart in (0.15, 0.30, 0.50):
        result.append(Config(restart, 0.0, 1.0, 0.0, 0.0))
        for blend in (0.25, 0.50, 0.75, 1.0):
            for w_string, w_cell, w_pathway in weights:
                result.append(Config(restart, blend, w_string, w_cell, w_pathway))
    return result


def safe_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def tune_historical(
    labels: np.ndarray,
    transitions: list[sp.csc_matrix],
    bins: list[np.ndarray],
    string_degree: np.ndarray,
    candidates: list[Config],
) -> tuple[pd.DataFrame, Config, Config, Config, Config]:
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    rows = []
    cache: dict[tuple[int, float], tuple[np.ndarray, ...]] = {}
    for split_index, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        seeds = train[labels[train] == 1]
        degree_test_rank = rankdata(np.log1p(string_degree[test]), method="average")
        for restart in sorted({config.restart for config in candidates}):
            cache[(split_index, restart)] = components(transitions, bins, seeds, len(labels), restart)
        for config in candidates:
            score = score_config(cache[(split_index, config.restart)], config)[test]
            roc, auprc = safe_metrics(labels[test], score)
            top_n = min(100, len(test))
            top = np.argsort(-score, kind="mergesort")[:top_n]
            precision_at_100 = float(labels[test][top].mean())
            hub_cutoff = float(np.quantile(string_degree, 0.95))
            top100_hub_fraction = float((string_degree[test][top] >= hub_cutoff).mean())
            correlation = float(spearmanr(score, degree_test_rank).statistic)
            rows.append(
                {
                    "split": split_index,
                    "repeat": split_index // 5,
                    "fold": split_index % 5,
                    "config_id": config.config_id,
                    **asdict(config),
                    "ROC_AUC": roc,
                    "AUPRC": auprc,
                    "precision_at_100": precision_at_100,
                    "top100_hub_fraction": top100_hub_fraction,
                    "score_degree_spearman": correlation,
                    "train_positives": int(labels[train].sum()),
                    "test_positives": int(labels[test].sum()),
                }
            )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby(["config_id", "restart", "blend", "weight_string", "weight_celltype", "weight_pathway"], as_index=False)
        .agg(
            mean_ROC_AUC=("ROC_AUC", "mean"),
            sd_ROC_AUC=("ROC_AUC", "std"),
            mean_AUPRC=("AUPRC", "mean"),
            sd_AUPRC=("AUPRC", "std"),
            mean_precision_at_100=("precision_at_100", "mean"),
            sd_precision_at_100=("precision_at_100", "std"),
            mean_top100_hub_fraction=("top100_hub_fraction", "mean"),
            mean_abs_score_degree_spearman=("score_degree_spearman", lambda values: np.mean(np.abs(values))),
        )
    )
    performance_row = summary.sort_values(
        ["mean_AUPRC", "mean_abs_score_degree_spearman"], ascending=[False, True]
    ).iloc[0]
    discovery_row = summary.sort_values(
        ["mean_precision_at_100", "mean_AUPRC", "mean_abs_score_degree_spearman"],
        ascending=[False, False, True],
    ).iloc[0]
    reference_id = Config(0.30, 0.0, 1.0, 0.0, 0.0).config_id
    reference_row = summary[summary.config_id == reference_id].iloc[0]
    constrained_eligible = summary[
        summary.mean_precision_at_100 >= 0.95 * float(reference_row.mean_precision_at_100)
    ]
    constrained_row = constrained_eligible.sort_values(
        ["mean_top100_hub_fraction", "mean_precision_at_100", "mean_AUPRC"],
        ascending=[True, False, False],
    ).iloc[0]
    threshold = 0.95 * float(performance_row["mean_AUPRC"])
    eligible = summary[summary["mean_AUPRC"] >= threshold]
    bias_row = eligible.sort_values(
        ["mean_abs_score_degree_spearman", "mean_AUPRC"], ascending=[True, False]
    ).iloc[0]

    def row_config(row: pd.Series) -> Config:
        return Config(
            float(row.restart),
            float(row.blend),
            float(row.weight_string),
            float(row.weight_celltype),
            float(row.weight_pathway),
        )

    return (
        summary,
        row_config(performance_row),
        row_config(discovery_row),
        row_config(constrained_row),
        row_config(bias_row),
    )


def evaluation_rows(
    cancer: str,
    historical: np.ndarray,
    current: np.ndarray,
    transitions: list[sp.csc_matrix],
    bins: list[np.ndarray],
    string_degree: np.ndarray,
    performance_config: Config,
    discovery_config: Config,
    constrained_config: Config,
    bias_config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    seeds = np.flatnonzero(historical == 1)
    candidate_nodes = np.flatnonzero(historical == 0)
    incident = ((current == 1) & (historical == 0)).astype(np.int8)
    configurations = {
        "RWR-STRING": Config(0.30, 0.0, 1.0, 0.0, 0.0),
        "DCPI-RWR performance": performance_config,
        "DCPI-RWR discovery": discovery_config,
        "DCPI-RWR constrained Top100": constrained_config,
        "DCPI-RWR bias-aware": bias_config,
        "Degree-conditioned STRING": Config(bias_config.restart, 1.0, 1.0, 0.0, 0.0),
        "Equal-weight conditioned multiview": Config(bias_config.restart, 1.0, 1 / 3, 1 / 3, 1 / 3),
    }
    component_cache = {
        restart: components(transitions, bins, seeds, len(historical), restart)
        for restart in sorted({config.restart for config in configurations.values()})
    }
    score_map = {
        method: score_config(component_cache[config.restart], config)
        for method, config in configurations.items()
    }
    metric_rows = []
    top_rows = []
    hub_cutoff = float(np.quantile(string_degree, 0.95))
    for method, full_score in score_map.items():
        labels = incident[candidate_nodes]
        score = full_score[candidate_nodes]
        roc, auprc = safe_metrics(labels, score)
        metric_rows.append(
            {
                "cancer": cancer,
                "method": method,
                "historical_positives": int(historical.sum()),
                "current_positives": int(current.sum()),
                "incident_positives": int(incident.sum()),
                "candidate_universe": len(candidate_nodes),
                "ROC_AUC": roc,
                "AUPRC": auprc,
                "score_degree_spearman": float(spearmanr(score, string_degree[candidate_nodes]).statistic),
            }
        )
        order = candidate_nodes[np.argsort(-score, kind="mergesort")]
        for k in (20, 50, 100, 200, 500, 1000):
            selected = order[:k]
            top_rows.append(
                {
                    "cancer": cancer,
                    "method": method,
                    "K": k,
                    "incident_hits": int(incident[selected].sum()),
                    "incident_recall": float(incident[selected].sum() / max(incident.sum(), 1)),
                    "precision_at_K": float(incident[selected].mean()),
                    "hub_fraction": float((string_degree[selected] >= hub_cutoff).mean()),
                    "median_STRING_degree": float(np.median(string_degree[selected])),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(top_rows), score_map


def stratified_bootstrap(
    labels: np.ndarray,
    reference: np.ndarray,
    alternatives: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    differences = {name: [] for name in alternatives}
    for _ in range(replicates):
        sampled = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        sampled_labels = labels[sampled]
        reference_ap = average_precision_score(sampled_labels, reference[sampled])
        for name, score in alternatives.items():
            differences[name].append(
                average_precision_score(sampled_labels, score[sampled]) - reference_ap
            )
    rows = []
    for name, values in differences.items():
        array = np.asarray(values)
        p_value = 2 * min((np.sum(array <= 0) + 1) / (len(array) + 1), (np.sum(array >= 0) + 1) / (len(array) + 1))
        rows.append(
            {
                "method": name,
                "reference": "RWR-STRING",
                "median_delta_AUPRC": float(np.median(array)),
                "ci_lower": float(np.quantile(array, 0.025)),
                "ci_upper": float(np.quantile(array, 0.975)),
                "p_two_sided": float(min(p_value, 1.0)),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "gene; positives and unlabelled genes sampled separately",
            }
        )
    return pd.DataFrame(rows)


def degree_matched_seed_null(
    historical: np.ndarray,
    current: np.ndarray,
    transitions: list[sp.csc_matrix],
    bins: list[np.ndarray],
    string_degree_bins: np.ndarray,
    config: Config,
    replicates: int,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED + 9000)
    true_seeds = np.flatnonzero(historical == 1)
    candidate_nodes = np.flatnonzero(historical == 0)
    incident = ((current == 1) & (historical == 0)).astype(np.int8)
    true_parts = components(transitions, bins, true_seeds, len(historical), config.restart)
    observed_score = score_config(true_parts, config)[candidate_nodes]
    observed = float(average_precision_score(incident[candidate_nodes], observed_score))
    eligible = np.flatnonzero(current == 0)
    null_rows = []
    for replicate in range(replicates):
        sampled_seeds = []
        for bin_id in np.unique(string_degree_bins[true_seeds]):
            count = int((string_degree_bins[true_seeds] == bin_id).sum())
            pool = eligible[string_degree_bins[eligible] == bin_id]
            sampled_seeds.extend(rng.choice(pool, size=count, replace=len(pool) < count).tolist())
        parts = components(
            transitions, bins, np.asarray(sampled_seeds, dtype=int), len(historical), config.restart
        )
        null_score = score_config(parts, config)[candidate_nodes]
        null_rows.append(
            {
                "replicate": replicate,
                "AUPRC": float(average_precision_score(incident[candidate_nodes], null_score)),
            }
        )
    null = pd.DataFrame(null_rows)
    summary = {
        "observed_AUPRC": observed,
        "null_mean_AUPRC": float(null["AUPRC"].mean()),
        "null_ci_lower": float(null["AUPRC"].quantile(0.025)),
        "null_ci_upper": float(null["AUPRC"].quantile(0.975)),
        "empirical_p_greater_equal": float((1 + (null["AUPRC"] >= observed).sum()) / (replicates + 1)),
        "replicates": replicates,
        "null": "Random seed sets matched to historical LUAD seeds by STRING-degree bin and sampled from genes without 26.06 evidence",
    }
    return null, summary


def plot_results(metrics: pd.DataFrame, topk: pd.DataFrame, tuning: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )
    colors = {
        "RWR-STRING": "#0072B2",
        "DCPI-RWR performance": "#D55E00",
        "DCPI-RWR discovery": "#56B4E9",
        "DCPI-RWR constrained Top100": "#CC79A7",
        "DCPI-RWR bias-aware": "#009E73",
        "Degree-conditioned STRING": "#CC79A7",
        "Equal-weight conditioned multiview": "#E69F00",
    }
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.9))
    cancers = [cancer for cancer in CANCERS if cancer in set(metrics["cancer"])]
    methods = ["RWR-STRING", "DCPI-RWR performance", "DCPI-RWR constrained Top100", "DCPI-RWR bias-aware"]
    x = np.arange(len(cancers))
    for method in methods:
        block = metrics[metrics.method == method].set_index("cancer").loc[cancers]
        axes[0].plot(x, block.AUPRC, marker="o", linewidth=1.4, color=colors[method], label=method)
    axes[0].set(xticks=x, xticklabels=cancers, ylabel="Cross-release incident-gene AUPRC")
    axes[0].set_title("a  Locked 26.03-to-26.06 evaluation", loc="left")
    axes[0].grid(color="#E5E5E5", linewidth=0.7)
    block = topk[(topk.K == 100) & topk.method.isin(methods)]
    width = 0.19
    for index, method in enumerate(methods):
        values = block[block.method == method].set_index("cancer").loc[cancers, "hub_fraction"]
        axes[1].bar(x + (index - 1.5) * width, values, width=width, color=colors[method], label=method)
    axes[1].set(
        xticks=x,
        xticklabels=cancers,
        ylabel="STRING top-5% hub fraction in top 100",
        ylim=(0, 1),
    )
    axes[1].set_title("b  Candidate centrality", loc="left")
    axes[1].grid(axis="y", color="#E5E5E5", linewidth=0.7)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.14, 1, 1))
    figure.savefig(output / "DCPI_RWR_multicancer_performance_and_bias.pdf", bbox_inches="tight")
    figure.savefig(output / "DCPI_RWR_multicancer_performance_and_bias.png", dpi=600, bbox_inches="tight")
    plt.close(figure)

    luad = tuning[tuning.cancer == "LUAD"]
    figure, axis = plt.subplots(figsize=(4.4, 3.5))
    axis.scatter(
        luad.mean_abs_score_degree_spearman,
        luad.mean_AUPRC,
        s=14,
        color="#BDBDBD",
        alpha=0.7,
        linewidth=0,
    )
    selected = luad[luad.selection != "candidate"]
    label_offsets = {
        "performance": (-6, 9),
        "discovery": (6, -13),
        "constrained-Top100": (6, 6),
        "bias-aware": (-6, -14),
    }
    horizontal_alignment = {
        "performance": "right",
        "discovery": "left",
        "constrained-Top100": "left",
        "bias-aware": "right",
    }
    for row in selected.itertuples(index=False):
        color = {"performance": "#D55E00", "discovery": "#56B4E9", "constrained-Top100": "#CC79A7", "bias-aware": "#009E73"}[row.selection]
        axis.scatter(row.mean_abs_score_degree_spearman, row.mean_AUPRC, s=45, color=color, edgecolor="white")
        axis.annotate(
            row.selection,
            (row.mean_abs_score_degree_spearman, row.mean_AUPRC),
            xytext=label_offsets[row.selection],
            textcoords="offset points",
            ha=horizontal_alignment[row.selection],
            va="center",
            fontsize=7,
        )
    axis.set(xlabel="Mean |Spearman(score, STRING degree)|", ylabel="Historical-CV AUPRC")
    axis.set_title("LUAD historical-development Pareto surface")
    axis.grid(color="#E5E5E5", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output / "DCPI_RWR_LUAD_tuning_pareto.pdf", bbox_inches="tight")
    figure.savefig(output / "DCPI_RWR_LUAD_tuning_pareto.png", dpi=600, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical-only tuning of degree-conditioned pathway-integrated RWR.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cancers", nargs="+", choices=CANCERS, default=CANCERS)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--null-permutations", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    mapping_raw = json.loads(args.mapping.read_text(encoding="utf-8-sig"))
    mapping = {str(symbol).upper(): int(node) for symbol, node in mapping_raw.items()}
    reverse_mapping = {node: symbol for symbol, node in mapping.items()}
    size = int(data.num_nodes)
    view_specifications = [
        ("edge_index_ppi", "edge_weight_ppi"),
        ("edge_index_coexp", "edge_weight_coexp"),
        ("edge_index_pathway", "edge_weight_pathway"),
    ]
    transitions = []
    degrees = []
    for edge_name, weight_name in view_specifications:
        edge = getattr(data, edge_name).cpu().numpy()
        weight = getattr(data, weight_name).cpu().numpy()
        transitions.append(transition(edge, weight, size))
        degrees.append(np.bincount(edge[0], minlength=size))
    bins = [degree_bins(degree) for degree in degrees]
    candidate_configs = configs()
    all_metrics = []
    all_topk = []
    all_tuning = []
    all_bootstrap = []
    all_top100_genes = []
    selections = {}
    luad_objects = None

    for cancer_index, cancer in enumerate(args.cancers):
        historical_path = args.labels_root / cancer / "historical_26.03.csv"
        current_path = args.labels_root / cancer / "future_26.06.csv"
        historical = read_labels(historical_path, mapping, size)
        current = read_labels(current_path, mapping, size)
        tuning, performance_config, discovery_config, constrained_config, bias_config = tune_historical(
            historical, transitions, bins, degrees[0], candidate_configs
        )
        tuning.insert(0, "cancer", cancer)
        tuning["selection"] = "candidate"
        tuning.loc[tuning.config_id == performance_config.config_id, "selection"] = "performance"
        tuning.loc[tuning.config_id == discovery_config.config_id, "selection"] = "discovery"
        tuning.loc[tuning.config_id == constrained_config.config_id, "selection"] = "constrained-Top100"
        tuning.loc[tuning.config_id == bias_config.config_id, "selection"] = "bias-aware"
        all_tuning.append(tuning)
        metrics, topk, score_map = evaluation_rows(
            cancer,
            historical,
            current,
            transitions,
            bins,
            degrees[0],
            performance_config,
            discovery_config,
            constrained_config,
            bias_config,
        )
        all_metrics.append(metrics)
        all_topk.append(topk)
        candidate_nodes = np.flatnonzero(historical == 0)
        incident_full = ((current == 1) & (historical == 0)).astype(np.int8)
        incident = incident_full[candidate_nodes]
        for method, full_score in score_map.items():
            order = candidate_nodes[np.argsort(-full_score[candidate_nodes], kind="mergesort")[:100]]
            for rank, node in enumerate(order, 1):
                all_top100_genes.append(
                    {
                        "cancer": cancer,
                        "method": method,
                        "rank": rank,
                        "node_id": int(node),
                        "Gene": reverse_mapping.get(int(node), str(node)),
                        "incident_26.06": int(incident_full[node]),
                        "STRING_degree": int(degrees[0][node]),
                    }
                )
        bootstrap = stratified_bootstrap(
            incident,
            score_map["RWR-STRING"][candidate_nodes],
            {
                "DCPI-RWR performance": score_map["DCPI-RWR performance"][candidate_nodes],
                "DCPI-RWR discovery": score_map["DCPI-RWR discovery"][candidate_nodes],
                "DCPI-RWR constrained Top100": score_map["DCPI-RWR constrained Top100"][candidate_nodes],
                "DCPI-RWR bias-aware": score_map["DCPI-RWR bias-aware"][candidate_nodes],
            },
            args.bootstrap,
            SEED + cancer_index,
        )
        bootstrap.insert(0, "cancer", cancer)
        all_bootstrap.append(bootstrap)
        selections[cancer] = {
            "performance": asdict(performance_config),
            "performance_config_id": performance_config.config_id,
            "discovery": asdict(discovery_config),
            "discovery_config_id": discovery_config.config_id,
            "discovery_selection_rule": "Highest historical-CV mean Precision@100; ties resolved by mean AUPRC then lower absolute score-degree Spearman correlation",
            "constrained_top100": asdict(constrained_config),
            "constrained_top100_config_id": constrained_config.config_id,
            "constrained_top100_selection_rule": "Lowest historical-CV mean top-100 STRING-hub fraction among configurations retaining at least 95% of the RWR historical-CV mean Precision@100",
            "bias_aware": asdict(bias_config),
            "bias_aware_config_id": bias_config.config_id,
            "bias_selection_rule": "Lowest mean absolute score-degree Spearman among configurations with historical-CV mean AUPRC >= 95% of the performance-optimal configuration",
        }
        if cancer == "LUAD":
            luad_objects = (historical, current, score_map, performance_config, discovery_config, constrained_config, bias_config)
        print(cancer, metrics[["method", "AUPRC", "score_degree_spearman"]].to_dict("records"), flush=True)

    metrics = pd.concat(all_metrics, ignore_index=True)
    topk = pd.concat(all_topk, ignore_index=True)
    tuning = pd.concat(all_tuning, ignore_index=True)
    bootstrap = pd.concat(all_bootstrap, ignore_index=True)
    top100_genes = pd.DataFrame(all_top100_genes)
    metrics.to_csv(args.output / "DCPI_RWR_multicancer_cross_release_metrics.csv", index=False, encoding="utf-8-sig")
    topk.to_csv(args.output / "DCPI_RWR_multicancer_topk_and_hubs.csv", index=False, encoding="utf-8-sig")
    tuning.to_csv(args.output / "DCPI_RWR_historical_tuning_surface.csv.gz", index=False, compression="gzip")
    bootstrap.to_csv(args.output / "DCPI_RWR_multicancer_bootstrap.csv", index=False, encoding="utf-8-sig")
    top100_genes.to_csv(args.output / "DCPI_RWR_multicancer_top100_genes.csv", index=False, encoding="utf-8-sig")
    paired_rows = []
    for alternative_name in ["DCPI-RWR performance", "DCPI-RWR constrained Top100"]:
        pooled_method_only_hits = 0
        pooled_reference_only_hits = 0
        for cancer in args.cancers:
            block = top100_genes[top100_genes.cancer == cancer]
            reference = block[block.method == "RWR-STRING"].set_index("node_id")
            alternative = block[block.method == alternative_name].set_index("node_id")
            method_only = alternative.index.difference(reference.index)
            reference_only = reference.index.difference(alternative.index)
            method_hits = int(alternative.loc[method_only, "incident_26.06"].sum())
            reference_hits = int(reference.loc[reference_only, "incident_26.06"].sum())
            discordant_hits = method_hits + reference_hits
            p_value = float(binomtest(method_hits, discordant_hits, 0.5).pvalue) if discordant_hits else 1.0
            pooled_method_only_hits += method_hits
            pooled_reference_only_hits += reference_hits
            paired_rows.append(
                {
                    "cancer": cancer,
                    "comparison": f"{alternative_name} vs RWR-STRING",
                    "overlap_genes": int(len(reference.index.intersection(alternative.index))),
                    "method_only_incident_hits": method_hits,
                    "reference_only_incident_hits": reference_hits,
                    "exact_p_two_sided": p_value,
                }
            )
        pooled_total = pooled_method_only_hits + pooled_reference_only_hits
        paired_rows.append(
            {
                "cancer": "POOLED_DESCRIPTIVE",
                "comparison": f"{alternative_name} vs RWR-STRING",
                "overlap_genes": np.nan,
                "method_only_incident_hits": pooled_method_only_hits,
                "reference_only_incident_hits": pooled_reference_only_hits,
                "exact_p_two_sided": float(binomtest(pooled_method_only_hits, pooled_total, 0.5).pvalue) if pooled_total else 1.0,
            }
        )
    pd.DataFrame(paired_rows).to_csv(
        args.output / "DCPI_RWR_top100_paired_incident_analysis.csv", index=False, encoding="utf-8-sig"
    )
    (args.output / "DCPI_RWR_selected_configs.json").write_text(
        json.dumps(selections, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if luad_objects is None:
        raise ValueError("LUAD must be included because the locked candidate and degree-matched null analyses are LUAD-specific.")
    historical, current, score_map, luad_performance_config, luad_discovery_config, luad_constrained_config, luad_bias_config = luad_objects
    null, null_summary = degree_matched_seed_null(
        historical,
        current,
        transitions,
        bins,
        bins[0],
        luad_bias_config,
        args.null_permutations,
    )
    null.to_csv(args.output / "DCPI_RWR_LUAD_degree_matched_seed_null.csv", index=False, encoding="utf-8-sig")
    (args.output / "DCPI_RWR_LUAD_degree_matched_seed_null_summary.json").write_text(
        json.dumps(null_summary, indent=2), encoding="utf-8"
    )

    # Lock future candidate hypotheses without using any external characterization outcome.
    current_seeds = np.flatnonzero(current == 1)
    unlabelled = np.flatnonzero(current == 0)
    hub_cutoff = float(np.quantile(degrees[0], 0.95))
    for label, config in {
        "performance": luad_performance_config,
        "discovery": luad_discovery_config,
        "constrained_top100": luad_constrained_config,
        "bias_aware": luad_bias_config,
    }.items():
        parts = components(transitions, bins, current_seeds, size, config.restart)
        current_scores = score_config(parts, config)
        order = unlabelled[np.argsort(-current_scores[unlabelled], kind="mergesort")]
        candidate_rows = []
        for rank, node in enumerate(order[:500], 1):
            candidate_rows.append(
                {
                    "Candidate_Rank": rank,
                    "Gene": reverse_mapping.get(int(node), str(node)),
                    "Gene_Symbol": reverse_mapping.get(int(node), str(node)),
                    "node_id": int(node),
                    "DCPI_RWR_score": float(current_scores[node]),
                    "STRING_degree": int(degrees[0][node]),
                    "STRING_hub_top5pct": bool(degrees[0][node] >= hub_cutoff),
                    "locked_objective": label,
                }
            )
        candidate_frame = pd.DataFrame(candidate_rows)
        candidate_frame.to_csv(args.output / f"DCPI_RWR_{label}_current26_unlabelled_top500.csv", index=False, encoding="utf-8-sig")
        candidate_frame.head(20).to_csv(args.output / f"DCPI_RWR_{label}_current26_locked_top20.csv", index=False, encoding="utf-8-sig")

    plot_results(metrics, topk, tuning, args.output)
    manifest = {
        "method_working_name": "DCPI-RWR (degree-conditioned pathway-integrated random walk with restart)",
        "development": "Hyperparameters selected using repeated five-fold CV of 26.03 labels only; AUPRC, Precision@100 and bias-aware objectives were locked separately",
        "validation": "Locked configurations evaluated against genes first present in the 26.06 disease-specific evidence set",
        "views": VIEW_NAMES,
        "candidate_configurations": len(candidate_configs),
        "bootstrap_replicates": args.bootstrap,
        "degree_matched_seed_null_replicates": args.null_permutations,
        "important_boundary": "Graph topology was not historically frozen; cross-release results are time-separated label validation, not a fully prospective frozen-feature experiment.",
        "input_sha256": {
            args.dataset.name: sha256(args.dataset),
            args.mapping.name: sha256(args.mapping),
        },
    }
    (args.output / "DCPI_RWR_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
