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
METHODS = ["RWR (historical seeds)", "XGBoost historical stack", "LightGBM historical stack"]
COLORS = {METHODS[0]: "#0072B2", METHODS[1]: "#E69F00", METHODS[2]: "#009E73"}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def safe_metrics(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if np.unique(y).size < 2:
        return np.nan, np.nan
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def bootstrap(oof: pd.DataFrame, y_incident: np.ndarray, eligible: np.ndarray, replicates: int) -> pd.DataFrame:
    wide = oof.pivot(index=["node_id", "repeat"], columns="method", values="score").reset_index()
    repeats = sorted(wide["repeat"].unique())
    matrices = {int(r): wide[wide["repeat"].eq(r)].set_index("node_id").sort_index() for r in repeats}
    nodes = matrices[repeats[0]].index.to_numpy(dtype=int)
    keep = eligible[nodes]
    nodes = nodes[keep]
    y = y_incident[nodes]
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(SEED)
    pairs = [(METHODS[1], METHODS[0]), (METHODS[2], METHODS[0]), (METHODS[2], METHODS[1])]
    distributions = {pair: np.empty(replicates) for pair in pairs}
    arrays = {r: {m: matrices[r].loc[nodes, m].to_numpy() for m in METHODS} for r in repeats}
    for b in range(replicates):
        sample = np.concatenate(
            [rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        )
        aps = {
            method: np.mean([average_precision_score(y[sample], arrays[r][method][sample]) for r in repeats])
            for method in METHODS
        }
        for pair in pairs:
            distributions[pair][b] = aps[pair[0]] - aps[pair[1]]
    rows = []
    for (method, reference), values in distributions.items():
        p = min(
            1.0,
            2
            * min(
                (np.count_nonzero(values <= 0) + 1) / (replicates + 1),
                (np.count_nonzero(values >= 0) + 1) / (replicates + 1),
            ),
        )
        rows.append(
            {
                "method": method,
                "reference": reference,
                "median_delta_incident_AUPRC": float(np.median(values)),
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "p_two_sided": float(p),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "gene retained across three historical OOF repeats",
            }
        )
    return pd.DataFrame(rows)


def plot_results(metrics: pd.DataFrame, topk: pd.DataFrame, output: Path) -> None:
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
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.38})
    summary = metrics.groupby("method").agg(AUPRC=("AUPRC", "mean"), SD=("AUPRC", "std")).loc[METHODS]
    x = np.arange(len(METHODS))
    axes[0].bar(x, summary["AUPRC"], yerr=summary["SD"], color=[COLORS[m] for m in METHODS], capsize=3)
    axes[0].axhline(metrics["incident_prevalence"].iloc[0], color="#666666", ls="--", lw=1, label="Incident prevalence")
    axes[0].set_xticks(x, ["RWR", "XGBoost", "LightGBM"], rotation=25, ha="right")
    axes[0].set_ylabel("Incident-label AUPRC")
    axes[0].set_title("26.03 to 26.06 temporal validation")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].text(-0.17, 1.05, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=11)
    for method in METHODS:
        part = topk[topk["method"].eq(method)]
        axes[1].plot(
            part["K"],
            part["incident_recall_at_k_mean"],
            marker="o",
            ms=3,
            color=COLORS[method],
            label=method.split()[0],
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Top K historical-OOF ranked genes")
    axes[1].set_ylabel("Recall of future incident targets")
    axes[1].set_title("Incident-target recovery")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].text(-0.17, 1.05, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=11)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E5E5E5", lw=0.6)
    fig.savefig(output / "temporal_validation.pdf", bbox_inches="tight")
    fig.savefig(output / "temporal_validation.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--historical-labels", type=Path, required=True)
    parser.add_argument("--future-labels", type=Path, required=True)
    parser.add_argument("--benchmark-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--historical-release", default="26.03")
    parser.add_argument("--future-release", default="26.06")
    parser.add_argument(
        "--design-label",
        default="three-repeat five-fold historical-label OOF prediction followed by incident-label evaluation",
    )
    parser.add_argument(
        "--endpoint-note",
        default="Disease-specific clinical-precedence evidence; cross-release changes can include pipeline and ontology updates.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = load_module(args.benchmark_code)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8-sig"))
    symbol_to_node = {str(symbol).upper(): int(node) for symbol, node in mapping.items()}
    historical_symbols = set(pd.read_csv(args.historical_labels)["symbol"].astype(str).str.upper())
    future_symbols = set(pd.read_csv(args.future_labels)["symbol"].astype(str).str.upper())
    historical_nodes = {symbol_to_node[s] for s in historical_symbols if s in symbol_to_node}
    future_nodes = {symbol_to_node[s] for s in future_symbols if s in symbol_to_node}
    labels = np.zeros(data.num_nodes, dtype=int)
    labels[list(historical_nodes)] = 1
    incident = np.zeros(data.num_nodes, dtype=int)
    incident[list(future_nodes - historical_nodes)] = 1
    eligible = labels == 0
    raw_expression = data.x.cpu().numpy().astype(np.float32)
    topology, topology_names = base.network_degree_features(data)
    centrality, centrality_names = base.string_centralities(data, 128)
    topology = np.column_stack([topology, centrality]).astype(np.float32)
    transition = base.prepare_transition(data)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    oof_rows, audit_rows = [], []
    cell_names = [str(x).replace(" ", "_") for x in data.cell_types]
    for split_number, (outer_train, outer_test) in enumerate(outer.split(np.zeros(len(labels)), labels)):
        repeat, fold = divmod(split_number, 5)
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
        core_base = base.calibrated_logit(calibrator, core_rwr)
        val_base = base.calibrated_logit(calibrator, all_rwr[val])
        test_base = base.calibrated_logit(calibrator, all_rwr[outer_test])
        x_core = np.column_stack([core_base, feature[core]])
        x_val = np.column_stack([val_base, feature[val]])
        x_test = np.column_stack([test_base, feature[outer_test]])
        xgb, _, xgb_score = base.fit_xgb_direct(x_core, labels[core], x_val, labels[val], x_test, split_seed)
        lgb, _, lgb_score = base.fit_lgb_direct(x_core, labels[core], x_val, labels[val], x_test, split_seed)
        scores = {METHODS[0]: all_rwr[outer_test], METHODS[1]: xgb_score, METHODS[2]: lgb_score}
        for method, values in scores.items():
            oof_rows.extend(
                {"repeat": repeat, "fold": fold, "node_id": int(node), "method": method, "score": float(score)}
                for node, score in zip(outer_test, values)
            )
        audit_rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "core_n": len(core),
                "core_positives": int(labels[core].sum()),
                "validation_positives": int(labels[val].sum()),
                "outer_test_positives": int(labels[outer_test].sum()),
                "incident_nodes_used_as_positive_labels": False,
                "future_snapshot_used_for_fitting": False,
                "feature_names": "|".join(["RWR_calibrated_logit", *topology_names, *centrality_names, *[f"CellType_{x}" for x in cell_names], *[f"PCA_{i+1}" for i in range(5)]]),
                "xgb_best_iteration": int(getattr(xgb, "best_iteration", 0) or 0),
                "lgb_best_iteration": int(lgb.best_iteration_ or 0),
            }
        )
        print(f"temporal fold {split_number + 1}/15 complete", flush=True)
    oof = pd.DataFrame(oof_rows)
    metric_rows, top_rows = [], []
    evaluation_nodes = np.flatnonzero(eligible)
    y = incident[evaluation_nodes]
    for repeat in range(3):
        block = oof[oof["repeat"].eq(repeat)].pivot(index="node_id", columns="method", values="score").loc[evaluation_nodes]
        for method in METHODS:
            roc, ap = safe_metrics(y, block[method].to_numpy())
            metric_rows.append(
                {
                    "repeat": repeat,
                    "method": method,
                    "ROC_AUC": roc,
                    "AUPRC": ap,
                    "incident_positives": int(y.sum()),
                    "eligible_genes": len(y),
                    "incident_prevalence": float(y.mean()),
                }
            )
            order = np.argsort(-block[method].to_numpy())
            for k in (20, 50, 100, 200, 500, 1000):
                selected = y[order[:k]]
                top_rows.append(
                    {
                        "repeat": repeat,
                        "method": method,
                        "K": k,
                        "incident_hits": int(selected.sum()),
                        "precision_at_k": float(selected.mean()),
                        "incident_recall_at_k": float(selected.sum() / y.sum()),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    topk = (
        pd.DataFrame(top_rows)
        .groupby(["method", "K"], as_index=False)
        .agg(
            incident_hits_mean=("incident_hits", "mean"),
            incident_hits_sd=("incident_hits", "std"),
            precision_at_k_mean=("precision_at_k", "mean"),
            precision_at_k_sd=("precision_at_k", "std"),
            incident_recall_at_k_mean=("incident_recall_at_k", "mean"),
            incident_recall_at_k_sd=("incident_recall_at_k", "std"),
        )
    )
    boot = bootstrap(oof, incident, eligible, args.bootstrap_replicates)
    oof.to_csv(args.output / "historical_model_oof_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(audit_rows).to_csv(args.output / "temporal_training_audit.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output / "temporal_validation_metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    metrics.groupby("method").agg(
        repeats=("AUPRC", "size"), ROC_AUC_mean=("ROC_AUC", "mean"), ROC_AUC_sd=("ROC_AUC", "std"), AUPRC_mean=("AUPRC", "mean"), AUPRC_sd=("AUPRC", "std")
    ).reset_index().to_csv(args.output / "temporal_validation_summary.csv", index=False, encoding="utf-8-sig")
    topk.to_csv(args.output / "temporal_validation_topk.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(args.output / "temporal_validation_gene_cluster_bootstrap.csv", index=False, encoding="utf-8-sig")
    plot_results(metrics, topk, args.output)
    manifest = {
        "historical_release": args.historical_release,
        "future_release": args.future_release,
        "historical_positive_count": len(historical_nodes),
        "future_positive_count": len(future_nodes),
        "incident_positive_count": int(incident.sum()),
        "evaluation_population": "all 26.03-unlabelled network genes",
        "design": args.design_label,
        "future_labels_used_for_fitting": False,
        "important_endpoint_note": args.endpoint_note,
    }
    (args.output / "temporal_validation_design.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
