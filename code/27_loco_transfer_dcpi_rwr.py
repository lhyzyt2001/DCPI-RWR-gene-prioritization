from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


CANCERS = ["LUAD", "BRCA", "CRC", "PRAD", "MEL", "OV"]
SEED = 20261001


def load_core(script: Path):
    spec = importlib.util.spec_from_file_location("dcpi_rwr_core", script)
    if spec is None or spec.loader is None:
        raise ImportError(script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select_loco(tuning: pd.DataFrame, target: str, core) -> tuple[object, pd.DataFrame]:
    training = tuning[tuning.cancer != target].copy()
    training["AUPRC_percentile_within_cancer"] = training.groupby("cancer")["mean_AUPRC"].rank(pct=True, method="average")
    aggregate = training.groupby(
        ["config_id", "restart", "blend", "weight_string", "weight_celltype", "weight_pathway"], as_index=False
    ).agg(
        training_cancers=("cancer", "nunique"),
        mean_AUPRC_percentile=("AUPRC_percentile_within_cancer", "mean"),
        mean_abs_score_degree_spearman=("mean_abs_score_degree_spearman", "mean"),
    )
    chosen = aggregate.sort_values(
        ["mean_AUPRC_percentile", "mean_abs_score_degree_spearman"], ascending=[False, True]
    ).iloc[0]
    config = core.Config(
        float(chosen.restart),
        float(chosen.blend),
        float(chosen.weight_string),
        float(chosen.weight_celltype),
        float(chosen.weight_pathway),
    )
    aggregate.insert(0, "target_cancer", target)
    aggregate["selected"] = aggregate.config_id.eq(config.config_id)
    return config, aggregate


def bootstrap(labels: np.ndarray, reference: np.ndarray, alternative: np.ndarray, replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    differences = []
    for _ in range(replicates):
        sampled = np.concatenate(
            [rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        )
        differences.append(
            average_precision_score(labels[sampled], alternative[sampled])
            - average_precision_score(labels[sampled], reference[sampled])
        )
    values = np.asarray(differences)
    p = 2 * min((1 + (values <= 0).sum()) / (replicates + 1), (1 + (values >= 0).sum()) / (replicates + 1))
    return {
        "median_delta_AUPRC": float(np.median(values)),
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "p_two_sided": float(min(p, 1.0)),
        "replicates": replicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-cancer-out historical transfer of DCPI-RWR hyperparameters.")
    parser.add_argument("--core-script", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--tuning", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    core = load_core(args.core_script)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    mapping = {str(k).upper(): int(v) for k, v in json.loads(args.mapping.read_text(encoding="utf-8-sig")).items()}
    reverse = {node: gene for gene, node in mapping.items()}
    size = int(data.num_nodes)
    transitions, degrees = [], []
    for edge_name, weight_name in [
        ("edge_index_ppi", "edge_weight_ppi"),
        ("edge_index_coexp", "edge_weight_coexp"),
        ("edge_index_pathway", "edge_weight_pathway"),
    ]:
        edge = getattr(data, edge_name).cpu().numpy()
        weight = getattr(data, weight_name).cpu().numpy()
        transitions.append(core.transition(edge, weight, size))
        degrees.append(np.bincount(edge[0], minlength=size))
    bins = [core.degree_bins(degree) for degree in degrees]
    tuning = pd.read_csv(args.tuning)
    selected_rows, metric_rows, top_rows, boot_rows, paired_rows, top100_rows = [], [], [], [], [], []
    luad_config = None
    for index, cancer in enumerate(CANCERS):
        config, aggregate = select_loco(tuning, cancer, core)
        selected_rows.append(aggregate[aggregate.selected].iloc[0].to_dict())
        historical = core.read_labels(args.labels_root / cancer / "historical_26.03.csv", mapping, size)
        current = core.read_labels(args.labels_root / cancer / "future_26.06.csv", mapping, size)
        incident = ((current == 1) & (historical == 0)).astype(np.int8)
        candidates = np.flatnonzero(historical == 0)
        seeds = np.flatnonzero(historical == 1)
        rwr_parts = core.components(transitions, bins, seeds, size, 0.30)
        alt_parts = core.components(transitions, bins, seeds, size, config.restart)
        reference_full = core.score_config(rwr_parts, core.Config(0.30, 0.0, 1.0, 0.0, 0.0))
        alternative_full = core.score_config(alt_parts, config)
        labels = incident[candidates]
        reference = reference_full[candidates]
        alternative = alternative_full[candidates]
        metric_rows.append(
            {
                "cancer": cancer,
                "method": "LOCO DCPI-RWR transfer",
                "config_id": config.config_id,
                "AUPRC": average_precision_score(labels, alternative),
                "ROC_AUC": roc_auc_score(labels, alternative),
                "score_degree_spearman": float(spearmanr(alternative, degrees[0][candidates]).statistic),
                "RWR_AUPRC": average_precision_score(labels, reference),
                "RWR_ROC_AUC": roc_auc_score(labels, reference),
            }
        )
        boot_rows.append({"cancer": cancer, **bootstrap(labels, reference, alternative, args.bootstrap, SEED + index)})
        hub_cutoff = np.quantile(degrees[0], 0.95)
        orders = {
            "RWR-STRING": candidates[np.argsort(-reference, kind="mergesort")],
            "LOCO DCPI-RWR transfer": candidates[np.argsort(-alternative, kind="mergesort")],
        }
        for method, order in orders.items():
            selected = order[:100]
            top_rows.append(
                {
                    "cancer": cancer,
                    "method": method,
                    "incident_hits_at_100": int(incident[selected].sum()),
                    "hub_fraction_at_100": float((degrees[0][selected] >= hub_cutoff).mean()),
                    "median_STRING_degree_at_100": float(np.median(degrees[0][selected])),
                }
            )
            for rank, node in enumerate(selected, 1):
                top100_rows.append(
                    {"cancer": cancer, "method": method, "rank": rank, "node_id": int(node), "Gene": reverse.get(int(node), str(node)), "incident_26.06": int(incident[node])}
                )
        reference_nodes = set(orders["RWR-STRING"][:100])
        method_nodes = set(orders["LOCO DCPI-RWR transfer"][:100])
        method_only_hits = int(incident[list(method_nodes - reference_nodes)].sum())
        reference_only_hits = int(incident[list(reference_nodes - method_nodes)].sum())
        discordant = method_only_hits + reference_only_hits
        paired_rows.append(
            {
                "cancer": cancer,
                "overlap_genes": len(method_nodes & reference_nodes),
                "method_only_incident_hits": method_only_hits,
                "reference_only_incident_hits": reference_only_hits,
                "exact_p_two_sided": float(binomtest(method_only_hits, discordant, 0.5).pvalue) if discordant else 1.0,
            }
        )
        if cancer == "LUAD":
            luad_config = config

    selected = pd.DataFrame(selected_rows)
    metrics = pd.DataFrame(metric_rows)
    top = pd.DataFrame(top_rows)
    boots = pd.DataFrame(boot_rows)
    paired = pd.DataFrame(paired_rows)
    pooled_method = int(paired.method_only_incident_hits.sum())
    pooled_reference = int(paired.reference_only_incident_hits.sum())
    pooled_discordant = pooled_method + pooled_reference
    paired = pd.concat(
        [
            paired,
            pd.DataFrame(
                [{
                    "cancer": "POOLED_DESCRIPTIVE",
                    "overlap_genes": np.nan,
                    "method_only_incident_hits": pooled_method,
                    "reference_only_incident_hits": pooled_reference,
                    "exact_p_two_sided": float(binomtest(pooled_method, pooled_discordant, 0.5).pvalue) if pooled_discordant else 1.0,
                }]
            ),
        ], ignore_index=True
    )
    selected.to_csv(args.output / "LOCO_selected_configs.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output / "LOCO_cross_release_metrics.csv", index=False, encoding="utf-8-sig")
    top.to_csv(args.output / "LOCO_top100_summary.csv", index=False, encoding="utf-8-sig")
    boots.to_csv(args.output / "LOCO_bootstrap_vs_RWR.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(args.output / "LOCO_top100_paired_incident_analysis.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(top100_rows).to_csv(args.output / "LOCO_top100_genes.csv", index=False, encoding="utf-8-sig")

    if luad_config is None:
        raise RuntimeError("LUAD configuration was not generated")
    current = core.read_labels(args.labels_root / "LUAD" / "future_26.06.csv", mapping, size)
    parts = core.components(transitions, bins, np.flatnonzero(current == 1), size, luad_config.restart)
    score = core.score_config(parts, luad_config)
    unlabelled = np.flatnonzero(current == 0)
    order = unlabelled[np.argsort(-score[unlabelled], kind="mergesort")[:500]]
    hub_cutoff = np.quantile(degrees[0], 0.95)
    candidates = pd.DataFrame(
        [
            {
                "Candidate_Rank": rank,
                "Gene": reverse.get(int(node), str(node)),
                "Gene_Symbol": reverse.get(int(node), str(node)),
                "node_id": int(node),
                "LOCO_DCPI_RWR_score": float(score[node]),
                "STRING_degree": int(degrees[0][node]),
                "STRING_hub_top5pct": bool(degrees[0][node] >= hub_cutoff),
            }
            for rank, node in enumerate(order, 1)
        ]
    )
    candidates.to_csv(args.output / "LOCO_LUAD_current26_unlabelled_top500.csv", index=False, encoding="utf-8-sig")
    candidates.head(20).to_csv(args.output / "LOCO_LUAD_current26_locked_top20.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "design": "For each target cancer, the configuration maximizing mean within-cancer historical-CV AUPRC percentile across the other five cancers was locked and transferred to the target cancer.",
        "future_label_use_in_selection": False,
        "target_cancer_historical_label_use_in_selection": False,
        "bootstrap_replicates": args.bootstrap,
        "boundary": "Graph features were not release-frozen; this is cross-release label validation, not a fully prospective experiment.",
    }
    (args.output / "LOCO_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
