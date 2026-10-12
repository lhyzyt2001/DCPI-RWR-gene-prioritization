from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from statsmodels.stats.multitest import multipletests


CANCERS = ["LUAD", "BRCA", "CRC", "PRAD", "MEL", "OV"]
SEED = 20261001


def load_core(script: Path):
    spec = importlib.util.spec_from_file_location("dcpi_rwr_core_perm", script)
    if spec is None or spec.loader is None:
        raise ImportError(script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Degree-conditional permutation of the pathway component.")
    parser.add_argument("--core-script", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--selected-configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    core = load_core(args.core_script)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    mapping = {str(k).upper(): int(v) for k, v in json.loads(args.mapping.read_text(encoding="utf-8-sig")).items()}
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
    selected = json.loads(args.selected_configs.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED + 2800)
    null_rows, summary_rows = [], []
    for cancer_index, cancer in enumerate(CANCERS):
        config = core.Config(**selected[cancer]["performance"])
        historical = core.read_labels(args.labels_root / cancer / "historical_26.03.csv", mapping, size)
        current = core.read_labels(args.labels_root / cancer / "future_26.06.csv", mapping, size)
        candidates = np.flatnonzero(historical == 0)
        labels = ((current == 1) & (historical == 0)).astype(np.int8)[candidates]
        parts = core.components(transitions, bins, np.flatnonzero(historical == 1), size, config.restart)
        raw_string, string_conditioned, cell_conditioned, pathway_conditioned = [part[candidates] for part in parts]
        fixed = (
            (1.0 - config.blend) * raw_string
            + config.blend * config.weight_string * string_conditioned
            + config.blend * config.weight_celltype * cell_conditioned
        )
        coefficient = config.blend * config.weight_pathway
        observed_score = fixed + coefficient * pathway_conditioned
        observed_top = np.argpartition(-observed_score, 99)[:100]
        observed_hits = int(labels[observed_top].sum())
        candidate_bins = bins[0][candidates]
        groups = [np.flatnonzero(candidate_bins == value) for value in np.unique(candidate_bins)]
        null_hits = []
        for replicate in range(args.permutations):
            shuffled = pathway_conditioned.copy()
            for group in groups:
                shuffled[group] = rng.permutation(shuffled[group])
            null_score = fixed + coefficient * shuffled
            top = np.argpartition(-null_score, 99)[:100]
            hits = int(labels[top].sum())
            null_hits.append(hits)
            null_rows.append({"cancer": cancer, "replicate": replicate, "top100_incident_hits": hits})
        array = np.asarray(null_hits)
        summary_rows.append(
            {
                "cancer": cancer,
                "config_id": config.config_id,
                "effective_pathway_coefficient": coefficient,
                "observed_top100_incident_hits": observed_hits,
                "null_mean_top100_incident_hits": float(array.mean()),
                "null_ci_lower": float(np.quantile(array, 0.025)),
                "null_ci_upper": float(np.quantile(array, 0.975)),
                "empirical_p_greater_equal": float((1 + (array >= observed_hits).sum()) / (args.permutations + 1)),
                "permutations": args.permutations,
                "permutation": "Pathway conditioned-percentile values shuffled within STRING-degree bins; all other score components fixed.",
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["p_holm_six_cancers"] = multipletests(summary.empirical_p_greater_equal, method="holm")[1]
    pd.DataFrame(null_rows).to_csv(args.output / "pathway_component_conditional_null.csv.gz", index=False, compression="gzip")
    summary.to_csv(args.output / "pathway_component_conditional_permutation_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "hypothesis": "The pathway component improves fixed-budget recovery beyond pathway values exchangeable among genes of comparable STRING degree.",
        "selection": "Cancer-specific performance configurations were selected using 26.03 historical cross-validation only.",
        "future_labels_used_for_configuration_selection": False,
        "multiplicity": "Holm correction across six cancer-specific pathway tests.",
        "permutations": args.permutations,
    }
    (args.output / "pathway_component_permutation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
