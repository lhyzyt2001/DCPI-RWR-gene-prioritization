from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm


STANDARD = "LightGBM direct stack"
PU = "BaggingPU LightGBM 5:1 unlabelled:positive"
RWR = "RWR (STRING graph)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a post hoc PU-robust candidate consensus.")
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(args.oof)
    data = torch.load(args.graph, map_location="cpu", weights_only=False)
    labels = data.y.cpu().numpy().astype(int)
    edge = data.edge_index_ppi.cpu().numpy()
    degree = np.bincount(edge[0], minlength=data.num_nodes)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8-sig"))
    symbols = np.empty(data.num_nodes, dtype=object)
    for symbol, node in mapping.items():
        symbols[int(node)] = symbol
    selected = oof[oof["method"].isin([STANDARD, PU, RWR])].copy()
    if set(selected["method"]) != {STANDARD, PU, RWR}:
        raise RuntimeError("Required standard, PU and RWR scores are incomplete")
    unlabelled = np.flatnonzero(labels == 0)

    rank_rows = []
    for (method, repeat), block in selected.groupby(["method", "repeat"]):
        block = block.groupby("node_id", as_index=False)["score"].mean().set_index("node_id")
        scores = block.loc[unlabelled, "score"]
        rank = scores.rank(method="average", ascending=False)
        for node in unlabelled:
            rank_rows.append(
                {
                    "node_id": int(node),
                    "method": method,
                    "repeat": int(repeat),
                    "score": float(scores.loc[node]),
                    "rank": float(rank.loc[node]),
                    "rank_percentile": float(rank.loc[node] / len(unlabelled)),
                }
            )
    ranks = pd.DataFrame(rank_rows)
    ensemble = ranks[ranks["method"].isin([STANDARD, PU])]
    consensus = (
        ensemble.groupby("node_id", as_index=False)
        .agg(
            consensus_rank_mean=("rank", "mean"),
            consensus_rank_sd=("rank", "std"),
            consensus_rank_percentile_mean=("rank_percentile", "mean"),
            top20_frequency=("rank", lambda x: float(np.mean(np.asarray(x) <= 20))),
            top100_frequency=("rank", lambda x: float(np.mean(np.asarray(x) <= 100))),
            top500_frequency=("rank", lambda x: float(np.mean(np.asarray(x) <= 500))),
        )
    )
    standard_mean = (
        ranks[ranks["method"].eq(STANDARD)].groupby("node_id")["score"].mean().rename("standard_score_mean")
    )
    pu_mean = ranks[ranks["method"].eq(PU)].groupby("node_id")["score"].mean().rename("pu_score_mean")
    rwr_mean = ranks[ranks["method"].eq(RWR)].groupby("node_id")["score"].mean().rename("rwr_score_mean")
    table = consensus.join(standard_mean, on="node_id").join(pu_mean, on="node_id").join(rwr_mean, on="node_id")
    table["Gene"] = symbols[table["node_id"].to_numpy(dtype=int)]
    table["Gene_Symbol"] = table["Gene"]
    table["STRING_Degree"] = degree[table["node_id"].to_numpy(dtype=int)]
    hub_cutoff = float(np.quantile(degree, 0.95))
    table["STRING_Hub_Top5pct"] = table["STRING_Degree"] >= hub_cutoff
    table = table.sort_values(
        ["consensus_rank_mean", "consensus_rank_sd", "Gene"], ascending=[True, True, True]
    ).reset_index(drop=True)
    table["Candidate_Rank"] = np.arange(1, len(table) + 1)

    # Post hoc topology-adjusted sensitivity: regress a normal-score transform of the
    # consensus rank on RWR rank and log degree. It is not used as the primary list.
    n = len(table)
    strength = norm.ppf(np.clip(1 - table["consensus_rank_percentile_mean"], 1e-6, 1 - 1e-6))
    rwr_rank = table["rwr_score_mean"].rank(pct=True, ascending=False).to_numpy()
    x = np.column_stack([np.ones(n), np.log1p(table["STRING_Degree"].to_numpy()), rwr_rank])
    beta = np.linalg.lstsq(x, strength, rcond=None)[0]
    table["topology_adjusted_residual"] = strength - x @ beta
    table["topology_adjusted_rank"] = table["topology_adjusted_residual"].rank(
        method="first", ascending=False
    ).astype(int)

    columns = [
        "Candidate_Rank",
        "Gene",
        "Gene_Symbol",
        "node_id",
        "consensus_rank_mean",
        "consensus_rank_sd",
        "top20_frequency",
        "top100_frequency",
        "top500_frequency",
        "standard_score_mean",
        "pu_score_mean",
        "rwr_score_mean",
        "STRING_Degree",
        "STRING_Hub_Top5pct",
        "topology_adjusted_residual",
        "topology_adjusted_rank",
    ]
    table[columns].to_csv(args.output / "pu_robust_consensus_all_unlabelled.csv.gz", index=False, compression="gzip")
    table[columns].head(20).to_csv(args.output / "pu_robust_consensus_top20.csv", index=False)
    table.sort_values("topology_adjusted_rank")[columns].head(20).to_csv(
        args.output / "topology_adjusted_sensitivity_top20.csv", index=False
    )
    ranks.to_csv(args.output / "method_repeat_candidate_ranks.csv.gz", index=False, compression="gzip")
    manifest = {
        "status": "completed",
        "selection": "Mean rank across standard LightGBM and 5:1 bagging-PU LightGBM, using all six method-repeat rankings; no downstream biological outcome used.",
        "timing": "Post hoc after diagnosing the positive-unlabelled label structure.",
        "positive_genes_excluded": int(labels.sum()),
        "unlabelled_genes_ranked": int((labels == 0).sum()),
        "topology_adjusted_list": "Sensitivity only; it does not replace the primary PU-robust consensus.",
        "hub_definition": f"STRING degree >= empirical 95th percentile ({hub_cutoff:g})",
    }
    (args.output / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(table[columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
