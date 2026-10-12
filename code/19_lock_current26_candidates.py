from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression


def percentile(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock outcome-blind candidate lists from the unified 26.06 OOF benchmark.")
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--old-candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(args.oof)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8-sig"))
    node_to_symbol = {int(node): str(symbol).upper() for symbol, node in mapping.items()}
    edge = data.edge_index_ppi.cpu().numpy()
    degree = np.bincount(edge[0], minlength=data.num_nodes)
    weighted_degree = np.bincount(
        edge[0], weights=data.edge_weight_ppi.cpu().numpy(), minlength=data.num_nodes
    )

    mean_scores = oof.groupby(["node_id", "label", "method"], as_index=False)["score"].mean()
    wide = mean_scores.pivot(index=["node_id", "label"], columns="method", values="score").reset_index()
    repeat_scores = oof.groupby(["node_id", "method", "repeat"], as_index=False)["score"].mean()
    rwr_repeat = repeat_scores[repeat_scores["method"] == "RWR"].pivot(
        index="node_id", columns="repeat", values="score"
    )
    rwr_rank_sd = rwr_repeat.rank(ascending=False).std(axis=1)
    wide["gene"] = wide["node_id"].map(node_to_symbol)
    wide["STRING_degree"] = wide["node_id"].map(lambda node: int(degree[int(node)]))
    wide["STRING_weighted_degree"] = wide["node_id"].map(lambda node: float(weighted_degree[int(node)]))
    wide["RWR_rank_sd_across_repeats"] = wide["node_id"].map(rwr_rank_sd)
    unlabelled = wide[wide["label"] == 0].copy()
    for method in ["RWR", "LightGBM + RWR/features", "XGBoost + RWR/features", "APPNP", "GCNII", "GATv2"]:
        if method in unlabelled:
            unlabelled[f"{method}_percentile"] = percentile(unlabelled[method])

    unlabelled["performance_aligned_score"] = unlabelled["RWR_percentile"]
    unlabelled["performance_aligned_rank"] = unlabelled["performance_aligned_score"].rank(
        ascending=False, method="first"
    ).astype(int)

    x = np.log1p(unlabelled["STRING_degree"].to_numpy(dtype=float))
    design = np.column_stack([x, x**2, x**3])
    model = LinearRegression().fit(design, unlabelled["RWR"].to_numpy(dtype=float))
    unlabelled["RWR_degree_residual"] = unlabelled["RWR"] - model.predict(design)
    unlabelled["topology_adjusted_rank"] = unlabelled["RWR_degree_residual"].rank(
        ascending=False, method="first"
    ).astype(int)
    hub_cutoff = float(np.quantile(degree, 0.95))
    unlabelled["STRING_hub_top5pct"] = unlabelled["STRING_degree"] >= hub_cutoff

    columns = [
        "gene",
        "node_id",
        "STRING_degree",
        "STRING_weighted_degree",
        "STRING_hub_top5pct",
        "RWR",
        "RWR_rank_sd_across_repeats",
        "LightGBM + RWR/features",
        "XGBoost + RWR/features",
        "APPNP",
        "GCNII",
        "GATv2",
        "performance_aligned_score",
        "performance_aligned_rank",
        "RWR_degree_residual",
        "topology_adjusted_rank",
    ]
    available = [column for column in columns if column in unlabelled]
    ranked = unlabelled.sort_values("performance_aligned_rank")[available]
    ranked.to_csv(args.output / "current26_all_unlabelled_ranked.csv.gz", index=False, compression="gzip")
    primary = ranked.head(20).copy()
    primary.insert(0, "candidate_rank", np.arange(1, len(primary) + 1))
    primary.insert(1, "Candidate_Rank", primary["candidate_rank"])
    primary.insert(2, "Gene", primary["gene"])
    primary.insert(3, "Gene_Symbol", primary["gene"])
    primary.to_csv(args.output / "current26_locked_RWR_top20.csv", index=False, encoding="utf-8-sig")
    adjusted = unlabelled.sort_values("topology_adjusted_rank")[available].head(20).copy()
    adjusted.insert(0, "candidate_rank", np.arange(1, len(adjusted) + 1))
    adjusted.insert(1, "Candidate_Rank", adjusted["candidate_rank"])
    adjusted.insert(2, "Gene", adjusted["gene"])
    adjusted.insert(3, "Gene_Symbol", adjusted["gene"])
    adjusted.to_csv(args.output / "current26_topology_adjusted_sensitivity_top20.csv", index=False, encoding="utf-8-sig")

    comparison = {
        "old_candidate_file_supplied": bool(args.old_candidates and args.old_candidates.exists()),
        "primary_top20_hubs": int(primary["STRING_hub_top5pct"].sum()),
        "topology_adjusted_top20_hubs": int(adjusted["STRING_hub_top5pct"].sum()),
        "primary_vs_topology_adjusted_overlap": len(set(primary["gene"]) & set(adjusted["gene"])),
    }
    if args.old_candidates and args.old_candidates.exists():
        old = pd.read_csv(args.old_candidates)
        gene_column = "Gene" if "Gene" in old else "gene"
        comparison["current_primary_vs_historical_PU_top20_overlap"] = len(
            set(primary["gene"]) & set(old[gene_column].astype(str).str.upper())
        )
        comparison["historical_PU_top20_genes"] = old[gene_column].astype(str).str.upper().head(20).tolist()
    comparison["current26_locked_primary_genes"] = primary["gene"].tolist()
    comparison["current26_topology_adjusted_genes"] = adjusted["gene"].tolist()
    (args.output / "current26_candidate_set_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )

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
    fig, ax = plt.subplots(figsize=(4.3, 3.4))
    ax.scatter(
        np.log1p(unlabelled["STRING_degree"]),
        unlabelled["RWR"],
        s=5,
        color="#BDBDBD",
        alpha=0.35,
        linewidth=0,
    )
    selected = unlabelled[unlabelled["gene"].isin(primary["gene"])]
    ax.scatter(
        np.log1p(selected["STRING_degree"]),
        selected["RWR"],
        s=22,
        color="#0072B2",
        edgecolor="white",
        linewidth=0.4,
        label="Locked RWR top 20",
    )
    ax.set_xlabel("log(1 + STRING degree)")
    ax.set_ylabel("Mean out-of-fold RWR score")
    ax.set_title("Current 26.06 unlabelled-gene ranking")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E5E5E5", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(args.output / "current26_candidate_degree_audit.pdf", bbox_inches="tight")
    fig.savefig(args.output / "current26_candidate_degree_audit.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
