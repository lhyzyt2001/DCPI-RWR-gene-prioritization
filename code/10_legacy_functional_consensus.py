from __future__ import annotations

import argparse
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit

from common import COLORS, SEED, configure_plotting, save_figure, sha256, write_json


def read_gmt(path: Path) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.rstrip().split("\t")
        if len(fields) >= 3:
            sets[fields[0]] = [gene.upper() for gene in fields[2:] if gene]
    return sets


def run_one(ranking: pd.DataFrame, sets: dict[str, list[str]], library: str, analysis: str, out: Path) -> pd.DataFrame:
    result = gp.prerank(
        rnk=ranking[["Gene", "Score"]], gene_sets=sets, min_size=15, max_size=500,
        permutation_num=2000, seed=SEED, threads=4, verbose=False, outdir=None,
    ).res2d.copy()
    result.insert(0, "Analysis", analysis)
    result.insert(1, "Library", library)
    result.to_csv(out / f"GSEA_{library}_{analysis}.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hallmark and GO-BP characterization without KEGG circularity.")
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--hallmark", type=Path, required=True)
    parser.add_argument("--go-bp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.ranking)
    candidates["Gene"] = candidates["Gene_Symbol"].astype(str).str.upper()
    candidates["Primary_Score"] = logit(
        ((candidates["standard_score_mean"] + candidates["pu_score_mean"]) / 2).clip(1e-6, 1 - 1e-6)
    )
    candidates["Degree_Residual_Score"] = candidates["topology_adjusted_residual"]
    candidates[["Gene", "Primary_Score", "Degree_Residual_Score", "STRING_Degree"]].to_csv(
        args.out / "GSEA_rankings.csv", index=False, encoding="utf-8-sig"
    )
    rankings = {
        "primary": candidates[["Gene", "Primary_Score"]].rename(columns={"Primary_Score": "Score"}).sort_values("Score", ascending=False),
        "degree_residualized": candidates[["Gene", "Degree_Residual_Score"]].rename(columns={"Degree_Residual_Score": "Score"}).sort_values("Score", ascending=False),
    }
    libraries = {"Hallmark": read_gmt(args.hallmark), "GO_BP": read_gmt(args.go_bp)}
    results = []
    for analysis, ranking in rankings.items():
        for library, sets in libraries.items():
            result_path = args.out / f"GSEA_{library}_{analysis}.csv"
            if args.resume and result_path.exists():
                results.append(pd.read_csv(result_path))
            else:
                results.append(run_one(ranking, sets, library, analysis, args.out))
    full = pd.concat(results, ignore_index=True)
    full.to_csv(args.out / "GSEA_all_results.csv", index=False, encoding="utf-8-sig")

    primary = full[(full["Analysis"].eq("primary")) & (full["Library"].eq("Hallmark"))].copy()
    fdr_col = next(c for c in primary.columns if str(c).lower().replace(" ", "").startswith("fdr"))
    nes_col = next(c for c in primary.columns if str(c).upper() == "NES")
    term_col = "Term" if "Term" in primary.columns else primary.columns[2]
    primary[fdr_col] = pd.to_numeric(primary[fdr_col], errors="coerce")
    primary[nes_col] = pd.to_numeric(primary[nes_col], errors="coerce")
    shown = primary.sort_values(fdr_col).head(15).sort_values(nes_col)
    configure_plotting()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = np.where(shown[nes_col] >= 0, COLORS["orange"], COLORS["blue"])
    ax.barh(np.arange(len(shown)), shown[nes_col], color=colors)
    ax.set(yticks=np.arange(len(shown)), yticklabels=shown[term_col].astype(str).str.replace("HALLMARK_", "", regex=False), xlabel="Normalized enrichment score")
    ax.axvline(0, color="0.5", linewidth=0.8)
    ax.set_title("Hallmark characterization of the PU-robust consensus")
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, args.out / "Hallmark_GSEA_primary")
    write_json(args.out / "GSEA_manifest.json", {
        "primary_ranking": "logit of the mean standard-LightGBM and 5:1 bagging-PU LightGBM OOF score among unlabelled genes",
        "sensitivity_ranking": "post hoc residual of consensus rank strength on log(1 + STRING degree) and RWR rank",
        "permutations": 2000, "seed": SEED, "minimum_gene_set_size": 15, "maximum_gene_set_size": 500,
        "libraries": {"Hallmark": args.hallmark.name, "GO_BP": args.go_bp.name},
        "KEGG_excluded": "KEGG contributed to pathway graph construction and is excluded to avoid circular characterization.",
        "input_sha256": {p.name: sha256(p) for p in [args.ranking, args.hallmark, args.go_bp]},
    })


if __name__ == "__main__":
    main()
