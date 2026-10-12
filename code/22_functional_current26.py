from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gseapy as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 20260926


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gmt(path: Path) -> dict[str, list[str]]:
    gene_sets: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.rstrip().split("\t")
        if len(fields) >= 3:
            gene_sets[fields[0]] = [gene.upper() for gene in fields[2:] if gene]
    return gene_sets


def break_exact_ties(scores: pd.Series, genes: pd.Series) -> tuple[pd.Series, int]:
    """Create a safe midrank statistic with reproducible within-tie ordering."""
    original = scores.to_numpy(dtype=float)
    values = (scores.rank(method="average", pct=True).to_numpy(dtype=float) * 2.0) - 1.0
    unique = np.sort(np.unique(original))
    tied_gene_count = 0
    for score in unique:
        locations = np.flatnonzero(original == score)
        if len(locations) <= 1:
            continue
        tied_gene_count += len(locations)
        order = sorted(
            locations,
            key=lambda index: hashlib.sha256(
                f"{SEED}|{genes.iloc[index]}".encode("utf-8")
            ).hexdigest(),
        )
        centre = float(values[locations[0]])
        replacements = centre + np.linspace(-1e-9, 1e-9, len(order), dtype=float)
        for index, replacement in zip(order, replacements):
            values[index] = replacement
    if pd.Series(values).duplicated().any():
        raise RuntimeError("Exact score ties remain after deterministic tie handling.")
    return pd.Series(values, index=scores.index), tied_gene_count


def run_prerank(
    ranking: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    library: str,
    analysis: str,
    output: Path,
) -> pd.DataFrame:
    result = gp.prerank(
        rnk=ranking[["Gene", "Score"]],
        gene_sets=gene_sets,
        min_size=15,
        max_size=500,
        permutation_num=2000,
        seed=SEED,
        threads=4,
        verbose=False,
        outdir=None,
    ).res2d.copy()
    result.insert(0, "Analysis", analysis)
    result.insert(1, "Library", library)
    result.to_csv(output / f"GSEA_{library}_{analysis}.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Characterize current 26.06 RWR rankings with Hallmark and GO-BP GSEA."
    )
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--hallmark", type=Path, required=True)
    parser.add_argument("--go-bp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ranked = pd.read_csv(args.ranking)
    gene_column = next((column for column in ("gene", "Gene", "Gene_Symbol") if column in ranked), None)
    if gene_column is None:
        raise ValueError("Ranking must contain gene, Gene, or Gene_Symbol.")
    required = {"RWR", "RWR_degree_residual"}
    missing = required.difference(ranked.columns)
    if missing:
        raise ValueError(f"Ranking is missing required columns: {sorted(missing)}")
    rankings = pd.DataFrame(
        {
            "Gene": ranked[gene_column].astype(str).str.upper(),
            "RWR": pd.to_numeric(ranked["RWR"], errors="coerce"),
            "RWR_degree_residual": pd.to_numeric(ranked["RWR_degree_residual"], errors="coerce"),
        }
    ).dropna()
    rankings = rankings.groupby("Gene", as_index=False).mean(numeric_only=True)
    rankings.to_csv(args.output / "current26_GSEA_rankings.csv", index=False, encoding="utf-8-sig")

    rank_tables = {}
    tie_counts = {}
    for analysis, column in {
        "primary_RWR": "RWR",
        "degree_residualized_RWR": "RWR_degree_residual",
    }.items():
        table = rankings[["Gene", column]].rename(columns={column: "Score"}).copy()
        table["Score"], tie_counts[analysis] = break_exact_ties(table["Score"], table["Gene"])
        rank_tables[analysis] = table.sort_values("Score", ascending=False)
    libraries = {
        "Hallmark": read_gmt(args.hallmark),
        "GO_BP": read_gmt(args.go_bp),
    }
    results = []
    for analysis, ranking in rank_tables.items():
        for library, gene_sets in libraries.items():
            result_path = args.output / f"GSEA_{library}_{analysis}.csv"
            if args.resume and result_path.exists():
                results.append(pd.read_csv(result_path))
            else:
                results.append(run_prerank(ranking, gene_sets, library, analysis, args.output))
    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(args.output / "current26_GSEA_all_results.csv", index=False, encoding="utf-8-sig")

    hallmark = combined[
        combined["Analysis"].eq("primary_RWR") & combined["Library"].eq("Hallmark")
    ].copy()
    fdr_column = next(column for column in hallmark if column.lower().replace(" ", "").startswith("fdr"))
    nes_column = next(column for column in hallmark if column.upper() == "NES")
    hallmark[fdr_column] = pd.to_numeric(hallmark[fdr_column], errors="coerce")
    hallmark[nes_column] = pd.to_numeric(hallmark[nes_column], errors="coerce")
    shown = hallmark.sort_values(fdr_column).head(15).sort_values(nes_column)
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
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    colors = np.where(shown[nes_column] >= 0, "#D55E00", "#0072B2")
    axis.barh(
        np.arange(len(shown)),
        shown[nes_column],
        color=colors,
    )
    labels = (
        shown["Term"]
        .astype(str)
        .str.replace("HALLMARK_", "", regex=False)
        .replace({"Pperoxisome": "Peroxisome", "heme Metabolism": "Heme Metabolism"})
    )
    axis.set(
        yticks=np.arange(len(shown)),
        yticklabels=labels,
        xlabel="Normalized enrichment score",
    )
    axis.axvline(0, color="0.45", linewidth=0.8)
    axis.set_title("Descriptive Hallmark characterization of the current 26.06 RWR ranking")
    axis.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(args.output / "current26_Hallmark_GSEA_primary.pdf", bbox_inches="tight")
    figure.savefig(args.output / "current26_Hallmark_GSEA_primary.png", dpi=600, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "endpoint": "Open Targets 26.06 disease-specific LUAD clinical-precedence evidence",
        "population": "Genes not labelled positive under the 26.06 endpoint",
        "primary_ranking": "Mean out-of-fold RWR score under the unified current-endpoint design",
        "sensitivity_ranking": "Residual of mean out-of-fold RWR score after cubic regression on log(1 + STRING degree)",
        "permutations": 2000,
        "seed": SEED,
        "minimum_gene_set_size": 15,
        "maximum_gene_set_size": 500,
        "exact_tie_handling": (
            "GSEA used a normalized midrank statistic. Exact-score ties retained the same practical midrank "
            "weight and were ordered by a SHA-256 hash of the gene symbol and fixed seed within a total "
            "perturbation range of 2e-9. The order of distinct scores was unchanged."
        ),
        "genes_in_exact_tie_groups": tie_counts,
        "KEGG_excluded": "KEGG contributed to graph construction and was excluded to avoid circular characterization.",
        "input_sha256": {
            path.name: sha256(path) for path in (args.ranking, args.hallmark, args.go_bp)
        },
    }
    (args.output / "current26_GSEA_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
