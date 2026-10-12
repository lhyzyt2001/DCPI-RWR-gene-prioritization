from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests


API = "https://www.cbioportal.org/api"
PROFILE = "luad_cptac_2020_protein_quantification"
SAMPLE_LIST = "luad_cptac_2020_protein_quantification"
SEED = 20260812


def rank_biserial(values: np.ndarray) -> float:
    values = values[np.isfinite(values) & (values != 0)]
    if not len(values):
        return np.nan
    ranks = rankdata(np.abs(values))
    return float(np.sum(np.sign(values) * ranks) / np.sum(ranks))


def median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(2000, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch CPTAC LUAD protein profiles for a fixed candidate set.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidates)
    gene_col = "Gene_Symbol" if "Gene_Symbol" in candidates else "Gene"
    rank_col = "Consensus_Rank" if "Consensus_Rank" in candidates else "Candidate_Rank"
    genes = candidates.sort_values(rank_col).head(20)[gene_col].astype(str).tolist()
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    gene_response = requests.post(
        f"{API}/genes/fetch?geneIdType=HUGO_GENE_SYMBOL&projection=SUMMARY",
        json=genes, headers=headers, timeout=90,
    )
    gene_response.raise_for_status()
    gene_records = gene_response.json()
    gene_map = {record["hugoGeneSymbol"]: int(record["entrezGeneId"]) for record in gene_records}
    reverse = {value: key for key, value in gene_map.items()}
    data_response = requests.post(
        f"{API}/molecular-profiles/{PROFILE}/molecular-data/fetch?projection=SUMMARY",
        json={"sampleListId": SAMPLE_LIST, "entrezGeneIds": list(reverse)},
        headers=headers, timeout=180,
    )
    data_response.raise_for_status()
    raw = pd.DataFrame(data_response.json())
    if raw.empty:
        raw = pd.DataFrame(columns=["sampleId", "entrezGeneId", "value"])
    raw["Gene"] = raw["entrezGeneId"].map(reverse)
    raw.to_csv(args.out / "CPTAC_LUAD_current26_raw.csv.gz", index=False, compression="gzip")
    pd.DataFrame(gene_records).to_csv(args.out / "CPTAC_LUAD_gene_mapping.csv", index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(SEED)
    rows = []
    for rank, gene in enumerate(genes, 1):
        values = pd.to_numeric(raw.loc[raw["Gene"].eq(gene), "value"], errors="coerce").dropna().to_numpy()
        if len(values):
            try:
                p = float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError:
                p = 1.0
            lo, hi = median_ci(values, rng)
            rows.append({
                "Candidate_Rank": rank, "Gene": gene, "Status": "tested",
                "N_Protein_Profiles": len(values), "Median_CPTAC_Profile_Value": float(np.median(values)),
                "Median_CI_Lower": lo, "Median_CI_Upper": hi,
                "Signed_Rank_Biserial": rank_biserial(values), "Wilcoxon_P_vs_Zero": p,
            })
        else:
            rows.append({
                "Candidate_Rank": rank, "Gene": gene, "Status": "not returned by CPTAC profile",
                "N_Protein_Profiles": 0, "Median_CPTAC_Profile_Value": np.nan,
                "Median_CI_Lower": np.nan, "Median_CI_Upper": np.nan,
                "Signed_Rank_Biserial": np.nan, "Wilcoxon_P_vs_Zero": np.nan,
            })
    result = pd.DataFrame(rows)
    result["Wilcoxon_FDR"] = np.nan
    valid = result["Wilcoxon_P_vs_Zero"].notna()
    if valid.any():
        result.loc[valid, "Wilcoxon_FDR"] = multipletests(result.loc[valid, "Wilcoxon_P_vs_Zero"], method="fdr_bh")[1]
    result.to_csv(args.out / "CPTAC_LUAD_current26_validation.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "source": "cBioPortal public API",
        "study": "luad_cptac_2020",
        "profile": PROFILE,
        "sample_list": SAMPLE_LIST,
        "requested_genes": genes,
        "mapped_genes": len(gene_map),
        "genes_with_values": int((result["N_Protein_Profiles"] > 0).sum()),
        "test": "two-sided one-sample Wilcoxon signed-rank test against the study-specific zero reference; Benjamini-Hochberg correction across 20 candidates",
        "interpretation_boundary": "The zero reference is study specific and is not an adjacent-normal comparison.",
        "candidate_selection": "Fixed outcome-blind top 20 from mean out-of-fold RWR scores under the unified Open Targets 26.06 LUAD endpoint.",
        "retrieval_date": date.today().isoformat(),
    }
    (args.out / "CPTAC_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
