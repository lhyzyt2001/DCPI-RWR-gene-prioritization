from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ENDPOINTS = [
    ("TCGA paired\nexpression", "TCGA_expr_effect", "TCGA_expr_FDR"),
    ("TCGA late vs early\nstage", "TCGA_stage_effect", "TCGA_stage_FDR"),
    ("TCGA overall\nsurvival", "TCGA_log_HR", "TCGA_Cox_FDR"),
    ("GSE68465 tumor vs\nnormal expression", "GSE_expr_effect", "GSE_expr_FDR"),
    ("GSE68465 overall\nsurvival", "GSE_log_HR", "GSE_Cox_FDR"),
    ("CPTAC tumor protein\nprofile vs zero", "CPTAC_effect", "CPTAC_FDR"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate computational characterization of the new PU-robust candidate set.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.candidates).sort_values("Candidate_Rank").copy()

    tcga_expr = pd.read_csv(args.external_dir / "TCGA_paired_tumor_normal_top20.csv")
    tcga_stage = pd.read_csv(args.external_dir / "TCGA_stage_top20.csv")
    tcga_surv = pd.read_csv(args.external_dir / "TCGA_overall_survival_top20.csv")
    gse_expr = pd.read_csv(args.external_dir / "GSE68465_tumor_normal_top20.csv")
    gse_surv = pd.read_csv(args.external_dir / "GSE68465_survival_top20.csv")
    cptac = pd.read_csv(args.external_dir / "CPTAC" / "CPTAC_LUAD_PU_consensus_validation.csv")
    tables = [
        tcga_expr[["Gene", "Median_Paired_Difference", "Wilcoxon_FDR"]].rename(columns={"Median_Paired_Difference": "TCGA_expr_effect", "Wilcoxon_FDR": "TCGA_expr_FDR"}),
        tcga_stage[["Gene", "Late_Minus_Early_Median", "Mann_Whitney_FDR"]].rename(columns={"Late_Minus_Early_Median": "TCGA_stage_effect", "Mann_Whitney_FDR": "TCGA_stage_FDR"}),
        tcga_surv[["Gene", "Adjusted_HR_per_SD", "Adjusted_Cox_FDR"]].assign(TCGA_log_HR=lambda x: np.log(x["Adjusted_HR_per_SD"])).drop(columns="Adjusted_HR_per_SD").rename(columns={"Adjusted_Cox_FDR": "TCGA_Cox_FDR"}),
        gse_expr[["Gene", "Tumor_Minus_Normal_Median", "Mann_Whitney_FDR"]].rename(columns={"Tumor_Minus_Normal_Median": "GSE_expr_effect", "Mann_Whitney_FDR": "GSE_expr_FDR"}),
        gse_surv[["Gene", "Adjusted_HR_per_SD", "Adjusted_Cox_FDR"]].assign(GSE_log_HR=lambda x: np.log(x["Adjusted_HR_per_SD"])).drop(columns="Adjusted_HR_per_SD").rename(columns={"Adjusted_Cox_FDR": "GSE_Cox_FDR"}),
        cptac[["Gene", "Median_CPTAC_Profile_Value", "Wilcoxon_FDR"]].rename(columns={"Median_CPTAC_Profile_Value": "CPTAC_effect", "Wilcoxon_FDR": "CPTAC_FDR"}),
    ]
    integrated = base
    for table in tables:
        integrated = integrated.merge(table, on="Gene", how="left", validate="one_to_one")
    integrated["TCGA_GSE_expression_direction_concordant"] = np.sign(integrated["TCGA_expr_effect"]) == np.sign(integrated["GSE_expr_effect"])
    integrated["Significant_molecular_endpoints"] = (
        (integrated["TCGA_expr_FDR"] < 0.05).astype(int)
        + (integrated["TCGA_stage_FDR"] < 0.05).astype(int)
        + (integrated["GSE_expr_FDR"] < 0.05).astype(int)
        + (integrated["CPTAC_FDR"] < 0.05).astype(int)
    )
    integrated["Significant_adjusted_Cox_endpoints"] = (
        (integrated["TCGA_Cox_FDR"] < 0.05).astype(int) + (integrated["GSE_Cox_FDR"] < 0.05).astype(int)
    )
    integrated.to_csv(args.out / "PU_robust_top20_integrated_characterization.csv", index=False, encoding="utf-8-sig")

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for x, (label, effect_col, fdr_col) in enumerate(ENDPOINTS):
        effects = pd.to_numeric(integrated[effect_col], errors="coerce")
        fdr = pd.to_numeric(integrated[fdr_col], errors="coerce")
        direction = np.sign(effects)
        colors = np.where(direction > 0, "#D55E00", np.where(direction < 0, "#0072B2", "#999999"))
        sizes = np.clip(-np.log10(fdr.fillna(1).clip(lower=1e-6)) * 18 + 12, 12, 105)
        edges = np.where(fdr.fillna(1) < 0.05, "#111111", "none")
        ax.scatter(np.full(len(integrated), x), np.arange(len(integrated)), s=sizes, c=colors, edgecolors=edges, linewidths=0.8)
    ax.set_xticks(range(len(ENDPOINTS)), [x[0] for x in ENDPOINTS], rotation=25, ha="right")
    ax.set_yticks(range(len(integrated)), integrated["Gene"])
    ax.invert_yaxis()
    ax.set_title("Computational characterization of the PU-robust consensus candidates", loc="left")
    ax.set_xlabel("Endpoint (orange: positive effect; blue: negative effect; black edge: FDR < 0.05)")
    ax.grid(color="#ECECEC", linewidth=0.55)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "PU_robust_top20_characterization.pdf", bbox_inches="tight")
    fig.savefig(args.out / "PU_robust_top20_characterization.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "candidate_count": int(len(integrated)),
        "STRING_top5pct_hubs": int(integrated["STRING_Hub_Top5pct"].astype(bool).sum()),
        "TCGA_paired_expression_FDR_lt_0.05": int((integrated["TCGA_expr_FDR"] < 0.05).sum()),
        "GSE68465_expression_FDR_lt_0.05": int((integrated["GSE_expr_FDR"] < 0.05).sum()),
        "expression_direction_concordant": int(integrated["TCGA_GSE_expression_direction_concordant"].sum()),
        "TCGA_adjusted_Cox_FDR_lt_0.05": int((integrated["TCGA_Cox_FDR"] < 0.05).sum()),
        "GSE68465_adjusted_Cox_FDR_lt_0.05": int((integrated["GSE_Cox_FDR"] < 0.05).sum()),
        "CPTAC_profile_FDR_lt_0.05": int((integrated["CPTAC_FDR"] < 0.05).sum()),
        "DepMap_status": "Not re-used: only the former candidate-specific cache was available locally. A new candidate-level DepMap claim requires the complete Public 26Q1 gene-effect matrix or a fresh versioned download.",
        "interpretation": "All endpoints are post hoc computational characterization. Survival endpoints did not select candidates; CPTAC zero is not an adjacent-normal reference.",
    }
    (args.out / "candidate_characterization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
