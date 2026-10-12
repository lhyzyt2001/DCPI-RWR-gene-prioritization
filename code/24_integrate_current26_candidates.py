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
    ("CPTAC protein\nprofile vs zero", "CPTAC_effect", "CPTAC_FDR"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate current-26.06 candidate characterization.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.candidates).sort_values("Candidate_Rank").copy()

    tcga_expr = pd.read_csv(args.external_dir / "TCGA_paired_tumor_normal_top20.csv")
    tcga_stage = pd.read_csv(args.external_dir / "TCGA_stage_top20.csv")
    tcga_surv = pd.read_csv(args.external_dir / "TCGA_overall_survival_top20.csv")
    gse_expr = pd.read_csv(args.external_dir / "GSE68465_tumor_normal_top20.csv")
    gse_surv = pd.read_csv(args.external_dir / "GSE68465_survival_top20.csv")
    cptac = pd.read_csv(args.external_dir / "CPTAC_LUAD_current26_validation.csv")
    tables = [
        tcga_expr[["Gene", "Median_Paired_Difference", "Wilcoxon_FDR"]].rename(
            columns={"Median_Paired_Difference": "TCGA_expr_effect", "Wilcoxon_FDR": "TCGA_expr_FDR"}
        ),
        tcga_stage[["Gene", "Late_Minus_Early_Median", "Mann_Whitney_FDR"]].rename(
            columns={"Late_Minus_Early_Median": "TCGA_stage_effect", "Mann_Whitney_FDR": "TCGA_stage_FDR"}
        ),
        tcga_surv[["Gene", "Adjusted_HR_per_SD", "Adjusted_Cox_FDR"]]
        .assign(TCGA_log_HR=lambda frame: np.log(frame["Adjusted_HR_per_SD"]))
        .drop(columns="Adjusted_HR_per_SD")
        .rename(columns={"Adjusted_Cox_FDR": "TCGA_Cox_FDR"}),
        gse_expr[["Gene", "Tumor_Minus_Normal_Median", "Mann_Whitney_FDR"]].rename(
            columns={"Tumor_Minus_Normal_Median": "GSE_expr_effect", "Mann_Whitney_FDR": "GSE_expr_FDR"}
        ),
        gse_surv[["Gene", "Adjusted_HR_per_SD", "Adjusted_Cox_FDR"]]
        .assign(GSE_log_HR=lambda frame: np.log(frame["Adjusted_HR_per_SD"]))
        .drop(columns="Adjusted_HR_per_SD")
        .rename(columns={"Adjusted_Cox_FDR": "GSE_Cox_FDR"}),
        cptac[["Gene", "Median_CPTAC_Profile_Value", "Wilcoxon_FDR"]].rename(
            columns={"Median_CPTAC_Profile_Value": "CPTAC_effect", "Wilcoxon_FDR": "CPTAC_FDR"}
        ),
    ]
    integrated = base
    for table in tables:
        integrated = integrated.merge(table, on="Gene", how="left", validate="one_to_one")
    both_expression = integrated["TCGA_expr_effect"].notna() & integrated["GSE_expr_effect"].notna()
    integrated["TCGA_GSE_expression_direction_concordant"] = False
    integrated.loc[both_expression, "TCGA_GSE_expression_direction_concordant"] = (
        np.sign(integrated.loc[both_expression, "TCGA_expr_effect"])
        == np.sign(integrated.loc[both_expression, "GSE_expr_effect"])
    )
    integrated["Significant_molecular_endpoints"] = (
        (integrated["TCGA_expr_FDR"] < 0.05).astype(int)
        + (integrated["TCGA_stage_FDR"] < 0.05).astype(int)
        + (integrated["GSE_expr_FDR"] < 0.05).astype(int)
        + (integrated["CPTAC_FDR"] < 0.05).astype(int)
    )
    integrated["Significant_adjusted_Cox_endpoints"] = (
        (integrated["TCGA_Cox_FDR"] < 0.05).astype(int)
        + (integrated["GSE_Cox_FDR"] < 0.05).astype(int)
    )
    integrated.to_csv(
        args.output / "current26_RWR_top20_integrated_characterization.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for x_position, (_, effect_column, fdr_column) in enumerate(ENDPOINTS):
        effects = pd.to_numeric(integrated[effect_column], errors="coerce")
        fdr = pd.to_numeric(integrated[fdr_column], errors="coerce")
        colors = np.where(effects > 0, "#D55E00", np.where(effects < 0, "#0072B2", "#999999"))
        sizes = np.clip(-np.log10(fdr.fillna(1).clip(lower=1e-6)) * 18 + 12, 12, 105)
        edges = np.where(fdr.fillna(1) < 0.05, "#111111", "none")
        axis.scatter(
            np.full(len(integrated), x_position),
            np.arange(len(integrated)),
            s=sizes,
            c=colors,
            edgecolors=edges,
            linewidths=0.8,
        )
    axis.set_xticks(range(len(ENDPOINTS)), [item[0] for item in ENDPOINTS], rotation=25, ha="right")
    axis.set_yticks(range(len(integrated)), integrated["Gene"])
    axis.invert_yaxis()
    axis.set_title("External characterization of the current 26.06 RWR top 20", loc="left")
    axis.set_xlabel("Orange: positive effect; blue: negative effect; black edge: FDR < 0.05")
    axis.grid(color="#ECECEC", linewidth=0.55)
    axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(args.output / "current26_RWR_top20_characterization.pdf", bbox_inches="tight")
    figure.savefig(args.output / "current26_RWR_top20_characterization.png", dpi=600, bbox_inches="tight")
    plt.close(figure)

    tcga_ph = pd.read_csv(args.external_dir / "TCGA_schoenfeld_diagnostics.csv")
    gse_ph = pd.read_csv(args.external_dir / "GSE68465_schoenfeld_diagnostics.csv")
    summary = {
        "candidate_count": int(len(integrated)),
        "STRING_top5pct_hubs": int(integrated["STRING_hub_top5pct"].astype(bool).sum()),
        "TCGA_mapped": int(integrated["TCGA_expr_effect"].notna().sum()),
        "GSE68465_mapped": int(integrated["GSE_expr_effect"].notna().sum()),
        "CPTAC_mapped": int(integrated["CPTAC_effect"].notna().sum()),
        "TCGA_paired_expression_FDR_lt_0.05": int((integrated["TCGA_expr_FDR"] < 0.05).sum()),
        "TCGA_stage_FDR_lt_0.05": int((integrated["TCGA_stage_FDR"] < 0.05).sum()),
        "GSE68465_expression_FDR_lt_0.05": int((integrated["GSE_expr_FDR"] < 0.05).sum()),
        "expression_direction_concordant_among_mapped": int(
            integrated.loc[both_expression, "TCGA_GSE_expression_direction_concordant"].sum()
        ),
        "TCGA_adjusted_Cox_FDR_lt_0.05": int((integrated["TCGA_Cox_FDR"] < 0.05).sum()),
        "GSE68465_adjusted_Cox_FDR_lt_0.05": int((integrated["GSE_Cox_FDR"] < 0.05).sum()),
        "CPTAC_profile_FDR_lt_0.05": int((integrated["CPTAC_FDR"] < 0.05).sum()),
        "TCGA_expression_term_PH_FDR_lt_0.05": int(
            ((tcga_ph["Covariate"] == "expression_z") & (tcga_ph["PH_FDR"] < 0.05)).sum()
        ),
        "GSE68465_expression_term_PH_FDR_lt_0.05": int(
            ((gse_ph["Covariate"] == "expression_z") & (gse_ph["PH_FDR"] < 0.05)).sum()
        ),
        "interpretation": (
            "All external endpoints are post hoc characterization and did not select candidates. "
            "CPTAC zero is study specific rather than an adjacent-normal reference; Cox estimates with "
            "expression-term PH violations are average associations over follow-up."
        ),
    }
    (args.output / "current26_candidate_characterization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
