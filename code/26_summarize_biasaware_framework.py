from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import binomtest, hypergeom
from statsmodels.stats.multitest import multipletests


CANCERS = ["LUAD", "BRCA", "CRC", "PRAD", "MEL", "OV"]
METHODS = ["RWR-STRING", "DCPI-RWR performance", "DCPI-RWR bias-aware"]
COLORS = {
    "RWR-STRING": "#0072B2",
    "DCPI-RWR performance": "#D55E00",
    "DCPI-RWR bias-aware": "#009E73",
}


def random_effects(effect: np.ndarray, se: np.ndarray) -> dict[str, float]:
    variance = se**2
    fixed_weight = 1.0 / variance
    fixed = float(np.sum(fixed_weight * effect) / np.sum(fixed_weight))
    q = float(np.sum(fixed_weight * (effect - fixed) ** 2))
    df = len(effect) - 1
    c_value = float(np.sum(fixed_weight) - np.sum(fixed_weight**2) / np.sum(fixed_weight))
    tau2 = max(0.0, (q - df) / c_value) if c_value > 0 else 0.0
    weight = 1.0 / (variance + tau2)
    pooled = float(np.sum(weight * effect) / np.sum(weight))
    pooled_se = float(np.sqrt(1.0 / np.sum(weight)))
    i2 = max(0.0, (q - df) / q * 100.0) if q > 0 else 0.0
    return {
        "pooled_delta_AUPRC": pooled,
        "ci_lower": pooled - 1.96 * pooled_se,
        "ci_upper": pooled + 1.96 * pooled_se,
        "tau_squared": tau2,
        "I_squared_percent": i2,
    }


def integrate_candidates(candidate_file: Path, external: Path) -> pd.DataFrame:
    frame = pd.read_csv(candidate_file)
    tcga = pd.read_csv(external / "TCGA_paired_tumor_normal_top20.csv")[
        ["Gene", "Median_Paired_Difference", "Wilcoxon_FDR"]
    ].rename(
        columns={
            "Median_Paired_Difference": "TCGA_RNA_tumor_minus_normal",
            "Wilcoxon_FDR": "TCGA_RNA_FDR",
        }
    )
    gse = pd.read_csv(external / "GSE68465_tumor_normal_top20.csv")[
        ["Gene", "Tumor_Minus_Normal_Median", "Mann_Whitney_FDR"]
    ].rename(
        columns={
            "Tumor_Minus_Normal_Median": "GSE68465_RNA_tumor_minus_normal",
            "Mann_Whitney_FDR": "GSE68465_RNA_FDR",
        }
    )
    stage = pd.read_csv(external / "TCGA_stage_top20.csv")[["Gene", "Late_Minus_Early_Median", "Mann_Whitney_FDR"]].rename(
        columns={"Mann_Whitney_FDR": "TCGA_stage_FDR"}
    )
    tcga_os = pd.read_csv(external / "TCGA_overall_survival_top20.csv")[["Gene", "Adjusted_HR_per_SD", "Adjusted_Cox_FDR"]].rename(
        columns={"Adjusted_HR_per_SD": "TCGA_OS_HR_per_SD", "Adjusted_Cox_FDR": "TCGA_OS_FDR"}
    )
    gse_os = pd.read_csv(external / "GSE68465_survival_top20.csv")[["Gene", "Adjusted_HR_per_SD", "Adjusted_Cox_FDR"]].rename(
        columns={"Adjusted_HR_per_SD": "GSE68465_OS_HR_per_SD", "Adjusted_Cox_FDR": "GSE68465_OS_FDR"}
    )
    cptac = pd.read_csv(external / "CPTAC_LUAD_current26_validation.csv")[["Gene", "Median_CPTAC_Profile_Value", "Wilcoxon_FDR"]].rename(
        columns={"Wilcoxon_FDR": "CPTAC_FDR"}
    )
    for table in (tcga, gse, stage, tcga_os, gse_os, cptac):
        frame = frame.merge(table, on="Gene", how="left")
    valid = frame["TCGA_RNA_tumor_minus_normal"].notna() & frame["GSE68465_RNA_tumor_minus_normal"].notna()
    frame["RNA_direction_concordant"] = np.where(
        valid,
        np.sign(frame["TCGA_RNA_tumor_minus_normal"]) == np.sign(frame["GSE68465_RNA_tumor_minus_normal"]),
        np.nan,
    )
    frame["both_RNA_cohorts_FDR_lt_0.05"] = (
        (frame["TCGA_RNA_FDR"] < 0.05)
        & (frame["GSE68465_RNA_FDR"] < 0.05)
        & frame["RNA_direction_concordant"].eq(True)
    )
    frame["independent_significant_endpoints"] = (
        (frame["TCGA_RNA_FDR"] < 0.05).astype(int)
        + (frame["GSE68465_RNA_FDR"] < 0.05).astype(int)
        + (frame["TCGA_stage_FDR"] < 0.05).astype(int)
        + (frame["TCGA_OS_FDR"] < 0.05).astype(int)
        + (frame["GSE68465_OS_FDR"] < 0.05).astype(int)
        + (frame["CPTAC_FDR"] < 0.05).astype(int)
    )
    return frame


def plot_summary(metrics: pd.DataFrame, topk: pd.DataFrame, boot: pd.DataFrame, candidates: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))
    perf = boot[boot.method == "DCPI-RWR performance"].set_index("cancer").loc[CANCERS]
    y = np.arange(len(CANCERS))
    effect = perf["median_delta_AUPRC"].to_numpy(float)
    lower = perf["ci_lower"].to_numpy(float)
    upper = perf["ci_upper"].to_numpy(float)
    axes[0, 0].errorbar(effect, y, xerr=np.vstack([effect - lower, upper - effect]), fmt="o", color="#D55E00", capsize=2)
    axes[0, 0].axvline(0, color="#555555", linewidth=1, linestyle="--")
    axes[0, 0].set(yticks=y, yticklabels=CANCERS, xlabel="ΔAUPRC vs STRING-RWR")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title("a  Time-separated global discrimination", loc="left")

    block = topk[(topk.K == 100) & topk.method.isin(METHODS)]
    x = np.arange(len(CANCERS))
    width = 0.25
    for index, method in enumerate(METHODS):
        values = block[block.method == method].set_index("cancer").loc[CANCERS, "incident_hits"]
        axes[0, 1].bar(x + (index - 1) * width, values, width, color=COLORS[method], label=method)
    axes[0, 1].set(xticks=x, xticklabels=CANCERS, ylabel="Incident genes recovered in top 100")
    axes[0, 1].set_title("b  Fixed-budget recovery", loc="left")
    axes[0, 1].legend(frameon=False, fontsize=7)

    for index, method in enumerate(METHODS):
        values = block[block.method == method].set_index("cancer").loc[CANCERS, "hub_fraction"]
        axes[1, 0].bar(x + (index - 1) * width, values, width, color=COLORS[method])
    axes[1, 0].set(xticks=x, xticklabels=CANCERS, ylabel="STRING top-5% hub fraction", ylim=(0, 1))
    axes[1, 0].set_title("c  Centrality burden in top 100", loc="left")

    endpoints = [
        ("TCGA_RNA_tumor_minus_normal", "TCGA RNA"),
        ("GSE68465_RNA_tumor_minus_normal", "GSE68465 RNA"),
        ("Median_CPTAC_Profile_Value", "CPTAC protein"),
    ]
    matrix = np.column_stack([np.sign(candidates[column].to_numpy(float)) for column, _ in endpoints])
    masked = np.ma.masked_invalid(matrix)
    cmap = mpl.colors.ListedColormap(["#0072B2", "#F7F7F7", "#D55E00"])
    axes[1, 1].imshow(masked, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    axes[1, 1].set(
        xticks=np.arange(len(endpoints)),
        xticklabels=[label for _, label in endpoints],
        yticks=np.arange(len(candidates)),
        yticklabels=candidates["Gene"],
    )
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[1, 1].set_title("d  Exploratory molecular direction", loc="left")
    axes[1, 1].legend(
        handles=[
            Patch(facecolor="#0072B2", label="Negative direction"),
            Patch(facecolor="#D55E00", label="Positive direction"),
            Patch(facecolor="#F7F7F7", edgecolor="#777777", label="Unavailable"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=3,
        frameon=False,
        fontsize=6.5,
        handlelength=1.2,
        columnspacing=0.9,
    )
    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
        if axis is not axes[1, 1]:
            axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output / "DCPI_RWR_bias_performance_biology_summary.pdf", bbox_inches="tight")
    fig.savefig(output / "DCPI_RWR_bias_performance_biology_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the degree-conditioned pathway-integrated RWR framework.")
    parser.add_argument("--dc-results", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.dc_results / "DCPI_RWR_multicancer_cross_release_metrics.csv")
    topk = pd.read_csv(args.dc_results / "DCPI_RWR_multicancer_topk_and_hubs.csv")
    boot = pd.read_csv(args.dc_results / "DCPI_RWR_multicancer_bootstrap.csv")
    paired = pd.read_csv(args.dc_results / "DCPI_RWR_top100_paired_incident_analysis.csv")
    boot["p_holm_all_comparisons"] = multipletests(boot["p_two_sided"], method="holm")[1]
    boot.to_csv(args.output / "DCPI_RWR_multicancer_bootstrap_with_holm.csv", index=False, encoding="utf-8-sig")
    performance = boot[boot.method == "DCPI-RWR performance"].copy()
    performance["approx_se"] = (performance["ci_upper"] - performance["ci_lower"]) / (2 * 1.96)
    meta = random_effects(performance["median_delta_AUPRC"].to_numpy(float), performance["approx_se"].to_numpy(float))
    pd.DataFrame([{**meta, "method": "DCPI-RWR performance", "reference": "RWR-STRING", "cancers": 6}]).to_csv(
        args.output / "DCPI_RWR_random_effects_meta_analysis.csv", index=False, encoding="utf-8-sig"
    )
    candidates = integrate_candidates(
        args.dc_results / "DCPI_RWR_bias_aware_current26_locked_top20.csv", args.external
    )
    candidates.to_csv(args.output / "DCPI_RWR_bias_aware_top20_integrated_evidence.csv", index=False, encoding="utf-8-sig")
    top100 = topk[(topk.K == 100) & topk.method.isin(METHODS)]
    grouped = top100.groupby("method", as_index=False).agg(
        cancers=("cancer", "nunique"),
        total_incident_hits=("incident_hits", "sum"),
        mean_hub_fraction=("hub_fraction", "mean"),
        median_STRING_degree=("median_STRING_degree", "median"),
    )
    grouped.to_csv(args.output / "DCPI_RWR_top100_cross_cancer_summary.csv", index=False, encoding="utf-8-sig")
    enrichment_rows = []
    top100_all = topk[topk.K == 100].copy()
    metric_lookup = metrics.set_index(["cancer", "method"])
    for row in top100_all.itertuples(index=False):
        metric = metric_lookup.loc[(row.cancer, row.method)]
        population = int(metric.candidate_universe)
        positives = int(metric.incident_positives)
        hits = int(row.incident_hits)
        enrichment_rows.append(
            {
                "cancer": row.cancer,
                "method": row.method,
                "K": 100,
                "incident_hits": hits,
                "incident_prevalence": positives / population,
                "precision_at_100": hits / 100,
                "fold_enrichment_over_prevalence": (hits / 100) / (positives / population),
                "hypergeometric_p_greater_equal": float(hypergeom.sf(hits - 1, population, positives, 100)),
            }
        )
    enrichment = pd.DataFrame(enrichment_rows)
    enrichment["p_holm_all_top100_tests"] = multipletests(enrichment.hypergeometric_p_greater_equal, method="holm")[1]
    enrichment.to_csv(args.output / "DCPI_RWR_top100_enrichment.csv", index=False, encoding="utf-8-sig")
    pivot_hubs = top100.pivot(index="cancer", columns="method", values="hub_fraction")
    hub_reductions = pivot_hubs["DCPI-RWR performance"] < pivot_hubs["RWR-STRING"]
    hub_direction_p = float(binomtest(int(hub_reductions.sum()), len(hub_reductions), 0.5).pvalue)
    valid_rna = candidates["RNA_direction_concordant"].notna()
    summary = {
        "positioning": "Bias-performance audit and time-separated candidate prioritization; not a claim of universal superiority or causal target discovery.",
        "performance_meta_analysis": meta,
        "top100_cross_cancer": grouped.to_dict("records"),
        "top100_paired_pooled": paired[paired.cancer == "POOLED_DESCRIPTIVE"].to_dict("records")[0],
        "top100_hub_reduction_direction": {
            "cancers_with_lower_hub_fraction": int(hub_reductions.sum()),
            "cancers_evaluated": int(len(hub_reductions)),
            "two_sided_sign_test_p": hub_direction_p,
        },
        "candidate_biology": {
            "RNA_direction_concordant": int(candidates.loc[valid_rna, "RNA_direction_concordant"].sum()),
            "RNA_direction_evaluable": int(valid_rna.sum()),
            "concordant_and_FDR_significant_in_both_RNA_cohorts": int(candidates["both_RNA_cohorts_FDR_lt_0.05"].sum()),
            "TCGA_survival_FDR_significant": int((candidates["TCGA_OS_FDR"] < 0.05).sum()),
            "GSE68465_survival_FDR_significant": int((candidates["GSE68465_OS_FDR"] < 0.05).sum()),
        },
        "hard_boundary": "External molecular and clinical analyses are post hoc characterization of a locked list and do not establish causality or therapeutic efficacy.",
    }
    (args.output / "DCPI_RWR_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_summary(metrics, topk, boot, candidates, args.output)


if __name__ == "__main__":
    main()
