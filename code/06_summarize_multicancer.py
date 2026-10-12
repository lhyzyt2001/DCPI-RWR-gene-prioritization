from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    result = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * float(p[index])))
        result[index] = running
    return result


def random_effects(effect: np.ndarray, se: np.ndarray) -> dict:
    variance = se**2
    fixed_weight = 1 / variance
    fixed = np.sum(fixed_weight * effect) / np.sum(fixed_weight)
    q = np.sum(fixed_weight * (effect - fixed) ** 2)
    df = len(effect) - 1
    c = np.sum(fixed_weight) - np.sum(fixed_weight**2) / np.sum(fixed_weight)
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    weight = 1 / (variance + tau2)
    pooled = np.sum(weight * effect) / np.sum(weight)
    pooled_se = np.sqrt(1 / np.sum(weight))
    return {
        "studies": len(effect),
        "pooled_delta_AUPRC": float(pooled),
        "ci_lower": float(pooled - 1.96 * pooled_se),
        "ci_upper": float(pooled + 1.96 * pooled_se),
        "tau_squared": float(tau2),
        "Q": float(q),
        "I_squared_percent": float(max(0.0, (q - df) / q) * 100 if q > 0 else 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--cross-release", type=Path, required=True)
    parser.add_argument("--fixed-schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counts = pd.read_csv(args.labels / "multicancer_label_counts.csv")
    rows = []
    for cancer in counts["cancer"]:
        summary = pd.read_csv(args.cross_release / cancer / "temporal_validation_summary.csv")
        boot = pd.read_csv(args.cross_release / cancer / "temporal_validation_gene_cluster_bootstrap.csv")
        for row in boot.itertuples(index=False):
            rows.append(
                {
                    "cancer": cancer,
                    "method": row.method,
                    "reference": row.reference,
                    "delta_AUPRC": row.median_delta_incident_AUPRC,
                    "ci_lower": row.ci_lower,
                    "ci_upper": row.ci_upper,
                    "p_two_sided": row.p_two_sided,
                    "method_AUPRC": float(summary.loc[summary["method"].eq(row.method), "AUPRC_mean"].iloc[0]),
                    "reference_AUPRC": float(summary.loc[summary["method"].eq(row.reference), "AUPRC_mean"].iloc[0]),
                }
            )
    result = pd.DataFrame(rows).merge(counts, on="cancer", how="left")
    result["p_holm_within_comparison"] = result.groupby(["method", "reference"])["p_two_sided"].transform(
        lambda x: holm(x.to_numpy())
    )
    result["approx_se"] = (result["ci_upper"] - result["ci_lower"]) / (2 * 1.96)
    meta_rows = []
    for (method, reference), group in result.groupby(["method", "reference"]):
        meta_rows.append({"method": method, "reference": reference, **random_effects(group["delta_AUPRC"].to_numpy(), group["approx_se"].to_numpy())})
    meta = pd.DataFrame(meta_rows)
    result.to_csv(args.output / "multicancer_cross_release_results.csv", index=False)
    meta.to_csv(args.output / "multicancer_random_effects_meta_analysis.csv", index=False)

    fixed_summary = pd.read_csv(args.fixed_schema / "temporal_validation_summary.csv")
    fixed_boot = pd.read_csv(args.fixed_schema / "temporal_validation_gene_cluster_bootstrap.csv")
    fixed_summary.to_csv(args.output / "LUAD_fixed_schema_date_split_summary.csv", index=False)
    fixed_boot.to_csv(args.output / "LUAD_fixed_schema_date_split_bootstrap.csv", index=False)

    focus = result[result["reference"].eq("RWR (historical seeds)")].copy()
    methods = focus["method"].unique().tolist()
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(methods), figsize=(7.2, 3.4), sharey=True, gridspec_kw={"wspace": 0.15})
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        block = focus[focus["method"].eq(method)].sort_values("cancer")
        y = np.arange(len(block))
        estimate = block["delta_AUPRC"].to_numpy()
        ax.errorbar(
            estimate,
            y,
            xerr=[estimate - block["ci_lower"].to_numpy(), block["ci_upper"].to_numpy() - estimate],
            fmt="o",
            color="#0072B2",
            capsize=2,
        )
        pooled = meta[(meta["method"].eq(method)) & (meta["reference"].eq("RWR (historical seeds)"))].iloc[0]
        ax.axvspan(pooled["ci_lower"], pooled["ci_upper"], color="#E69F00", alpha=0.18)
        ax.axvline(pooled["pooled_delta_AUPRC"], color="#E69F00", lw=1.2)
        ax.axvline(0, color="#666666", ls="--", lw=0.8)
        ax.set_yticks(y, block["cancer"])
        ax.set_title(method.replace(" historical stack", ""))
        ax.set_xlabel("Incident-target ΔAUPRC vs RWR")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#E5E5E5", lw=0.6)
    fig.suptitle("Disease-specific cross-release replication", y=1.01)
    fig.savefig(args.output / "multicancer_cross_release_forest.pdf", bbox_inches="tight")
    fig.savefig(args.output / "multicancer_cross_release_forest.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    manifest = {
        "design": "Open Targets 26.03-to-26.06 disease-specific clinical-precedence cross-release replication",
        "cancers": counts["cancer"].tolist(),
        "multiplicity": "Holm correction across cancers within each method-reference comparison",
        "meta_analysis": "DerSimonian-Laird random effects using CI-derived standard errors",
        "boundary": "Not a strict prospective validation because release 26.06 changed clinical-mining and disease-ontology pipelines.",
        "fixed_schema_boundary": "The within-26.06 evidence-date split has few incident LUAD targets and is reported as low-powered sensitivity evidence.",
    }
    (args.output / "multicancer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(meta.to_string(index=False))


if __name__ == "__main__":
    main()
