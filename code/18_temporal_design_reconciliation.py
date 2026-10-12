from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "RWR (historical seeds)": "#0072B2",
    "XGBoost historical stack": "#E69F00",
    "LightGBM historical stack": "#009E73",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile the two temporal designs and combine Top-K results.")
    parser.add_argument("--cross-design", type=Path, required=True)
    parser.add_argument("--cross-topk", type=Path, required=True)
    parser.add_argument("--date-design", type=Path, required=True)
    parser.add_argument("--date-topk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cross = json.loads(args.cross_design.read_text(encoding="utf-8"))
    date = json.loads(args.date_design.read_text(encoding="utf-8"))
    table = pd.DataFrame(
        [
            {
                "design_id": "cross_release_26.03_to_26.06",
                "historical_definition": cross["historical_release"],
                "future_definition": cross["future_release"],
                "historical_positive_count": cross["historical_positive_count"],
                "future_positive_count": cross["future_positive_count"],
                "incident_positive_count": cross["incident_positive_count"],
                "feature_snapshot_fully_historical": False,
                "interpretation": "Cross-release replication; release-pipeline and ontology changes can contribute.",
            },
            {
                "design_id": "within_26.06_evidence_date",
                "historical_definition": date["historical_release"],
                "future_definition": date["future_release"],
                "historical_positive_count": date["historical_positive_count"],
                "future_positive_count": date["future_positive_count"],
                "incident_positive_count": date["incident_positive_count"],
                "feature_snapshot_fully_historical": False,
                "interpretation": "Fixed-schema date-split sensitivity; low powered and not prospective.",
            },
        ]
    )
    table.to_csv(args.output / "temporal_design_reconciliation.csv", index=False, encoding="utf-8-sig")

    cross_topk = pd.read_csv(args.cross_topk)
    cross_topk.insert(0, "design_id", "cross_release_26.03_to_26.06")
    date_topk = pd.read_csv(args.date_topk)
    date_topk.insert(0, "design_id", "within_26.06_evidence_date")
    topk = pd.concat([cross_topk, date_topk], ignore_index=True)
    topk.to_csv(args.output / "temporal_topk_reconciliation.csv", index=False, encoding="utf-8-sig")

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
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    titles = {
        "cross_release_26.03_to_26.06": "a  Cross-release: 120 incident genes",
        "within_26.06_evidence_date": "b  Within-release date split: 14 incident genes",
    }
    for axis, design_id in zip(axes, titles):
        block = topk[topk["design_id"] == design_id]
        for method, part in block.groupby("method"):
            part = part.sort_values("K")
            axis.plot(
                part["K"],
                part["incident_recall_at_k_mean"],
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                color=COLORS[method],
                label=method.replace(" (historical seeds)", "").replace(" historical stack", ""),
            )
        axis.set_xscale("log")
        axis.set_xlabel("Top K")
        axis.set_title(titles[design_id], loc="left")
        axis.grid(color="#E5E5E5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Incident-gene recall at K")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.output / "temporal_design_topk.pdf", bbox_inches="tight")
    fig.savefig(args.output / "temporal_design_topk.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    explanation = {
        "critical_correction": "The 120 and 14 incident-gene counts are not successive attrition stages.",
        "cross_release_120": "Genes present in the 26.06 disease-specific set but absent from the 26.03 disease-specific set.",
        "within_release_14": "Genes whose earliest dated clinical-precedence evidence in the 26.06 fixed-schema data occurs after 2016-12-31.",
        "why_counts_differ": "The two designs use different historical definitions and answer different sensitivity questions.",
        "claim_boundary": "Neither design is called fully prospective because graph and expression feature snapshots were not independently frozen at the historical cutoff.",
    }
    (args.output / "temporal_design_explanation.json").write_text(
        json.dumps(explanation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
