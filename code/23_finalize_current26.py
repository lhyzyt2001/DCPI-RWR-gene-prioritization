from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def any_true(values: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(values):
        return bool(values.fillna(False).any())
    return bool(values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).any())


def release_file(path: Path, root: Path) -> bool:
    excluded = {"__pycache__", "GCNII_repeat12", "GATv2_repeat2"}
    return not any(part in excluded for part in path.relative_to(root).parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize and audit the unified current-26.06 package.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results"
    combined_dir = results / "12_current26_unified" / "combined"
    qa_dir = results / "12_current26_unified" / "QA"
    qa_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(combined_dir / "current26_all_model_summary.csv")
    bootstrap = pd.read_csv(combined_dir / "current26_all_model_gene_cluster_bootstrap.csv")
    fold_metrics = pd.read_csv(combined_dir / "current26_all_model_fold_metrics.csv")
    deep_audit = pd.read_csv(combined_dir / "current26_all_deep_training_audit.csv")
    fold_assignments = pd.read_csv(combined_dir / "current26_fold_assignments.csv.gz")
    candidate = pd.read_csv(results / "14_current26_candidates" / "current26_locked_RWR_top20.csv")
    candidate_adjusted = pd.read_csv(
        results / "14_current26_candidates" / "current26_topology_adjusted_sensitivity_top20.csv"
    )
    temporal = pd.read_csv(results / "13_temporal_reconciliation" / "temporal_design_reconciliation.csv")
    gsea = pd.read_csv(results / "16_current26_functional" / "current26_GSEA_all_results.csv")

    method_count = summary["method"].nunique()
    methods = sorted(summary["method"].astype(str).unique())
    expected_methods = {
        "RWR",
        "Network-degree logistic",
        "XGBoost + RWR/features",
        "LightGBM + RWR/features",
        "APPNP",
        "GCNII",
        "GATv2",
    }
    current_positives = int(fold_assignments.groupby("node_id")["label"].first().sum())
    temporal_designs = set(temporal["design_id"].astype(str))
    gsea_designs = gsea[["Analysis", "Library"]].drop_duplicates().shape[0]

    checks = {
        "current_endpoint_positive_count_is_197": current_positives == 197,
        "unified_benchmark_has_expected_7_methods": set(methods) == expected_methods and method_count == 7,
        "unified_benchmark_has_105_method_fold_rows": len(fold_metrics) == 105,
        "unified_benchmark_has_15_outer_folds": fold_metrics[["repeat", "fold"]].drop_duplicates().shape[0] == 15,
        "deep_has_135_uniform_fits": len(deep_audit) == 135,
        "deep_fit_identifiers_unique": not deep_audit[["repeat", "fold", "method", "seed_label"]].duplicated().any(),
        "deep_outer_test_labels_not_used": not any_true(deep_audit["outer_test_labels_used_for_training"]),
        "deep_outer_test_nodes_not_RWR_seeds": not any_true(deep_audit["outer_test_nodes_used_as_rwr_seeds"]),
        "fold_assignments_cover_three_repeats": len(fold_assignments) == 25242 * 3,
        "candidate_count_is_20": len(candidate) == 20,
        "topology_adjusted_candidate_count_is_20": len(candidate_adjusted) == 20,
        "temporal_designs_are_explicitly_separate": temporal_designs
        == {"cross_release_26.03_to_26.06", "within_26.06_evidence_date"},
        "four_current_endpoint_GSEA_tables_present": gsea_designs == 4,
        "RWR_is_in_bootstrap_reference_comparisons": set(bootstrap["reference"].astype(str)) == {"RWR"},
    }

    figure_rows = []
    new_result_dirs = [
        results / "12_current26_unified",
        results / "13_temporal_reconciliation",
        results / "14_current26_candidates",
        results / "15_current26_external_characterization",
        results / "16_current26_functional",
    ]
    for directory in new_result_dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("*.png"):
            with Image.open(path) as image:
                dpi = image.info.get("dpi", (None, None))
                figure_rows.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "width_px": image.width,
                        "height_px": image.height,
                        "dpi_x": dpi[0],
                        "dpi_y": dpi[1],
                        "pdf_counterpart": path.with_suffix(".pdf").exists(),
                    }
                )
    pd.DataFrame(figure_rows).to_csv(
        qa_dir / "current26_figure_pixel_and_dpi_audit.csv", index=False, encoding="utf-8-sig"
    )
    checks["all_new_png_figures_at_least_300_dpi"] = bool(figure_rows) and all(
        row["dpi_x"] is not None
        and row["dpi_y"] is not None
        and row["dpi_x"] >= 299
        and row["dpi_y"] >= 299
        for row in figure_rows
    )
    checks["all_new_png_figures_have_vector_pdf_counterparts"] = bool(figure_rows) and all(
        row["pdf_counterpart"] for row in figure_rows
    )

    compact_summary = summary.sort_values("mean_AUPRC", ascending=False).copy()
    compact_summary.to_csv(qa_dir / "current26_key_model_results.csv", index=False, encoding="utf-8-sig")
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "endpoint": "Open Targets 26.06 disease-specific LUAD clinical-precedence evidence",
        "positive_count": current_positives,
        "methods": methods,
        "checks": checks,
        "interpretation": (
            "A PASS confirms endpoint alignment, shared folds, leakage controls and package completeness. "
            "It does not establish causal-gene discovery, therapeutic validity or journal acceptance."
        ),
    }
    (qa_dir / "CURRENT26_QA_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest_lines = []
    for path in sorted(
        candidate_path
        for candidate_path in root.rglob("*")
        if candidate_path.is_file()
        and candidate_path.name != "FILE_MANIFEST.sha256"
        and release_file(candidate_path, root)
    ):
        manifest_lines.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "FILE_MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
