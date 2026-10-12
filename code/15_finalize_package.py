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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(rows: list[dict], section: str, name: str, value, detail: str = "") -> None:
    rows.append({"section": section, "metric": name, "value": value, "detail": detail})


def any_true(values: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(values):
        return bool(values.fillna(False).any())
    return bool(values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).any())


def is_release_file(path: Path, root: Path) -> bool:
    """Exclude restart shards and interpreter caches from the final release."""
    relative_parts = path.relative_to(root).parts
    excluded_parts = {"__pycache__", "GCNII_repeat12", "GATv2_repeat2"}
    return not any(part in excluded_parts for part in relative_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final result summary, QA report and SHA-256 manifest.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results"
    summary_dir = results / "00_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    label_counts = pd.read_csv(results / "01_labels" / "multicancer_label_counts.csv")
    for record in label_counts.itertuples(index=False):
        metric(rows, "disease_specific_labels", f"{record.cancer} historical positives", record.historical_targets, "Open Targets 26.03 clinical-precedence evidence")
        metric(rows, "disease_specific_labels", f"{record.cancer} current positives", record.future_targets, "Open Targets 26.06 clinical-precedence evidence")
        metric(rows, "disease_specific_labels", f"{record.cancer} incident positives", record.incident_targets, "Present in 26.06 but absent from 26.03")

    current = pd.read_csv(results / "02b_disease_specific_current_cv" / "disease_specific_summary.csv")
    for record in current.itertuples(index=False):
        metric(rows, "current_disease_specific_CV", f"{record.method} mean AUPRC", record.mean_AUPRC, f"SD={record.sd_AUPRC:.6f}; 15 outer folds")
    current_boot = pd.read_csv(results / "02b_disease_specific_current_cv" / "disease_specific_gene_cluster_bootstrap.csv")
    for record in current_boot.itertuples(index=False):
        metric(rows, "current_disease_specific_CV", f"{record.method} minus {record.reference}", record.median_delta_AUPRC, f"95% CI {record.ci_lower:.6f} to {record.ci_upper:.6f}; P={record.p_two_sided:.6g}")
    current_degree = pd.read_csv(results / "02c_current_endpoint_centrality" / "current_endpoint_degree_matched_bootstrap.csv")
    for record in current_degree.itertuples(index=False):
        metric(rows, "current_endpoint_degree_matched", f"{record.method} minus {record.reference}", record.median_delta_AUPRC, f"95% CI {record.ci_lower:.6f} to {record.ci_upper:.6f}; P={record.p_two_sided:.6g}")

    date_split = pd.read_csv(results / "03_fixed_schema_date_split" / "temporal_validation_summary.csv")
    for record in date_split.itertuples(index=False):
        metric(rows, "fixed_schema_date_split", f"{record.method} mean AUPRC", record.AUPRC_mean, f"SD={record.AUPRC_sd:.6f}; only 14 incident LUAD genes")

    pu = pd.read_csv(results / "06_pu_sensitivity" / "pu_summary.csv")
    for record in pu.itertuples(index=False):
        metric(rows, "historical_endpoint_PU", f"{record.method} mean AUPRC", record.AUPRC_mean, f"SD={record.AUPRC_sd:.6f}")
    exact = pd.read_csv(results / "05_exact_centrality_audit" / "exact_degree_only_gene_cluster_bootstrap.csv")
    for record in exact.itertuples(index=False):
        metric(rows, "historical_endpoint_exact_degree", f"{record.method} minus {record.reference}", record.median_delta_AUPRC, f"95% CI {record.ci_lower:.6f} to {record.ci_upper:.6f}; Holm P={record.p_holm:.6g}")

    permutation = pd.read_csv(results / "05_exact_centrality_audit" / "conditional_permutation_results.csv.gz")
    perm_summary = permutation.groupby(["method", "permuted_block"], as_index=False)["AUPRC_drop"].mean()
    for record in perm_summary.itertuples(index=False):
        metric(rows, "historical_endpoint_conditional_permutation", f"{record.method}: {record.permuted_block}", record.AUPRC_drop)

    meta = pd.read_csv(results / "04_multicancer_summary" / "multicancer_random_effects_meta_analysis.csv")
    for record in meta.itertuples(index=False):
        comparison = f"{record.method} minus {record.reference}"
        metric(rows, "cross_release_multicancer", f"{comparison} pooled delta", record.pooled_delta_AUPRC, f"95% CI {record.ci_lower:.6f} to {record.ci_upper:.6f}; I2={record.I_squared_percent:.2f}%")

    deep = pd.read_csv(results / "07_uniform_multiseed" / "combined" / "uniform_deep_vs_rwr_summary.csv")
    for record in deep.itertuples(index=False):
        metric(rows, "uniform_deep_multiseed", f"{record.method} mean AUPRC", record.mean_AUPRC, f"SD={record.sd_AUPRC:.6f}")
    deep_boot = pd.read_csv(results / "07_uniform_multiseed" / "combined" / "uniform_deep_vs_rwr_gene_cluster_bootstrap.csv")
    for record in deep_boot.itertuples(index=False):
        metric(rows, "uniform_deep_multiseed", f"{record.method} minus RWR", record.bootstrap_median, f"95% CI {record.ci_lower:.6f} to {record.ci_upper:.6f}; P={record.two_sided_p:.6g}")

    candidate_summary = json.loads((results / "11_integrated_candidates" / "candidate_characterization_summary.json").read_text(encoding="utf-8"))
    for name, value in candidate_summary.items():
        metric(rows, "candidate_characterization", name, value)

    gsea = pd.read_csv(results / "10_functional" / "GSEA_all_results.csv")
    for (analysis, library), block in gsea.groupby(["Analysis", "Library"]):
        fdr_col = next(col for col in block if col.lower().replace(" ", "").startswith("fdr"))
        significant = int((pd.to_numeric(block[fdr_col], errors="coerce") < 0.05).sum())
        metric(rows, "GSEA", f"{analysis} {library} FDR<0.05", significant, f"of {len(block)} tested gene sets")
    pd.DataFrame(rows).to_csv(summary_dir / "key_results.csv", index=False, encoding="utf-8-sig")

    expected = [
        results / "02b_disease_specific_current_cv" / "disease_specific_training_audit.csv",
        results / "02c_current_endpoint_centrality" / "current_endpoint_degree_matched_bootstrap.csv",
        results / "04_multicancer_summary" / "multicancer_cross_release_forest.pdf",
        results / "05_exact_centrality_audit" / "network_centrality_audit.pdf",
        results / "06_pu_sensitivity" / "pu_gene_cluster_bootstrap.csv",
        results / "07_uniform_multiseed" / "combined" / "uniform_multiseed_combined_manifest.json",
        results / "10_functional" / "GSEA_all_results.csv",
        results / "11_integrated_candidates" / "PU_robust_top20_integrated_characterization.csv",
    ]
    current_audit = pd.read_csv(expected[0])
    deep_audit = pd.read_csv(results / "07_uniform_multiseed" / "combined" / "uniform_multiseed_training_audit.csv")
    current_metrics = pd.read_csv(results / "02b_disease_specific_current_cv" / "disease_specific_fold_metrics.csv")
    checks = {
        "all_expected_outputs_exist": all(path.exists() for path in expected),
        "current_endpoint_has_45_method_fold_rows": len(current_metrics) == 45,
        "current_endpoint_has_15_outer_folds": current_metrics[["repeat", "fold"]].drop_duplicates().shape[0] == 15,
        "current_outer_test_labels_not_used": not any_true(current_audit["outer_test_labels_used_for_fitting"]),
        "current_outer_test_nodes_not_RWR_seeds": not any_true(current_audit["outer_test_nodes_used_as_RWR_seeds"]),
        "deep_has_135_uniform_fits": len(deep_audit) == 135,
        "deep_fit_identifiers_unique": not deep_audit[["repeat", "fold", "method", "seed_label"]].duplicated().any(),
        "deep_outer_test_labels_not_used": not any_true(deep_audit["outer_test_labels_used_for_training"]),
        "candidate_count_is_20": candidate_summary["candidate_count"] == 20,
        "all_four_GSEA_tables_present": gsea[["Analysis", "Library"]].drop_duplicates().shape[0] == 4,
    }
    image_audit = []
    for path in results.rglob("*.png"):
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (None, None))
            image_audit.append({"path": path.relative_to(root).as_posix(), "width_px": image.width, "height_px": image.height, "dpi_x": dpi[0], "dpi_y": dpi[1]})
    pd.DataFrame(image_audit).to_csv(summary_dir / "figure_pixel_and_dpi_audit.csv", index=False, encoding="utf-8-sig")
    checks["all_png_figures_at_least_300_dpi"] = bool(image_audit) and all(
        row["dpi_x"] is not None and row["dpi_y"] is not None and row["dpi_x"] >= 299 and row["dpi_y"] >= 299
        for row in image_audit
    )
    checks["all_png_figures_have_vector_pdf_counterparts"] = bool(image_audit) and all(
        (root / Path(row["path"]).with_suffix(".pdf")).exists() for row in image_audit
    )

    qa = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "result_file_count": sum(1 for p in results.rglob("*") if p.is_file() and is_release_file(p, root)),
        "code_file_count": sum(1 for p in (root / "code").rglob("*.py")),
        "png_figure_count": len(image_audit),
        "note": "A PASS confirms internal package completeness and leakage checks; it does not imply biological validation or journal acceptance.",
    }
    (summary_dir / "QA_REPORT.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_lines = []
    for path in sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.name != "FILE_MANIFEST.sha256" and is_release_file(p, root)
    ):
        manifest_lines.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "FILE_MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(json.dumps(qa, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
