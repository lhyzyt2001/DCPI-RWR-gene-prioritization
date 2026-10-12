from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


RESULT_DIRS = [
    "01_labels",
    "02_cross_release",
    "04_multicancer_summary",
    "05_exact_centrality_audit",
    "07_uniform_multiseed",
    "12_current26_unified",
    "18_dcpi_rwr_external",
    "20_loco_transfer",
    "21_constrained_dcpi_rwr",
    "22_final_innovation_summary",
    "23_pathway_component_permutation",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing release directory: {target}")
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble and verify the DCPI-RWR publication package.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    release = args.release.resolve()
    if release.exists():
        raise FileExistsError(f"Release already exists: {release}")
    release.mkdir(parents=True)
    shutil.copy2(root / "README_INNOVATION_PACKAGE.md", release / "README.md")
    shutil.copytree(root / "code", release / "code", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(root / "docs", release / "docs")
    shutil.copytree(root / "inputs", release / "inputs")
    (release / "results").mkdir()
    for name in RESULT_DIRS:
        shutil.copytree(root / "results" / name, release / "results" / name)

    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    primary = release / "results" / "21_constrained_dcpi_rwr"
    metrics = pd.read_csv(primary / "DCPI_RWR_multicancer_cross_release_metrics.csv")
    topk = pd.read_csv(primary / "DCPI_RWR_multicancer_topk_and_hubs.csv")
    bootstrap = pd.read_csv(primary / "DCPI_RWR_multicancer_bootstrap.csv")
    configs = json.loads((primary / "DCPI_RWR_selected_configs.json").read_text(encoding="utf-8"))
    check("six cancer endpoints", metrics.cancer.nunique() == 6, f"cancers={sorted(metrics.cancer.unique())}")
    check("four locked alternatives", bootstrap.method.nunique() == 4, f"methods={sorted(bootstrap.method.unique())}")
    check("2,000 bootstrap replicates", set(bootstrap.bootstrap_replicates) == {2000}, str(set(bootstrap.bootstrap_replicates)))
    check("historical-only configurations", set(configs) == {"LUAD", "BRCA", "CRC", "PRAD", "MEL", "OV"}, "all six present")
    top100 = topk[(topk.K == 100) & topk.method.isin(["RWR-STRING", "DCPI-RWR performance", "DCPI-RWR bias-aware"])]
    summary = top100.groupby("method").agg(hits=("incident_hits", "sum"), hubs=("hub_fraction", "mean"))
    check("Top100 descriptive recovery", int(summary.loc["DCPI-RWR performance", "hits"]) == 44 and int(summary.loc["RWR-STRING", "hits"]) == 34, summary.to_json())
    check("hub reduction", float(summary.loc["DCPI-RWR performance", "hubs"]) < float(summary.loc["RWR-STRING", "hubs"]), summary.to_json())
    paired = pd.read_csv(primary / "DCPI_RWR_top100_paired_incident_analysis.csv")
    pooled = paired[(paired.cancer == "POOLED_DESCRIPTIVE") & paired.comparison.str.startswith("DCPI-RWR performance")].iloc[0]
    check("paired test retained", abs(float(pooled.exact_p_two_sided) - 0.14330665429588407) < 1e-12, f"P={pooled.exact_p_two_sided}")
    candidates = pd.read_csv(primary / "DCPI_RWR_bias_aware_current26_locked_top20.csv")
    check("locked LUAD Top20", len(candidates) == 20 and candidates.Candidate_Rank.tolist() == list(range(1, 21)), f"rows={len(candidates)}")
    pathway = pd.read_csv(release / "results" / "23_pathway_component_permutation" / "pathway_component_conditional_permutation_summary.csv")
    check("pathway permutation", len(pathway) == 6 and set(pathway.permutations) == {2000}, f"rows={len(pathway)}")
    integrated = pd.read_csv(release / "results" / "22_final_innovation_summary" / "DCPI_RWR_bias_aware_top20_integrated_evidence.csv")
    check("external characterization", len(integrated) == 20, f"rows={len(integrated)}")

    figure_rows = []
    for png in release.rglob("*.png"):
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0, 0))
            figure_rows.append({"file": png.relative_to(release).as_posix(), "width": image.width, "height": image.height, "dpi_x": float(dpi[0]), "dpi_y": float(dpi[1])})
    figures = pd.DataFrame(figure_rows)
    figures.to_csv(release / "FIGURE_QA.csv", index=False, encoding="utf-8-sig")
    relevant = figures[figures.file.str.contains("21_constrained|22_final")]
    check("publication PNG resolution", len(relevant) > 0 and (relevant.dpi_x >= 300).all() and (relevant.dpi_y >= 300).all(), f"checked={len(relevant)}")
    pdf_names = {path.stem for path in release.rglob("*.pdf")}
    for png in relevant.file:
        check(f"vector companion: {Path(png).stem}", Path(png).stem in pdf_names, "PDF present")

    key_results = pd.DataFrame(
        [
            {"result": "RWR total incident hits at Top100 across six cancers", "value": 34, "interpretation": "reference"},
            {"result": "performance DCPI-RWR total incident hits at Top100", "value": 44, "interpretation": "descriptive increase; paired P=0.143"},
            {"result": "RWR mean Top100 hub fraction", "value": 0.790, "interpretation": "STRING top-5% hubs"},
            {"result": "performance DCPI-RWR mean Top100 hub fraction", "value": 0.485, "interpretation": "lower in all six cancers"},
            {"result": "bias-aware DCPI-RWR mean Top100 hub fraction", "value": 0.380, "interpretation": "larger de-biasing with performance cost"},
            {"result": "global AUPRC random-effects delta", "value": -0.006307, "interpretation": "95% CI -0.014575 to 0.001961"},
            {"result": "concordant FDR-significant RNA candidates", "value": 3, "interpretation": "VEGFB, PDGFD, EPHA4"},
            {"result": "adjusted survival associations", "value": 0, "interpretation": "TCGA and GSE68465"},
        ]
    )
    key_results.to_csv(release / "KEY_RESULTS.csv", index=False, encoding="utf-8-sig")
    status = "PASS" if all(item["passed"] for item in checks) else "FAIL"
    qa = {
        "status": status,
        "checks": checks,
        "claim_boundary": "The package supports a performance-centrality trade-off and fixed-budget enrichment, not universal superiority, causality or therapeutic validation.",
    }
    (release / "QA_REPORT.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    manifest_rows = []
    for path in sorted(release.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256.tsv":
            manifest_rows.append({"path": path.relative_to(release).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(release / "MANIFEST.sha256.tsv", sep="\t", index=False, encoding="utf-8")
    print(json.dumps({"release": str(release), "status": status, "files": len(manifest_rows)}, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
