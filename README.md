# DCPI-RWR: degree-conditioned pathway-integrated network diffusion for multicancer gene prioritization

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release: v1.0.0](https://img.shields.io/badge/release-v1.0.0-5c6bc0.svg)](https://github.com/lhyzyt2001/DCPI-RWR-gene-prioritization/releases/tag/v1.0.0)

> Public release: `DCPI-RWR-release-v1.0.0`. The directory name contains only the method name and semantic version; internal target-tier, date and `final` labels are not part of the public identity.

This package extends the endpoint-unified Open Targets 26.06 benchmark with a deliberately conservative innovation track. The proposed analysis is **not** positioned as a universally superior algorithm. Its defensible contribution is a time-separated, multicancer study of the performance–centrality trade-off in network-based target prioritization.

## Study positioning

**Degree-conditioned pathway-integrated network diffusion reveals a performance–centrality trade-off in time-separated multicancer gene prioritization**

The endpoint is the later appearance of cancer-specific clinical-stage target evidence in Open Targets, not causal driver-gene status. Open Targets 26.03 labels are used for development and the genes first present in the corresponding 26.06 label set form the time-separated endpoint. Network topology was not independently frozen at 26.03, so the design must be described as **cross-release label validation**, not a fully prospective study.

## What was added

1. **DCPI-RWR** blends raw STRING-RWR rank with degree-conditioned STRING and KEGG shared-pathway ranks. A TISCH cell-type similarity graph was evaluated as an optional auxiliary view, but it received zero weight in every performance-oriented locked configuration.
2. Hyperparameters are selected exclusively with repeated five-fold cross-validation of 26.03 labels.
3. Separate locked objectives assess global AUPRC, Precision@100 and centrality-aware trade-offs. None uses 26.06 labels during selection.
4. Locked configurations are evaluated in LUAD, breast, colorectal, prostate, melanoma and ovarian cancer.
5. A leave-one-cancer-out analysis selects each cancer's configuration from the other five cancers only.
6. Label-stratified gene bootstrap, paired Top-100 incident-hit analysis, degree-matched seed nulls and degree-conditional pathway-component permutation quantify uncertainty and specificity.
7. A locked LUAD bias-aware Top-20 list is characterized in TCGA-LUAD, GSE68465 and CPTAC-LUAD without feeding those outcomes back into model selection.

## Legacy analyses retained for provenance

Scripts and outputs from earlier positive-unlabelled and multiview-GCN sensitivity analyses are retained only to document the analysis history. They are not components of the primary DCPI-RWR method. The primary reproduction path is defined in `docs/RUN_ORDER.md`; the fixed-candidate characterization entry point is `code/09_characterize_dcpi_candidates.py`.

Legacy output labels such as `LightGBM direct stack`, `XGBoost direct stack` and filenames containing `temporal_validation` are preserved only where changing them would alter frozen provenance records. In the manuscript, these comparators are reported as `LightGBM + RWR/features` and `XGBoost + RWR/features`, and the design is described as cross-release validation.

## Main results

- The performance-oriented DCPI-RWR configuration did **not** improve global incident-gene AUPRC across cancers: random-effects pooled delta versus STRING-RWR was -0.0063 (95% CI -0.0146 to 0.0020; I2 64.3%).
- LUAD showed only a small, non-significant AUPRC increase (0.0360 versus 0.0331; bootstrap median delta 0.0030, 95% CI -0.0036 to 0.0120).
- CRC showed a significant loss after Holm correction; universal superiority must not be claimed.
- At a fixed Top-100 budget, the performance configuration recovered 44 incident targets across six cancers versus 34 for STRING-RWR, but the pooled paired exact comparison was not significant (24 versus 14 method-only incident hits; P=0.143).
- Mean Top-100 STRING top-5% hub fraction fell from 0.790 for RWR to 0.485 for the performance configuration and 0.380 for the bias-aware configuration. The direction of hub reduction was consistent in all six cancers (two-sided sign-test P=0.0313).
- Top-100 lists remained strongly enriched over endpoint prevalence in LUAD, CRC, melanoma and ovarian cancer after multiplicity correction. BRCA was borderline after correction and PRAD was low powered.
- Conditional pathway permutation gave nominal evidence in LUAD and ovarian cancer, but neither survived Holm correction (adjusted P=0.144). Pathway contribution therefore remains suggestive rather than established.
- The TISCH cell-type graph received zero optimal weight in nearly all locked configurations. It should be reported as a negative result, not presented as a demonstrated performance driver.
- Of 17 LUAD candidates evaluable in both RNA cohorts, 9 had concordant tumor-normal direction; VEGFB, PDGFD and EPHA4 were concordant and FDR-significant in both. No candidate was FDR-significant for adjusted survival in TCGA or GSE68465.

## Interpretation boundary

The strongest defensible conclusion is that modest pathway-conditioned diffusion can substantially reduce hub concentration while preserving or numerically improving fixed-budget recovery in several cancers, but this does not translate into a reproducible gain in global AUPRC. External molecular concordance supports biological plausibility for a subset of LUAD hypotheses but does not validate causality or therapeutic efficacy.

## Key folders

- `code/09_characterize_dcpi_candidates.py`: outcome-blind characterization of a fixed DCPI-RWR candidate set.
- `code/25_degree_conditioned_pathway_rwr.py`: historical-only tuning, time-separated evaluation and locked candidate generation.
- `code/26_summarize_biasaware_framework.py`: multiplicity correction, meta-analysis and integrated molecular characterization.
- `code/27_loco_transfer_dcpi_rwr.py`: leave-one-cancer-out transfer analysis.
- `code/28_pathway_component_permutation.py`: degree-conditional pathway-component permutation.
- `results/21_constrained_dcpi_rwr`: complete primary and sensitivity outputs.
- `results/18_dcpi_rwr_external`: TCGA, GSE68465 and CPTAC characterization.
- `results/20_loco_transfer`: cross-cancer transfer results.
- `results/22_final_innovation_summary`: manuscript-facing summary tables and figures.
- `results/23_pathway_component_permutation`: pathway-specific conditional null.
- `results/manuscript_figures`: the five manuscript figures and the workflow source figure.

## Installation

Python 3.11 is recommended. From the repository root:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch and PyTorch Geometric wheels must match the local operating system and accelerator. The study was run with the CPU build of PyTorch 2.6.0; a GPU is optional.

## Reproduction

The primary analysis order and exact command-line arguments are documented in [`docs/RUN_ORDER.md`](docs/RUN_ORDER.md). The principal entry points are:

```bash
python code/25_degree_conditioned_pathway_rwr.py --help
python code/27_loco_transfer_dcpi_rwr.py --help
python code/28_pathway_component_permutation.py --help
python code/26_summarize_biasaware_framework.py --help
```

Frozen derived outputs are included to permit claim-level verification without rerunning the most computationally intensive analyses. Source data that are access-controlled or governed by third-party terms are not redistributed; see [`docs/ANALYSIS_SCOPE.md`](docs/ANALYSIS_SCOPE.md) and the provenance files in `docs/`.

## Repository integrity

`MANIFEST.sha256.tsv` records the byte size and SHA-256 checksum of every released file except the manifest itself and Git metadata. It can be verified before reproduction or interpretation.

## Citation

Please cite the software release described in [`CITATION.cff`](CITATION.cff). A version-specific Zenodo DOI will be added here after the GitHub `v1.0.0` release has been archived.

## License

Source code is available under the [MIT License](LICENSE). Public derived tables and figures remain subject to the terms of their underlying data sources; restricted or third-party raw data are not redistributed.

## Contact

Corresponding author: Jihua Feng (`fengjihua@ymu.edu.cn`).

See `docs/RUN_ORDER.md`, `docs/METHODS_ADDENDUM.md` and `docs/RESULTS_AND_CLAIMS.md` before reproduction or interpretation.
