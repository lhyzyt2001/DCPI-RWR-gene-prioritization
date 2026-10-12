# Manuscript-to-release alignment

This document maps the public `v1.0.0` software release to the manuscript titled:

> Degree-conditioned pathway-integrated network diffusion reveals a performance–centrality trade-off in time-separated multicancer gene prioritization

## Terminology

- **DCPI-RWR** is the degree-conditioned pathway-integrated random walk with restart framework implemented in `code/25_degree_conditioned_pathway_rwr.py`.
- **STRING-RWR** is the unconditioned diffusion baseline on the STRING functional-association network.
- **LightGBM + RWR/features** and **XGBoost + RWR/features** use cross-fitted RWR logits together with degree, centrality and cell-type features. Frozen legacy outputs may retain the earlier internal label `direct stack`.
- **Cross-release validation** uses Open Targets 26.03 labels for development and genes newly labelled in 26.06 for evaluation. It is not described as fully prospective because network topology was not independently frozen at 26.03.
- The cell-type similarity view was evaluated but received zero weight in all performance-optimal locked configurations; this is reported as a negative result.

## Figure map

| Manuscript figure | Released file | Primary supporting results |
|---|---|---|
| Figure 1, workflow | `results/manuscript_figures/Figure_1_DCPI_RWR_workflow.png` | `code/30_make_workflow_figure.py`, `docs/RUN_ORDER.md` |
| Figure 2, contemporary LUAD benchmark | `results/manuscript_figures/Figure_2_contemporary_benchmark.png` | `results/12_current26_unified/combined/current26_all_model_summary.csv` |
| Figure 3, multicancer cross-release performance | `results/manuscript_figures/Figure_3_cross_release_performance.png` | `results/22_final_innovation_summary/DCPI_RWR_multicancer_bootstrap_with_holm.csv` |
| Figure 4, performance-centrality trade-off | `results/manuscript_figures/Figure_4_centrality_tradeoff.png` | `results/22_final_innovation_summary` and `results/21_constrained_dcpi_rwr` |
| Figure 5, locked LUAD candidate characterization | `results/manuscript_figures/Figure_5_LUAD_candidate_characterization.png` | `results/18_dcpi_rwr_external` and `results/22_final_innovation_summary` |

All PNG files were exported at 600 dpi. Figure 1 is also supplied as a vector PDF. The frozen numerical outputs, candidate set and statistical conclusions are unchanged by public-package renaming.

## Interpretation boundary

The release supports computational prioritization and the study of centrality bias. It does not establish causal cancer genes, clinical utility or therapeutic efficacy. Global AUPRC superiority over STRING-RWR is not claimed.
