# Results index

| Question | Primary files |
|---|---|
| What are the disease-specific label counts and their provenance? | `results/01_labels/multicancer_label_counts.csv`, `results/01_labels/label_provenance.json`, `results/01_labels/disease_scope_audit.csv` |
| Do LightGBM + RWR/features or XGBoost + RWR/features improve the current LUAD endpoint? | `results/02b_disease_specific_current_cv/disease_specific_summary.csv`, `disease_specific_gene_cluster_bootstrap.csv`, `disease_specific_training_audit.csv` |
| Does the conclusion persist after controlling for degree? | `results/02c_current_endpoint_centrality/current_endpoint_degree_matched_bootstrap.csv`, `current_endpoint_exact_degree_only_bootstrap.csv`, `current_endpoint_hub_removal.csv`, `current_endpoint_degree_stratified_performance.csv` |
| Does performance generalize across releases and cancers? | `results/04_multicancer_summary/multicancer_random_effects_meta_analysis.csv`, `multicancer_cross_release_forest.pdf` |
| How much does topology drive the historical endpoint? | `results/05_exact_centrality_audit/exact_degree_only_gene_cluster_bootstrap.csv`, `conditional_permutation_results.csv.gz`, `network_centrality_audit.pdf` |
| Legacy sensitivity: are earlier conclusions sensitive to the positive-unlabelled assumption? | `results/06_pu_sensitivity/pu_summary.csv`, `pu_gene_cluster_bootstrap.csv`, `candidate_rank_overlap.csv` |
| Are deep-model findings stable across optimization seeds? | `results/07_uniform_multiseed/combined/uniform_deep_vs_rwr_summary.csv`, `uniform_deep_vs_rwr_gene_cluster_bootstrap.csv`, `uniform_multiseed_training_audit.csv` |
| Legacy candidate analysis: which post hoc hypotheses were retained before DCPI-RWR locking? | `results/08_candidates/pu_robust_consensus_top20.csv`, `topology_adjusted_sensitivity_top20.csv` |
| What external characterization is available, including the locked DCPI-RWR candidates? | `results/09_external_characterization`, `results/11_integrated_candidates/PU_robust_top20_integrated_characterization.csv` |
| Which pathways are associated with the rankings? | `results/10_functional/GSEA_all_results.csv`, `Hallmark_GSEA_primary.pdf` |
| Did the package pass internal completeness/leakage checks? | `results/00_summary/QA_REPORT.json`, `results/00_summary/key_results.csv`, `FILE_MANIFEST.sha256` |
| What is the final endpoint-aligned seven-method result? | `results/12_current26_unified/combined/current26_all_model_summary.csv`, `current26_all_model_gene_cluster_bootstrap.csv`, `current26_unified_benchmark.pdf` |
| Are all current-endpoint deep fits complete and leakage controlled? | `results/12_current26_unified/combined/current26_all_deep_training_audit.csv`, `results/12_current26_unified/QA/CURRENT26_QA_REPORT.json` |
| Why are 120 and 14 different? | `results/13_temporal_reconciliation/temporal_design_reconciliation.csv`, `temporal_topk_recall_comparison.pdf` |
| Which candidates follow from the best current-endpoint method? | `results/14_current26_candidates/current26_locked_RWR_top20.csv`, `current26_topology_adjusted_sensitivity_top20.csv` |
| How do the fixed current candidates behave externally? | `results/15_current26_external_characterization/current26_RWR_top20_integrated_characterization.csv`, `current26_candidate_characterization_summary.json` |
| What does current-endpoint functional characterization show? | `results/16_current26_functional/current26_GSEA_all_results.csv`, `current26_GSEA_manifest.json` |

| Which DCPI-RWR configurations were selected without using 26.06 labels? | `results/21_constrained_dcpi_rwr/DCPI_RWR_selected_configs.json`, `DCPI_RWR_historical_tuning_surface.csv.gz` |
| What are the primary performance-centrality results? | `results/22_final_innovation_summary/DCPI_RWR_multicancer_bootstrap_with_holm.csv`, `DCPI_RWR_bias_performance_biology_summary.pdf` |
| Does pathway conditioning exceed a degree-conditional null? | `results/23_pathway_component_permutation/pathway_component_conditional_permutation_summary.csv` |
| Which files correspond to the submitted manuscript figures? | `results/manuscript_figures/Figure_1_DCPI_RWR_workflow.png` through `Figure_5_LUAD_candidate_characterization.png`; see `docs/MANUSCRIPT_ALIGNMENT.md` |

Files under `results/00_summary` are generated last and therefore appear only after all long-running analyses finish.
