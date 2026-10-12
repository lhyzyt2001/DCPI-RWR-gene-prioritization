# Revised analysis scope

This package adds only the analyses required by the final methodological review:

- disease-specific Open Targets label reconstruction;
- a current-release LUAD repeated-cross-validation benchmark;
- 26.03-to-26.06 cross-release analyses in LUAD and five additional cancers;
- a fixed-schema LUAD evidence-date sensitivity analysis;
- complete network-centrality auditing, including degree strata, exact/nearest degree matching, hub removal and conditional feature permutation;
- bagging positive-unlabelled sensitivity analysis;
- three uniformly trained optimization seeds for APPNP, GCNII and GATv2 on every fixed outer fold;
- regeneration and post hoc characterization of a PU-robust consensus candidate list;
- TCGA-LUAD, GSE68465 and CPTAC computational characterization; and
- Hallmark and GO-BP GSEA with a degree/topology-adjusted sensitivity ranking.

No old candidate-specific DepMap result is reused. The locally available DepMap cache contains only the former candidate set; a claim for the revised candidates requires the complete versioned gene-effect matrix.

The revised package does not claim a universally superior algorithm. Its intended contribution is a bias-aware comparison of RWR, boosted-tree stacks and neural propagation models under incomplete positive labels, accompanied by explicit failure-to-replicate and endpoint-shift analyses.
