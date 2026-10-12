# Results, allowed claims and prohibited claims

## Allowed principal claims

1. STRING-RWR remained the strongest or statistically indistinguishable method for global time-separated AUPRC in five of six cancers; DCPI-RWR was significantly worse in CRC after Holm correction.
2. Performance-oriented DCPI-RWR reduced Top-100 hub fraction in all six cancers and increased the descriptive total of incident Top-100 hits from 34 to 44.
3. The Top-100 hit increase was not statistically conclusive (pooled paired P=0.143), whereas the direction of hub reduction was consistent (P=0.0313).
4. The global AUPRC meta-analysis did not support superiority (pooled delta -0.0063, 95% CI -0.0146 to 0.0020).
5. Aggressive de-biasing sharply reduced centrality but lost time-separated discrimination, demonstrating a genuine performance-bias trade-off rather than a free improvement.
6. Pathway information was nominally helpful for LUAD and ovarian Top-100 recovery under conditional permutation, but the evidence did not survive six-cancer correction.
7. The cell-type similarity view did not make a stable incremental contribution to the clinical-evidence endpoint.
8. VEGFB, PDGFD and EPHA4 showed concordant FDR-significant tumor-normal RNA differences in TCGA-LUAD and GSE68465. No adjusted survival association replicated.

## Claims that must not appear

- “DCPI-RWR significantly outperformed RWR across cancers.”
- “The cell-type graph improved prediction.”
- “The method discovers LUAD driver genes.”
- “The Top-20 genes are validated therapeutic targets.”
- “TCGA/GSE68465 survival validates the candidates.”
- “The 26.03-to-26.06 experiment is fully prospective.”
- “Pathway information has an independently confirmed effect after multiple testing.”

## Recommended narrative

The paper should be written as a rigorous bias-aware evaluation of network diffusion for fixed-budget cancer target prioritization. The methodological contribution is the joint treatment of time-separated evidence accrual, degree-conditioned multiview scores, explicit performance-bias Pareto analysis and multicancer transport. The negative and mixed results are integral to the contribution and reduce the risk of reviewer accusations of cherry-picking.

## Remaining risks for journal submission

- The graph views were not historically frozen, so only labels are time separated.
- The main endpoint is clinical evidence accumulation and is affected by research and commercial attention.
- Six cancers are not independent biological replicates because networks and many genes are shared.
- The fixed-budget hit improvement is descriptive rather than statistically definitive.
- External characterization provides limited biological triangulation and no causal or wet-lab validation.
- The proposed method is an adaptation of established multiview RWR and degree normalization, not a fundamentally new algorithm.

These limitations mean that a methods-oriented publication is plausible only with restrained claims and a methods/benchmarking-oriented journal fit; it is not guaranteed.
