# Unified Open Targets 26.06 endpoint report

## What was corrected

All primary methods now use the same disease-specific Open Targets 26.06 LUAD endpoint (197 positives among 25,242 genes), the same three-repeat five-fold outer splits and the same leakage boundary. The former deep-model results based on the archived 1,184-positive endpoint are retained only as historical sensitivity analyses and are not mixed into the current benchmark.

APPNP, GCNII and GATv2 were each fitted with three optimization seeds in every outer fold. This produced 135 completed deep fits and 681,534 seed-specific outer-test predictions. Audit checks found no duplicate fit identifiers, no duplicate prediction identifiers, no use of outer-test labels for training, and no use of outer-test nodes as RWR seeds. No fit selected the maximum 300th epoch.

## Main result

| Method | ROC-AUC, mean ± SD | AUPRC, mean ± SD |
|---|---:|---:|
| RWR | 0.976 ± 0.010 | 0.554 ± 0.094 |
| LightGBM + RWR/features | 0.977 ± 0.009 | 0.536 ± 0.084 |
| XGBoost + RWR/features | 0.978 ± 0.007 | 0.534 ± 0.080 |
| GATv2 | 0.968 ± 0.013 | 0.362 ± 0.087 |
| APPNP | 0.963 ± 0.013 | 0.346 ± 0.060 |
| GCNII | 0.965 ± 0.012 | 0.330 ± 0.069 |
| Network-degree logistic | 0.948 ± 0.009 | 0.154 ± 0.016 |

The label-stratified gene-cluster bootstrap retained each sampled gene across all repeats. Relative to RWR, median AUPRC differences were −0.0195 for LightGBM + RWR/features (95% CI −0.0506 to 0.0145; Holm P=0.484), −0.0198 for XGBoost + RWR/features (−0.0539 to 0.0148; Holm P=0.484), −0.1810 for GATv2 (−0.2366 to −0.1257; Holm P=0.006), −0.1970 for APPNP (−0.2381 to −0.1512; Holm P=0.006), and −0.2130 for GCNII (−0.2555 to −0.1662; Holm P=0.006). Thus no tested model improved global AUPRC over RWR.

At K=20, XGBoost + RWR/features recovered a mean of 17 current positives across repeats, RWR 16.7 and LightGBM + RWR/features 14. This local Top-K difference does not override the prespecified global AUPRC result and was not used to claim overall superiority.

## Centrality and candidate consequences

In the existing fold-specific 1:1 degree-matched analysis, LightGBM + RWR/features minus RWR had median ΔAUPRC 0.0086 (95% CI −0.0016 to 0.0187; P=0.086), whereas XGBoost + RWR/features minus RWR was −0.0041 (−0.0156 to 0.0072; P=0.492). The new performance-aligned candidate set was therefore locked from mean out-of-fold RWR scores.

Nineteen of the RWR top 20 are STRING degree top-5% hubs. The list is dominated by mitochondrial respiratory-chain and ubiquitous structural genes. A cubic log-degree residual sensitivity retained 17 of the 20 primary candidates and still contained 19 top-5% hubs. This is a substantive negative result: simple degree residualization does not remove the broader topology dependence of RWR.

## External characterization

Eighteen candidates mapped in TCGA and GSE68465, and 15 had CPTAC protein profiles. Nine showed FDR-significant paired TCGA tumor-normal expression differences, 12 showed FDR-significant GSE68465 tumor-normal differences, and only four mapped genes had concordant expression direction across both datasets. No candidate had FDR-significant TCGA stage or adjusted overall-survival association. One GSE68465 adjusted Cox association passed FDR, but six GSE68465 expression terms violated the proportional-hazards assumption and therefore require average-over-follow-up interpretation. Ten candidates differed from the study-specific CPTAC zero reference; this is not an adjacent-normal comparison.

The old candidate-specific DepMap cache was not reused because it does not contain the newly locked genes. A new DepMap claim requires a complete, versioned gene-effect matrix or fresh candidate-specific extraction.

## Temporal-design clarification

The cross-release design contains 77 LUAD genes in 26.03, 197 in 26.06 and 120 genes present only in the later release. The within-26.06 evidence-date design contains 182 genes dated on or before 2016-12-31 and 196 genes with usable dates, leaving 14 post-cutoff additions. These are different temporal questions; 14 is not a filtered subset of 120.

## Functional characterization boundary

All 50 Hallmark sets and 2,678 of 2,721 GO-BP sets were significant for the primary RWR ranking; after cubic degree residualization, all 50 Hallmark sets and 2,229 GO-BP sets remained significant. These results are too broad to support selective mechanistic claims. They instead reinforce that the ranking captures pervasive network and annotation structure. Complete tables are retained for transparency.

## Publication judgment

The unified experiment is internally consistent and supports the conclusion that leakage-controlled RWR is a very strong baseline for recovering the current Open Targets LUAD clinical-evidence annotation. It does not support a claim that LightGBM + RWR/features, XGBoost + RWR/features or the tested deep models improve RWR. A methods-oriented journal submission is most defensible as an endpoint-reconciliation and bias-aware benchmarking study with negative results, not as a new-model paper. The remaining scientific weakness is the absence of a genuinely frozen prospective endpoint and the hub-dominated nature of the candidate ranking.
