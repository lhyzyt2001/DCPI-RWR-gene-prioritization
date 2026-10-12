# Methods addendum: degree-conditioned pathway-integrated network diffusion

## Endpoint and temporal design

For each cancer, genes with cancer-specific clinical-precedence evidence in Open Targets release 26.03 were historical positives. Genes first present in the disease-specific 26.06 set were incident positives. All other genes in the 25,242-node analysis universe were treated as unlabelled, not confirmed negatives. The endpoint captures later clinical-stage evidence and must not be interpreted as causal cancer-gene status.

## Network views

The three disease-independent views were the weighted STRING functional-association network, a mutual 12-nearest-neighbour graph constructed from 15 TISCH cell-type expression profiles, and a weighted shared-KEGG-pathway graph. Each adjacency matrix was column-normalized. RWR was iterated until the L1 change was below 1e-10 or 300 iterations were reached.

## Degree conditioning and fusion

For each view, nodes were divided into up to 20 quantile bins of log(1 + view degree). The RWR score was transformed to a within-bin percentile. A configuration combined the global percentile rank of the raw STRING-RWR score with a weighted sum of the three within-degree-bin percentiles. Restart probabilities were 0.15, 0.30 or 0.50; blend coefficients were 0, 0.25, 0.50, 0.75 or 1; view weights followed a simplex grid with step 0.25.

## Historical-only selection

Candidate configurations were assessed by repeated stratified five-fold cross-validation with three repeats using only 26.03 labels. The performance configuration maximized mean AUPRC. The discovery configuration maximized mean Precision@100. The bias-aware configuration minimized mean absolute Spearman correlation between score and STRING degree among configurations retaining at least 95% of the best historical-CV AUPRC. An aggressive constrained sensitivity selected the lowest historical Top-100 hub fraction among configurations retaining at least 95% of historical RWR Precision@100. The latter failed time-separated transport and is not the primary configuration.

## Time-separated evaluation

After configuration locking, all 26.03 positives were used as RWR seeds and genes first appearing in 26.06 were evaluated over the complete historical-unlabelled universe. Reported metrics include ROC-AUC, AUPRC, Precision/Recall at K, Top-K incident hits, score-degree Spearman correlation, Top-K hub fraction and median STRING degree. This deployment-style analysis differs from the repeated out-of-fold cross-release analysis in the earlier benchmark and should be reported separately.

## Uncertainty and null analyses

A paired label-stratified gene bootstrap with 2,000 replicates compared AUPRC with STRING-RWR. Holm adjustment covered all locked alternatives across six cancers. For Top-100 comparisons, the exact binomial test compared incident hits found exclusively by the alternative list with those found exclusively by RWR. Hypergeometric tests quantified incident-gene enrichment relative to each cancer's candidate-universe prevalence and were Holm adjusted across Top-100 tests.

In the LUAD seed null, random seed sets were sampled from current-evidence-negative genes while matching the historical seeds' STRING-degree-bin counts. The analysis used 100 replicates and is explicitly low-resolution. For pathway-component testing, pathway conditioned-percentile values were permuted within STRING-degree bins while all other score components remained fixed; 2,000 permutations were performed per cancer and P values were Holm adjusted across six cancers.

## Leave-one-cancer-out transfer

For each target cancer, configurations were ranked within each of the other five cancers by historical-CV AUPRC. The configuration with the highest mean percentile rank was locked and transferred to the target cancer. Neither target-cancer historical labels nor any 26.06 labels were used in this selection.

## External LUAD characterization

The bias-aware LUAD Top-20 list was fixed before external analysis. TCGA-LUAD paired tumor-normal expression, stage and adjusted overall survival; GSE68465 tumor-normal expression and adjusted survival; and CPTAC-LUAD protein profiles were evaluated with Benjamini-Hochberg correction within each endpoint. These analyses are secondary characterization and were not used to tune, filter or reorder candidates.
