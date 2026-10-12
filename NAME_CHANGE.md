# Method nomenclature update

## Recommended name

**DCPI-RWR: Degree-Conditioned Pathway-Integrated Random Walk with Restart**

## Rationale

The legacy analysis name, DC-MV-RWR, could imply that all network views consistently
contribute to the locked primary method. In the historical-only tuning results,
the TISCH cell-type view received zero weight in all six performance-oriented
configurations, whereas STRING diffusion, degree conditioning and the KEGG
shared-pathway view define the selected method. DCPI-RWR therefore describes
the implemented and selected method more precisely.

The TISCH graph remains in the code and results as a prespecified evaluated
auxiliary view and an informative negative ablation; it is not erased from the
study.

## Scope

This is a nomenclature-only update. No input, split, hyperparameter, prediction,
ranking, statistical result, candidate list or interpretation boundary was
changed. The three figures containing the old label were regenerated from the
unchanged tables.

## Public release identity

The public software package is `DCPI-RWR-release-v1.0.0`. Older PU-learning and multiview-GCN labels appear only in provenance-preserving legacy analyses and do not name the primary method.
