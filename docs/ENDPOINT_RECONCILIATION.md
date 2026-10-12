# Open Targets endpoint reconciliation

## Why the endpoint was revised

The former graph dataset contains 1,184 positive genes derived from a March 2026 LUAD association export using the field `maxClinicalTrialPhase > 0`. This historical export remains useful for reproducing the earlier method-development benchmark, but it is not interchangeable with the disease-specific `evidence_clinical_precedence` endpoint distributed in current Open Targets releases.

The revised analysis reconstructs clinical-precedence labels directly from evidence rows whose `diseaseId` or `diseaseFromSourceMappedId` maps to the prespecified cancer. For LUAD, 77 graph genes were positive in release 26.03 and 197 in release 26.06; 120 were present only in the later release. These counts and the contributing evidence rows are stored under `results/01_labels`.

## Reporting hierarchy

1. The release-26.06 disease-specific LUAD endpoint is the preferred current internal endpoint.
2. The 26.03-to-26.06 analysis is a cross-release validation, not a strictly prospective experiment. Changes may reflect new evidence, ontology mapping and Open Targets processing changes.
3. The within-26.06 evidence-date split fixes the endpoint schema, but only 14 post-cutoff incident genes were available and the graph-feature snapshots were not independently frozen at the cutoff. It is therefore a low-powered date-split sensitivity analysis, not a prospective validation.
4. The 1,184-positive historical endpoint is retained solely to reproduce the earlier benchmark, network-bias audit, PU-label sensitivity and uniformly trained neural comparators. Results from this endpoint must not be described as direct validation of the current disease-specific endpoint.

## Interpretation boundary

In every endpoint, zero denotes absence of the specified clinical evidence in the relevant snapshot, not proof that a gene is biologically unrelated to cancer. The task is evidence-status prediction and computational prioritization; it is not causal-gene discovery or therapeutic-efficacy prediction.
