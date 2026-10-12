# DCPI-RWR v1.0.0 reproduction order

Run from the package root with the same Python environment used by the endpoint-unified benchmark.

```powershell
python code/25_degree_conditioned_pathway_rwr.py `
  --dataset inputs/GNN_Dataset_Raw.pt `
  --mapping inputs/node_mapping.json `
  --labels-root results/01_labels `
  --output results/21_constrained_dcpi_rwr `
  --cancers LUAD BRCA CRC PRAD MEL OV `
  --bootstrap 2000 `
  --null-permutations 100

python code/27_loco_transfer_dcpi_rwr.py `
  --core-script code/25_degree_conditioned_pathway_rwr.py `
  --dataset inputs/GNN_Dataset_Raw.pt `
  --mapping inputs/node_mapping.json `
  --labels-root results/01_labels `
  --tuning results/21_constrained_dcpi_rwr/DCPI_RWR_historical_tuning_surface.csv.gz `
  --output results/20_loco_transfer `
  --bootstrap 2000

python code/28_pathway_component_permutation.py `
  --core-script code/25_degree_conditioned_pathway_rwr.py `
  --dataset inputs/GNN_Dataset_Raw.pt `
  --mapping inputs/node_mapping.json `
  --labels-root results/01_labels `
  --selected-configs results/21_constrained_dcpi_rwr/DCPI_RWR_selected_configs.json `
  --output results/23_pathway_component_permutation `
  --permutations 2000

python code/26_summarize_biasaware_framework.py `
  --dc-results results/21_constrained_dcpi_rwr `
  --external results/18_dcpi_rwr_external `
  --output results/22_final_innovation_summary
```

The TCGA/GSE68465 and CPTAC characterization scripts require the same external inputs used in the endpoint-unified package. They must be run only after the candidate file is locked.

## Fixed-candidate characterization

Use `code/09_characterize_dcpi_candidates.py` for outcome-blind TCGA-LUAD and GSE68465 characterization after a candidate table has been locked. The filename reflects the current DCPI-RWR study and replaces an earlier PU-stage filename; the implementation and statistical calculations are unchanged.
