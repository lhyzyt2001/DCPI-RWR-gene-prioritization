from __future__ import annotations

import argparse
import csv
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, rankdata, wilcoxon, zscore
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import survdiff

from common import BOOTSTRAPS, SEED, fdr, sha256, write_json


def signed_rank_biserial(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values != 0]
    if not len(values):
        return 0.0
    ranks = rankdata(np.abs(values))
    return float(np.sum(np.sign(values) * ranks) / np.sum(ranks))


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    medians = np.median(rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True), axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def tcga_expression(tcga_expression: Path, tcga_clinical: Path, genes: list[str], out: Path) -> None:
    expression = pd.read_csv(tcga_expression, sep="\t", index_col=0).groupby(level=0).mean()
    clinical = pd.read_csv(tcga_clinical, sep="\t")
    mapped = [gene for gene in genes if gene in expression.index]
    sample_data = expression.loc[mapped].T
    sample_data.index = sample_data.index.astype(str)
    sample_data["patient"] = sample_data.index.str[:12]
    sample_data["sample_type"] = sample_data.index.str[13:15]
    tumor = sample_data[sample_data["sample_type"].eq("01")].groupby("patient")[mapped].mean()
    normal = sample_data[sample_data["sample_type"].eq("11")].groupby("patient")[mapped].mean()
    patients = tumor.index.intersection(normal.index)
    rng = np.random.default_rng(SEED)
    paired_rows = []
    for rank, gene in enumerate(genes, start=1):
        if gene not in mapped:
            paired_rows.append({"Candidate_Rank": rank, "Gene": gene, "Status": "not mapped"})
            continue
        pair = pd.DataFrame({"Normal": normal.loc[patients, gene], "Tumor": tumor.loc[patients, gene]}).dropna()
        if pair.to_numpy().max() > 50:
            pair = np.log2(pair + 1)
        diff = (pair["Tumor"] - pair["Normal"]).to_numpy()
        try:
            p_value = float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            p_value = 1.0
        lower, upper = bootstrap_median_ci(diff, rng)
        paired_rows.append({
            "Candidate_Rank": rank, "Gene": gene, "Status": "tested", "Paired_Patients": len(pair),
            "Normal_Median": float(pair["Normal"].median()), "Tumor_Median": float(pair["Tumor"].median()),
            "Median_Paired_Difference": float(np.median(diff)), "Difference_CI_Lower": lower,
            "Difference_CI_Upper": upper, "Rank_Biserial_Effect": signed_rank_biserial(diff),
            "Wilcoxon_P": p_value,
        })
    paired = pd.DataFrame(paired_rows)
    paired["Wilcoxon_FDR"] = fdr(paired.get("Wilcoxon_P", pd.Series(np.nan, index=paired.index)))
    paired.to_csv(out / "TCGA_paired_tumor_normal_top20.csv", index=False, encoding="utf-8-sig")

    expr = expression.loc[mapped].T.reset_index().rename(columns={"index": "sample"})
    expr = expr[expr["sample"].astype(str).str[13:15].eq("01")]
    stage = clinical[["sampleID", "pathologic_stage"]].drop_duplicates("sampleID").rename(columns={"sampleID": "sample"})
    merged = expr.merge(stage, on="sample", how="inner")

    def group_stage(value):
        text = str(value).upper().replace("STAGE", "").strip()
        if text.startswith(("III", "IV")):
            return "late"
        if text.startswith(("I", "II")):
            return "early"
        return np.nan

    merged["stage_group"] = merged["pathologic_stage"].map(group_stage)
    stage_rows = []
    for rank, gene in enumerate(genes, start=1):
        if gene not in mapped:
            stage_rows.append({"Candidate_Rank": rank, "Gene": gene, "Status": "not mapped"})
            continue
        frame = merged[[gene, "stage_group"]].dropna()
        values = pd.to_numeric(frame[gene], errors="coerce")
        if values.max() > 50:
            values = np.log2(values + 1)
        early = values[frame["stage_group"].eq("early")].dropna().to_numpy()
        late = values[frame["stage_group"].eq("late")].dropna().to_numpy()
        test = mannwhitneyu(late, early, alternative="two-sided")
        stage_rows.append({
            "Candidate_Rank": rank, "Gene": gene, "Status": "tested", "Early_N": len(early),
            "Late_N": len(late), "Early_Median": float(np.median(early)), "Late_Median": float(np.median(late)),
            "Late_Minus_Early_Median": float(np.median(late) - np.median(early)),
            "Cliffs_Delta_Late_vs_Early": float(2 * test.statistic / (len(late) * len(early)) - 1),
            "Mann_Whitney_P": float(test.pvalue),
        })
    stage_result = pd.DataFrame(stage_rows)
    stage_result["Mann_Whitney_FDR"] = fdr(stage_result.get("Mann_Whitney_P", pd.Series(np.nan, index=stage_result.index)))
    stage_result.to_csv(out / "TCGA_stage_top20.csv", index=False, encoding="utf-8-sig")
    run_tcga_survival(expression, clinical, genes, out)


def map_stage(value) -> float:
    text = str(value).upper().replace("STAGE", "").strip()
    if text.startswith(("III", "IV")):
        return 1.0
    if text.startswith(("I", "II")):
        return 0.0
    return np.nan


def ph_fit(time: pd.Series, event: pd.Series, design: pd.DataFrame, cohort: str, endpoint: str, gene: str):
    fit = PHReg(time.to_numpy(), design.to_numpy(), status=event.to_numpy(), ties="efron").fit(disp=0)
    residuals = np.asarray(fit.schoenfeld_residuals, dtype=float)
    event_mask = event.to_numpy(dtype=int) == 1
    event_time = time.to_numpy(dtype=float)[event_mask]
    time_rank = rankdata(event_time) / (len(event_time) + 1.0)
    diagnostics = []
    for column_index, column in enumerate(design.columns):
        residual = residuals[event_mask, column_index]
        valid = np.isfinite(residual) & np.isfinite(time_rank)
        if valid.sum() >= 5 and np.nanstd(residual[valid]) > 0:
            rho, p_value = pearsonr(residual[valid], time_rank[valid])
        else:
            rho, p_value = np.nan, np.nan
        diagnostics.append({
            "Cohort": cohort, "Endpoint": endpoint, "Gene": gene, "Covariate": column,
            "N": len(time), "Events": int(event.sum()), "Schoenfeld_R": rho, "PH_P": p_value,
        })
    return fit, diagnostics


def run_tcga_survival(expression: pd.DataFrame, clinical: pd.DataFrame, genes: list[str], out: Path) -> None:
    clinical = clinical.copy()
    clinical["sampleID"] = clinical["sampleID"].astype(str)
    clinical = clinical[clinical["sampleID"].str[13:15].eq("01")].copy()
    clinical["patient"] = clinical["sampleID"].str[:12]
    clinical = clinical.sort_values("sampleID").drop_duplicates("patient", keep="first")
    survival = pd.DataFrame({
        "sample": clinical["sampleID"], "patient": clinical["patient"],
        "event": clinical["vital_status"].astype(str).str.upper().eq("DECEASED").astype(int),
        "death": pd.to_numeric(clinical["days_to_death"], errors="coerce"),
        "followup": pd.to_numeric(clinical["days_to_last_followup"], errors="coerce"),
        "age": pd.to_numeric(clinical["age_at_initial_pathologic_diagnosis"], errors="coerce"),
        "male": clinical["gender"].astype(str).str.upper().map({"MALE": 1.0, "FEMALE": 0.0}),
        "stage_high": clinical["pathologic_stage"].map(map_stage),
    })
    survival["time_months"] = np.where(survival["event"].eq(1), survival["death"], survival["followup"]) / 30.4375
    expression = expression.groupby(level=0).mean()
    rows, diagnostics = [], []
    for rank, gene in enumerate(genes, start=1):
        if gene not in expression.index:
            rows.append({"Candidate_Rank": rank, "Gene": gene, "Status": "not mapped"})
            continue
        work = survival[["sample", "time_months", "event", "age", "male", "stage_high"]].copy()
        work["expression"] = expression.loc[gene].reindex(work["sample"].astype(str)).to_numpy()
        work = work.replace([np.inf, -np.inf], np.nan).dropna()
        work = work[work["time_months"] > 0]
        design = pd.DataFrame({
            "expression_z": zscore(work["expression"]), "age_z": zscore(work["age"]),
            "male": work["male"].to_numpy(), "stage_high": work["stage_high"].to_numpy(),
        }, index=work.index)
        try:
            fit, detail = ph_fit(work["time_months"], work["event"], design, "TCGA-LUAD", "overall survival", gene)
            diagnostics.extend(detail)
            idx = design.columns.get_loc("expression_z")
            ci = np.exp(fit.conf_int()[idx])
            high = (work["expression"] >= work["expression"].median()).astype(int)
            _, logrank_p = survdiff(work["time_months"], work["event"], high)
            rows.append({
                "Candidate_Rank": rank, "Gene": gene, "Status": "tested", "N": len(work),
                "Events": int(work["event"].sum()), "Adjusted_HR_per_SD": float(np.exp(fit.params[idx])),
                "CI_Lower": float(ci[0]), "CI_Upper": float(ci[1]), "Adjusted_Cox_P": float(fit.pvalues[idx]),
                "LogRank_P": float(logrank_p),
            })
        except Exception as error:
            rows.append({"Candidate_Rank": rank, "Gene": gene, "Status": f"failed: {type(error).__name__}", "N": len(work), "Events": int(work["event"].sum())})
    result = pd.DataFrame(rows)
    result["Adjusted_Cox_FDR"] = fdr(result.get("Adjusted_Cox_P", pd.Series(np.nan, index=result.index)))
    result["LogRank_FDR"] = fdr(result.get("LogRank_P", pd.Series(np.nan, index=result.index)))
    result.to_csv(out / "TCGA_overall_survival_top20.csv", index=False, encoding="utf-8-sig")
    save_ph_diagnostics(pd.DataFrame(diagnostics), out / "TCGA_schoenfeld_diagnostics.csv")


def save_ph_diagnostics(detail: pd.DataFrame, path: Path) -> None:
    if len(detail):
        detail["PH_FDR"] = detail.groupby(["Cohort", "Endpoint", "Covariate"])["PH_P"].transform(
            lambda x: fdr(x)
        )
    detail.to_csv(path, index=False, encoding="utf-8-sig")


def read_gse_metadata(path: Path) -> pd.DataFrame:
    sample_ids = None
    records: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            values = next(csv.reader([line.rstrip("\n")], delimiter="\t", quotechar='"'))
            key, entries = values[0], values[1:]
            if key == "!Sample_geo_accession":
                sample_ids = entries
            elif key == "!Sample_title":
                records["title"] = entries
            elif key == "!Sample_characteristics_ch1":
                parsed = [entry.split(":", 1) for entry in entries]
                names = [item[0].strip().lower() if len(item) == 2 else "unparsed" for item in parsed]
                name = pd.Series(names).mode().iloc[0]
                records[name] = [item[1].strip() if len(item) == 2 else item[0].strip() for item in parsed]
    if sample_ids is None:
        raise ValueError("GSE68465 sample accessions were not found")
    frame = pd.DataFrame(index=sample_ids)
    for key, values in records.items():
        if len(values) == len(frame):
            frame[key] = values
    frame.index.name = "Sample"
    return frame


def read_gpl(path: Path) -> pd.DataFrame:
    header = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if line.startswith("!platform_table_begin"):
                header = line_number + 1
                break
    annotation = pd.read_csv(path, sep="\t", comment="!", compression="gzip", dtype=str, skiprows=header)
    id_column = next(c for c in annotation.columns if c.strip().upper() in {"ID", "ID_REF"})
    symbol_column = next(c for c in annotation.columns if c.strip().lower() in {"gene symbol", "gene_symbol"})
    mapping = annotation[[id_column, symbol_column]].rename(columns={id_column: "Probe", symbol_column: "Raw"}).dropna()
    mapping["Gene"] = mapping["Raw"].astype(str).str.split(r"\s*///\s*|\s*//\s*|;", regex=True)
    mapping = mapping.explode("Gene")
    mapping["Gene"] = mapping["Gene"].astype(str).str.strip().str.upper()
    return mapping.loc[mapping["Gene"].str.match(r"^[A-Z0-9_.-]+$"), ["Probe", "Gene"]].drop_duplicates()


def parse_tnm(value: str) -> tuple[float, float]:
    text = str(value).upper().replace(" ", "")
    n_match, t_match = re.search(r"PN([0-3])", text), re.search(r"PT([1-4])", text)
    return (
        float(int(n_match.group(1)) >= 1) if n_match else np.nan,
        float(int(t_match.group(1)) >= 3) if t_match else np.nan,
    )


def gse_characterization(matrix_file: Path, gpl_file: Path, genes: list[str], out: Path) -> None:
    metadata = read_gse_metadata(matrix_file)
    expression = pd.read_csv(matrix_file, sep="\t", comment="!", compression="gzip", quotechar='"', index_col=0)
    expression = expression.apply(pd.to_numeric, errors="coerce")
    if np.nanpercentile(expression.to_numpy(dtype=float), 99) > 50:
        expression = np.log2(expression.clip(lower=0) + 1)
    mapping = read_gpl(gpl_file)
    selected, audit = {}, []
    for rank, gene in enumerate(genes, start=1):
        probes = [p for p in mapping.loc[mapping["Gene"].eq(gene), "Probe"].unique() if p in expression.index]
        if probes:
            iqr = expression.loc[probes].quantile(0.75, axis=1) - expression.loc[probes].quantile(0.25, axis=1)
            probe = str(iqr.idxmax())
            selected[gene] = expression.loc[probe]
            audit.append({"Candidate_Rank": rank, "Gene": gene, "Selected_Probe": probe, "Probe_Count": len(probes), "Rule": "maximum IQR; outcome blind"})
        else:
            audit.append({"Candidate_Rank": rank, "Gene": gene, "Selected_Probe": np.nan, "Probe_Count": 0, "Rule": "not mapped"})
    gene_expression = pd.DataFrame(selected).T
    pd.DataFrame(audit).to_csv(out / "GSE68465_probe_selection_audit.csv", index=False, encoding="utf-8-sig")

    disease = metadata.get("disease_state", pd.Series(index=metadata.index, dtype=str)).astype(str).str.lower()
    tumor_samples = metadata.index[disease.str.contains("adenocarcinoma", na=False)]
    normal_samples = metadata.index[disease.str.contains("normal", na=False)]
    expression_rows = []
    for rank, gene in enumerate(genes, start=1):
        if gene not in gene_expression.index:
            expression_rows.append({"Candidate_Rank": rank, "Gene": gene, "Status": "not mapped"})
            continue
        tumor = gene_expression.loc[gene, gene_expression.columns.intersection(tumor_samples)].dropna().to_numpy(dtype=float)
        normal = gene_expression.loc[gene, gene_expression.columns.intersection(normal_samples)].dropna().to_numpy(dtype=float)
        test = mannwhitneyu(tumor, normal, alternative="two-sided")
        expression_rows.append({
            "Candidate_Rank": rank, "Gene": gene, "Status": "tested", "Tumor_N": len(tumor), "Normal_N": len(normal),
            "Tumor_Minus_Normal_Median": float(np.median(tumor) - np.median(normal)),
            "Cliffs_Delta": float(2 * test.statistic / (len(tumor) * len(normal)) - 1), "Mann_Whitney_P": float(test.pvalue),
        })
    exp_result = pd.DataFrame(expression_rows)
    exp_result["Mann_Whitney_FDR"] = fdr(exp_result.get("Mann_Whitney_P", pd.Series(np.nan, index=exp_result.index)))
    exp_result.to_csv(out / "GSE68465_tumor_normal_top20.csv", index=False, encoding="utf-8-sig")

    frame = metadata.loc[disease.str.contains("adenocarcinoma", na=False)].copy()
    frame["event"] = frame.get("vital_status", "").astype(str).str.lower().eq("dead").astype(int)
    frame["time_months"] = pd.to_numeric(frame.get("months_to_last_contact_or_death"), errors="coerce")
    frame["age"] = pd.to_numeric(frame.get("age"), errors="coerce")
    frame["male"] = frame.get("sex", "").astype(str).str.lower().map({"male": 1.0, "female": 0.0})
    tnm = frame.get("disease_stage", pd.Series(index=frame.index, dtype=str)).map(parse_tnm)
    frame["node_positive"] = [x[0] for x in tnm]
    frame["t_high"] = [x[1] for x in tnm]
    frame = frame[(frame["time_months"] > 0) & frame["time_months"].notna()]
    survival_rows, diagnostics = [], []
    for rank, gene in enumerate(genes, start=1):
        if gene not in gene_expression.index:
            survival_rows.append({"Candidate_Rank": rank, "Gene": gene, "Status": "not mapped"})
            continue
        work = frame[["time_months", "event", "age", "male", "node_positive", "t_high"]].copy()
        work["expression"] = gene_expression.loc[gene].reindex(work.index)
        work = work.replace([np.inf, -np.inf], np.nan).dropna()
        design = pd.DataFrame({
            "expression_z": zscore(work["expression"]), "age_z": zscore(work["age"]), "male": work["male"].to_numpy(),
            "node_positive": work["node_positive"].to_numpy(), "t_high": work["t_high"].to_numpy(),
        }, index=work.index)
        try:
            fit, detail = ph_fit(work["time_months"], work["event"], design, "GSE68465", "overall survival", gene)
            diagnostics.extend(detail)
            idx = design.columns.get_loc("expression_z")
            ci = np.exp(fit.conf_int()[idx])
            high = (work["expression"] >= work["expression"].median()).astype(int)
            _, logrank_p = survdiff(work["time_months"], work["event"], high)
            survival_rows.append({
                "Candidate_Rank": rank, "Gene": gene, "Status": "tested", "N": len(work), "Events": int(work["event"].sum()),
                "Adjusted_HR_per_SD": float(np.exp(fit.params[idx])), "CI_Lower": float(ci[0]), "CI_Upper": float(ci[1]),
                "Adjusted_Cox_P": float(fit.pvalues[idx]), "LogRank_P": float(logrank_p),
            })
        except Exception as error:
            survival_rows.append({"Candidate_Rank": rank, "Gene": gene, "Status": f"failed: {type(error).__name__}", "N": len(work), "Events": int(work["event"].sum())})
    survival = pd.DataFrame(survival_rows)
    survival["Adjusted_Cox_FDR"] = fdr(survival.get("Adjusted_Cox_P", pd.Series(np.nan, index=survival.index)))
    survival["LogRank_FDR"] = fdr(survival.get("LogRank_P", pd.Series(np.nan, index=survival.index)))
    survival.to_csv(out / "GSE68465_survival_top20.csv", index=False, encoding="utf-8-sig")
    save_ph_diagnostics(pd.DataFrame(diagnostics), out / "GSE68465_schoenfeld_diagnostics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Characterize fixed candidates in TCGA-LUAD and GSE68465.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--tcga-expression", type=Path, required=True)
    parser.add_argument("--tcga-clinical", type=Path, required=True)
    parser.add_argument("--gse-matrix", type=Path, required=True)
    parser.add_argument("--gpl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--selection-description",
        default="Fixed outcome-blind candidate set supplied in --candidates.",
        help="Auditable description of how the fixed candidate set was selected.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidates).sort_values("Candidate_Rank")
    gene_column = next(
        (column for column in ("Gene_Symbol", "Gene", "gene") if column in candidates.columns),
        None,
    )
    if gene_column is None:
        raise ValueError("Candidate table must contain Gene_Symbol, Gene, or gene.")
    genes = candidates[gene_column].astype(str).tolist()
    tcga_expression(args.tcga_expression, args.tcga_clinical, genes, args.out)
    gse_characterization(args.gse_matrix, args.gpl, genes, args.out)
    write_json(args.out / "characterization_manifest.json", {
        "candidate_selection": args.selection_description,
        "TCGA": "Paired Wilcoxon, stage Mann-Whitney, and age/sex/stage-adjusted continuous-expression Cox model.",
        "GSE68465": "Maximum-IQR probe selected without outcome labels; tumor-normal Mann-Whitney and age/sex/nodal/pT-adjusted Cox model.",
        "multiple_testing": "Benjamini-Hochberg within each endpoint across the fixed top 20.",
        "ph_diagnostic": "Pearson correlation of Schoenfeld residuals with ranked event time; FDR by cohort/endpoint/covariate.",
        "input_sha256": {p.name: sha256(p) for p in [args.candidates, args.tcga_expression, args.tcga_clinical, args.gse_matrix, args.gpl]},
        "interpretation": "Secondary molecular and clinical characterization, not experimental validation of causality or therapeutic efficacy.",
    })


if __name__ == "__main__":
    main()
