from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


POSITIVE_STAGES = {
    "EARLY_PHASE_1",
    "PHASE_1",
    "PHASE_1_2",
    "PHASE_2",
    "PHASE_2_3",
    "PHASE_3",
    "PHASE_4",
    "APPROVAL",
}

CANCERS = {
    "LUAD": "lung adenocarcinoma",
    "BRCA": "breast cancer",
    "CRC": "colorectal cancer",
    "PRAD": "prostate cancer",
    "MEL": "melanoma",
    "OV": "ovarian cancer",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_parquet_parts(directory: Path, columns: list[str] | None = None) -> pd.DataFrame:
    parts = sorted(directory.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Parquet parts in {directory}")
    return pd.concat([pd.read_parquet(p, columns=columns) for p in parts], ignore_index=True)


def disease_root_and_scope(path: Path, exact_name: str) -> tuple[str, set[str]]:
    disease = pd.read_parquet(path)
    hit = disease.loc[disease["name"].astype(str).str.casefold().eq(exact_name.casefold())]
    if len(hit) != 1:
        raise RuntimeError(f"Expected one exact disease match for {exact_name!r}, found {len(hit)}")
    row = hit.iloc[0]
    descendants = row["descendants"]
    scope = {str(row["id"])}
    if isinstance(descendants, (list, tuple, np.ndarray)):
        scope.update(map(str, descendants))
    return str(row["id"]), scope


def target_map(raw_root: Path) -> pd.DataFrame:
    target = read_parquet_parts(raw_root / "26.03" / "target", ["id", "approvedSymbol"])
    target = target.drop_duplicates("id").rename(columns={"id": "targetId", "approvedSymbol": "symbol"})
    target["symbol"] = target["symbol"].astype(str).str.upper()
    return target


def evidence_for_release(raw_root: Path, release: str) -> pd.DataFrame:
    columns = [
        "targetId",
        "diseaseId",
        "diseaseFromSourceMappedId",
        "clinicalStage",
        "evidenceDate",
        "studyStartDate",
        "clinicalReportId",
        "drugId",
    ]
    return read_parquet_parts(raw_root / release, columns)


def scoped_evidence(evidence: pd.DataFrame, scope: set[str]) -> pd.DataFrame:
    direct = evidence["diseaseId"].astype(str).isin(scope)
    mapped = evidence["diseaseFromSourceMappedId"].astype(str).isin(scope)
    stage = evidence["clinicalStage"].isin(POSITIVE_STAGES)
    return evidence.loc[(direct | mapped) & stage].copy()


def network_targets(evidence: pd.DataFrame, mapping: pd.DataFrame, network: set[str]) -> pd.DataFrame:
    out = evidence.merge(mapping, on="targetId", how="left", validate="many_to_one")
    out = out.loc[out["symbol"].isin(network)].copy()
    return out


def write_label(path: Path, symbols: set[str], column: str) -> None:
    pd.DataFrame({"symbol": sorted(symbols), column: 1}).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build disease-specific clinical-precedence labels.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--network-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-schema-cutoff", default="2016-12-31")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    mapping = target_map(args.raw_root)
    node_mapping = json.loads(args.network_mapping.read_text(encoding="utf-8-sig"))
    network = set(map(str.upper, node_mapping.keys()))
    evidence = {release: evidence_for_release(args.raw_root, release) for release in ("26.03", "26.06")}

    rows: list[dict] = []
    root_rows: list[dict] = []
    for cancer, disease_name in CANCERS.items():
        label_dir = args.output / cancer
        label_dir.mkdir(parents=True, exist_ok=True)
        sets: dict[str, set[str]] = {}
        for release in ("26.03", "26.06"):
            root_id, scope = disease_root_and_scope(
                args.raw_root / release / "disease" / "disease.parquet", disease_name
            )
            scoped = scoped_evidence(evidence[release], scope)
            mapped = network_targets(scoped, mapping, network)
            symbols = set(mapped["symbol"].dropna().astype(str))
            sets[release] = symbols
            mapped.to_csv(label_dir / f"{release}_disease_specific_evidence.csv.gz", index=False)
            root_rows.append(
                {
                    "cancer": cancer,
                    "disease_name": disease_name,
                    "release": release,
                    "root_id": root_id,
                    "scope_size": len(scope),
                    "evidence_rows": len(scoped),
                    "network_targets": len(symbols),
                    "evidence_date_nonmissing_fraction": float(
                        pd.to_datetime(scoped["evidenceDate"], errors="coerce").notna().mean()
                    ),
                }
            )
        incident = sets["26.06"] - sets["26.03"]
        retired = sets["26.03"] - sets["26.06"]
        write_label(label_dir / "historical_26.03.csv", sets["26.03"], "historical_positive")
        write_label(label_dir / "future_26.06.csv", sets["26.06"], "future_positive")
        write_label(label_dir / "incident_26.06.csv", incident, "incident_positive")
        rows.append(
            {
                "cancer": cancer,
                "disease_name": disease_name,
                "historical_targets": len(sets["26.03"]),
                "future_targets": len(sets["26.06"]),
                "incident_targets": len(incident),
                "retired_targets": len(retired),
            }
        )

    # Fixed-schema, fixed-release LUAD split. This removes release-pipeline and ontology changes,
    # but remains retrospective unless every feature source is also frozen before the cutoff.
    _, luad_scope = disease_root_and_scope(
        args.raw_root / "26.06" / "disease" / "disease.parquet", CANCERS["LUAD"]
    )
    luad = network_targets(scoped_evidence(evidence["26.06"], luad_scope), mapping, network)
    luad["date"] = pd.to_datetime(luad["evidenceDate"], errors="coerce")
    cutoff = pd.Timestamp(args.fixed_schema_cutoff)
    earliest = luad.dropna(subset=["date"]).groupby("symbol", as_index=False)["date"].min()
    historical = set(earliest.loc[earliest["date"] <= cutoff, "symbol"])
    future = set(earliest["symbol"])
    incident = set(earliest.loc[earliest["date"] > cutoff, "symbol"])
    fixed_dir = args.output / "LUAD_fixed_schema_date_split"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    write_label(fixed_dir / "historical_pre_cutoff.csv", historical, "historical_positive")
    write_label(fixed_dir / "future_all_dated.csv", future, "future_positive")
    write_label(fixed_dir / "incident_post_cutoff.csv", incident, "incident_positive")
    earliest.to_csv(fixed_dir / "earliest_evidence_date_by_network_gene.csv", index=False)
    fixed_design = {
        "release": "26.06",
        "dataset": "evidence_clinical_precedence",
        "disease": CANCERS["LUAD"],
        "cutoff": str(cutoff.date()),
        "historical_targets": len(historical),
        "future_dated_targets": len(future),
        "incident_targets": len(incident),
        "interpretation": "Fixed-release evidence-date split; not called prospective because graph features were not independently version-frozen before the cutoff.",
    }
    (fixed_dir / "design.json").write_text(json.dumps(fixed_design, indent=2), encoding="utf-8")

    pd.DataFrame(rows).to_csv(args.output / "multicancer_label_counts.csv", index=False)
    pd.DataFrame(root_rows).to_csv(args.output / "disease_scope_audit.csv", index=False)
    provenance = {
        "positive_stages": sorted(POSITIVE_STAGES),
        "cancers": CANCERS,
        "input_files": [
            {
                "path": str(p.resolve()),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
            for release in ("26.03", "26.06")
            for p in sorted((args.raw_root / release).glob("part-*.parquet"))
        ],
        "fixed_schema": fixed_design,
    }
    (args.output / "label_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(fixed_design, indent=2))


if __name__ == "__main__":
    main()
