"""
Prepare MIMIC-III discharge summaries with SNOMED labels.

Inputs:
  - NOTEEVENTS.csv (from MIMIC-III)
  - DIAGNOSES_ICD.csv (from MIMIC-III)
  - icd9_to_snomed.csv: mapping file with columns:
        icd9,snomed_id
    (one SNOMED per row; duplicates allowed)

Output:
  - notes_snomed.csv with columns:
        text,concept_ids
    where concept_ids is a semicolon-separated list of SNOMED IDs.
"""

import argparse
import pandas as pd
from pathlib import Path

def load_discharge_summaries(noteevents_path, max_notes=None):
    df = pd.read_csv(noteevents_path)
    # Keep only discharge summaries
    df = df[df["CATEGORY"] == "Discharge summary"]
    # Basic cleanup
    df = df[["ROW_ID", "SUBJECT_ID", "HADM_ID", "TEXT"]].dropna(subset=["HADM_ID", "TEXT"])
    if max_notes is not None:
        df = df.sample(n=min(max_notes, len(df)), random_state=0)
    return df

def load_diagnoses_icd(diag_path):
    # DIAGNOSES_ICD: SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE
    diag = pd.read_csv(diag_path)
    diag = diag[["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]].dropna(subset=["ICD9_CODE"])
    return diag

def load_icd9_to_snomed(map_path):
    # Expect columns: icd9,snomed_id
    m = pd.read_csv(map_path, dtype=str)
    m = m.dropna(subset=["icd9", "snomed_id"])
    # Normalize ICD9 format (strip dots and spaces)
    m["icd9_norm"] = m["icd9"].str.replace(".", "", regex=False).str.strip()
    return m

def map_icd9_to_snomed(icd_codes, mapping_df):
    if len(icd_codes) == 0:
        return []
    df = pd.DataFrame({"ICD9_CODE": icd_codes})
    df["icd9_norm"] = df["ICD9_CODE"].str.replace(".", "", regex=False).str.strip()
    merged = df.merge(mapping_df, on="icd9_norm", how="left")
    snomed = merged["snomed_id"].dropna().unique().tolist()
    return snomed

def build_notes_snomed(noteevents_path, diag_path, map_path, out_path, max_notes=None):
    notes = load_discharge_summaries(noteevents_path, max_notes=max_notes)
    diag = load_diagnoses_icd(diag_path)
    mapping = load_icd9_to_snomed(map_path)

    # Join diagnoses to notes on SUBJECT_ID + HADM_ID
    merged = notes.merge(
        diag,
        on=["SUBJECT_ID", "HADM_ID"],
        how="left",
        suffixes=("", "_diag"),
    )

    # Group ICD codes per note
    grouped = merged.groupby("ROW_ID").agg(
        {
            "TEXT": "first",
            "ICD9_CODE": lambda codes: [c for c in codes.dropna().unique()],
        }
    ).reset_index()

    texts = []
    concept_lists = []
    for _, row in grouped.iterrows():
        text = row["TEXT"]
        icd_list = row["ICD9_CODE"]
        snomed_ids = map_icd9_to_snomed(icd_list, mapping)
        if not snomed_ids:
            continue  # skip notes with no mapped SNOMED codes
        texts.append(text.replace("\n", " ").strip())
        concept_lists.append(";".join(sorted(set(snomed_ids))))

    out_df = pd.DataFrame({"text": texts, "concept_ids": concept_lists})
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} notes with SNOMED labels to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_root", required=True, help="Path to folder containing NOTEEVENTS.csv and DIAGNOSES_ICD.csv")
    ap.add_argument("--icd9_snomed_map", required=True, help="Path to icd9_to_snomed.csv")
    ap.add_argument("--out_csv", default="notes_snomed.csv")
    ap.add_argument("--max_notes", type=int, default=None, help="Optional max number of discharge summaries to sample")
    args = ap.parse_args()

    root = Path(args.mimic_root)
    noteevents_path = root / "NOTEEVENTS.csv"
    diag_path = root / "DIAGNOSES_ICD.csv"

    build_notes_snomed(
        noteevents_path=str(noteevents_path),
        diag_path=str(diag_path),
        map_path=args.icd9_snomed_map,
        out_path=args.out_csv,
        max_notes=args.max_notes,
    )

if __name__ == "__main__":
    main()