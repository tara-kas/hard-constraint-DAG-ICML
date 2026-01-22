## DAG-Proj: Hard-Constraint Projection for DAG Ontologies in Neural Classification
Scalable Hard-Constraint Projection for DAG Ontologies in Neural Classification

Problem statement: Real-world knowledge is structured in direct acyclic graphs (DAGs) rather than trees. In the application of healthcare, a “viral pneumonia” is both a “viral infection” and a “respiratory disease”. A child word can have multiple given parents. Major ontological datasets such as Gene Ontology (45,000 concepts) or SNOMED-CT (350,000 concepts) are explicit DAGs (as opposed to trees forcefully formatted into DAGs) with heavy multiple-inheritance. 

Neural classifiers ignore this structure and are prone to producing logically fallible outputs, like a high confidence for “viral pneumonia” but a low confidence for “respiratory disease”. This violates semantics of the inherent ontology which can undermine credibility and trust in critical domains such as medical diagnosis.

This repository implements DAG-Proj, a scalable hard-constraint projection layer for enforcing DAG ontology constraints (e.g., SNOMED CT) in neural classification. It provides:

- RF2 to DAG parsing for SNOMED CT International Edition 2026
- A PyTorch DAG projection layer (DAGConstraintLayer)
- An end-to-end text to multi-label classification pipeline
- Hooks for benchmarking vs. standard QP solvers (e.g., Gurobi)

# 1. Data requirements
1.1 SNOMED CT International Edition 2026
You need the official RF2 release, e.g.:
SnomedCT_InternationalRF2_PRODUCTION_20260101T120000Z

Steps:
1. Create an account and accept the SNOMED CT license via your national release center or SNOMED International portal.
2. Download the International Edition in RF2 format (Full/Snapshot/Delta).
3. Unzip so you have a structure like:

    SnomedCT_InternationalRF2_PRODUCTION_20260101T120000Z/
        Snapshot/
            Terminology/
            sct2_Concept_Snapshot_*.txt
            sct2_Relationship_Snapshot_*.txt
            ...
        Full/
        Delta/

    This repo only uses the Snapshot/Terminology file

1.2 MIMIC-III clinical notes
Steps: 
1. Request access to MIMIC-III via PhysioNet (complete the training + data use agreement).
2. After approval, download the CSV bundle and extract it:
    You need NOTEEVENTS.csv and DIAGNOSES_ICD.csv from MIMIC-III
    You will also need a simple ICD9 → SNOMED mapping, and name it icd9_to_snomed.csv: [mapping](https://www.nlm.nih.gov/research/umls/mapping_projects/icd9cmv3_to_snomedct.html) or try [here](https://athena.ohdsi.org/search-terms/start)

Run: python prepare_mimic_notes.py --mimic_root /path/to/file --icd9_snomed_map /path/to/file

This will give you notes_snomed.csv

# 2. Building SNOMED-DAG
The first step is to parse RF2 and build:\
    snomed_idx.json: mapping SNOMED concept ID (string) -> node index (int)\
    snomed_adj.json: adjacency dict parent_index (int) -> [child_indices]

2.1 snomed_dag.py\
Run snomed_dag.py with arg --rf2_root {path to file}\
ie. python snomed_dag.py --rf2_root /path/to/file\


# 3. Classification Pipeline
Once you have snomed_idx.json, snomed_adj.json, notes_snomed.csv, run:

python3 train_eval.py \
  --rf2_idx snomed_idx.json \
  --adj_json snomed_adj.json \
  --notes_csv notes_snomed.csv

Example with explicit hyperparameters:
Run
python3 train_eval.py \
  --rf2_idx snomed_idx.json \
  --adj_json snomed_adj.json \
  --notes_csv notes_snomed.csv \
  --backbone distilbert-base-uncased \
  --epochs 3 \
  --batch_size 8 \
  --lr 2e-5

# 4. File Overview
- dag_projection.py\
    PyTorch implementation of the hard-constraint DAG projection (DAGProjection, DAGConstraintLayer).

- snomed_dag.py\
    Parses SNOMED CT RF2 and creates snomed_idx.json and snomed_adj.json.

- dataset.py\
    ClinicalNotesDataset for (text, concept_ids) pairs and a simple collate function.

- model.py\
    TextDAGClassifier which combines a transformer encoder, linear classifier, and DAGConstraintLayer.

- train_eval.py\
    End-to-end training and evaluation: trains the model, computes F1 and violation rate.