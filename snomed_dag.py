"""
read and parse SNOMED CT dataset
prioritises snapshot ver first, then defaults to full
"""
import csv
from collections import defaultdict
from pathlib import Path

IS_A = "116680003"

def load_active_concepts(concept_file):
    active = set()
    with open(concept_file, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row["active"] == "1":
                active.add(row["id"])
    return active

def load_is_a_relationships(rel_file, active_concepts):
    parents = defaultdict(list)
    with open(rel_file, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row["active"] != "1":
                continue
            if row["typeId"] != IS_A:
                continue
            src = row["sourceId"]
            dst = row["destinationId"]
            if src in active_concepts and dst in active_concepts:
                # dst is parent of src
                parents[dst].append(src)
    return parents

def build_index(concepts, parents_dict):
    nodes = sorted(concepts)
    idx = {cid: i for i, cid in enumerate(nodes)}
    adj = {}
    for p, children in parents_dict.items():
        if p not in idx:
            continue
        p_idx = idx[p]
        ch_idx = [idx[c] for c in children if c in idx]
        if ch_idx:
            adj[p_idx] = ch_idx
    return idx, adj

def build_snomed_dag(base_dir, use_snapshot=True):
    base = Path(base_dir)
    term_dir = base / "Snapshot" / "Terminology" if use_snapshot else base / "Full" / "Terminology"
    concept_file = next(term_dir.glob("sct2_Concept_Snapshot_*.txt"))
    rel_file = next(term_dir.glob("sct2_Relationship_Snapshot_*.txt"))
    active = load_active_concepts(concept_file)
    parents_dict = load_is_a_relationships(rel_file, active)
    idx_map, adj = build_index(active, parents_dict)
    return idx_map, adj

if __name__ == "__main__":
    # Example usage
    import json, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf2_root", required=True)
    ap.add_argument("--idx_out", default="snomed_idx.json")
    ap.add_argument("--adj_out", default="snomed_adj.json")
    args = ap.parse_args()

    idx, adj = build_snomed_dag(args.rf2_root)
    import json
    with open(args.idx_out, "w") as f:
        json.dump(idx, f)
    with open(args.adj_out, "w") as f:
        json.dump(adj, f)
