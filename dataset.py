"""
mapping from clinical notes to SNOMED concept IDs
"""
import csv
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List

class ClinicalNotesDataset(Dataset):
    def __init__(self, csv_path, idx_map: Dict[str, int], tokenizer, max_len=256):
        self.samples = []
        self.idx_map = idx_map
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                text = row["text"]
                cids = [c for c in row["concept_ids"].split(";") if c]
                self.samples.append((text, cids))

        self.num_nodes = len(idx_map)

    def __len__(self):
        return len(self.samples)

    def encode_labels(self, cids: List[str]):
        y = torch.zeros(self.num_nodes, dtype=torch.float32)
        for cid in cids:
            if cid in self.idx_map:
                y[self.idx_map[cid]] = 1.0
        return y

    def __getitem__(self, idx):
        text, cids = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.encode_labels(cids)
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = labels
        return item

def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
