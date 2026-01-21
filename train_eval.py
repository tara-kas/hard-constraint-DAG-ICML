"""
train and evaluation for F1 and violation rate
"""
import argparse
import json
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from dataset import ClinicalNotesDataset, collate_fn
from model import TextDAGClassifier

def violation_rate(probs, adj):
    # probs: (B, N)
    B, N = probs.shape
    total = 0
    viol = 0
    for p, children in adj.items():
        p_col = probs[:, p]
        for c in children:
            c_col = probs[:, c]
            total += B
            viol += (c_col > p_col + 1e-6).sum().item()
    return viol / max(total, 1)

def f1_scores(logits, labels, threshold=0.5):
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = (preds * labels).sum(dim=0)
    fp = (preds * (1 - labels)).sum(dim=0)
    fn = ((1 - preds) * labels).sum(dim=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_per_label = 2 * precision * recall / (precision + recall + 1e-8)
    macro_f1 = f1_per_label.mean().item()
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_precision = micro_tp / (micro_tp + micro_fp + 1e-8)
    micro_recall = micro_tp / (micro_tp + micro_fn + 1e-8)
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)).item()
    return macro_f1, micro_f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf2_idx", required=True)
    ap.add_argument("--adj_json", required=True)
    ap.add_argument("--notes_csv", required=True)
    ap.add_argument("--backbone", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    with open(args.rf2_idx) as f:
        idx_map = json.load(f)
    with open(args.adj_json) as f:
        raw_adj = json.load(f)
        adj = {int(k): v for k, v in raw_adj.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    dataset = ClinicalNotesDataset(args.notes_csv, idx_map, tokenizer)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextDAGClassifier(args.backbone, args.adj_json).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        all_logits = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
                all_logits.append(out["logits"].cpu())
                all_labels.append(batch["labels"].cpu())
                all_probs.append(out["probs"].cpu())
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        probs = torch.cat(all_probs, dim=0)
        macro_f1, micro_f1 = f1_scores(logits, labels)
        v_rate = violation_rate(probs, adj)
        print(f"Epoch {epoch+1}: macro F1={macro_f1:.4f}, micro F1={micro_f1:.4f}, violations={v_rate*100:.2f}%")

if __name__ == "__main__":
    main()
