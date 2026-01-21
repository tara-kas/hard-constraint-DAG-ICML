"""
wraps a simple Transformer-based encoder for text (can change encoder)
"""
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from new_core import DAGConstraintLayer

class TextDAGClassifier(nn.Module):
    def __init__(self, backbone_name, adj_json, max_iters=10):
        super().__init__()
        with open(adj_json) as f:
            adj = {int(k): v for k, v in json.load(f).items()}
        num_nodes = 1 + max(adj.keys()) if adj else 0
        self.encoder_config = AutoConfig.from_pretrained(backbone_name)
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden = self.encoder_config.hidden_size
        self.classifier = nn.Linear(hidden, num_nodes)
        self.dag_layer = DAGConstraintLayer(adj, num_nodes, max_iters=max_iters)

    def forward(self, input_ids, attention_mask, labels=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = h[:, 0, :]
        logits = self.classifier(cls)
        probs = self.dag_layer(logits)
        out = {"logits": logits, "probs": probs}
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            out["loss"] = loss
        return out
