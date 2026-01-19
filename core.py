""" 
Core DAG projection layer
Pytorch implementation of hard-constraint projection for DAG ontologies in neural classification
"""

import torch
import torch.nn as nn
from collections import deque

class DAGProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_logits, adj_list, topo_order):
        """
        input_logits: (batch_size, num_nodes) raw sigmoid outputs
        adj_list: dict {parent_idx: [child_indices]}
        topo_order: list of nodes in topological order
        """
        p = input_logits.clone()
        
        # Downward Pass: Enforce p_child <= p_parent
        # Following topological order ensures we only need one pass for feasibility
        for parent in topo_order:
            if parent in adj_list:
                for child in adj_list[parent]:
                    p[:, child] = torch.min(p[:, child], p[:, parent])
        
        ctx.save_for_backward(input_logits, p)
        return p

    @staticmethod
    def backward(ctx, grad_output):
        # Using the Straight-Through Estimator (STE) for the backward pass
        # This allows gradients to flow directly to the violative logits
        input_logits, projected_p = ctx.saved_tensors
        return grad_output, None, None

class DAGConstraintLayer(nn.Module):
    def __init__(self, adj_list, num_nodes):
        super(DAGConstraintLayer, self).__init__()
        self.adj_list = adj_list
        self.num_nodes = num_nodes
        self.topo_order = self._compute_topo_sort()
        
        # Precompute edge list for violation checking
        self.edge_list = []
        for parent, children in adj_list.items():
            for child in children:
                self.edge_list.append((parent, child))

    def _compute_topo_sort(self):
        in_degree = {i: 0 for i in range(self.num_nodes)}
        for u in self.adj_list:
            for v in self.adj_list[u]:
                in_degree[v] += 1
        
        queue = deque([i for i in in_degree if in_degree[i] == 0])
        topo_order = []
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            if u in self.adj_list:
                for v in self.adj_list[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)
        return topo_order

    def forward(self, x):
        # x is assumed to be raw logits; apply sigmoid first
        probs = torch.sigmoid(x)
        return DAGProjection.apply(probs, self.adj_list, self.topo_order)

