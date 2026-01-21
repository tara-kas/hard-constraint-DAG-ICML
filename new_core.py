""" 
Core DAG projection layer
Pytorch implementation of hard-constraint projection for DAG ontologies in neural classification
ADDED: upward pass for optimality
"""

import torch
import torch.nn as nn
from collections import deque

class DAGProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_probs, adj_list, reverse_adj, topo_order, max_iters=10, tol=1e-6):
        """
        input_probs: (batch_size, num_nodes) raw sigmoid outputs
        adj_list: dict {parent_idx: [child_indices]}
        reverse_adj: dict {child_idx: [parent_indices]}
        topo_order: list of nodes in topological order
        max_iters: maximum iterations for convergence
        tol: convergence tolerance
        """
        q = input_probs.clone()
        p = input_probs  # original predictions
        
        for iteration in range(max_iters):
            q_old = q.clone()
            
            # DOWNWARD PASS (Feasibility)
            # Enforce q_child <= q_parent
            for node in topo_order:
                if node in adj_list:
                    for child in adj_list[node]:
                        q[:, child] = torch.min(q[:, child], q[:, node])
            
            # UPWARD PASS (Optimality)
            # Adjust parent nodes toward original while maintaining feasibility
            for node in reversed(topo_order):
                # Get constraints from children
                if node in adj_list and adj_list[node]:
                    # Parent must be >= all children
                    max_child = torch.max(
                        torch.stack([q[:, c] for c in adj_list[node]], dim=-1),
                        dim=-1
                    ).values
                else:
                    max_child = torch.zeros_like(q[:, node])
                
                # Get constraints from parents
                if node in reverse_adj and reverse_adj[node]:
                    # Node must be <= all parents
                    min_parent = torch.min(
                        torch.stack([q[:, par] for par in reverse_adj[node]], dim=-1),
                        dim=-1
                    ).values
                else:
                    min_parent = torch.ones_like(q[:, node])
                
                # Optimal projection: closest to original p within [max_child, min_parent]
                q[:, node] = torch.clamp(p[:, node], min=max_child, max=min_parent)
            
            # Check convergence
            if torch.max(torch.abs(q - q_old)) < tol:
                break
        
        ctx.save_for_backward(input_probs, q)
        ctx.mark_non_differentiable(q)
        return q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator
        return grad_output, None, None, None, None, None


class DAGConstraintLayer(nn.Module):
    def __init__(self, adj_list, num_nodes, max_iters=10):
        super().__init__()
        self.adj_list = adj_list
        self.num_nodes = num_nodes
        self.max_iters = max_iters
        
        # Build reverse adjacency (child -> parents)
        self.reverse_adj = {i: [] for i in range(num_nodes)}
        for parent, children in adj_list.items():
            for child in children:
                self.reverse_adj[child].append(parent)
        
        self.topo_order = self._compute_topo_sort()

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
        probs = torch.sigmoid(x)
        return DAGProjection.apply(
            probs, 
            self.adj_list, 
            self.reverse_adj,
            self.topo_order,
            self.max_iters
        )