"""
model_mat.py
------------
Molecular Attention Transformer (Maziarka et al., 2020), as used by CPMP
(Step 3). Self-attention at every layer blends three signals:

    A = ( lambda_a * softmax(Q K^T / sqrt(d_k))
        + lambda_d * g(D)
        + lambda_g * A_adj ) V

  - softmax(QK^T/sqrt(dk)): ordinary learned attention.
  - g(D): a kernel over the 3D distance matrix (atoms close in the folded
    3D structure attend to each other regardless of bond-graph distance).
  - A_adj: the raw bond adjacency matrix (directly bonded atoms get a
    dedicated boost).

lambda_a, lambda_d, lambda_g are scalar mixing weights that sum to 1. The
paper finds them by grid search rather than learning them end-to-end, so
they're exposed here as fixed hyperparameters (see LAMBDA presets below) --
pass your own via `lambdas=(la, ld, lg)` to MATModel.

Set `use_distance=False` / `use_adjacency=False` / `use_dummy_node=False`
to reproduce the Step 8 ablations.
"""
import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# A reasonable starting grid-search preset (paper doesn't publish exact
# values; these weight distance heavily, consistent with the ablation
# finding that the distance matrix matters most).
LAMBDA_PRESETS = {
    "balanced": (0.34, 0.33, 0.33),
    "distance_heavy": (0.20, 0.60, 0.20),
    "attention_heavy": (0.60, 0.20, 0.20),
}


def distance_kernel(dist: torch.Tensor, mask: torch.Tensor, kind: str = "softmax_neg") -> torch.Tensor:
    """
    g(D): turn a raw Euclidean distance matrix into an attention-like weight
    matrix where closer atoms get higher weight. `kind='softmax_neg'` applies
    softmax(-D) row-wise (closer -> higher weight); masked (padding) entries
    are pushed to -inf before the softmax so they contribute ~0.
    """
    big_neg = -1e9
    pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (B, N, N)
    if kind == "softmax_neg":
        logits = -dist
        logits = logits.masked_fill(pair_mask == 0, big_neg)
        return F.softmax(logits, dim=-1)
    elif kind == "exp_neg":
        w = torch.exp(-dist) * pair_mask
        denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return w / denom
    else:
        raise ValueError(f"unknown distance kernel kind: {kind}")


def normalize_adjacency(adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Row-normalize the binary adjacency matrix so it behaves like an attention distribution."""
    pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    adj = adj * pair_mask
    denom = adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return adj / denom


class MoleculeMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, lambdas: Tuple[float, float, float],
                 use_distance: bool = True, use_adjacency: bool = True, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.lambda_a, self.lambda_d, self.lambda_g = lambdas
        self.use_distance = use_distance
        self.use_adjacency = use_adjacency

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dist_kernel: torch.Tensor, adj_kernel: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        q = self.q_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, N, N)
        scores = scores.masked_fill(pair_mask.unsqueeze(1) == 0, -1e9)
        attn = F.softmax(scores, dim=-1)

        combined = self.lambda_a * attn
        if self.use_distance:
            combined = combined + self.lambda_d * dist_kernel.unsqueeze(1)
        if self.use_adjacency:
            combined = combined + self.lambda_g * adj_kernel.unsqueeze(1)

        combined = self.dropout(combined)
        out = torch.matmul(combined, v)  # (B, H, N, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(out)


class MATLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, lambdas, use_distance, use_adjacency,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = MoleculeMultiHeadAttention(d_model, n_heads, lambdas, use_distance, use_adjacency, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, dist_kernel, adj_kernel, mask):
        a = self.attn(x, dist_kernel, adj_kernel, mask)
        x = self.norm1(x + self.dropout(a))
        f = self.ff(x)
        x = self.norm2(x + self.dropout(f))
        return x


class MATModel(nn.Module):
    """
    Full CPMP/MAT regressor.

    Forward inputs (all padded to the same N, dummy node included if used):
        atom_feats: (B, N, F_in)
        adjacency:  (B, N, N) binary
        distance:   (B, N, N) Euclidean (or topological fallback)
        mask:       (B, N) 1 for real/dummy atoms, 0 for padding

    Output: (B,) predicted LogPexp.
    """
    def __init__(self, atom_feat_dim: int, d_model: int = 128, n_heads: int = 8, n_layers: int = 4,
                 d_ff: int = 256, lambdas: Tuple[float, float, float] = LAMBDA_PRESETS["balanced"],
                 use_distance: bool = True, use_adjacency: bool = True, use_dummy_node: bool = True,
                 distance_kernel_kind: str = "softmax_neg", dropout: float = 0.1):
        super().__init__()
        self.use_dummy_node = use_dummy_node
        self.distance_kernel_kind = distance_kernel_kind

        self.input_proj = nn.Linear(atom_feat_dim, d_model)
        self.layers = nn.ModuleList([
            MATLayer(d_model, n_heads, d_ff, lambdas, use_distance, use_adjacency, dropout)
            for _ in range(n_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, atom_feats: torch.Tensor, adjacency: torch.Tensor, distance: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(atom_feats)
        dist_kernel = distance_kernel(distance, mask, kind=self.distance_kernel_kind)
        adj_kernel = normalize_adjacency(adjacency, mask)

        for layer in self.layers:
            x = layer(x, dist_kernel, adj_kernel, mask)

        # Global pooling. With a dummy node, its own representation after
        # the attention stack already accumulates whole-molecule info
        # (Step 3), so we read it out directly; otherwise mask-aware mean.
        if self.use_dummy_node:
            dummy_repr = x[:, -1, :]  # dummy node is always the last slot
            pooled = dummy_repr
        else:
            m = mask.unsqueeze(-1)
            pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-8)

        return self.readout(pooled).squeeze(-1)


if __name__ == "__main__":
    # smoke test with random tensors
    torch.manual_seed(0)
    B, N, F_in = 4, 20, 43
    model = MATModel(atom_feat_dim=F_in, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    atom_feats = torch.randn(B, N, F_in)
    adjacency = (torch.rand(B, N, N) > 0.85).float()
    distance = torch.rand(B, N, N) * 10
    mask = torch.ones(B, N)
    out = model(atom_feats, adjacency, distance, mask)
    print("output shape:", out.shape, "sample values:", out[:3].detach().numpy())
