# models/spatial_vit.py
#
# CHANGED vs recovered version:
#   OLD: attn_score = out[:, 1, :].norm(dim=-1)
#        This is the L2 norm of a post-LayerNorm token. LayerNorm projects every
#        token onto a sphere of radius sqrt(d), so this quantity is EXACTLY
#        sqrt(256) = 16.0 for every patch, with zero variance. It is not
#        attention, and it cannot vary. It is the root cause of gate saturation.
#
#   NEW: real attention weights are extracted from the final encoder layer via
#        need_weights=True, and the target-token embedding is returned so the
#        gate can build a spatial map from it.
#
# Token layout (10 tokens): [CLS, target, nb_0 ... nb_7]
#   index 0 = CLS
#   index 1 = target patch
#   index 2..9 = the 8 spatial neighbours

import torch
import torch.nn as nn
from einops import repeat

CLS_IDX = 0
TARGET_IDX = 1
N_NEIGHBOURS = 8
N_TOKENS = 2 + N_NEIGHBOURS


class SpatialViT(nn.Module):
    def __init__(self, in_dim=512, vit_dim=256, n_heads=8, n_layers=4,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        assert n_layers >= 1
        self.vit_dim = vit_dim

        self.proj = nn.Linear(in_dim, vit_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, vit_dim) * 0.02)
        self.pos_emb = nn.Linear(2, vit_dim)

        def _layer():
            return nn.TransformerEncoderLayer(
                d_model=vit_dim, nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True)

        # All layers but the last run as a stack. The final layer is held
        # separately so we can call its self-attention with need_weights=True.
        self.blocks = nn.TransformerEncoder(_layer(), num_layers=n_layers - 1) \
            if n_layers > 1 else None
        self.final = _layer()
        self.norm = nn.LayerNorm(vit_dim)

    def _final_layer_with_attn(self, x):
        """Run the final encoder layer manually to capture attention weights.

        Mirrors nn.TransformerEncoderLayer's norm_first=False (post-norm) path,
        which is the PyTorch default and what the recovered model used.
        """
        L = self.final
        attn_out, attn_w = L.self_attn(
            x, x, x, need_weights=True, average_attn_weights=True)
        x = L.norm1(x + L.dropout1(attn_out))
        ff = L.linear2(L.dropout(L.activation(L.linear1(x))))
        x = L.norm2(x + L.dropout2(ff))
        return x, attn_w

    def forward(self, tgt, nbs, pos):
        """
        tgt : [B, in_dim]      target patch backbone features
        nbs : [B, 8, in_dim]   neighbour patch backbone features
        pos : [B, 2]           target patch (x, y) position

        Returns
        -------
        z_vit      : [B, vit_dim]  CLS embedding (unchanged contract)
        target_tok : [B, vit_dim]  target-token embedding, for the gate
        attn_info  : dict
            'cls_to_target'    [B]      real attention, CLS -> target
            'cls_to_neighbours'[B, 8]   real attention, CLS -> each neighbour
            'target_to_all'    [B, 10]  real attention, target -> all tokens
        """
        B = tgt.size(0)
        t = self.proj(tgt).unsqueeze(1) + self.pos_emb(pos).unsqueeze(1)
        nb = self.proj(nbs)
        cls = repeat(self.cls, '1 1 d -> b 1 d', b=B)

        tokens = torch.cat([cls, t, nb], dim=1)           # [B, 10, D]
        if self.blocks is not None:
            tokens = self.blocks(tokens)

        out, attn_w = self._final_layer_with_attn(tokens)  # attn_w [B, 10, 10]
        out = self.norm(out)

        attn_info = {
            'cls_to_target': attn_w[:, CLS_IDX, TARGET_IDX],
            'cls_to_neighbours': attn_w[:, CLS_IDX, TARGET_IDX + 1:],
            'target_to_all': attn_w[:, TARGET_IDX, :],
        }
        return out[:, CLS_IDX, :], out[:, TARGET_IDX, :], attn_info
