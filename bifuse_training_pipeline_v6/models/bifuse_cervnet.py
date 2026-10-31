# models/bifuse_cervnet.py
#
# CHANGED vs recovered version:
#   - SpatialViT now returns (z_vit, target_token, attn_info) instead of
#     (z_vit, attn_score). attn_score was a constant; see spatial_vit.py.
#   - BidirectionalGate -> SpatialAttentionGate, with gate_mode selectable.
#   - get_gate_values() returns the full [B,1,H,W] map plus summary statistics,
#     so "did the gate vary" is answerable directly instead of by inference.
#   - Added gate_mode='none' — this is M3 (the paper's "unidirectional
#     CNN-to-ViT, no gate" baseline), not the same thing as 'identity'.
#     'identity' still routes CNN spatial features through gate_pool/
#     gate_proj (an extra learned layer) with g forced to 1 — the pathway
#     exists, it just does nothing multiplicatively. 'none' removes the
#     pathway entirely: no gate, no gate_pool, no gate_proj, zero related
#     parameters. Fusion instead uses f1, the CNN's OWN natural pooled+
#     projected global feature (self.proj(GAP(feat)) inside CNNEncoder) —
#     already computed every forward pass to feed the ViT, previously
#     discarded afterward rather than reaching fusion.
#     NOTE: the original M3 code no longer exists (lost in the same data-loss
#     event as everything else this project rebuilt from). This is a
#     reconstruction from the paper's textual description and standard
#     architectural convention, not recovered original code — worth a second
#     pair of eyes before treating results as a like-for-like replication of
#     the original ablation table.
#
# gate_mode:
#   'spatial'  proposed gate
#   'scalar'   old expressive power, real (non-constant) input
#   'identity' control arm, g == 1.0, pathway still present
#   'none'     M3 — no gate pathway at all

import torch
import torch.nn as nn

from models.cnn_encoder import CNNEncoder
from models.spatial_vit import SpatialViT
from models.spatial_attention_gate import SpatialAttentionGate


class BiFuseCervNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gate_mode = getattr(cfg, 'GATE_MODE', 'spatial')

        self.cnn1 = CNNEncoder(pretrained=True, out_dim=cfg.CNN_OUT_DIM)
        for p in self.cnn1.parameters():
            p.requires_grad = False

        self.vit = SpatialViT(
            in_dim=cfg.CNN_OUT_DIM,
            vit_dim=cfg.VIT_DIM,
            n_heads=cfg.VIT_HEADS,
            n_layers=cfg.VIT_LAYERS)

        if self.gate_mode != 'none':
            self.gate = SpatialAttentionGate(
                vit_dim=cfg.VIT_DIM,
                grid_h=getattr(cfg, 'SPATIAL_GRID_H', 8),
                grid_w=getattr(cfg, 'SPATIAL_GRID_W', 8),
                mode=self.gate_mode,
                normalize=getattr(cfg, 'GATE_NORMALIZE', True),
                use_attention=getattr(cfg, 'GATE_USE_ATTENTION', True))
            self.gate_pool = nn.AdaptiveAvgPool2d(1)
            self.gate_proj = nn.Linear(2048, cfg.CNN_OUT_DIM)
        else:
            self.gate = None  # M3: no gate module at all, zero related params

        fusion_in = cfg.CNN_OUT_DIM + cfg.VIT_DIM
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, cfg.FUSION_DIM),
            nn.BatchNorm1d(cfg.FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT))
        self.detect_head = nn.Linear(cfg.FUSION_DIM, cfg.NUM_CLASSES_DETECT)
        self.grade_head = nn.Linear(cfg.FUSION_DIM, cfg.NUM_CLASSES_GRADE)

    def _encode(self, patch, neighbours, positions):
        B = patch.size(0)
        H, W = patch.shape[2], patch.shape[3]
        nb_flat = neighbours.view(B * 8, 3, H, W)
        all_imgs = torch.cat([patch, nb_flat], dim=0)

        with torch.no_grad():
            all_feats, all_spatial = self.cnn1(all_imgs)

        f1 = all_feats[:B]
        nb_feat = all_feats[B:].view(B, 8, -1)
        f_spatial = all_spatial[:B]

        z_vit, target_tok, attn_info = self.vit(f1, nb_feat, positions)
        return z_vit, target_tok, attn_info, f_spatial, f1

    def forward(self, patch, neighbours, positions, return_gate=False):
        z_vit, target_tok, attn_info, f_spatial, f1 = self._encode(
            patch, neighbours, positions)

        if self.gate_mode == 'none':
            f2 = f1  # M3: CNN's own natural pooled feature, no gate at all
            g = None
        else:
            f_gated, g = self.gate(target_tok, attn_info, f_spatial)
            f2 = self.gate_proj(self.gate_pool(f_gated).flatten(1))

        fused = self.fusion(torch.cat([f2, z_vit], dim=1))
        det = self.detect_head(fused).squeeze(1)
        grd = self.grade_head(fused)
        if return_gate:
            return det, grd, g
        return det, grd

    @torch.no_grad()
    def get_gate_values(self, patch, neighbours, positions):
        """Returns the gate map and the statistics that diagnose saturation.

        spatial_std is the key number: it was 0.0 by construction in the old
        model. within_patch_range near zero means the gate is still acting as
        a uniform amplifier regardless of its shape.

        Not meaningful for gate_mode='none' (M3) — there is no gate to
        diagnose. Returns (None, {}) rather than crashing, so calling code
        doesn't need a special case for M3 specifically.
        """
        if self.gate_mode == 'none':
            return None, {}
        z_vit, target_tok, attn_info, f_spatial, f1 = self._encode(
            patch, neighbours, positions)
        g = self.gate.compute_gate(target_tok, attn_info,
                                   spatial_hw=f_spatial.shape[-2:])
        flat = g.flatten(1)
        stats = {
            'mean': g.mean().item(),
            'spatial_std': flat.std(dim=1, unbiased=False).mean().item(),
            'across_patch_std': flat.mean(dim=1).std(unbiased=False).item(),
            'within_patch_range': (flat.max(dim=1).values
                                   - flat.min(dim=1).values).mean().item(),
            'attn_cls_to_target_std': attn_info['cls_to_target'].std(unbiased=False).item(),
        }
        return g.detach(), stats
