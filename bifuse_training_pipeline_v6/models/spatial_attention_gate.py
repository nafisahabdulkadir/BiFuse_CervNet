# models/spatial_attention_gate.py
#
# Replaces BidirectionalGate.
#
# Why the old gate could not work:
#   g = sigmoid(w_g * attn + b_g), w_g and b_g scalars, attn = ||LayerNorm tok||.
#   LayerNorm forces ||tok|| == sqrt(d) exactly, so attn was a dataset-wide
#   constant (16.0 for d=256). g was therefore constant too (0.9998 at
#   w_g~0.53), and .view(-1,1,1,1) broadcast that one number over every spatial
#   position. Two learnable parameters, zero spatial selectivity, and a sigmoid
#   derivative of ~1e-7 so effectively no gradient either.
#
# This module produces g of shape [B, 1, H, W] from real ViT signals, and
# supports three modes so the gate's contribution can be measured rather than
# assumed:
#
#   'spatial'  - the proposed gate: per-position map, learned from the target
#                token embedding, modulated by real CLS->target attention.
#   'scalar'   - per-patch scalar broadcast spatially. Matches the OLD gate's
#                expressive power but driven by real attention, so it separates
#                "scalar gating is too weak" from "the input was constant".
#   'identity' - g == 1.0 everywhere. CONTROL ARM. Any M4-vs-M3 gain that
#                survives this mode did not come from the gate.
#
# Normalisation:
#   normalize=True applies a softmax over spatial positions rescaled to mean 1.
#   This forces the gate to ALLOCATE emphasis rather than uniformly amplify,
#   which is the failure mode we are trying to rule out. It cannot saturate to
#   a constant map without also flattening, which the diagnostics will catch.

import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_MODES = ('spatial', 'scalar', 'identity')


class SpatialAttentionGate(nn.Module):
    def __init__(self, vit_dim=256, grid_h=8, grid_w=8, mode='spatial',
                 normalize=True, temperature=1.0, use_attention=True):
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f'mode must be one of {VALID_MODES}, got {mode!r}')
        self.mode = mode
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.normalize = normalize
        self.use_attention = use_attention

        # Temperature is learned in log space so it stays positive. Not
        # created for 'identity': that mode returns a constant before this
        # parameter would ever be used, and a genuine control arm should have
        # exactly zero trainable parameters, not one inert one — a prior
        # version left this parameter in place for all modes, which showed up
        # as "gate_params=1" for identity in diagnostics rather than 0.
        if mode != 'identity':
            self.log_temp = nn.Parameter(torch.tensor(float(temperature)).log())

        if mode == 'spatial':
            self.to_map = nn.Linear(vit_dim, grid_h * grid_w)
            # Zero-init the bias: start at logit 0 -> sigmoid 0.5, mid-range,
            # where the gradient is maximal. The old gate started deep in the
            # flat tail.
            nn.init.zeros_(self.to_map.bias)
            nn.init.normal_(self.to_map.weight, std=0.02)
        elif mode == 'scalar':
            self.to_scalar = nn.Linear(vit_dim, 1)
            nn.init.zeros_(self.to_scalar.bias)
            nn.init.normal_(self.to_scalar.weight, std=0.02)

        # Scales real CLS->target attention into the gate logits. Attention
        # weights over 10 tokens sit near 0.1, so they are centred and scaled
        # rather than used raw.
        #
        # IMPORTANT — what this term does and does not do: cls_to_target is a
        # scalar per patch (one number, not a spatial map), so attn_scale * a
        # is a UNIFORM shift applied identically to all H*W logit positions.
        # It can change a patch's overall gate magnitude relative to other
        # patches; it cannot, by itself, make the gate vary spatially WITHIN
        # a patch. All within-patch (spatial) variation comes from to_map
        # (mode='spatial') acting on target_tok. This term is initialised at
        # zero, so at the start of training it contributes nothing at all —
        # its only role is letting the gate additionally weight patches by
        # how strongly the CLS token attended to them, which is a legitimate
        # but secondary signal, not the source of spatial selectivity.
        if use_attention and mode != 'identity':
            self.attn_scale = nn.Parameter(torch.zeros(1))

    def compute_gate(self, target_tok, attn_info, spatial_hw=None):
        """Returns g with shape [B, 1, H, W].

        Spatial (within-patch) variation comes entirely from to_map(target_tok)
        in 'spatial' mode. attn_scale (see __init__) only ever contributes a
        per-patch uniform shift — check spatial_std in
        BiFuseCervNet.get_gate_values() to confirm the gate is actually
        varying spatially, not attn_cls_to_target_std, which only confirms
        that attention itself is non-degenerate (the original bug).
        """
        B = target_tok.size(0)
        device = target_tok.device
        H, W = spatial_hw if spatial_hw is not None else (self.grid_h, self.grid_w)

        if self.mode == 'identity':
            return torch.ones(B, 1, H, W, device=device, dtype=target_tok.dtype)

        if self.mode == 'spatial':
            logits = self.to_map(target_tok)                      # [B, HW]
            logits = logits.view(B, 1, self.grid_h, self.grid_w)
        else:  # scalar
            logits = self.to_scalar(target_tok).view(B, 1, 1, 1)
            logits = logits.expand(B, 1, self.grid_h, self.grid_w)

        if self.use_attention:
            # Centre attention on uniform (1/n_tokens) so a neutral attention
            # distribution contributes nothing.
            a = attn_info['cls_to_target']                          # [B]
            n_tokens = attn_info['target_to_all'].size(-1)
            a = (a - 1.0 / n_tokens).view(B, 1, 1, 1)
            logits = logits + self.attn_scale * a

        logits = logits / self.log_temp.exp().clamp(min=1e-3)

        # Spatial softmax is a WITHIN-patch normalisation. Applying it to a
        # constant map yields exactly uniform 1.0 — this is a DIFFERENT
        # failure mode than the old gate's saturation, worth naming
        # explicitly: the old gate saturated toward a sigmoid ceiling because
        # its input was mathematically constant, and could never recover.
        # This gate cannot get stuck the same way (its input, target_tok,
        # genuinely varies), but weight decay or a poor local optimum could
        # still shrink to_map's weights toward zero over training, at which
        # point softmax over near-zero logits also approaches uniform and the
        # gate quietly becomes a no-op. Unlike the old bug this is gradual and
        # recoverable, not permanent — but it means spatial_std
        # (BiFuseCervNet.get_gate_values()) should be logged as a
        # TRAINING-TIME CURVE, not just checked once at the end, so a drift
        # toward uniformity during training is visible when it happens rather
        # than discovered post-hoc the way the original bug was.
        # spatially constant map yields exactly uniform 1.0, which would make
        # 'scalar' mode indistinguishable from 'identity' and destroy the
        # control. Scalar mode therefore always uses sigmoid, scaled to mean 1
        # so its average magnitude is comparable to the normalised spatial map.
        if self.normalize and self.mode == 'spatial':
            flat = logits.view(B, -1)
            g = F.softmax(flat, dim=-1) * flat.size(-1)   # mean 1 by construction
            g = g.view(B, 1, self.grid_h, self.grid_w)
        else:
            g = torch.sigmoid(logits) * 2.0   # sigmoid(0)=0.5 -> 1.0 at init

        if (H, W) != (self.grid_h, self.grid_w):
            g = F.interpolate(g, size=(H, W), mode='bilinear', align_corners=False)
        return g

    def forward(self, target_tok, attn_info, spatial):
        """spatial: [B, C, H, W] backbone feature map. Returns gated map."""
        g = self.compute_gate(target_tok, attn_info, spatial_hw=spatial.shape[-2:])
        return g * spatial, g
