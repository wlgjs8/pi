#!/usr/bin/env python3
"""C2: pi05_base's pretrained Gemma-300M ACTION EXPERT as the VA flow head. No 2B trunk.

Why (measured, 2026-08-11..17): with language content (null), the encoder (transplanted), and the
state discretization (reproduced) all controlled, every from-scratch head failed the closed-loop
terminal grasp at a different layer (gate / ratchet / final sequence) while pi05 and pi05-nolang
succeeded. The remaining variable is the pretrained decoder. C2 transplants it: SigLIP tokens
(pi05_base weights) -> new 1152->1024 projection -> the 18-layer expert with its adaRMS flow
conditioning and action projections, all initialised from pi05_base. The ONLY random-init parts are
the vision projection and the 14 discrete-state token embeddings.

Faithfulness notes (verified against openpi gemma.py / pi0.py):
  - width 1024, depth 18, heads 8, kv_heads 1 (GQA), head_dim 256, mlp 4096 GeGLU(gelu-tanh)
  - q scaled by head_dim**-0.5; RoPE max_wavelength 10_000 applied to q and k
  - adaRMS: per-norm Dense(1024->3072) of the time cond -> (scale, shift, gate);
    normed = rms(x)*(1+scale)+shift; residual = x + gate*branch_out; NO static scale param
  - final norm is adaRMS too (gate unused on the output)
  - time cond = swish(time_mlp_out(swish(time_mlp_in(posemb_sincos(t, 1024, 4e-3, 4.0)))))
  - attention groups: prefix (vision+state) attends prefix; suffix (H action tokens) attends all
  - actions are 32-dim padded (loss masked to the real 14 by the trainer), matching the
    transplanted action_in/out projections exactly
  - deviation vs pi05, accepted: the prefix is processed by the EXPERT weights (pi05 used the 2B).
  - PREFIX IS TIME-INDEPENDENT BY DESIGN: the prefix pass uses cond = zeros, so each adaRMS
    modulation Dense contributes only its BIAS (a fixed learned affine). This makes the per-layer
    prefix K/V cacheable across flow steps exactly like pi05's trunk KV cache (measured: without
    it, 10-step sampling recomputes 550 tokens x 18 layers x 10 = 106.7 ms; with it, one prefix
    pass + 10 cheap suffix passes). Training uses the SAME two-pass structure, so train and
    inference are self-consistent.
"""

from __future__ import annotations

import math
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

D, DEPTH, HEADS, HEAD_DIM, MLP = 1024, 18, 8, 256, 4096
ACT_DIM = 32
REAL_DIM = 14


def posemb_sincos(pos: torch.Tensor, dim: int, min_period: float, max_period: float) -> torch.Tensor:
    """openpi pi0.posemb_sincos: pos (B,) -> (B, dim)."""
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=pos.device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    ang = pos.float()[:, None] * (2.0 * math.pi / period)[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """openpi gemma._apply_rope: x (B,L,H,Dh), positions (B,L). fp32 inside, cast back."""
    dt = x.dtype
    x = x.float()
    d = x.shape[-1]
    freq_exp = (2.0 / d) * torch.arange(d // 2, device=x.device, dtype=torch.float32)
    timescale = 10_000.0**freq_exp
    radians = positions.float()[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]  # (B, L, 1, d/2)
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dt)


def _rms(x: torch.Tensor) -> torch.Tensor:
    """Uncentered RMS normalisation in fp32 (JAX RMSNorm), returned in fp32."""
    xf = x.float()
    var = xf.square().mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(var + 1e-6)


class ExpertBlock(nn.Module):
    """One transplanted Gemma-300M layer with adaRMS conditioning (raw-einsum params)."""

    def __init__(self) -> None:
        super().__init__()
        self.q_w = nn.Parameter(torch.empty(HEADS, D, HEAD_DIM))
        self.kv_w = nn.Parameter(torch.empty(2, 1, D, HEAD_DIM))
        self.o_w = nn.Parameter(torch.empty(HEADS, HEAD_DIM, D))
        self.mlp_gating = nn.Parameter(torch.empty(2, D, MLP))
        self.mlp_linear = nn.Parameter(torch.empty(MLP, D))
        self.attn_mod_k = nn.Parameter(torch.empty(D, 3 * D))
        self.attn_mod_b = nn.Parameter(torch.empty(3 * D))
        self.ffw_mod_k = nn.Parameter(torch.empty(D, 3 * D))
        self.ffw_mod_b = nn.Parameter(torch.empty(3 * D))

    def _modulate(self, x, cond, k, b):
        mod = cond @ k + b  # (B, 3D) in cond dtype
        scale, shift, gate = mod[:, None, :].chunk(3, dim=-1)
        h = _rms(x) * (1.0 + scale.float()) + shift.float()
        return h.to(x.dtype), gate

    def _qkv(self, h, positions):
        q = torch.einsum("btd,hdk->bthk", h, self.q_w) * (HEAD_DIM**-0.5)
        k = torch.einsum("btd,dk->btk", h, self.kv_w[0, 0])[:, :, None, :]
        v = torch.einsum("btd,dk->btk", h, self.kv_w[1, 0])  # (B,T,Dh)
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)[:, :, 0, :]  # (B,T,Dh)
        return q, k, v

    def _mlp(self, x, cond):
        h, gate = self._modulate(x, cond, self.ffw_mod_k, self.ffw_mod_b)
        ff = F.gelu(h @ self.mlp_gating[0], approximate="tanh") * (h @ self.mlp_gating[1])
        return x + gate * (ff @ self.mlp_linear)

    @staticmethod
    def _attend(q, k, v):
        """SDPA (flash) attention; q is PRE-SCALED by head_dim^-0.5 so scale=1.0 here.
        Memory: avoids materialising the (T x T) probs that OOM'd the einsum path at batch 64.
        q (B,T,H,Dh); k, v (B,S,Dh) single KV head broadcast to all query heads (GQA kv=1)."""
        B, T, H, Dh = q.shape
        qp = q.permute(0, 2, 1, 3)  # (B,H,T,Dh)
        kp = k[:, None].expand(B, H, k.shape[1], Dh)
        vp = v[:, None].expand(B, H, v.shape[1], Dh)
        o = F.scaled_dot_product_attention(qp, kp, vp, scale=1.0)
        return o.permute(0, 2, 1, 3)  # (B,T,H,Dh)

    def prefix_forward(self, x, zero_cond, positions):
        """Full self-attention within the prefix; returns (x_out, (k, v)) for the suffix cache."""
        h, gate = self._modulate(x, zero_cond, self.attn_mod_k, self.attn_mod_b)
        q, k, v = self._qkv(h, positions)
        o = self._attend(q, k, v)
        o = torch.einsum("bqhd,hdD->bqD", o, self.o_w)
        x = x + gate * o
        return self._mlp(x, zero_cond), (k, v)

    def suffix_forward(self, x, cond, positions, kv_prefix):
        """Suffix tokens attend [cached prefix; suffix] (full, no mask -- suffix sees everything)."""
        h, gate = self._modulate(x, cond, self.attn_mod_k, self.attn_mod_b)
        q, k_s, v_s = self._qkv(h, positions)
        k = torch.cat([kv_prefix[0], k_s], dim=1)
        v = torch.cat([kv_prefix[1], v_s], dim=1)
        o = self._attend(q, k, v)
        o = torch.einsum("bqhd,hdD->bqD", o, self.o_w)
        x = x + gate * o
        return self._mlp(x, cond)


class C2Policy(nn.Module):
    """SigLIP (pi05_base) -> [512 vision + 14 state tokens] -> pretrained 300M expert flow head."""

    def __init__(self, size: str, weights_dir: pathlib.Path, horizon: int, state_bins: int = 256, **_ignored) -> None:
        super().__init__()
        from transformers import SiglipVisionConfig
        from transformers import SiglipVisionModel

        weights_dir = pathlib.Path(weights_dir)
        cfg = SiglipVisionConfig(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
            vision_use_head=False,
        )
        self.backbone = SiglipVisionModel(cfg)
        sd = torch.load(weights_dir / "siglip_so400m_pi05base.pth", map_location="cpu")
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing[:3], unexpected[:3])

        self.horizon = horizon
        self.state_bins = state_bins
        # The ONLY random-init modules:
        self.vis_proj = nn.Linear(1152, D)
        self.state_vocab = nn.Embedding(state_bins, D)
        self.state_dim_emb = nn.Parameter(torch.randn(1, REAL_DIM, D) * 0.02)

        # Transplanted expert + flow projections.
        ex = torch.load(weights_dir / "c2_expert_pi05base.pth", map_location="cpu")
        self.blocks = nn.ModuleList([ExpertBlock() for _ in range(DEPTH)])
        with torch.no_grad():
            for i, blk in enumerate(self.blocks):
                blk.q_w.copy_(ex["q_w"][i])
                blk.kv_w.copy_(ex["kv_w"][i])
                blk.o_w.copy_(ex["o_w"][i])
                blk.mlp_gating.copy_(ex["mlp_gating"][i])
                blk.mlp_linear.copy_(ex["mlp_linear"][i])
                blk.attn_mod_k.copy_(ex["pre_attention_norm_1_dense_k"][i])
                blk.attn_mod_b.copy_(ex["pre_attention_norm_1_dense_b"][i])
                blk.ffw_mod_k.copy_(ex["pre_ffw_norm_1_dense_k"][i])
                blk.ffw_mod_b.copy_(ex["pre_ffw_norm_1_dense_b"][i])
        self.final_mod_k = nn.Parameter(ex["final_norm_dense_k"].clone())
        self.final_mod_b = nn.Parameter(ex["final_norm_dense_b"].clone())

        def _linear(k, b):
            lin = nn.Linear(k.shape[0], k.shape[1])
            with torch.no_grad():
                lin.weight.copy_(k.T)
                lin.bias.copy_(b)
            return lin

        self.action_in = _linear(ex["action_in_proj_k"], ex["action_in_proj_b"])  # 32 -> 1024
        self.action_out = _linear(ex["action_out_proj_k"], ex["action_out_proj_b"])  # 1024 -> 32
        self.time_in = _linear(ex["time_mlp_in_k"], ex["time_mlp_in_b"])
        self.time_out = _linear(ex["time_mlp_out_k"], ex["time_mlp_out_b"])

    # --- VAPolicy-compatible surface -------------------------------------------------------
    def encode(self, img_l, img_r, state):
        b = img_l.shape[0]
        tok = self.backbone(torch.cat([img_l, img_r], 0)).last_hidden_state  # (2B,256,1152)
        tok = self.vis_proj(tok)
        vis = torch.cat([tok[:b], tok[b:]], dim=1)  # (B,512,1024)
        idx = torch.clamp(torch.floor((state.float() + 1.0) / 2.0 * self.state_bins).long(), 0, self.state_bins - 1)
        st_tok = (self.state_vocab(idx) + self.state_dim_emb).to(vis.dtype)  # (B,14,1024)
        return torch.cat([vis, st_tok], dim=1)  # (B,526,1024)

    def precompute(self, ctx):
        """One time-independent prefix pass (cond = zeros -> bias-only adaRMS); returns the
        per-layer (k, v) cache the suffix passes attend."""
        b, p = ctx.shape[0], ctx.shape[1]
        zero_cond = torch.zeros(b, D, device=ctx.device, dtype=ctx.dtype)
        positions = torch.arange(p, device=ctx.device)[None].expand(b, -1)
        x, cache = ctx, []
        for blk in self.blocks:
            x, kv = blk.prefix_forward(x, zero_cond, positions)
            cache.append(kv)
        return cache, p

    def _time_cond(self, t, dtype):
        cond = posemb_sincos(t, D, 4e-3, 4.0).to(dtype)
        return F.silu(self.time_out(F.silu(self.time_in(cond))))

    def suffix_pass(self, cache, prefix_len, x_t, t, dtype):
        cond = self._time_cond(t, dtype)
        x = self.action_in(x_t.to(dtype))  # (B,H,1024)
        b = x.shape[0]
        positions = (prefix_len + torch.arange(self.horizon, device=x.device))[None].expand(b, -1)
        for blk, kv in zip(self.blocks, cache, strict=True):
            x = blk.suffix_forward(x, cond, positions, kv)
        mod = cond @ self.final_mod_k + self.final_mod_b
        scale, shift, _ = mod[:, None, :].chunk(3, dim=-1)
        out = (_rms(x) * (1.0 + scale.float()) + shift.float()).to(dtype)
        return self.action_out(out)  # (B,H,32)

    def decode(self, ctx, x_t, t):
        cache, p = self.precompute(ctx)
        return self.suffix_pass(cache, p, x_t, t, ctx.dtype)

    def forward(self, img_l, img_r, state, x_t=None, t=None):
        return self.decode(self.encode(img_l, img_r, state), x_t, t)

    @torch.no_grad()
    def sample_actions(self, img_l, img_r, state, num_steps: int = 10, noise_scale: float = 1.0, branch: str = "flow"):
        """Euler integration of openpi's convention (x_t = t*noise + (1-t)*a, u = noise - a).
        Returns (B, H, 14) -- sliced to the real dims for drop-in eval/serve compatibility."""
        ctx = self.encode(img_l, img_r, state)
        cache, p = self.precompute(ctx)
        b = ctx.shape[0]
        x = noise_scale * torch.randn(b, self.horizon, ACT_DIM, device=ctx.device, dtype=torch.float32)
        dt = -1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((b,), 1.0 + i * dt, device=ctx.device, dtype=torch.float32)
            u = self.suffix_pass(cache, p, x, t, ctx.dtype).float()
            x = x + dt * u
        return x[:, :, :REAL_DIM]
