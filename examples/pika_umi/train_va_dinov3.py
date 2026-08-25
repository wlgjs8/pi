#!/usr/bin/env python3
"""VA policy: DINOv3 wrist encoder -> action chunk. No LLM; flow-matching head by default.

Why drop the language model. The dataset carries ONE prompt string across all 177,945 frames and it
never varies within an episode, so the text branch has zero discriminative information -- there is
nothing to lose by removing it. Measured cost of keeping it: the LLM prefix is 26.2 ms of the 50 ms
model latency, and on this hardware DINOv3 ViT-B plus an action head runs both wrist images in
3.4 ms.

Why latency is the thing worth buying. Val says perception is not the bottleneck: at the deployment
operating point the policy already localises the grasp to 2.68 mm in z, better than a non-parametric
kNN ceiling over frozen DINOv3 features. What actually costs millimetres is OBSERVATION AGE -- the
measured precision curve is 1.04 / 2.68 / 6.78 / 20.63 mm at 2 / 5 / 12 / 23 steps of lookahead, and
raising CHUNK_EXECUTE_STEPS from 4 to 6 (to escape a latency stall) moved the last executed row from
234 ms to 367 ms stale and degraded the grasp exactly as that curve predicts. A faster policy can
run a shorter execute window, which is fresher, which is more accurate.

Head choice is a FLAG, not a decision baked in. The dominant latency term is the LLM (26.2 ms of
50 ms), not flow matching (17 ms = 10 steps x 1.7 ms), and in this architecture only the small head
repeats per denoising step -- the backbone runs once. So `--head flow` keeps openpi's exact
formulation and still lands far under pi05. Run it as the primary arm: replacing PaliGemma with
DINOv3 is then the SINGLE variable against pi05, which is the only way a negative result stays
interpretable. `--head l2` is a secondary arm that prices the head choice on its own; note it also
makes medoid sampling meaningless, since that needs multiple draws.

Data comes through openpi's own transform stack (create_torch_dataset + transform_dataset), so
normalisation, resize and padding are IDENTICAL to the pi05 runs and the val numbers are comparable
cell-for-cell. Actions/state arrive padded to 32 dims; only the first 14 are real.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REAL_DIM = 14

# From dinov3/hub/backbones.py -- NOT the DinoVisionTransformer defaults. Building with the defaults
# and loading strict=False silently drops 37 tensors (storage tokens, LayerScale, qkv bias_mask) and
# yields a differently shaped network whose outputs still look plausible.
VIT_COMMON = dict(
    img_size=224,
    patch_size=16,
    in_chans=3,
    pos_embed_rope_base=100,
    pos_embed_rope_normalize_coords="separate",
    pos_embed_rope_rescale_coords=2,
    pos_embed_rope_dtype="fp32",
    ffn_ratio=4,
    qkv_bias=True,
    drop_path_rate=0.0,
    layerscale_init=1.0e-05,
    norm_layer="layernormbf16",
    ffn_layer="mlp",
    ffn_bias=True,
    proj_bias=True,
    n_storage_tokens=4,
    mask_k_bias=True,
)
VIT_SIZES = {
    "s": (dict(embed_dim=384, depth=12, num_heads=6), "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    "b": (dict(embed_dim=768, depth=12, num_heads=12), "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    "l": (dict(embed_dim=1024, depth=24, num_heads=16), "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
}


def time_emb(t: torch.Tensor, d: int) -> torch.Tensor:
    """Sinusoidal embedding of the flow time, same shape convention as openpi's posemb_sincos."""
    half = d // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
    ang = t[:, None] * freq[None, :] * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class VAPolicy(nn.Module):
    def __init__(
        self,
        size: str,
        weights_dir: pathlib.Path,
        horizon: int,
        layers: int = 4,
        head_mode: str = "flow",
        drop_path: float = 0.0,
        split_head: bool = False,
        backbone_type: str = "dinov3",
        wm_head: bool = False,
        state_tokens: int = 0,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.state_tokens = state_tokens
        if backbone_type == "siglip":
            # The C-experiment encoder: pi05's OWN vision tower (So400m/14, 428M), weights extracted
            # from the pi05_base JAX checkpoint (extract_siglip_pi05base.py). 256 tokens per wrist
            # image -> 512 vision tokens into the action head, exactly the vision half of what pi05's
            # prefix carries. Input convention is [-1, 1] (openpi model.py), NOT ImageNet stats.
            from transformers import SiglipVisionConfig
            from transformers import SiglipVisionModel

            assert drop_path == 0.0, "drop_path is a DINOv3 knob; not plumbed for SigLIP"
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
            d = 1152
        elif backbone_type == "sigdino":
            # Dual encoder: SigLIP (pi05's own, language-aligned semantics) + DINOv3 (self-sup dense
            # geometry), token streams concatenated (256+196 per wrist). Complementarity hypothesis
            # targets the observed rollout failure "grasps at bolt-less spots that match the floor
            # colour": semantic features can confuse low-contrast bolt/floor where dense self-sup
            # features still separate texture/geometry. Input is raw [0,1]; each branch applies its
            # own normalisation inside encode() (SigLIP [-1,1], DINOv3 ImageNet).
            from dinov3.models.vision_transformer import DinoVisionTransformer
            from transformers import SiglipVisionConfig
            from transformers import SiglipVisionModel

            assert drop_path == 0.0, "drop_path not plumbed for the dual-encoder arm"
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
            dims, ckpt2 = VIT_SIZES[size]
            self.backbone2 = DinoVisionTransformer(**dict(VIT_COMMON, **dims))
            sd2 = torch.load(weights_dir / ckpt2, map_location="cpu")
            self.backbone2.load_state_dict(sd2.get("model", sd2), strict=True)
            d = 1152
            self.dino_proj = nn.Linear(dims["embed_dim"], d)
        else:
            from dinov3.models.vision_transformer import DinoVisionTransformer

            dims, ckpt = VIT_SIZES[size]
            d = dims["embed_dim"]
            vit_kw = dict(VIT_COMMON, **dims)
            vit_kw["drop_path_rate"] = drop_path
            self.backbone = DinoVisionTransformer(**vit_kw)
            sd = torch.load(weights_dir / ckpt, map_location="cpu")
            self.backbone.load_state_dict(sd.get("model", sd), strict=True)  # strict: see module docstring
        self.horizon = horizon
        # One learned embedding per camera so the head can tell the two wrists apart after the
        # token streams are concatenated (the backbone is shared and sees them as separate images).
        self.cam_emb = nn.Parameter(torch.zeros(2, 1, d))
        if state_tokens > 0:
            # VLA-style DISCRETE state input, minus the LLM: quantize each of the 14 normalized
            # state dims into `state_tokens` bins over [-1,1] (pi05's exact np.digitize grid) and
            # embed each bin as a learned token (+ per-dim embedding), 14 tokens total. This
            # reproduces the VLA's state bottleneck without any language model: within-bin
            # information is destroyed, the symbol->value map must be memorized per bin (no linear
            # readout), and no gradient flows through the continuous state. Motivation (measured
            # 2026-08-16): the continuous state_proj makes "grip ~= state grip - eps" a near-linear
            # shortcut, which on robot frames becomes an echo spiral; pi05's discretized state is
            # one of the two structural reasons it resists that shortcut.
            self.state_vocab = nn.Embedding(state_tokens, d)
            self.state_dim_emb = nn.Parameter(torch.randn(1, REAL_DIM, d) * 0.02)
        else:
            self.state_proj = nn.Sequential(nn.Linear(REAL_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.head_mode = head_mode
        # l2: the H action slots are free-standing learned queries.
        # flow: each slot instead carries the noisy action x_t for that row, so the queries act as
        # per-row positional embeddings that x_t is added into.
        # dual: BOTH, sharing the backbone and the transformer trunk. Motivation (measured): the two
        # arms want different heads -- the right arm faces PILED bolts (high conditional ambiguity;
        # flow's mode-following wins, 2.32 vs L2 2.69) while the left faces SCATTERED singles
        # (near-deterministic; the direct-regression head wins, 1.97 vs flow 2.66 even at tau=0).
        # One model with both heads lets deployment pick per arm without running two networks.
        flowish = head_mode in ("flow", "dual")
        self.queries = nn.Parameter(torch.randn(1, horizon, d) * 0.02)
        self.act_proj = nn.Linear(REAL_DIM, d) if flowish else None
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)) if flowish else None
        if head_mode == "dual":
            self.queries_l2 = nn.Parameter(torch.randn(1, horizon, d) * 0.02)
            self.out_l2 = nn.Linear(d, REAL_DIM)

        def _trunk():
            layer = nn.TransformerEncoderLayer(
                d, nhead=8, dim_feedforward=4 * d, batch_first=True, norm_first=True, dropout=0.0
            )
            return nn.TransformerEncoder(layer, layers)

        self.head = _trunk()
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, REAL_DIM)
        # dual normally runs BOTH branches through one trunk, so they compete for the same 4 layers
        # while only the l2 branch is what deployment reads for precision. split_head gives the l2
        # branch its own trunk: same idea as the shared version, minus the contention.
        self.split_head = split_head and head_mode == "dual"
        if self.split_head:
            self.head_l2 = _trunk()
            self.norm_l2 = nn.LayerNorm(d)
        # World-model auxiliary (wm arm): predict the FROZEN-encoder pooled latent of frame t+K,
        # conditioned on the GT action chunk. Given obs AND actions the future is ~deterministic --
        # the task's multimodality (which bolt) lives in the actions, which are given -- so this
        # regression cannot mode-average, and the flow action path is untouched. Runs as a SECOND
        # pass through the SHARED trunk (2 query tokens + ctx), so the flow pass stays byte-identical
        # to the baseline and inference never sees the aux tokens.
        self.wm_head = wm_head
        if wm_head:
            self.wm_queries = nn.Parameter(torch.randn(1, 2, d) * 0.02)  # one per wrist
            self.wm_act = nn.Linear(horizon * REAL_DIM, d)
            self.wm_out = nn.Linear(d, d)

    def wm_forward(self, ctx, gt_actions):
        """(b,ctx,d),(b,H,14) -> (b,2,d) predicted pooled future latents (left, right)."""
        b = ctx.shape[0]
        q = self.wm_queries.expand(b, -1, -1) + self.wm_act(gt_actions.reshape(b, -1).to(ctx.dtype)).unsqueeze(1)
        h = self.head(torch.cat([q, ctx], dim=1))[:, :2]
        return self.wm_out(self.norm(h))

    def encode(self, img_l, img_r, state):
        """Vision + proprio context. Runs ONCE per inference, even for 10 denoising steps."""
        b = img_l.shape[0]
        # Both wrists through the backbone in ONE batch -- two separate calls would halve GPU
        # utilisation at this batch size for no benefit.
        if self.backbone_type == "siglip":
            tok = self.backbone(torch.cat([img_l, img_r], 0)).last_hidden_state
        elif self.backbone_type == "sigdino":
            x = torch.cat([img_l, img_r], 0)  # raw [0,1]
            xs = x * 2.0 - 1.0  # SigLIP convention
            m = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            s = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            ts = self.backbone(xs).last_hidden_state  # (2b, 256, 1152)
            td = self.dino_proj(self.backbone2.forward_features((x - m) / s)["x_norm_patchtokens"])  # (2b, 196, 1152)
            tok = torch.cat([ts, td], dim=1)  # (2b, 452, 1152)
        else:
            o = self.backbone.forward_features(torch.cat([img_l, img_r], 0))
            tok = o["x_norm_patchtokens"]
        tl, tr = tok[:b] + self.cam_emb[0], tok[b:] + self.cam_emb[1]
        if self.state_tokens > 0:
            # pi05 grid: digitize(state, linspace(-1,1,N+1)[:-1]) - 1  ==  floor((x+1)/2 * N), clamped
            idx = torch.clamp(
                torch.floor((state.float() + 1.0) / 2.0 * self.state_tokens).long(), 0, self.state_tokens - 1
            )
            st_tok = (self.state_vocab(idx) + self.state_dim_emb).to(tl.dtype)  # (B, 14, d)
            return torch.cat([st_tok, tl, tr], dim=1)
        return torch.cat([self.state_proj(state).unsqueeze(1), tl, tr], dim=1)

    def decode_l2(self, ctx):
        """dual: deterministic x0-regression branch through the SHARED trunk."""
        b = ctx.shape[0]
        trunk = self.head_l2 if self.split_head else self.head
        norm = self.norm_l2 if self.split_head else self.norm
        h = trunk(torch.cat([self.queries_l2.expand(b, -1, -1), ctx], dim=1))[:, : self.horizon]
        return self.out_l2(norm(h))

    def decode(self, ctx, x_t=None, t=None):
        b = ctx.shape[0]
        q = self.queries.expand(b, -1, -1)
        if x_t is not None:
            q = q + self.act_proj(x_t)
            d = q.shape[-1]
            # time_emb builds from a float32 t; the module runs in bf16, so cast before the MLP.
            q = q + self.time_mlp(time_emb(t, d).to(q.dtype)).unsqueeze(1)
        h = self.head(torch.cat([q, ctx], dim=1))[:, : self.horizon]
        return self.out(self.norm(h))

    def forward(self, img_l, img_r, state, x_t=None, t=None):
        return self.decode(self.encode(img_l, img_r, state), x_t, t)

    @torch.no_grad()
    def sample_actions(self, img_l, img_r, state, num_steps: int = 10, noise_scale: float = 1.0, branch: str = "flow"):
        """Euler integration of openpi's convention: x_t = t*noise + (1-t)*a, u_t = noise - a, so the
        field points from noise (t=1) toward the action (t=0) and we step with -dt."""
        if self.head_mode == "l2":
            return self.forward(img_l, img_r, state)
        ctx = self.encode(img_l, img_r, state)
        if self.head_mode == "dual" and branch == "l2":
            return self.decode_l2(ctx)
        b = ctx.shape[0]
        # noise_scale = sampling temperature. 1.0 draws from the full learned distribution (modes
        # preserved); tau<1 starts integration nearer the origin, contracting draws toward the
        # distribution centre; 0 is deterministic. Motivation (measured): the left arm's conditional
        # is NARROW (low kNN ambiguity) and there flow's draw scatter is the whole deficit vs L2
        # (bias ~0, all scatter, 1.97 vs 2.4-3.1 across conditions), while the right arm's is wide
        # and flow wins. Temperature is settable PER CALL, so deployment can run each arm at its
        # own tau.
        x = noise_scale * torch.randn(b, self.horizon, REAL_DIM, device=ctx.device, dtype=torch.float32)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((b,), 1.0 - i * dt, device=ctx.device, dtype=torch.float32)
            v = self.decode(ctx, x.to(ctx.dtype), t).float()
            x = x - dt * v
        return x


_AUG = {"tf": None}


def make_aug(mode):
    from torchvision.transforms import v2

    ops = []
    if mode in ("photo", "photo_geo"):
        ops.append(v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03))
    if mode == "photo_geo":
        # fill=0 matches the black padding bars resize_with_pad already puts on these frames.
        ops.append(v2.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=0))
    return v2.Compose(ops) if ops else None


def _rot_img(x, th):
    """Rotate a float CHW batch by per-sample angles th (rad), numerically matching
    torchvision rotate(+deg) -- the SAME convention the yaw stress probe used to pin the action
    transform sign (verified: M=[[c,-s],[s,c]] sampling grid, mean|diff| 4e-4 vs TVF at 30 deg)."""
    c, s = torch.cos(th), torch.sin(th)
    M = torch.zeros(len(th), 2, 3, device=x.device, dtype=x.dtype)
    M[:, 0, 0] = c
    M[:, 0, 1] = -s
    M[:, 1, 0] = s
    M[:, 1, 1] = c
    g = torch.nn.functional.affine_grid(M, x.shape, align_corners=False)
    return torch.nn.functional.grid_sample(x, g, align_corners=False, padding_mode="zeros")


def _rot_pairs(v, th, pairs):
    """Apply R(-th) to the given (i,j) column pairs of v (probe-pinned sign: image +th pairs with
    x' = c x + s y, y' = -s x + c y). v is (B, ..., 14); th is (B,)."""
    c, s = torch.cos(th), torch.sin(th)
    shape = (-1,) + (1,) * (v.dim() - 2)
    c, s = c.view(shape), s.view(shape)
    for i, j in pairs:
        vi, vj = v[..., i].clone(), v[..., j].clone()
        v[..., i] = c * vi + s * vj
        v[..., j] = -s * vi + c * vj
    return v


def build_batch(batch, dev, aug=None, no_state=False, img_norm="imagenet", rotaug_deg=0.0, grip_state_random=0.0):
    """openpi's stack yields uint8 HWC images; DINOv3 wants ImageNet-normalised CHW floats,
    SigLIP wants [-1, 1] (openpi model.py convention -- match it exactly for the pi05-encoder arm).

    rotaug (yaw-equivariant augmentation): per sample, per arm, rotate the wrist image by
    th ~ U(-deg, +deg) about its centre and transform that arm's targets consistently -- grasping is
    equivariant to rotation about the approach axis, and the UMI wrist camera's optical axis is
    ~aligned with it, so this synthesises the 45-90-deg-yaw bolt states missing from the demos
    (the measured rollout failure A; the yaw stress probe measured 2x xy degradation at 30-45 deg).
    Transform, per arm: pos-xy and rotvec-xy rows by R(-th); the remaining net yaw -th is spread
    uniformly over the chunk's rz; z/grip untouched; velocity proprio xy pairs rotated identically
    (image and proprio must tell the same story). Each sample stays ONE demo, just rotated -- no
    mode averaging, flow loss untouched."""
    B = batch["state"].shape[0]
    th_l = th_r = None
    if rotaug_deg > 0:
        amax = math.radians(rotaug_deg)
        th_l = (torch.rand(B, device=dev) * 2 - 1) * amax
        th_r = (torch.rand(B, device=dev) * 2 - 1) * amax

    def img(x, th=None):
        x = x.to(dev, non_blocking=True).permute(0, 3, 1, 2)
        if aug is not None:
            x = aug(x)  # on uint8, batched, GPU
        x = x.float() / 255.0
        if th is not None:
            x = _rot_img(x, th)
        if img_norm == "raw01":  # sigdino: per-branch norm happens inside encode()
            return x.to(torch.bfloat16)
        if img_norm == "pm1":
            return (x * 2.0 - 1.0).to(torch.bfloat16)
        m = torch.tensor(IMAGENET_MEAN, device=dev)[None, :, None, None]
        s = torch.tensor(IMAGENET_STD, device=dev)[None, :, None, None]
        return ((x - m) / s).to(torch.bfloat16)

    il = img(batch["image"]["left_wrist_0_rgb"], th_l)
    ir = img(batch["image"]["right_wrist_0_rgb"], th_r)
    st = batch["state"][:, :REAL_DIM].to(dev, non_blocking=True).float()
    ac = batch["actions"][:, :, :REAL_DIM].to(dev, non_blocking=True).float()
    if th_l is not None:
        H = ac.shape[1]
        # left arm: pos xy (0,1), rotvec xy (3,4); right arm: (7,8), (10,11)
        ac = _rot_pairs(ac, th_l, [(0, 1), (3, 4)])
        ac = _rot_pairs(ac, th_r, [(7, 8), (10, 11)])
        ac[:, :, 5] -= (th_l / H)[:, None]  # distribute the net extra yaw over the chunk
        ac[:, :, 12] -= (th_r / H)[:, None]
        st = _rot_pairs(st, th_l, [(0, 1), (3, 4)])
        st = _rot_pairs(st, th_r, [(7, 8), (10, 11)])
    if grip_state_random > 0:
        mask = torch.rand(B, device=dev) < grip_state_random
        if mask.any():
            st[mask.nonzero(as_tuple=True)[0][:, None], torch.tensor([6, 13], device=dev)] = (
                torch.rand(int(mask.sum()), 2, device=dev) * 2.0 - 1.0
            )
    st = st.to(torch.bfloat16)
    if no_state:
        st = torch.zeros_like(st)
    pad = batch.get("actions_is_pad")
    pad = pad.to(dev, non_blocking=True) if pad is not None else None
    return il, ir, st, ac, pad


def param_groups(model, base_lr, head_lr, decay=0.75):
    """Layer-wise decay: the pretrained stem should move least. 722 episodes is a small set for an
    85 M-parameter backbone, and the effective sample count is closer to the number of episodes than
    to the 178 k frames, because frames within an episode are near-duplicates.
    decay=1.0 turns this into a flat backbone lr -- use that for the SigLIP arm, whose 27 layers
    would otherwise get 0.75^27 ~ 4e-4x at the bottom (frozen in all but name), and whose reference
    treatment (pi05) trains the encoder at a single flat lr."""
    groups = []

    def bb(module, blocks, blk_prefix):
        n = len(blocks)
        stem = [p for nm, p in module.named_parameters() if not nm.startswith(blk_prefix)]
        groups.append({"params": stem, "lr": base_lr * decay ** (n + 1)})
        for i, blk in enumerate(blocks):
            groups.append({"params": list(blk.parameters()), "lr": base_lr * decay ** (n - i)})

    if model.backbone_type == "siglip":
        bb(model.backbone, model.backbone.vision_model.encoder.layers, "vision_model.encoder.layers.")
    elif model.backbone_type == "sigdino":
        bb(model.backbone, model.backbone.vision_model.encoder.layers, "vision_model.encoder.layers.")
        bb(model.backbone2, model.backbone2.blocks, "blocks.")
    else:
        bb(model.backbone, model.backbone.blocks, "blocks.")
    rest = [p for nm, p in model.named_parameters() if not (nm.startswith("backbone.") or nm.startswith("backbone2."))]
    groups.append({"params": rest, "lr": head_lr})
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="pi05_pika_umi_wrist_velgrip_k1_h8_40k",
        help="only its DataConfig is used: repo, norm stats, resize, horizon",
    )
    ap.add_argument("--weights-dir", default="/home/plaif/dinov3_weights")
    ap.add_argument("--size", default="b", choices=["s", "b", "l"])
    ap.add_argument(
        "--backbone",
        default="dinov3",
        choices=["dinov3", "siglip", "sigdino"],
        help="siglip = the C-experiment (true LLM-removal): pi05's OWN So400m/14 vision "
        "tower (weights from pi05_base via extract_siglip_pi05base.py), 256 tok/img "
        "-> 512 vision tokens into the action head. Same encoder, same init, same "
        "data/norm stats/EMA as pi05; the ONLY thing missing is the Gemma trunk. "
        "Pair with --lr 2.5e-5 --lwd 1.0 (pi05 trains its encoder flat at 2.5e-5) "
        "and --weight-decay 1e-10 (openpi AdamW default).",
    )
    ap.add_argument(
        "--lwd", type=float, default=0.75, help="layer-wise lr decay for the backbone; 1.0 = flat (use for siglip)"
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="AdamW weight decay. openpi/pi05 uses 1e-10; VA line has used 0.05",
    )
    ap.add_argument(
        "--layers",
        type=int,
        default=4,
        help="transformer layers in the action head. Never swept until now: at d=768 the "
        "4-layer default is ~28 M params against pi05's ~300 M action expert, and "
        "head capacity is a DIFFERENT axis from backbone capacity (ViT-L was "
        "rejected because perception is not the bottleneck -- action decoding may "
        "still be).",
    )
    ap.add_argument(
        "--split-head",
        action="store_true",
        help="dual only: give the l2 branch its own trunk instead of sharing one with "
        "flow. Deployment reads l2 for precision, so the contention is one-sided.",
    )
    ap.add_argument(
        "--row-decay",
        type=float,
        default=1.0,
        help="per-row loss weight gamma**row, renormalised to mean 1. gamma<1 puts the "
        "capacity on the rows that are actually executed and that the L=5 metric "
        "reads, while keeping H=24 so the long-horizon mode structure survives -- "
        "the benefit that made H=8 win for pi05, without losing multimodality.",
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=0,
        help="train the model on only the first N rows of the config's chunk. The DATASET "
        "is untouched (same repo, same norm stats, same 24-row chunks); the target is "
        "sliced, so this stays a single variable against an H=24 control. Rationale: "
        "at 3.4 ms the policy can replan every 33 ms control step, so rows 0-1 are the "
        "only ones that ever execute -- and L=5, the metric the whole campaign was "
        "judged on, is pi05's operating point (it must run 4-6 rows to cover 76 ms), "
        "not this policy's. H=8 was REJECTED on L=5 while it improved L=2 "
        "(1.19/0.79 vs h24 1.35/0.90), so that rejection was metric-dependent.",
    )
    ap.add_argument(
        "--exec-window",
        type=int,
        default=0,
        help="rows 0..K-1 -- the ones a K-step execute window actually runs -- get "
        "--exec-weight and the rest 1.0, renormalised to mean 1. Unlike --row-decay's "
        "smooth gamma (measured neutral: 2.40/2.04 vs 2.47/2.00), this is a hard step "
        "at the deployment execute window, and it keeps H=24 so the long-horizon mode "
        "structure that makes this task multimodal survives. Tests whether "
        "short-horizon precision requires GIVING UP multimodality (--horizon) or not.",
    )
    ap.add_argument("--exec-weight", type=float, default=10.0)
    ap.add_argument(
        "--distill-cache",
        type=str,
        default=None,
        help="directory of per-episode teacher chunks (cache_teacher_chunks.py). "
        "Distills pi05-H8 into the L2 BRANCH ONLY: measured 2026-08-06, the teacher "
        "wins by a better prior, not a smaller train/val gap (it memorises train "
        "HARDER, 0.61/0.71 vs 1.08/1.23, yet wins val 2.37/1.63 vs 2.47/2.00), so "
        "transfer the function instead of constraining the student. The flow branch "
        "keeps pure GT so mode structure is not narrowed by a short-horizon teacher. "
        "Cache is UNNORMALISED actions; normalised here with the same quantile "
        "formula as the targets.",
    )
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument(
        "--wm-cache",
        type=str,
        default=None,
        help="directory of per-frame pooled frozen-encoder latents "
        "(cache_future_latents.py). Enables the world-model auxiliary: predict the "
        "latent of frame t+K conditioned on the GT action chunk, cosine loss, via a "
        "second pass through the shared trunk. Flow action path untouched; given "
        "obs+actions the future is ~deterministic, so the aux cannot mode-average. "
        "Motivated by the measured pi05 gap being ONLY long-horizon (L12/23 x/y).",
    )
    ap.add_argument(
        "--wm-k",
        type=int,
        default=16,
        help="lookahead in frames (30 Hz; 16 = 0.53 s, the L12-23 territory where the " "gap lives)",
    )
    ap.add_argument("--wm-weight", type=float, default=0.25)
    ap.add_argument(
        "--state-tokens",
        type=int,
        default=0,
        help="N>0: replace the continuous state_proj with VLA-style DISCRETE state "
        "tokens -- quantize each normalized state dim into N bins over [-1,1] "
        "(pi05's np.digitize grid; use 256 to match) and embed per bin. Tests "
        "whether the VLA's echo resistance comes from the discretization "
        "bottleneck alone, with no dropout trick and no LLM. See VAPolicy.",
    )
    ap.add_argument(
        "--grip-state-random",
        type=float,
        default=0.0,
        help="per-sample prob of replacing the STATE grip dims (6/13) with uniform "
        "random values in [-1,1] (normalized space). Breaks the grip echo shortcut "
        "measured on-robot 2026-08-16: the grip target is ~99%% explained by "
        "'current state grip minus epsilon', so on OOD (robot) images the vision "
        "gate goes silent and the policy spirals closed (corr(cmd,meas)=0.993, "
        "-4.7%%/tick). Randomised state forces the grip decision onto VISION, which "
        "the forced-open probe showed is learnable. Deploy keeps honest state.",
    )
    ap.add_argument(
        "--rotaug",
        type=float,
        default=0.0,
        help="yaw-equivariant augmentation: per-arm image rotation up to DEG with the "
        "action chunk, rz yaw budget and velocity proprio transformed consistently. "
        "See build_batch docstring; sign pinned by the yaw stress probe (2026-08-15).",
    )
    ap.add_argument(
        "--drop-path",
        type=float,
        default=0.0,
        help="stochastic depth rate for the backbone. MEASURED train/val gap on the same "
        "grasp metric (gw ckpt, l2 branch): left_L5 1.23 train vs 2.00 val, and "
        "left_L23 2.9 vs ~14+ -- the long horizon is memorized outright. The "
        "backbone was fine-tuning with drop_path 0.0; this is the standard ViT "
        "regulariser for that failure. Parameter-free, so strict weight loading is "
        "unaffected.",
    )
    ap.add_argument(
        "--grasp-weight",
        type=float,
        default=1.0,
        help=">1 upweights frames within --grasp-window steps BEFORE either arm's grasp "
        "instant. Rationale: precision only matters at the grasp (allowance ~3 mm) "
        "but those frames are a small minority of the 178k training frames; the loss "
        "spends most of its gradient on transport/idle where cm-level error is fine. "
        "Grasp instants come from phase_segmentation over the gripper channels, the "
        "same detector the eval uses.",
    )
    ap.add_argument("--grasp-window", type=int, default=30)
    ap.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="input side length. openpi's ModelTransformFactory hardcodes "
        "ResizeImages(224,224); when this differs we surgically replace that "
        "transform in the loaded DataConfig so frames come straight from the "
        "480x640 source to NxN (RoPE handles any grid; 384 -> 576 tok/img). "
        "Resolution has been null THREE times (no-pad, SigLIP-224, DINOv3 "
        "480x640 ceiling probe) -- this arm exists to close the question for "
        "the fine-tuned-DINOv3 recipe specifically, not because it is favoured.",
    )
    ap.add_argument(
        "--no-state",
        action="store_true",
        help="zero the proprio input (architecture unchanged -> single variable). "
        "Rationale: proprio carries ZERO bolt-position information (measured: "
        "proprio-only regressor = blind floor, 31.5 vs 34.1 mm), pi05's low "
        "first-step error was proprio COPYING, and closed-loop the state feeds "
        "back the policy's own commands -- the copycat/causal-confusion failure "
        "mode (Fighting Copycat Agents, NeurIPS 2020). If accuracy holds without "
        "state, deployment drops the contraction-loop risk entirely.",
    )
    ap.add_argument(
        "--ema-decay",
        type=float,
        default=0.99,
        help="pi05 trains with ema_decay=0.99 and SERVES the EMA params "
        "(checkpoints.py:146). The first VA runs had no EMA at all -- the likely "
        "reason single-draw flow lost to L2 (3.15 vs 2.69 mm): a raw final iterate "
        "of a stochastic objective is noisier than its average. 0 disables.",
    )
    ap.add_argument(
        "--aug",
        default="none",
        choices=["none", "photo", "photo_geo"],
        help="photo = color jitter only (appearance robustness, geometry untouched). "
        "photo_geo adds a small random affine (translate 5%%, scale 0.95-1.05): this "
        "DOES perturb the image<->action geometric mapping, deliberately -- it is a "
        "cheap stand-in for the viewpoint shift the policy meets closed-loop when "
        "the arm strays off the demo trajectory (the teacher-forced/rollout gap; "
        "cf. EgoDemoGen, arXiv 2509.22578, which attacks the same axis with "
        "generated novel-view demos).",
    )
    ap.add_argument(
        "--head",
        default="flow",
        choices=["flow", "l2", "dual"],
        help="flow = openpi's exact rectified-flow objective (primary arm: then DINOv3-vs-"
        "PaliGemma is the single variable vs pi05). l2 = direct regression (secondary "
        "arm; also breaks medoid sampling, which needs multiple draws).",
    )
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--out", default="/home/plaif/va_runs/va_dinov3_b_h8")
    args = ap.parse_args()

    import openpi.training.config as _config
    import openpi.training.data_loader as dl

    cfg = _config.get_config(args.config)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    if args.resolution != 224:
        import dataclasses

        import openpi.transforms as _tf

        new_inputs = tuple(
            _tf.ResizeImages(args.resolution, args.resolution, pad=t.pad) if isinstance(t, _tf.ResizeImages) else t
            for t in dc.model_transforms.inputs
        )
        assert any(isinstance(t, _tf.ResizeImages) and t.height == args.resolution for t in new_inputs)
        dc = dataclasses.replace(dc, model_transforms=dataclasses.replace(dc.model_transforms, inputs=new_inputs))
        print(f"resolution override: {args.resolution}x{args.resolution}", flush=True)
    H = H_data = cfg.model.action_horizon
    if args.horizon:
        assert 0 < args.horizon <= H_data, f"--horizon {args.horizon} outside config chunk {H_data}"
        H = args.horizon
    # Build the dataset at the CONFIG horizon regardless, so the data pipeline is byte-identical to
    # the H=24 control; the target is sliced to H in the loop.
    ds = dl.transform_dataset(dl.create_torch_dataset(dc, H_data, cfg.model), dc)
    print(
        f"dataset {len(ds)} samples, horizon {H_data}"
        + (f" -> model horizon {H} (rows 0..{H - 1})" if H_data != H else ""),
        flush=True,
    )

    teacher = None
    if args.distill_cache:
        import glob as _glob

        ns = dc.norm_stats
        aq01 = np.asarray(ns["actions"].q01)[:REAL_DIM]
        aq99 = np.asarray(ns["actions"].q99)[:REAL_DIM]
        TH = None
        t_arr = t_msk = None
        for fp in sorted(_glob.glob(str(pathlib.Path(args.distill_cache) / "ep_*.npz"))):
            z = np.load(fp)
            ch, f0 = z["chunks"], int(z["frame0"])
            if TH is None:
                TH = ch.shape[1]
                t_arr = np.zeros((len(ds), TH, REAL_DIM), dtype=np.float32)
                t_msk = np.zeros(len(ds), dtype=bool)
            n = min(ch.shape[0], len(ds) - f0)
            t_arr[f0 : f0 + n] = ch[:n]
            t_msk[f0 : f0 + n] = True
        t_arr = (t_arr - aq01) / (aq99 - aq01 + 1e-6) * 2.0 - 1.0  # same normalisation as targets
        teacher = (torch.from_numpy(t_arr), torch.from_numpy(t_msk), TH)
        print(f"distill cache: {int(t_msk.sum())}/{len(ds)} frames, teacher H={TH}", flush=True)

    wm = None
    if args.wm_cache:
        # Per-frame pooled frozen-encoder latents (cache_future_latents.py). The +K shift and the
        # same-episode validity mask are computed HERE; the cache itself is shift-free.
        import glob as _glob

        la = msk = None
        ends = np.zeros(len(ds), dtype=np.int64)  # exclusive episode end per flat index
        for fp in sorted(_glob.glob(str(pathlib.Path(args.wm_cache) / "ep_*.npz"))):
            z = np.load(fp)
            lat, f0 = z["lat"], int(z["frame0"])  # (n, 2, D) fp16
            if la is None:
                la = np.zeros((len(ds), 2, lat.shape[2]), dtype=np.float16)
                msk = np.zeros(len(ds), dtype=bool)
            n = min(lat.shape[0], len(ds) - f0)
            la[f0 : f0 + n] = lat[:n]
            msk[f0 : f0 + n] = True
            ends[f0 : f0 + n] = f0 + n
        idx = np.arange(len(ds)) + args.wm_k
        valid = msk & (idx < ends)  # future frame exists in the SAME episode
        wm = (torch.from_numpy(la), torch.from_numpy(valid), idx)
        print(
            f"wm cache: {int(valid.sum())}/{len(ds)} frames usable at K={args.wm_k}, " f"latent dim {la.shape[2]}",
            flush=True,
        )

    if args.grasp_weight > 1.0:
        # Per-index weights: 1.0 everywhere, grasp_weight in the window before each arm's grasp.
        import importlib.util as _ilu
        import pathlib as _pl

        _sp = _ilu.spec_from_file_location(
            "ps", "/home/plaif/workspace/robotics_lab/policy_runner/policy_runner/phase_segmentation.py"
        )
        _ps = _ilu.module_from_spec(_sp)
        sys.modules["ps"] = _ps
        _sp.loader.exec_module(_ps)
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as _LRD

        _root = _pl.Path(os.environ["HF_LEROBOT_HOME"]) / dc.repo_id
        _lds = _LRD(dc.repo_id, root=_root, video_backend="pyav")
        _hf = _lds.hf_dataset.with_format("numpy")
        _S = np.stack(_hf["state"]).astype(np.float64)
        _f = list(_lds.episode_data_index["from"])
        _t = list(_lds.episode_data_index["to"])
        w = np.ones(len(ds), dtype=np.float32)
        n_ev = 0
        for _ei in range(len(_f)):
            _a, _b = int(_f[_ei]), int(_t[_ei])
            _st = _S[_a : _b - 1]
            _T = _st.shape[0]
            if _st.shape[1] <= 13 or _T < 40:
                continue
            _bo = _ps.extract_phase_boundaries(_st[:, 6] * 100.0, _st[:, 13] * 100.0, _T)
            if not _bo.clean:
                continue
            for _bk in ("b1", "b3"):
                _bf = int(getattr(_bo, _bk, -1))
                if 0 <= _bf < _T:
                    w[_a + max(0, _bf - args.grasp_window) : _a + _bf + 1] = args.grasp_weight
                    n_ev += 1
        print(
            f"grasp-weight {args.grasp_weight} on {int((w > 1).sum())} frames "
            f"({100 * (w > 1).mean():.1f}%), {n_ev} events",
            flush=True,
        )

        class _WeightedDS(torch.utils.data.Dataset):
            def __init__(self, base, weights):
                self.base, self.w = base, weights

            def __len__(self):
                return len(self.base)

            def __getitem__(self, i):
                it = dict(self.base[i])
                it["_w"] = np.float32(self.w[i])
                return it

        ds = _WeightedDS(ds, w)

    if teacher is not None:

        class _TeacherDS(torch.utils.data.Dataset):
            def __init__(self, base, t):
                self.base, (self.ta, self.tm, self.th) = base, t

            def __len__(self):
                return len(self.base)

            def __getitem__(self, i):
                it = dict(self.base[i])
                it["_teacher"] = self.ta[i].numpy()
                it["_tmask"] = np.float32(self.tm[i].item())
                return it

        ds = _TeacherDS(ds, teacher)

    if wm is not None:

        class _WMDS(torch.utils.data.Dataset):
            def __init__(self, base, w):
                self.base, (self.la, self.va, self.fi) = base, w

            def __len__(self):
                return len(self.base)

            def __getitem__(self, i):
                it = dict(self.base[i])
                if bool(self.va[i]):
                    it["_futlat"] = self.la[self.fi[i]].numpy().astype(np.float32)
                    it["_futmask"] = np.float32(1.0)
                else:
                    it["_futlat"] = np.zeros_like(self.la[0].numpy(), dtype=np.float32)
                    it["_futmask"] = np.float32(0.0)
                return it

        ds = _WMDS(ds, wm)

    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )

    dev = "cuda"
    if wm is not None:
        assert not args.horizon, "--wm-cache conditions on the full config chunk; drop --horizon"
        assert not args.rotaug, (
            "wm latent targets are cached from UNROTATED frames; combining "
            "with --rotaug would pair rotated ctx with unrotated futures"
        )
    model = (
        VAPolicy(
            args.size,
            pathlib.Path(args.weights_dir),
            H,
            layers=args.layers,
            head_mode=args.head,
            drop_path=args.drop_path,
            split_head=args.split_head,
            backbone_type=args.backbone,
            wm_head=wm is not None,
            state_tokens=args.state_tokens,
        )
        .to(dev)
        .to(torch.bfloat16)
    )
    if wm is not None:
        assert (
            wm[0].shape[2] == model.wm_out.out_features
        ), f"wm latent dim {wm[0].shape[2]} != head width {model.wm_out.out_features}"
    img_norm = {"siglip": "pm1", "sigdino": "raw01"}.get(args.backbone, "imagenet")
    row_w = None
    if args.exec_window:
        assert args.row_decay == 1.0, "--exec-window and --row-decay both shape row_w; pick one"
        rw = np.ones(H, dtype=np.float32)
        rw[: min(args.exec_window, H)] = args.exec_weight
        rw = rw / rw.mean()
        row_w = torch.from_numpy(rw).to(dev).view(1, H, 1)
        print(
            f"exec-window {args.exec_window} @ w={args.exec_weight}: row0 {rw[0]:.2f} .. "
            f"row{H - 1} {rw[-1]:.2f} (executed rows carry "
            f"{100 * rw[:args.exec_window].sum() / rw.sum():.0f}% of the loss mass)",
            flush=True,
        )
    if args.row_decay != 1.0:
        rw = args.row_decay ** np.arange(H, dtype=np.float32)
        rw = rw / rw.mean()  # mean 1 keeps the loss scale comparable to gamma=1
        row_w = torch.from_numpy(rw).to(dev).view(1, H, 1)
        print(f"row-decay {args.row_decay}: row0 {rw[0]:.2f} .. row{H - 1} {rw[-1]:.2f}", flush=True)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params {n_par:.1f} M", flush=True)
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.head_lr, decay=args.lwd), weight_decay=args.weight_decay)
    warm = max(1, int(0.02 * args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warm) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))
    )

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    aug = make_aug(args.aug)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()} if args.ema_decay > 0 else None
    step, t0, run = 0, time.perf_counter(), 0.0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            il, ir, st, ac, pad = build_batch(
                batch, dev, aug, args.no_state, img_norm, args.rotaug, args.grip_state_random
            )
            if ac.shape[1] > H:  # --horizon: keep only the rows that execute
                ac = ac[:, :H]
                pad = pad[:, :H] if pad is not None else None
            if args.head in ("flow", "dual"):
                # openpi's convention exactly (pi0.py compute_loss): time ~ Beta(1.5, 1) skews samples
                # toward t=1 (mostly noise), x_t interpolates noise->action, and the regression target
                # is the constant velocity u_t = noise - actions.
                noise = torch.randn_like(ac)
                t = torch.distributions.Beta(1.5, 1.0).sample((ac.shape[0],)).to(dev) * 0.999 + 0.001
                te = t[:, None, None]
                x_t = te * noise + (1 - te) * ac
                u_t = noise - ac
                if args.head == "dual":
                    ctx = model.encode(il, ir, st)
                    se = (model.decode(ctx, x_t.to(torch.bfloat16), t).float() - u_t) ** 2
                    l2_pred = model.decode_l2(ctx).float()
                    se = se + (l2_pred - ac) ** 2  # equal-weight sum
                    if args.distill_cache and "_teacher" in batch:
                        tch = batch["_teacher"].to(dev, non_blocking=True).float()
                        tmk = batch["_tmask"].to(dev, non_blocking=True).float().view(-1, 1, 1)
                        TH_ = tch.shape[1]
                        dse = ((l2_pred[:, :TH_] - tch) ** 2) * tmk
                        se[:, :TH_] = se[:, :TH_] + args.distill_weight * dse
                else:
                    ctx = model.encode(il, ir, st)  # exposed so the wm aux can share it
                    pred = model.decode(ctx, x_t.to(torch.bfloat16), t).float()
                    se = (pred - u_t) ** 2
            else:
                pred = model(il, ir, st).float()
                se = (pred - ac) ** 2
            wgt = batch.get("_w")
            if wgt is not None:
                wgt = wgt.to(dev, non_blocking=True).float().view(-1, 1, 1)
                se = se * wgt
                norm = wgt
            else:
                norm = torch.ones(se.shape[0], 1, 1, device=dev)
            if row_w is not None:
                se = se * row_w
                norm = norm * row_w  # weight the denominator too, so the loss scale is unchanged
            if pad is not None:
                # Episode-boundary padding repeats the last real action; averaging over it teaches the
                # policy to freeze at the end of every episode.
                keep = (~pad).float().unsqueeze(-1)
                loss = (se * keep).sum() / (keep * norm).sum().clamp(min=1) / REAL_DIM
            else:
                loss = se.sum() / (norm.sum().clamp(min=1) * se.shape[1] * se.shape[2])
            if wm is not None and "_futlat" in batch:
                fl = batch["_futlat"].to(dev, non_blocking=True).float()  # (b, 2, D)
                fm = batch["_futmask"].to(dev, non_blocking=True).float()  # (b,)
                wm_pred = model.wm_forward(ctx, ac).float()
                cosl = 1.0 - torch.nn.functional.cosine_similarity(wm_pred, fl, dim=-1)
                loss = loss + args.wm_weight * (cosl.mean(dim=1) * fm).sum() / fm.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema[k].mul_(args.ema_decay).add_(v.float(), alpha=1 - args.ema_decay)
            run += loss.item()
            step += 1
            if step % 100 == 0:
                dt = time.perf_counter() - t0
                print(
                    f"step {step:6d}/{args.steps}  loss {run / 100:.5f}  "
                    f"{dt / 100 * 1000:.0f} ms/it  lr {sched.get_last_lr()[-1]:.2e}",
                    flush=True,
                )
                run, t0 = 0.0, time.perf_counter()
            if step % args.save_every == 0 or step == args.steps:
                p = out / f"step_{step}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step,
                        "size": args.size,
                        "horizon": H,
                        "config": args.config,
                        "head": args.head,
                        "aug": args.aug,
                        # eval/serving should read EMA, matching what pi05 checkpoints contain
                        "no_state": args.no_state,
                        "resolution": args.resolution,
                        "grasp_weight": args.grasp_weight,
                        "drop_path": args.drop_path,
                        "distill": bool(args.distill_cache),
                        "layers": args.layers,
                        "split_head": args.split_head,
                        "row_decay": args.row_decay,
                        "exec_window": args.exec_window,
                        "exec_weight": args.exec_weight,
                        "backbone": args.backbone,
                        "weight_decay": args.weight_decay,
                        "lwd": args.lwd,
                        "wm": wm is not None,
                        "wm_k": args.wm_k,
                        "wm_weight": args.wm_weight,
                        "rotaug": args.rotaug,
                        "grip_state_random": args.grip_state_random,
                        "state_tokens": args.state_tokens,
                        "ema": ({k: v.to(torch.bfloat16) for k, v in ema.items()} if ema is not None else None),
                    },
                    p,
                )
                print(f"saved {p}", flush=True)
    (out / "done.json").write_text(json.dumps({"steps": step, "params_M": n_par}))
    print("done", flush=True)


if __name__ == "__main__":
    main()
