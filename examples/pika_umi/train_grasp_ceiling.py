#!/usr/bin/env python3
"""CEILING probe: DINOv3 regressor, wrist image at t -> displacement to the grasp frame.

Answers a prior question, not a modelling one. Six VLA trainings (proprio content x3, delta vs
anchored, LR/is_pad/loss-dim fixes, resize_pad A/B) all sit at right-arm z scatter 18.5-21.1 mm at
L23, and a PAIRED test killed the one apparent win. So: is that scatter reducible from these frames
AT ALL? This regressor gets direct supervision on exactly the measured quantity -- no action chunk,
no flow matching, no language, no proprio feedback loop. If it cannot beat the VLA, the information
is not in the images and further model work is misdirected.

Decision rule, fixed BEFORE any result was seen:
    val right_L23 z scatter  <= 12 mm  -> information is there, the VLA leaves it on the table
                             >= 17 mm  -> not recoverable from these frames; a data/sensor problem
                             12-17 mm  -> partial; check multimodality before concluding

DINOv3 weights are an INITIALISATION only -- the whole backbone trains. Note the constructor kwargs
below are not the library defaults: loading a released checkpoint into a default-built
DinoVisionTransformer silently drops 37 tensors (storage tokens, LayerScale, qkv bias_mask) under
strict=False and yields a differently-shaped network with plausible-looking outputs. These match
dinov3/hub/backbones.py, and the load is strict=True so a mismatch is an error, not a surprise.

NO GEOMETRIC AUGMENTATION. The target is a geometric quantity in the camera frame; a crop or flip
changes the correct answer. Photometric jitter only.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# From dinov3/hub/backbones.py. NOT the DinoVisionTransformer defaults -- see module docstring.
VIT_COMMON = dict(
    img_size=224, patch_size=16, in_chans=3, pos_embed_rope_base=100,
    pos_embed_rope_normalize_coords="separate", pos_embed_rope_rescale_coords=2,
    pos_embed_rope_dtype="fp32", ffn_ratio=4, qkv_bias=True, drop_path_rate=0.0,
    layerscale_init=1.0e-05, norm_layer="layernormbf16", ffn_layer="mlp",
    ffn_bias=True, proj_bias=True, n_storage_tokens=4, mask_k_bias=True,
)
VIT_SIZES = {
    "s": (dict(embed_dim=384, depth=12, num_heads=6), "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    "b": (dict(embed_dim=768, depth=12, num_heads=12), "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    "l": (dict(embed_dim=1024, depth=24, num_heads=16), "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
}


class GraspSet(Dataset):
    def __init__(self, root: pathlib.Path, split: str, hw, arm: str | None, jitter: bool, blind: bool):
        m = np.load(root / f"meta_{split}.npz", allow_pickle=True)
        keep = np.ones(len(m["name"]), bool)
        if arm is not None:
            keep &= m["arm"] == (1 if arm == "right" else 0)
        self.name = m["name"][keep]
        self.ep = m["ep"][keep]
        self.arm = m["arm"][keep]
        self.L = m["L"][keep]
        self.y = m["target_pos"][keep].astype(np.float32)     # metres, ee_local at t
        self.state = m["state"][keep].astype(np.float32)
        self.dir = root / "images" / split
        self.hw = hw
        self.blind = blind
        self.jitter = jitter
        if jitter:
            from torchvision.transforms import v2
            self.aug = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, i):
        if self.blind:
            x = torch.zeros(3, *self.hw)
        else:
            im = Image.open(self.dir / str(self.name[i])).convert("RGB")
            if im.size != (self.hw[1], self.hw[0]):
                im = im.resize((self.hw[1], self.hw[0]), Image.BILINEAR)
            if self.jitter:
                im = self.aug(im)
            x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
            x = (x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]
        return x, torch.from_numpy(self.y[i]), torch.from_numpy(self.state[i]), i


class Regressor(nn.Module):
    def __init__(self, size: str, weights_dir: pathlib.Path, use_state: bool, image: bool = True):
        super().__init__()
        self.image = image
        self.use_state = use_state
        d = 0
        if image:
            from dinov3.models.vision_transformer import DinoVisionTransformer
            dims, ckpt = VIT_SIZES[size]
            self.backbone = DinoVisionTransformer(**VIT_COMMON, **dims)
            sd = torch.load(weights_dir / ckpt, map_location="cpu")
            self.backbone.load_state_dict(sd.get("model", sd), strict=True)  # strict: see docstring
            d += dims["embed_dim"] * 2                                       # cls + mean patch token
        if use_state:
            d += 14
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 3))

    def forward(self, x, state):
        feats = []
        if self.image:
            o = self.backbone.forward_features(x)
            feats += [o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)]
        if self.use_state:
            feats.append(state)
        return self.head(torch.cat(feats, dim=-1))


def param_groups(model, base_lr, head_lr, decay=0.75):
    """Layer-wise LR decay: earlier blocks move less. Standard for fine-tuning a pretrained ViT on a
    small set, and this set is small -- 45k frames but only ~1.4k independent grasp events."""
    groups = []
    if model.image:
        blocks = model.backbone.blocks
        n = len(blocks)
        stem = [p for nm, p in model.backbone.named_parameters() if not nm.startswith("blocks.")]
        groups.append({"params": stem, "lr": base_lr * decay ** (n + 1)})
        for i, blk in enumerate(blocks):
            groups.append({"params": list(blk.parameters()), "lr": base_lr * decay ** (n - i)})
    groups.append({"params": list(model.head.parameters()), "lr": head_lr})
    return groups


@torch.no_grad()
def evaluate(model, loader, dev, ds):
    model.eval()
    P, Y, I = [], [], []
    for x, y, s, idx in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x.to(dev, non_blocking=True), s.to(dev, non_blocking=True))
        P.append(p.float().cpu()); Y.append(y); I.append(idx)
    P = torch.cat(P).numpy(); Y = torch.cat(Y).numpy(); I = torch.cat(I).numpy()
    e = (P - Y) * 1000.0                         # mm
    out = {"n_all": int(len(I)), "rmse_all_mm": float(np.sqrt((e ** 2).sum(1).mean()))}
    # Report at the SAME lookaheads the VLA metric uses so the cells line up one-for-one.
    for L in (12, 23):
        for a, an in ((1, "right"), (0, "left")):
            m = (ds.L[I] == L) & (ds.arm[I] == a)
            if m.sum() < 5:
                continue
            ee, yy = e[m], Y[m] * 1000.0
            out[f"{an}_L{L}"] = {
                "n": int(m.sum()),
                "bias_mm": {ax: float(ee[:, k].mean()) for k, ax in enumerate("xyz")},
                "scatter_mm": {ax: float(ee[:, k].std(ddof=1)) for k, ax in enumerate("xyz")},
                "rmse_mm": {ax: float(np.sqrt((ee[:, k] ** 2).mean())) for k, ax in enumerate("xyz")},
                "rmse_norm_mm": float(np.sqrt((ee ** 2).sum(1).mean())),
                # r2 against "predict the mean required displacement" -- the same reference the VLA
                # metric uses, so 0 means "ignores where the bolt is".
                "r2_vs_mean": {ax: float(1 - ee[:, k].var() / max(yy[:, k].var(), 1e-12))
                               for k, ax in enumerate("xyz")},
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/plaif/grasp_ceiling")
    ap.add_argument("--weights-dir", default="/home/plaif/dinov3_weights")
    ap.add_argument("--size", default="s", choices=["s", "b", "l"])
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--arm", default=None, choices=[None, "right", "left"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--use-state", action="store_true", help="concat 14-D proprio to the head input")
    ap.add_argument("--blind", action="store_true", help="CONTROL: zero images -> conditional-mean floor")
    ap.add_argument("--no-image", action="store_true", help="CONTROL: proprio only, no backbone")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    hw = (args.height, args.width)
    dev = "cuda"
    tr = GraspSet(root, "train", hw, args.arm, jitter=True, blind=args.blind)
    va = GraspSet(root, "val", hw, args.arm, jitter=False, blind=args.blind)
    print(f"train {len(tr)} rows / {len(set(tr.ep.tolist()))} episodes | "
          f"val {len(va)} rows / {len(set(va.ep.tolist()))} episodes", flush=True)

    ltr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                     pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    lva = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                     pin_memory=True, persistent_workers=args.workers > 0)

    model = Regressor(args.size, pathlib.Path(args.weights_dir),
                      use_state=args.use_state or args.no_image, image=not args.no_image).to(dev)
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.head_lr), weight_decay=0.05)
    steps = args.epochs * len(ltr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, int(0.05 * steps))) * 0.5 * (1 + math.cos(math.pi * s / steps)))

    best, best_ep, hist = None, -1, []
    for ep in range(args.epochs):
        model.train(); tot = n = 0.0
        for x, y, s, _ in ltr:
            x, y, s = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), s.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.mse_loss(model(x, s), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(x); n += len(x)
        m = evaluate(model, lva, dev, va)
        key = m.get("right_L23", {}).get("scatter_mm", {}).get("z", float("inf"))
        hist.append({"epoch": ep, "train_mse": tot / n, "val_rmse_all_mm": m["rmse_all_mm"],
                     "right_L23_z_scatter_mm": key})
        print(f"ep {ep:2d}  train_mse {tot / n:.5f}  val_rmse_all {m['rmse_all_mm']:6.2f} mm  "
              f"right_L23_z_scatter {key:6.2f} mm", flush=True)
        # Select on the pre-registered quantity, not on the training loss.
        if best is None or key < best.get("right_L23", {}).get("scatter_mm", {}).get("z", float("inf")):
            best, best_ep = m, ep

    out = {"tag": args.tag, "size": args.size, "hw": hw, "arm": args.arm,
           "blind": args.blind, "no_image": args.no_image, "use_state": args.use_state,
           "epochs": args.epochs, "best_epoch": best_ep, "history": hist, "best": best}
    p = pathlib.Path(args.root) / f"result_{args.tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps({"tag": args.tag, "best_epoch": best_ep,
                      "right_L23": best.get("right_L23"), "right_L12": best.get("right_L12")}, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
