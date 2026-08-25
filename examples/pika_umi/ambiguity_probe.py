#!/usr/bin/env python3
"""Is the grasp target DETERMINED by the wrist image, or is the scene ambiguous?

Motivation: inspecting the frames killed the obvious hypothesis. The bolts are large and sharply
resolved -- this was never a resolution problem (three separate resize experiments agreed). What the
frames actually show is a variable number of bolts, from one to about ten. If the demonstrator's
choice among them is not determined by the image, then no encoder, no resolution and no amount of
data fixes it, because two near-identical scenes carry different correct answers.

Two non-parametric measurements, neither of which can overfit:

  1. kNN regression. Frozen DINOv3 features, val frame -> nearest TRAIN frames -> use their targets.
     No trained head, no capacity to memorise. If this reaches the fine-tuned models' ~17 mm, the
     information is present and accessible; if it sits at the blind floor (~34 mm), it is not.

  2. Neighbour target spread. Among the k nearest TRAIN frames of each val frame -- i.e. frames the
     encoder calls near-identical -- how much do the targets themselves disagree? That disagreement
     is an irreducible noise floor: a perfect predictor still cannot beat it, because the same
     picture legitimately maps to several answers.

Comparing (1) against (2) separates "model is weak" from "task is ambiguous".
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
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
}


class Frames(Dataset):
    def __init__(self, root, split, hw, arm):
        m = np.load(root / f"meta_{split}.npz", allow_pickle=True)
        keep = m["arm"] == (1 if arm == "right" else 0)
        self.name = m["name"][keep]
        self.ep = m["ep"][keep]
        self.L = m["L"][keep]
        self.y = m["target_pos"][keep].astype(np.float32)
        self.dir = root / "images" / split
        self.hw = hw

    def __len__(self):
        return len(self.name)

    def __getitem__(self, i):
        im = Image.open(self.dir / str(self.name[i])).convert("RGB").resize((self.hw[1], self.hw[0]), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
        x = (x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]
        return x, i


@torch.no_grad()
def features(ds, model, dev, bs, workers):
    out = []
    for x, _ in DataLoader(ds, batch_size=bs, num_workers=workers, pin_memory=True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model.forward_features(x.to(dev, non_blocking=True))
        f = torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], -1).float()
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu())
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/plaif/grasp_ceiling")
    ap.add_argument("--weights-dir", default="/home/plaif/dinov3_weights")
    ap.add_argument("--size", default="b")
    ap.add_argument("--arm", default="right")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="/home/plaif/grasp_ceiling/ambiguity.json")
    args = ap.parse_args()

    from dinov3.models.vision_transformer import DinoVisionTransformer

    root = pathlib.Path(args.root)
    dims, ckpt = VIT_SIZES[args.size]
    model = DinoVisionTransformer(**VIT_COMMON, **dims)
    sd = torch.load(pathlib.Path(args.weights_dir) / ckpt, map_location="cpu")
    model.load_state_dict(sd.get("model", sd), strict=True)
    model.eval().cuda()

    tr = Frames(root, "train", (224, 224), args.arm)
    va = Frames(root, "val", (224, 224), args.arm)
    Ftr = features(tr, model, "cuda", args.batch, args.workers)
    Fva = features(va, model, "cuda", args.batch, args.workers)
    print(f"features: train {tuple(Ftr.shape)}  val {tuple(Fva.shape)}", flush=True)

    Ytr = torch.from_numpy(tr.y) * 1000.0  # mm
    Yva = torch.from_numpy(va.y) * 1000.0
    res = {"arm": args.arm, "size": args.size, "n_train": int(len(tr)), "n_val": int(len(va))}

    # Restrict to the lookaheads the VLA metric reports, so the numbers are directly comparable.
    for L in (23, 12, 5, 2):
        vm = torch.from_numpy((va.L == L).astype(bool))
        if vm.sum() < 5:
            continue
        # Neighbours are drawn from the SAME lookahead: a frame 23 steps out should be compared with
        # frames 23 steps out, otherwise "nearest" trivially means "same distance-to-grasp".
        tm = torch.from_numpy((tr.L == L).astype(bool))
        S = Fva[vm] @ Ftr[tm].T  # cosine similarity, both L2-normalised
        yq, yb = Yva[vm], Ytr[tm]
        entry = {"n_val": int(vm.sum()), "n_train": int(tm.sum())}
        for k in (1, 5, 20):
            idx = S.topk(k, dim=1).indices
            nb = yb[idx]  # (nval, k, 3)
            pred = nb.mean(1)
            e = (pred - yq).numpy()
            entry[f"knn{k}"] = {
                "rmse_norm_mm": float(np.sqrt((e**2).sum(1).mean())),
                "z_scatter_mm": float(e[:, 2].std(ddof=1)),
                "z_bias_mm": float(e[:, 2].mean()),
                "z_rmse_mm": float(np.sqrt((e[:, 2] ** 2).mean())),
            }
            if k > 1:
                # Disagreement AMONG the neighbours = irreducible floor for any predictor that only
                # sees the image, since these frames are what the encoder considers the same scene.
                spread = nb.std(dim=1).numpy()
                entry[f"neighbour_target_spread_mm_k{k}"] = {
                    ax: float(np.median(spread[:, j])) for j, ax in enumerate("xyz")
                }
                entry[f"neighbour_cosine_k{k}"] = float(S.topk(k, dim=1).values[:, -1].mean())
        res[f"L{L}"] = entry
        print(f"L={L}: " + json.dumps(entry), flush=True)

    pathlib.Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
