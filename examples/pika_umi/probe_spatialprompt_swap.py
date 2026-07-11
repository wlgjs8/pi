"""C1 probe: does the spatialprompt+obs-dropout policy FOLLOW the spatial words?

Two branches per checkpoint (llm-wiki vla-rollout-diagnosis 2026-07-11, C1 eval plan):
  (a) dropped-obs — zero every camera image AND set image masks False (exactly the trained
      train_obs_dropout branch; PikaUmiInputs hardcodes masks True, so we monkeypatch).
      CAPACITY readout: with no scene, the prompt is the only target info — swapped spatial
      words MUST move the endpoint. Failure here = language-pathway bottleneck.
  (b) full-obs — real images. TRANSFER readout: does the prompt still modulate when the
      geometry shortcut is available?

Measurement (threshold-free, direction-signed): at t=0, N draws under the episode's CORRECT
spatial prompt and N under the descriptor-SWAPPED prompt (near<->far, left<->right for the
bolt words ONLY — arm names untouched). Integrate chunks per arm, mean endpoint per prompt.
  side_follow  = P(sign(x_swap - x_correct) matches the side-word swap direction)   [random=0.5]
  depth_follow = P(sign(|E|_swap - |E|_correct) matches the depth-word swap)        [random=0.5]
  |dE| = mean endpoint displacement induced by the swap (A1 noise floor ~1 mm).
Only episodes with a spatial label (val task_index != 0) are used.

Run (low-mem, alongside training):
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_ALLOCATOR=platform .venv/bin/python \
    examples/pika_umi/probe_spatialprompt_swap.py --ckpt-base <...> --ckpt-step 10000
"""

import argparse
import json
import pathlib
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

LEROBOT_LOCAL = pathlib.Path("/home/plaif/workspace/lerobot_local")
VAL_REPO = "plaif/pika_umi_video_val_tcp_gripabs_velproprio_depth_z50_spatialprompt"
LABELS = pathlib.Path("/home/plaif/workspace/openpi_runs/analysis/spatialprompt_labels.json")
OUT_DIR = pathlib.Path("/home/plaif/workspace/openpi_runs/analysis")

PROMPT_RE = re.compile(
    r"pick up the (near|far) (left|right) bolt with the right arm, "
    r"and the (near|far) (left|right) bolt with the left arm"
)
INV = {"near": "far", "far": "near", "left": "right", "right": "left"}

# module-level toggle read by the monkeypatched PikaUmiInputs (branch (a) vs (b))
DROP_OBS = False


def swap_prompt(p):
    m = PROMPT_RE.fullmatch(p)
    assert m, f"unexpected prompt: {p}"
    d1, s1, d2, s2 = m.groups()
    return (f"pick up the {INV[d1]} {INV[s1]} bolt with the right arm, "
            f"and the {INV[d2]} {INV[s2]} bolt with the left arm"), (d1, s1, d2, s2)


def _integrate_local(deltas):
    cp = np.zeros(3)
    cr = Rotation.identity()
    for k in range(len(deltas)):
        cp = cp + cr.apply(deltas[k, :3])
        cr = cr * Rotation.from_rotvec(deltas[k, 3:6])
    return cp


def _video_frame0(path):
    cap = cv2.VideoCapture(str(path))
    ok, f = cap.read()
    cap.release()
    assert ok, path
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-base", required=True)
    ap.add_argument("--ckpt-step", type=int, required=True)
    ap.add_argument("--config", default="pi05_pika_umi_video_tcp_gripabs_velproprio_depth_z50_spatialprompt_h24")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import openpi.policies.pika_umi_policy as pika_umi_policy
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    # ---- monkeypatch: branch (a) zeroes images + masks False AFTER the normal transform ----
    _orig_call = pika_umi_policy.PikaUmiInputs.__call__

    def _patched_call(self, data):
        out = _orig_call(self, data)
        if DROP_OBS:
            out["image"] = {k: np.zeros_like(v) for k, v in out["image"].items()}
            out["image_mask"] = {k: np.False_ for k in out["image_mask"]}
        return out

    pika_umi_policy.PikaUmiInputs.__call__ = _patched_call

    root = LEROBOT_LOCAL / VAL_REPO
    ds = LeRobotDataset(VAL_REPO, root=root, video_backend="pyav")
    hf = ds.hf_dataset.with_format("numpy")
    states_all = np.stack(hf["state"]).astype(np.float64)
    task_idx = np.asarray(hf["task_index"])
    tmap = {int(k): v for k, v in ds.meta.tasks.items()}
    ep_from = list(ds.episode_data_index["from"])
    n_ep = len(ep_from)
    has_depth = (root / "videos/chunk-000/left_wrist_0_depth").exists()

    cfg = _config.get_config(args.config)
    ckpt_dir = pathlib.Path(args.ckpt_base) / str(args.ckpt_step)
    policy = _policy_config.create_trained_policy(cfg, str(ckpt_dir))
    ARMS = [("left", 0), ("right", 7)]

    global DROP_OBS
    rows = []
    n_done = 0
    for ei in range(n_ep):
        a = int(ep_from[ei])
        ti = int(task_idx[a])
        if ti == 0:
            continue  # unlabeled episode (original prompt) — no spatial ground truth
        prompt = tmap[ti]
        swapped, (d1, s1, d2, s2) = swap_prompt(prompt)
        words = {"right": (d1, s1), "left": (d2, s2)}

        obs_base = {
            "observation/left_wrist_0_rgb": _video_frame0(root / "videos/chunk-000/left_wrist_0_rgb" / f"episode_{ei:06d}.mp4"),
            "observation/right_wrist_0_rgb": _video_frame0(root / "videos/chunk-000/right_wrist_0_rgb" / f"episode_{ei:06d}.mp4"),
            "observation/state": states_all[a].astype(np.float32),
        }
        if has_depth:
            obs_base["observation/left_wrist_0_depth"] = _video_frame0(root / "videos/chunk-000/left_wrist_0_depth" / f"episode_{ei:06d}.mp4")
            obs_base["observation/right_wrist_0_depth"] = _video_frame0(root / "videos/chunk-000/right_wrist_0_depth" / f"episode_{ei:06d}.mp4")

        for branch in ("dropped", "full"):
            DROP_OBS = branch == "dropped"
            ends = {}
            for tag, p in [("correct", prompt), ("swap", swapped)]:
                obs = dict(obs_base)
                obs["prompt"] = p
                # N draws; each infer returns the full dual-arm chunk -> per-arm endpoints
                chunks = [np.asarray(policy.infer(obs)["actions"], dtype=np.float64)[:24] for _ in range(args.draws)]
                E = {}
                for arm, b in ARMS:
                    pts = np.stack([_integrate_local(ch[:, b:b + 6]) for ch in chunks])
                    E[arm] = pts.mean(0)
                ends[tag] = E
            for arm, _ in ARMS:
                Ec, Es = ends["correct"][arm], ends["swap"][arm]
                dword, sword = words[arm]
                # side word swapped: left->right means x should INCREASE (right = x >= median)
                exp_dx = +1.0 if sword == "left" else -1.0
                # depth word swapped: near->far means |E| should INCREASE
                exp_dn = +1.0 if dword == "near" else -1.0
                rows.append({
                    "ep": ei, "arm": arm, "branch": branch,
                    "side_ok": bool(np.sign(Es[0] - Ec[0]) == exp_dx),
                    "depth_ok": bool(np.sign(np.linalg.norm(Es) - np.linalg.norm(Ec)) == exp_dn),
                    "dE_m": float(np.linalg.norm(Es - Ec)),
                    "Ec_norm_m": float(np.linalg.norm(Ec)),
                })
        n_done += 1
        if args.limit and n_done >= args.limit:
            break
        if n_done % 10 == 0:
            print(f"episode {n_done} done", flush=True)

    tag = args.tag or f"step{args.ckpt_step}"
    out = OUT_DIR / f"spatialprompt_swap_{tag}.json"
    json.dump({"args": vars(args), "rows": rows}, open(out, "w"))

    for branch in ("dropped", "full"):
        sub = [r for r in rows if r["branch"] == branch]
        if not sub:
            continue
        side = np.mean([r["side_ok"] for r in sub])
        depth = np.mean([r["depth_ok"] for r in sub])
        dE = np.median([r["dE_m"] for r in sub])
        print(f"RESULT step={args.ckpt_step} branch={branch} n={len(sub)} "
              f"side_follow={side:.2f} depth_follow={depth:.2f} median|dE|={dE * 1000:.1f}mm "
              f"(random=0.50, A1 noise~1mm)", flush=True)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
