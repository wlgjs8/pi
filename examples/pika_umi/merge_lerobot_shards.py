"""Merge N LeRobot v2.1 shard datasets (produced by convert_pika_umi_storage_video.py --num-shards)
into ONE standard dataset, so a sharded parallel conversion yields the same single dataset a one-process
run would. Re-indexes episode_index + the global `index` sequentially, copies the per-episode parquet +
per-camera mp4s, and rebuilds meta (info.json totals, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl).

Assumes ≤1000 episodes total (single chunk-000) and identical features/fps/task across shards.

  python examples/pika_umi/merge_lerobot_shards.py \
    --lerobot-home /home/plaif/workspace/lerobot_home \
    --shard-glob 'plaif/pika_umi_video_train_dvg_s*of12' \
    --dest-repo-id plaif/pika_umi_video_train_tcp_gripabs_velproprio_depth
"""

import glob as _glob
import json
import pathlib
import shutil

import pyarrow.parquet as pq
import tyro


def _read_jsonl(p: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def main(
    lerobot_home: pathlib.Path,
    shard_glob: str,  # e.g. 'plaif/pika_umi_video_train_dvg_s*of12'  (relative to lerobot_home)
    dest_repo_id: str,
):
    import pyarrow as pa

    shard_dirs = sorted(pathlib.Path(lerobot_home).glob(shard_glob))
    if not shard_dirs:
        raise FileNotFoundError(f"no shards match {shard_glob} under {lerobot_home}")
    dest = pathlib.Path(lerobot_home) / dest_repo_id
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "data/chunk-000").mkdir(parents=True)
    (dest / "meta").mkdir(parents=True)

    # camera/video keys from the first shard's info.json
    info0 = json.loads((shard_dirs[0] / "meta/info.json").read_text())
    video_keys = [k for k, v in info0["features"].items() if v["dtype"] == "video"]
    for vk in video_keys:
        (dest / f"videos/chunk-000/{vk}").mkdir(parents=True)

    g_ep = 0
    g_frame = 0
    episodes_meta: list[dict] = []
    episodes_stats: list[dict] = []
    n_shard_eps = []
    for sd in shard_dirs:
        eps = _read_jsonl(sd / "meta/episodes.jsonl")
        stats = {s["episode_index"]: s for s in _read_jsonl(sd / "meta/episodes_stats.jsonl")}
        n_shard_eps.append(len(eps))
        for e in eps:
            src_idx = e["episode_index"]
            # --- parquet: rewrite episode_index + global index ---
            t = pq.read_table(sd / f"data/chunk-000/episode_{src_idx:06d}.parquet")
            n = t.num_rows
            cols = {name: t.column(name) for name in t.column_names}
            cols["episode_index"] = pa.array([g_ep] * n, type=t.schema.field("episode_index").type)
            cols["index"] = pa.array(list(range(g_frame, g_frame + n)), type=t.schema.field("index").type)
            pq.write_table(pa.table(cols), dest / f"data/chunk-000/episode_{g_ep:06d}.parquet")
            # --- videos: copy each camera mp4 renamed ---
            for vk in video_keys:
                shutil.copy(
                    sd / f"videos/chunk-000/{vk}/episode_{src_idx:06d}.mp4",
                    dest / f"videos/chunk-000/{vk}/episode_{g_ep:06d}.mp4",
                )
            # --- meta lines ---
            em = dict(e)
            em["episode_index"] = g_ep
            episodes_meta.append(em)
            sm = dict(stats[src_idx])
            sm["episode_index"] = g_ep
            episodes_stats.append(sm)
            g_ep += 1
            g_frame += n

    # --- write merged meta ---
    info = dict(info0)
    info["total_episodes"] = g_ep
    info["total_frames"] = g_frame
    info["total_videos"] = g_ep * len(video_keys)
    info["total_chunks"] = (g_ep + info0.get("chunks_size", 1000) - 1) // info0.get("chunks_size", 1000)
    info["splits"] = {"train": f"0:{g_ep}"}
    (dest / "meta/info.json").write_text(json.dumps(info, indent=4))
    (dest / "meta/episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes_meta) + "\n")
    (dest / "meta/episodes_stats.jsonl").write_text("\n".join(json.dumps(e) for e in episodes_stats) + "\n")
    shutil.copy(shard_dirs[0] / "meta/tasks.jsonl", dest / "meta/tasks.jsonl")

    print(f"merged {len(shard_dirs)} shards ({n_shard_eps}) -> {dest}")
    print(f"  total_episodes={g_ep}  total_frames={g_frame}  video_keys={video_keys}")


if __name__ == "__main__":
    tyro.cli(main)
