"""Convert PiKA-UMI HDF5 episodes to a local LeRobot dataset.

This script intentionally builds only the session-holdout train episodes from the
in-house split manifest. It never pushes to the Hugging Face Hub.
"""

import json
import pathlib
import shutil

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from scipy.spatial.transform import Rotation
import tyro

REPO_ID = "plaif/pika_umi_openpi_train"
PROMPT = (
    "pick up the black bolt with the right arm and put it in the right box, then pick up the gray bolt with the "
    "left arm and put it in the left box"
)


def _decode_jpeg(encoded: np.ndarray) -> np.ndarray:
    bgr = cv2.imdecode(np.asarray(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _state(left_pose: np.ndarray, right_pose: np.ndarray, left_grip: np.ndarray, right_grip: np.ndarray) -> np.ndarray:
    # RESET-RELATIVE proprio (v3): each frame pose relative to the episode-first
    # frame, expressed in the reset body frame -- per arm pos_rel(3), rotvec_rel(3),
    # then gripper percent/100. Cancels the absolute capture-world frame that made
    # the v2 absolute-pose checkpoint fail on the robot (live stand-frame proprio
    # was ~2.7 m / z-score ~28 out of distribution). MUST match the inference
    # anchor in policy_runner OpenpiRemoteActionSource._proprio_state.
    def _rel(pose: np.ndarray):
        r0 = Rotation.from_quat(pose[0, 3:7])
        pos_rel = r0.inv().apply(pose[:, :3] - pose[0, :3])
        rot_rel = (r0.inv() * Rotation.from_quat(pose[:, 3:7])).as_rotvec()
        return pos_rel, rot_rel

    left_pos_rel, left_rotvec = _rel(left_pose)
    right_pos_rel, right_rotvec = _rel(right_pose)
    return np.concatenate(
        [
            left_pos_rel,
            left_rotvec,
            (left_grip[:, None] / 100.0),
            right_pos_rel,
            right_rotvec,
            (right_grip[:, None] / 100.0),
        ],
        axis=1,
    ).astype(np.float32)


def _arm_actions(pose: np.ndarray, grip: np.ndarray) -> np.ndarray:
    cur = Rotation.from_quat(pose[:-1, 3:7])
    nxt = Rotation.from_quat(pose[1:, 3:7])
    pos_delta_world = pose[1:, :3] - pose[:-1, :3]
    pos_delta_local = cur.inv().apply(pos_delta_world)
    rot_delta = (cur.inv() * nxt).as_rotvec()
    grip_delta = ((grip[1:] - grip[:-1]) / 100.0)[:, None]
    return np.concatenate([pos_delta_local, rot_delta, grip_delta], axis=1).astype(np.float32)


def _actions(left_pose: np.ndarray, right_pose: np.ndarray, left_grip: np.ndarray, right_grip: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            _arm_actions(left_pose, left_grip),
            _arm_actions(right_pose, right_grip),
        ],
        axis=1,
    ).astype(np.float32)


def _local_path(data_root: pathlib.Path, relative_path: str) -> pathlib.Path:
    path = data_root / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main(
    data_root: pathlib.Path = pathlib.Path("/home/plaif/workspace/robotics_lab/data_tcp"),
    split_manifest: pathlib.Path = pathlib.Path(
        "/home/plaif/workspace/robotics_lab/outputs/flow_runs/ee_local_seed5_lr3e4/split_manifest.json"
    ),
    repo_id: str = REPO_ID,
    summary_path: pathlib.Path = pathlib.Path("/home/plaif/workspace/openpi_runs/pika_umi_conversion_summary.json"),
):
    with split_manifest.open() as f:
        manifest = json.load(f)

    train_entries = manifest["session_holdout_train"]
    val_entries = manifest["session_holdout_val"]

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="pika_umi_dual_arm",
        fps=30,
        features={
            "left_wrist_0_rgb": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "right_wrist_0_rgb": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["actions"],
            },
        },
        image_writer_threads=16,
        image_writer_processes=4,
    )

    converted = []
    total_frames = 0
    total_action_frames = 0
    for entry in train_entries:
        episode_path = _local_path(data_root, entry["relative_path"])
        with h5py.File(episode_path, "r") as f:
            left_pose = np.asarray(f["observations/tcp_stand_left"], dtype=np.float64)
            right_pose = np.asarray(f["observations/tcp_stand_right"], dtype=np.float64)
            left_grip = np.asarray(f["observations/gripper_left"], dtype=np.float64)
            right_grip = np.asarray(f["observations/gripper_right"], dtype=np.float64)
            left_images = f["observations/images/left_realsense_color"]
            right_images = f["observations/images/right_realsense_color"]

            states = _state(left_pose, right_pose, left_grip, right_grip)
            actions = _actions(left_pose, right_pose, left_grip, right_grip)

            if actions.shape[0] != states.shape[0] - 1:
                raise ValueError(f"Action/state length mismatch in {episode_path}")

            for t in range(actions.shape[0]):
                dataset.add_frame(
                    {
                        "left_wrist_0_rgb": _decode_jpeg(left_images[t]),
                        "right_wrist_0_rgb": _decode_jpeg(right_images[t]),
                        "state": states[t],
                        "actions": actions[t],
                        "task": PROMPT,
                    }
                )
            dataset.save_episode()

        converted.append(
            {
                "relative_path": entry["relative_path"],
                "source_frame_count": int(entry["frame_count"]),
                "converted_frames": int(actions.shape[0]),
                "sha256": entry["sha256"],
            }
        )
        total_frames += int(entry["frame_count"])
        total_action_frames += int(actions.shape[0])
        print(f"converted {entry['relative_path']}: {actions.shape[0]} frames")

    summary = {
        "repo_id": repo_id,
        "output_path": str(output_path),
        "prompt": PROMPT,
        "fps": 30,
        "train_episode_count": len(train_entries),
        "val_episode_count": len(val_entries),
        "source_train_frames": total_frames,
        "converted_train_frames": total_action_frames,
        "converted_train_episodes": converted,
        "heldout_val_episodes": [entry["relative_path"] for entry in val_entries],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
