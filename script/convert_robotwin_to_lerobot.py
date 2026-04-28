"""
Convert RoboTwin HDF5 data to LeRobot dataset format with 30-dim action/state.

Action/State convention (Lingbot-VA, 30 dimensions):
    [0:7]   left arm EEF:  x, y, z, qx, qy, qz, qw   (SciPy quaternion order)
    [7:14]  right arm EEF: x, y, z, qx, qy, qz, qw
    [14:21] left arm joint angles (7 dims, padded with 0 if only 6 joints)
    [21:28] right arm joint angles (7 dims, padded with 0 if only 6 joints)
    [28]    left gripper
    [29]    right gripper

Note on quaternion conventions:
    - RoboTwin (transforms3d) stores quaternions as [w, x, y, z]
    - This script converts to SciPy convention [qx, qy, qz, qw]

RoboTwin HDF5 layout:
    /endpose/left_endpose     (T, 7)   [x, y, z, qw, qx, qy, qz]
    /endpose/left_gripper     (T,)     normalized gripper value
    /endpose/right_endpose    (T, 7)   [x, y, z, qw, qx, qy, qz]
    /endpose/right_gripper    (T,)     normalized gripper value
    /joint_action/left_arm    (T, 6)   joint angles
    /joint_action/left_gripper(T,)     gripper joint
    /joint_action/right_arm   (T, 6)   joint angles
    /joint_action/right_gripper(T,)    gripper joint
    /joint_action/vector      (T, 14)  [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
    /observation/{cam}/rgb    (T,)     JPEG compressed bytes
    /observation/{cam}/intrinsic_cv   (T, 3, 3)
    /observation/{cam}/extrinsic_cv   (T, 4, 4)

Usage:
    python script/convert_robotwin_to_lerobot.py \\
        --raw-dir ./data/beat_block_hammer/demo_clean \\
        --repo-id local/robotwin-beat-block-hammer \\
        --fps 30 \\
        --mode image

    # With task-specific instruction from instructions/*.json:
    python script/convert_robotwin_to_lerobot.py \\
        --raw-dir ./data/beat_block_hammer/demo_clean \\
        --repo-id local/robotwin-beat-block-hammer \\
        --fps 30 \\
        --task auto
"""

import dataclasses
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Literal

import cv2
import h5py
import numpy as np
import torch
import tqdm
import tyro

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except ModuleNotFoundError:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


# ==================== Constants ====================

# RoboTwin camera names → LeRobot observation key names
# Note: cv2.imdecode produces BGR images, consistent with the existing
# convert_aloha_data_to_lerobot_robotwin.py script.
CAMERA_MAP = {
    "head_camera": "cam_high",
    "front_camera": "cam_front",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}

# 30-dim action/state feature names
ACTION_DIM_NAMES = [
    # 0-6: left arm EEF
    "left_eef_x", "left_eef_y", "left_eef_z",
    "left_eef_qx", "left_eef_qy", "left_eef_qz", "left_eef_qw",
    # 7-13: right arm EEF
    "right_eef_x", "right_eef_y", "right_eef_z",
    "right_eef_qx", "right_eef_qy", "right_eef_qz", "right_eef_qw",
    # 14-20: left arm joints (7 dims, pad if needed)
    "left_joint_0", "left_joint_1", "left_joint_2",
    "left_joint_3", "left_joint_4", "left_joint_5", "left_joint_6",
    # 21-27: right arm joints (7 dims, pad if needed)
    "right_joint_0", "right_joint_1", "right_joint_2",
    "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6",
    # 28-29: grippers
    "left_gripper", "right_gripper",
]

ACTION_DIM = 30


# ==================== Config ====================

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


# ==================== Quaternion Conversion ====================

def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion from [w, x, y, z] (transforms3d) to [x, y, z, w] (SciPy).

    Args:
        quat_wxyz: array of shape (..., 4) with [w, x, y, z] order.

    Returns:
        array of shape (..., 4) with [x, y, z, w] order.
    """
    return np.concatenate([quat_wxyz[..., 1:], quat_wxyz[..., :1]], axis=-1)


def endpose_to_scipy(endpose: np.ndarray) -> np.ndarray:
    """Convert RoboTwin endpose [x,y,z, qw,qx,qy,qz] to [x,y,z, qx,qy,qz,qw].

    Args:
        endpose: array of shape (T, 7) with [x, y, z, qw, qx, qy, qz].

    Returns:
        array of shape (T, 7) with [x, y, z, qx, qy, qz, qw].
    """
    pos = endpose[:, :3]  # (T, 3)
    quat_wxyz = endpose[:, 3:]  # (T, 4) [w, x, y, z]
    quat_xyzw = quat_wxyz_to_xyzw(quat_wxyz)  # (T, 4) [x, y, z, w]
    return np.concatenate([pos, quat_xyzw], axis=-1)


# ==================== Data Loading ====================

def decode_jpeg_array(data: np.ndarray) -> np.ndarray:
    """Decode an array of JPEG byte strings to (T, H, W, 3) uint8 BGR images."""
    imgs = []
    for buf in data.ravel():
        if isinstance(buf, (bytes, bytearray)):
            arr = np.frombuffer(buf, dtype=np.uint8)
        elif isinstance(buf, np.ndarray) and buf.dtype == np.uint8:
            arr = buf
        else:
            raise TypeError(f"Unsupported buffer type: {type(buf)}")
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        imgs.append(img)
    return np.stack(imgs, axis=0)


def load_robotwin_episode(ep_path: str) -> dict:
    """Load a single RoboTwin HDF5 episode and convert to 30-dim action/state.

    Returns:
        dict with keys:
            "state":  (T, 30) float32 — current observation state
            "action": (T, 30) float32 — action (using next-frame state as action)
            "images": {cam_name: (T, H, W, 3) uint8}
    """
    with h5py.File(ep_path, "r") as f:
        T = None

        # --- Load endpose ---
        has_endpose = "endpose" in f and "left_endpose" in f["endpose"]
        if has_endpose:
            left_endpose_raw = np.array(f["endpose/left_endpose"])   # (T, 7) [x,y,z,qw,qx,qy,qz]
            right_endpose_raw = np.array(f["endpose/right_endpose"]) # (T, 7)
            left_gripper = np.array(f["endpose/left_gripper"])       # (T,)
            right_gripper = np.array(f["endpose/right_gripper"])     # (T,)

            T = left_endpose_raw.shape[0]

            # Convert quaternion: [w,x,y,z] → [x,y,z,w]
            left_endpose = endpose_to_scipy(left_endpose_raw)    # (T, 7) [x,y,z,qx,qy,qz,qw]
            right_endpose = endpose_to_scipy(right_endpose_raw)  # (T, 7)
        else:
            raise KeyError(f"Missing endpose data in {ep_path}. "
                           "Ensure data_type.endpose=true in task config.")

        # --- Load joint action ---
        has_joints = "joint_action" in f and "left_arm" in f["joint_action"]
        if has_joints:
            left_arm_joints = np.array(f["joint_action/left_arm"])     # (T, 6)
            right_arm_joints = np.array(f["joint_action/right_arm"])   # (T, 6)
        else:
            # Fallback: zeros
            left_arm_joints = np.zeros((T, 6), dtype=np.float64)
            right_arm_joints = np.zeros((T, 6), dtype=np.float64)

        # --- Build 30-dim state vector ---
        # Pad joints from 6 to 7 dims (append zero column)
        left_joints_padded = np.zeros((T, 7), dtype=np.float64)
        left_joints_padded[:, :left_arm_joints.shape[1]] = left_arm_joints

        right_joints_padded = np.zeros((T, 7), dtype=np.float64)
        right_joints_padded[:, :right_arm_joints.shape[1]] = right_arm_joints

        left_gripper_col = left_gripper.reshape(-1, 1)    # (T, 1)
        right_gripper_col = right_gripper.reshape(-1, 1)  # (T, 1)

        # state: [left_eef(7), right_eef(7), left_joints(7), right_joints(7), left_grip(1), right_grip(1)]
        state_30 = np.concatenate([
            left_endpose,         # 0:7
            right_endpose,        # 7:14
            left_joints_padded,   # 14:21
            right_joints_padded,  # 21:28
            left_gripper_col,     # 28
            right_gripper_col,    # 29
        ], axis=-1).astype(np.float32)  # (T, 30)

        assert state_30.shape == (T, ACTION_DIM), f"Expected (T, 30), got {state_30.shape}"

        # action: use next-frame state as the action target for the current frame.
        # For the last frame, repeat the last state.
        action_30 = np.copy(state_30)
        action_30[:-1] = state_30[1:]
        # Last frame action = same as last state (no change)

        # --- Load camera images ---
        images = {}
        for rtwin_cam, lerobot_cam in CAMERA_MAP.items():
            key = f"observation/{rtwin_cam}/rgb"
            if key in f:
                images[lerobot_cam] = decode_jpeg_array(np.array(f[key]))

    return {
        "state": state_30,
        "action": action_30,
        "images": images,
        "num_frames": T,
    }


# ==================== Dataset Creation ====================

def create_empty_dataset(
    repo_id: str,
    robot_type: str = "robotwin",
    fps: int = 30,
    mode: Literal["video", "image"] = "image",
    cameras: list[str] | None = None,
    image_height: int = 240,
    image_width: int = 320,
    root: Path | None = None,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """Create an empty LeRobotDataset with 30-dim action/state features."""

    if cameras is None:
        cameras = list(CAMERA_MAP.values())

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [ACTION_DIM_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [ACTION_DIM_NAMES],
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, image_height, image_width),
            "names": ["channels", "height", "width"],
        }

    output_root = root if root is not None else HF_LEROBOT_HOME
    if Path(output_root / repo_id).exists():
        shutil.rmtree(output_root / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=root,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_task_instruction(ep_path: str, ep_idx: int) -> str:
    """Try to load a task instruction from the instructions directory.

    Looks for:
        1. {data_dir}/../instructions/episode{ep_idx}.json  (per-episode)
        2. {data_dir}/../instructions.json                   (shared)

    Returns a randomly sampled instruction string, or empty string if not found.
    """
    data_dir = os.path.dirname(ep_path)
    base_dir = os.path.dirname(data_dir)

    # Per-episode instruction file
    per_ep_path = os.path.join(base_dir, "instructions", f"episode{ep_idx}.json")
    if os.path.exists(per_ep_path):
        with open(per_ep_path, "r") as f:
            instr_data = json.load(f)
        # Use "seen" instructions for training
        if "seen" in instr_data:
            return np.random.choice(instr_data["seen"])
        elif "instructions" in instr_data:
            return np.random.choice(instr_data["instructions"])

    # Shared instructions file
    shared_path = os.path.join(base_dir, "instructions.json")
    if os.path.exists(shared_path):
        with open(shared_path, "r") as f:
            instr_data = json.load(f)
        if "instructions" in instr_data:
            return np.random.choice(instr_data["instructions"])

    return ""


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[str],
    task: str = "auto",
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    """Populate the LeRobot dataset from RoboTwin HDF5 files."""

    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes, desc="Converting episodes"):
        ep_path = hdf5_files[ep_idx]
        ep_data = load_robotwin_episode(ep_path)

        # Determine task instruction
        if task == "auto":
            instruction = load_task_instruction(ep_path, ep_idx)
        else:
            instruction = task

        num_frames = ep_data["num_frames"]

        for i in range(num_frames):
            frame = {
                "observation.state": torch.from_numpy(ep_data["state"][i]),
                "action": torch.from_numpy(ep_data["action"][i]),
            }

            if instruction:
                frame["task"] = instruction

            for cam_name, img_array in ep_data["images"].items():
                frame[f"observation.images.{cam_name}"] = img_array[i]

            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


# ==================== Main ====================

@dataclasses.dataclass
class ConvertArgs:
    """Convert RoboTwin HDF5 data to LeRobot format (30-dim action)."""
    raw_dir: Path
    """Path to the RoboTwin data directory (e.g., ./data/beat_block_hammer/demo_clean)."""
    repo_id: str
    """LeRobot dataset repo ID (e.g., local/robotwin-task-name)."""
    fps: int = 30
    """Frames per second for the dataset."""
    task: str = "auto"
    """Task description. Use 'auto' to load from instructions/*.json."""
    robot_type: str = "robotwin"
    """Robot type identifier."""
    output_dir: Path | None = None
    """Output directory. Default: saves to ~/.cache/huggingface/lerobot/."""
    mode: Literal["video", "image"] = "image"
    """Storage mode for images."""
    push_to_hub: bool = False
    """Whether to push the dataset to Hugging Face Hub."""
    episodes: list[int] | None = None
    """Specific episode indices to convert. None = all."""


def main(args: ConvertArgs):
    raw_dir = args.raw_dir
    data_subdir = raw_dir / "data"

    if not data_subdir.exists():
        raise FileNotFoundError(
            f"Expected data subdirectory at {data_subdir}. "
            f"RoboTwin data should be at <raw_dir>/data/episode*.hdf5"
        )

    # Collect HDF5 files
    hdf5_files = sorted(
        [str(f) for f in data_subdir.glob("episode*.hdf5")],
        key=lambda x: int(os.path.basename(x).replace("episode", "").replace(".hdf5", "")),
    )

    if not hdf5_files:
        raise FileNotFoundError(f"No episode*.hdf5 files found in {data_subdir}")

    print(f"Found {len(hdf5_files)} episodes in {data_subdir}")
    print(f"  First: {os.path.basename(hdf5_files[0])}")
    print(f"  Last:  {os.path.basename(hdf5_files[-1])}")

    # Detect available cameras and image resolution from the first episode
    with h5py.File(hdf5_files[0], "r") as f:
        available_cameras = []
        image_height, image_width = 240, 320  # default
        for rtwin_cam, lerobot_cam in CAMERA_MAP.items():
            key = f"observation/{rtwin_cam}/rgb"
            if key in f:
                available_cameras.append(lerobot_cam)
                # Detect image resolution from first camera found
                if image_height == 240:
                    first_buf = f[key][0]
                    if isinstance(first_buf, (bytes, bytearray)):
                        arr = np.frombuffer(first_buf, dtype=np.uint8)
                    else:
                        arr = np.array(first_buf, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        image_height, image_width = img.shape[:2]
        print(f"  Cameras: {available_cameras}")
        print(f"  Image resolution: {image_width}x{image_height}")

        # Verify endpose exists
        if "endpose" not in f or "left_endpose" not in f["endpose"]:
            raise KeyError(
                "endpose data not found in HDF5. "
                "Make sure data_type.endpose=true in your task config."
            )

        # Print first frame info
        T = f["endpose/left_endpose"].shape[0]
        print(f"  Frames per episode (first): {T}")

    # Create dataset
    dataset = create_empty_dataset(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        mode=args.mode,
        cameras=available_cameras,
        image_height=image_height,
        image_width=image_width,
        root=args.output_dir,
    )

    # Populate
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=args.task,
        episodes=args.episodes,
    )

    dataset.finalize()

    output_root = args.output_dir if args.output_dir is not None else HF_LEROBOT_HOME
    print(f"\nDataset saved to: {output_root / args.repo_id}")
    print(f"  Total episodes: {len(hdf5_files) if args.episodes is None else len(args.episodes)}")
    print(f"  Action dim: {ACTION_DIM}")
    print(f"  FPS: {args.fps}")

    if args.push_to_hub:
        dataset.push_to_hub()
        print("Pushed to Hugging Face Hub.")


if __name__ == "__main__":
    args = tyro.cli(ConvertArgs)
    main(args)
