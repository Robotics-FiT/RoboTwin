"""Extract joint_action/vector and per-camera RGB streams from a RoboTwin
episode hdf5 (as produced by `script/collect_data.py`).

Typical layout of an input hdf5::

    /joint_action/vector           (T, 14)  float64
    /observation/head_camera/rgb   (T,)     |SN   (JPEG bytes)
    /observation/front_camera/rgb  (T,)     |SN
    /observation/left_camera/rgb   (T,)     |SN
    /observation/right_camera/rgb  (T,)     |SN

Usage examples::

    # Quick summary only
    python script/extract_hdf5.py data/random_dance/random_dance/data/episode0.hdf5

    # Dump qpos + split the 4 cameras into 4 mp4 files + per-frame jpgs
    python script/extract_hdf5.py \
        data/random_dance/random_dance/data/episode0.hdf5 \
        --out-dir ./extracted/ep0 \
        --save-qpos --save-videos --save-frames --fps 30
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import h5py
import numpy as np

from envs.utils.parse_hdf5 import parse_img_array
from envs.utils.images_to_video import images_to_video


CAMERA_NAMES = ("head_camera", "front_camera", "left_camera", "right_camera")


def load_joint_action_vector(f: h5py.File) -> np.ndarray:
    """Return /joint_action/vector as a (T, 14) float64 array."""
    key = "joint_action/vector"
    if key not in f:
        raise KeyError(f"Missing dataset '{key}' in {f.filename}")
    return np.asarray(f[key][()])


def load_camera_rgb(f: h5py.File, camera: str) -> np.ndarray:
    """Return decoded RGB frames for one camera as (T, H, W, 3) uint8 BGR.

    RoboTwin stores each frame as a JPEG byte string; ``parse_img_array``
    decodes them via ``cv2.imdecode`` which yields BGR images.
    """
    key = f"observation/{camera}/rgb"
    if key not in f:
        raise KeyError(f"Missing dataset '{key}' in {f.filename}")
    return parse_img_array(f[key][()])


def save_qpos(vector: np.ndarray, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, f"{stem}_joint_action_vector.npy")
    csv_path = os.path.join(out_dir, f"{stem}_joint_action_vector.csv")
    np.save(npy_path, vector)
    # 14 dims: [left_arm(6) | left_gripper(1) | right_arm(6) | right_gripper(1)]
    header = (
        "left_arm_j1,left_arm_j2,left_arm_j3,left_arm_j4,left_arm_j5,left_arm_j6,"
        "left_gripper,"
        "right_arm_j1,right_arm_j2,right_arm_j3,right_arm_j4,right_arm_j5,right_arm_j6,"
        "right_gripper"
    )
    np.savetxt(csv_path, vector, delimiter=",", header=header, comments="")
    print(f"✅ joint_action/vector saved: {npy_path}  ({vector.shape})")
    print(f"✅ joint_action/vector saved: {csv_path}")


def save_camera_video(frames_bgr: np.ndarray, out_path: str, fps: float) -> None:
    # frames from cv2.imdecode are BGR, so tell ffmpeg is_rgb=False.
    images_to_video(frames_bgr, out_path=out_path, fps=fps, is_rgb=False)


def save_camera_frames(frames_bgr: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n = frames_bgr.shape[0]
    width = max(4, len(str(n - 1)))
    for i in range(n):
        cv2.imwrite(
            os.path.join(out_dir, f"frame_{str(i).zfill(width)}.jpg"),
            frames_bgr[i],
        )
    print(f"✅ {n} JPG frames saved under {out_dir}/")


def print_summary(vector: np.ndarray, cameras: dict) -> None:
    T = vector.shape[0]
    print("============= HDF5 Summary =============")
    print(f"num_frames (T)                : {T}")
    print(f"joint_action/vector shape     : {vector.shape}  dtype={vector.dtype}")
    print("joint_action/vector stats per-dim (min / max / mean):")
    for i in range(vector.shape[1]):
        col = vector[:, i]
        print(f"  dim[{i:02d}] : {col.min(): .4f} / {col.max(): .4f} / {col.mean(): .4f}")
    print("---------------- Cameras ----------------")
    for name, frames in cameras.items():
        if frames is None:
            print(f"  {name:<14s}: <missing>")
        else:
            print(
                f"  {name:<14s}: frames={frames.shape[0]:4d}  "
                f"size={frames.shape[2]}x{frames.shape[1]}  dtype={frames.dtype}  (BGR)"
            )
    print("========================================")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hdf5_path", type=str, help="Path to episode{N}.hdf5")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write extracted artefacts. Defaults to "
        "<hdf5_dir>/../extracted/<episode_stem>.",
    )
    parser.add_argument("--save-qpos", action="store_true", help="Dump joint_action/vector to .npy and .csv")
    parser.add_argument("--save-videos", action="store_true", help="Dump one mp4 per camera")
    parser.add_argument("--save-frames", action="store_true", help="Dump per-frame JPGs per camera")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS used when --save-videos is set")
    parser.add_argument(
        "--cameras",
        type=str,
        default=",".join(CAMERA_NAMES),
        help="Comma-separated camera names to split out. Default: all four.",
    )
    parser.add_argument(
        "--with-video-frame",
        action="store_true",
        help="Also extract /video_frame (the stream that was used to render "
        "the original video/episode{N}.mp4). Its source depends on "
        "task_config.camera.third_person_view (default/observer/random).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.hdf5_path):
        raise FileNotFoundError(args.hdf5_path)

    stem = os.path.splitext(os.path.basename(args.hdf5_path))[0]
    if args.out_dir is None:
        hdf5_dir = os.path.dirname(os.path.abspath(args.hdf5_path))
        args.out_dir = os.path.join(os.path.dirname(hdf5_dir), "extracted", stem)

    cameras_to_extract = [c.strip() for c in args.cameras.split(",") if c.strip()]
    for c in cameras_to_extract:
        if c not in CAMERA_NAMES:
            print(f"⚠️  Warning: '{c}' is not a standard RoboTwin camera ({CAMERA_NAMES})")

    with h5py.File(args.hdf5_path, "r") as f:
        vector = load_joint_action_vector(f)
        cameras = {}
        for cam in cameras_to_extract:
            try:
                cameras[cam] = load_camera_rgb(f, cam)
            except KeyError as e:
                print(f"⚠️  {e}")
                cameras[cam] = None
        if args.with_video_frame:
            if "video_frame" in f:
                cameras["video_frame"] = parse_img_array(f["video_frame"][()])
            else:
                print("⚠️  /video_frame not found in this hdf5, skipped.")
                cameras["video_frame"] = None

    print_summary(vector, cameras)

    do_anything = args.save_qpos or args.save_videos or args.save_frames
    if not do_anything:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"📦 Output directory: {args.out_dir}")

    if args.save_qpos:
        save_qpos(vector, args.out_dir, stem)

    for cam, frames in cameras.items():
        if frames is None:
            continue
        if args.save_videos:
            save_camera_video(
                frames,
                out_path=os.path.join(args.out_dir, f"{stem}_{cam}.mp4"),
                fps=args.fps,
            )
        if args.save_frames:
            save_camera_frames(
                frames,
                out_dir=os.path.join(args.out_dir, f"{stem}_{cam}"),
            )


if __name__ == "__main__":
    main()
