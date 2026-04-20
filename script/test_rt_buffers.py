"""
SAPIEN 3.x RT-buffer probe.

Goal:
    Figure out *which* auxiliary G-buffers (Albedo / Normal / ...) the current
    SAPIEN ray-tracing shader actually exposes via `camera.get_picture(name)`.
    We need this to decide whether we can feed OIDN offline with
    color + albedo + normal (high-quality denoising path).

Design:
    1. Build a minimal RT scene (same style as test_render_upgrade.py).
    2. Set denoiser="none" so the renderer never touches OIDN/OptiX
       (we are trying to capture the *noisy* G-buffers, not the denoised one,
        and we also want this script to survive on hosts where OIDN crashes).
    3. Call cam.take_picture() once.
    4. For each candidate buffer name, try cam.get_picture(name) and report
         - whether it succeeded
         - shape / dtype / min / max / mean
         - save a preview PNG when possible
    5. Summarize which names are usable for an offline-OIDN pipeline.

Run:
    python script/test_rt_buffers.py
    # optional:
    python script/test_rt_buffers.py --width 640 --height 480 --spp 32

Exit codes:
    0 = probe finished (see per-buffer results in the table at the bottom)
    1 = could not even build the RT scene
"""

import argparse
import os
import sys
import traceback
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Candidate buffer names to probe.
#
# SAPIEN historically exposes render targets whose names come from the shader
# YAML (e.g. sapien/vulkan_shader/rt/camera.yml). Different SAPIEN versions
# ship different shader sets, and different shader sets expose different
# names. We probe a superset and report what actually works.
# ---------------------------------------------------------------------------
CANDIDATE_BUFFERS = [
    # Color variants
    "Color",
    "ColorRaw",
    "HdrColor",
    "DenoisedColor",
    # Geometry / depth / position
    "Position",
    "Depth",
    "ViewPosition",
    "WorldPosition",
    # Normal variants (we want world-space or view-space, non-denoised)
    "Normal",
    "WorldNormal",
    "ViewNormal",
    "ShadingNormal",
    "GeometryNormal",
    # Albedo / base color variants
    "Albedo",
    "BaseColor",
    "Diffuse",
    "DiffuseAlbedo",
    # Segmentation (sanity check, we know it should exist)
    "Segmentation",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72, flush=True)


def _build_probe_scene(sapien_mod, width: int, height: int):
    """Build a minimal textured scene so Albedo/Normal are non-trivial."""
    from sapien.render import RenderMaterial

    scene = sapien_mod.Scene()
    scene.set_timestep(1 / 250)
    scene.add_ground(altitude=0)
    scene.set_ambient_light([0.4, 0.4, 0.4])
    scene.add_directional_light([0.3, 1, -1], [0.8, 0.8, 0.8])

    # A red box
    mat_red = RenderMaterial()
    mat_red.set_base_color([0.85, 0.15, 0.15, 1.0])
    b1 = scene.create_actor_builder()
    b1.add_box_visual(half_size=[0.12, 0.12, 0.12], material=mat_red)
    b1.build_static(name="box_red")

    # A green box, offset, different normal facing
    mat_green = RenderMaterial()
    mat_green.set_base_color([0.15, 0.75, 0.15, 1.0])
    b2 = scene.create_actor_builder()
    b2.add_box_visual(half_size=[0.08, 0.20, 0.08], material=mat_green)
    from sapien import Pose
    try:
        b2.set_initial_pose(Pose([0.0, 0.25, 0.08]))
    except AttributeError:
        # Fallback for older builder APIs that expose initial_pose directly.
        b2.initial_pose = Pose([0.0, 0.25, 0.08])
    b2.build_static(name="box_green")

    cam = scene.add_camera(
        name="probe_cam",
        width=width,
        height=height,
        fovy=1.0,
        near=0.01,
        far=10.0,
    )
    cam.set_local_pose(sapien_mod.Pose([0.8, -0.4, 0.5], [0, 0, 0, 1]))
    # Point roughly at origin: rotate to look down-forward-left
    # (we don't bother with a precise look-at; default orientation is fine
    # for probing because we only care that the buffers are non-empty).
    return scene, cam


def _array_stats(arr):
    import numpy as np
    a = np.asarray(arr)
    try:
        amin = float(a.min())
        amax = float(a.max())
        amean = float(a.mean())
    except Exception:
        amin = amax = amean = float("nan")
    return a.shape, str(a.dtype), amin, amax, amean


def _save_preview_png(arr, out_path: str, name: str) -> bool:
    """Best-effort preview: map whatever the buffer is into a uint8 PNG."""
    try:
        import numpy as np
        import imageio.v2 as imageio
    except Exception:
        return False

    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[..., None]

    if a.shape[-1] >= 3:
        img = a[..., :3]
    else:
        img = np.repeat(a[..., :1], 3, axis=-1)

    if name.lower().startswith(("normal", "worldnormal", "viewnormal",
                                "shadingnormal", "geometrynormal")):
        # Map [-1,1] to [0,1]
        img = (img * 0.5) + 0.5

    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)

    try:
        imageio.imwrite(out_path, img)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--spp", type=int, default=32,
                        help="ray tracing samples per pixel")
    parser.add_argument("--path-depth", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="directory to write preview PNGs "
                             "(default: alongside this script)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_rt_buffers_out",
    )
    os.makedirs(out_dir, exist_ok=True)

    # ---- import + configure RT with denoiser=none ----
    section("[1] import sapien & configure RT (denoiser=none)")
    try:
        import sapien
        from sapien.render import set_global_config

        set_global_config(max_num_materials=50000, max_num_textures=50000)
        print(f"sapien version : {getattr(sapien, '__version__', '?')}")
        print(f"sapien path    : {os.path.dirname(sapien.__file__)}")

        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(args.spp)
        sapien.render.set_ray_tracing_path_depth(args.path_depth)
        sapien.render.set_ray_tracing_denoiser("none")
        print(f"shader_dir     : rt")
        print(f"spp            : {args.spp}")
        print(f"path_depth     : {args.path_depth}")
        print(f"denoiser       : "
              f"{sapien.render.get_ray_tracing_denoiser()}")
    except Exception:
        print("[1 FAIL] cannot import / configure sapien RT")
        traceback.print_exc()
        return 1

    # ---- build scene + take_picture ----
    section("[2] build RT scene + take_picture()")
    try:
        scene, cam = _build_probe_scene(sapien, args.width, args.height)
        scene.step()
        scene.update_render()
        cam.take_picture()
        print("[2 OK] take_picture() returned")
    except Exception:
        print("[2 FAIL] could not render RT frame")
        traceback.print_exc()
        return 1

    # ---- probe buffers ----
    section("[3] probe candidate G-buffers")
    results = []  # list of (name, ok, info)
    for name in CANDIDATE_BUFFERS:
        try:
            arr = cam.get_picture(name)
        except Exception as e:
            msg = type(e).__name__
            print(f"  [--] {name:<18s}  NOT AVAILABLE ({msg})")
            results.append((name, False, msg))
            continue

        try:
            shape, dtype, amin, amax, amean = _array_stats(arr)
        except Exception:
            print(f"  [??] {name:<18s}  got object but stats failed")
            results.append((name, True, "stats-failed"))
            continue

        print(f"  [ok] {name:<18s}  shape={shape}  dtype={dtype}  "
              f"min={amin:+.4f}  max={amax:+.4f}  mean={amean:+.4f}")

        # Save preview
        preview_path = os.path.join(out_dir, f"probe_{name}.png")
        if _save_preview_png(arr, preview_path, name):
            print(f"        preview -> {preview_path}")

        results.append((name, True, (shape, dtype, amin, amax, amean)))

    # ---- summary ----
    section("[4] summary")
    usable = [n for n, ok, _ in results if ok]
    missing = [n for n, ok, _ in results if not ok]
    print(f"available buffers ({len(usable)}):")
    for n in usable:
        print(f"  + {n}")
    print(f"\nmissing buffers ({len(missing)}):")
    for n in missing:
        print(f"  - {n}")

    # Tell the user what this implies for offline OIDN
    section("[5] implication for offline OIDN (color + albedo + normal)")
    color_name = _first_present(usable,
                                ["Color", "ColorRaw", "HdrColor"])
    albedo_name = _first_present(usable,
                                 ["Albedo", "BaseColor",
                                  "DiffuseAlbedo", "Diffuse"])
    normal_name = _first_present(usable,
                                 ["Normal", "WorldNormal", "ViewNormal",
                                  "ShadingNormal", "GeometryNormal"])
    print(f"  color  buffer : {color_name or '*** NOT FOUND ***'}")
    print(f"  albedo buffer : {albedo_name or '*** NOT FOUND ***'}")
    print(f"  normal buffer : {normal_name or '*** NOT FOUND ***'}")

    if color_name and albedo_name and normal_name:
        print("\n  => High-quality offline OIDN pipeline IS feasible.")
        print("     Modify envs/camera/camera.py to additionally grab")
        print(f"     '{albedo_name}' and '{normal_name}' per frame.")
        return 0
    if color_name:
        print("\n  => Only color is available. Fall back to single-channel")
        print("     offline OIDN (lower quality, still works).")
        return 0
    print("\n  => No usable color buffer. Something is deeply wrong; stop.")
    return 1


def _first_present(pool, candidates):
    for c in candidates:
        if c in pool:
            return c
    return None


if __name__ == "__main__":
    sys.exit(main())
