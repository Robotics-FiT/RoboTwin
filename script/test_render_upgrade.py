"""
Post-upgrade SAPIEN verification script (for SAPIEN 3.0.x).

Goal:
    After upgrading `sapien`, reproduce the exact code path that triggered the
    `OIDN Error: an illegal memory access was encountered` error, and compare
    the OIDN denoiser against the OptiX denoiser on this machine.

Design:
    Each RT-denoiser test runs in a FRESH subprocess, so a C++ crash
    (SIGSEGV / Access Violation) in one denoiser does not prevent the other
    from being tested.

Run:
    python script/test_render_upgrade.py

Exit codes:
    0 = L1/L2 passed AND at least one of OptiX / OIDN produced a valid image.
    1 = L1 import failed
    2 = L2 rasterizer scene failed
    3 = both denoisers failed
"""

import os
import subprocess
import sys
import traceback
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)


# ===========================================================================
# Helpers (shared by main process and child process)
# ===========================================================================

def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60, flush=True)


def _build_probe_scene(sapien_mod):
    """Build a minimal scene with one colored box, a light, and one camera."""
    from sapien.render import RenderMaterial

    scene = sapien_mod.Scene()
    scene.set_timestep(1 / 250)
    scene.add_ground(altitude=0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.6, 0.6, 0.6])

    mat = RenderMaterial()
    mat.set_base_color([0.8, 0.2, 0.2, 1.0])  # RGBA in [0,1]

    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=[0.1, 0.1, 0.1], material=mat)
    builder.build_static(name="probe_box")

    cam = scene.add_camera(
        name="probe_cam",
        width=320,
        height=240,
        fovy=1.0,
        near=0.01,
        far=10.0,
    )
    cam.set_local_pose(sapien_mod.Pose([1.0, 0, 0.5], [0, 0, 0, 1]))
    return scene, cam


# ===========================================================================
# Child process entry: render one RT frame with the requested denoiser
# ===========================================================================

def _child_render_rt(denoiser_name: str) -> int:
    """Executed in a fresh subprocess. Returns 0 on success."""
    import numpy as np
    import sapien

    tag = f"child[{denoiser_name}]"
    try:
        print(f"{tag}.1 configuring RT globals ...", flush=True)
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser(denoiser_name)
        print(f"      denoiser  = {sapien.render.get_ray_tracing_denoiser()}",
              flush=True)

        print(f"{tag}.2 building RT scene ...", flush=True)
        scene_rt, cam_rt = _build_probe_scene(sapien)

        print(f"{tag}.3 scene.step() + update_render() ...", flush=True)
        scene_rt.step()
        scene_rt.update_render()

        print(f"{tag}.4 cam.take_picture()  <-- denoiser kernel fires here",
              flush=True)
        cam_rt.take_picture()

        print(f"{tag}.5 cam.get_picture('Color') ...", flush=True)
        rgba = cam_rt.get_picture("Color")
        arr = np.asarray(rgba)
        print(f"      rgba shape     : {arr.shape}")
        print(f"      rgba dtype     : {arr.dtype}")
        print(f"      rgba min/max   : "
              f"{float(arr.min()):.4f} / {float(arr.max()):.4f}")
        print(f"      rgba mean      : {float(arr.mean()):.4f}", flush=True)

        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"test_render_upgrade_out_{denoiser_name}.png",
        )
        try:
            import imageio.v2 as imageio

            img_uint8 = (np.clip(arr[..., :3], 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(out_path, img_uint8)
            print(f"      wrote image    : {out_path}", flush=True)
        except Exception:
            print("      (imageio write skipped)", flush=True)

        print(f"{tag} OK", flush=True)
        return 0
    except Exception:
        print(f"{tag} FAIL (python exception)")
        traceback.print_exc()
        return 1


# ===========================================================================
# Main process: dispatch children, collect results
# ===========================================================================

def _run_rt_in_subprocess(denoiser_name: str) -> bool:
    """Launch ourselves as a subprocess to isolate C++ crashes."""
    script_path = os.path.abspath(__file__)
    cmd = [sys.executable, script_path, "--run-rt", denoiser_name]
    print(f">>> spawning child: {' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(cmd)
    except Exception:
        print(f"subprocess launch failed for {denoiser_name!r}")
        traceback.print_exc()
        return False

    rc = proc.returncode
    print(f">>> child[{denoiser_name}] exited with returncode = {rc}",
          flush=True)
    if rc == 0:
        return True

    # Non-zero return code ==> decode what happened
    if rc < 0:
        import signal as _signal
        try:
            sig_name = _signal.Signals(-rc).name
        except (ValueError, AttributeError):
            sig_name = f"signal {-rc}"
        print(f"    child was killed by {sig_name} "
              f"(likely a C++ crash in the RT/{denoiser_name} path)")
    elif rc == 0xC0000005 or rc == -1073741819:
        print("    child aborted with Windows STATUS_ACCESS_VIOLATION "
              "(C++ crash in the RT path)")
    else:
        print(f"    child returned non-zero exit code {rc}")
    return False


def _main_dispatcher() -> int:
    # -----------------------------------------------------------------------
    # Level 1: import sapien and print version (main process)
    # -----------------------------------------------------------------------
    section("[L1] import sapien & print version")
    try:
        import sapien

        version = getattr(sapien, "__version__", "unknown")
        print(f"sapien version : {version}")
        print(f"sapien path    : {os.path.dirname(sapien.__file__)}")
    except Exception:
        print("[L1 FAIL] cannot import sapien")
        traceback.print_exc()
        return 1

    # -----------------------------------------------------------------------
    # Level 2: rasterizer scene (main process, no RT)
    # -----------------------------------------------------------------------
    section("[L2] create rasterizer scene")
    try:
        from sapien.render import set_global_config

        set_global_config(max_num_materials=50000, max_num_textures=50000)
        scene, cam = _build_probe_scene(sapien)
        scene.step()
        scene.update_render()
        cam.take_picture()
        print("[L2 OK] rasterizer scene rendered one frame")
        del cam
        del scene
    except Exception:
        print("[L2 FAIL] rasterizer scene")
        traceback.print_exc()
        return 2

    # -----------------------------------------------------------------------
    # Level 3: RT + OptiX (in subprocess)
    # -----------------------------------------------------------------------
    section("[L3] ray tracing + OptiX (in subprocess)")
    optix_ok = _run_rt_in_subprocess("optix")

    # -----------------------------------------------------------------------
    # Level 4: RT + OIDN (in subprocess)
    # -----------------------------------------------------------------------
    section("[L4] ray tracing + OIDN (in subprocess)")
    oidn_ok = _run_rt_in_subprocess("oidn")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    section("SUMMARY")
    print(f"  L3 (optix) : {'OK' if optix_ok else 'FAIL'}")
    print(f"  L4 (oidn)  : {'OK' if oidn_ok else 'FAIL'}")

    if optix_ok:
        print("\n>>> OptiX works. Recommended: switch denoiser to 'optix' "
              "in envs/_base_task.py")
        return 0
    if oidn_ok:
        print("\n>>> OIDN works, OptiX failed. Keep using 'oidn'.")
        return 0

    print("\n>>> Both denoisers failed. Fallback: denoiser='none' "
          "or disable ray tracing entirely.")
    return 3


# ===========================================================================
# Entrypoint
# ===========================================================================

if __name__ == "__main__":
    # Child-mode: `python test_render_upgrade.py --run-rt <denoiser>`
    if len(sys.argv) == 3 and sys.argv[1] == "--run-rt":
        sys.exit(_child_render_rt(sys.argv[2]))
    # Normal mode
    sys.exit(_main_dispatcher())
