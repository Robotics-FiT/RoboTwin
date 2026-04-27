"""Sanity-check HDRI-driven shadows.

Renders a small scene (plate + sphere) with:
* HDRI skybox       -> scene.set_environment_map
* Directional light -> direction + colour extracted from the HDRI's sun
                       (shadow=True)

The shadow of the sphere on the plate should point *away from* the visual
sun in the HDRI behind it. Saved to ``script/shadow_preview.png``.
"""
import os
import sys
import warnings

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import sapien
import sapien.core as sc
from sapien.render import set_global_config

from envs.utils.hdri_light import estimate_sun_from_hdri

HDRI = os.path.join(REPO_ROOT, "assets", "scenes", "HDRIs", "grasslands_sunset_4k.exr")
OUT = os.path.join(REPO_ROOT, "script", "shadow_preview.png")


def main():
    engine = sc.Engine()
    set_global_config(max_num_materials=5000, max_num_textures=5000)
    renderer = sc.SapienRenderer()
    engine.set_renderer(renderer)

    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(64)
    sapien.render.set_ray_tracing_path_depth(6)
    sapien.render.set_ray_tracing_denoiser("oidn")

    scene = engine.create_scene(sc.SceneConfig())
    scene.set_timestep(1 / 240)
    scene.add_ground(0.0, render=True)
    scene.set_ambient_light([0.15, 0.15, 0.18])
    scene.set_environment_map(HDRI)

    sun = estimate_sun_from_hdri(HDRI, intensity_scale=2.5)
    assert sun is not None, "failed to extract sun from HDRI"
    sun_dir, sun_col = sun
    elev = float(np.rad2deg(np.arcsin(-sun_dir[2])))
    azim = float(np.rad2deg(np.arctan2(-sun_dir[1], -sun_dir[0])))
    print(f"sun light dir (light -> scene) = {sun_dir.tolist()}")
    print(f"sun altitude                    = {elev:.1f} deg")
    print(f"sun azimuth                     = {azim:.1f} deg (0 = +x, 90 = +y)")
    print(f"sun col                         = {sun_col.tolist()}")

    # Big ortho shadow volume because the sun is ~low in the sky -> long shadow.
    scene.add_directional_light(sun_dir.tolist(), sun_col.tolist(), shadow=True,
                                shadow_scale=8.0, shadow_near=-15.0, shadow_far=25.0,
                                shadow_map_size=4096)

    # A matte sphere sitting on the ground, offset so the shadow direction
    # (roughly opposite of azim) is clearly visible inside the camera view.
    builder = scene.create_actor_builder()
    mat = renderer.create_material()
    mat.set_base_color([0.85, 0.25, 0.20, 1.0])  # red-ish so it doesn't blend with the ground
    mat.set_metallic(0.0)
    mat.set_roughness(0.55)
    builder.add_sphere_visual(radius=0.25, material=mat)
    ball = builder.build(name="ball")
    ball.set_pose(sapien.Pose(p=[0.0, 0.0, 0.25]))

    # Camera looks along -x (a bit elevated), which is roughly opposite to the
    # HDRI sun (which sits at azim ~+36 deg). Shadow should lie on +x side
    # and skew slightly +y, visibly long because the sun is close to the horizon.
    cam = scene.add_camera("cam", width=960, height=540, fovy=np.deg2rad(45),
                           near=0.05, far=100.0)
    cam.set_local_pose(sapien.Pose(p=[2.5, 0.0, 1.2], q=[0.0, 0.0, 0.0, 1.0]))

    scene.step()
    scene.update_render()
    cam.take_picture()
    rgba = cam.get_picture("Color")
    rgb = (np.clip(rgba[..., :3], 0.0, 1.0) * 255).astype(np.uint8)
    import cv2
    cv2.imwrite(OUT, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
