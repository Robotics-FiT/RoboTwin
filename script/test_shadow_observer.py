"""Reproduce the observer camera view from random_dance with a simplified
scene (ground + table + a few cuboids standing in for the robot) to verify
that shadows from the HDRI-derived sun are actually visible.
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
from envs.utils.hdri_light import estimate_sun_from_hdri, estimate_ground_color_from_hdri

HDRI = os.path.join(REPO_ROOT, "assets", "scenes", "HDRIs", "grasslands_sunset_4k.exr")
OUT = os.path.join(REPO_ROOT, "script", "shadow_observer_preview.png")


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
    scene.add_ground(0.0, render=False)

    # shadow catcher
    ground_col = estimate_ground_color_from_hdri(HDRI)
    builder = scene.create_actor_builder()
    mat = renderer.create_material()
    mat.set_base_color([float(ground_col[0]), float(ground_col[1]), float(ground_col[2]), 1.0])
    mat.set_metallic(0.0); mat.set_roughness(1.0)
    builder.add_box_visual(half_size=[10, 10, 0.001], material=mat)
    catcher = builder.build_static(name="catcher")
    catcher.set_pose(sapien.Pose(p=[0, 0, 0.001]))

    # table (1.2 x 0.7 x 0.74)
    builder = scene.create_actor_builder()
    mat = renderer.create_material()
    mat.set_base_color([0.70, 0.55, 0.35, 1.0]); mat.set_metallic(0.0); mat.set_roughness(0.8)
    builder.add_box_visual(half_size=[0.6, 0.35, 0.025], material=mat)
    table_top = builder.build_static(name="table")
    table_top.set_pose(sapien.Pose(p=[0, 0, 0.74]))

    # "robot torso" (cylinder approximated as box) standing behind the table
    builder = scene.create_actor_builder()
    mat = renderer.create_material()
    mat.set_base_color([0.2, 0.2, 0.25, 1.0]); mat.set_metallic(0.1); mat.set_roughness(0.4)
    builder.add_box_visual(half_size=[0.15, 0.15, 0.5], material=mat)
    torso = builder.build_static(name="torso")
    torso.set_pose(sapien.Pose(p=[0.0, -0.65, 0.6]))

    # A couple of objects on the table to cast visible shadows.
    for i, (x, y, c) in enumerate([(-0.2, 0.05, [0.9, 0.1, 0.1]),
                                   ( 0.25, -0.05, [0.1, 0.6, 0.1]),
                                   ( 0.0, -0.15, [0.1, 0.3, 0.9])]):
        b = scene.create_actor_builder()
        m = renderer.create_material()
        m.set_base_color(c + [1.0]); m.set_metallic(0.0); m.set_roughness(0.5)
        b.add_box_visual(half_size=[0.03, 0.03, 0.05], material=m)
        a = b.build_static(name=f"obj{i}")
        a.set_pose(sapien.Pose(p=[x, y, 0.74 + 0.025 + 0.05]))

    # Lighting (matches _base_task.py after recent edits)
    scene.set_ambient_light([0.25, 0.25, 0.25])
    scene.set_environment_map(HDRI)
    sun_dir, sun_col = estimate_sun_from_hdri(HDRI, intensity_scale=4.0)
    print(f"sun dir = {sun_dir.tolist()}")
    print(f"sun col = {sun_col.tolist()}")
    scene.add_directional_light(sun_dir.tolist(), sun_col.tolist(),
                                shadow=True, shadow_scale=8.0,
                                shadow_near=-15.0, shadow_far=25.0,
                                shadow_map_size=4096)
    # the two default point lights at the new low intensity
    scene.add_point_light([ 1, 0, 1.8], [0.25, 0.25, 0.25], shadow=True)
    scene.add_point_light([-1, 0, 1.8], [0.25, 0.25, 0.25], shadow=True)

    # observer camera matching envs/camera/camera.py "observer" branch
    cam_pos = np.array([0.0, 0.65, 1.20])
    fwd = np.array([0.0, -1.0, -0.4]); fwd /= np.linalg.norm(fwd)
    left = np.array([1.0, 0.0, 0.0])
    up = np.cross(fwd, left); up /= np.linalg.norm(up)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([fwd, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    cam = scene.add_camera("obs", width=960, height=720, fovy=np.deg2rad(70),
                           near=0.05, far=100.0)
    cam.set_local_pose(sapien.Pose(mat44))

    scene.step()
    scene.update_render()
    cam.take_picture()
    rgba = cam.get_picture("Color")
    rgb = (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)
    import cv2
    cv2.imwrite(OUT, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
