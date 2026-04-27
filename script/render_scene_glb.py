"""Render a static preview PNG of a .glb scene mesh (no physics, no robot).

Defaults to ``assets/scenes/3d_front/00110bde-f58_Kitchen-38252_Kitchen/room_mesh.glb``
rotated from the Y-up convention used by 3D-Front into SAPIEN's Z-up world, then
parked so that the room's bbox floor sits at ``z = 0``. A single off-screen
camera shoots a 640x480 RGB image and saves it to a PNG.

Examples::

    # Default: render the preset kitchen scene
    python script/render_scene_glb.py

    # Render any glb, custom resolution & output path
    python script/render_scene_glb.py \
        --glb assets/scenes/3d_front/<room>/room_mesh.glb \
        --out script/scene_preview.png \
        --width 1280 --height 720

    # Skip the Y-up -> Z-up auto fix (if the mesh is already Z-up)
    python script/render_scene_glb.py --no-yup-to-zup

    # Override camera manually (position + target in world frame, metres)
    python script/render_scene_glb.py --cam-pos 2 2 1.6 --cam-target 0 0 1.0
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import sapien.core as sapien
from sapien.render import set_global_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLB = (
    REPO_ROOT
    / "assets/scenes/3d_front/00110bde-f58_Kitchen-38252_Kitchen/room_mesh.glb"
)


def make_pose(glb_path: Path, yup_to_zup: bool, sit_floor_on_zero: bool) -> sapien.Pose:
    """Build the visual actor pose for the scene mesh.

    * ``yup_to_zup``: rotate +90 deg around X so that glb Y-up becomes SAPIEN Z-up.
    * ``sit_floor_on_zero``: shift along world-Z so that the post-rotation bbox
      minimum Z lands on 0 (the room's floor sits on the SAPIEN ground plane).
    """
    p = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if yup_to_zup:
        # +90 deg around world X maps (x, y, z)_glb -> (x, -z, y)_world, i.e.
        # glb-Y becomes world-Z, glb-Z becomes world -Y. Quaternion [w,x,y,z]:
        q = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0], dtype=np.float32)
    else:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    if sit_floor_on_zero:
        world_min, _ = _mesh_world_aabb(glb_path, yup_to_zup)
        p[2] = float(-world_min[2])
    return sapien.Pose(p=p, q=q)


def _mesh_world_aabb(glb_path: Path, yup_to_zup: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return the world-frame AABB of the mesh given our chosen pose (no Z-shift
    applied yet). Computed from actual vertices via trimesh, not room_info.json
    (that file's bbox only covers listed furniture, not the walls)."""
    import trimesh  # lazy import; heavy

    mesh = trimesh.load(str(glb_path), force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if yup_to_zup:
        # Rx(+90deg): (x, y, z) -> (x, -z, y)
        verts = np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=1)
    return verts.min(axis=0), verts.max(axis=0)


def setup_scene(
    glb_path: Path,
    yup_to_zup: bool,
    sit_floor_on_zero: bool,
    shader: str,
    use_oidn: bool,
    rt_spp: int,
    rt_depth: int,
) -> sapien.Scene:
    engine = sapien.Engine()
    set_global_config(max_num_materials=50000, max_num_textures=50000)
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    # Match high-quality settings from script/test_render.py only when using
    # the path-tracer; "default" is the rasterizer (much faster first run).
    sapien.render.set_camera_shader_dir(shader)
    if shader == "rt":
        sapien.render.set_ray_tracing_samples_per_pixel(rt_spp)
        sapien.render.set_ray_tracing_path_depth(rt_depth)
        if use_oidn:
            sapien.render.set_ray_tracing_denoiser("oidn")

    scene = engine.create_scene(sapien.SceneConfig())
    scene.set_timestep(1 / 240)

    # Figure out where to park the lights: above the actual room, not at the
    # world origin (the 3D-Front meshes are often several metres off-origin).
    world_min, world_max = _mesh_world_aabb(glb_path, yup_to_zup)
    if sit_floor_on_zero:
        world_max = world_max + np.array([0.0, 0.0, -world_min[2]])
        world_min = world_min + np.array([0.0, 0.0, -world_min[2]])
    centre = 0.5 * (world_min + world_max)
    ceiling_z = float(world_max[2] - 0.05)

    # Ambient + a couple of directional/point lights so PBR materials in the
    # glb actually show up without looking flat. Point lights are placed just
    # under the ceiling so they illuminate the whole volume.
    scene.set_ambient_light([0.35, 0.35, 0.35])
    scene.add_directional_light([-0.5, -1.0, -1.0], [1.0, 1.0, 1.0], shadow=True)
    scene.add_point_light([centre[0], centre[1], ceiling_z], [30.0, 30.0, 30.0])
    scene.add_point_light(
        [world_min[0] + 0.25 * (world_max[0] - world_min[0]),
         world_min[1] + 0.25 * (world_max[1] - world_min[1]),
         ceiling_z],
        [20.0, 20.0, 20.0],
    )
    scene.add_point_light(
        [world_min[0] + 0.75 * (world_max[0] - world_min[0]),
         world_min[1] + 0.75 * (world_max[1] - world_min[1]),
         ceiling_z],
        [20.0, 20.0, 20.0],
    )

    # Add the scene mesh as a visual-only static actor.
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static")
    builder.add_visual_from_file(filename=str(glb_path))
    actor = builder.build(name="scene_mesh")
    actor.set_pose(make_pose(glb_path, yup_to_zup, sit_floor_on_zero))
    return scene


def compute_auto_camera(
    glb_path: Path,
    yup_to_zup: bool,
    sit_floor_on_zero: bool,
    inside: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick a reasonable camera for an unknown room.

    * ``inside=True`` (default): stand inside, near one corner of the room's
      true vertex AABB at eye height, look at the opposite corner.
    * ``inside=False``: stand outside at ~1x the diagonal away, look at the
      bbox centre (useful for orbit-style previews of standalone props).
    """
    world_min, world_max = _mesh_world_aabb(glb_path, yup_to_zup)

    if sit_floor_on_zero:
        shift = -world_min[2]
        world_min = world_min + np.array([0.0, 0.0, shift])
        world_max = world_max + np.array([0.0, 0.0, shift])

    centre = 0.5 * (world_min + world_max)
    extent = world_max - world_min

    if inside:
        # Stand near the room centre (not hugging a wall) at eye height, look
        # towards one of the far corners. This avoids staring straight at the
        # nearest cabinet face and reveals the whole interior.
        eye_height = min(1.4, 0.75 * extent[2]) + world_min[2]
        cam_pos = np.array(
            [
                centre[0] - 0.05 * extent[0],
                centre[1] - 0.05 * extent[1],
                eye_height,
            ]
        )
        cam_target = np.array(
            [
                world_max[0] - 0.1 * extent[0],
                world_max[1] - 0.1 * extent[1],
                world_min[2] + 0.3 * extent[2],
            ]
        )
    else:
        diag = float(np.linalg.norm(extent))
        cam_pos = centre + np.array([diag * 0.9, diag * 0.9, diag * 0.25])
        cam_target = centre
    return cam_pos, cam_target


def lookat_quat(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return SAPIEN camera-mount quaternion [w,x,y,z] for a camera placed at
    ``cam_pos`` looking at ``target``.

    SAPIEN camera-local convention: +X forward, +Y left, +Z up.
    """
    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    world_up = np.array([0.0, 0.0, 1.0])
    left = np.cross(world_up, forward)
    if np.linalg.norm(left) < 1e-6:
        left = np.array([0.0, 1.0, 0.0])
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    # Columns of rotation matrix are the camera-frame axes expressed in world.
    R = np.stack([forward, left, up], axis=1)
    # Matrix -> quaternion (w, x, y, z). Avoid the transforms3d dependency.
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def render_once(
    scene: sapien.Scene,
    out_path: Path,
    width: int,
    height: int,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    fovy_deg: float,
) -> None:
    cam = scene.add_camera(
        name="preview_cam",
        width=width,
        height=height,
        fovy=np.deg2rad(fovy_deg),
        near=0.05,
        far=100.0,
    )
    pose = sapien.Pose(p=cam_pos.astype(np.float32), q=lookat_quat(cam_pos, cam_target))
    cam.set_local_pose(pose)

    scene.step()
    scene.update_render()
    cam.take_picture()

    rgba = cam.get_picture("Color")  # (H, W, 4) float32 in [0,1]
    rgb = (np.clip(rgba[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)

    import cv2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # cv2 writes BGR
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"🎨 Preview saved to {out_path}  ({width}x{height})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--glb", type=str, default=str(DEFAULT_GLB), help="Path to .glb scene mesh")
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "script/scene_preview.png"),
        help="Output PNG path",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fovy", type=float, default=45.0, help="Vertical FOV in degrees")

    parser.add_argument(
        "--no-yup-to-zup",
        dest="yup_to_zup",
        action="store_false",
        help="Disable the automatic Y-up -> Z-up rotation",
    )
    parser.set_defaults(yup_to_zup=True)

    parser.add_argument(
        "--no-sit-floor",
        dest="sit_floor",
        action="store_false",
        help="Do not auto-translate the bbox minimum to z=0",
    )
    parser.set_defaults(sit_floor=True)

    parser.add_argument("--cam-pos", type=float, nargs=3, default=None,
                        help="Override camera position (x y z). Otherwise auto from bbox.")
    parser.add_argument("--cam-target", type=float, nargs=3, default=None,
                        help="Override camera target (x y z). Otherwise bbox centre.")
    parser.add_argument(
        "--outside",
        dest="inside",
        action="store_false",
        help="Stand outside the room (orbit view) instead of inside one corner.",
    )
    parser.set_defaults(inside=True)

    parser.add_argument(
        "--shader",
        type=str,
        default="default",
        choices=["default", "rt"],
        help="Renderer to use. 'default' = rasterizer (fast), 'rt' = path tracer.",
    )
    parser.add_argument("--rt-spp", type=int, default=32, help="Ray-tracing samples per pixel")
    parser.add_argument("--rt-depth", type=int, default=8, help="Ray-tracing path depth")
    parser.add_argument("--no-oidn", dest="oidn", action="store_false", help="Disable OIDN denoiser")
    parser.set_defaults(oidn=True)

    args = parser.parse_args()

    glb_path = Path(args.glb).resolve()
    if not glb_path.is_file():
        raise FileNotFoundError(glb_path)

    scene = setup_scene(
        glb_path=glb_path,
        yup_to_zup=args.yup_to_zup,
        sit_floor_on_zero=args.sit_floor,
        shader=args.shader,
        use_oidn=args.oidn,
        rt_spp=args.rt_spp,
        rt_depth=args.rt_depth,
    )

    if args.cam_pos is not None and args.cam_target is not None:
        cam_pos = np.asarray(args.cam_pos, dtype=np.float64)
        cam_target = np.asarray(args.cam_target, dtype=np.float64)
    else:
        cam_pos, cam_target = compute_auto_camera(
            glb_path, args.yup_to_zup, args.sit_floor, inside=args.inside
        )
        if args.cam_pos is not None:
            cam_pos = np.asarray(args.cam_pos, dtype=np.float64)
        if args.cam_target is not None:
            cam_target = np.asarray(args.cam_target, dtype=np.float64)

    print(f"🏠 scene glb : {glb_path}")
    print(f"📷 cam pos   : {cam_pos.tolist()}")
    print(f"📷 cam target: {cam_target.tolist()}")

    render_once(
        scene=scene,
        out_path=Path(args.out).resolve(),
        width=args.width,
        height=args.height,
        cam_pos=cam_pos,
        cam_target=cam_target,
        fovy_deg=args.fovy,
    )


if __name__ == "__main__":
    main()
