import sapien.core as sapien
import numpy as np
import pdb
from PIL import Image, ImageColor
import open3d as o3d
import json
import transforms3d as t3d
import os
# 启用 OpenCV 的 OpenEXR 支持（必须在 import cv2 之前设置）。
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import torch
import yaml
import trimesh
import math
from .._GLOBAL_CONFIGS import CONFIGS_PATH
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig


def get_rotated_hdri_path(file_path, yaw_deg):
    """
    读取 equirectangular HDRI (.exr)，在水平方向按 yaw 角度做 np.roll 旋转，
    然后写到 /tmp 下的临时文件（不占额外磁盘），供 SAPIEN 加载。
    使用 cv2 读/写 EXR，避免 imageio 的 pyav 插件问题。
    """
    import tempfile
    temp_path = os.path.join(tempfile.gettempdir(), f"rotated_hdri_{os.getpid()}.exr")
    # IMREAD_UNCHANGED 保留原始 float32 及通道数 (含 alpha)
    hdri = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if hdri is None:
        raise RuntimeError(f"Failed to read HDRI: {file_path}. "
                           f"Ensure OPENCV_IO_ENABLE_OPENEXR=1 and the file exists.")
    width = hdri.shape[1]
    shift = int(width * (yaw_deg / 360.0))
    rotated = np.roll(hdri, shift, axis=1)
    # 保留 float32 精度写出 EXR
    cv2.imwrite(temp_path, rotated.astype(np.float32))
    return temp_path

try:
    import pytorch3d.ops as torch3d_ops

    def fps(points, num_points=1024, use_cuda=True):
        K = [num_points]
        if use_cuda:
            points = torch.from_numpy(points).cuda()
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.cpu().numpy()
        else:
            points = torch.from_numpy(points)
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.numpy()

        return sampled_points, indices

except:
    print("missing pytorch3d")

    def fps(points, num_points=1024, use_cuda=True):
        print("fps error: missing pytorch3d")
        exit()


class Camera:

    def __init__(self, bias=0, random_head_camera_dis=0, **kwags):
        """ """
        self.pcd_crop = kwags.get("pcd_crop", False)
        self.pcd_down_sample_num = kwags.get("pcd_down_sample_num", 0)
        self.pcd_crop_bbox = kwags.get("bbox", [[-0.6, -0.35, 0.7401], [0.6, 0.35, 2]])
        self.pcd_crop_bbox[0][2] += bias
        self.table_z_bias = bias
        self.random_head_camera_dis = random_head_camera_dis

        self.static_camera_config = []
        self.head_camera_type = kwags["camera"].get("head_camera_type", "D435")
        self.wrist_camera_type = kwags["camera"].get("wrist_camera_type", "D435")

        self.collect_head_camera = kwags["camera"].get("collect_head_camera", True)
        self.collect_wrist_camera = kwags["camera"].get("collect_wrist_camera", True)
        # Which camera feed is used for the final mp4 video & data-augmentation.
        #   - "default"  : use head_camera (original behavior)
        #   - "observer" : use the static observer_camera (third-person view)
        #   - "random"   : (reserved for future) observer_camera with randomised pose
        self.third_person_view = kwags["camera"].get("third_person_view", "default")
        if self.third_person_view not in ("default", "observer", "random"):
            print(f"[Camera] Unknown third_person_view '{self.third_person_view}', fallback to 'default'.")
            self.third_person_view = "default"

        # Parameters for the "random" third-person view.
        # Position is sampled on a partial sphere around the tabletop centre;
        # look-at point is the tabletop centre plus a small jitter.
        # Ranges may be overridden in the task config under camera.random_view.
        _rv = kwags["camera"].get("random_view", {}) or {}
        # azimuth=0 -> camera stands on -y side (front of robot); positive -> +x side.
        self.random_view_azimuth_range = tuple(_rv.get("azimuth_range", [-90.0, 90.0]))
        self.random_view_elevation_range = tuple(_rv.get("elevation_range", [15.0, 70.0]))
        self.random_view_radius_range = tuple(_rv.get("radius_range", [1.1, 1.8]))
        # Tabletop centre in world frame. table_z_bias is applied automatically
        # so the centre follows random_table_height.
        self.random_view_look_at = np.array(_rv.get("look_at", [0.0, 0.0, 0.74]), dtype=np.float64)
        self.random_view_look_at_jitter = float(_rv.get("look_at_jitter", 0.05))

        # embodiment = kwags.get('embodiment')
        # embodiment_config_path = os.path.join(CONFIGS_PATH, '_embodiment_config.yml')
        # with open(embodiment_config_path, 'r', encoding='utf-8') as f:
        #     embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        # robot_file = embodiment_types[embodiment]['file_path']
        # if robot_file is None:
        #     raise "No embodiment files"

        # robot_config_file = os.path.join(robot_file, 'config.yml')
        # with open(robot_config_file, 'r', encoding='utf-8') as f:
        #     embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        # TODO
        self.static_camera_info_list = kwags["left_embodiment_config"]["static_camera_list"]
        self.static_camera_num = len(self.static_camera_info_list)

    def load_camera(self, scene, hdri_path=None):
        """
        Add cameras and set camera parameters
            - Including four cameras: left, right, front, head.
        """
        near, far = 0.1, 100
        camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")

        assert os.path.isfile(camera_config_path), "task config file is missing"

        with open(camera_config_path, "r", encoding="utf-8") as f:
            camera_args = yaml.load(f.read(), Loader=yaml.FullLoader)

        # 每个 episode 初始化时：如果传入了 HDRI 路径，随机一个 yaw 旋转并重新加载，
        # 使得同一张底图在不同 episode 中呈现不同朝向的背景与 IBL 光照。
        if hdri_path and os.path.exists(hdri_path):
            yaw_deg = np.random.uniform(0, 360)
            rotated_path = get_rotated_hdri_path(hdri_path, yaw_deg)
            scene.set_environment_map(rotated_path)

        # sensor_mount_actor = scene.create_actor_builder().build_kinematic()

        # camera_args = get_camera_config()
        def create_camera(camera_info, random_head_camera_dis=0):
            if camera_info["type"] not in camera_args.keys():
                raise ValueError(f"Camera type {camera_info['type']} not supported")

            camera_config = camera_args[camera_info["type"]]
            cam_pos = np.array(camera_info["position"])
            vector = np.random.randn(3)
            random_dir = vector / np.linalg.norm(vector)
            cam_pos = cam_pos + random_dir * np.random.uniform(low=0, high=random_head_camera_dis)
            cam_forward = np.array(camera_info["forward"]) / np.linalg.norm(np.array(camera_info["forward"]))
            cam_left = np.array(camera_info["left"]) / np.linalg.norm(np.array(camera_info["left"]))
            up = np.cross(cam_forward, cam_left)
            mat44 = np.eye(4)
            mat44[:3, :3] = np.stack([cam_forward, cam_left, up], axis=1)
            mat44[:3, 3] = cam_pos

            # ========================= sensor camera =========================
            # sensor_config = StereoDepthSensorConfig()
            # sensor_config.rgb_resolution = (camera_config['w'], camera_config['h'])

            camera = scene.add_camera(
                name=camera_info["name"],
                width=camera_config["w"],
                height=camera_config["h"],
                fovy=np.deg2rad(camera_config["fovy"]),
                near=near,
                far=far,
            )
            camera.entity.set_pose(sapien.Pose(mat44))

            # ========================= sensor camera =========================
            # sensor_camera = StereoDepthSensor(
            #     sensor_config,
            #     sensor_mount_actor,
            #     sapien.Pose(mat44)
            # )
            # camera.entity.set_pose(sapien.Pose(camera_info['position']))
            # return camera, sensor_camera, camera_config
            return camera, camera_config

        # ================================= wrist camera =================================
        if self.collect_wrist_camera:
            wrist_camera_config = camera_args[self.wrist_camera_type]
            self.left_camera = scene.add_camera(
                name="left_camera",
                width=wrist_camera_config["w"],
                height=wrist_camera_config["h"],
                fovy=np.deg2rad(wrist_camera_config["fovy"]),
                near=near,
                far=far,
            )

            self.right_camera = scene.add_camera(
                name="right_camera",
                width=wrist_camera_config["w"],
                height=wrist_camera_config["h"],
                fovy=np.deg2rad(wrist_camera_config["fovy"]),
                near=near,
                far=far,
            )

        # ================================= sensor camera =================================
        # sensor_config = StereoDepthSensorConfig()
        # sensor_config.rgb_resolution = (wrist_camera_config['w'], wrist_camera_config['h'])
        # self.left_sensor_camera = StereoDepthSensor(
        #     sensor_config,
        #     sensor_mount_actor,
        #     sapien.Pose([0,0,0],[1,0,0,0])
        # )

        # self.right_sensor_camera = StereoDepthSensor(
        #     sensor_config,
        #     sensor_mount_actor,
        #     sapien.Pose([0,0,0],[1,0,0,0])
        # )

        # ================================= static camera =================================
        self.head_camera_id = None
        self.static_camera_list = []
        # self.static_sensor_camera_list = []
        self.static_camera_name = []
        # static camera list
        for i, camera_info in enumerate(self.static_camera_info_list):
            if camera_info.get("forward") == None:
                camera_info["forward"] = (-1 * np.array(camera_info["position"])).tolist()
            if camera_info.get("left") == None:
                camera_info["left"] = [
                    -camera_info["forward"][1],
                    camera_info["forward"][0],
                ] + [0]

            if camera_info["name"] == "head_camera":
                if self.collect_head_camera:
                    self.head_camera_id = i
                    camera_info["type"] = self.head_camera_type
                    # camera, sensor_camera, camera_config = create_camera(camera_info)
                    camera, camera_config = create_camera(camera_info,
                                                          random_head_camera_dis=self.random_head_camera_dis)
                    self.static_camera_list.append(camera)
                    self.static_camera_name.append(camera_info["name"])
                    # self.static_sensor_camera_list.append(sensor_camera)
                    self.static_camera_config.append(camera_config)
                    # ================================= sensor camera =================================
                    # camera_config = get_camera_config(camera_info['type'])
                    # cam_pos = np.array(camera_info['position'])
                    # cam_forward = np.array(camera_info['forward']) / np.linalg.norm(np.array(camera_info['forward']))
                    # cam_left = np.array(camera_info['left']) / np.linalg.norm(np.array(camera_info['left']))
                    # up = np.cross(cam_forward, cam_left)
                    # mat44 = np.eye(4)
                    # mat44[:3, :3] = np.stack([cam_forward, cam_left, up], axis=1)
                    # mat44[:3, 3] = cam_pos
                    # sensor_config = StereoDepthSensorConfig()
                    # sensor_config.rgb_resolution = (camera_config['w'], camera_config['h'])

                    # self.head_sensor = StereoDepthSensor(
                    #     sensor_config,
                    #     sensor_mount_actor,
                    #     sapien.Pose(mat44)
                    # )
            else:
                # camera, sensor_camera, camera_config = create_camera(camera_info)
                camera, camera_config = create_camera(camera_info)
                self.static_camera_list.append(camera)
                self.static_camera_name.append(camera_info["name"])
                # self.static_sensor_camera_list.append(sensor_camera)
                self.static_camera_config.append(camera_config)

        # observer camera (third-person view).
        # Resolution is bumped to 640x480 so the resulting mp4 is usable for data
        # augmentation. The pose is chosen to cover the whole dual-arm workspace.
        if self.third_person_view == "default":
            observer_w, observer_h, observer_fovy_deg = 320, 240, 93
            observer_cam_pos = np.array([0.0, 0.23, 1.33])
            observer_cam_forward = np.array([0, -1, -1.02])
            observer_cam_left = np.array([1, 0, 0])
        elif self.third_person_view == "random":
            # Per-episode random pose on a partial sphere around the tabletop.
            observer_w, observer_h, observer_fovy_deg = 640, 480, 70
            observer_cam_pos, observer_cam_forward, observer_cam_left = (
                self._sample_random_observer_pose()
            )
        else:
            # "observer": fixed third-person view facing the robot *across*
            # the table, so the table sits between the camera and the robot.
            # Robot base is around y=-0.65, table centre at y=0, table depth
            # ~0.7m (front edge at y=+0.35). We stand just past the front
            # edge, around head height, with a gentle downward tilt so the
            # tabletop + any objects + the robot's upper body all sit in
            # frame while the camera is mostly horizontal (not a heavy
            # top-down pitch).
            observer_w, observer_h, observer_fovy_deg = 1280, 720, 70
            observer_cam_pos = np.array([0.0, 0.65, 1.20])
            observer_cam_forward = np.array([0.0, -1.0, -0.4])
            # Camera-local +Y ("left"). Keep it as world +x so up = fwd x left
            # ends up with +z component (image stays right-side up). With this
            # convention, screen-left = world +x, i.e. the robot's left arm
            # (x>0) lands on the left side of the frame -- the natural
            # "face-to-face, mirror" reading.
            observer_cam_left = np.array([1.0, 0.0, 0.0])
        self.observer_camera = scene.add_camera(
            name="observer_camera",
            width=observer_w,
            height=observer_h,
            fovy=np.deg2rad(observer_fovy_deg),
            near=near,
            far=far,
        )
        observer_cam_forward = observer_cam_forward / np.linalg.norm(observer_cam_forward)
        observer_cam_left = observer_cam_left / np.linalg.norm(observer_cam_left)
        observer_up = np.cross(observer_cam_forward, observer_cam_left)
        observer_mat44 = np.eye(4)
        observer_mat44[:3, :3] = np.stack([observer_cam_forward, observer_cam_left, observer_up], axis=1)
        observer_mat44[:3, 3] = observer_cam_pos
        self.observer_camera.entity.set_pose(sapien.Pose(observer_mat44))

        # world pcd camera
        self.world_camera1 = scene.add_camera(
            name="world_camera1",
            width=640,
            height=480,
            fovy=np.deg2rad(50),
            near=near,
            far=far,
        )
        world_cam_pos = np.array([0.4, -0.4, 1.6])
        world_cam_forward = np.array([-1, 1, -1.4])
        world_cam_left = np.array([-1, -1, 0])
        world_cam_up = np.cross(world_cam_forward, world_cam_left)
        world_cam_mat44 = np.eye(4)
        world_cam_mat44[:3, :3] = np.stack([world_cam_forward, world_cam_left, world_cam_up], axis=1)
        world_cam_mat44[:3, 3] = world_cam_pos
        self.world_camera1.entity.set_pose(sapien.Pose(world_cam_mat44))

        self.world_camera2 = scene.add_camera(
            name="world_camera1",
            width=640,
            height=480,
            fovy=np.deg2rad(50),
            near=near,
            far=far,
        )
        world_cam_pos = np.array([-0.4, -0.4, 1.6])
        world_cam_forward = np.array([1, 1, -1.4])
        world_cam_left = np.array([-1, 1, 0])
        world_cam_up = np.cross(world_cam_forward, world_cam_left)
        world_cam_mat44 = np.eye(4)
        world_cam_mat44[:3, :3] = np.stack([world_cam_forward, world_cam_left, world_cam_up], axis=1)
        world_cam_mat44[:3, 3] = world_cam_pos
        self.world_camera2.entity.set_pose(sapien.Pose(world_cam_mat44))

    def update_picture(self):
        # camera
        if self.collect_wrist_camera:
            self.left_camera.take_picture()
            self.right_camera.take_picture()

        for camera in self.static_camera_list:
            camera.take_picture()

        # ================================= sensor camera =================================
        # self.head_sensor.take_picture()
        # self.head_sensor.compute_depth()

    def update_wrist_camera(self, left_pose, right_pose):
        """
        Update rendering to refresh the camera's RGBD information
        (rendering must be updated even when disabled, otherwise data cannot be collected).
        """
        if self.collect_wrist_camera:
            self.left_camera.entity.set_pose(left_pose)
            self.right_camera.entity.set_pose(right_pose)

    def get_config(self) -> dict:
        res = {}

        def _get_config(camera):
            camera_intrinsic_cv = camera.get_intrinsic_matrix()
            camera_extrinsic_cv = camera.get_extrinsic_matrix()
            camera_model_matrix = camera.get_model_matrix()
            return {
                "intrinsic_cv": camera_intrinsic_cv,
                "extrinsic_cv": camera_extrinsic_cv,
                "cam2world_gl": camera_model_matrix,
            }

        if self.collect_wrist_camera:
            res["left_camera"] = _get_config(self.left_camera)
            res["right_camera"] = _get_config(self.right_camera)

        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = _get_config(camera)
            else:
                res[camera_name] = _get_config(camera)
        # ================================= sensor camera =================================
        # res['head_sensor'] = res['head_camera']
        # print(res)
        return res

    def get_rgb(self) -> dict:
        rgba = self.get_rgba()
        rgb = {}
        for camera_name, camera_data in rgba.items():
            rgb[camera_name] = {}
            rgb[camera_name]["rgb"] = camera_data["rgba"][:, :, :3]  # Exclude alpha channel
        return rgb
    
    # Get Camera RGBA
    def get_rgba(self) -> dict:

        def _get_rgba(camera):
            camera_rgba = camera.get_picture("Color")
            camera_rgba_img = (camera_rgba * 255).clip(0, 255).astype("uint8")
            return camera_rgba_img

        # ================================= sensor camera =================================
        # def _get_sensor_rgba(sensor):
        #     camera_rgba = sensor.get_rgb()
        #     camera_rgba_img = (camera_rgba * 255).clip(0, 255).astype("uint8")[:,:,:3]
        #     return camera_rgba_img

        res = {}

        if self.collect_wrist_camera:
            res["left_camera"] = {}
            res["right_camera"] = {}
            res["left_camera"]["rgba"] = _get_rgba(self.left_camera)
            res["right_camera"]["rgba"] = _get_rgba(self.right_camera)

        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = {}
                    res[camera_name]["rgba"] = _get_rgba(camera)
            else:
                res[camera_name] = {}
                res[camera_name]["rgba"] = _get_rgba(camera)
        # ================================= sensor camera =================================
        # res['head_sensor']['rgb'] = _get_sensor_rgba(self.head_sensor)

        return res

    def _sample_random_observer_pose(self):
        """
        Sample a per-episode random pose for the third-person observer camera.

        The camera position is drawn on a partial sphere around the tabletop
        centre (``random_view_look_at`` + ``table_z_bias``) with:
            * azimuth  in ``random_view_azimuth_range``   (deg; 0 = robot front)
            * elevation in ``random_view_elevation_range`` (deg; above horizon)
            * radius   in ``random_view_radius_range``    (metres)
        Then the camera looks towards the tabletop centre perturbed by a small
        uniform jitter (``random_view_look_at_jitter``) to avoid always framing
        the scene dead-centre.

        Returns three numpy vectors (``pos``, ``forward``, ``left``) consumed by
        the generic observer pose-building code in ``load_camera``.
        """
        # --- sample spherical coordinates
        az_lo, az_hi = self.random_view_azimuth_range
        el_lo, el_hi = self.random_view_elevation_range
        r_lo, r_hi = self.random_view_radius_range
        azimuth = np.deg2rad(np.random.uniform(az_lo, az_hi))
        elevation = np.deg2rad(np.random.uniform(el_lo, el_hi))
        radius = np.random.uniform(r_lo, r_hi)

        # Tabletop centre (follows random_table_height via table_z_bias).
        centre = self.random_view_look_at.copy()
        centre[2] += self.table_z_bias

        # azimuth = 0 -> camera stands on the +y side (opposite the robot,
        # looking across the tabletop towards the robot, which sits at -y).
        # azimuth = +pi/2 -> +x side, -pi/2 -> -x side.
        offset = np.array([
            np.sin(azimuth) * np.cos(elevation),
            np.cos(azimuth) * np.cos(elevation),
            np.sin(elevation),
        ]) * radius
        pos = centre + offset

        # Jittered look-at point, then forward vector.
        j = self.random_view_look_at_jitter
        look_at = centre + np.random.uniform(-j, j, size=3)
        forward = look_at - pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            forward = np.array([0.0, 1.0, -0.5])
        else:
            forward = forward / forward_norm

        # Build a "left" axis orthogonal to forward, using world up as reference.
        world_up = np.array([0.0, 0.0, 1.0])
        # If forward is nearly parallel to world_up, fall back to world +x.
        if abs(np.dot(forward, world_up)) > 0.999:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = world_up
        left = np.cross(ref, forward)
        left_norm = np.linalg.norm(left)
        if left_norm < 1e-8:
            left = np.array([1.0, 0.0, 0.0])
        else:
            left = left / left_norm
        return pos, forward, left

    def get_observer_rgb(self) -> dict:
        self.observer_camera.take_picture()

        def _get_rgb(camera):
            camera_rgba = camera.get_picture("Color")
            camera_rgb_img = (camera_rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
            return camera_rgb_img

        return _get_rgb(self.observer_camera)

    def get_video_frame(self, head_rgb=None):
        """
        Return a single RGB frame (H, W, 3) that will be used to render the final mp4.
        The source is selected by ``self.third_person_view``:
          - "default"  : head_camera (caller may pass its already-rendered rgb to save a re-render).
          - "observer" : static observer_camera (third-person).
          - "random"   : currently aliased to "observer"; reserved for future per-episode
                         or per-frame random camera pose.
        """
        if self.third_person_view == "default":
            if head_rgb is not None:
                return head_rgb
            if self.collect_head_camera and self.head_camera_id is not None:
                head_cam = self.static_camera_list[self.head_camera_id]
                rgba = head_cam.get_picture("Color")
                return (rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
            # fallback: observer
            return self.get_observer_rgb()
        # "observer" or "random" (random reuses observer for now)
        return self.get_observer_rgb()

    # Get Camera Segmentation
    def get_segmentation(self, level="mesh") -> dict:

        def _get_segmentation(camera, level="mesh"):
            # visual_id is the unique id of each visual shape
            seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
            colormap = sorted(set(ImageColor.colormap.values()))
            color_palette = np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)
            if level == "mesh":
                label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
            elif level == "actor":
                label0_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
            return color_palette[label0_image]

        res = {
            # 'left_camera':{},
            # 'right_camera':{}
        }

        if self.collect_wrist_camera:
            res["left_camera"] = {}
            res["right_camera"] = {}
            res["left_camera"][f"{level}_segmentation"] = _get_segmentation(self.left_camera, level=level)
            res["right_camera"][f"{level}_segmentation"] = _get_segmentation(self.right_camera, level=level)

        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = {}
                    res[camera_name][f"{level}_segmentation"] = _get_segmentation(camera, level=level)
            else:
                res[camera_name] = {}
                res[camera_name][f"{level}_segmentation"] = _get_segmentation(camera, level=level)
        return res

    # Get Camera Depth
    def get_depth(self) -> dict:

        def _get_depth(camera):
            position = camera.get_picture("Position")
            depth = -position[..., 2]
            depth_image = (depth * 1000.0).astype(np.float64)
            return depth_image

        def _get_sensor_depth(sensor):
            depth = sensor.get_depth()
            depth = (depth * 1000.0).astype(np.float64)
            return depth

        res = {}
        rgba = self.get_rgba()

        if self.collect_wrist_camera:
            res["left_camera"] = {}
            res["right_camera"] = {}
            res["left_camera"]["depth"] = _get_depth(self.left_camera)
            res["right_camera"]["depth"] = _get_depth(self.right_camera)
            res["left_camera"]["depth"] *= rgba["left_camera"]["rgba"][:, :, 3] / 255
            res["right_camera"]["depth"] *= rgba["right_camera"]["rgba"][:, :, 3] / 255
        
        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = {}
                    res[camera_name]["depth"] = _get_depth(camera)
                    res[camera_name]["depth"] *= rgba[camera_name]["rgba"][:, :, 3] / 255
            else:
                res[camera_name] = {}
                res[camera_name]["depth"] = _get_depth(camera)
                res[camera_name]["depth"] *= rgba[camera_name]["rgba"][:, :, 3] / 255
        # res['head_sensor']['depth'] = _get_sensor_depth(self.head_sensor)

        return res

    # Get World PointCloud
    def get_world_pcd(self):
        self.world_camera1.take_picture()
        self.world_camera2.take_picture()

        def _get_camera_pcd(camera, color=True):
            rgba = camera.get_picture_cuda("Color").torch()  # [H, W, 4]
            position = camera.get_picture_cuda("Position").torch()
            model_matrix = camera.get_model_matrix()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_matrix = torch.tensor(model_matrix, dtype=torch.float32).to(device)

            # Extract valid three-dimensional points and corresponding color data.
            valid_mask = position[..., 3] < 1
            points_opengl = position[..., :3][valid_mask]
            points_color = rgba[valid_mask][:, :3]
            # Transform into the world coordinate system.
            points_world = (torch.bmm(
                points_opengl.view(1, -1, 3),
                model_matrix[:3, :3].transpose(0, 1).view(-1, 3, 3),
            ).squeeze(1) + model_matrix[:3, 3])

            # Format color data.
            points_color = torch.clamp(points_color, 0, 1)
            points_world = points_world.squeeze(0)

            # Convert the tensor back to a NumPy array for use with Open3D.
            points_world_np = points_world.cpu().numpy()
            points_color_np = points_color.cpu().numpy()
            # print(points_world_np.shape, points_color_np.shape)

            res_pcd = (np.hstack((points_world_np, points_color_np)) if color else points_world_np)
            return res_pcd

        pcd1 = _get_camera_pcd(self.world_camera1, color=True)
        pcd2 = _get_camera_pcd(self.world_camera2, color=True)
        res_pcd = np.vstack((pcd1, pcd2))

        return res_pcd
        pcd_array, index = fps(res_pcd[:, :3], 2000)
        index = index.detach().cpu().numpy()[0]

        return pcd_array

    # Get Camera PointCloud
    def get_pcd(self, if_combine=False):

        def _get_camera_pcd(camera, point_num=0):
            rgba = camera.get_picture_cuda("Color").torch()  # [H, W, 4]
            position = camera.get_picture_cuda("Position").torch()
            model_matrix = camera.get_model_matrix()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_matrix = torch.tensor(model_matrix, dtype=torch.float32).to(device)

            # Extract valid three-dimensional points and corresponding color data.
            valid_mask = position[..., 3] < 1
            points_opengl = position[..., :3][valid_mask]
            points_color = rgba[valid_mask][:, :3]
            # Transform into the world coordinate system.
            points_world = (torch.bmm(
                points_opengl.view(1, -1, 3),
                model_matrix[:3, :3].transpose(0, 1).view(-1, 3, 3),
            ).squeeze(1) + model_matrix[:3, 3])

            # Format color data.
            points_color = torch.clamp(points_color, 0, 1)

            points_world = points_world.squeeze(0)

            # If crop is needed
            if self.pcd_crop:
                min_bound = torch.tensor(self.pcd_crop_bbox[0], dtype=torch.float32).to(device)
                max_bound = torch.tensor(self.pcd_crop_bbox[1], dtype=torch.float32).to(device)
                inside_bounds_mask = (points_world.squeeze(0) >= min_bound).all(dim=1) & (points_world.squeeze(0)
                                                                                          <= max_bound).all(dim=1)
                points_world = points_world[inside_bounds_mask]
                points_color = points_color[inside_bounds_mask]

            # Convert the tensor back to a NumPy array for use with Open3D.
            points_world_np = points_world.cpu().numpy()
            points_color_np = points_color.cpu().numpy()

            if point_num > 0:
                # points_world_np,index = fps(points_world_np,point_num)
                index = index.detach().cpu().numpy()[0]
                points_color_np = points_color_np[index, :]

            return np.hstack((points_world_np, points_color_np))

        if self.head_camera_id is None:
            print("No head camera in static camera list, pointcloud save error!")
            return None

        combined_pcd = np.array([])

        # Merge pointcloud
        if if_combine:
            # combined_pcd = np.vstack((head_pcd , left_pcd , right_pcd, front_pcd))
            if self.collect_wrist_camera:
                combined_pcd = np.vstack((
                    _get_camera_pcd(self.left_camera),
                    _get_camera_pcd(self.right_camera),
                ))
            for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
                if camera_name == "head_camera":
                    if self.collect_head_camera:
                        combined_pcd = np.vstack((combined_pcd, _get_camera_pcd(camera)))
                else:
                    combined_pcd = np.vstack((combined_pcd, _get_camera_pcd(camera)))
        elif self.collect_head_camera:
            combined_pcd = _get_camera_pcd(self.static_camera_list[self.head_camera_id])
        
        def pad_array(numpy_array, target_num):
            current_num = numpy_array.shape[0]
            if current_num < target_num:
                if current_num == 0:
                    numpy_array = np.zeros((target_num, 6))
                else:
                    pad_num = target_num - current_num
                    padding = np.zeros((pad_num, 6))
                    numpy_array = np.vstack([numpy_array, padding])
            return numpy_array

        combined_pcd = pad_array(combined_pcd, self.pcd_down_sample_num)

        pcd_array, index = combined_pcd[:, :3], np.array(range(len(combined_pcd)))

        if self.pcd_down_sample_num > 0:
            pcd_array, index = fps(combined_pcd[:, :3], self.pcd_down_sample_num)
            index = index.detach().cpu().numpy()[0]

        return combined_pcd[index]
