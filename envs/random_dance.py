"""
random_dance
============
A utility "task" whose sole purpose is **data augmentation**: no goal is solved,
the robot simply performs random joint-space motions ("dances") while the scene
randomization pipeline (cluttered table, random light, random background ...)
produces diverse visual observations.

Two-stage integration with ``script/collect_data.py``:

* ``need_plan=True`` (seed pass)
    - Draw random target joint vectors for left / right arms plus random gripper
      commands for a fixed number of dance steps.
    - Execute them via the SAPIEN drive-target interface and record observations
      through ``_take_picture``.
    - The sequence of random targets is stored in ``self.left_joint_path`` /
      ``self.right_joint_path`` so that the second pass can replay it.

* ``need_plan=False`` (data pass)
    - Replay the previously recorded targets, producing an identical trajectory
      but with full observation logging.

``check_success`` always returns ``True``: there is nothing to "achieve".
"""

from ._base_task import Base_Task
from .utils import *
import numpy as np
import math


class random_dance(Base_Task):
    # Default dance parameters. They can be overridden from the task config file
    # under the top-level key ``random_dance`` (optional).
    DEFAULT_N_STEPS = 30            # number of random key-frames per episode
    DEFAULT_ARM_DELTA = 0.35        # max |delta| (rad) around home state per joint
    DEFAULT_GRIPPER_TOGGLE_P = 0.3  # probability of changing gripper state each step
    DEFAULT_HOLD_SUBSTEPS = 40      # physics substeps spent holding each key-frame
    DEFAULT_SAVE_EVERY = 5          # save an observation every N physics substeps

    # Independent "dance home" used as the centre of the random joint sampling.
    # The embodiment's own ``homestate`` (e.g. all-zeros for aloha-agilex) makes
    # both arms sit vertically along the body midline, which looks cramped on
    # the observer camera. Here we pre-pose the arms with a mild shoulder abduct
    # + shoulder lift + elbow bend so the two arms naturally spread open and
    # reach *forward* (toward the observer camera, +y world direction).
    # Joint order: [j1, j2, j3, j4, j5, j6] (6-DoF arm).
    #  j1: shoulder yaw   (positive -> outward abduction on each side)
    #  j2: shoulder pitch (positive -> arm swings forward / toward +y)
    #  j3: elbow          (positive -> bend; smaller -> more forward reach)
    #  j4,j5,j6: wrist
    # Tweaking to lean further forward:
    #   - increase  j2 (shoulder pitch)   -> whole arm rotates forward
    #   - decrease  j3 (elbow bend)       -> straighter -> longer reach
    DEFAULT_LEFT_HOME = [0.30, 0.60, 0.40, 0.0, 0.80, 0.0]
    DEFAULT_RIGHT_HOME = [-0.30, 0.60, 0.40, 0.0, 0.80, 0.0]

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)
        # Optional per-task overrides from task_config.
        dance_cfg = kwargs.get("random_dance", {}) or {}
        self._dance_n_steps = int(dance_cfg.get("n_steps", self.DEFAULT_N_STEPS))
        self._dance_arm_delta = float(dance_cfg.get("arm_delta", self.DEFAULT_ARM_DELTA))
        self._dance_gripper_toggle_p = float(dance_cfg.get("gripper_toggle_p", self.DEFAULT_GRIPPER_TOGGLE_P))
        self._dance_hold_substeps = int(dance_cfg.get("hold_substeps", self.DEFAULT_HOLD_SUBSTEPS))
        self._dance_save_every = max(1, int(dance_cfg.get("save_every", self.DEFAULT_SAVE_EVERY)))

        # ``mode`` selects the keyframe source:
        #   "joint"      (default): legacy random joint-space sampling around home
        #   "task"       : random end-effector (Cartesian) sampling per arm,
        #                  each arm restricted to its own half of the table;
        #                  keyframe orientation is inherited from the current EE
        #                  pose so the planner only has to solve for position.
        #   "ik_debug"   : single-shot IK test at fixed target(s), logs joint
        #                  angles. Used to validate reach/mapping.
        #   "home_debug" : drive to the configured ``left_home``/``right_home``
        #                  and hold. Useful for iterating on the dance home
        #                  joint values without the IK step moving the arms
        #                  off the pose you are trying to inspect.
        self._dance_mode = str(dance_cfg.get("mode", "joint")).lower()
        # Hover height (metres) above the table top used by ik_debug mode.
        self._dance_ik_debug_hover = float(dance_cfg.get("ik_debug_hover", 0.20))
        # Forward offset along world +y (metres), i.e. "toward the observer
        # camera". 0 = table-half centre (y=0). Positive values push the IK
        # target closer to the camera; max reasonable is ~0.30 before the
        # target leaves the tabletop (table width is 0.7 -> front edge at
        # y=+0.35).
        self._dance_ik_debug_forward = float(dance_cfg.get("ik_debug_forward", 0.20))
        # Lateral spread (metres) along world x for ik_debug targets. The left
        # arm aims at x = -lateral, the right arm at x = +lateral (both relative
        # to ``self.table_xy_bias``). Increase to spread the arms further apart
        # (good for showing both arms cleanly in the observer view); decrease
        # to bring the grippers closer to the midline. Table is 1.2 m wide in
        # x, so anything beyond ~0.55 risks IK failure / off-table targets.
        self._dance_ik_debug_lateral = float(dance_cfg.get("ik_debug_lateral", 0.30))

        # home_debug-only knob: force grippers to this normalised value
        # (0 = close, 1 = open) while holding the dance home pose. None
        # means "leave them at whatever they are (usually 0)".
        hd_grip = dance_cfg.get("home_debug_gripper", None)
        self._dance_home_debug_gripper = (
            None if hd_grip is None else float(hd_grip)
        )

        # Task-space sampling box ('task' mode only). Each arm samples a
        # target inside its own [xmin, xmax] x [ymin, ymax] x [zmin, zmax]
        # in world coordinates. Defaults are conservative: they cover
        # roughly the arm's half of the table, stay a small margin inside
        # the tabletop edges, and hover 8-25 cm above the table top. All
        # values overridable per-task in the yaml.
        task_cfg = dance_cfg.get("task", {}) or {}
        self._dance_task_left_x = tuple(task_cfg.get("left_x_range", [-0.55, -0.05]))
        self._dance_task_right_x = tuple(task_cfg.get("right_x_range", [0.05, 0.55]))
        self._dance_task_y = tuple(task_cfg.get("y_range", [-0.30, 0.25]))
        # z is relative to the *table top* so it tracks ``table_z_bias``.
        self._dance_task_z_rel = tuple(task_cfg.get("z_rel_range", [0.08, 0.25]))
        # How many times to retry when a sampled target is IK-unreachable.
        self._dance_task_max_retries = int(task_cfg.get("max_retries", 8))
        # Max orientation perturbation (deg) around the home EE orientation.
        # Each waypoint right-multiplies the home quaternion by a random
        # axis-angle within this cone, so 0 = fixed orientation (old
        # behaviour), 15 = mild wrist wobble, 40+ = quite expressive.
        # IK success drops off quickly past ~45 deg.
        self._dance_task_pose_perturb_deg = float(task_cfg.get("pose_perturb_deg", 20.0))

        # Per-joint Cartesian-bypass offset, applied AFTER IK and AFTER
        # unwrap_toward() but BEFORE the max_joint_step / span check.
        # Each entry [low, high] is a uniform sampling range (radians)
        # added to the corresponding joint of the IK solution.
        #
        # Why this exists. IK in task mode only constrains the EE
        # pose, so the wrist joints (j4..j6) are free to slip into
        # whatever value minimises the IK cost. In practice that
        # leaves j6 nearly untouched (std ~0.12 over 100 episodes;
        # see docs/dance_home_tuning.md) and pulls j4 toward one
        # side. This knob lets the user broaden the joint-space
        # distribution without re-doing the whole IK pipeline.
        #
        # Side-effects:
        #   * j6 is essentially gripper twist about its own axis -- it
        #     does not move the EE, so perturbing it has zero impact
        #     on the EE position.
        #   * j4/j5 do shift the EE by a few cm and rotate the EE
        #     pose by a few degrees. That is fine for the random
        #     dance task (we are after visual coverage, not precise
        #     placement) but obviously inappropriate for any task
        #     that grasps an object.
        #
        # Default: all zeros => disabled, identical to legacy behaviour.
        # Format in yaml: a list of 6 [lo, hi] pairs, one per joint.
        jpp_raw = task_cfg.get("joint_post_perturb", None)
        if jpp_raw is None:
            self._dance_task_joint_post_perturb = None
        else:
            jpp = np.asarray(jpp_raw, dtype=np.float64)
            if jpp.shape != (6, 2):
                raise ValueError(
                    f"random_dance.task.joint_post_perturb must be a list of "
                    f"6 [lo, hi] pairs, got shape {jpp.shape}")
            # Normalise so [0, 0] entries are a no-op and lo <= hi.
            jpp = np.stack([np.minimum(jpp[:, 0], jpp[:, 1]),
                            np.maximum(jpp[:, 0], jpp[:, 1])], axis=1)
            if np.allclose(jpp, 0.0):
                self._dance_task_joint_post_perturb = None
            else:
                self._dance_task_joint_post_perturb = jpp
                # Friendly summary so the user can confirm at startup.
                rngs = ", ".join(
                    f"j{i+1}=[{jpp[i,0]:+.2f},{jpp[i,1]:+.2f}]"
                    for i in range(6) if not (jpp[i,0] == 0.0 and jpp[i,1] == 0.0)
                )
                print(f"[random_dance task] joint_post_perturb active: {rngs}")
        # Upper bound on the single-joint step size between consecutive
        # IK waypoints, in radians. Curobo often returns equivalent
        # configurations differing by >1 rad on redundant joints even when
        # the EE target moves only slightly; without a cap the spline then
        # has to cover, say, 2 rad in ``hold_substeps`` physics ticks,
        # which renders as a whip-fast arm sweep. Whenever a fresh IK
        # solution exceeds this cap, we retry (sampling a new EE target
        # / new wrist perturbation).
        self._dance_task_max_joint_step = float(task_cfg.get("max_joint_step", 1.0))
        # To keep visual *peak* angular speed roughly constant across
        # waypoints that happen to have very different joint-space span,
        # we stretch the playback segment duration proportionally when a
        # waypoint's joint delta exceeds this reference. Duration is
        # clamped to [1x, stretch_cap x] of ``hold_substeps``.
        self._dance_task_speed_ref_rad = float(task_cfg.get("speed_ref_rad", 0.8))
        self._dance_task_stretch_cap = float(task_cfg.get("stretch_cap", 3.0))

        # Named presets. These are thin shortcuts that overwrite the
        # keyframe count / per-segment hold so you can switch between
        # "dance personalities" from the yaml without re-tuning every
        # knob. A preset only overrides the fields listed in its dict;
        # anything else (task box, pose_perturb_deg, ...) is inherited
        # from the explicit yaml values above. Pick "default" (or leave
        # ``preset`` unset) to keep the legacy behaviour.
        #
        #   default : 30 keyframes, 80 substeps/segment  (baseline)
        #   slow    : 10 keyframes, 160 substeps/segment (half the key
        #             poses, each traversed twice as slowly -> calmer,
        #             more deliberate motion; same wall-clock per episode)
        _PRESETS = {
            "default": {},
            "slow":    {"n_steps": 10, "hold_substeps": 160},
        }
        preset_name = str(dance_cfg.get("preset", "default")).lower()
        preset_over = _PRESETS.get(preset_name)
        if preset_over is None:
            print(f"[random_dance] WARNING: unknown preset '{preset_name}', "
                  f"falling back to 'default'. Known presets: {list(_PRESETS)}")
            preset_name = "default"
            preset_over = {}
        # Only overwrite when the user did NOT explicitly set the field
        # in the yaml -- that way a yaml value always wins over a preset
        # default, so you can e.g. pick `slow` but bump `n_steps` to 12.
        if "n_steps" not in dance_cfg and "n_steps" in preset_over:
            self._dance_n_steps = int(preset_over["n_steps"])
        if "hold_substeps" not in dance_cfg and "hold_substeps" in preset_over:
            self._dance_hold_substeps = int(preset_over["hold_substeps"])
        print(f"[random_dance] preset='{preset_name}' -> "
              f"n_steps={self._dance_n_steps}, "
              f"hold_substeps={self._dance_hold_substeps}")

        # Dance home: explicit yaml override > built-in default > embodiment homestate.
        n_left = len(self.robot.left_arm_joints)
        n_right = len(self.robot.right_arm_joints)
        left_home_cfg = dance_cfg.get("left_home", None)
        right_home_cfg = dance_cfg.get("right_home", None)
        if left_home_cfg is None:
            left_home_cfg = self.DEFAULT_LEFT_HOME if n_left == len(self.DEFAULT_LEFT_HOME) \
                else list(self.robot.left_homestate)
        if right_home_cfg is None:
            right_home_cfg = self.DEFAULT_RIGHT_HOME if n_right == len(self.DEFAULT_RIGHT_HOME) \
                else list(self.robot.right_homestate)
        self._dance_left_home = np.asarray(left_home_cfg, dtype=np.float64)
        self._dance_right_home = np.asarray(right_home_cfg, dtype=np.float64)

    # ------------------------------------------------------------------
    # Scene objects
    # ------------------------------------------------------------------
    def load_actors(self):
        """
        The cluttered_table randomisation (enabled via
        ``domain_randomization.cluttered_table: true`` in the task config) will
        automatically spawn random objects on the table, so we only need to
        reserve a generous prohibited area in front of the robot to keep the
        spawning sane -- we don't want tiny objects directly under the end
        effectors when the arms dance around.
        """
        # A rectangular band right in front of the robot stays empty so that
        # arms have free space to move. Values are expressed in the table frame
        # (robot base is roughly at y = -0.65).
        self.prohibited_area.append([-0.30, -0.15, 0.30, 0.15])

    # ------------------------------------------------------------------
    # Dance logic
    # ------------------------------------------------------------------
    def _sample_arm_targets(self, home_arm, arm_tag):
        """Sample a random target joint vector around the home state.

        The raw uniform sample in ``[home-delta, home+delta]`` is additionally
        clipped to each joint's physical limit (with a small safety margin),
        so that large ``arm_delta`` values still produce reachable -- and
        therefore actually moving -- targets.
        """
        home_arm = np.asarray(home_arm, dtype=np.float64)
        delta = self._dance_arm_delta
        raw = home_arm + np.random.uniform(-delta, delta, size=home_arm.shape)

        joint_lst = self.robot.left_arm_joints if arm_tag == "left" else self.robot.right_arm_joints
        limits_low = np.full_like(home_arm, -np.inf)
        limits_high = np.full_like(home_arm, np.inf)
        try:
            for j_idx, joint in enumerate(joint_lst):
                lim = joint.get_limits()
                # `get_limits()` returns shape (1, 2); robust to ndarray / list.
                lim = np.asarray(lim).reshape(-1, 2)
                if lim.size >= 2 and np.all(np.isfinite(lim[0])):
                    limits_low[j_idx] = lim[0, 0]
                    limits_high[j_idx] = lim[0, 1]
        except Exception:
            pass
        # Leave a 5% safety margin inside the hard limits.
        span = np.where(np.isfinite(limits_high - limits_low),
                        (limits_high - limits_low) * 0.05, 0.0)
        safe_low = limits_low + span
        safe_high = limits_high - span
        return np.clip(raw, safe_low, safe_high)

    def _drive_spline(self, left_waypoints, right_waypoints,
                      left_grips, right_grips):
        """Drive both arms through their full list of waypoints using a
        Catmull-Rom spline (C^1 continuous) so consecutive segments are
        stitched seamlessly -- no start-stop-start-stop stuttering at the
        key-frame boundaries.

        Inputs
        ------
        left_waypoints, right_waypoints : list of np.ndarray, each shape (ndof,)
            Sequence starting with the current pose and followed by every
            sampled keyframe. ``len(left_waypoints) == N + 1`` where ``N``
            is the number of random keyframes.
        left_grips, right_grips : list of float
            Same length as ``*_waypoints``. Gripper references are linearly
            interpolated within each segment -- no spline -- since gripper
            open/close is a low-DoF binary-ish signal that doesn't need C^1
            continuity.

        For each of the N segments we run ``hold_substeps`` physics steps,
        sampling the spline at ``t in [0, 1]``. Because the spline uses
        Catmull-Rom tangents at every interior waypoint, the robot's joint
        reference velocity is continuous across the key-frame boundaries.

        At the two ends we duplicate the first / last waypoint to form
        "virtual" neighbours, which makes the tangents at the first and last
        real waypoint equal to zero -- i.e. the dance starts and ends at
        rest, avoiding an initial velocity jump.
        """
        left_waypoints = [np.asarray(p, dtype=np.float64) for p in left_waypoints]
        right_waypoints = [np.asarray(p, dtype=np.float64) for p in right_waypoints]
        left_grips = [float(g) for g in left_grips]
        right_grips = [float(g) for g in right_grips]

        assert len(left_waypoints) == len(right_waypoints) == \
               len(left_grips) == len(right_grips), \
               "spline driver expects matched-length waypoint / gripper lists"
        n_waypoints = len(left_waypoints)
        if n_waypoints < 2:
            return  # nothing to play

        # Simulation timestep (s). Fall back to the default if unavailable.
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            dt = 1.0 / 250.0

        steps_per_seg_base = max(1, int(self._dance_hold_substeps))
        # Per-segment substep budget. We stretch segments whose joint-space
        # span exceeds ``speed_ref_rad`` so that the peak angular speed
        # seen in the video stays roughly constant (see settings in
        # ``setup_demo``). Only used in task mode -- joint-mode waypoints
        # are already tightly bounded by ``arm_delta`` so they never need
        # stretching.
        ref_rad = float(getattr(self, "_dance_task_speed_ref_rad", 0.0))
        stretch_cap = float(getattr(self, "_dance_task_stretch_cap", 1.0))

        # Pad with duplicated ends -> zero-velocity boundary tangents.
        left_padded = [left_waypoints[0]] + left_waypoints + [left_waypoints[-1]]
        right_padded = [right_waypoints[0]] + right_waypoints + [right_waypoints[-1]]

        # Catmull-Rom (uniform, tau=0.5) position and velocity at t in [0,1]:
        #   h00(t) = 2t^3 - 3t^2 + 1
        #   h10(t) = t^3 - 2t^2 + t
        #   h01(t) = -2t^3 + 3t^2
        #   h11(t) = t^3 - t^2
        # with tangent m_i = (p_{i+1} - p_{i-1}) / 2.  Return values are in
        # *normalised* units (per segment). To get rad/s we divide by
        # the *actual* segment time (which may be stretched).
        def eval_segment(padded, i, t):
            p_prev = padded[i]       # p_{i-1}
            p0 = padded[i + 1]       # p_i
            p1 = padded[i + 2]       # p_{i+1}
            p_next = padded[i + 3]   # p_{i+2}
            m0 = 0.5 * (p1 - p_prev)
            m1 = 0.5 * (p_next - p0)
            t2 = t * t
            t3 = t2 * t
            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 = t3 - t2
            pos = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
            # d/dt of the Hermite bases
            dh00 = 6.0 * t2 - 6.0 * t
            dh10 = 3.0 * t2 - 4.0 * t + 1.0
            dh01 = -6.0 * t2 + 6.0 * t
            dh11 = 3.0 * t2 - 2.0 * t
            vel_per_seg = dh00 * p0 + dh10 * m0 + dh01 * p1 + dh11 * m1
            return pos, vel_per_seg

        n_segments = n_waypoints - 1
        for seg in range(n_segments):
            # Decide this segment's playback length. We look at the
            # larger of the two arms' joint-space delta for this segment
            # so that whichever arm moves the most drives the pacing.
            if ref_rad > 0.0:
                l_span = float(np.max(np.abs(left_waypoints[seg + 1] - left_waypoints[seg])))
                r_span = float(np.max(np.abs(right_waypoints[seg + 1] - right_waypoints[seg])))
                span = max(l_span, r_span)
                stretch = max(1.0, min(stretch_cap, span / ref_rad))
            else:
                stretch = 1.0
            steps_per_seg = max(1, int(round(steps_per_seg_base * stretch)))
            seg_time = steps_per_seg * dt  # seconds for THIS segment

            g_prev_l = left_grips[seg]
            g_next_l = left_grips[seg + 1]
            g_prev_r = right_grips[seg]
            g_next_r = right_grips[seg + 1]
            for sub in range(steps_per_seg):
                # t goes (1/steps_per_seg, 2/steps_per_seg, ..., 1) so we
                # always include the exact endpoint on the last substep.
                t = (sub + 1) / float(steps_per_seg)

                l_pos, l_vel_norm = eval_segment(left_padded, seg, t)
                r_pos, r_vel_norm = eval_segment(right_padded, seg, t)
                l_vel = l_vel_norm / seg_time
                r_vel = r_vel_norm / seg_time
                # Grippers: plain linear interpolation inside the segment.
                l_grip = g_prev_l + (g_next_l - g_prev_l) * t
                r_grip = g_prev_r + (g_next_r - g_prev_r) * t

                self.robot.set_arm_joints(l_pos, l_vel, "left")
                self.robot.set_arm_joints(r_pos, r_vel, "right")
                self.robot.set_gripper(float(l_grip), "left")
                self.robot.set_gripper(float(r_grip), "right")
                self.scene.step()
                self._update_render()
                if self.render_freq and hasattr(self, "viewer"):
                    try:
                        self.viewer.render()
                    except Exception:
                        pass
                if sub % self._dance_save_every == 0:
                    self._take_picture()
            # One snapshot at the end of every segment (keeps coverage
            # comparable to the previous _drive_to_keyframe implementation).
            self._take_picture()

    # ------------------------------------------------------------------
    # Task-space (Cartesian) sampling
    # ------------------------------------------------------------------
    @staticmethod
    def _random_quat_perturb(rng, max_angle_deg):
        """Return a unit quaternion (w, x, y, z) uniform on an SO(3) cap of
        half-angle ``max_angle_deg`` (degrees), i.e. a "small random
        rotation" centred on identity. If ``max_angle_deg <= 0`` we return
        the identity quaternion.

        We sample the axis uniformly on S^2 and the angle magnitude uniformly
        in [0, max_angle_deg]. That is *not* Haar-uniform on the spherical
        cap (which would bias toward the edge), but for data augmentation
        a uniform-angle cap is what people actually want -- it gives a
        predictable, bounded "how much the wrist wobbles" knob.
        """
        if max_angle_deg <= 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        # Uniform axis on the unit sphere via normalised Gaussian.
        axis = rng.normal(size=3)
        n = float(np.linalg.norm(axis))
        if n < 1e-8:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis / n
        angle = float(rng.uniform(-np.deg2rad(max_angle_deg), np.deg2rad(max_angle_deg)))
        half = 0.5 * angle
        s = float(np.sin(half))
        c = float(np.cos(half))
        return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)

    @staticmethod
    def _quat_mul(q_a, q_b):
        """Hamilton product q_a * q_b, with quaternion convention (w, x, y, z)."""
        w1, x1, y1, z1 = q_a
        w2, x2, y2, z2 = q_b
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=np.float64)

    def _apply_joint_post_perturb(self, q, rng):
        """Add per-joint uniform offsets to an IK solution ``q``.

        See ``self._dance_task_joint_post_perturb`` (set in
        ``setup_demo``) for the rationale and shape spec. Returns ``q``
        unchanged when the knob is disabled, so the caller can wrap this
        unconditionally.
        """
        jpp = getattr(self, "_dance_task_joint_post_perturb", None)
        if jpp is None:
            return q
        # Sample one offset per joint per call so each IK retry sees an
        # independent draw (otherwise we'd just keep adding the same
        # constant offset and learn nothing about coverage).
        offset = rng.uniform(jpp[:, 0], jpp[:, 1])
        return q + offset

    def _sample_task_target(self, arm_tag, rng):
        """Uniform sample a 3D target within the arm's reachable box.

        ``rng`` is a ``np.random.Generator``; we pass it in explicitly so
        the caller controls seeding (one dedicated stream per episode,
        so task-space sampling does not poison global ``np.random``).

        Returns world-frame (x, y, z).
        """
        if arm_tag == "left":
            xr = self._dance_task_left_x
        else:
            xr = self._dance_task_right_x
        yr = self._dance_task_y
        zr_rel = self._dance_task_z_rel

        table_x_bias, table_y_bias = self.table_xy_bias
        table_top_z = 0.74 + self.table_z_bias

        x = table_x_bias + float(rng.uniform(xr[0], xr[1]))
        y = table_y_bias + float(rng.uniform(yr[0], yr[1]))
        z = table_top_z + float(rng.uniform(zr_rel[0], zr_rel[1]))
        return np.array([x, y, z], dtype=np.float64)

    def _plan_ik_keep_orientation(self, target_xyz, arm_tag, seed_qpos=None,
                                  target_quat=None):
        """IK toward ``target_xyz`` at orientation ``target_quat`` (or the
        current EE orientation when ``target_quat is None``).

        ``seed_qpos`` (optional) is the **arm-only** joint vector to warm-start
        the IK with (length = len(arm_joints)). Passing the previous sampled
        waypoint's joint config dramatically improves IK success rates for
        task-space chaining because the search starts close to a known
        feasible solution. Under the hood we splice this arm-only vector
        into the articulation's *full* active-joint qpos (which is what the
        planner actually expects), so the arm indices get warm-started
        while gripper / base joints keep their current values.

        ``target_quat`` is a length-4 (w, x, y, z) quaternion in the *world*
        frame -- same layout as ``get_left_ee_pose()[3:7]``. When left as
        ``None`` we copy the current EE quaternion (old behaviour).

        Returns ``(qpos_arm_only or None, ee_pose_before)``.
        """
        if arm_tag == "left":
            entity = self.robot.left_entity
            arm_joints = self.robot.left_arm_joints
            ee_pose_now = self.robot.get_left_ee_pose()
        else:
            entity = self.robot.right_entity
            arm_joints = self.robot.right_arm_joints
            ee_pose_now = self.robot.get_right_ee_pose()

        if target_quat is None:
            qw, qx, qy, qz = ee_pose_now[3:7]
        else:
            qw, qx, qy, qz = target_quat
        target_pose = [
            float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2]),
            float(qw), float(qx), float(qy), float(qz),
        ]

        kwargs = {}
        if seed_qpos is not None:
            # Build full active-joint qpos with our seed overwriting only the
            # arm joints' slots. The planner internally indexes into this
            # array via ``self.all_joints`` -> ``active_joints_name``.
            # Keep the dtype identical to ``entity.get_qpos()`` -- curobo
            # wraps it in ``torch.tensor(...)`` which defaults to float32,
            # so passing float64 triggers "expected scalar type Float but
            # found Double" inside the planner.
            entity_qpos = np.asarray(entity.get_qpos())
            full_qpos = entity_qpos.copy()
            active_joints = entity.get_active_joints()
            arm_joint_names = [j.get_name() for j in arm_joints]
            for i, aj in enumerate(active_joints):
                if aj.get_name() in arm_joint_names:
                    idx_in_seed = arm_joint_names.index(aj.get_name())
                    full_qpos[i] = full_qpos.dtype.type(seed_qpos[idx_in_seed])
            kwargs["last_qpos"] = full_qpos

        if arm_tag == "left":
            plan = self.robot.left_plan_path(target_pose, **kwargs)
        else:
            plan = self.robot.right_plan_path(target_pose, **kwargs)

        if plan is None or plan.get("status") != "Success":
            return None, ee_pose_now
        # Planner returns arm-joint qpos (length = len(arm_joints)).
        return np.asarray(plan["position"][-1], dtype=np.float64), ee_pose_now

    def _sample_task_keyframes(self, seed_left_qpos, seed_right_qpos,
                               n_steps, rng):
        """Generate a sequence of ``n_steps`` IK-valid joint waypoints per
        arm by sampling random 3D targets inside the arm's box and solving
        IK. Each waypoint uses the previous successful joint config as the
        IK seed (warm-start) for continuity and higher success rates.

        If every retry fails for a given step, the previous successful
        waypoint is repeated (the arm stays put for that step rather than
        throwing). This keeps the dance running even near box edges.

        Returns ``(left_waypoints, right_waypoints)`` -- lists of length
        ``n_steps`` of ndarray(ndof,).
        """
        left_waypoints = []
        right_waypoints = []
        cur_left = np.asarray(seed_left_qpos, dtype=np.float64)
        cur_right = np.asarray(seed_right_qpos, dtype=np.float64)
        # Home EE quaternions (captured once at the start of sampling -- the
        # robot is still at dance home). Each waypoint will right-multiply
        # this with a small random perturbation so the wrist "wobbles"
        # around the home orientation instead of being pinned to it.
        left_home_quat = np.asarray(self.robot.get_left_ee_pose()[3:7], dtype=np.float64)
        right_home_quat = np.asarray(self.robot.get_right_ee_pose()[3:7], dtype=np.float64)
        max_deg = self._dance_task_pose_perturb_deg

        def unwrap_toward(prev, new):
            """Return ``new`` shifted by 2*pi multiples per joint so it lies
            within +/- pi of ``prev``. Handles the "IK may return equivalent
            angles differing by 2*pi" issue, which otherwise makes the
            Catmull-Rom spline sweep 355 degrees in a fraction of a second
            ("whip-around" artefact in the rendered video).
            """
            prev = np.asarray(prev, dtype=np.float64)
            new = np.asarray(new, dtype=np.float64).copy()
            diff = new - prev
            # Number of full turns to add/subtract so each joint diff lands
            # in (-pi, pi]. ``np.round`` rounds half toward even but that's
            # fine because it's only hit when diff == exactly pi.
            turns = np.round(diff / (2.0 * np.pi))
            new -= turns * (2.0 * np.pi)
            return new

        l_fail = r_fail = 0
        for step in range(n_steps):
            # Left arm: try many IK attempts and pick the one with the
            # *smallest* joint-space step from ``cur_left``. This prefers
            # low-speed segments in the rendered video, while still
            # accepting large steps when no nearby solution exists (rather
            # than freezing the arm completely).
            best_q_l = None
            best_span_l = float("inf")
            for attempt in range(self._dance_task_max_retries):
                tgt = self._sample_task_target("left", rng)
                # First retries use the perturbed quat; the last one falls
                # back to the pristine home quat.
                use_home = attempt >= self._dance_task_max_retries - 1
                if use_home or max_deg <= 0.0:
                    target_q = left_home_quat
                else:
                    delta = self._random_quat_perturb(rng, max_deg)
                    target_q = self._quat_mul(left_home_quat, delta)
                q, _ = self._plan_ik_keep_orientation(
                    tgt, "left", seed_qpos=cur_left, target_quat=target_q,
                )
                if q is None:
                    continue
                q = unwrap_toward(cur_left, q)
                # Optional per-joint post-perturbation. Applied here so
                # the span / max_joint_step check below covers the
                # *full* movement (IK + perturbation), and unwrap_toward
                # has already brought q close to cur_left so the offset
                # math doesn't fight 2*pi equivalences.
                q = self._apply_joint_post_perturb(q, rng)
                span = float(np.max(np.abs(q - cur_left)))
                if span < best_span_l:
                    best_q_l = q
                    best_span_l = span
                # Early exit once we find a "well-behaved" solution.
                if span <= self._dance_task_max_joint_step:
                    break
            if best_q_l is not None:
                cur_left = best_q_l
            else:
                l_fail += 1  # all retries returned None; repeat cur_left
            left_waypoints.append(cur_left.copy())

            # Right arm
            best_q_r = None
            best_span_r = float("inf")
            for attempt in range(self._dance_task_max_retries):
                tgt = self._sample_task_target("right", rng)
                use_home = attempt >= self._dance_task_max_retries - 1
                if use_home or max_deg <= 0.0:
                    target_q = right_home_quat
                else:
                    delta = self._random_quat_perturb(rng, max_deg)
                    target_q = self._quat_mul(right_home_quat, delta)
                q, _ = self._plan_ik_keep_orientation(
                    tgt, "right", seed_qpos=cur_right, target_quat=target_q,
                )
                if q is None:
                    continue
                q = unwrap_toward(cur_right, q)
                q = self._apply_joint_post_perturb(q, rng)
                span = float(np.max(np.abs(q - cur_right)))
                if span < best_span_r:
                    best_q_r = q
                    best_span_r = span
                if span <= self._dance_task_max_joint_step:
                    break
            if best_q_r is not None:
                cur_right = best_q_r
            else:
                r_fail += 1
            right_waypoints.append(cur_right.copy())

        if l_fail or r_fail:
            print(f"\033[93m[random_dance task]\033[0m IK retries exhausted: "
                  f"left={l_fail}/{n_steps}  right={r_fail}/{n_steps}  "
                  f"(these keyframes reused the previous pose)")

        # ---- "Frozen keyframe" check ------------------------------------
        # If a waypoint ends up nearly identical to the previous one (even
        # though IK reported "Success"), the arm will appear *frozen* in
        # the video. This can happen when the planner clamps an
        # unreachable target to the nearest feasible qpos, which is
        # frequently the same as the seed. We only warn when this
        # actually happens -- the per-keyframe delta dump was too noisy
        # for normal collection runs.
        seed_left = np.asarray(seed_left_qpos, dtype=np.float64)
        seed_right = np.asarray(seed_right_qpos, dtype=np.float64)
        frozen_thr = 1e-3  # < 0.06 deg per joint -> effectively frozen
        l_prev = seed_left
        r_prev = seed_right
        l_frozen_steps, r_frozen_steps = [], []
        for i, (lw, rw) in enumerate(zip(left_waypoints, right_waypoints)):
            if float(np.max(np.abs(lw - l_prev))) < frozen_thr:
                l_frozen_steps.append(i)
            if float(np.max(np.abs(rw - r_prev))) < frozen_thr:
                r_frozen_steps.append(i)
            l_prev, r_prev = lw, rw
        if l_frozen_steps:
            print(f"\033[93m[random_dance task] LEFT frozen at steps "
                  f"{l_frozen_steps}\033[0m")
        if r_frozen_steps:
            print(f"\033[93m[random_dance task] RIGHT frozen at steps "
                  f"{r_frozen_steps}\033[0m")
        return left_waypoints, right_waypoints

    # ------------------------------------------------------------------
    # Task-space (IK) debug mode
    # ------------------------------------------------------------------
    def _plan_ik_for_arm(self, target_xyz, arm_tag):
        """Run IK for ``arm_tag`` to reach ``target_xyz`` while *keeping*
        the current end-effector orientation -- i.e. only ask the planner
        to move the EE position. This is the most robust thing for debug
        reachability tests: any orientation constraint (like "fingers down")
        drastically shrinks the reachable set and is the #1 reason IK fails
        for nearby targets.

        Returns
        -------
        qpos : np.ndarray(ndof,) of float
            Planned final joint configuration (arm joints only, no gripper).
        ee_pose : list[float] of length 7
            The EE pose at the time of planning (before replay).
        """
        if arm_tag == "left":
            ee_pose_now = self.robot.get_left_ee_pose()
        else:
            ee_pose_now = self.robot.get_right_ee_pose()

        # Keep the current orientation (quaternion), change only position.
        qw, qx, qy, qz = ee_pose_now[3], ee_pose_now[4], ee_pose_now[5], ee_pose_now[6]
        target_pose = [
            float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2]),
            float(qw), float(qx), float(qy), float(qz),
        ]

        if arm_tag == "left":
            plan = self.robot.left_plan_path(target_pose)
        else:
            plan = self.robot.right_plan_path(target_pose)

        if plan is None or plan.get("status") != "Success":
            return None, ee_pose_now
        final_qpos = np.asarray(plan["position"][-1], dtype=np.float64)
        return final_qpos, ee_pose_now

    def _play_ik_debug(self, left_home, right_home,
                       left_grip, right_grip):
        """Drive each arm to a hover pose above the corresponding half of
        the table (left arm -> +x half, right arm -> -x half), print the
        resulting joint state, and stop. Used to validate that the IK and
        workspace mapping are correct before enabling random task-space
        sampling in the main dance loop.
        """
        # Table-frame centre of each half (relative to the table's own
        # centre, which itself is offset by ``self.table_xy_bias``). The
        # table is 1.2 m long (x) and 0.7 m wide (y) -- see
        # Base_Task.create_table_and_wall.
        #
        # *** Left/right mapping ***
        # Confirmed by printing ``get_left_ee_pose()`` / ``get_right_ee_pose()``
        # at home for the aloha-agilex embodiment:
        #   left EE  sits at x = -0.298  (so left arm owns the -x half)
        #   right EE sits at x = +0.306  (so right arm owns the +x half)
        # i.e. the *urdf's* "left arm" is actually on world -x. We therefore
        # aim the left arm at (-0.3, ...) and the right arm at (+0.3, ...).
        #
        # Positive ``forward`` pushes the target toward the observer camera
        # (world +y).
        table_x_bias, table_y_bias = self.table_xy_bias
        table_top_z = 0.74 + self.table_z_bias  # matches create_table_and_wall
        hover_z = table_top_z + self._dance_ik_debug_hover
        fwd_y = self._dance_ik_debug_forward
        lat_x = self._dance_ik_debug_lateral

        left_target = np.array([table_x_bias - lat_x, table_y_bias + fwd_y, hover_z])
        right_target = np.array([table_x_bias + lat_x, table_y_bias + fwd_y, hover_z])

        print("\033[96m[ik-debug]\033[0m table_top_z = "
              f"{table_top_z:.3f}  hover_z = {hover_z:.3f}  forward_y = {fwd_y:.3f}")
        print(f"\033[96m[ik-debug]\033[0m left  target world pos = {left_target}")
        print(f"\033[96m[ik-debug]\033[0m right target world pos = {right_target}")

        # Before any IK we log where each EE currently sits so we can sanity-
        # check the sign convention (+x side vs -x side).
        left_ee_before = self.robot.get_left_ee_pose()
        right_ee_before = self.robot.get_right_ee_pose()
        print(f"\033[96m[ik-debug]\033[0m left  EE @ home = "
              f"[{left_ee_before[0]:+.3f}, {left_ee_before[1]:+.3f}, {left_ee_before[2]:+.3f}]")
        print(f"\033[96m[ik-debug]\033[0m right EE @ home = "
              f"[{right_ee_before[0]:+.3f}, {right_ee_before[1]:+.3f}, {right_ee_before[2]:+.3f}]")

        l_q, _ = self._plan_ik_for_arm(left_target, "left")
        r_q, _ = self._plan_ik_for_arm(right_target, "right")

        if l_q is None:
            print("\033[91m[ik-debug] left  IK FAILED -> keeping home\033[0m")
            l_q = left_home.copy()
        if r_q is None:
            print("\033[91m[ik-debug] right IK FAILED -> keeping home\033[0m")
            r_q = right_home.copy()

        # Replace the planned path with a two-waypoint list: (home) -> (IK
        # result, held). We repeat the IK result so the spline plays
        # "go to target and stop" without wobbling past it.
        self.left_joint_path = [
            {"arm": l_q.tolist(), "gripper": left_grip},
            {"arm": l_q.tolist(), "gripper": left_grip},
        ]
        self.right_joint_path = [
            {"arm": r_q.tolist(), "gripper": right_grip},
            {"arm": r_q.tolist(), "gripper": right_grip},
        ]

        left_waypoints = [left_home.copy(), l_q, l_q]
        right_waypoints = [right_home.copy(), r_q, r_q]
        left_grips = [left_grip, left_grip, left_grip]
        right_grips = [right_grip, right_grip, right_grip]
        self._drive_spline(left_waypoints, right_waypoints, left_grips, right_grips)

        # Report where the arms actually ended up + the j1..j3 values requested.
        left_ee_after = self.robot.get_left_ee_pose()
        right_ee_after = self.robot.get_right_ee_pose()
        l_err = np.linalg.norm(np.array(left_ee_after[:3]) - left_target)
        r_err = np.linalg.norm(np.array(right_ee_after[:3]) - right_target)
        print("\033[92m[ik-debug] result\033[0m ----------------------------")
        print(f"  left  qpos (j1..jN): {np.round(l_q, 3).tolist()}")
        print(f"  left  j1,j2,j3     : {float(l_q[0]):+.3f}, {float(l_q[1]):+.3f}, {float(l_q[2]):+.3f}")
        print(f"  left  EE reached   : [{left_ee_after[0]:+.3f}, {left_ee_after[1]:+.3f}, {left_ee_after[2]:+.3f}]")
        print(f"  left  EE error     : {l_err*1000:.1f} mm")
        print(f"  right qpos (j1..jN): {np.round(r_q, 3).tolist()}")
        print(f"  right j1,j2,j3     : {float(r_q[0]):+.3f}, {float(r_q[1]):+.3f}, {float(r_q[2]):+.3f}")
        print(f"  right EE reached   : [{right_ee_after[0]:+.3f}, {right_ee_after[1]:+.3f}, {right_ee_after[2]:+.3f}]")
        print(f"  right EE error     : {r_err*1000:.1f} mm")
        # Copy-paste-ready dance home block: drop the two lines below
        # straight into ``random_dance.left_home`` / ``right_home`` in
        # the task yaml to freeze this IK pose as the dance start pose.
        l_home_str = "[" + ", ".join(f"{float(v):+.4f}" for v in l_q) + "]"
        r_home_str = "[" + ", ".join(f"{float(v):+.4f}" for v in r_q) + "]"
        print("\033[92m[ik-debug] yaml-ready dance home\033[0m")
        print(f"  left_home:  {l_home_str}")
        print(f"  right_home: {r_home_str}")
        print("\033[92m[ik-debug] end\033[0m ----------------------------")

    def _settle_to_dance_home(self, left_home, right_home,
                              left_grip, right_grip):
        """Drive the robot from its current pose to the dance home and
        hold there until physics has actually converged.

        Why this exists. The base class's ``move_to_homestate`` only sets
        the PD drive *target* to the embodiment homestate (typically all
        zeros for aloha-agilex) -- it does not call ``scene.step`` so no
        physics is integrated, which means at the start of ``play_once``
        the robot is still near the embodiment's home, NOT at our task-
        level dance home. Without this settle phase the very first
        Catmull-Rom segment ``home -> keyframe_1`` plays out while the
        joints are still racing toward home, so the dance never visibly
        passes through the dance-home pose at all (the user would see
        the robot mostly skip home and lurch directly toward the first
        IK waypoint).

        We side-step that by driving a tiny three-waypoint spline
        ``[current, home, home]`` first. The current-pose waypoint is
        read straight off the articulation, so the spline knows where
        the joints really are. The duplicated home at the end gives PD
        a full segment to converge before the main dance starts.
        """
        # Read the *physical* current arm qpos (drop the gripper which
        # ``get_left_arm_real_jointState`` appends at the end).
        cur_left = np.asarray(
            self.robot.get_left_arm_real_jointState()[:-1], dtype=np.float64)
        cur_right = np.asarray(
            self.robot.get_right_arm_real_jointState()[:-1], dtype=np.float64)
        # ``[current, home, home]`` so PD has time to settle on home.
        left_waypoints = [cur_left, left_home.copy(), left_home.copy()]
        right_waypoints = [cur_right, right_home.copy(), right_home.copy()]
        left_grips = [left_grip, left_grip, left_grip]
        right_grips = [right_grip, right_grip, right_grip]
        self._drive_spline(left_waypoints, right_waypoints,
                           left_grips, right_grips)

    def play_once(self):
        # Use the task-level "dance home" (not the embodiment homestate) as the
        # sampling centre so the arms start in a spread-open pose.
        left_home = self._dance_left_home.astype(np.float64, copy=True)
        right_home = self._dance_right_home.astype(np.float64, copy=True)

        # Current gripper values (normalised [0,1]).
        left_grip = float(self.robot.get_left_gripper_val() or 0.0)
        right_grip = float(self.robot.get_right_gripper_val() or 0.0)

        # -- IK DEBUG MODE -----------------------------------------------
        if self._dance_mode == "ik_debug":
            # Only run the debug trajectory on the first pass (need_plan=True).
            # Subsequent replays just follow whatever we recorded.
            if self.need_plan:
                self._play_ik_debug(left_home, right_home, left_grip, right_grip)
            else:
                # Replay path: use whatever IK joint angles were saved.
                left_waypoints = [left_home.copy()]
                right_waypoints = [right_home.copy()]
                left_grips = [left_grip]
                right_grips = [right_grip]
                n = min(len(self.left_joint_path), len(self.right_joint_path))
                for i in range(n):
                    l_kf = self.left_joint_path[i]
                    r_kf = self.right_joint_path[i]
                    left_waypoints.append(np.asarray(l_kf["arm"], dtype=np.float64))
                    right_waypoints.append(np.asarray(r_kf["arm"], dtype=np.float64))
                    left_grips.append(float(l_kf["gripper"]))
                    right_grips.append(float(r_kf["gripper"]))
                self._drive_spline(left_waypoints, right_waypoints, left_grips, right_grips)
            self.info["info"] = {"{A}": "random_dance", "{a}": "dual"}
            return self.info

        # -- HOME DEBUG MODE ---------------------------------------------
        # Drive to the configured dance home and hold, no IK, no sampling.
        # Intended for eyeballing / iterating on ``left_home``/``right_home``
        # values without the IK step overwriting the held pose.
        if self._dance_mode == "home_debug":
            # Optional yaml override: set ``home_debug_gripper`` to a float
            # in [0, 1] (0 = close, 1 = open) to force the grippers to a
            # particular state while inspecting the home pose. Default is
            # "whatever they currently hold" (i.e. usually 0 = closed).
            hd_grip = getattr(self, "_dance_home_debug_gripper", None)
            if hd_grip is not None:
                left_grip = float(np.clip(hd_grip, 0.0, 1.0))
                right_grip = float(np.clip(hd_grip, 0.0, 1.0))

            # Duplicate the home waypoint so the spline plays
            # (current_pose -> home -> hold). Three copies give a calm
            # hold phase for video inspection.
            left_waypoints = [left_home.copy(), left_home.copy(), left_home.copy()]
            right_waypoints = [right_home.copy(), right_home.copy(), right_home.copy()]
            left_grips = [left_grip, left_grip, left_grip]
            right_grips = [right_grip, right_grip, right_grip]
            self._drive_spline(left_waypoints, right_waypoints, left_grips, right_grips)

            # Log the *actual* EE poses once we're settled at home -- this
            # is the thing to eyeball when tuning shoulder yaw / pitch /
            # elbow values. Compare against shoulder positions and upper-
            # arm / forearm lengths to check "arm vertical" etc.
            left_ee = self.robot.get_left_ee_pose()
            right_ee = self.robot.get_right_ee_pose()
            print("\033[92m[home-debug]\033[0m "
                  f"requested left  home qpos = {np.round(left_home, 3).tolist()}")
            print("\033[92m[home-debug]\033[0m "
                  f"requested right home qpos = {np.round(right_home, 3).tolist()}")
            print("\033[92m[home-debug]\033[0m "
                  f"left  EE @ home = [{left_ee[0]:+.3f}, {left_ee[1]:+.3f}, {left_ee[2]:+.3f}]")
            print("\033[92m[home-debug]\033[0m "
                  f"right EE @ home = [{right_ee[0]:+.3f}, {right_ee[1]:+.3f}, {right_ee[2]:+.3f}]")
            print("\033[92m[home-debug]\033[0m "
                  f"grippers set to left={left_grip:.2f}, right={right_grip:.2f} "
                  f"(0=close, 1=open)")
            # Approx shoulder world position for aloha-agilex
            # (see _play_ik_debug comment); handy for sanity checks.
            print("\033[92m[home-debug]\033[0m "
                  f"(shoulder approx: left=(-0.30, -0.42, 0.78), "
                  f"right=(+0.30, -0.42, 0.78); upper arm ~0.25m, forearm ~0.26m)")
            self.info["info"] = {"{A}": "random_dance", "{a}": "dual"}
            return self.info

        # -- TASK-SPACE DANCE (random Cartesian targets) -----------------
        if self._dance_mode == "task":
            if self.need_plan:
                # Dedicated RNG so task sampling is reproducible per-seed
                # and does not disturb other global random consumers
                # (HDRI rotation, domain randomisation, etc).
                base_seed = int(getattr(self, "ep_num", 0))
                rng = np.random.default_rng(
                    int(np.uint32(base_seed * 2246822519 + 0xBEEF))
                )
                # Seed IK search at the dance home (the arms sit there at
                # the start of the episode). ``left_home`` / ``right_home``
                # already have the same length as ``arm_joints`` (6 for
                # aloha-agilex), and ``_plan_ik_keep_orientation`` will
                # splice these into the articulation's full qpos internally.
                left_wps, right_wps = self._sample_task_keyframes(
                    left_home, right_home, self._dance_n_steps, rng,
                )
                # Randomise gripper exactly like joint mode (reuses global
                # np.random so ``gripper_toggle_p`` behaves the same).
                self.left_joint_path = []
                self.right_joint_path = []
                for l_arm, r_arm in zip(left_wps, right_wps):
                    if np.random.rand() < self._dance_gripper_toggle_p:
                        left_grip = float(np.random.uniform(0.0, 1.0))
                    if np.random.rand() < self._dance_gripper_toggle_p:
                        right_grip = float(np.random.uniform(0.0, 1.0))
                    self.left_joint_path.append({"arm": l_arm.tolist(), "gripper": left_grip})
                    self.right_joint_path.append({"arm": r_arm.tolist(), "gripper": right_grip})
            # -- Settle to dance home before the dance starts ------------
            # See ``_settle_to_dance_home`` docstring. Without this the
            # very first dance segment starts before the robot has
            # reached home, so the home pose is never actually held.
            # Run for both fresh-sampled and replayed paths so the held
            # video frames at the start are consistent.
            self._settle_to_dance_home(left_home, right_home,
                                       left_grip, right_grip)

            # -- Play back (fresh or replayed traj data) ------------------
            left_waypoints = [left_home.copy()]
            right_waypoints = [right_home.copy()]
            left_grips = [left_grip]
            right_grips = [right_grip]
            n = min(len(self.left_joint_path), len(self.right_joint_path))
            for i in range(n):
                l_kf = self.left_joint_path[i]
                r_kf = self.right_joint_path[i]
                left_waypoints.append(np.asarray(l_kf["arm"], dtype=np.float64))
                right_waypoints.append(np.asarray(r_kf["arm"], dtype=np.float64))
                left_grips.append(float(l_kf["gripper"]))
                right_grips.append(float(r_kf["gripper"]))
            self._drive_spline(left_waypoints, right_waypoints, left_grips, right_grips)
            self.info["info"] = {"{A}": "random_dance", "{a}": "dual"}
            return self.info

        # -- DEFAULT JOINT-SPACE DANCE -----------------------------------
        if self.need_plan:
            # ---------- generate a fresh random dance ----------
            self.left_joint_path = []
            self.right_joint_path = []
            for _ in range(self._dance_n_steps):
                l_arm = self._sample_arm_targets(left_home, "left")
                r_arm = self._sample_arm_targets(right_home, "right")

                if np.random.rand() < self._dance_gripper_toggle_p:
                    left_grip = float(np.random.uniform(0.0, 1.0))
                if np.random.rand() < self._dance_gripper_toggle_p:
                    right_grip = float(np.random.uniform(0.0, 1.0))

                self.left_joint_path.append({
                    "arm": l_arm.tolist(),
                    "gripper": left_grip,
                })
                self.right_joint_path.append({
                    "arm": r_arm.tolist(),
                    "gripper": right_grip,
                })

        # Settle to dance home so the very first played-back segment
        # actually starts from home (and not from wherever PD happened
        # to be when ``play_once`` was called). See
        # ``_settle_to_dance_home`` docstring.
        self._settle_to_dance_home(left_home, right_home,
                                   left_grip, right_grip)

        # Build the full waypoint / gripper lists. Index 0 is the current
        # pose (dance home); 1..N are the sampled keyframes. Works for both
        # the fresh-sampling pass above and the replay pass (when
        # self.left_joint_path was loaded from disk).
        n = min(len(self.left_joint_path), len(self.right_joint_path))
        left_waypoints = [left_home.copy()]
        right_waypoints = [right_home.copy()]
        left_grips = [left_grip]
        right_grips = [right_grip]
        for i in range(n):
            l_kf = self.left_joint_path[i]
            r_kf = self.right_joint_path[i]
            l_arm = np.asarray(l_kf["arm"] if isinstance(l_kf, dict) else l_kf, dtype=np.float64)
            r_arm = np.asarray(r_kf["arm"] if isinstance(r_kf, dict) else r_kf, dtype=np.float64)
            l_grip = float(l_kf["gripper"]) if isinstance(l_kf, dict) else left_grips[-1]
            r_grip = float(r_kf["gripper"]) if isinstance(r_kf, dict) else right_grips[-1]
            left_waypoints.append(l_arm)
            right_waypoints.append(r_arm)
            left_grips.append(l_grip)
            right_grips.append(r_grip)

        # Drive the whole sequence with one continuous spline playback.
        self._drive_spline(left_waypoints, right_waypoints, left_grips, right_grips)

        self.info["info"] = {"{A}": "random_dance", "{a}": "dual"}
        return self.info

    def check_success(self):
        # No explicit goal -- the episode is always considered successful so
        # that collect_data.py saves the trajectory.
        return True
