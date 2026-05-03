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
    # + shoulder lift + elbow bend so the two arms naturally spread open.
    # Joint order: [j1, j2, j3, j4, j5, j6] (6-DoF arm).
    #  j1: shoulder yaw   (positive -> outward abduction on each side)
    #  j2: shoulder pitch (sign depends on URDF; tune after first rollout)
    #  j3: elbow          (positive -> bend)
    #  j4,j5,j6: wrist
    DEFAULT_LEFT_HOME = [0.30, -0.40, 0.80, 0.0, 0.80, 0.0]
    DEFAULT_RIGHT_HOME = [-0.30, -0.40, 0.80, 0.0, 0.80, 0.0]

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)
        # Optional per-task overrides from task_config.
        dance_cfg = kwargs.get("random_dance", {}) or {}
        self._dance_n_steps = int(dance_cfg.get("n_steps", self.DEFAULT_N_STEPS))
        self._dance_arm_delta = float(dance_cfg.get("arm_delta", self.DEFAULT_ARM_DELTA))
        self._dance_gripper_toggle_p = float(dance_cfg.get("gripper_toggle_p", self.DEFAULT_GRIPPER_TOGGLE_P))
        self._dance_hold_substeps = int(dance_cfg.get("hold_substeps", self.DEFAULT_HOLD_SUBSTEPS))
        self._dance_save_every = max(1, int(dance_cfg.get("save_every", self.DEFAULT_SAVE_EVERY)))

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

        steps_per_seg = max(1, int(self._dance_hold_substeps))
        seg_time = steps_per_seg * dt  # seconds per segment

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
        # ``seg_time``.
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

    def play_once(self):
        # Use the task-level "dance home" (not the embodiment homestate) as the
        # sampling centre so the arms start in a spread-open pose.
        left_home = self._dance_left_home.astype(np.float64, copy=True)
        right_home = self._dance_right_home.astype(np.float64, copy=True)

        # Current gripper values (normalised [0,1]).
        left_grip = float(self.robot.get_left_gripper_val() or 0.0)
        right_grip = float(self.robot.get_right_gripper_val() or 0.0)

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
