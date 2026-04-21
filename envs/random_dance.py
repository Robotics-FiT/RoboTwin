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

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)
        # Optional per-task overrides from task_config.
        dance_cfg = kwargs.get("random_dance", {}) or {}
        self._dance_n_steps = int(dance_cfg.get("n_steps", self.DEFAULT_N_STEPS))
        self._dance_arm_delta = float(dance_cfg.get("arm_delta", self.DEFAULT_ARM_DELTA))
        self._dance_gripper_toggle_p = float(dance_cfg.get("gripper_toggle_p", self.DEFAULT_GRIPPER_TOGGLE_P))
        self._dance_hold_substeps = int(dance_cfg.get("hold_substeps", self.DEFAULT_HOLD_SUBSTEPS))
        self._dance_save_every = max(1, int(dance_cfg.get("save_every", self.DEFAULT_SAVE_EVERY)))
        # Fraction of ``hold_substeps`` used for the *cruise* (linear interp)
        # phase; the remaining fraction is used to let the PD controller
        # settle on the target. Set to 1.0 for the smoothest motion (no
        # deceleration between key-frames at all).
        self._dance_cruise_ratio = float(dance_cfg.get("cruise_ratio", 1.0))
        self._dance_cruise_ratio = min(max(self._dance_cruise_ratio, 0.05), 1.0)

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

    def _drive_to_keyframe(self, left_arm_prev, left_arm_target,
                           right_arm_prev, right_arm_target,
                           left_grip_prev, left_grip_target,
                           right_grip_prev, right_grip_target):
        """Drive the robot from the previous key-frame to the given one using
        a *linear interpolation* reference trajectory with velocity feed-
        forward, so that consecutive key-frames are stitched into a smooth
        continuous motion instead of a start-stop-start-stop pattern.

        The first ``cruise_ratio * hold_substeps`` substeps linearly
        interpolate the position reference from ``prev`` to ``target`` and
        feed the matching constant velocity to the PD controller. The
        remaining substeps hold the final target with zero velocity reference
        (letting the PD controller settle, avoiding overshoot on the very
        last key-frame).
        """
        left_arm_prev = np.asarray(left_arm_prev, dtype=np.float64)
        left_arm_target = np.asarray(left_arm_target, dtype=np.float64)
        right_arm_prev = np.asarray(right_arm_prev, dtype=np.float64)
        right_arm_target = np.asarray(right_arm_target, dtype=np.float64)

        # Simulation timestep (s). Fall back to the default if unavailable.
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            dt = 1.0 / 250.0

        total = max(1, self._dance_hold_substeps)
        cruise = max(1, int(round(total * self._dance_cruise_ratio)))
        cruise_time = cruise * dt  # seconds spent cruising

        # Constant velocity feed-forward during the cruise phase.
        left_arm_vel = (left_arm_target - left_arm_prev) / cruise_time
        right_arm_vel = (right_arm_target - right_arm_prev) / cruise_time
        zero_vel_l = np.zeros_like(left_arm_target)
        zero_vel_r = np.zeros_like(right_arm_target)

        for sub in range(total):
            if sub < cruise:
                alpha = (sub + 1) / float(cruise)
                l_pos = left_arm_prev + (left_arm_target - left_arm_prev) * alpha
                r_pos = right_arm_prev + (right_arm_target - right_arm_prev) * alpha
                l_grip = left_grip_prev + (left_grip_target - left_grip_prev) * alpha
                r_grip = right_grip_prev + (right_grip_target - right_grip_prev) * alpha
                l_vel = left_arm_vel
                r_vel = right_arm_vel
            else:
                # Settle phase: hold target position, zero velocity reference.
                l_pos = left_arm_target
                r_pos = right_arm_target
                l_grip = left_grip_target
                r_grip = right_grip_target
                l_vel = zero_vel_l
                r_vel = zero_vel_r

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
        # Always take one final snapshot at the end of the key-frame.
        self._take_picture()

    def play_once(self):
        left_home = np.array(self.robot.left_homestate, dtype=np.float64)
        right_home = np.array(self.robot.right_homestate, dtype=np.float64)

        # Current gripper values (normalised [0,1]).
        left_grip = float(self.robot.get_left_gripper_val() or 0.0)
        right_grip = float(self.robot.get_right_gripper_val() or 0.0)

        # "prev_*" tracks where the robot is *coming from* for the current
        # key-frame so that _drive_to_keyframe can build a smooth ramp.
        prev_left_arm = left_home.copy()
        prev_right_arm = right_home.copy()
        prev_left_grip = left_grip
        prev_right_grip = right_grip

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

                self._drive_to_keyframe(
                    prev_left_arm, l_arm,
                    prev_right_arm, r_arm,
                    prev_left_grip, left_grip,
                    prev_right_grip, right_grip,
                )
                prev_left_arm = l_arm
                prev_right_arm = r_arm
                prev_left_grip = left_grip
                prev_right_grip = right_grip
        else:
            # ---------- replay the recorded dance ----------
            n = min(len(self.left_joint_path), len(self.right_joint_path))
            for i in range(n):
                l_kf = self.left_joint_path[i]
                r_kf = self.right_joint_path[i]
                l_arm = np.asarray(l_kf["arm"] if isinstance(l_kf, dict) else l_kf, dtype=np.float64)
                r_arm = np.asarray(r_kf["arm"] if isinstance(r_kf, dict) else r_kf, dtype=np.float64)
                l_grip = float(l_kf["gripper"]) if isinstance(l_kf, dict) else prev_left_grip
                r_grip = float(r_kf["gripper"]) if isinstance(r_kf, dict) else prev_right_grip
                self._drive_to_keyframe(
                    prev_left_arm, l_arm,
                    prev_right_arm, r_arm,
                    prev_left_grip, l_grip,
                    prev_right_grip, r_grip,
                )
                prev_left_arm = l_arm
                prev_right_arm = r_arm
                prev_left_grip = l_grip
                prev_right_grip = r_grip

        self.info["info"] = {"{A}": "random_dance", "{a}": "dual"}
        return self.info

    def check_success(self):
        # No explicit goal -- the episode is always considered successful so
        # that collect_data.py saves the trajectory.
        return True
