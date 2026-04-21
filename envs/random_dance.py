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

    def _drive_to_keyframe(self, left_arm_target, right_arm_target,
                           left_gripper_target, right_gripper_target):
        """Drive the robot towards the given key-frame for a fixed number of
        simulation substeps, saving observations periodically.
        """
        # Velocity target is set to zero; sapien PD controller will handle it.
        zero_vel_l = np.zeros_like(left_arm_target)
        zero_vel_r = np.zeros_like(right_arm_target)

        for sub in range(self._dance_hold_substeps):
            self.robot.set_arm_joints(left_arm_target, zero_vel_l, "left")
            self.robot.set_arm_joints(right_arm_target, zero_vel_r, "right")
            self.robot.set_gripper(left_gripper_target, "left")
            self.robot.set_gripper(right_gripper_target, "right")
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

                self._drive_to_keyframe(l_arm, r_arm, left_grip, right_grip)
        else:
            # ---------- replay the recorded dance ----------
            n = min(len(self.left_joint_path), len(self.right_joint_path))
            for i in range(n):
                l_kf = self.left_joint_path[i]
                r_kf = self.right_joint_path[i]
                l_arm = np.asarray(l_kf["arm"] if isinstance(l_kf, dict) else l_kf, dtype=np.float64)
                r_arm = np.asarray(r_kf["arm"] if isinstance(r_kf, dict) else r_kf, dtype=np.float64)
                l_grip = float(l_kf["gripper"]) if isinstance(l_kf, dict) else left_grip
                r_grip = float(r_kf["gripper"]) if isinstance(r_kf, dict) else right_grip
                self._drive_to_keyframe(l_arm, r_arm, l_grip, r_grip)

        self.info["info"] = {"{A}": "random_dance", "{a}": "dual"}
        return self.info

    def check_success(self):
        # No explicit goal -- the episode is always considered successful so
        # that collect_data.py saves the trajectory.
        return True
