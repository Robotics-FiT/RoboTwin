# Scene Geometry — RoboTwin (aloha-agilex)

This is a practical reference for the **world-frame geometry** of the
RoboTwin scene used by `random_dance` (and most other tasks): where the
robot sits, where the cameras look from, what the joints do, and how
"left / right" map to +/- x.

All coordinates are **world frame, metres**. Convention: +z up, +x right
(from a bird's-eye view), +y forward-from-robot / toward the observer
camera. The robot faces +y.

---

## Quick reference card

| What | Pose (x, y, z) | Notes |
|---|---|---|
| **Robot base** (`aloha-agilex`) | (0, −0.65, 0) | Facing +y (see `config.yml: robot_pose`). The base quaternion `[0.707, 0, 0, 0.707]` is a 90° rotation about +z. |
| Table centre | (0, 0, 0.74) | Length 1.2 (x) × width 0.7 (y), 5 cm thick. Height configurable via `random_table_height`, see `table_z_bias`. |
| Table-top z | 0.74 + `table_z_bias` | Randomised per episode, see Tunable knobs. |
| Left-half centre | (−0.3, 0, z_top) | "Left" = from robot's own POV (see §left/right). |
| Right-half centre | (+0.3, 0, z_top) | |
| Table front edge (toward camera) | y = +0.35 | Going past this puts the EE over empty space. |
| Table back edge (behind robot) | y = −0.35 | The robot is another 30 cm behind this. |
| **Observer camera** (`third_person_view: observer`) | (0, +0.65, 1.20) | Forward = (0, −1, −0.4); looks across the table at the robot, slight downward pitch (~22°). 70° fovy, 1280×720. Used for videos / augmentation. |
| **Head camera** (`head_camera`) | (−0.032, −0.45, 1.35) | Forward = (0, 0.6, −0.8); mounted on the robot, looks down at the tabletop from behind. D435 realsense. |
| **Front camera** | (0, −0.45, 0.85) | Forward = (0, 1, −0.1); almost horizontal, shoots from the robot's own chest toward the table. |

> **Reminder:** `table_z_bias` is sampled **symmetrically** per episode in
> `[−random_table_height, +random_table_height]`, so the tabletop moves
> both up and down from the nominal 0.74 m. See
> `envs/_base_task.py::_init_task_env_`.

---

## Left / right arm mapping

`aloha-agilex` is bilateral with two 6-DoF arms, named `fl_` and `fr_`
in the URDF. Before IK work we empirically probed the EE positions at
home and found:

```
left  EE @ home  ≈ (x = −0.298, y = −0.314, z = +0.942)
right EE @ home  ≈ (x = +0.306, y = −0.313, z = +0.941)
```

So the mapping is:

| URDF prefix | Arm alias | Occupies world half | Home EE position |
|---|---|---|---|
| `fl_*` | **left**  | **−x** half of the table | (−0.30, −0.31, 0.94) |
| `fr_*` | **right** | **+x** half of the table | (+0.31, −0.31, 0.94) |

Counter-intuitive as it may read, this is because the base quaternion
is a 90° rotation about +z. If you stand *behind* the robot looking
forward (+y), the "left" arm really is on your left — i.e. world −x
side. This matches what you see in the `observer_camera` view, where
screen-left is world +x (camera-local "left" is set to world +x), so
the video actually shows the **left arm on the right of the frame**.

When hand-writing Cartesian targets, **pick the half that the arm
naturally owns** — making the left arm reach into +x will force a huge
cross-body motion and IK will almost always fail.

---

## Arm joint semantics

Order inside `arm_joints_name`:
`[fl_joint1, fl_joint2, fl_joint3, fl_joint4, fl_joint5, fl_joint6]`
(and the mirror for `fr_*`).

> **Heads up — these directional intuitions only hold near the original
> spread-forward home `[0.30, 0.60, 0.40, 0, 0.80, 0]`.** Several joints
> (notably j3 because of `link3`'s built-in `rpy="π"` flip) reverse
> their effective direction once you push past ~π/4. For poses far from
> that home, read the URDF axes directly and verify with `home_debug`.
> See **`docs/dance_home_tuning.md`** for the full method, the
> verification ruler, and the final L-shape pose.

| # | URDF joint | Role | Sign convention (confirmed by `DEFAULT_LEFT_HOME`) |
|---|---|---|---|
| j1 | fl_joint1 | shoulder **yaw** | +: outward abduction (away from body midline) |
| j2 | fl_joint2 | shoulder **pitch** | +: arm swings **forward** (toward +y / camera) |
| j3 | fl_joint3 | **elbow** | +: bend; smaller value → straighter arm → longer reach |
| j4 | fl_joint4 | wrist **roll** | around the forearm axis |
| j5 | fl_joint5 | wrist **pitch** | +: fingers drop toward −z |
| j6 | fl_joint6 | wrist **yaw** | rolls the gripper about its own axis |

Approximate joint ranges (URDF limits vary, but the sampler clips to
5 % safety inside them). For random dance motion the defaults are:

```yaml
arm_delta: 0.6        # rad (~34°) per joint around the dance home
hold_substeps: 80     # physics substeps per keyframe segment (dt=1/250s → 0.32 s)
```

### Default dance home (`envs/random_dance.py::DEFAULT_*_HOME`)

```python
DEFAULT_LEFT_HOME  = [ 0.30, 0.60, 0.40, 0.0, 0.80, 0.0]
DEFAULT_RIGHT_HOME = [-0.30, 0.60, 0.40, 0.0, 0.80, 0.0]
#                      j1    j2    j3    j4   j5    j6
```

i.e. both arms:

* j1 = ±0.30 → shoulders abducted outward
* j2 = +0.60 → shoulders pitched forward (tilting toward camera)
* j3 = +0.40 → elbows slightly bent (straighter than a grasp pose)
* j5 = +0.80 → wrists pitched down so grippers face the table

Overridable per task under `random_dance.left_home` / `right_home` in
the yaml.

---

## Example: IK to (x = ±0.30, y = 0.20, z ≈ 0.93) using `mode: ik_debug`

Running with `ik_debug_hover: 0.20`, `ik_debug_forward: 0.20` and
**orientation set to whatever the home EE currently holds** (much more
robust than hard-constraining fingers down) we get:

```
[ik-debug] table_top_z = 0.731  hover_z = 0.931  forward_y = 0.200
[ik-debug] left  target world pos = [-0.300, +0.200, +0.931]
[ik-debug] right target world pos = [+0.300, +0.200, +0.931]
[ik-debug] left  qpos = [-0.030, +2.776, +2.359, +0.417, +0.001, +0.006]
[ik-debug] right qpos = [-0.014, +2.768, +2.347, +0.421, +0.017, +0.006]
[ik-debug] left  EE reached = [-0.291, +0.198, +0.914]   error ≈ 20 mm
[ik-debug] right EE reached = [+0.289, +0.197, +0.946]   error ≈ 18 mm
```

Interpretation:

1. IK has a large residual (~2 cm) because we currently **don't** do a
   post-IK convergence / re-plan loop — the reported `qpos` is just what
   curobo's first plan converged to. For visually smooth dancing this is
   fine; for precise Cartesian targets you'd want to re-invoke
   `left_plan_path` with the result as seed, or tighten the planner's
   tolerance.
2. Both arms chose **j2 ≈ +2.77 and j3 ≈ +2.35** to reach forward &
   down. Those are very different from the dance home (+0.60 / +0.40) —
   the whole shoulder is rotated most of the way forward and the elbow
   nearly folded. This is a reachable but aggressive pose; values close
   to joint limits can be a hint that the target is near the workspace
   boundary.
3. j1 stayed close to 0 for both arms: the IK didn't need any outward
   yaw to place the hand at that x coordinate, because the tangential
   direction from the shoulder joint already points well into the
   target's quadrant.

---

## Where each number comes from

| Number | Source |
|---|---|
| Robot base `(0, -0.65, 0)` | `assets/embodiments/aloha-agilex/config.yml: robot_pose` |
| Base quaternion `[0.707, 0, 0, 0.707]` | same file — 90° rotation about +z, hence URDF "left" = world −x |
| Table dims (1.2 × 0.7 × 0.05) | `envs/_base_task.py::create_table_and_wall` |
| Table centre `(0, 0, 0.74)` | same, via `table_height=0.74` + `table_xy_bias` |
| Observer camera pose | `envs/camera/camera.py`, `third_person_view == "observer"` branch |
| Head / front cameras | `assets/embodiments/aloha-agilex/config.yml: static_camera_list` |
| Joint order & names | same `config.yml: arm_joints_name` |
| Joint sign conventions | Empirical — confirmed by running `ik_debug` and watching the video |

---

## Tunable knobs — recommended ranges

These are the main geometry-adjacent randomisation knobs with empirically
safe ranges for the `aloha-agilex` embodiment + the default `observer`
camera. Exceed them at your own peril (IK failures, base/tabletop
interpenetration, or visible clipping).

### `domain_randomization.random_table_height` (metres)

Symmetric jitter: per-episode `table_z_bias ∈ [−h, +h]`.

| Value | Effect |
|---|---|
| 0.00 | Table fixed at z = 0.74. Boring. |
| 0.05 | ±5 cm jitter. Barely visible; IK untouched. |
| **0.08 – 0.10** | **Recommended.** ±8–10 cm; clear visual variation with ~no IK cost. |
| 0.15 | Aggressive; task-mode IK success rate starts dipping (rough estimate: 80–90 %). Good for training generalisation. |
| ≥ 0.20 | Not recommended — table can start clipping into the robot base when sinking, and task-mode `z_rel_range` targets become unreachable when rising. |

### `task.z_rel_range` (metres, relative to tabletop)

EE sampling height above the (possibly jittered) table top. Tracks
`table_z_bias` automatically.

| Default | Typical safe window |
|---|---|
| `[0.08, 0.25]` | `[0.05, 0.30]` — below 0.05 the fingers may graze the table surface; above 0.30 the arm's reach starts failing. |

### `task.left_x_range` / `task.right_x_range` (metres, world)

| Default | Safe window |
|---|---|
| `[-0.55, -0.05]` / `[0.05, 0.55]` | `[-0.60, 0]` / `[0, 0.60]` — keep the two ranges strictly non-overlapping (≥ 5 cm gap) to prevent the two arms IK-ing to the same spot. |

### `task.y_range` (metres, world)

| Default | Safe window |
|---|---|
| `[-0.30, 0.25]` | `[-0.35, 0.30]` — table spans `y ∈ [−0.35, +0.35]`, so these limits stay strictly above the tabletop. Stretch y < −0.35 if you want the EE to reach behind the robot (IK will cope up to y ≈ −0.45). |

### `task.pose_perturb_deg` (degrees)

Wrist SO(3) perturbation around the home EE orientation.

| Value | Feel |
|---|---|
| 0 | Fingers always point the same way (old behaviour). |
| 10 | Gentle wrist wobble. |
| **20 – 25** | **Recommended.** Clearly visible variation, IK failure < 1 %. |
| 40+ | Expressive but IK failure climbs fast; last-attempt fallback to home quat kicks in often. |

### `task.max_joint_step` (radians)

Reject IK solutions whose single-joint step exceeds this. Tighter →
smoother motion, more IK retries.

| Default | Notes |
|---|---|
| 1.0 | Fine with the "best-of-N" retry strategy. Below ~0.6 you'll start seeing frequent "best span > cap" warnings. |

### `task.speed_ref_rad` + `task.stretch_cap`

`hold_substeps` per segment is stretched by `max(1, span / speed_ref_rad)`,
clamped at `stretch_cap`. Keeps peak angular speed ~constant.

| Key | Default | What to change |
|---|---|---|
| `speed_ref_rad` | 0.8 | Raise to 1.2 for faster overall motion; drop to 0.5 for a slow-motion dance. 0 disables stretching. |
| `stretch_cap` | 3.0 | Raise when you see very large waypoint jumps still rendering too fast. |

### `hold_substeps`

Base physics-step budget per keyframe segment before stretching. 80 @
dt = 1/250 s ⇒ **0.32 s per segment** when not stretched.

### `preset` (motion personality)

Thin shortcut that overrides `n_steps` and `hold_substeps` without
forcing you to re-tune every other knob. Explicit yaml values **always
win** over a preset, so you can e.g. pick `slow` but bump `n_steps` to 12.

| Preset | `n_steps` | `hold_substeps` | Feel |
|---|---|---|---|
| `default` | 30 | 80  | Baseline -- ~0.32 s per segment, lively. |
| `slow`    | 10 | 160 | Half as many key poses, each held twice as long. Calmer / more deliberate; same wall-clock per episode. Good when you want the policy to emphasise steady-state poses over transitions. |

Adding a new preset: edit the `_PRESETS` dict at the top of
`random_dance.setup_demo` in `envs/random_dance.py`.



When designing a new trajectory:

1. Forward = world +y. Camera sits at y = +0.65.
2. Table front edge = y = +0.35. Above the tabletop means z > ~0.74.
3. Left arm reaches x ∈ [−0.60, 0]; right arm reaches x ∈ [0, +0.60].
4. EE reachable box (empirical, hover height 0.15–0.30 m):
   roughly `x ∈ [±0.1, ±0.55]`, `y ∈ [−0.35, +0.30]`, `z ∈ [0.80, 1.15]`.
   Anywhere inside this is very likely IK-solvable if you keep the
   current home quaternion; outside, expect failures near the boundary.
