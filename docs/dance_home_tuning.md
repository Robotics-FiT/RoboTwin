# Tuning the `random_dance` "Dance Home" Pose

This is a hands-on log of how we tuned the per-arm 6-DoF "dance home" pose
in `random_dance` for the `aloha-agilex` embodiment. The goal of dance
home tuning is to pick a starting / centring pose that:

1. Is **easy for the observer camera to read** — the camera should see
   the full length of each arm rather than a self-occluded folded blob.
2. Sits **roughly in the middle of the IK-friendly workspace** so the
   `task` mode keyframe sampler doesn't have to fight for solutions.
3. Is **bilaterally symmetric** in world space (so demos look natural
   left-vs-right).

The final pose is an "L" — upper arm vertical up, forearm horizontal
out to the side, perpendicular to the +y robot→camera line. That makes
the robot look a bit like a rugby goalpost from the camera, which
displays the full arm length unambiguously.

> **Companion doc.** See `docs/scene_geometry.md` for the world-frame
> coordinates (robot/table/camera) and the left/right ↔ ±x mapping
> referenced throughout this document.

---

## TL;DR — final values

`task_config/random_dance.yml` (and `random_dance_slow.yml`):

```yaml
random_dance:
  # Joint order: [j1 shoulder-yaw, j2 shoulder-pitch, j3 elbow,
  #               j4, j5 wrist-roll, j6]
  left_home:  [ 1.10, 1.5708, 1.5708, 0.0,  1.5708, 0.0]
  right_home: [-1.10, 1.5708, 1.5708, 0.0, -1.5708, 0.0]
```

Visual semantics:

| Joint | Value (rad) | Meaning |
|---|---|---|
| j1 (shoulder yaw) | `±1.10` (~63°) | Outward abduction. Mirrored sign per arm. |
| j2 (shoulder pitch) | `+π/2` | Upper arm rotated from default-horizontal up to **vertical up**. |
| j3 (elbow) | `+π/2` | Forearm folded 90° from upper arm → horizontal. |
| j4 | `0` | — |
| j5 (wrist roll) | `±π/2` | Rotates the gripper ~90° around the forearm axis. **Mirrored** sign per arm so the two grippers face symmetrically; same sign on both arms makes one side look right and one side wrong. |
| j6 | `0` | — |

Approximate end-effector world positions at this home (left shoulder is
at world ~(−0.30, −0.42, 0.78), upper arm 0.25 m, forearm 0.26 m):

```
left  EE @ home ≈ (-0.57, -0.24, +1.20)
right EE @ home ≈ (+0.60, -0.28, +1.20)
```

---

## Joint geometry (corrected; fl_* arm)

Reading the URDF directly is more reliable than guessing — naive sign
intuitions have bitten us several times during this tuning session.

```
fl_joint1: axis (0,0,1)   origin (0, 0, 0.058)
fl_joint2: axis (0,1,0)   origin (0.025, 0.0006, 0.042)
fl_joint3: axis (0,1,0)   origin (-0.264, 0.0045, 0)   rpy=(-π, 0, -0.016)  ← 180° flip
fl_joint4: axis (0,1,0)   origin (0.246, -0.00025, -0.06)
fl_joint5: axis (0,0,1)   origin (0.0678, 0.0015, -0.0855) rpy=(0, 0, -0.016)
fl_joint6: axis (1,0,0)   origin (0.031, 0, 0.0855)    rpy=(-π, 0, 0)       ← 180° flip
```

Two important consequences of the rpy="π" flips on link3 and link6:

* **j3 = 0 does NOT mean "forearm continues along upper arm"**, because
  link3's frame is pre-rotated 180° around x. At j3 = 0 the forearm
  actually points **anti-parallel** to the upper arm (folded all the
  way back). To get the forearm at +90° relative to the upper arm, j3
  has to rotate ~π/2 from the anti-parallel state, which empirically
  came out as **j3 = +π/2** (not 0 + small offset).
* **The signs of j2 and j3 together** decide both the elbow's world
  position and the forearm direction. In our final pose `j2 = +π/2`
  swings the upper-arm chain so it points up; `j3 = +π/2` then folds
  the forearm out perpendicular.

For `fr_*` (right arm) the URDF is **structurally identical**, not
pre-mirrored. That's why `j1` has to be sign-flipped per arm to get
symmetric outward abduction, and (surprisingly) `j5` has to be
sign-flipped too to get symmetric wrist roll. `j2`, `j3`, `j4`, `j6`
keep the **same** sign on both arms.

> **The legacy "joint semantics" table in `scene_geometry.md`** describes
> directional intuitions ("+ swings forward", etc.) that were measured
> at the *original* spread-forward home `[0.30, 0.60, 0.40, 0, 0.80, 0]`.
> Those intuitions don't generalise once you push joints toward π/2 or
> beyond, because pose-dependent kinematic coupling becomes large. Use
> the URDF axes + rpy and the `home_debug` mode to verify rather than
> reasoning from the table.

---

## Joint value ranges

There are four different "ranges" to keep straight for each joint.

### URDF hard limits

Read from
`assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf`:

```
fl_joint1..6: <limit lower="-10" upper="10" effort="100" velocity="1000" />
fr_joint1..6: <limit lower="-10" upper="10" effort="100" velocity="1000" />
```

These are **effectively unlimited** (±10 rad ≈ ±573°). This is a RoboTwin
convention — the real arx5 hardware has tight mechanical limits (see
"physically plausible" column below) but the simulation URDF leaves them
open so IK is free to explore. Do **not** treat these as a real range.

### curobo retract config

From `curobo_{left,right}_tmp.yml`:

```
retract_config = [0, 0, 0, 0, 0, 0, 0.04, 0.04]
```

All six arm joints retract to 0. This is what curobo's IK gravitates
toward when multiple solutions exist (and also the reason you sometimes
see a candidate "2π away" from the dance home — the retract pulls the
solution toward 0 even if the seed was farther out).

### Physically plausible range (real arx5 hardware)

Approximate values for the arx5 mechanical limits, useful when checking
whether an IK solution is realistically reachable:

| # | Joint | Plausible range | Notes |
|---|---|---|---|
| j1 | shoulder yaw | ≈ [−π, +π] | Full rotation around vertical axis is common on arx5. |
| j2 | shoulder pitch | ≈ [0, +π] | Cannot pitch below horizontal inwards (collides with base). |
| j3 | elbow | ≈ [0, +π] | Folding beyond this starts self-colliding with the upper arm. |
| j4 | wrist pitch | ≈ [−π/2, +π/2] | Forearm's orientation limits this. |
| j5 | wrist roll | ≈ [−π, +π] | Continuous-rotation wrist; no hard stop on hardware. |
| j6 | gripper twist | ≈ [−π, +π] | Continuous-rotation gripper joint. |

These are **not enforced** by SAPIEN (because the URDF says ±10), so
random / IK-returned joint values outside them will still execute; they
will just look unphysical in the video (elbow bent backwards, etc.).

### Recommended ranges for `random_dance`

Empirical, derived from staring at 10 sample episodes and the safe-area
of IK convergence. Values outside these are not forbidden, just likely
to produce visually weird frames or low IK yield.

| # | Joint | Dance home | Recommended variation around home | Comment |
|---|---|---|---|---|
| j1 | shoulder yaw | `±1.10` | ±1.2 rad | URDF gives full rotation but past ±1.4 the hand starts swinging into the robot's own body. |
| j2 | shoulder pitch | `+π/2` | ±1.0 rad around π/2 | Keep above ~0.3 to avoid "arm swinging down past the base". |
| j3 | elbow | `+π/2` | [0, π] | Going beyond π puts the elbow "inside out"; curobo occasionally returns such solutions (see `max_joint_step`). |
| j4 | wrist pitch | `0` | ±1.2 rad | Far outside hits self-collision with forearm. |
| j5 | wrist roll | `±π/2` | ±π around home | Continuous joint; large values are fine. |
| j6 | gripper twist | `0` | ±π/2 | Currently under-excited (0.12 std) because IK alone doesn't drive it. Increase `task.pose_perturb_deg` to wake it up. |

### Observed ranges in collected data

From `data/random_dance/random_dance/` (10 episodes, 4049 timesteps):

| # | Left range | Right range | Left std | Right std | Comment |
|---|---|---|---|---|---|
| j1 | 2.32 | 2.44 | 0.47 | 0.47 | Well covered, symmetric. |
| j2 | 4.14 | 3.01 | 0.63 | 0.51 | Left has a 2π-equivalent outlier episode. |
| j3 | **5.03** | 2.74 | 0.92 | 0.52 | Same outlier episode (curobo picked an "elbow flipped" solution). |
| j4 | 3.29 | 2.36 | 0.59 | 0.46 | Well covered. |
| j5 | 2.80 | 2.89 | 0.54 | 0.55 | Well covered, symmetric. |
| j6 | **0.98** | **0.91** | **0.12** | **0.12** | **Under-covered** — IK barely touches this joint; bump `pose_perturb_deg` to fix. |

To reproduce these stats:

```bash
conda activate RoboTwin
python -c "
import h5py, numpy as np, glob
files = sorted(glob.glob('data/random_dance/random_dance/data/episode*.hdf5'))
all_l, all_r = [], []
for f in files:
    with h5py.File(f) as h:
        all_l.append(h['joint_action']['left_arm'][:])
        all_r.append(h['joint_action']['right_arm'][:])
L, R = np.concatenate(all_l), np.concatenate(all_r)
for j in range(6):
    c = L[:,j]
    print(f'L j{j+1}: min={c.min():+.3f} max={c.max():+.3f} std={c.std():.3f} range={c.max()-c.min():.3f}')
"
```

### Mirroring rules (summary)

When setting `left_home` / `right_home` for a symmetric-looking pose:

| Joint | Mirror sign? |
|---|---|
| j1 | **Yes** (shoulder yaw flips left↔right) |
| j2 | No (both arms pitch the same direction in world frame) |
| j3 | No |
| j4 | No |
| j5 | **Yes** (URDF is not pre-mirrored, so same numeric value rotates both grippers the same direction in world space — one side will look wrong) |
| j6 | No |

---

## The `home_debug` mode

`mode: home_debug` (in `task_config/random_dance.yml`, under
`random_dance:`) drives the robot to `left_home` / `right_home` and
holds, with no IK and no random sampling. It also prints the actual EE
poses so you can compare against shoulder + arm-length geometry.

```yaml
random_dance:
  mode: home_debug
  home_debug_gripper: 1.0   # 0 = closed, 1 = open, null = current value
  left_home:  [ ..., ..., ..., 0.0, ..., 0.0]
  right_home: [ ..., ..., ..., 0.0, ..., 0.0]
```

Sample log:

```
[home-debug] requested left  home qpos = [1.1, 1.571, 1.571, 0.0, 1.571, 0.0]
[home-debug] requested right home qpos = [-1.1, 1.571, 1.571, 0.0, -1.571, 0.0]
[home-debug] left  EE @ home = [-0.569, -0.241, +1.205]
[home-debug] right EE @ home = [+0.599, -0.278, +1.204]
[home-debug] grippers set to left=1.00, right=1.00 (0=close, 1=open)
[home-debug] (shoulder approx: left=(-0.30, -0.42, 0.78), right=(+0.30, -0.42, 0.78); upper arm ~0.25m, forearm ~0.26m)
```

The `(shoulder approx, upper arm ~0.25m, forearm ~0.26m)` line is the
ruler you'll use throughout tuning.

> **Why a dedicated debug mode?** `mode: ik_debug` first drives to
> dance home and *then* runs IK to a fixed table-surface point — so the
> "stable" pose you see in the video is the IK solution, not the dance
> home. We initially wasted a round of tuning thinking the home values
> were ignored, when actually we just couldn't see them. `home_debug`
> exists because of that misadventure.

---

## The verification ruler

For any candidate `left_home`, compute:

```
shoulder_left ≈ (-0.30, -0.42, 0.78)    # constant; from URDF + base pose
delta         = EE - shoulder_left
distance      = |delta|
```

Match `distance` against geometry expectations:

| Pose | Expected `|delta|` | EE.z (left shoulder z = 0.78) |
|---|---|---|
| Arm fully straight, vertical up | ~0.51 m (= 0.25 + 0.26) | ~1.29 m |
| Arm fully straight, vertical down | ~0.51 m | ~0.27 m |
| **L-shape (target):** upper arm up, forearm horizontal | **~0.36 m** (= √(0.25² + 0.26²)) | **~1.03 m** |
| Both segments folded together (zero-length collapse) | ~0.01 m | ~0.78 m |
| Anti-parallel along the same line (j3=0 with link3 flip) | ~0.01 m | ~0.78 m |

In intermediate poses, if `EE.z` jumps from 1.03 to 1.20 after a tweak
that should *only* rotate around the forearm axis (e.g. j5), you have
not actually changed the dance home shape — you have changed where the
gripper body offsets sit relative to the wrist joint. The EE point is
**not** the wrist; it is the gripper's tip frame.

For a 90° L-shape specifically, the **dihedral angle** θ between upper
arm and forearm satisfies the law of cosines:

```
|delta|² = a² + b² + 2 a b cos(θ)        where a = 0.25, b = 0.26
```

(The `+` sign instead of `−` accounts for the link3 rpy="π" flip,
which makes "j3 = 0" the anti-parallel configuration, i.e. θ_geom = 0
in this convention rather than π.) So:

```
cos(θ) = (|delta|² − a² − b²) / (2 a b)
```

| Observed `|delta|` | θ | Interpretation |
|---|---|---|
| 0.157 m | ~144° | forearm folded much more than 90° |
| 0.228 m | ~127° | forearm slightly folded back |
| **0.361 m** | **~90°** | **L-shape (target)** |
| 0.466 m | ~60° | forearm almost continuing the upper arm |

This is the formula we used to extrapolate j3 between trials.

---

## Tuning history (as it actually happened)

This is preserved verbatim because the failure modes here are
genuinely instructive — every other team that tries to tweak this pose
will hit at least one of them.

```
[ 0.30, 0.40, 0.80, 0.0, 0.80, 0.0]            # folded forward, original
[ 1.10, 0.10, 0.10, 0.0, 0.10, 0.0]            # spread open, both arms ~horizontal outward
[ 1.10,-1.5708,-1.5708, 0.0, 0.0, 0.0]         # collapsed at shoulder (|d|=0.16)
[ 1.10,+1.5708, 0.0,    0.0, 0.0, 0.0]         # upper-arm up, forearm folded back (|d|=0.23, θ~127°)
[ 1.10,+1.5708,-1.5708, 0.0, 0.0, 0.0]         # angle widened wrong way (|d|=0.16, θ~144°)
[ 1.10,+1.5708,+1.5708, 0.0, 0.0, 0.0]         # L-shape! (target geometry; wrist not yet rolled)
[ 1.10,+1.5708,+1.5708, 0.0,+1.5708,0.0]       # both arms wrists rolled SAME way
                                               #   -> picture-right OK, picture-left visually wrong
[-1.5708 / +1.5708 swap]                       # mirrored the wrong way -> BOTH arms wrong
[+1.5708 / -1.5708]                            # final, symmetric
```

### Lessons

1. **Don't trust j-direction intuition past ~30°.** The `link3 rpy="π"`
   twist makes "elbow positive = bend" intuition reverse beyond a
   certain point. Use `home_debug`, read `|delta|`, and apply the law
   of cosines to triangulate.
2. **Always change one knob at a time.** When we tried to set
   `j2 = -π/2, j3 = -π/2` simultaneously we got the trivial collapse
   (`|delta|=0.157`) and couldn't tell which knob was off. After we
   forced ourselves to first verify `j2` alone (j3=0), then add j3,
   the whole thing converged in two more iterations.
3. **Mirror only the joints that actually need mirroring.**
   `aloha-agilex` is **not** pre-mirrored in URDF — both `fl_*` and
   `fr_*` arms have the same axis directions and the same rpy values.
   Empirically, `j1` (shoulder yaw) and `j5` (wrist roll) need
   sign-flipping per arm; `j2`, `j3`, `j4`, `j6` keep the same sign.
4. **EE position ≠ wrist position.** A change that looks like it
   should preserve `|delta|` (e.g. wrist roll) actually shifts EE by
   ~10–15 cm because the gripper body has its own rigid offset
   relative to the last joint. Use `|delta|` as a structural sanity
   check, not as a millimetre-level match.

---

## Cookbook: tuning a different shape

Steps to adapt to a different desired posture:

1. Set `mode: home_debug` and `episode_num: 1` (faster iteration).
2. Pick joint values one at a time, starting with the proximal-most
   joint that affects the limb you care about (j1 for left/right
   spread, j2 for shoulder up/down, j3 for elbow open/close, j5 for
   wrist roll). After each change, run
   `./collect_data.sh random_dance random_dance 0` and compare
   `[home-debug] left/right EE @ home` to the expected value from the
   "verification ruler" table.
3. For symmetry: set `right_home` ≡ mirror of `left_home`. Start with
   only `j1` mirrored. If you observe asymmetric gripper orientation,
   also mirror `j5`. Keep `j2`, `j3`, `j4`, `j6` identical.
4. Once happy, switch `mode` back to `task` (or `joint`/`ik_debug` as
   needed). The same `left_home` / `right_home` is reused as the
   IK seed in `task` mode, so the chosen pose also influences the
   shape of the dance.
5. Sync the values to `task_config/random_dance_slow.yml` (the slow
   preset re-uses the same home).

---

## Related knobs

* `random_dance.task.pose_perturb_deg` — adds a random wrist quaternion
  perturbation around the home EE orientation (in `task` mode). Higher
  values give more visual variety; IK failures climb past ~45°.
* `random_dance.task.max_joint_step` — caps single-joint step between
  consecutive IK waypoints. If the home pose is far from the centre of
  curobo's IK-easy region, you may need to widen this from `1.0` to
  `1.5` to keep success rates up.
* `random_dance.gripper_toggle_p` — probability that gripper state
  changes at each keyframe in `task` / `joint` mode. Independent of
  `home_debug_gripper`, which only sets the held value during home
  inspection.

---

## File map

| What | Where |
|---|---|
| Dance home values | `task_config/random_dance.yml`, `task_config/random_dance_slow.yml` (`random_dance.left_home` / `right_home`) |
| `home_debug` implementation | `envs/random_dance.py::play_once` (the `home_debug` branch) |
| `home_debug_gripper` plumbing | `envs/random_dance.py::setup_demo` reads it; the `home_debug` branch applies it |
| URDF | `assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf` |
| Embodiment config (joint names, base pose) | `assets/embodiments/aloha-agilex/config.yml` |
| World-frame geometry overview | `docs/scene_geometry.md` |
