# Per-Link Robot Recoloring — RoboTwin (aloha-agilex)

This is a hands-on log of how we added per-link flat-color repainting
of the `aloha-agilex` (ARX5) bimanual robot in RoboTwin, what tripped
us up, and how to control it from a task yaml.

The goal is purely **visual**: when a policy / dance demo plays back as
video, the observer should be able to tell instantly which joint
segment they are looking at. We do this by overwriting the visual
material of each link with a distinctive color, **without touching
physics, collision or joint behaviour**.

The implementation lives in:

- `envs/utils/robot_coloring.py` — the recolor utility + default scheme
- `envs/_base_task.py::load_robot` — wires it into the URDF load path
- `task_config/*.yml` — opt-in knobs (`recolor_robot`, `recolor_scheme`, …)

---

## TL;DR — How to use it

In any task yaml under `task_config/`:

```yaml
# Turn on per-link recoloring (off by default)
recolor_robot: true

# Optional: override the per-link rules. Same format as DEFAULT_SCHEME
# in envs/utils/robot_coloring.py. First-match-wins on the link name.
# recolor_scheme:
#   - { match: link4, color: [0.95, 0.15, 0.15] }   # forearm -- red
#   - { match: camera, color: [0.95, 0.95, 0.95] }  # camera mount -- white

# Optional: catch-all color for links matched by no rule. ``null`` (the
# default) leaves unmatched links with their original material.
# recolor_default_color: [0.6, 0.6, 0.6]

# Optional: keep the dae's baked diffuse texture (NOT recommended; see
# below). Defaults to true since v2.
# recolor_clear_texture: false

# Optional: print one line per repainted link at load time.
# recolor_verbose: true
```

That's it. The next `bash collectData.sh <task> ...` run will render
the new colors into the video.

---

## Final default scheme

The `DEFAULT_SCHEME` in `envs/utils/robot_coloring.py` is designed so
that a single still frame from the head camera lets you name every
joint:

| Rule (`match`)                | Region              | Color   |
|-------------------------------|---------------------|---------|
| `link1`                       | shoulder yaw        | blue    |
| `link2`                       | shoulder pitch      | purple  |
| `link3`                       | elbow               | cyan    |
| `link4`                       | forearm             | red     |
| `link5`                       | wrist 1             | orange  |
| `link6`                       | wrist 2             | yellow  |
| `link7`, `link8`              | gripper fingers     | green   |
| `fl_base_link`, `fr_base_link`| shoulder mount      | magenta |
| `camera`                      | head-camera tower   | white   |
| `wheel`                       | drive wheels        | near black |
| `castor`                      | castor wheels       | dark grey |
| `box1` / `box2`               | torso boxes         | mid / dark grey |
| `base_link`                   | mobile base chassis | dark grey |
| `inertial`                    | inertial-only links | dark grey |

Rule ordering matters because the matcher is first-match-wins.

---

## Per-link geometry (axis-aligned, in the link's own frame)

Sizes below are the **axis-aligned bounding box** of each link's *visual*
mesh, after applying the URDF `<origin rpy=...>` of the visual but
**before** any joint transform — i.e. they describe the shape of the part
itself, not where it ends up in the world. All values are in mm and were
read directly from
`assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf`
(visual `<mesh>` files under `aloha_maniskill_sim/meshes/`) by computing
`vertices.max(0) - vertices.min(0)` after the visual rpy.

The aloha-agilex URDF defines four arm chains:
`fl_*` (front-left, our visible left arm),
`fr_*` (front-right),
`lr_*` and `rr_*` (a second pair, lower-rear in the URDF; uses the
"back" finger mesh `back_link7.dae` instead of `link7.dae`).
`fl_*` / `fr_*` / `lr_*` / `rr_*` share the same per-segment meshes
except for `link7` vs `back_link7`, so only `fl_*` is listed.

### Mobile base & torso

| Link                     | X (mm) | Y (mm) | Z (mm) | Mesh                          | Notes |
|--------------------------|-------:|-------:|-------:|-------------------------------|-------|
| `base_link`              | 685    | 570    | 161    | `tracer_base_link.dae`        | mobile chassis (visual rpy = 1.57,0,0; values shown after rotation) |
| `right_wheel_link`       | 121    | 84     | 121    | `tracer_wheel.dae`            | drive wheel (∅ ≈ 121 mm, width 84 mm) |
| `left_wheel_link`        | 121    | 84     | 121    | `tracer_wheel.dae`            | drive wheel |
| `fl_castor_link` etc. ×4 | 97     | 85     | 84     | `castor_joint.dae`            | castor swivel housing |
| `fl_wheel_link` etc. ×4  | 75     | 75     | 53     | `castor.dae`                  | castor wheel itself |
| `box1_Link`              | 578    | 476    | 53     | `box1_Link.STL`               | thin lower torso plate |
| `box2_Link`              | 679    | 700    | 787    | `box2_Link.dae`               | upper-torso shell (mesh is one big shaped part, not a small box) |

### Head camera tower

| Link               | X (mm) | Y (mm) | Z (mm) | Mesh                       | Notes |
|--------------------|-------:|-------:|-------:|----------------------------|-------|
| `camera_base_link` | 94     | 70     | 647    | `camera_base_link.STL`     | the tall vertical pole (this is what dominates the silhouette) |
| `camera_link1`     | 78     | 102    | 36     | `camera_link1.dae`         | horizontal arm |
| `camera_link2`     | 78     | 102    | 36     | `camera_link2.dae`         | head holding the front cam |
| `left_camera`      | 25     | 90     | 25     | `d435.dae`                 | RealSense D435 stub on `fl_link6` (also `right_camera` on fr) |

### Single arm (`fl_*`; `fr_*` / `lr_*` / `rr_*` identical except `link7`)

| Link              | X (mm) | Y (mm) | Z (mm) | Mesh             | Region              |
|-------------------|-------:|-------:|-------:|------------------|---------------------|
| `fl_base_link`    | 60     | 60     | 63     | `base_arm.dae`   | shoulder mount cube |
| `fl_link1`        | 61     | 64     | 61     | `link1.dae`      | shoulder yaw nubbin |
| `fl_link2`        | **325**| 74     | 60     | `link2.dae`      | upper arm (long axis = X) |
| `fl_link3`        | **303**| 85     | 118    | `link3.dae`      | forearm-front (long axis = X) |
| `fl_link4`        | 116    | 62     | 104    | `link4.dae`      | forearm short segment |
| `fl_link5`        | 71     | 61     | 79     | `link5.dae`      | wrist 1             |
| `fl_link6`        | 77     | **174**| 89     | `link6.dae`      | wrist 2 (long axis = Y; this is what the recolor scheme calls the "wrist 2" yellow band) |
| `fl_link7`        | 86     | 37     | 61     | `link7.dae`      | gripper finger      |
| `fl_link8`        | 86     | 37     | 61     | `link8.dae`      | gripper finger (mirror) |
| `lr_link7` / `rr_link7` | 121 | 173 | 228 | `back_link7.dae` | rear-arm "finger"; longer/heavier than `link7.dae` |

Quick sanity check from these numbers: shoulder→EE for one arm is about
`link2.X + link3.X + link4.X ≈ 0.325 + 0.303 + 0.116 ≈ 0.74 m` of
serial reach. That's why the `ik_debug` lateral spread (`±0.45 m` from
the table-bias midline) is well inside the reachable envelope: each
shoulder is at `x ≈ ±0.30 m`, so the wrist only needs to travel
`~0.15 m` sideways to reach the target.

---

## Two bugs we hit (and the fixes)

Both bugs caused the **same** visible symptom — the video looks like
"only the shoulder turned blue, everything else looks original" — but
the root causes are completely different. We chased the first one for a
while before realising the second one existed.

### Bug #1 — naive substring match painted the camera blue

The original matcher just did `match.lower() in link_name.lower()`.
That sounds fine until you remember that the URDF also contains
`camera_link1` and `camera_base_link`. With the scheme above:

- Rule `link1` (blue) appears **before** rule `camera` (white).
- Substring test `"link1" in "camera_link1"` → True.
- Camera tower gets painted blue.

The fix is in `_matches_link(...)` in `envs/utils/robot_coloring.py`:

- Exact equality wins immediately.
- For "joint-style" rules whose `match` looks like `linkN` / `linkNN`,
  the link name's **last underscore-separated token** must equal the
  rule, AND the **second-to-last token** must be a known arm prefix
  (`fl`, `fr`, `lr`, `rr`). That makes `link1` match `fl_link1` /
  `fr_link1` but never `camera_link1`.
- For multi-token rules (e.g. `base_link`), we require a `_match` suffix
  of the link name. That keeps `base_link` from swallowing
  `camera_base_link`.
- Other single-token rules (`camera`, `wheel`, `box1`, …) must appear
  as a full underscore-separated token of the link name.

After this fix the camera tower stayed white, BUT the arms still looked
mostly unchanged. That was bug #2.

### Bug #2 — baked diffuse textures hid the flat `base_color`

The bundled aloha-agilex dae meshes carry a **baked diffuse texture**
(the "AGILE-X" branded paint job). SAPIEN's PBR shader computes:

```
final_albedo = base_color_factor × sample(base_color_texture, uv)
```

So when we write `mat.base_color = [r, g, b, 1]` but leave the texture
attached, the rendered color is roughly `[r, g, b] × texture`. The
consequences:

- `fl_link1`'s baked texture is the AGILE-X **blue** paint job, and our
  scheme writes blue on top of it. Result: looks blue. We thought we
  were succeeding.
- The other arm links (`fl_link2..7`) have **near-black** plastic
  textures. `near_black × anything ≈ near_black`. They looked black no
  matter what color we wrote.
- The camera tower's mesh is **white plastic with no texture**, so the
  flat `base_color` showed through directly. That misled us into
  thinking the pipeline was wired up correctly.

We verified the assumption with a small SAPIEN script that listed each
`RenderMaterial`'s attributes — every arm part reported
`base_color_texture = SET`, while `base_link` was textureless.

**Fix**: clear the texture handles on the material **at the same time**
we write the flat color. Concretely, in `_mutate_material_inplace(...)`
we call (best-effort, since binding names vary between SAPIEN versions):

```
mat.set_base_color_texture(None)
mat.set_metallic_roughness_texture(None)
mat.set_normal_texture(None)
mat.set_diffuse_texture(None)
mat.set_metallic_texture(None)
mat.set_roughness_texture(None)
mat.set_emission_texture(None)
```

This makes the PBR shader fall back to pure flat shading on the
modified material, and the scheme colors finally show up as intended.

#### Sub-bug 2.5 — the yaml-default override

After flipping the function default to `clear_texture=True` in
`recolor_robot(...)`, the video **still** showed the original behaviour.
The reason was that `envs/_base_task.py::load_robot` read
`recolor_clear_texture` from the yaml with a hard-coded default of
`False` and explicitly passed it through:

```python
clear_texture = bool(kwags.get("recolor_clear_texture", False))
recolor_robot(..., clear_texture=clear_texture, ...)
```

So the wrapper was silently shadowing the function's new default with
the old "keep textures" behaviour. Lesson: when a function has an
opinionated default, every wrapper that re-supplies the kwarg must
agree with it. We changed the wrapper's default to `True` as well, with
a comment explaining why; a user that wants the legacy behaviour can
still opt back in via `recolor_clear_texture: false`.

---

## SAPIEN-specific gotchas worth remembering

These are quirks of SAPIEN 3.0.0b1 we relied on (and got bitten by)
while implementing the recolor utility. The full version of each note
is in `envs/utils/robot_coloring.py`'s docstring; here is the short
version.

1. **Mutate, don't reassign.** `part.material = new_material` is a
   silent no-op on SAPIEN 3.0.0b1's Python bindings — the attribute
   write succeeds but never reaches the renderer. Worse, doing it on
   only some parts trips SAPIEN's `Triangle shape contains multiple
   parts with different materials` consistency check. We mutate
   `mat.base_color`, `mat.metallic`, `mat.roughness` (and clear
   textures) in-place on the existing material handle. That actually
   propagates to the renderer.

2. **A `RenderShapeTriangleMesh` may have multiple `parts`.** GLTF /
   dae files frequently bundle several sub-meshes with their own
   materials. We iterate `shape.get_parts()` (falling back to
   `shape.parts`) and mutate every part's material. For primitive
   shapes (box / capsule / sphere / plane) there are no parts; we use
   `shape.material` directly.

3. **Property vs setter name varies.** Some builds expose `base_color`
   as an assignable property, some only as `set_base_color(...)`. The
   helper tries the property first and falls back to the setter.

4. **`base_color_texture` is what actually drives albedo.** If you
   write a color but leave the texture, you have not changed what gets
   rendered (see Bug #2). Always clear the texture (or accept the
   modulation) when you want flat shading.

---

## Files touched

- `envs/utils/robot_coloring.py` — new module: matcher, default scheme,
  `recolor_robot(...)`, `_mutate_material_inplace(...)`.
- `envs/_base_task.py::load_robot` — opt-in hook reading
  `recolor_robot`, `recolor_scheme`, `recolor_default_color`,
  `recolor_clear_texture` (default `True`) and `recolor_verbose` from
  the task yaml.
- `task_config/random_dance.yml`, `task_config/random_dance_slow.yml`,
  `task_config/demo_clean.yml`, `task_config/demo_randomized.yml` —
  enabled `recolor_robot: true` and use `DEFAULT_SCHEME`.
- `envs/camera/camera.py` — unrelated tweaks (see git history).

---

## How to verify locally

The fastest no-renderer check is to load the URDF, run the recolor and
read materials back:

```python
import sapien, sapien.render as sr
from envs.utils.robot_coloring import recolor_robot

scene = sapien.Scene()
loader = scene.create_urdf_loader(); loader.fix_root_link = True
art = loader.load('assets/embodiments/aloha-agilex/urdf/'
                  'arx5_description_isaac.urdf')
recolor_robot(art)

for link in art.get_links():
    rb = link.entity.find_component_by_type(sr.RenderBodyComponent)
    if rb is None: continue
    for shape in rb.render_shapes:
        for p in shape.get_parts():
            m = p.material
            tex = 'SET' if m.base_color_texture else None
            print(link.get_name(), [round(x,2) for x in m.base_color],
                  'tex=', tex)
```

Every arm link should print `tex= None` and the expected RGB triple
from the scheme. For an end-to-end check, run any task with
`recolor_robot: true`:

```
bash collectData.sh random_dance 0 1 0
```

and inspect `data/random_dance/random_dance/video/episode0.mp4`. The
shoulder should be a flat solid blue with **no** "AGILE-X" logo
visible — that is the easiest one-glance confirmation that the texture
clearing actually fired.

---

## Hiding decorative links (`hide_robot_dressing`)

Recoloring solved "I can't tell the joints apart"; it did NOT solve
"the head_camera view is blocked by stuff that isn't actually part of
this robot". The aloha-agilex URDF carries several visual-only meshes
that are scenery rather than functional joints — most importantly, an
unused **rear pair of arms** (`lr_*` and `rr_*`) and a **decorative
head-mounted RealSense tower** (`camera_base_link` + `camera_link1..3`).
None of them are commanded by the policy, but they all happily render
into the head_camera image as foreground occluders.

The implementation is in the same module as recoloring:

- `envs/utils/robot_coloring.py::hide_robot_dressing(...)` — the hider
  + `DEFAULT_HIDE_RULES`
- `envs/_base_task.py::load_robot` — opt-in hook reading
  `hide_robot_dressing` and `hide_robot_dressing_rules` from the yaml

### TL;DR — How to use it

```yaml
# Turn on dressing-hide (off by default)
hide_robot_dressing: true

# Optional: override the per-link match list. Same token-aware matcher as
# the recolor scheme (so ``box1_link`` doesn't swallow ``camera_base_link``).
# hide_robot_dressing_rules:
#   - box1_link
#   - camera_base_link
#   - camera_link1
#   - camera_link2
#   - camera_link3
#   - inertial_link
#   - lr_base_link
#   - lr_link1
#   - lr_link2
#   - lr_link3
#   - lr_link4
#   - lr_link5
#   - lr_link6
#   - lr_link7
#   - rr_base_link
#   - rr_link1
#   - rr_link2
#   - rr_link3
#   - rr_link4
#   - rr_link5
#   - rr_link6
#   - rr_link7
#   # extras you might also want to hide:
#   - box2_link   # shoulder pedestal under the arms
#   - base_link   # mobile chassis
#   - wheel
#   - castor
```

### Default hide list

| Rule              | What it hides                      | Why |
|-------------------|------------------------------------|-----|
| `box1_link`       | the central lower-torso pillar     | dominates the head_camera silhouette |
| `camera_base_link`+`camera_link1..3` | decorative D435 tower mesh | cosmetic; not the real head_camera |
| `inertial_link`   | inertia-only helper link           | no visual anyway, kept for clarity |
| `lr_base_link`+`lr_link1..7`         | unused rear-left arm    | URDF leftover, never commanded |
| `rr_base_link`+`rr_link1..7`         | unused rear-right arm   | URDF leftover, never commanded |

What is **kept** by default and why:

- `base_link`, `wheel`, `castor`, `footprint` — the mobile chassis +
  wheels. Removing them makes the robot look like it is floating.
- `box2_link` — the **shoulder pedestal** directly under the arms.
  Despite the name "box", its mesh is a wide, low pedestal slab
  (`box2_Link.dae`, see the geometry table above; ~700 mm wide × 787 mm
  tall once you account for its visual rpy) that visually anchors the
  arms to the chassis. It does NOT obstruct the head_camera, so it
  stays.
- `fl_*`, `fr_*` — the two arms the policy actually commands.

If you need a more aggressive hide (e.g. you want a totally floating
pair of arms on a clean background for synthetic pretraining), copy the
list above and append `box2_link` / `base_link` / `wheel` / `castor` /
`footprint`.

### The matcher reuses the token-aware logic from recoloring

`hide_robot_dressing_rules` go through the same `_matches_link(...)`
function used by the recolor scheme. That means:

- `box1_link` matches `box1_Link` (case-insensitive) but does **not**
  match `camera_base_link` or `box2_Link`.
- `lr_link1` matches `lr_link1` but does NOT match `fl_link1`.
- Multi-token rules require a `_match` suffix of the link name (so
  `base_link` matches `base_link` itself but not `fl_base_link`).

This means it is safe to copy/paste any subset of `DEFAULT_HIDE_RULES`
into a yaml override without accidentally hiding shoulders or wheels.

### Why hiding is harder than it looks on SAPIEN 3.x

Naive "iterate `RenderBodyComponent.render_shapes` and call
`set_visibility(0)` on each shape" works on **some** SAPIEN builds but
silently no-ops on others. Symptoms:

- The verbose output prints `0 shape(s) hidden` for every matched link
  even though `_matches_link` is happy. → The link's visuals live on a
  different component (`RenderShapeComponent`, `VisualBodyComponent`,
  …), not on the one we expected, OR they live on a component that
  doesn't expose `render_shapes` as an iterable Python attribute at
  all.
- `shape.visibility = 0` succeeds as a Python attribute write but the
  renderer keeps drawing the shape, because that build only honours
  `set_visibility(...)` (or only honours `disable()` on the parent
  component).

`hide_robot_dressing` therefore tries every reasonable path and
counts how many it actually managed to silence:

1. Walk **every** component on the entity whose class name contains
   `Render` or `Visual`.
2. For each such component, try `render_shapes` first, then
   `get_render_shapes()`, then `visual_shapes`, then
   `get_visual_shapes()`. Whatever comes back is iterated.
3. For each shape: try property write `visibility = 0`, then
   `set_visibility(0)`, then `disable()`.
4. If the component still looks alive (e.g. the build exposes neither
   shapes nor a per-shape visibility), fall back to **component-level**
   `disable()` / `set_disabled(True)`, and as a last resort
   `entity.remove_component(component)`.

Verbose mode prints `N shape(s), K component(s) disabled; comps=[…]` so
you can tell which path actually fired on your build. If you see `0,
0` but the link is still visible, the link's visuals live on yet
another component class — extend `_iter_render_components(...)` rather
than re-reading `render_shapes` from a different angle.

### Bug we hit and the fix

**Symptom**: ran with `hide_robot_dressing_rules: [box1_link,
camera_base_link, lr_link1, …]` + `hide_robot_dressing_verbose: true`,
got per-link logs like `box1_link -> hidden (0 shape(s), …)`, and the
collected video still showed the central pillar and the rear arms.

**Root cause**: this build's link entities had a `RenderBodyComponent`
on which `render_shapes` evaluated to an empty tuple, AND the actual
visual meshes were attached as separate `RenderShapeComponent`
instances on the **same** entity. Iterating only the body's
`render_shapes` thus always returned 0.

**Fix**: the multi-component / multi-attribute fallback chain
described above. After the change every matched link reports a
non-zero `(shape(s), component(s) disabled)` pair, and the head_camera
view is finally clear.

### Related sub-bug — `box2_link` looks like "the chassis"

The first iteration of `DEFAULT_HIDE_RULES` also hid `box2_link`
because the name suggests "another box on the torso". In the rendered
scene this looks dramatic: with `box2_link` hidden, the arms appear
to float just above the wheels, with no visible attachment.

The user wanted to keep that piece (the shoulder pedestal) visible,
even though the URDF *name* sounds incidental. Lesson: link names like
`box1` / `box2` don't tell you what the mesh actually looks like —
always cross-reference with the bbox table in the "Per-link geometry"
section above before deciding what's "decorative".

### How to verify locally

Pre-render check (no GPU video, just the head_camera PNG that
`generate_pic: true` saves):

```yaml
# in your task config
generate_pic: true
hide_robot_dressing: true
hide_robot_dressing_verbose: true
```

```
bash collect_data.sh random_dance <task_config> 0
```

Then open `data/<task>/<task_config>/images/episode0_head.png` and
confirm:

- the central pillar (`box1_Link`) is gone,
- there's only one pair of arms (no rear `lr_*`/`rr_*` ghost arms),
- the chassis + wheels + shoulder pedestal (`box2_Link`) are still
  there.

If anything looks wrong, re-run with `hide_robot_dressing_verbose:
true` and check the per-link `N shape(s), K component(s)` summary —
zeros there mean the matcher hit the link but no rendering path was
silenced (extend `_iter_render_components`).
