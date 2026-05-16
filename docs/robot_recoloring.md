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
