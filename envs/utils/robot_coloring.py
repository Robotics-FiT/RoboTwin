"""Utility for repainting robot links with distinct flat colors.

This module is intentionally side-effect free: it only mutates the
``RenderMaterial`` of visual shapes already attached to the loaded URDF.
It does NOT alter physics, collision, or joint behaviour.

Typical usage from ``_base_task.load_robot``::

    from envs.utils.robot_coloring import recolor_robot
    recolor_robot(self.robot.left_entity, scheme=cfg)
    recolor_robot(self.robot.right_entity, scheme=cfg)

The ``scheme`` is a list of ``{match: <substr>, color: [r,g,b]}`` rules.
The first rule whose ``match`` substring appears in the link name wins.
A ``default`` color (optional) is applied to links that match no rule.
RGB values are in [0, 1].

Notes on SAPIEN internals (3.x):
  * Each articulation link's entity carries a ``RenderBodyComponent``;
    that component exposes ``render_shapes`` (one per ``<visual>`` in
    the URDF). Each shape owns a ``RenderMaterial`` with ``base_color``,
    ``metallic`` and ``roughness`` writable at runtime.
  * URDF meshes often carry a baked diffuse texture; setting
    ``base_color`` alone is then invisible. We clear the texture via
    ``set_base_color_texture(None)`` so the flat color actually shows.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import sapien.core as sapien
import sapien.render as sr


# Default color scheme for the aloha-agilex (ARX5) bimanual robot. Order
# matters: the first matching ``match`` substring wins. The matching is
# case-insensitive substring, so ``link7`` also catches ``fl_link7``.
DEFAULT_SCHEME: List[dict] = [
    # --- grippers (two finger links per arm) ---
    {"match": "link7", "color": [0.10, 0.85, 0.20]},   # gripper finger -- green
    {"match": "link8", "color": [0.10, 0.85, 0.20]},   # gripper finger -- green
    # --- forearm + wrist (links 4..6) ---
    {"match": "link4", "color": [0.95, 0.15, 0.15]},   # forearm -- red
    {"match": "link5", "color": [1.00, 0.55, 0.00]},   # wrist 1 -- orange
    {"match": "link6", "color": [1.00, 0.85, 0.10]},   # wrist 2 -- yellow
    # --- shoulder + upper arm (links 1..3) ---
    {"match": "link1", "color": [0.20, 0.45, 0.95]},   # shoulder yaw -- blue
    {"match": "link2", "color": [0.55, 0.25, 0.85]},   # shoulder pitch -- purple
    {"match": "link3", "color": [0.10, 0.75, 0.85]},   # elbow -- cyan
    # --- shoulder mount / base of each arm ---
    {"match": "fl_base_link", "color": [0.95, 0.20, 0.65]},  # left shoulder mount -- magenta
    {"match": "fr_base_link", "color": [0.95, 0.20, 0.65]},  # right shoulder mount -- magenta
    # --- camera links (leave as bright white so renders still look like cameras) ---
    {"match": "camera", "color": [0.95, 0.95, 0.95]},
    # --- mobile base / torso ---
    {"match": "wheel", "color": [0.10, 0.10, 0.10]},
    {"match": "castor", "color": [0.30, 0.30, 0.30]},
    {"match": "box1", "color": [0.55, 0.55, 0.60]},
    {"match": "box2", "color": [0.40, 0.40, 0.45]},
    {"match": "base_link", "color": [0.35, 0.35, 0.40]},
    {"match": "inertial", "color": [0.35, 0.35, 0.40]},
]


def _matches_link(link_name_lc: str, match_lc: str) -> bool:
    """Token-aware match between a link name and a rule's ``match`` string.

    A naive substring test is wrong for this URDF: ``link1`` would match
    not just ``fl_link1`` but also ``camera_link1`` -- and because the
    DEFAULT_SCHEME lists ``link1`` before ``camera``, the camera mount
    ends up painted blue (the shoulder color). Same trap with
    ``base_link`` swallowing ``camera_base_link``.

    Rules:
      * exact equality wins;
      * otherwise ``match`` must be a full token of the link name,
        i.e. surrounded by ``_`` boundaries (or string start/end).
        ``link1`` matches ``fl_link1`` (``..._link1``) and
        ``link1_tip`` (``link1_...``) but NOT ``camera_link1``
        because there ``link1`` is preceded by ``camera_`` AND the
        whole token is ``link1`` -- wait, that would still match.
        We additionally require that ``match`` be either the WHOLE
        name or the LAST underscore-separated token of the name.
        That makes ``link1`` match ``fl_link1`` (last token) but
        not ``camera_link1`` (last token is ``link1`` too -> still
        matches!). So we go one step further: any rule whose
        ``match`` is a ``link[0-9]`` style joint name only matches
        the LAST token AND requires the SECOND-TO-LAST token to be
        a 2-character arm prefix (``fl``, ``fr``, ``lr``, ``rr``).
        Any other rule (e.g. ``camera``, ``wheel``, ``box1``,
        ``base_link``) uses the simple "exact-or-last-token" rule.
    """
    if not match_lc:
        return False
    if link_name_lc == match_lc:
        return True
    tokens = link_name_lc.split("_")
    last = tokens[-1] if tokens else ""
    # Joint-style match (linkN, linkNN): require an arm-prefix tier
    # so it only catches the actual arm joints, never camera_linkN.
    if (len(match_lc) >= 5 and match_lc.startswith("link")
            and match_lc[4:].isdigit()):
        if last != match_lc:
            return False
        # Need at least one more token before "linkN", and that token
        # must be a known arm prefix.
        if len(tokens) < 2:
            return False
        return tokens[-2] in ("fl", "fr", "lr", "rr")
    # ``base_link`` style: rule is itself multi-token. Treat it as a
    # suffix of the full underscore-separated name.
    if "_" in match_lc:
        return link_name_lc.endswith("_" + match_lc) or link_name_lc == match_lc
    # Single-token rule (e.g. ``camera``, ``wheel``, ``castor``,
    # ``box1``, ``inertial``): must appear as one of the tokens.
    return match_lc in tokens


def _resolve_color(link_name: str,
                   scheme: Sequence[dict],
                   default: Optional[Sequence[float]]) -> Optional[List[float]]:
    """Return the RGB triple for ``link_name``, or ``None`` to skip."""
    name_lc = link_name.lower()
    for rule in scheme:
        m = str(rule.get("match", "")).lower()
        if _matches_link(name_lc, m):
            c = rule.get("color")
            if c is None:
                return None
            return list(c)[:3]
    if default is None:
        return None
    return list(default)[:3]


def _iter_render_components(entity_or_articulation) -> Iterable:
    """Yield (link_name, render_body_component) for every link.

    Accepts a SAPIEN ``PhysxArticulation`` (URDF root) directly.
    """
    links = entity_or_articulation.get_links()
    for link in links:
        # ``link`` is a PhysxArticulationLinkComponent; its sapien.Entity
        # owns the renderable component.
        entity = link.entity
        rb = entity.find_component_by_type(sr.RenderBodyComponent)
        if rb is None:
            continue
        yield link.get_name(), rb


def recolor_robot(articulation,
                  scheme: Optional[Sequence[dict]] = None,
                  default_color: Optional[Sequence[float]] = None,
                  metallic: float = 0.0,
                  roughness: float = 0.7,
                  clear_texture: bool = True,
                  verbose: bool = False) -> None:
    """Repaint visual shapes of every link in ``articulation``.

    Parameters
    ----------
    articulation : sapien PhysxArticulation
        The loaded robot (e.g. ``self.robot.left_entity``).
    scheme : list of {match, color} dicts, optional
        Substring -> RGB triple. First match wins. Defaults to
        ``DEFAULT_SCHEME`` (sensible aloha-agilex coloring).
    default_color : RGB triple, optional
        Applied to links matching no rule. ``None`` (default) leaves
        unmatched links untouched.
    metallic, roughness : float
        PBR parameters; defaults give a flat, non-shiny look that makes
        the colors pop in screenshots.
    clear_texture : bool
        If True, attempt to clear the diffuse / metallic-roughness /
        normal textures so the flat ``base_color`` is fully visible.
        ⚠ On SAPIEN 3.x this can trigger
        ``Triangle shape contains multiple parts with different
        materials`` when a single mesh holds multiple sub-parts that
        share an underlying triangle shape (very common with the bundled
        ARX5 / aloha-agilex meshes). For that reason this is **off by
        default**; the PBR ``base_color`` modulates the texture and is
        typically visually sufficient to disambiguate links.
    verbose : bool
        Print a one-line summary per repainted link.

    Implementation notes
    --------------------
    A SAPIEN ``RenderShapeTriangleMesh`` (one per ``<visual>`` mesh in
    the URDF) may internally consist of multiple
    ``RenderShapeTriangleMeshPart`` objects -- one per material group
    inside the source mesh file (e.g. a GLTF with several
    sub-meshes/materials).  Each part has its **own** ``material``
    handle.  We learned (the hard way) two things specific to
    SAPIEN 3.0.0b1:

    1. ``part.material = some_new_material`` is a **no-op** -- the
       Python binding doesn't expose a real setter; the attribute write
       silently succeeds but never reaches the renderer.  Worse, doing
       it on only some parts trips SAPIEN's consistency check::

           Triangle shape contains multiple parts with different
           materials.

    2. ``part.material`` returns a Python handle that **does share the
       underlying C++ material state**.  Mutating attributes
       in-place (``mat.base_color = ...``, ``mat.metallic = ...``)
       persists across re-fetches and shows up in the rendered frame.

    Therefore we mutate **each part's material in place** -- never
    create new materials, never reassign ``part.material``.  All parts
    of a triangle mesh keep their original (now-mutated) materials, so
    SAPIEN's sanity check is happy and the renderer actually sees the
    new color.
    """
    if scheme is None:
        scheme = DEFAULT_SCHEME

    for link_name, rb in _iter_render_components(articulation):
        rgb = _resolve_color(link_name, scheme, default_color)
        if rgb is None:
            continue
        rgba = [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]

        n_touched = 0
        for shape in rb.render_shapes:
            # ---- collect every (part) material handle we should mutate ----
            mats = []
            getter = getattr(shape, "get_parts", None)
            parts = []
            if getter is not None:
                try:
                    parts = list(getter()) or []
                except Exception:
                    parts = []
            if not parts:
                parts = list(getattr(shape, "parts", []) or [])

            for part in parts:
                m = getattr(part, "material", None)
                if m is not None:
                    mats.append(m)

            # Primitive shapes (box / capsule / sphere / plane) have no
            # parts but expose ``shape.material`` directly.
            if not mats:
                m = getattr(shape, "material", None)
                if m is not None:
                    mats.append(m)

            for mat in mats:
                _mutate_material_inplace(mat, rgba, metallic, roughness,
                                         clear_texture=clear_texture)
                n_touched += 1

        if verbose:
            print(f"[recolor_robot] {link_name:>20s} -> {rgba[:3]} "
                  f"({n_touched} material handle(s) mutated)")


def _mutate_material_inplace(mat, rgba, metallic, roughness,
                              clear_texture: bool = False) -> None:
    """Mutate a SAPIEN ``RenderMaterial`` handle in place.

    Do NOT reassign the material on its owning shape/part -- that is a
    no-op on SAPIEN 3.0.0b1's bindings.  Only in-place attribute writes
    propagate to the renderer.
    """
    try:
        mat.base_color = rgba
    except Exception:
        # Older builds use ``set_base_color`` instead of property.
        try:
            mat.set_base_color(rgba)
        except Exception:
            pass
    try:
        mat.metallic = float(metallic)
    except Exception:
        try:
            mat.set_metallic(float(metallic))
        except Exception:
            pass
    try:
        mat.roughness = float(roughness)
    except Exception:
        try:
            mat.set_roughness(float(roughness))
        except Exception:
            pass

    if not clear_texture:
        return
    # Best-effort texture clearing. Only enable if you've confirmed your
    # meshes are single-part (otherwise harmless on this code path
    # because we no longer rebind materials, but textures may still
    # confuse SAPIEN's consistency check on some builds).
    for setter in ("set_base_color_texture",
                   "set_metallic_roughness_texture",
                   "set_normal_texture",
                   "set_diffuse_texture",
                   "set_metallic_texture",
                   "set_roughness_texture",
                   "set_emission_texture"):
        fn = getattr(mat, setter, None)
        if fn is None:
            continue
        try:
            fn(None)
        except Exception:
            pass
