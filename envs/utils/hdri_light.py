"""Estimate the dominant directional light (usually the sun) from an HDRI.

SAPIEN's path tracer consumes the environment map as image-based lighting,
which gives you soft/ambient illumination and ambient-occlusion-like contact
darkening for free but **does not cast hard shadows** (the HDRI is integrated
as a smooth distribution, not treated as a point/parallel light with a shadow
map).

To recover crisp shadows that still match the HDRI look, we analyse the
equirectangular HDR image, find the brightest blob (the sun), and turn it
into a ``add_directional_light(..., shadow=True)`` call. The light's
direction and colour are derived from the HDRI itself, so the shadows point
the same way as the visual sun in the background.

Convention used here (matches the rest of the repo and SAPIEN's default
Z-up world):

* Longitude  ``phi``  : 0 at image centre column, +x at phi=0, grows +y for
  phi>0 (i.e. image column :math:`u` maps linearly to :math:`[-\\pi, \\pi]`).
* Latitude   ``theta``: 0 at the horizon, ``+\\pi/2`` at the zenith (top
  row), ``-\\pi/2`` at the nadir (bottom row).
* World dir  ``(x, y, z)``: unit vector pointing *from the scene toward the
  sun*. The directional-light API in SAPIEN wants the light direction
  (from sun toward scene), i.e. the negative of this.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np


def _read_hdri(path: str) -> np.ndarray:
    """Read an .exr/.hdr equirectangular image into an (H, W, 3) float32 RGB.

    Uses OpenCV when available (fast, handles .exr via OpenEXR bindings
    baked into opencv-python on most platforms). Falls back to imageio for
    .hdr files. Values are linear radiance, can be >>1.
    """
    ext = os.path.splitext(path)[1].lower()

    # OpenCV needs this env var set *before* the import for OpenEXR support
    # on some builds. Setting it here only matters the very first time
    # cv2 is imported in the process.
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if img is not None:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            # cv2 returns BGR -> flip to RGB
            return np.ascontiguousarray(img[:, :, ::-1]).astype(np.float32)
    except Exception:
        pass

    # Fallback: imageio
    import imageio.v2 as imageio  # type: ignore

    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img.astype(np.float32)


def _pixel_to_world_dir(u: float, v: float, width: int, height: int) -> np.ndarray:
    """Convert equirectangular pixel (u, v) -> unit direction *toward* the
    sky hemisphere point, in a Z-up world frame.

    Equirect layout assumed:

    * column u=0       -> phi = -pi      (behind, -x half-space)
      column u=W-1     -> phi = +pi
      column u=W/2     -> phi = 0        (in front, +x direction)
    * row    v=0       -> theta = +pi/2  (zenith)
      row    v=H-1     -> theta = -pi/2  (nadir)

    Returns (x, y, z) with +z up.
    """
    phi = (u + 0.5) / width * 2.0 * np.pi - np.pi        # [-pi, pi]
    theta = np.pi / 2.0 - (v + 0.5) / height * np.pi      # [+pi/2, -pi/2]
    cos_t = np.cos(theta)
    x = cos_t * np.cos(phi)
    y = cos_t * np.sin(phi)
    z = np.sin(theta)
    return np.array([x, y, z], dtype=np.float32)


def estimate_sun_from_hdri(
    hdri_path: str,
    max_elevation_deg: float = 85.0,
    min_elevation_deg: float = 20.0,
    color_clip: float = 10.0,
    intensity_scale: float = 1.5,
    downscale_max_side: int = 1024,
    blob_window_deg: float = 4.0,
    diffuse_fallback_percentile: float = 99.5,
    auto_intensity: bool = True,
    base_intensity: float = 1.0,
    max_intensity: float = 20.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return ``(light_direction, light_color)`` for a SAPIEN directional
    light that approximates the HDRI's strongest light source.

    * ``light_direction``: unit vec pointing *from the sun toward the scene*
      (what SAPIEN's ``add_directional_light`` expects). Z-up world.
    * ``light_color``: (R, G, B), tone-mapped so values stay in a sensible
      range for SAPIEN (roughly ``[0, ~color_clip]``).

    Algorithm:
      1. Read HDRI, downscale for speed, compute per-pixel luminance.
      2. If there is a clear "sun" (a single very bright spike -- typical of
         outdoor HDRIs with a visible sun), take the brightest pixel and
         average a small angular window around it (``blob_window_deg``) to
         nail down its direction and radiance. This avoids the pitfall of a
         99.5-percentile centroid, which for a sunset HDRI spans the whole
         warm sky and drifts the sun's estimated elevation upward.
      3. Otherwise (overcast / studio / diffuse HDRIs with no distinct sun),
         fall back to a luminance-weighted centroid of the brightest
         ``diffuse_fallback_percentile`` percentile.

    Low-elevation suns are *clamped up* to ``min_elevation_deg`` (default
    20 deg) so that the resulting shadows remain visually meaningful on a
    tabletop scene. A nearly-horizontal sun (e.g. a sunset HDRI whose sun
    sits at 3 deg elevation) would cast shadows that graze parallel to the
    floor and produce almost no contrast under the robot -- not useful for
    data augmentation. Override ``min_elevation_deg`` if you want the
    physically correct direction back.

    Intensity:
      * ``auto_intensity=True`` (default): pick the directional-light
        intensity as a function of how much the sun outshines the rest of
        the sky in *this particular* HDRI. Concretely
        ``intensity = base_intensity * log1p(sun_lum / sky_lum)``, clipped
        to ``max_intensity``. This keeps bright noon HDRIs getting a
        punchy sun and dim overcast / night HDRIs getting barely any
        directional contribution -- the visual tone of each HDRI is
        preserved instead of being flattened to a single global scale.
      * ``auto_intensity=False``: fall back to the old behaviour of
        normalising the peak channel to exactly ``intensity_scale``.

    Returns ``None`` when the HDRI cannot be read.
    """
    if not os.path.exists(hdri_path):
        return None

    img = _read_hdri(hdri_path)  # (H, W, 3) linear RGB
    if img.size == 0:
        return None

    # Downscale for speed; keeps the brightest-blob centroid stable enough.
    H, W = img.shape[:2]
    scale = max(H, W) / float(downscale_max_side)
    if scale > 1.0:
        import cv2
        new_w = int(round(W / scale))
        new_h = int(round(H / scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]

    # Luminance (Rec.709). Keep non-negative.
    lum = np.clip(0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2], 0.0, None)

    # --- Branch: spikey sun vs. diffuse sky ---
    peak = float(lum.max())
    p99 = float(np.percentile(lum, 99.0))
    spikey = peak > max(10.0 * p99, 5.0)  # sun >> everything else ~ crisp sun

    # ``sun_lum_repr`` represents the representative luminance of the sun
    # blob (peak brightness for spikey case, percentile-weighted for
    # diffuse). ``sky_lum`` represents the rest of the sky (median of the
    # non-sun pixels). Both feed auto_intensity.
    sun_lum_repr = peak
    if spikey:
        y0, x0 = np.unravel_index(int(np.argmax(lum)), lum.shape)
        # Convert the chosen pixel's neighbourhood to a small angular window
        # (``blob_window_deg``). Width in pixels varies with the latitude of
        # the argmax because equirectangular images squish longitudes near
        # the poles; just use a fixed pixel radius from the angular window.
        px_per_deg_lon = W / 360.0
        px_per_deg_lat = H / 180.0
        rx = max(1, int(round(blob_window_deg * px_per_deg_lon)))
        ry = max(1, int(round(blob_window_deg * px_per_deg_lat)))
        y_lo = max(0, y0 - ry)
        y_hi = min(H, y0 + ry + 1)
        x_lo = max(0, x0 - rx)
        x_hi = min(W, x0 + rx + 1)
        sub_img = img[y_lo:y_hi, x_lo:x_hi]
        sub_lum = lum[y_lo:y_hi, x_lo:x_hi]

        # Within the window, weight by luminance to find a sub-pixel centre.
        w_sum = float(sub_lum.sum())
        if w_sum <= 0:
            return None
        ys_sub, xs_sub = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        # Weighted centroid. Longitude wraps -- but within a small window we
        # can just use raw pixel indices, no wrap correction needed.
        mean_x = float(np.sum(xs_sub * sub_lum) / w_sum)
        mean_y = float(np.sum(ys_sub * sub_lum) / w_sum)

        mean_color = np.array([
            float(np.sum(sub_img[..., c] * sub_lum) / w_sum) for c in range(3)
        ], dtype=np.float32)

        phi = (mean_x + 0.5) / W * 2.0 * np.pi - np.pi
        theta = np.pi / 2.0 - (mean_y + 0.5) / H * np.pi
        # Representative blob luminance: mean of the blob window (more
        # robust than the single peak pixel which is often a single spike).
        sun_lum_repr = float(sub_lum.mean())
        # Sky luminance = median over all pixels *outside* the blob window.
        blob_mask = np.zeros_like(lum, dtype=bool)
        blob_mask[y_lo:y_hi, x_lo:x_hi] = True
        sky_pixels = lum[~blob_mask]
        sky_lum = float(np.median(sky_pixels)) if sky_pixels.size else float(np.median(lum))
    else:
        thr = np.percentile(lum, diffuse_fallback_percentile)
        mask = lum >= max(thr, 1e-6)
        ys, xs = np.where(mask)
        if xs.size == 0:
            return None
        weights = lum[ys, xs].astype(np.float64)
        w_sum = float(weights.sum())
        if w_sum <= 0:
            return None

        phi_pix = (xs.astype(np.float64) + 0.5) / W * 2.0 * np.pi - np.pi
        mean_cos = float(np.sum(np.cos(phi_pix) * weights) / w_sum)
        mean_sin = float(np.sum(np.sin(phi_pix) * weights) / w_sum)
        phi = np.arctan2(mean_sin, mean_cos)

        theta_pix = np.pi / 2.0 - (ys.astype(np.float64) + 0.5) / H * np.pi
        theta = float(np.sum(theta_pix * weights) / w_sum)

        mean_color = np.array([
            float(np.sum(img[ys, xs, c] * weights) / w_sum) for c in range(3)
        ], dtype=np.float32)
        # Diffuse case: use the mean of the top-percentile region as the
        # sun proxy, and the median of everything else as the sky proxy.
        sun_lum_repr = float(lum[ys, xs].mean())
        non_top_mask = lum < max(thr, 1e-6)
        non_top_pixels = lum[non_top_mask]
        sky_lum = float(np.median(non_top_pixels)) if non_top_pixels.size else float(np.median(lum))

    # Clamp the elevation to avoid degenerate shadows but allow low suns.
    max_t = np.deg2rad(max_elevation_deg)
    min_t = np.deg2rad(min_elevation_deg)
    theta = float(np.clip(theta, min_t, max_t))

    cos_t = np.cos(theta)
    sun_dir_world = np.array([cos_t * np.cos(phi), cos_t * np.sin(phi), np.sin(theta)], dtype=np.float32)
    sun_dir_world /= (np.linalg.norm(sun_dir_world) + 1e-8)

    # SAPIEN wants the direction *from* the light *to* the scene.
    light_direction = -sun_dir_world

    # ---- Determine the effective intensity ----
    if auto_intensity:
        solar_ratio = sun_lum_repr / max(sky_lum, 1e-4)
        # log1p compresses huge midday ratios; base_intensity lets the user
        # globally boost / attenuate the result.
        effective = float(base_intensity) * float(np.log1p(max(solar_ratio, 0.0)))
        effective = float(np.clip(effective, 0.0, float(max_intensity)))
    else:
        effective = float(intensity_scale)

    # Normalise so the brightest channel sits at ``effective``.
    peak_c = float(mean_color.max())
    if peak_c > 1e-4:
        mean_color = mean_color / peak_c * effective
    mean_color = np.clip(mean_color, 0.0, color_clip).astype(np.float32)

    return light_direction, mean_color


def estimate_ground_color_from_hdri(
    hdri_path: str,
    horizon_band_deg: float = 15.0,
    downscale_max_side: int = 1024,
    tone_map_exposure: float = 0.4,
    max_channel: float = 0.9,
) -> Optional[np.ndarray]:
    """Sample an (R, G, B) 'ground tint' from the HDRI.

    We average the part of the HDRI that sits just below the horizon -- a
    horizontal band of ``horizon_band_deg`` degrees underneath the
    mathematical horizon. For a grasslands HDRI this lands squarely on the
    grass, giving a plausible ``base_color`` for a shadow-catcher plane so
    that it visually blends with the HDRI's own ground.

    The radiance is tone-mapped (Reinhard-ish: ``c / (c + 1)`` scaled by
    ``tone_map_exposure``) because HDRI pixel values can exceed 1 but a
    PBR ``base_color`` should live in ``[0, 1]``. Finally each channel is
    clipped to ``max_channel`` so the plane stays dark enough for shadows
    to actually be visible on it.

    Returns ``None`` on failure.
    """
    if not os.path.exists(hdri_path):
        return None

    img = _read_hdri(hdri_path)
    if img.size == 0:
        return None

    H, W = img.shape[:2]
    scale = max(H, W) / float(downscale_max_side)
    if scale > 1.0:
        import cv2
        new_w = int(round(W / scale))
        new_h = int(round(H / scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]

    # Band spans rows just below the horizon row (H/2 ~ equator).
    band_px = max(4, int(round(horizon_band_deg * H / 180.0)))
    y0 = H // 2
    y1 = min(H, y0 + band_px)
    band = img[y0:y1]
    if band.size == 0:
        return None

    mean_rgb = np.asarray(band.reshape(-1, 3).mean(axis=0), dtype=np.float32)
    # Simple exposure + Reinhard tone map -> [0, 1].
    exposed = mean_rgb * float(tone_map_exposure)
    mapped = exposed / (1.0 + exposed)
    mapped = np.clip(mapped, 0.0, float(max_channel)).astype(np.float32)
    return mapped
