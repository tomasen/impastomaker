#!/usr/bin/env python3
"""
ImpastoMaker — Painting to Sculptural 3MF Converter

Two-layer sculptural approach with paint-layer stacking:
  - Layer 1 (Sculpture): Luminance-driven bas-relief depth
  - Layer 2 (Brushstrokes): Medial-axis swept rounded stroke ridges
  - Colors stack in Z like real paint layers (background at bottom, accents on top)

Direct Bambu PLA palette quantization in CIE Lab color space for accurate
filament color matching. All computation at target resolution (default 0.2 mm/pixel).
Targets Bambu Lab multi-material printers (H2C with AMS, 8 filaments).
"""

import argparse
import io
import json
import math
import os
import sys
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import medial_axis

# ---------------------------------------------------------------------------
# Bambu Lab PLA Basic filament palette
# ---------------------------------------------------------------------------
BAMBU_PLA_BASIC = [
    ("Jade White",      "#FFFFFF", (255, 255, 255)),
    ("Black",           "#000000", (0, 0, 0)),
    ("Red",             "#C12E1F", (193, 46, 31)),
    ("Orange",          "#FF6A13", (255, 106, 19)),
    ("Pumpkin Orange",  "#FF9016", (255, 144, 22)),
    ("Gold",            "#E4BD68", (228, 189, 104)),
    ("Sunflower Yellow","#FEC600", (254, 198, 0)),
    ("Yellow",          "#F4EE2A", (244, 238, 42)),
    ("Bright Green",    "#BECF00", (190, 207, 0)),
    ("Mistletoe Green", "#3F8E43", (63, 142, 67)),
    ("Bambu Green",     "#00AE42", (0, 174, 66)),
    ("Turquoise",       "#00B1B7", (0, 177, 183)),
    ("Cyan",            "#0086D6", (0, 134, 214)),
    ("Cobalt Blue",     "#0056B8", (0, 86, 184)),
    ("Blue Grey",       "#5B6579", (91, 101, 121)),
    ("Blue",            "#0A2989", (10, 41, 137)),
    ("Purple",          "#5E43B7", (94, 67, 183)),
    ("Indigo Purple",   "#482960", (72, 41, 96)),
    ("Magenta",         "#EC008C", (236, 0, 140)),
    ("Hot Pink",        "#F5547C", (245, 84, 124)),
    ("Pink",            "#F55A74", (245, 90, 116)),
    ("Maroon Red",      "#9D2235", (157, 34, 53)),
    ("Cocoa Brown",     "#6F5034", (111, 80, 52)),
    ("Bronze",          "#847D48", (132, 125, 72)),
    ("Dark Gray",       "#545454", (84, 84, 84)),
    ("Gray",            "#8E9089", (142, 144, 137)),
    ("Silver",          "#A6A9AA", (166, 169, 170)),
    ("Light Gray",      "#D1D3D5", (209, 211, 213)),
]

TEXTURE_PRESETS = {
    "none": 0.0, "light": 0.3, "medium": 0.6, "heavy": 1.0, "extreme": 1.5,
}

CANVAS_COLOR = ("Canvas", "#F0F0F0", (240, 240, 240))

# Pre-compute Bambu palette in Lab and HSV color spaces
_bambu_rgb_arr = np.array([[c[2] for c in BAMBU_PLA_BASIC]], dtype=np.uint8)
_BAMBU_LAB = cv2.cvtColor(_bambu_rgb_arr, cv2.COLOR_RGB2Lab).reshape(-1, 3).astype(np.float64)
_BAMBU_HSV = cv2.cvtColor(_bambu_rgb_arr, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float64)


# ---------------------------------------------------------------------------
# Stage 1: Image Loading
# ---------------------------------------------------------------------------
def load_image(path):
    """Load image as RGB numpy array at full resolution."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def resize_image(image, max_dim):
    """Resize preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(w, h) <= max_dim:
        return image
    scale = max_dim / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return np.array(Image.fromarray(image).resize((new_w, new_h), Image.LANCZOS),
                    dtype=np.uint8)


# Bayer 4x4 ordered dithering matrix (normalized 0-1)
BAYER_4X4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
], dtype=np.float32) / 16.0


# ---------------------------------------------------------------------------
# Stage 2+3: Direct Bambu PLA Palette Quantization (Lab color space)
# ---------------------------------------------------------------------------
def quantize_to_bambu_palette(image, n_colors=8, morph_kernel=5, min_area=100):
    """Assign each pixel to Bambu PLA colors using greedy error minimization.

    Instead of simple nearest-neighbor counting, iteratively selects whichever
    Bambu color most reduces total assignment error.  This naturally produces a
    diverse palette: adding a redundant shade barely helps when a similar color
    already covers those pixels, so distinct colors that serve under-represented
    regions get selected instead.

    Returns (label_map, palette_rgb, bambu_colors).
    """
    H, W, _ = image.shape
    n_bambu = len(BAMBU_PLA_BASIC)
    n_pixels = H * W
    chunk = 50000

    # Convert image to Lab for perceptual distance
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab).astype(np.float64)
    pixels_lab = img_lab.reshape(-1, 3)

    # Convert to HSV for hue-aware matching
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float64)
    pixels_hsv = img_hsv.reshape(-1, 3)

    # Step 1: precompute hue-aware distance from every pixel to every Bambu
    # color.  Lab gives perceptual accuracy; a hue penalty (weighted by
    # saturation of both colors) ensures saturated pixels prefer filaments
    # of matching hue rather than greys or wrong-hue matches.  Two terms:
    #   1) Hue penalty: penalises wrong-hue matches (scaled by min saturation)
    #   2) Desat penalty: penalises achromatic filaments for chromatic pixels
    HUE_WEIGHT = 150.0  # scale for hue angle mismatch
    SAT_WEIGHT = 100.0  # scale for desaturation penalty
    print("    Computing pixel-to-palette distances (Lab + hue/sat penalty)...")
    all_dists = np.zeros((n_pixels, n_bambu), dtype=np.float32)
    for c in range(n_bambu):
        bam_hsv = _BAMBU_HSV[c]
        bam_sat_norm = bam_hsv[1] / 255.0
        for s in range(0, n_pixels, chunk):
            e = min(s + chunk, n_pixels)
            # Lab distance
            lab_d = np.linalg.norm(pixels_lab[s:e] - _BAMBU_LAB[c], axis=1)
            px_sat_norm = pixels_hsv[s:e, 1] / 255.0
            # Hue penalty: circular hue diff * min saturation of both colors
            hue_diff = np.abs(pixels_hsv[s:e, 0] - bam_hsv[0])
            hue_diff = np.minimum(hue_diff, 180.0 - hue_diff) / 90.0  # [0, 1]
            min_sat = np.minimum(px_sat_norm, bam_sat_norm)
            hue_penalty = HUE_WEIGHT * hue_diff * min_sat
            # Desat penalty: chromatic pixel matched to achromatic filament
            desat = np.maximum(0.0, px_sat_norm - bam_sat_norm) * px_sat_norm
            sat_penalty = SAT_WEIGHT * desat
            all_dists[s:e, c] = (lab_d + hue_penalty + sat_penalty).astype(np.float32)

    # Step 2: greedy selection — each step picks the color that most reduces
    # total assignment error (sum of nearest-selected-color distances)
    best_dist = np.full(n_pixels, 1e30, dtype=np.float32)
    selected = []
    for i in range(n_colors):
        remaining = [c for c in range(n_bambu) if c not in selected]
        errors = np.array([
            np.minimum(best_dist, all_dists[:, c]).sum() for c in remaining])
        best_c = remaining[int(np.argmin(errors))]
        selected.append(best_c)
        best_dist = np.minimum(best_dist, all_dists[:, best_c])
        pct = (all_dists[:, best_c] <= best_dist + 0.01).sum() / n_pixels * 100
        print(f"    #{i+1} {BAMBU_PLA_BASIC[best_c][0]:20s} "
              f"{BAMBU_PLA_BASIC[best_c][1]}  ({pct:.1f}% closest)")

    # Step 2b: diversity check — replace redundant close colors
    MIN_PALETTE_DELTA_E = 25.0
    blacklist = set()  # colors that have been dropped (prevent re-selection)
    replaced = True
    while replaced:
        replaced = False
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                d = np.linalg.norm(_BAMBU_LAB[selected[i]] - _BAMBU_LAB[selected[j]])
                if d < MIN_PALETTE_DELTA_E:
                    drop_idx = j  # later pick = less impactful
                    dropped_c = selected[drop_idx]
                    dropped_name = BAMBU_PLA_BASIC[dropped_c][0]
                    blacklist.add(dropped_c)
                    selected.pop(drop_idx)
                    # Re-run greedy for the freed slot (excluding blacklist)
                    best_dist_temp = np.full(n_pixels, 1e30, dtype=np.float32)
                    for s_idx in selected:
                        best_dist_temp = np.minimum(best_dist_temp,
                                                    all_dists[:, s_idx])
                    remaining = [c for c in range(n_bambu)
                                 if c not in selected and c not in blacklist]
                    if not remaining:
                        # No alternatives left, restore the dropped color
                        selected.insert(drop_idx, dropped_c)
                        blacklist.discard(dropped_c)
                        break
                    errors = np.array([
                        np.minimum(best_dist_temp, all_dists[:, c]).sum()
                        for c in remaining])
                    best_c = remaining[int(np.argmin(errors))]
                    selected.append(best_c)
                    print(f"    Replaced {dropped_name} (too close) → "
                          f"{BAMBU_PLA_BASIC[best_c][0]}")
                    replaced = True
                    break
            if replaced:
                break

    del all_dists  # free ~250MB
    top_indices = np.array(selected)

    # Step 3: re-assign pixels to nearest among selected (same hue-aware metric)
    selected_lab = _BAMBU_LAB[top_indices]
    selected_hsv = _BAMBU_HSV[top_indices]
    labels = np.zeros(H * W, dtype=np.int32)
    for s in range(0, H * W, chunk):
        e = min(s + chunk, H * W)
        lab_d = np.linalg.norm(
            pixels_lab[s:e, np.newaxis, :] - selected_lab[np.newaxis, :, :], axis=2)
        # Hue penalty for each selected color
        hue_d = np.abs(pixels_hsv[s:e, 0:1] - selected_hsv[np.newaxis, :, 0])
        hue_d = np.minimum(hue_d, 180.0 - hue_d) / 90.0
        px_sat = pixels_hsv[s:e, 1:2] / 255.0
        bam_sat = selected_hsv[np.newaxis, :, 1] / 255.0
        min_s = np.minimum(px_sat, bam_sat)
        desat = np.maximum(0.0, px_sat - bam_sat) * px_sat
        dists = lab_d + HUE_WEIGHT * hue_d * min_s + SAT_WEIGHT * desat
        labels[s:e] = np.argmin(dists, axis=1)
    labels = labels.reshape(H, W)

    # Step 3b: ordered dithering between close color pairs
    DITHER_THRESHOLD = 45.0
    flat_lab = img_lab.reshape(-1, 3)
    n_sel = len(selected_lab)
    # Compute Lab distances from each pixel to each selected color
    dither_dists = np.zeros((H * W, n_sel), dtype=np.float32)
    for c in range(n_sel):
        for s in range(0, H * W, chunk):
            e = min(s + chunk, H * W)
            dither_dists[s:e, c] = np.linalg.norm(
                flat_lab[s:e] - selected_lab[c], axis=1).astype(np.float32)
    # Find 1st and 2nd nearest for each pixel
    top2 = np.argpartition(dither_dists, 2, axis=1)[:, :2]
    idx = np.arange(H * W)
    d1 = dither_dists[idx, top2[:, 0]]
    d2 = dither_dists[idx, top2[:, 1]]
    # Only dither where both colors are close to the pixel
    dither_mask = (d1 + d2) < DITHER_THRESHOLD
    if dither_mask.any():
        t = d1 / (d1 + d2 + 1e-10)
        bayer = np.tile(BAYER_4X4, (H // 4 + 1, W // 4 + 1))[:H, :W].ravel()
        swap = dither_mask & (bayer < t)
        flat_labels = labels.ravel()
        flat_labels[swap] = top2[swap, 1]
        labels = flat_labels.reshape(H, W)
        n_dithered = swap.sum()
        print(f"    Dithered {n_dithered:,} pixels ({n_dithered/H/W*100:.1f}%)")
    del dither_dists

    # Step 4: morphological smoothing
    kern = np.ones((morph_kernel, morph_kernel), np.uint8)
    smoothed_masks = []
    for c in range(n_colors):
        mask = (labels == c).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
        smoothed_masks.append(mask)

    counts_sm = np.array([m.sum() for m in smoothed_masks])
    priority = np.argsort(-counts_sm)
    new_labels = np.full((H, W), -1, dtype=np.int32)
    for c in priority:
        new_labels[smoothed_masks[c] > 0] = c

    # Fill unclaimed pixels
    unclaimed = new_labels == -1
    if unclaimed.any():
        unc_lab = img_lab[unclaimed].reshape(-1, 3)
        unc_hsv = img_hsv[unclaimed].reshape(-1, 3)
        lab_d = np.linalg.norm(
            unc_lab[:, np.newaxis, :] - selected_lab[np.newaxis, :, :], axis=2)
        hue_d = np.abs(unc_hsv[:, 0:1] - selected_hsv[np.newaxis, :, 0])
        hue_d = np.minimum(hue_d, 180.0 - hue_d) / 90.0
        px_sat = unc_hsv[:, 1:2] / 255.0
        bam_sat = selected_hsv[np.newaxis, :, 1] / 255.0
        min_s = np.minimum(px_sat, bam_sat)
        desat = np.maximum(0.0, px_sat - bam_sat) * px_sat
        dists = lab_d + HUE_WEIGHT * hue_d * min_s + SAT_WEIGHT * desat
        new_labels[unclaimed] = np.argmin(dists, axis=1)
    labels = new_labels

    # Step 5: remove small connected components
    for c in range(n_colors):
        mask_c = (labels == c).astype(np.uint8)
        n_comp, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mask_c)
        for comp_id in range(1, n_comp):
            if stats[comp_id, cv2.CC_STAT_AREA] < min_area:
                comp_pixels = comp_labels == comp_id
                dilated = cv2.dilate(comp_pixels.astype(np.uint8), kern, iterations=1)
                neighbor_region = (dilated > 0) & ~comp_pixels
                if neighbor_region.any():
                    neighbor_labels = labels[neighbor_region]
                    neighbor_labels = neighbor_labels[neighbor_labels != c]
                    if len(neighbor_labels) > 0:
                        replacement = np.bincount(
                            neighbor_labels.astype(np.int32),
                            minlength=n_colors).argmax()
                        labels[comp_pixels] = replacement

    # Step 6: sort by coverage — most frequent = index 0 = background
    counts_final = np.bincount(labels.ravel(), minlength=n_colors)
    order = np.argsort(-counts_final)
    remap = np.zeros(n_colors, dtype=np.int32)
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    labels = remap[labels]

    # Build output in sorted order
    sorted_bambu = [top_indices[i] for i in order]
    palette = np.array([BAMBU_PLA_BASIC[i][2] for i in sorted_bambu], dtype=np.uint8)
    bambu_colors = [BAMBU_PLA_BASIC[i] for i in sorted_bambu]

    return labels, palette, bambu_colors


# ---------------------------------------------------------------------------
# Stage 4: Sculptural Heightmap (Two-Layer + Paint Stacking)
# ---------------------------------------------------------------------------
def compute_sculpture_layer(image, min_h, max_h, strategy="brightness"):
    """Layer 1: Bas-relief depth from image luminance with bilateral smoothing."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma_space = max(image.shape[:2]) * 0.05
    smoothed = cv2.bilateralFilter(gray, d=0, sigmaColor=50,
                                   sigmaSpace=sigma_space)
    smin, smax = float(smoothed.min()), float(smoothed.max())
    if smax - smin > 1e-6:
        norm = (smoothed - smin) / (smax - smin)
    else:
        norm = np.zeros_like(smoothed)

    if strategy == "brightness":
        norm = 1.0 - norm  # darker = taller

    norm = np.power(norm, 0.8)
    return (min_h + norm * (max_h - min_h)).astype(np.float64)


def segment_strokes(image, mask, min_stroke_area=15):
    """Segment a color region into individual brushstrokes via edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    median_val = float(np.median(gray[mask > 0])) if mask.any() else 128
    low_t = max(10, int(median_val * 0.3))
    high_t = max(30, int(median_val * 0.7))
    edges = cv2.Canny(gray, low_t, high_t)

    mask_u8 = mask.astype(np.uint8)
    boundary = mask_u8 - cv2.erode(mask_u8, np.ones((3, 3), np.uint8))
    edges = edges | (boundary > 0).astype(np.uint8) * 255
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    interior = mask_u8.copy()
    interior[edges > 0] = 0

    n_labels, labels = cv2.connectedComponents(interior)
    strokes = []
    for lbl in range(1, n_labels):
        stroke_mask = (labels == lbl).astype(np.uint8)
        if stroke_mask.sum() >= min_stroke_area:
            strokes.append(stroke_mask)
    return strokes


def stroke_to_height(mask):
    """Rounded cross-section height profile via medial axis sweep."""
    bool_mask = mask.astype(bool)
    if bool_mask.sum() < 3:
        return np.zeros_like(mask, dtype=np.float64)

    skel, distance = medial_axis(bool_mask, return_distance=True)
    dist_on_skel = distance * skel

    if dist_on_skel.max() < 0.5:
        dt = distance_transform_edt(bool_mask)
        mx = dt.max()
        if mx < 1e-8:
            return np.zeros_like(mask, dtype=np.float64)
        frac = dt / mx
        h = np.sqrt(np.maximum(1.0 - (1.0 - frac) ** 2, 0.0))
        h[~bool_mask] = 0
        return h

    skel_inv = ~skel
    dist_to_skel, indices = distance_transform_edt(skel_inv, return_indices=True)
    local_width = dist_on_skel[indices[0], indices[1]]

    perp_frac = np.where(local_width > 0,
                         dist_to_skel / (local_width + 1e-8), 1.0)
    perp_frac = np.clip(perp_frac, 0.0, 1.0)

    height = np.sqrt(np.maximum(1.0 - perp_frac ** 2, 0.0))

    max_width = local_width.max()
    if max_width > 0:
        height *= 0.3 + 0.7 * (local_width / max_width)

    height[~bool_mask] = 0.0
    return height


def generate_heightmap(image, label_map, n_colors,
                       min_h, max_h, strategy, texture_mm, steepness,
                       smooth_sigma, layer_step):
    """Build sculptural heightmap with paint-layer stacking.

    Each color gets a base Z offset (layer stacking) so that background colors
    sit lower and accent colors sit higher — like real paint layers.

    Returns (heightmap, base_z_per_color).
    """
    H, W = label_map.shape

    print("    Computing sculptural base from luminance...")
    sculpture = compute_sculpture_layer(image, min_h, max_h, strategy)

    # Steepness contrast curve
    s_min, s_max = sculpture.min(), sculpture.max()
    if s_max - s_min > 1e-6:
        s_norm = (sculpture - s_min) / (s_max - s_min)
        s_norm = np.power(s_norm, steepness / 2.0)
        sculpture = s_min + s_norm * (s_max - s_min)

    # Paint-layer stacking: most frequent color (index 0) = background = lowest Z
    # Rarest color = foreground = highest Z base offset
    base_z = {}
    for c in range(n_colors):
        base_z[c] = c * layer_step

    # Build per-pixel heightmap = sculpture height + base_z offset for each color
    heightmap = np.zeros((H, W), dtype=np.float64)
    for c in range(n_colors):
        mask = label_map == c
        heightmap[mask] = sculpture[mask] + base_z[c]

    # Brushstroke texture layer
    stroke_layer = np.zeros((H, W), dtype=np.float64)
    if texture_mm > 0:
        print("    Segmenting brushstrokes...")
        total_strokes = 0
        for c in range(n_colors):
            mask_c = (label_map == c).astype(np.uint8)
            if mask_c.sum() < 15:
                continue
            strokes = segment_strokes(image, mask_c)
            for sm in strokes:
                sh = stroke_to_height(sm)
                stroke_layer = np.maximum(stroke_layer, sh)
                total_strokes += 1

            # Fallback for uncovered areas
            covered = np.zeros((H, W), dtype=bool)
            for sm in strokes:
                covered |= sm.astype(bool)
            uncovered = (mask_c > 0) & ~covered
            if uncovered.sum() > 10:
                dt = distance_transform_edt(uncovered)
                mx = dt.max()
                if mx > 0:
                    stroke_layer = np.maximum(
                        stroke_layer,
                        np.where(uncovered, np.sqrt(dt / mx) * 0.3, 0))

        print(f"    {total_strokes} brushstrokes profiled")

        if stroke_layer.max() > 0:
            stroke_layer = (stroke_layer / stroke_layer.max()) * texture_mm

    heightmap += stroke_layer

    # Smoothing
    if smooth_sigma <= 0:
        smooth_sigma = max(0.5, min(H, W) * 0.004)
    heightmap = gaussian_filter(heightmap, sigma=smooth_sigma)
    heightmap = np.clip(heightmap, 0.0, None)

    return heightmap, base_z


# ---------------------------------------------------------------------------
# Stage 5: Mesh Generation (optimized for high resolution)
# ---------------------------------------------------------------------------
def generate_mesh_for_region(label_map, heightmap, color_idx, canvas_z, px_mm):
    """Generate watertight triangle mesh for one color region.

    Uses bounding-box optimization to avoid allocating full image-sized arrays
    for small color regions.
    """
    H, W = label_map.shape
    mask = label_map == color_idx

    if not mask.any():
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)

    # Find bounding box of this color region
    active = np.argwhere(mask)
    if len(active) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)

    r_min, c_min = active.min(axis=0)
    r_max, c_max = active.max(axis=0)

    r0 = max(0, r_min)
    c0 = max(0, c_min)
    r1 = min(H, r_max + 1)
    c1 = min(W, c_max + 1)
    bH = r1 - r0
    bW = c1 - c0

    local_mask = mask[r0:r1, c0:c1]
    local_hmap = heightmap[r0:r1, c0:c1]

    VH, VW = bH + 1, bW + 1
    total_verts = VH * VW

    local_active = np.argwhere(local_mask)
    rows, cols = local_active[:, 0], local_active[:, 1]
    h_vals = local_hmap[rows, cols]

    top_z_sum = np.zeros((VH, VW), dtype=np.float64)
    top_z_count = np.zeros((VH, VW), dtype=np.float64)
    for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        np.add.at(top_z_sum, (rows + dr, cols + dc), h_vals)
        np.add.at(top_z_count, (rows + dr, cols + dc), 1.0)

    valid = top_z_count > 0
    top_z = np.full((VH, VW), canvas_z, dtype=np.float64)
    top_z[valid] = (top_z_sum[valid] / top_z_count[valid]) + canvas_z

    x_all = np.tile(np.arange(VW, dtype=np.float64), VH) * px_mm + c0 * px_mm
    y_all = np.repeat(np.arange(VH, dtype=np.float64), VW) * px_mm + r0 * px_mm
    z_top = top_z.ravel()
    z_bot = np.full_like(z_top, canvas_z)

    tl = rows * VW + cols
    tr = rows * VW + (cols + 1)
    bl = (rows + 1) * VW + cols
    br = (rows + 1) * VW + (cols + 1)

    top_tris = np.empty((len(rows) * 2, 3), dtype=np.int64)
    top_tris[0::2] = np.stack([tl, tr, br], axis=1)
    top_tris[1::2] = np.stack([tl, br, bl], axis=1)

    off = total_verts
    bot_tris = np.empty((len(rows) * 2, 3), dtype=np.int64)
    bot_tris[0::2] = np.stack([tl + off, br + off, tr + off], axis=1)
    bot_tris[1::2] = np.stack([tl + off, bl + off, br + off], axis=1)

    wall_parts = []

    def add_walls(bnd_rows, bnd_cols, v1_top, v2_top, flip=False):
        if len(bnd_rows) == 0:
            return
        v1b = v1_top + off
        v2b = v2_top + off
        w = np.empty((len(bnd_rows) * 2, 3), dtype=np.int64)
        if flip:
            w[0::2] = np.stack([v1_top, v2b, v1b], axis=1)
            w[1::2] = np.stack([v1_top, v2_top, v2b], axis=1)
        else:
            w[0::2] = np.stack([v1_top, v1b, v2b], axis=1)
            w[1::2] = np.stack([v1_top, v2b, v2_top], axis=1)
        wall_parts.append(w)

    # Top edge
    tb = local_mask.copy()
    tb[1:, :] &= ~local_mask[:-1, :]
    tb[0, :] = local_mask[0, :] if r0 == 0 else (local_mask[0, :] & ~mask[r0 - 1, c0:c1])
    r, c = np.where(tb)
    add_walls(r, c, r * VW + c, r * VW + (c + 1))

    # Bottom edge
    bb = local_mask.copy()
    bb[:-1, :] &= ~local_mask[1:, :]
    bb[-1, :] = local_mask[-1, :] if r1 == H else (local_mask[-1, :] & ~mask[r1, c0:c1])
    r, c = np.where(bb)
    add_walls(r, c, (r + 1) * VW + c, (r + 1) * VW + (c + 1), flip=True)

    # Left edge
    lb = local_mask.copy()
    lb[:, 1:] &= ~local_mask[:, :-1]
    lb[:, 0] = local_mask[:, 0] if c0 == 0 else (local_mask[:, 0] & ~mask[r0:r1, c0 - 1])
    r, c = np.where(lb)
    add_walls(r, c, r * VW + c, (r + 1) * VW + c, flip=True)

    # Right edge
    rb = local_mask.copy()
    rb[:, :-1] &= ~local_mask[:, 1:]
    rb[:, -1] = local_mask[:, -1] if c1 == W else (local_mask[:, -1] & ~mask[r0:r1, c1])
    r, c = np.where(rb)
    add_walls(r, c, r * VW + (c + 1), (r + 1) * VW + (c + 1))

    all_tris = np.vstack([top_tris, bot_tris] + wall_parts)

    verts = np.empty((total_verts * 2, 3), dtype=np.float64)
    verts[:total_verts, 0] = x_all
    verts[:total_verts, 1] = y_all
    verts[:total_verts, 2] = z_top
    verts[total_verts:, 0] = x_all
    verts[total_verts:, 1] = y_all
    verts[total_verts:, 2] = z_bot

    used_v = np.unique(all_tris.ravel())
    remap = np.full(len(verts), -1, dtype=np.int32)
    remap[used_v] = np.arange(len(used_v), dtype=np.int32)
    cverts = verts[used_v]
    ctris = remap[all_tris].astype(np.int32)

    v0 = cverts[ctris[:, 0]]
    v1 = cverts[ctris[:, 1]]
    v2 = cverts[ctris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    ok = np.sum(cross ** 2, axis=1) > 1e-12
    ctris = ctris[ok]

    return cverts, ctris


def generate_canvas_mesh(w_mm, h_mm, t_mm):
    if t_mm <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    v = np.array([[0,0,0],[w_mm,0,0],[w_mm,h_mm,0],[0,h_mm,0],
                  [0,0,t_mm],[w_mm,0,t_mm],[w_mm,h_mm,t_mm],[0,h_mm,t_mm]],
                 dtype=np.float64)
    t = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                  [2,3,7],[2,7,6],[0,4,7],[0,7,3],[1,2,6],[1,6,5]],
                 dtype=np.int32)
    return v, t


# ---------------------------------------------------------------------------
# Stage 6: 3MF Export (streaming XML with numpy vectorized formatting)
# ---------------------------------------------------------------------------
def _write_vertices_xml(f, vertices, chunk_size=100000):
    """Write vertex XML using numpy savetxt for speed."""
    for start in range(0, len(vertices), chunk_size):
        chunk = vertices[start:start + chunk_size]
        buf = io.BytesIO()
        np.savetxt(buf, chunk, fmt='     <vertex x="%.3f" y="%.3f" z="%.3f"/>')
        f.write(buf.getvalue())


def _write_triangles_xml(f, triangles, chunk_size=100000):
    """Write triangle XML using numpy savetxt for speed."""
    for start in range(0, len(triangles), chunk_size):
        chunk = triangles[start:start + chunk_size]
        buf = io.BytesIO()
        np.savetxt(buf, chunk, fmt='     <triangle v1="%d" v2="%d" v3="%d"/>')
        f.write(buf.getvalue())


def _make_uuid():
    return str(uuid.uuid4())


def write_3mf(output_path, meshes, material_entries, n_painting_colors):
    """Write Bambu Studio native 3MF with per-part filament color assignment.

    Uses the Bambu-native format:
      - Separate model file for meshes (3D/Objects/object_1.model)
      - Assembly object with components in root model
      - model_settings.config with extruder assignment per part
      - project_settings.config (JSON) with filament_colour array

    meshes[0..n_painting_colors-1] are painting regions.
    meshes[n_painting_colors..] are canvas/support (share extruder 1 with color 0).
    """
    print("    Streaming XML to ZIP...")

    # Filter out empty meshes, track which are valid
    valid = [(i, verts, tris) for i, (verts, tris) in enumerate(meshes)
             if len(verts) > 0 and len(tris) > 0]

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED,
                         compresslevel=1) as zf:
        # --- [Content_Types].xml ---
        zf.writestr("[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
            ' <Default Extension="rels" ContentType='
            '"application/vnd.openxmlformats-package.relationships+xml"/>\n'
            ' <Default Extension="model" ContentType='
            '"application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
            '</Types>')

        # --- _rels/.rels ---
        zf.writestr("_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns='
            '"http://schemas.openxmlformats.org/package/2006/relationships">\n'
            ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
            '</Relationships>')

        # --- 3D/_rels/3dmodel.model.rels ---
        zf.writestr("3D/_rels/3dmodel.model.rels",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns='
            '"http://schemas.openxmlformats.org/package/2006/relationships">\n'
            ' <Relationship Target="/3D/Objects/object_1.model" Id="rel-1" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
            '</Relationships>')

        # --- 3D/Objects/object_1.model (mesh objects) ---
        with zf.open("3D/Objects/object_1.model", 'w') as f:
            f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(b'<model unit="millimeter" xml:lang="en-US"'
                    b' xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"'
                    b' xmlns:BambuStudio="http://schemas.bambulab.com/package/2021"'
                    b' xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06"'
                    b' requiredextensions="p">\n')
            f.write(b' <metadata name="BambuStudio:3mfVersion">1</metadata>\n')
            f.write(b' <resources>\n')

            for idx, (orig_i, verts, tris) in enumerate(valid):
                obj_id = idx + 1
                obj_uuid = f"0001{idx:04d}-81cb-4c03-9d28-80fed5dfa1dc"
                f.write(f'  <object id="{obj_id}" p:UUID="{obj_uuid}"'
                        f' type="model">\n'.encode('utf-8'))
                f.write(b'   <mesh>\n    <vertices>\n')
                _write_vertices_xml(f, verts)
                f.write(b'    </vertices>\n    <triangles>\n')
                _write_triangles_xml(f, tris)
                f.write(b'    </triangles>\n   </mesh>\n')
                f.write(b'  </object>\n')

            f.write(b' </resources>\n <build/>\n</model>\n')

        # --- 3D/3dmodel.model (root assembly) ---
        assembly_id = len(valid) + 1
        assembly_uuid = _make_uuid()
        components_xml = ""
        for idx in range(len(valid)):
            obj_id = idx + 1
            comp_uuid = f"0001{idx:04d}-b206-40ff-9872-83e8017abed1"
            components_xml += (
                f'    <component p:path="/3D/Objects/object_1.model"'
                f' objectid="{obj_id}" p:UUID="{comp_uuid}"'
                f' transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n')

        root_model = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<model unit="millimeter" xml:lang="en-US"'
            ' xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"'
            ' xmlns:BambuStudio="http://schemas.bambulab.com/package/2021"'
            ' xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06"'
            ' requiredextensions="p">\n'
            ' <metadata name="Application">BambuStudio-02.05.00.66</metadata>\n'
            ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
            ' <resources>\n'
            f'  <object id="{assembly_id}" p:UUID="{assembly_uuid}" type="model">\n'
            '   <components>\n'
            f'{components_xml}'
            '   </components>\n'
            '  </object>\n'
            ' </resources>\n'
            f' <build p:UUID="{_make_uuid()}">\n'
            f'  <item objectid="{assembly_id}" p:UUID="{_make_uuid()}"'
            f' transform="1 0 0 0 1 0 0 0 1 0 0 0" printable="1"/>\n'
            ' </build>\n'
            '</model>\n')
        zf.writestr("3D/3dmodel.model", root_model)

        # --- Metadata/model_settings.config (extruder assignments) ---
        parts_xml = ""
        for idx, (orig_i, verts, tris) in enumerate(valid):
            obj_id = idx + 1
            if orig_i < n_painting_colors:
                extruder = orig_i + 1  # 1-based extruder number
            else:
                extruder = 1  # canvas shares extruder with color 0
            name = material_entries[orig_i][0]
            parts_xml += (
                f'    <part id="{obj_id}" subtype="normal_part">\n'
                f'      <metadata key="name" value="{name}"/>\n'
                f'      <metadata key="matrix"'
                f' value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
                f'      <metadata key="source_file"'
                f' value="{os.path.basename(output_path)}"/>\n'
                f'      <metadata key="source_object_id" value="0"/>\n'
                f'      <metadata key="source_volume_id" value="{idx}"/>\n'
                f'      <metadata key="extruder" value="{extruder}"/>\n'
                f'    </part>\n')

        model_settings = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<config>\n'
            f'  <object id="{assembly_id}">\n'
            f'    <metadata key="name" value="ImpastoMaker"/>\n'
            f'    <metadata key="extruder" value="1"/>\n'
            f'{parts_xml}'
            '  </object>\n'
            '  <plate>\n'
            '    <metadata key="plater_id" value="1"/>\n'
            '    <metadata key="plater_name" value=""/>\n'
            '    <metadata key="locked" value="false"/>\n'
            '    <model_instance>\n'
            f'      <metadata key="object_id" value="{assembly_id}"/>\n'
            '      <metadata key="instance_id" value="0"/>\n'
            '      <metadata key="identify_id" value="85"/>\n'
            '    </model_instance>\n'
            '  </plate>\n'
            '  <assemble>\n'
            f'   <assemble_item object_id="{assembly_id}" instance_id="0"'
            f' transform="1 0 0 0 1 0 0 0 1 0 0 0" offset="0 0 0" />\n'
            '  </assemble>\n'
            '</config>\n')
        zf.writestr("Metadata/model_settings.config", model_settings)

        # --- Metadata/project_settings.config (JSON with filament colours) ---
        # Only painting colors go into filament slots (canvas shares color 0)
        n_slots = max(8, n_painting_colors)
        filament_colours = []
        filament_types = []
        filament_settings = []
        for i in range(n_slots):
            if i < n_painting_colors:
                filament_colours.append(material_entries[i][1])
                filament_types.append("PLA")
                filament_settings.append("Bambu PLA Basic @BBL H2C")
            else:
                filament_colours.append("#00000000")
                filament_types.append("PLA")
                filament_settings.append("Bambu PLA Basic @BBL H2C")

        project_config = {
            "name": "project_settings",
            "from": "project",
            "version": "02.05.00.66",
            "filament_colour": filament_colours,
            "filament_multi_colour": filament_colours,
            "filament_colour_type": ["0"] * n_slots,
            "filament_type": filament_types,
            "filament_settings_id": filament_settings,
            "filament_ids": ["GFA00"] * n_slots,
            "filament_vendor": ["Bambu Lab"] * n_slots,
            "filament_density": ["1.26"] * n_slots,
            "filament_diameter": ["1.75"] * n_slots,
            "filament_cost": ["24.99"] * n_slots,
            "filament_is_support": ["0"] * n_slots,
            "filament_soluble": ["0"] * n_slots,
            "filament_map": ["1"] * n_slots,
            "filament_map_mode": "Auto For Flush",
            "default_filament_colour": [""] * n_slots,
            "default_filament_profile": ["Bambu PLA Basic @BBL H2C"],
        }
        zf.writestr("Metadata/project_settings.config",
                     json.dumps(project_config, indent=4))

        # --- Metadata/slice_info.config ---
        zf.writestr("Metadata/slice_info.config",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<config>\n  <header>\n'
            '    <header_item key="X-BBL-Client-Type" value="slicer"/>\n'
            '    <header_item key="X-BBL-Client-Version"'
            ' value="02.05.00.66"/>\n'
            '  </header>\n</config>\n')


# ---------------------------------------------------------------------------
# Stage 7: Preview
# ---------------------------------------------------------------------------
def generate_preview(original, label_map, palette, heightmap, bambu_colors,
                     output_path, max_preview=800):
    H, W = label_map.shape
    n_colors = len(palette)

    if max(H, W) > max_preview:
        scale = max_preview / max(H, W)
        pH = max(1, int(H * scale))
        pW = max(1, int(W * scale))
    else:
        pH, pW = H, W

    panel1 = Image.fromarray(original).resize((pW, pH), Image.LANCZOS)
    quantized = palette[label_map]
    panel2 = Image.fromarray(quantized).resize((pW, pH), Image.LANCZOS)

    if heightmap.max() > 0:
        h_norm = heightmap / heightmap.max()
    else:
        h_norm = np.zeros((H, W))
    gy, gx = np.gradient(h_norm)
    shade = np.clip(0.5 + 0.5 * (-gx - gy), 0.15, 1.0)
    shaded = (quantized.astype(np.float64) * shade[:, :, np.newaxis])
    panel3 = Image.fromarray(np.clip(shaded, 0, 255).astype(np.uint8)).resize(
        (pW, pH), Image.LANCZOS)

    gap = 4
    total_w = pW * 3 + gap * 2
    line_h = 20
    legend_h = max(60, n_colors * line_h + 30)

    canvas = Image.new("RGB", (total_w, pH + legend_h), (255, 255, 255))
    canvas.paste(panel1, (0, 0))
    canvas.paste(panel2, (pW + gap, 0))
    canvas.paste(panel3, (pW * 2 + gap * 2, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    y0 = pH + 2
    draw.text((0, y0), "Original", fill=(0, 0, 0), font=font)
    draw.text((pW + gap, y0), "Quantized (Bambu PLA)", fill=(0, 0, 0), font=font)
    draw.text((pW * 2 + gap * 2, y0), "Sculpted Relief", fill=(0, 0, 0), font=font)

    counts = np.bincount(label_map.ravel(), minlength=n_colors)
    total_px = label_map.size
    y = y0 + line_h + 4
    for c in range(n_colors):
        cov = counts[c] / total_px * 100
        draw.rectangle([4, y, 20, y + 14], fill=tuple(palette[c]),
                        outline=(0, 0, 0))
        draw.text((24, y),
                  f"  {c+1}. {bambu_colors[c][0]} ({bambu_colors[c][1]}) | {cov:.1f}%",
                  fill=(0, 0, 0), font=font)
        y += line_h

    canvas.save(output_path)


# ---------------------------------------------------------------------------
# Stage 8: Metadata
# ---------------------------------------------------------------------------
def save_metadata(output_path, args, palette, bambu_colors, label_map,
                  heightmap, tile_files, out_w, out_h, px_mm):
    H, W = label_map.shape
    n_colors = len(palette)
    counts = np.bincount(label_map.ravel(), minlength=n_colors)
    total_px = label_map.size

    filaments = [{"name": CANVAS_COLOR[0], "hex": CANVAS_COLOR[1],
                  "rgb": list(CANVAS_COLOR[2]), "role": "base plate"}]
    for c in range(n_colors):
        rh = heightmap[label_map == c]
        filaments.append({
            "index": c + 1, "name": bambu_colors[c][0],
            "hex": bambu_colors[c][1], "rgb": list(bambu_colors[c][2]),
            "height_range_mm": [round(float(rh.min()), 2),
                                round(float(rh.max()), 2)],
            "coverage_percent": round(counts[c] / total_px * 100, 1),
        })

    meta = {
        "source": os.path.basename(args.input),
        "resolution_px": [W, H],
        "pixel_size_mm": round(px_mm, 4),
        "output_size_mm": f"{out_w:.1f} x {out_h:.1f}",
        "n_colors": n_colors,
        "strategy": args.strategy,
        "filaments": filaments,
        "output_files": [os.path.basename(f) for f in tile_files],
    }
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_texture(value):
    if value in TEXTURE_PRESETS:
        return TEXTURE_PRESETS[value]
    try:
        return max(0.0, float(value))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Must be preset ({', '.join(TEXTURE_PRESETS)}) or mm value")


def main():
    parser = argparse.ArgumentParser(
        description="ImpastoMaker: Sculptural painting to multi-color 3MF",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", help="Input image (PNG, JPG, BMP, TIFF)")
    parser.add_argument("-o", "--output", help="Output 3MF path")
    parser.add_argument("-c", "--colors", type=int, default=8,
                        help="Number of filament colors (default: 8)")
    parser.add_argument("-r", "--resolution", type=int, default=1525,
                        help="Max pixels on longest edge (default: 1525 = 0.2mm/pixel at 305mm)")
    parser.add_argument("--output-width", type=float, default=305.0,
                        help="Output width mm (default: 305 = H2C right nozzle)")
    parser.add_argument("--output-height", type=float, default=None,
                        help="Output height mm (default: auto)")
    parser.add_argument("--min-height", type=float, default=0.5,
                        help="Min sculpture height mm (default: 0.5)")
    parser.add_argument("--max-height", type=float, default=5.0,
                        help="Max sculpture height mm (default: 5.0)")
    parser.add_argument("--layer-step", type=float, default=0.4,
                        help="Z offset between paint layers mm (default: 0.4)")
    parser.add_argument("--canvas", type=float, default=0.6,
                        help="Canvas base thickness mm (default: 0.6)")
    parser.add_argument("--strategy", choices=["brightness", "frequency", "equal"],
                        default="brightness",
                        help="Height mapping (default: brightness)")
    parser.add_argument("--texture", default="heavy",
                        help="Brushstroke amplitude: preset or mm value")
    parser.add_argument("--steepness", type=float, default=2.0,
                        help="Relief contrast (default: 2.0)")
    parser.add_argument("--smooth", type=float, default=0.0,
                        help="Gaussian sigma, 0=auto")
    parser.add_argument("--no-preview", action="store_true")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.colors < 2 or args.colors > len(BAMBU_PLA_BASIC):
        print(f"Error: Colors must be 2-{len(BAMBU_PLA_BASIC)}", file=sys.stderr)
        sys.exit(1)

    texture_mm = parse_texture(args.texture)

    output_base = args.output or str(
        Path(args.input).parent / f"{Path(args.input).stem}_3d.3mf")

    print("ImpastoMaker — Sculptural Painting to 3MF")
    print("=" * 55)
    print(f"Input:       {args.input}")
    print(f"Colors:      {args.colors} Bambu PLA filaments")
    print(f"Resolution:  {args.resolution}px (all computation)")
    print(f"Strategy:    {args.strategy}")
    print(f"Height:      {args.min_height}-{args.max_height} mm + {texture_mm:.1f} mm strokes")
    print(f"Layer step:  {args.layer_step} mm between paint layers")
    print()

    # --- Stage 1: Load and resize to target resolution ---
    print("[1/6] Loading image...")
    full_image = load_image(args.input)
    orig_h, orig_w = full_image.shape[:2]
    print(f"  Original: {orig_w} x {orig_h}")

    image = resize_image(full_image, args.resolution)
    del full_image
    ih, iw = image.shape[:2]
    print(f"  Working:  {iw} x {ih}")

    aspect = orig_h / orig_w
    out_w = args.output_width
    out_h = args.output_height or out_w * aspect
    if out_h > 320.0 and args.output_height is None:
        scale = 320.0 / out_h
        out_h = 320.0
        out_w *= scale
    px_mm = out_w / iw
    print(f"  Output:   {out_w:.1f} x {out_h:.1f} mm ({px_mm:.3f} mm/pixel)")
    print()

    # --- Stage 2+3: Direct Bambu palette quantization (Lab space) ---
    print(f"[2/6] Quantizing to {args.colors} Bambu PLA colors (Lab space)...")
    label_map, palette, bambu_colors = quantize_to_bambu_palette(
        image, args.colors, morph_kernel=5, min_area=100)
    n_colors = len(palette)

    counts = np.bincount(label_map.ravel(), minlength=n_colors)
    total_px = label_map.size
    for i, (name, hx, _) in enumerate(bambu_colors):
        print(f"  {i+1}. {name:20s} {hx}  ({counts[i]/total_px*100:.1f}%)")
    print()

    # --- Stage 4: Heightmap ---
    print("[3/6] Generating sculptural heightmap...")
    heightmap, base_z = generate_heightmap(
        image, label_map, n_colors,
        args.min_height, args.max_height, args.strategy,
        texture_mm, args.steepness, args.smooth, args.layer_step)

    unique_z = len(np.unique(np.round(heightmap[heightmap > 0], decimals=4)))
    print(f"  {unique_z} unique Z values")
    print(f"  Height: {heightmap.min():.2f} - {heightmap.max():.2f} mm")
    for c in range(n_colors):
        rh = heightmap[label_map == c]
        if len(rh) > 0:
            print(f"  Color {c+1} ({bambu_colors[c][0]:15s}): "
                  f"base +{base_z[c]:.1f}mm, "
                  f"range {rh.min():.1f}-{rh.max():.1f}mm")
    print()

    # --- Stage 5: Mesh generation + 3MF export ---
    print("[4/6] Generating meshes...")
    meshes = []
    material_entries = []
    total_verts = 0
    total_tris = 0

    for c in range(n_colors):
        print(f"    Color {c+1}/{n_colors} ({bambu_colors[c][0]})...")
        verts, tris = generate_mesh_for_region(
            label_map, heightmap, c, args.canvas, px_mm)
        meshes.append((verts, tris))
        material_entries.append((bambu_colors[c][0], bambu_colors[c][1]))
        total_verts += len(verts)
        total_tris += len(tris)
        print(f"      {len(verts):,} vertices, {len(tris):,} triangles")

    cv, ct = generate_canvas_mesh(out_w, out_h, args.canvas)
    meshes.append((cv, ct))
    # Canvas shares extruder with color 0 (most common = background)
    material_entries.append((bambu_colors[0][0], bambu_colors[0][1]))
    total_verts += len(cv)
    total_tris += len(ct)

    print(f"  Total: {total_verts:,} vertices, {total_tris:,} triangles")
    print()

    print("[5/6] Exporting 3MF...")
    write_3mf(output_base, meshes, material_entries, n_colors)
    del meshes
    size_mb = os.path.getsize(output_base) / (1024 * 1024)
    print(f"  Written: {os.path.basename(output_base)} ({size_mb:.1f} MB)")
    print()

    # --- Stage 6: Preview + metadata ---
    print("[6/6] Preview and metadata...")
    if not args.no_preview:
        pp = str(Path(output_base).with_suffix("")) + "_preview.png"
        generate_preview(image, label_map, palette,
                         heightmap, bambu_colors, pp)
        print(f"  Preview: {os.path.basename(pp)}")

    mp = str(Path(output_base).with_suffix("")) + "_metadata.json"
    save_metadata(mp, args, palette, bambu_colors, label_map,
                  heightmap, [output_base], out_w, out_h, px_mm)
    print(f"  Metadata: {os.path.basename(mp)}")

    print()
    print("Done!")
    print(f"  Pixel size: {px_mm:.3f} mm ({px_mm*1000:.0f} um)")
    print(f"  Paint layers: {n_colors} colors stacked with {args.layer_step}mm step")
    print("  Open in Bambu Studio -> assign each object to an AMS slot")


if __name__ == "__main__":
    main()
