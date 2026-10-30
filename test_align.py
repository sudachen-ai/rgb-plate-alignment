"""Regression checks for align.py.

These tests were added after the course submission, during portfolio cleanup.
They build a synthetic three-plate scan with a known offset and check that the
search functions recover it.
"""

import numpy as np

from align import (get_best_displacement, get_best_displacement_pyramid,
                   get_color_image, normalized_cross_correlation, split_raw_image)


def make_plate(seed=0, h=120, w=96):
    rng = np.random.default_rng(seed)
    base = rng.random((h, w)).astype(np.float32) * 255
    # smooth it a little so neighbouring offsets score differently but not randomly
    k = np.ones((3, 3), dtype=np.float32) / 9.0
    out = base.copy()
    for _ in range(2):
        padded = np.pad(out, 1, mode="edge")
        out = sum(padded[i:i + h, j:j + w] * k[i, j] for i in range(3) for j in range(3))
    return out


def test_split_returns_equal_thirds_in_r_g_b_order():
    raw = np.vstack([np.full((10, 4), 1.0), np.full((10, 4), 2.0), np.full((10, 4), 3.0)])
    im_r, im_g, im_b = split_raw_image(raw)
    assert im_b.mean() == 1.0 and im_g.mean() == 2.0 and im_r.mean() == 3.0
    assert im_r.shape == im_g.shape == im_b.shape == (10, 4)


def test_ncc_is_one_for_identical_and_lower_for_shifted():
    plate = make_plate()
    assert abs(normalized_cross_correlation(plate, plate) - 1.0) < 1e-6
    assert normalized_cross_correlation(plate, np.roll(plate, 5, axis=0)) < 0.999


def test_exhaustive_search_recovers_known_offset():
    ref = make_plate()
    moved = np.roll(np.roll(ref, -7, axis=0), 4, axis=1)   # moved = ref shifted by (dy=-7, dx=4)
    dx, dy = get_best_displacement(moved, ref)             # rolling moved by (dy, dx) should undo it
    assert (dx, dy) == (-4, 7)


def test_pyramid_search_matches_exhaustive_on_synthetic_plate():
    ref = make_plate(seed=3, h=256, w=192)
    moved = np.roll(np.roll(ref, 10, axis=0), -6, axis=1)
    assert get_best_displacement_pyramid(moved, ref) == get_best_displacement(moved, ref) == (6, -10)


def test_color_image_shape_and_alignment():
    ref = make_plate(seed=5)
    im_g = np.roll(ref, 3, axis=0)
    im_b = np.roll(ref, -2, axis=1)
    im_color, g_disp, b_disp = get_color_image(ref, im_g, im_b)
    assert im_color.shape == ref.shape + (3,)
    assert g_disp == (0, -3) and b_disp == (2, 0)
    assert np.allclose(im_color[:, :, 1], ref) and np.allclose(im_color[:, :, 2], ref)
