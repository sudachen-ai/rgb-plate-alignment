"""Align the three colour plates of a Prokudin-Gorskii glass negative.

Each scanned negative is one tall grayscale image that holds three exposures
stacked top to bottom in B, G, R order. The functions below split the scan,
score two plates with normalized cross-correlation (NCC), search for the
displacement with the best score, and stack the aligned plates into an RGB
image. Only NumPy is used for the maths; OpenCV is used to read, resize and
write images.

Usage:
    python align.py data/00153v.jpg                 # exhaustive +/-15 px search
    python align.py data/emir.tif --pyramid         # 4-level coarse-to-fine search
    python align.py data/00153v.jpg -o out.png      # save the colour image
"""

import argparse
import time

import cv2
import numpy as np


def split_raw_image(raw_im):
    """Split the scan vertically into equal thirds and return (R, G, B).

    The plates are stored B, G, R from top to bottom, so the first third is
    the blue plate and the last third is the red plate.
    """
    h = raw_im.shape[0] // 3
    im_b = raw_im[0:h, :]
    im_g = raw_im[h:2 * h, :]
    im_r = raw_im[2 * h:3 * h, :]
    return im_r, im_g, im_b


def normalized_cross_correlation(ref_im, input_im):
    """Return the NCC of two single-channel images.

    Each image is flattened to a vector, divided by its L2 norm, and the two
    unit vectors are dotted. The result is 1.0 for identical images.
    """
    assert ref_im.ndim == 2 and input_im.ndim == 2
    ref_vec = ref_im.flatten()
    input_vec = input_im.flatten()
    ref_unit = ref_vec / np.linalg.norm(ref_vec)
    input_unit = input_vec / np.linalg.norm(input_vec)
    return float(np.sum(ref_unit * input_unit))


def get_best_displacement(im1, im2, x_disp=(-15, 15), y_disp=(-15, 15)):
    """Exhaustively search the (dx, dy) that best aligns im1 onto im2.

    im1 is rolled by every integer offset in the given ranges (inclusive) and
    scored against im2 with NCC. np.roll is circular, so pixels that leave one
    edge re-enter on the other; for small offsets this is harmless.
    Returns (best_x_disp, best_y_disp).
    """
    best_x_disp, best_y_disp = 0, 0
    best_score = -np.inf
    for dy in range(y_disp[0], y_disp[1] + 1):
        for dx in range(x_disp[0], x_disp[1] + 1):
            moved = np.roll(im1, dy, axis=0)
            moved = np.roll(moved, dx, axis=1)
            ncc = normalized_cross_correlation(im2, moved)
            if ncc > best_score:
                best_score = ncc
                best_x_disp, best_y_disp = dx, dy
    return best_x_disp, best_y_disp


def get_best_displacement_pyramid(im1, im2, x_disp=(-15, 15), y_disp=(-15, 15), num_pyramid=4):
    """Coarse-to-fine search over an image pyramid.

    Both images are halved (num_pyramid - 1) times. The full search window is
    used only on the smallest level. Each finer level doubles the previous
    displacement and refines it within +/-2 px, so the cost stays small even
    for large scans.
    """
    pyramid1 = [im1]
    pyramid2 = [im2]
    for _ in range(num_pyramid - 1):
        pyramid1.append(cv2.resize(pyramid1[-1], None, fx=0.5, fy=0.5))
        pyramid2.append(cv2.resize(pyramid2[-1], None, fx=0.5, fy=0.5))

    best_x_disp, best_y_disp = get_best_displacement(
        pyramid1[-1], pyramid2[-1], x_disp=x_disp, y_disp=y_disp)

    for level in range(num_pyramid - 2, -1, -1):
        dx_guess = best_x_disp * 2
        dy_guess = best_y_disp * 2
        best_x_disp, best_y_disp = get_best_displacement(
            pyramid1[level], pyramid2[level],
            x_disp=(dx_guess - 2, dx_guess + 2),
            y_disp=(dy_guess - 2, dy_guess + 2))
    return best_x_disp, best_y_disp


def get_color_image(im_r, im_g, im_b, disp_search_func=get_best_displacement):
    """Align G and B to R and stack them into an (H, W, 3) RGB image.

    Returns (im_color, best_g_disp, best_b_disp) where each disp is (dx, dy).
    """
    best_g_disp = disp_search_func(im_g, im_r)
    best_b_disp = disp_search_func(im_b, im_r)
    aligned_g = np.roll(np.roll(im_g, best_g_disp[1], axis=0), best_g_disp[0], axis=1)
    aligned_b = np.roll(np.roll(im_b, best_b_disp[1], axis=0), best_b_disp[0], axis=1)
    im_color = np.stack((im_r, aligned_g, aligned_b), axis=2)
    return im_color, best_g_disp, best_b_disp


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("image", help="path to a scanned negative (jpg/tif)")
    parser.add_argument("--pyramid", action="store_true", help="use the 4-level coarse-to-fine search")
    parser.add_argument("-o", "--output", help="where to save the colour image")
    args = parser.parse_args()

    raw_im = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    search = get_best_displacement_pyramid if args.pyramid else get_best_displacement

    tic = time.time()
    im_r, im_g, im_b = split_raw_image(raw_im)
    im_color, g_disp, b_disp = get_color_image(im_r, im_g, im_b, search)
    toc = time.time()

    print(f"{args.image}: raw {raw_im.shape}, {'pyramid' if args.pyramid else 'exhaustive'} search")
    print(f"G channel, x-disp: {g_disp[0]}, y-disp: {g_disp[1]}")
    print(f"B channel, x-disp: {b_disp[0]}, y-disp: {b_disp[1]}")
    print(f"Elapsed time: {toc - tic:.3f}s")

    if args.output:
        bgr = np.clip(im_color, 0, 255).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(args.output, bgr)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
