"""Prepare a Gemini-made start frame for Google Flow: remove Gemini's sparkle watermark and upscale
to exactly 1920x1080.

    python tools/clean_frame.py "gemini_image.webp" "site/media/frames/<team>.png"

The sparkle sits at the same relative spot in every Gemini image (about x=935, y=482 in a 1024x572
image, scaled for other sizes). It's covered by blending in floor texture from a nearby patch: the
candidate patch whose brightness and texture best match the pixels around the sparkle wins.
Then a Lanczos resize to 1080 px high, a centre crop to 1920 wide, and gentle sharpening.

Prints a before/after crop path so the result can be checked visually.
"""
import os
import sys

import cv2
import numpy as np


def clean_frame(src, dst):
    im = cv2.imread(src)
    if im is None:
        raise SystemExit(f"Can't read {src}")
    im = im.astype(np.float32)
    H, W = im.shape[:2]
    cx, cy, r = round(935 * W / 1024), round(482 * H / 572), round(21 * W / 1024)

    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], np.int32)], 255)
    k = max(7, r // 3) | 1
    mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    soft = cv2.GaussianBlur(mask, (0, 0), max(4, r / 5)).astype(np.float32)[..., None] / 255.0
    ring = (cv2.dilate(mask, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0) & (mask == 0)
    m = mask > 0

    step = round(2.2 * r)
    best = None
    for dx, dy in [(0, step), (0, -step), (-step, 0), (step, 0), (-step, step), (step, step)]:
        patch = np.roll(np.roll(im, -dy, axis=0), -dx, axis=1)
        score = abs(patch[m].mean() - im[ring].mean()) + abs(patch[m].std() - im[ring].std())
        if best is None or score < best[0]:
            best = (score, patch)
    clean = np.clip(im * (1 - soft) + best[1] * soft, 0, 255).astype(np.uint8)

    h = 1080
    w = round(W * h / H)
    up = cv2.resize(clean, (w, h), interpolation=cv2.INTER_LANCZOS4)
    if w >= 1920:
        x0 = (w - 1920) // 2
        up = up[:, x0:x0 + 1920]
    else:  # narrower than 16:9: scale to width instead and crop height
        up = cv2.resize(clean, (1920, round(H * 1920 / W)), interpolation=cv2.INTER_LANCZOS4)
        y0 = (up.shape[0] - 1080) // 2
        up = up[y0:y0 + 1080]
    blur = cv2.GaussianBlur(up, (0, 0), 1.2)
    up = cv2.addWeighted(up, 1.35, blur, -0.35, 0)
    cv2.imwrite(dst, up)

    y0, y1, x0, x1 = max(cy - 3 * r, 0), cy + 3 * r, max(cx - 4 * r, 0), cx + 4 * r
    check = np.hstack([im[y0:y1, x0:x1].astype(np.uint8), clean[y0:y1, x0:x1]])
    check_path = os.path.splitext(dst)[0] + "_check.png"
    cv2.imwrite(check_path, cv2.resize(check, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    print(f"source {W}x{H} -> {dst} (1920x1080). Before/after watermark crop: {check_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    clean_frame(sys.argv[1], sys.argv[2])
