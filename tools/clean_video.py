"""Remove the fixed sparkle watermark that Google Flow / Gemini adds to the bottom-right corner
of generated videos, and re-encode as web-friendly H.264 (no audio).

    python tools/clean_video.py "input.mp4" "output.mp4" ["reference.mp4"]

If the clip doesn't end on fully black frames, pass a reference clip that does (any other Flow
clip: the watermark is identical). Clips that don't fully fade out get their fade finished, so every
clip ends on pure black.

How it works: the watermark is a semi-transparent overlay at a fixed position. For every pixel:
    seen = background * (1 - alpha) + colour * alpha  =  background * slope + offset
1. The clip's black frames show where the watermark is (only it is visible there).
2. For every frame, the background behind the watermark is estimated from the pixels around it.
3. Across all frames the background goes dark and bright, so a straight-line fit of `seen` against
   the estimated background gives each pixel's real slope and offset (alpha and colour).
4. Every frame is un-blended: background = (seen - offset) / slope.
5. The watermark's thin outlines are re-filled from the clean pixels on either side.
This recovers the real picture underneath instead of painting over it.
"""
import sys

import cv2
import imageio_ffmpeg
import numpy as np

REGION = (0.80, 0.70)  # search area: bottom-right corner (fractions of width, height)


def read_frames(path):
    cap = cv2.VideoCapture(path)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f
    cap.release()


def locate(path):
    """Watermark footprint from the black frames of `path`, as a box plus mask within that box."""
    acc, n, shape = None, 0, None
    for f in read_frames(path):
        shape = f.shape
        if f.mean() < 2:
            acc = f.astype(np.float32) if acc is None else acc + f
            n += 1
    if not n:
        return None
    H, W = shape[:2]
    aw = (acc / n).max(axis=2)
    aw[: int(H * REGION[1])] = 0
    aw[:, : int(W * REGION[0])] = 0
    mask = (aw > 2.5).astype(np.uint8) * 255
    # keep only the blobs close to the strongest watermark pixel (ignores stray lights in the scene)
    py, px = np.unravel_index(np.argmax(aw), aw.shape)
    count, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    keep = [i for i in range(1, count) if np.hypot(*(cents[i] - (px, py))) < 120]
    mask = np.isin(labels, keep).astype(np.uint8) * 255
    ys, xs = np.nonzero(mask)
    m = 24
    box = (max(xs.min() - m, 0), max(ys.min() - m, 0), min(xs.max() + m, W), min(ys.max() + m, H))
    x0, y0, x1, y1 = box
    strength = (acc / n)[y0:y1, x0:x1] * (mask[y0:y1, x0:x1, None] > 0)
    return box, mask[y0:y1, x0:x1], strength, n


def fit_overlay(path, box, mask, strength):
    """Per-pixel slope/offset of seen vs. surrounding-based background estimate."""
    x0, y0, x1, y1 = box
    hole = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    S = {k: 0.0 for k in ("n", "x", "y", "xx", "xy")}
    for f in read_frames(path):
        crop = f[y0:y1, x0:x1]
        est = cv2.inpaint(crop, hole, 7, cv2.INPAINT_TELEA).astype(np.float64)
        seen = crop.astype(np.float64)
        S["n"] += 1; S["x"] += est; S["y"] += seen; S["xx"] += est * est; S["xy"] += est * seen
    n = S["n"]
    var = n * S["xx"] - S["x"] ** 2
    slope = np.where(var > 1e-6, (n * S["xy"] - S["x"] * S["y"]) / np.maximum(var, 1e-6), 1.0)
    slope = np.clip(slope, 0.15, 1.0)
    offset = (S["y"] - slope * S["x"]) / n
    # where the background behind a pixel hardly changed, the fit is unreliable: fall back to the
    # white-overlay model measured from the black frames (alpha = strength / 255)
    sd = np.sqrt(np.maximum(S["xx"] / n - (S["x"] / n) ** 2, 0))
    trust = np.clip((sd - 10) / 25, 0, 1)
    white_slope = 1 - np.clip(strength / 255.0, 0, 0.85)
    slope = trust * slope + (1 - trust) * white_slope
    offset = trust * offset + (1 - trust) * strength
    inside = (cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0)[..., None]
    slope = np.where(inside, slope, 1.0)
    offset = np.where(inside, offset, 0.0)
    # the watermark is neutral grey, so red, green and blue share one alpha: averaging the channels
    # stops small per-channel fitting errors from showing up as coloured fringes
    slope = np.repeat(slope.mean(axis=2, keepdims=True), 3, axis=2)
    offset = np.repeat(offset.mean(axis=2, keepdims=True), 3, axis=2)
    # smooth the fitted maps slightly so compression noise doesn't create speckle
    slope = cv2.GaussianBlur(slope.astype(np.float32), (0, 0), 0.8)
    offset = cv2.GaussianBlur(offset.astype(np.float32), (0, 0), 0.8)
    return slope, offset


def clean(src_path, dst_path, ref_path=None):
    found = locate(src_path)
    if found is None:
        if not ref_path:
            raise SystemExit("This clip has no fully black frames. Pass a reference clip that does, e.g. "
                             'python tools/clean_video.py in.mp4 out.mp4 "Red BUll Test 2.mp4"')
        found = locate(ref_path)
        if found is None:
            raise SystemExit("The reference clip has no black frames either.")
        print(f"no black frames in this clip; measured the watermark from {ref_path}")
    box, mask, strength, nblack = found
    print(f"watermark box {box}, located from {nblack} black frames")
    slope, offset = fit_overlay(src_path, box, mask, strength)
    print(f"fitted transparency: background kept {slope.min():.2f}-{slope.max():.2f}")
    x0, y0, x1, y1 = box
    edge = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    soft = cv2.GaussianBlur(edge, (0, 0), 1.5).astype(np.float32)[..., None] / 255.0
    # the watermark's anti-aliased outlines (including where two sparkles overlap) are the hardest
    # pixels to un-blend; mark those thin lines and re-fill them from the clean pixels either side
    strength = cv2.GaussianBlur((offset.max(axis=2) if offset.ndim == 3 else offset), (0, 0), 0.7)
    lines = cv2.morphologyEx(strength, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 4
    lines = cv2.dilate(lines.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))

    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ok, last = cap.read()
    cap.release()
    # finish the fade if the clip doesn't reach black: darken the last half-second down to 0
    fade_frames = int(fps * 0.5) if ok and last.mean() > 1 else 0
    if fade_frames:
        print(f"clip ends at brightness {last.mean():.1f}; finishing the fade to black over {fade_frames} frames")
    writer = imageio_ffmpeg.write_frames(
        dst_path, (W, H), fps=fps, codec="libx264", quality=None, macro_block_size=1,
        output_params=["-crf", "18", "-preset", "slow", "-movflags", "+faststart", "-an"],
        pix_fmt_out="yuv420p")
    writer.send(None)
    for i, f in enumerate(read_frames(src_path)):
        c = f[y0:y1, x0:x1].astype(np.float32)
        restored = np.clip((c - offset) / slope, 0, 255)
        smooth = cv2.medianBlur(restored.astype(np.uint8), 3).astype(np.float32)
        restored = (restored * (1 - soft) + smooth * soft).astype(np.uint8)
        f[y0:y1, x0:x1] = cv2.inpaint(restored, lines, 3, cv2.INPAINT_TELEA)
        left = total - 1 - i
        if fade_frames and left < fade_frames:
            f = (f.astype(np.float32) * (left / fade_frames)).astype(np.uint8)
        writer.send(np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    writer.close()


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    clean(*sys.argv[1:])
