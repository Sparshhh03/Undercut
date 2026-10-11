"""Remove the moving Gemini sparkle (the white four-pointed star) that Google Flow sometimes adds to a
clip, drifting across the dark studio for a second or so. (The fixed watermark in the bottom-right
corner is a different thing: tools/clean_video.py removes that.)

    python tools/remove_sparkle.py "input.mp4" "output.mp4"

How it works:
1. The dark studio above the horizon (top 40% of the picture) is searched for star-shaped bright spots
   (square-ish, mostly empty box, 12-75 px, bright core, dark all round, and dark there a second
   earlier or later), so lights, reflections and the car don't count.
2. Detections in neighbouring frames that move smoothly are grouped into "passes", and each pass is
   widened by a few frames (following its motion) to catch the star fading in and out.
3. For every frame of a pass, the area round the star is rebuilt from the clean frames just before and
   just after the pass (blended by time, so slow camera movement and light changes carry through),
   with a soft edge. The editing happens directly on the video's own YUV planes, so the untouched
   parts of the picture keep their exact colours (an RGB round trip darkens a clip slightly).
Output: H.264, CRF 18, same frame rate, no audio (like clean_video.py).
"""
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np

FF = imageio_ffmpeg.get_ffmpeg_exe()
TOP = 0.4   # search the top 40% of the picture (the dark studio above the horizon)
TAIL = 4    # frames added before/after each pass for the star's fade in/out


def read_all(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return np.array(frames), fps


def find_stars(grey):
    """[(frame, cx, cy, size)] for every star-like spot."""
    n = len(grey)
    out = []
    for i in range(n):
        m = (grey[i] > 35).astype(np.uint8)
        m[int(grey.shape[1] * TOP):] = 0
        cnt, lab, st, cen = cv2.connectedComponentsWithStats(m)
        for k in range(1, cnt):
            x, y, w, h, a = st[k]
            if not (12 <= w <= 75 and 12 <= h <= 75 and 0.6 < w / h < 1.6 and a < 0.6 * w * h):
                continue
            # a star floats on dark wall: bright core, and dark all round it (not part of the car)
            if grey[i][lab == k].max() < 100:
                continue
            y0, y1, x0, x1 = max(y - 15, 0), min(y + h + 15, grey.shape[1]), max(x - 15, 0), min(x + w + 15, grey.shape[2])
            ring = grey[i][y0:y1, x0:x1].astype(np.float32)
            inner = np.zeros(ring.shape, bool)
            inner[y - y0:y - y0 + h, x - x0:x - x0 + w] = True
            if ring[~inner].mean() > 25 or np.percentile(ring[~inner], 95) > 45:
                continue
            # and the same spot is dark ~1 s before or after (a moving star, not a lamp)
            box = (slice(y, y + h), slice(x, x + w))
            others = [grey[j][box].mean() for j in (i - 24, i + 24) if 0 <= j < n]
            if others and min(others) < 20:
                out.append((i, float(cen[k][0]), float(cen[k][1]), int(max(w, h))))
    return out


def group_passes(stars):
    """Group detections frame by frame into smooth passes: [[(frame, cx, cy, size), ...], ...]."""
    passes = []
    for s in sorted(stars):
        for p in passes:
            f, x, y, _ = p[-1]
            if 0 < s[0] - f <= 2 and np.hypot(s[1] - x, s[2] - y) < 30:
                p.append(s)
                break
        else:
            passes.append([s])
    passes = [p for p in passes if len(p) >= 5]
    # widen each pass along its own motion, for the dim frames as the star fades in and out
    wide = []
    for p in passes:
        (f0, x0, y0, s0), (f1, x1, y1, s1) = p[0], p[-1]
        vx, vy = (x1 - x0) / max(f1 - f0, 1), (y1 - y0) / max(f1 - f0, 1)
        head = [(f0 - k, x0 - vx * k, y0 - vy * k, s0) for k in range(TAIL, 0, -1)]
        tail = [(f1 + k, x1 + vx * k, y1 + vy * k, s1) for k in range(1, TAIL + 1)]
        wide.append(head + p + tail)
    return wide


def read_yuv(path, W, H):
    """Every frame as its raw YUV 4:2:0 planes (Y full size, U and V half size), no colour conversion."""
    raw = subprocess.run([FF, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, W * H * 3 // 2)
    q = W * H // 4
    return (a[:, :W * H].reshape(-1, H, W).astype(np.float32),
            a[:, W * H:W * H + q].reshape(-1, H // 2, W // 2).astype(np.float32),
            a[:, W * H + q:].reshape(-1, H // 2, W // 2).astype(np.float32))


def main(src, dst):
    frames, fps = read_all(src)
    n, H, W = frames.shape[:3]
    grey = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    passes = group_passes(find_stars(grey))
    print(f"{n} frames, {len(passes)} sparkle pass(es)")
    del frames, grey

    planes = read_yuv(src, W, H)
    grids = [np.mgrid[0:H, 0:W], np.mgrid[0:H // 2, 0:W // 2]]
    for p in passes:
        p = [s for s in p if 0 <= s[0] < n]
        first, last = p[0][0], p[-1][0]
        before, after = max(first - 2, 0), min(last + 2, n - 1)
        print(f"  frames {first}-{last} ({first / fps:.2f}-{last / fps:.2f} s), "
              f"rebuilt from frames {before} and {after}")
        for f, cx, cy, size in p:
            t = (f - before) / max(after - before, 1)
            r = size / 2 + 14  # star plus its soft glow
            for k, plane in enumerate(planes):
                sc = 1 if k == 0 else 0.5  # U and V are half size
                yy, xx = grids[0 if k == 0 else 1]
                mask = np.clip(((r + 6) * sc - np.hypot(xx - cx * sc, yy - cy * sc)) / (12 * sc), 0, 1)
                clean = (1 - t) * plane[before] + t * plane[after]
                plane[f] = plane[f] * (1 - mask) + clean * mask

    enc = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                            "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                            "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", "-an", dst], stdin=subprocess.PIPE)
    for i in range(n):
        for plane in planes:
            enc.stdin.write(np.clip(plane[i] + 0.5, 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
