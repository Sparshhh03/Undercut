"""Smooth out the one-frame "hitches" Veo/Flow clips sometimes have: the whole picture jumps a step too
far in one frame (usually where the generator joins its internal chunks, e.g. at 1.0 s and 6.0 s), so
the camera move stutters.

    python tools/fix_hitches.py "input.mp4" "output.mp4" [frame frame ...]

Without frame numbers, hitches are found automatically: a frame whose change from the previous frame is
much bigger than the changes around it. Because the clip really skips ahead there (it doesn't just have
one bad frame), the frames in a short window round each hitch (K before and after) are re-timed: they're
rebuilt by optical-flow interpolation at evenly spaced points of the motion, so the extra step is spread
over the whole window and the camera move stays steady. The editing happens directly on the video's own YUV planes (like
remove_sparkle.py), so the rest of the clip keeps its exact colours. Output H.264 CRF 18, no audio.
"""
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np

FF = imageio_ffmpeg.get_ffmpeg_exe()
K = 6  # frames re-timed on each side of a hitch


def find_hitches(Y, ratio=1.35):
    small = [cv2.resize(y, (480, 270), interpolation=cv2.INTER_AREA).astype(np.float32) for y in Y]
    d = np.array([0] + [np.abs(small[i] - small[i - 1]).mean() for i in range(1, len(small))])
    out = []
    for i in range(2, len(d) - 2):
        around = np.median(np.r_[d[i - 2:i], d[i + 1:i + 3]])
        if around > 1 and d[i] > ratio * around:
            out.append(i)
    return out


def between(a, b, flow, t, scale):
    """Plane at fraction t of the way from a to b (optical flow a -> b, computed at full luma size)."""
    h, w = a.shape
    f = cv2.resize(flow, (w, h)) * scale
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    wa = cv2.remap(a, gx - t * f[..., 0], gy - t * f[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    wb = cv2.remap(b, gx + (1 - t) * f[..., 0], gy + (1 - t) * f[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return (1 - t) * wa + t * wb


def main(src, dst, frames=None):
    cap = cv2.VideoCapture(src)
    W, H, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    raw = subprocess.run([FF, "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, W * H * 3 // 2).copy()
    q = W * H // 4
    Y = a[:, :W * H].reshape(-1, H, W)
    U = a[:, W * H:W * H + q].reshape(-1, H // 2, W // 2)
    V = a[:, W * H + q:].reshape(-1, H // 2, W // 2)

    frames = frames or find_hitches(Y)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    src_planes = (Y.copy(), U.copy(), V.copy())
    small = [cv2.resize(y, (480, 270), interpolation=cv2.INTER_AREA).astype(np.float32) for y in src_planes[0]]
    for c in frames:
        lo, hi = max(c - K, 0), min(c + K, len(a) - 1)
        # progress of the motion through the window (frame-to-frame change, summed)
        steps = [np.abs(small[j + 1] - small[j]).mean() for j in range(lo, hi)]
        prog = np.r_[0, np.cumsum(steps)]
        flows = {}
        for j in range(lo + 1, hi):
            target = prog[-1] * (j - lo) / (hi - lo)       # evenly spaced progress
            m = int(np.clip(np.searchsorted(prog, target) - 1, 0, hi - lo - 1))
            t = float((target - prog[m]) / max(prog[m + 1] - prog[m], 1e-6))
            if m not in flows:
                flows[m] = dis.calc(src_planes[0][lo + m], src_planes[0][lo + m + 1], None)
            for P, S, s_ in ((Y, src_planes[0], 1), (U, src_planes[1], 0.5), (V, src_planes[2], 0.5)):
                P[j] = np.clip(between(S[lo + m].astype(np.float32), S[lo + m + 1].astype(np.float32),
                                       flows[m], t, s_) + 0.5, 0, 255)
        print(f"  re-timed frames {lo + 1}-{hi - 1} round the hitch at frame {c} ({c / fps:.2f} s)")
    print(f"{len(a)} frames, {len(frames)} hitch(es) fixed")

    enc = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}",
                            "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst], stdin=subprocess.PIPE)
    enc.stdin.write(a.tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]])
