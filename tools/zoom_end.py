"""Push the camera in further at the end of a hero clip, so every team's clip ends on a tight
nose/cockpit close-up (Flow often stops the dolly-in too early).

    python tools/zoom_end.py "Team - clean v2.mp4" "Team - clean v3.mp4" [cx cy] [zoom] [start s]

From `start` seconds (default 5.5) to the end, the picture is smoothly enlarged up to `zoom` (default
1.4x), drifting from the frame centre towards (cx, cy) (default 960 470: nose and cockpit). Run it
after clean_video.py, which adds the fade to black, so the zoom happens under the fade.
Frames are edited in their own YUV planes, so colours don't shift; output H.264 CRF 18, no audio.
"""
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np


def zoom_end(src, dst, cx=960.0, cy=470.0, zmax=1.4, t0=5.5):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cap = cv2.VideoCapture(src)
    W, H, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    raw = subprocess.run([ff, "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, W * H * 3 // 2)
    n, ys, cs = len(frames), W * H, W * H // 4
    t1 = (n - 1) / fps
    enc = subprocess.Popen([ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}",
                            "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst], stdin=subprocess.PIPE)
    for i in range(n):
        f = np.clip((i / fps - t0) / (t1 - t0), 0, 1)
        e = f * f * (3 - 2 * f)  # eased
        s = 1 + (zmax - 1) * e
        if s == 1:
            enc.stdin.write(frames[i].tobytes())
            continue
        planes = [frames[i][:ys].reshape(H, W), frames[i][ys:ys + cs].reshape(H // 2, W // 2),
                  frames[i][ys + cs:].reshape(H // 2, W // 2)]
        out = []
        for k, p in enumerate(planes):
            sc = 1 if k == 0 else 0.5  # U and V are half size
            ccx = p.shape[1] / 2 + (cx * sc - p.shape[1] / 2) * f
            ccy = p.shape[0] / 2 + (cy * sc - p.shape[0] / 2) * f
            M = np.float32([[s, 0, ccx - s * ccx], [0, s, ccy - s * ccy]])
            out.append(cv2.warpAffine(p, M, (p.shape[1], p.shape[0]), flags=cv2.INTER_LANCZOS4,
                                      borderMode=cv2.BORDER_REPLICATE))
        enc.stdin.write(np.concatenate([o.ravel() for o in out]).tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"wrote {dst} ({n} frames, zoom to {zmax}x from {t0} s towards {cx:.0f},{cy:.0f})")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (2, 4, 5, 6):
        sys.exit(__doc__)
    zoom_end(a[0], a[1], *(float(x) for x in a[2:]))
