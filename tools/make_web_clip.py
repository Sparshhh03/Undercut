"""Turn a cleaned hero clip into the files the website plays.

    python tools/make_web_clip.py "cleaned.mp4" 05-racing_bulls

Writes:
    media_masters/hero/<name>.mp4         copy of the cleaned master (not uploaded; gitignored)
    media_masters/hero/<name>-60fps.mp4   motion-interpolated to 60 fps (smooth on 60 Hz screens)
    site/media/hero/<name>-1080.mp4       desktop version (H.264, CRF 21, no audio, faststart)
    site/media/hero/<name>-720.mp4        phone / data-saver version (CRF 23)
    site/media/hero/<name>-poster.jpg     still shown while the video loads / for reduced motion

Then add the clip to HERO_CLIPS in site/js/app.js, e.g.
    { base: 'media/hero/05-racing_bulls', team: 'rb' },
`team` must be a key of the TEAMS map in app.js (red_bull, mercedes, ferrari, mclaren, rb, alpine,
williams, aston_martin, haas, audi, cadillac).

The 60 fps interpolation takes about 3 minutes per 8-second clip.
"""
import os
import shutil
import subprocess
import sys

import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF = imageio_ffmpeg.get_ffmpeg_exe()


def run(*args):
    subprocess.run([FF, "-v", "error", "-y", *args], check=True)


def make(src, name):
    masters = os.path.join(ROOT, "media_masters", "hero")
    web = os.path.join(ROOT, "site", "media", "hero")
    os.makedirs(masters, exist_ok=True)
    os.makedirs(web, exist_ok=True)

    master = os.path.join(masters, f"{name}.mp4")
    if os.path.abspath(src) != os.path.abspath(master):
        shutil.copyfile(src, master)
    smooth = os.path.join(masters, f"{name}-60fps.mp4")

    print("interpolating to 60 fps (about 3 minutes)...")
    run("-i", master, "-vf", "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
        "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-an", smooth)
    common = ["-c:v", "libx264", "-preset", "slow", "-tune", "film", "-profile:v", "high",
              "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"]
    print("encoding 1080p and 720p...")
    run("-i", smooth, "-crf", "21", *common, os.path.join(web, f"{name}-1080.mp4"))
    run("-i", smooth, "-vf", "scale=-2:720", "-crf", "23", *common, os.path.join(web, f"{name}-720.mp4"))
    run("-i", master, "-vf", "scale=1280:-2", "-frames:v", "1", "-q:v", "4", os.path.join(web, f"{name}-poster.jpg"))
    for suffix in ("-1080.mp4", "-720.mp4", "-poster.jpg"):
        p = os.path.join(web, name + suffix)
        print(f"  {os.path.relpath(p, ROOT)}  {os.path.getsize(p) / 1e6:.1f} MB")
    print(f"done. Now add {{ base: 'media/hero/{name}', team: '...' }} to HERO_CLIPS in site/js/app.js")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    make(sys.argv[1], sys.argv[2])
