"""Cut a helmet out of a product photo (plain light backdrop) for the driver cards.

    python tools/cut_helmet.py "photo.webp" site/media/helmets/<driver>.webp [x0,y0,x1,y1]

Already transparent photos (PNG/WebP with alpha) are used as they are (only the helmet is kept, not a
separate floor shadow). For a busy backdrop, pass the box round the helmet shell (without visor tabs): the
helmet-shaped outline fitted to it seeds GrabCut, and the cut never leaves that outline.

The backdrop is found from the border inwards: on a flat backdrop, pixels within a few shades of its
colour; on a gradient backdrop, light-to-mid grey, colourless, smooth pixels (this also catches the
shadow under the helmet). Small notches in the outline are closed. OpenCV's GrabCut then
refines the outline, holes are filled and the edge gets a soft 1 px anti-alias. Nothing on the helmet
itself is redrawn, so sponsors stay exactly as photographed.

Every output has the same layout so all cards match: the helmet's width fills the image width
(WIDTH px, plus a small margin), and the height follows the helmet's own proportions. The card CSS
gives every photo the same width, so all helmets show at the same size.
Also writes <out>_check.png (the cut-out on the dark card colour) for a visual check.
"""
import sys

import cv2
import numpy as np
from PIL import Image

WIDTH = 520   # helmet width in the output, px
MARGIN = 8    # transparent margin round the helmet, px


def cut(src, dst, box=None):
    im = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if im is None:
        sys.exit(f"Can't read {src}")
    if im.ndim == 3 and im.shape[2] == 4 and (im[..., 3] < 250).mean() > 0.05:
        a = (im[..., 3] > 200).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(a)
        keep = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))
        return save(im[..., :3], im[..., 3].astype(np.float32) / 255 * keep, dst)
    im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    h, w = im.shape[:2]
    if box:
        # a front-view helmet is a rounded dome on top and squarer at the chin: a superellipse fitted to
        # the box. Outside it (a little grown) is backdrop, well inside it is helmet, GrabCut decides the rim.
        x0, y0, x1, y1 = box
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u, v = np.abs(xx - cx) / rx, (yy - cy) / ry
        shape = u ** 2.2 + np.abs(v) ** np.where(v < 0, 2.2, 4.0)   # <= 1 inside the helmet
        gc = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
        gc[shape > 1.0] = cv2.GC_PR_BGD
        gc[shape > 1.12] = cv2.GC_BGD
        gc[shape < 0.75] = cv2.GC_FGD
        cv2.grabCut(im, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 8, cv2.GC_INIT_WITH_MASK)
        fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
        fg = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        fill = np.zeros_like(fg)
        cv2.drawContours(fill, cnts, -1, 1, -1)
        # never outside the grown shape, never a bite out of its inner part (dark logos near a dark wall)
        fill = ((fill == 1) & (shape <= 1.08)) | (shape <= 0.97)
        return save(im, cv2.GaussianBlur(fill.astype(np.float32), (0, 0), 0.8), dst)
    f = im.astype(np.float32)
    bright = f.mean(2)
    colour = f.max(2) - f.min(2)
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))
    grad = cv2.GaussianBlur(grad, (0, 0), 1.5)

    # backdrop candidates, then keep the parts that touch the border.
    # A perfectly flat backdrop: only pixels within a few shades of its colour (white helmet paint is
    # usually a little off it, so it survives). A gradient backdrop with a shadow: light-to-mid grey,
    # colourless and smooth pixels (the helmet's dark bottom rim stops it).
    edge = np.r_[f[0], f[-1], f[:, 0], f[:, -1]]
    if edge.std(0).max() < 3:
        cand = (np.abs(f - np.median(edge, 0)).max(2) < 8).astype(np.uint8)
    else:
        cand = ((bright > 130) & (colour < 14) & (grad < 40)).astype(np.uint8)
    n, lab = cv2.connectedComponents(cand)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    bg = np.isin(lab, list(border))

    # GrabCut: far from the object = surely backdrop, deep inside = surely helmet, the rest is refined
    k = max(9, (min(h, w) // 40) | 1)
    gc = np.where(bg, cv2.GC_PR_BGD, cv2.GC_PR_FGD).astype(np.uint8)
    gc[cv2.erode(bg.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)] = cv2.GC_BGD
    gc[cv2.erode((~bg).astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)] = cv2.GC_FGD
    cv2.grabCut(im, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)

    # the helmet = the biggest piece, with its holes filled
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    fg = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    # close small notches; padded first, because near the photo's edge closing would fill the backdrop
    fg = cv2.copyMakeBorder(fg, 2 * k, 2 * k, 2 * k, 2 * k, cv2.BORDER_CONSTANT, value=0)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1)))
    fg = fg[2 * k:-2 * k, 2 * k:-2 * k]
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fill = np.zeros_like(fg)
    cv2.drawContours(fill, cnts, -1, 1, -1)
    alpha = cv2.GaussianBlur(fill.astype(np.float32), (0, 0), 0.8)

    save(im, alpha, dst)


def save(im, alpha, dst):
    ys, xs = np.where(alpha > 0.5)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    rgba = np.dstack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (alpha * 255).astype(np.uint8)])[y0:y1, x0:x1]
    scale = WIDTH / (x1 - x0)
    out = Image.fromarray(rgba).resize((WIDTH, round((y1 - y0) * scale)), Image.LANCZOS)
    canvas = Image.new("RGBA", (out.width + 2 * MARGIN, out.height + 2 * MARGIN))
    canvas.paste(out, (MARGIN, MARGIN))
    canvas.save(dst, "WEBP", quality=90, method=6)

    check = Image.new("RGBA", canvas.size, (12, 15, 21, 255))
    check.alpha_composite(canvas)
    check.convert("RGB").save(dst.rsplit(".", 1)[0] + "_check.png")
    print(f"{dst}: {canvas.width}x{canvas.height} (helmet {x1 - x0}x{y1 - y0} px in the photo, "
          f"height/width {(y1 - y0) / (x1 - x0):.2f})")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    cut(sys.argv[1], sys.argv[2], tuple(int(v) for v in sys.argv[3].split(",")) if len(sys.argv) == 4 else None)
