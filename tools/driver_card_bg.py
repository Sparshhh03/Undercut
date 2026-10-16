"""Build a driver card's background photo: the driver cut out of their photo and placed on a team
backdrop, framed exactly like the Verstappen card (same head size and position), 720 x 1006 px.

    python tools/driver_card_bg.py FIGURE BACKDROP OUT HEAD_TOP CHIN FACE_X [BACKDROP_BOX]

FIGURE        photo of the driver: flat light backdrop, pale textured backdrop, or any backdrop if the
              driver fills the middle
BACKDROP      team backdrop image, e.g. the Ferrari shield
HEAD_TOP CHIN FACE_X   in FIGURE pixels: top of the hair, bottom of the chin, centre of the face
BACKDROP_BOX  optional "x0,y0,x1,y1" part of the backdrop to use (default: the Ferrari framing below)

The head (hair top to chin) ends up HEAD px tall with its top HEAD_Y px from the top, and the face
FACE_DX px right of the centre: the Verstappen card's proportions. Light backdrops are removed with the
same rules as tools/cut_helmet.py; dark/busy ones with GrabCut from a box round the figure. Edge pixels
are "decontaminated" (the backdrop colour is taken out of them), so no white halo shows on the new
backdrop.
"""
import sys

import cv2
import numpy as np

W, H = 720, 1006
HEAD, HEAD_Y, FACE_DX = 201, 60, 43
FERRARI_BOX = (402, 590, 1118, 1590)


def figure_alpha(im, head=None):
    h, w = im.shape[:2]
    f = im.astype(np.float32)
    edge = np.r_[f[0], f[-1], f[:, 0], f[:, -1]]
    light = edge.mean() > 200
    busy = light and edge.std(0).max() > 6   # pale but textured backdrop (feathers, a faint logo...)
    k = max(5, (min(h, w) // 60) | 1)
    bright, colour = f.mean(2), f.max(2) - f.min(2)
    if busy:
        # seed the driver from what the backdrop isn't: saturated (suit, skin) or dark (hair, shoes)
        pale = (bright > 175) & (colour < 30)
        seed = cv2.morphologyEx(((colour > 55) | (bright < 95)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(seed)
        core = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > h * w * 0.002
                             and 0.15 * w < cen[i][0] < 0.85 * w]).astype(np.uint8)
        ell = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
        near, far = cv2.dilate(core, ell(41)).astype(bool), ~cv2.dilate(core, ell(121)).astype(bool)
        gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        gc[near & ~pale] = cv2.GC_PR_FGD
        gc[far] = cv2.GC_BGD
        gc[cv2.erode(core, ell(9)).astype(bool)] = cv2.GC_FGD
        if head:   # the white race-suit collar sits just under the chin: never let it count as backdrop
            top, chin, fx = head
            hh = chin - top
            collar = np.array([[fx - .22 * hh, chin - .12 * hh], [fx + .22 * hh, chin - .12 * hh],
                               [fx + .40 * hh, chin + .28 * hh], [fx - .24 * hh, chin + .28 * hh]], np.int32)
            cmask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(cmask, [collar], 1)
            gc[cmask.astype(bool)] = cv2.GC_FGD
        cv2.grabCut(im, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 6, cv2.GC_INIT_WITH_MASK)
        cand = pale.astype(np.uint8)
        bgc = np.median(f[pale & near], 0)
    elif light:
        bgc = np.median(edge, 0)
        cand = (np.abs(f - bgc).max(2) < 10).astype(np.uint8)
        n, lab = cv2.connectedComponents(cand)
        border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
        bg = np.isin(lab, list(border))
        gc = np.where(bg, cv2.GC_PR_BGD, cv2.GC_PR_FGD).astype(np.uint8)
        gc[cv2.erode(bg.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)] = cv2.GC_BGD
        gc[cv2.erode((~bg).astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)] = cv2.GC_FGD
        cv2.grabCut(im, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    else:
        bgc = None
        gc = np.zeros((h, w), np.uint8)
        rect = (int(w * .3), int(h * .1), int(w * .45), int(h * .82))
        cv2.grabCut(im, gc, rect, np.zeros((1, 65)), np.zeros((1, 65)), 8, cv2.GC_INIT_WITH_RECT)
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    fg = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fill = np.zeros_like(fg)
    cv2.drawContours(fill, cnts, -1, 1, -1)
    if busy:    # enclosed gaps showing the pale backdrop stay transparent
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(((fill == 1) & (fg == 0) & (cand == 1)).astype(np.uint8))
        fill[np.isin(lab2, [i for i in range(1, n2) if st2[i, cv2.CC_STAT_AREA] >= 300])] = 0
    elif light:   # enclosed gaps (between an arm and the body) that show the backdrop stay transparent:
        # patches of the exact backdrop colour bigger than a speck (white suit parts are a few shades off)
        exact = (np.abs(f - bgc).max(2) < 4).astype(np.uint8)
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(exact)
        big = np.isin(lab2, [i for i in range(1, n2) if st2[i, cv2.CC_STAT_AREA] >= 40]).astype(np.uint8)
        big = cv2.dilate(big, np.ones((5, 5), np.uint8)).astype(bool)   # and their anti-aliased rims
        fill[big | ((fill == 1) & (fg == 0) & (cand == 1))] = 0
    alpha = cv2.GaussianBlur(fill.astype(np.float32), (0, 0), max(0.8, min(h, w) / 1500))
    return alpha, bgc


def build(figure, backdrop, out, head_top, chin, face_x, box=FERRARI_BOX):
    im = cv2.imread(figure)
    alpha, bgc = figure_alpha(im, (head_top, chin, face_x))
    fgc = im.astype(np.float32)
    if bgc is not None:   # take the light backdrop's colour out of the semi-transparent edge pixels
        a = np.clip(alpha, 0.05, 1)[..., None]
        fgc = np.clip((fgc - (1 - a) * bgc) / a, 0, 255)

    s = HEAD / (chin - head_top)
    M = np.float32([[s, 0, W / 2 + FACE_DX - s * face_x], [0, s, HEAD_Y - s * head_top]])
    interp = cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_AREA
    fig = cv2.warpAffine(fgc, M, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT)
    a = np.clip(cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0), 0, 1)[..., None]

    bd = cv2.imread(backdrop)
    x0, y0, x1, y1 = box
    bd = cv2.resize(bd[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    # a soft contact shadow behind the driver so they sit in the scene rather than float on it
    sh = cv2.GaussianBlur(a[..., 0], (0, 0), 14)[..., None]
    bd = bd * (1 - 0.45 * sh)
    comp = np.clip(fig * a + bd * (1 - a), 0, 255).astype(np.uint8)
    cv2.imwrite(out.rsplit(".", 1)[0] + "_full.png", comp)
    ok = cv2.imwrite(out, comp, [cv2.IMWRITE_WEBP_QUALITY, 86])
    print(f"wrote {out} (scale x{s:.2f}{'  - upscaled, will look soft' if s > 1.6 else ''})", ok)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (6, 7):
        sys.exit(__doc__)
    box = tuple(int(v) for v in a[6].split(",")) if len(a) == 7 else FERRARI_BOX
    build(a[0], a[1], a[2], float(a[3]), float(a[4]), float(a[5]), box)
