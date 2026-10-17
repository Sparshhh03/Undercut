"""Build a driver card's background photo: the driver cut out of their photo and placed on a team
backdrop, framed exactly like the Verstappen card (same head size and position), 720 x 1006 px.

    python tools/driver_card_bg.py FIGURE BACKDROP OUT HEAD_TOP CHIN FACE_X [BACKDROP_BOX] [CHOKE] [SHADOW] [MATCH_REF] [--key | --solid]

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


def figure_alpha(im, head=None, key=False):
    if key == 'solid':
        return solid_alpha(im, head)
    h, w = im.shape[:2]
    f = im.astype(np.float32)
    edge = np.r_[f[0], f[-1], f[:, 0], f[:, -1]]
    light = edge.mean() > 200
    busy = light and edge.std(0).max() > 6   # pale but textured backdrop (feathers, a faint logo...)
    k = max(5, (min(h, w) // 60) | 1)
    bright, colour = f.mean(2), f.max(2) - f.min(2)
    if key:
        return key_alpha(im, f, bright, head)
    if busy:
        # seed the driver from what the backdrop isn't: saturated (suit, skin) or dark (hair, shoes)
        pale = (bright > 175) & (colour < 30)
        seed = cv2.morphologyEx((((colour > 55) & (bright < 185)) | (bright < 95)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
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
        cand = (pale | ((bright > 165) & (colour < 80))).astype(np.uint8)   # pale tints count for enclosed gaps
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
        # gaps are big; white printed letters on the suit are small and must stay (Russell's PETRONAS)
        big = np.isin(lab2, [i for i in range(1, n2) if st2[i, cv2.CC_STAT_AREA] >= h * w * 4e-5]).astype(np.uint8)
        big = cv2.dilate(big, np.ones((5, 5), np.uint8)).astype(bool)   # and their anti-aliased rims
        fill[big | ((fill == 1) & (fg == 0) & (cand == 1))] = 0
    alpha = cv2.GaussianBlur(fill.astype(np.float32), (0, 0), max(0.8, min(h, w) / 1500))
    if bgc is not None:
        alpha = refine_edges(f, fill, alpha, bgc, max(6, min(h, w) // 200))
    return alpha, bgc


def refine_edges(f, fill, alpha, bgc, band):
    """Soft photos fade from hair or skin into the backdrop over several pixels. In a band just inside
    the outline, estimate how much of each pixel is driver and how much is backdrop (projecting its
    colour between the backdrop colour and the nearby solid driver colour), so the fade becomes real
    transparency instead of a light outline."""
    inner = cv2.erode(fill, np.ones((2 * band + 1, 2 * band + 1), np.uint8)).astype(np.float32)
    ring = (cv2.dilate(fill, np.ones((9, 9), np.uint8)) == 1) & (inner == 0)   # incl. the blurred rim just outside
    sig = band * 1.5
    den = cv2.GaussianBlur(inner, (0, 0), sig)[..., None]
    local = cv2.GaussianBlur(f * inner[..., None], (0, 0), sig) / np.maximum(den, 1e-3)   # nearby solid colour
    d = local - bgc
    a = ((f - bgc) * d).sum(2) / np.maximum((d * d).sum(2), 1.0)
    # tighten the fade: some photos were already cut out once with a soft, slightly grey feather, which
    # would show as a light rim on a dark backdrop; squaring the low end pulls that rim in
    # clearly coloured pixels (skin, suit colours) are solid, even if lighter than the nearby driver colour:
    # a brightly lit cheek next to dark hair would otherwise turn see-through. Grey mixes of dark hair and
    # a white backdrop have no colour, so their soft fade is kept.
    chroma = f.max(2) - f.min(2)
    a = np.maximum(a, np.clip((chroma - 18) / 22, 0, 1) * (np.abs(f - bgc).max(2) > 30))
    a = np.clip(a * 1.1, 0, 1) ** 1.8
    ok = ring & (den[..., 0] > 0.02) & ((d * d).sum(2) > 900)   # only where driver and backdrop differ clearly
    out = alpha.copy()
    out[ok] = a[ok]
    out[ring & (np.abs(f - bgc).max(2) < 4)] = 0   # pure backdrop is never part of the driver
    return out


def key_alpha(im, f, bright, head=None):
    """--key: a pale TINTED backdrop (teal watercolour) behind a white suit. Neither brightness nor
    saturation separates them, but the hue does: the backdrop is green-leaning (G > R) or plain white,
    the suit's white is blue-leaning (B > G), skin is red-leaning (R > G), hair and stripes are darker.
    Those rules seed GrabCut; anything the rules leave unsure, GrabCut decides from the colours."""
    h, w = f.shape[:2]
    B, G, R = f[..., 0], f[..., 1], f[..., 2]
    back = (bright > 150) & (B - G < 10) & ((G - R > 10) | (f.min(2) > 236))
    driver = ((B - G > 14) & (bright > 60)) | (bright < 130) | (R - G > 15)
    ell = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    drv = cv2.morphologyEx(driver.astype(np.uint8), cv2.MORPH_OPEN, ell(5))
    n, lab, st, _ = cv2.connectedComponentsWithStats(drv)
    drv = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)   # the figure, not specks
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[cv2.dilate(drv, ell(15)).astype(bool) & ~back] = cv2.GC_PR_FGD
    gc[cv2.erode(drv, ell(7)).astype(bool)] = cv2.GC_FGD
    gc[cv2.erode(back.astype(np.uint8), ell(9)).astype(bool) & ~cv2.dilate(drv, ell(31)).astype(bool)] = cv2.GC_BGD
    cv2.grabCut(im, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 6, cv2.GC_INIT_WITH_MASK)
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    fg = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, ell(5))
    cnts, hier = cv2.findContours(fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    fill = np.zeros_like(fg)
    for i, c in enumerate(cnts):   # fill small holes (logo letters), keep big ones (gaps by the arms)
        if hier[0][i][3] < 0:
            cv2.drawContours(fill, cnts, i, 1, -1)
    tint = (G - R > 10) & (bright > 150)   # the backdrop's teal; white logos on the suit don't have it
    for i, c in enumerate(cnts):
        if hier[0][i][3] >= 0 and cv2.contourArea(c) > h * w * 4e-4:
            hole = np.zeros((h, w), np.uint8)
            cv2.drawContours(hole, cnts, i, 1, -1)
            if tint[hole.astype(bool)].mean() > 0.3:
                cv2.drawContours(fill, cnts, i, 0, -1)
    if head:   # flecks of backdrop caught in curly hair: only in the top half of the head, never on the suit
        hair = np.zeros((h, w), bool)
        hair[:int(head[0] + 0.5 * (head[1] - head[0]))] = True
        fill[back & ~(bright < 130) & hair] = 0
    fill = cv2.morphologyEx(fill, cv2.MORPH_OPEN, ell(3))
    alpha = cv2.GaussianBlur(fill.astype(np.float32), (0, 0), 1.0)
    return alpha, None


def solid_alpha(im, head):
    """--solid: a saturated single-hue backdrop (Audi orange-red, Alpine blue), shaded and with big
    darker numbers on it. Backdrop = the border's hue, strongly saturated; driver = low saturation (black,
    grey, white suit, hair) or a clearly different hue (skin, pink suit). GrabCut settles the rest, and the
    face and neck (from the head measurements) are always kept."""
    h, w = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, Sat, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    edge_h = np.r_[H[0], H[:, 0], H[:, -1]]
    hb = float(np.median(edge_h))
    dh = np.minimum(np.abs(H - hb), 180 - np.abs(H - hb))
    back = (dh < 9) & (Sat > 150) & (V > 70)
    driver = (Sat < 80) | (dh > 25) | (V < 45)
    ell = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    gc = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
    gc[back] = cv2.GC_PR_BGD
    gc[cv2.erode(back.astype(np.uint8), ell(7)).astype(bool)] = cv2.GC_BGD
    gc[cv2.erode((driver & ~back).astype(np.uint8), ell(5)).astype(bool)] = cv2.GC_FGD
    top, chin, fx = head
    hh = chin - top
    face = np.zeros((h, w), np.uint8)
    cv2.ellipse(face, (int(fx), int((top + chin) / 2 + .05 * hh)), (int(.3 * hh), int(.42 * hh)), 0, 0, 360, 1, -1)
    gc[face.astype(bool)] = cv2.GC_FGD
    cv2.grabCut(im, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 6, cv2.GC_INIT_WITH_MASK)
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    fg = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, ell(3))
    cnts, hier = cv2.findContours(fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    fill = np.zeros_like(fg)
    for i, c in enumerate(cnts):
        if hier[0][i][3] < 0:
            cv2.drawContours(fill, cnts, i, 1, -1)
    # holes are kept filled: suit panels can share the backdrop's colour (Audi's red mesh panel);
    # but backdrop-coloured specks in the hair (top half of the head) are removed
    hair = np.zeros((h, w), bool)
    hair[:int(top + .5 * hh)] = True
    if 30 < hb < 160:   # only when the backdrop hue is far from skin tones (blue yes, orange-red no)
        fill[hair & (dh < 14) & (Sat > 70)] = 0
    fill = cv2.morphologyEx(fill, cv2.MORPH_OPEN, ell(3))
    return cv2.GaussianBlur(fill.astype(np.float32), (0, 0), 0.8), None


def match_colour(fgc, mask, ref_path, amount=0.85):
    """Shift the driver's colours (Lab mean and spread) towards the driver in a reference card photo.
    The reference driver is everything in the middle of that photo that isn't bright backdrop colour."""
    ref = cv2.imread(ref_path).astype(np.float32)
    rh, rw = ref.shape[:2]
    sat = ref.max(2) - ref.min(2)
    rmask = np.zeros((rh, rw), bool)
    rmask[int(rh * .05):int(rh * .75), int(rw * .25):int(rw * .75)] = True
    rmask &= ~((ref[..., 2] > 140) & (sat > 90))          # not the yellow / red backdrop
    to_lab = lambda x: cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    src, dst = to_lab(fgc), to_lab(ref)
    out = src.copy()
    for c in range(3):
        sm, ss = src[..., c][mask].mean(), src[..., c][mask].std() + 1e-3
        dm, ds = dst[..., c][rmask].mean(), dst[..., c][rmask].std() + 1e-3
        out[..., c] = (src[..., c] - sm) * (ds / ss) + dm
    out = src + amount * (out - src)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def build(figure, backdrop, out, head_top, chin, face_x, box=FERRARI_BOX, choke=0, shadow=0.45, match=None, key=False):
    im = cv2.imread(figure, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3 and im.shape[2] == 4 and (im[..., 3] < 250).mean() > 0.05:   # already cut out
        alpha, bgc, im = im[..., 3].astype(np.float32) / 255, None, np.ascontiguousarray(im[..., :3])
    else:
        im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        alpha, bgc = figure_alpha(im, (head_top, chin, face_x), key)
    rows = np.where(alpha[-3:].max(1) > 0.5)[0]
    if len(rows):   # cut off by the photo's bottom edge (waist-up shots): fade out instead of a hard line
        fh = int(alpha.shape[0] * 0.22)
        ramp = np.ones(alpha.shape[0], np.float32)
        ramp[-fh:] = np.linspace(1, 0, fh) ** 1.5
        alpha = alpha * ramp[:, None]
    if choke:   # soft photos: pull the edge in by `choke` px so the blur into the old backdrop doesn't show
        alpha = cv2.erode(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * choke + 1, 2 * choke + 1)))
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)
    fgc = im.astype(np.float32)
    if bgc is not None:   # take the light backdrop's colour out of the semi-transparent edge pixels
        a = np.clip(alpha, 0.05, 1)[..., None]
        fgc = np.clip((fgc - (1 - a) * bgc) / a, 0, 255)

    if match is not None:   # grade the driver like a reference driver photo (same light and colour)
        fgc = match_colour(fgc, alpha > 0.9, match)
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
    bd = bd * (1 - shadow * sh)   # 0 on bright backdrops, where it reads as a dark outline
    comp = np.clip(fig * a + bd * (1 - a), 0, 255).astype(np.uint8)
    cv2.imwrite(out.rsplit(".", 1)[0] + "_full.png", comp)
    ok = cv2.imwrite(out, comp, [cv2.IMWRITE_WEBP_QUALITY, 86])
    print(f"wrote {out} (scale x{s:.2f}{'  - upscaled, will look soft' if s > 1.6 else ''})", ok)


if __name__ == "__main__":
    key = "solid" if "--solid" in sys.argv else "--key" in sys.argv
    a = [v for v in sys.argv[1:] if v not in ("--key", "--solid")]
    if len(a) not in (6, 7, 8, 9, 10):
        sys.exit(__doc__)
    box = tuple(int(v) for v in a[6].split(",")) if len(a) >= 7 else FERRARI_BOX
    build(a[0], a[1], a[2], float(a[3]), float(a[4]), float(a[5]), box, int(a[7]) if len(a) >= 8 else 0,
          float(a[8]) if len(a) >= 9 else 0.45, a[9] if len(a) == 10 else None, key)
