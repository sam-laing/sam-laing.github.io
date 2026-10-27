"""Turn line art into Fourier epicycle data for the page background.

    python trace.py IMAGE NAME LABEL [--thresh 140] [--blank x0,y0,x1,y1] [--close 0] [--size 700] [--edges 120]

Line art -> ink mask -> 1px skeleton -> one walk per stroke -> strokes chained into a single
closed path with pen-up jumps -> DFT. Writes data/NAME.json and data/NAME.png (a preview of the
reconstruction), then rebuilds ../characters.js from everything in data/.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
N_SAMPLES = 4096


# ---------- image -> strokes ----------

def ink_mask(path, size=700, thresh=140, blank=(), hollow=6, dot=30, edges=0):
    """Dark pixels = ink, or with `edges` (a Canny threshold) the boundaries between colour regions.
    `blank` is a list of (x0, y0, x1, y1) fractions to wipe (watermarks, backgrounds)."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"could not read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:                                        # transparent PNG: put it on white
        a = img[:, :, 3:4] / 255.0
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    s = size / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    if edges:                                                    # coloured renders: boundaries between flat colour regions
        smooth = cv2.bilateralFilter(img, 9, 60, 9)
        e = np.max([cv2.Canny(smooth[:, :, c], edges // 2, edges) for c in range(3)], axis=0)
        mask = (e > 0).astype(np.uint8)
        hollow = 0                                               # edges are already thin
    else:
        mask = (img.max(axis=2) < thresh).astype(np.uint8)      # near-black in every channel (ignores e.g. a red tie)
    h, w = mask.shape
    for x0, y0, x1, y1 in blank:
        mask[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0
    if hollow:
        mask = _hollow(mask, hollow)
    return _clean(mask, dot)


def _hollow(mask, hollow):
    """Solid blobs thicker than `hollow` px (shoes, pupils) are reduced to their outlines."""
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * hollow + 1, 2 * hollow + 1))
    core = cv2.erode(mask, k)                                    # only survives inside thick blobs
    return mask & (1 - cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * hollow - 3, 2 * hollow - 3))))


def _clean(mask, dot):
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 25                 # drop specks
    out = keep[lab].astype(np.uint8)
    for i in range(1, n):                                        # small solid dots (pupils, buttons) -> tiny rings,
        x, y, bw, bh, area = stats[i]                            # otherwise they skeletonise to a single pixel
        if keep[i] and max(bw, bh) <= dot and area > 0.45 * bw * bh:
            blob = (lab == i).astype(np.uint8)
            cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            out[blob.astype(bool)] = 0
            (cx, cy), r = cv2.minEnclosingCircle(cs[0])
            cv2.circle(out, (int(round(cx)), int(round(cy))), max(4, int(round(r))), 1, 1)
    return out.astype(bool)


def strokes(mask, min_len, close=0):
    """Each skeleton component as a closed walk (lines are traversed there and back)."""
    import cv2
    from skimage.morphology import skeletonize
    if close:                                                    # bridge small gaps in sketchy pencil lines
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))).astype(bool)
    sk = skeletonize(mask).astype(np.uint8) * 255
    # every contour, holes included: lines inside a closed loop (a face inside a head) are only on the hole side
    contours, _ = cv2.findContours(sk, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        if len(c) < min_len:
            continue
        p = c[:, 0, :].astype(float)
        k = 5                                                    # smooth the pixel staircase (closed walk)
        p = np.stack([np.convolve(np.r_[p[-k:, i], p[:, i], p[:k, i]], np.ones(2 * k + 1) / (2 * k + 1), "valid")
                      for i in range(2)], 1)
        out.append(p)
    return out


def chain(walks):
    """Greedy nearest-neighbour ordering. Returns points and a pen-down flag per point."""
    walks = sorted(walks, key=len, reverse=True)
    path, pen = [walks[0]], [np.ones(len(walks[0]), bool)]
    rest = walks[1:]
    end = walks[0][-1]
    while rest:
        best = min(((np.hypot(*(w - end).T).min(), i) for i, w in enumerate(rest)))
        w = rest.pop(best[1])
        j = int(np.hypot(*(w - end).T).argmin())
        w = np.r_[w[j:], w[:j]]
        jump = np.linspace(end, w[0], 12)[1:-1]
        path += [jump, w]
        pen += [np.zeros(len(jump), bool), np.ones(len(w), bool)]
        end = w[-1]
    jump = np.linspace(end, path[0][0], 12)[1:-1]                # close the loop
    path.append(jump)
    pen.append(np.zeros(len(jump), bool))
    return np.vstack(path), np.concatenate(pen)


# ---------- path -> Fourier ----------

def resample(pts, pen, n=N_SAMPLES):
    seg = np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T)
    d = np.r_[0, np.cumsum(seg)]
    t = np.linspace(0, d[-1], n, endpoint=False)
    closed = np.vstack([pts, pts[:1]])
    xy = np.c_[np.interp(t, d, closed[:, 0]), np.interp(t, d, closed[:, 1])]
    idx = np.clip(np.searchsorted(d, t, side="right") - 1, 0, len(pen) - 1)
    return xy, pen[idx]


def fourier(xy, pen, terms):
    z = xy[:, 0] - 1j * xy[:, 1]                                 # y up
    lo, hi = xy.min(0), xy.max(0)
    z -= complex((lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2)
    z /= np.abs(np.r_[z.real, z.imag]).max()
    t = np.arange(len(z)) / len(z)
    ks = sorted(range(-terms, terms + 1), key=lambda k: (abs(k), -k))
    cs = [np.mean(z * np.exp(-2j * np.pi * k * t)) for k in ks]
    ups, i = [], 0                                               # pen-up runs as [t0, t1]
    while i < len(pen):
        if not pen[i]:
            j = i
            while j < len(pen) and not pen[j]:
                j += 1
            ups.append([round(i / len(pen), 5), round(j / len(pen), 5)])
            i = j
        else:
            i += 1
    return ks, cs, ups


def preview(ks, cs, ups, out):
    import cv2
    ts = np.linspace(0, 1, 6000, endpoint=False)
    z = (np.array(cs)[None] * np.exp(2j * np.pi * np.array(ks)[None] * ts[:, None])).sum(1)
    img = np.full((600, 600), 255, np.uint8)
    P = np.c_[300 + 280 * z.real, 300 - 280 * z.imag].astype(np.int32)
    up = np.zeros(len(ts), bool)
    for a, b in ups:
        up |= (ts >= a) & (ts < b)
    for i in range(len(ts) - 1):
        if not up[i]:
            cv2.line(img, tuple(P[i]), tuple(P[i + 1]), 0, 2, cv2.LINE_AA)
    cv2.imwrite(str(out), img)


def save(name, label, ks, cs, ups):
    DATA.mkdir(exist_ok=True)
    rec = {"name": name, "label": label, "pen_up": ups,
           "coeffs": [[int(k), round(c.real, 5), round(c.imag, 5)] for k, c in zip(ks, cs)]}
    (DATA / f"{name}.json").write_text(json.dumps(rec, separators=(",", ":")))
    preview(ks, cs, ups, DATA / f"{name}.png")
    order = json.loads((HERE / "order.json").read_text()) if (HERE / "order.json").exists() else []
    recs = {p.stem: json.loads(p.read_text()) for p in DATA.glob("*.json")}
    names = [n for n in order if n in recs] + sorted(n for n in recs if n not in order)
    js = "/* generated by trace/trace.py — Fourier coefficients for the background drawings */\n"
    js += "window.FOURIER_CHARACTERS = " + json.dumps([recs[n] for n in names], separators=(",", ":")) + ";\n"
    (HERE.parent / "characters.js").write_text(js)
    print(f"{name}: {len(ks)} terms, {len(ups)} pen lifts -> data/{name}.png, characters.js ({len(js) // 1024} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("name")
    ap.add_argument("label")
    ap.add_argument("--terms", type=int, default=250)
    ap.add_argument("--min-len", type=int, default=16)
    ap.add_argument("--thresh", type=int, default=140, help="ink = every channel below this")
    ap.add_argument("--blank", action="append", default=[], help="x0,y0,x1,y1 fractions to wipe (watermarks)")
    ap.add_argument("--close", type=int, default=0, help="closing kernel (px) to bridge gaps in pencil lines")
    ap.add_argument("--hollow", type=int, default=6, help="solid blobs thicker than this become outlines")
    ap.add_argument("--edges", type=int, default=0, help="Canny threshold: trace colour boundaries instead of dark ink")
    ap.add_argument("--size", type=int, default=700, help="working resolution (longest side, px)")
    a = ap.parse_args()
    blank = [tuple(map(float, b.split(","))) for b in a.blank]
    walks = strokes(ink_mask(a.image, size=a.size, thresh=a.thresh, blank=blank, hollow=a.hollow, dot=a.size // 25, edges=a.edges), a.min_len, a.close)
    xy, pen = resample(*chain(walks))
    save(a.name, a.label, *fourier(xy, pen, a.terms))


if __name__ == "__main__":
    main()
