"""Single-camera broadcast / replay footage (e.g. a domestic match feed).

What is different from our own synced high-speed cameras:
  * ONE camera, usually behind the bowler, often a long zoom (unknown focal);
  * the clip is edited: wide shot -> slow-motion replay -> wide shot (cuts);
  * the "slow motion" is made by holding and blending frames, so the gap in
    real time between frames is irregular and unknown -- frame counts give no
    usable timing;
  * graphics may be burned in (here: a red in-line corridor);
  * the ball is small (~15-25 px) and shows mostly as a DARK blob on the pitch.

This module finds the replay shot, flags held frames, builds a background,
calibrates the camera from the pitch markings, detects ball candidates and
links them into one delivery. The 3D reconstruction is in monofit.py.
"""
import json
import cv2
import numpy as np

from .camera import Camera
from .pitch import BALL_RADIUS, PITCH_LENGTH, CREASE_OFFSET, RETURN_CREASE_HALF


# ---------------------------------------------------------------- reading
def read_frames(path, frames=None, gray=False, scale=1.0):
    """Yield (index, frame) for the given indices (sorted) or all frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    want = None if frames is None else set(int(f) for f in frames)
    last = None if want is None else max(want)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok or (last is not None and i > last):
            break
        if want is None or i in want:
            if scale != 1.0:
                f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if gray:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            yield i, f
        i += 1
    cap.release()


def video_info(path):
    cap = cv2.VideoCapture(path)
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS) or 30.0, n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                size=(int(cap.get(3)), int(cap.get(4))))
    cap.release()
    return info


# ---------------------------------------------------------------- shots and holds
def scan(path, cut_thresh=18.0, hold_thresh=0.6):
    """Per-frame change measure -> shots (between hard cuts) and held frames.

    diff[i] = mean |frame i - frame i-1| on a downscaled grey image.
    A cut is a big jump; a hold is (nearly) no change at all."""
    diffs, prev = [], None
    for i, g in read_frames(path, gray=True, scale=0.25):
        g = g.astype(np.int16)
        diffs.append(0.0 if prev is None else float(np.mean(np.abs(g - prev))))
        prev = g
    diffs = np.array(diffs)
    if not len(diffs):
        raise RuntimeError(f"could not decode any frames from {path}")
    cuts = [i for i in range(1, len(diffs)) if diffs[i] > cut_thresh]
    bounds = [0] + cuts + [len(diffs)]
    shots = [(a, b - 1) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 5]
    held = diffs < hold_thresh
    held[0] = False
    for c in cuts:
        held[c] = False
    return dict(diffs=diffs, cuts=cuts, shots=shots, held=held)


def camera_motion(path, shot, step=6):
    """90th-percentile image shift (px) across a shot, measured against the
    shot's first frame -- ~0 for a locked-off replay camera, large for
    coverage that pans or zooms at any point."""
    a, b = shot
    idx = list(range(a, b + 1, step))
    frames = dict(read_frames(path, idx, gray=True, scale=0.5))
    ref = frames[idx[0]].astype(np.float32)
    shifts = []
    for k in idx[1:]:
        (dx, dy), _ = cv2.phaseCorrelate(ref, frames[k].astype(np.float32))
        shifts.append(np.hypot(dx, dy) * 2)
    return float(np.percentile(shifts, 90)) if shifts else 0.0


def pitch_fraction(img):
    """Share of the image that looks like a cricket pitch (light tan), as
    opposed to outfield grass. The zoomed delivery replay from behind the
    bowler is mostly pitch; wide shots are mostly grass."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s_, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    tan = (h >= 8) & (h <= 28) & (s_ >= 35) & (s_ <= 150) & (v >= 110)
    return float(tan.mean())


def pick_replay_shot(path, sc, allow_motion=False):
    """Default shot to analyse: a locked-off shot (camera not moving) with the
    most pitch in view. Returns (shot, info). The click tool lets the operator
    choose another shot."""
    rows = []
    for s in sc["shots"]:
        a, b = s
        mid = dict(read_frames(path, [(a + b) // 2]))[(a + b) // 2]
        rows.append(dict(shot=s, frames=b - a + 1, motion_px=round(camera_motion(path, s), 2),
                         pitch_fraction=round(pitch_fraction(mid), 3),
                         held_fraction=round(float(np.mean(sc["held"][a + 1:b + 1])) if b > a else 0.0, 3)))
    if not rows:
        raise RuntimeError("no continuous replay shot found")
    pool = [r for r in rows if r["frames"] >= 20 and (allow_motion or r["motion_px"] < 3.0)] or rows
    best = max(pool, key=lambda r: r["pitch_fraction"])
    return best["shot"], dict(best, all_shots=rows)


def background(path, shot, n=60):
    a, b = shot
    idx = np.linspace(a, b, min(n, b - a + 1)).astype(int)
    stack = np.stack([f for _, f in read_frames(path, idx)])
    return np.median(stack, axis=0).astype(np.uint8)


# ---------------------------------------------------------------- calibration
# World frame (cbt.pitch): x positive = IMAGE LEFT when the camera is behind the
# bowler ("off side for a right-hander"), y from the striker's stumps (0) to the
# bowler's stumps (20.12), z up.
CREASE_POINTS = {
    "bowler_bowling_crease_L": (RETURN_CREASE_HALF, PITCH_LENGTH, 0.0),
    "bowler_bowling_crease_R": (-RETURN_CREASE_HALF, PITCH_LENGTH, 0.0),
    "bowler_popping_crease_L": (RETURN_CREASE_HALF, PITCH_LENGTH - CREASE_OFFSET, 0.0),
    "bowler_popping_crease_R": (-RETURN_CREASE_HALF, PITCH_LENGTH - CREASE_OFFSET, 0.0),
    "striker_popping_crease_L": (RETURN_CREASE_HALF, CREASE_OFFSET, 0.0),
    "striker_popping_crease_R": (-RETURN_CREASE_HALF, CREASE_OFFSET, 0.0),
    "striker_bowling_crease_L": (RETURN_CREASE_HALF, 0.0, 0.0),
    "striker_bowling_crease_R": (-RETURN_CREASE_HALF, 0.0, 0.0),
}
CREASE_HELP = {   # what to click, in the order the click tool asks for them
    "bowler_popping_crease_L": "NEAR end (bowler's): popping crease meets the LEFT return crease",
    "bowler_popping_crease_R": "NEAR end: popping crease meets the RIGHT return crease",
    "bowler_bowling_crease_L": "NEAR end: bowling crease (line of the stumps) meets the LEFT return crease",
    "bowler_bowling_crease_R": "NEAR end: bowling crease meets the RIGHT return crease",
    "striker_popping_crease_L": "FAR end (batter's): popping crease meets the LEFT return crease",
    "striker_popping_crease_R": "FAR end: popping crease meets the RIGHT return crease",
    "striker_bowling_crease_L": "FAR end: bowling crease meets the LEFT return crease",
    "striker_bowling_crease_R": "FAR end: bowling crease meets the RIGHT return crease",
}


def _ridge_profile(I, pts_along, normal, half):
    """Sub-pixel centre of a bright line crossing each sample point along
    `normal`: centroid of brightness above the local background."""
    out = []
    h, w = I.shape
    for p in pts_along:
        ts = np.arange(-half, half + 1)
        q = p[None, :] + ts[:, None] * normal[None, :]
        xs, ys = q[:, 0], q[:, 1]
        if xs.min() < 1 or ys.min() < 1 or xs.max() > w - 2 or ys.max() > h - 2:
            continue
        v = cv2.remap(I, xs.astype(np.float32)[None, :], ys.astype(np.float32)[None, :],
                      cv2.INTER_LINEAR)[0]
        base = np.percentile(v, 30)
        wgt = np.clip(v - base - 8, 0, None)
        if wgt.sum() < 20 or wgt.max() < 12:
            continue
        # keep the strongest connected run only
        k = int(np.argmax(wgt))
        lo, hi = k, k
        while lo > 0 and wgt[lo - 1] > 0:
            lo -= 1
        while hi < len(wgt) - 1 and wgt[hi + 1] > 0:
            hi += 1
        ww = wgt[lo:hi + 1]
        tc = float((ts[lo:hi + 1] * ww).sum() / ww.sum())
        out.append(p + tc * normal)
    return np.array(out).reshape(-1, 2)


def _fit_line(P):
    P = np.asarray(P, float)
    for _ in range(4):
        if len(P) < 4:
            break
        m = P.mean(0)
        _, _, vt = np.linalg.svd(P - m)
        r = np.abs((P - m) @ vt[1])
        keep = r < max(0.8, 3 * np.median(r))
        if keep.all():
            break
        P = P[keep]
    m = P.mean(0)
    _, _, vt = np.linalg.svd(P - m)
    rms = float(np.sqrt(np.mean(((P - m) @ vt[1]) ** 2))) if len(P) > 2 else np.inf
    return m, vt[0], len(P), rms


def refine_intersection(img, click, d_crease=(1.0, 0.0), d_return=(0.0, 1.0), reach=(70, 45), half=9):
    """Snap a rough click to the exact crossing of the crease line and the
    return crease, by fitting both painted lines near the click."""
    I = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
    c = np.asarray(click, float)
    lines = []
    for d, L in ((np.asarray(d_crease, float), reach[0]), (np.asarray(d_return, float), reach[1])):
        d = d / np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        # sample along the line on both sides of the crossing, skipping the crossing itself
        ts = np.r_[np.arange(-L, -12, 2), np.arange(12, L, 2)]
        pts = _ridge_profile(I, c[None, :] + ts[:, None] * d[None, :], n, half)
        if len(pts) < 6:
            lines.append(None)
            continue
        m, dd, k, rms = _fit_line(pts)
        lines.append((m, dd) if rms <= 2.0 and k >= 6 else None)
    if lines[0] is not None and lines[1] is not None:
        (p, d), (q, e) = lines
        t = np.linalg.solve(np.array([d, -e]).T, q - p)
        x = p + t[0] * d
        status = "both lines"
    elif lines[0] is not None or lines[1] is not None:
        # one line hidden (player, shadow): snap onto the visible line only
        p, d = lines[0] if lines[0] is not None else lines[1]
        x = p + ((c - p) @ d) * d
        status = "one line"
    else:
        return c, "click only"
    if np.linalg.norm(x - c) > 25:
        return c, "click only"
    return x, status


def calibrate_unknown_focal(image_points, world_points, size, f_range=(300, 80000)):
    """Pose AND focal length (square pixels, centred principal point, no lens
    distortion -- fine for a long broadcast zoom). Returns (Camera, rms_px, f)."""
    from scipy.optimize import minimize_scalar
    obj = np.asarray(world_points, np.float64).reshape(-1, 3)
    img = np.asarray(image_points, np.float64).reshape(-1, 2)
    cx, cy = size[0] / 2, size[1] / 2
    planar = np.ptp(obj[:, 2]) < 1e-6

    def solve(f):
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        flags = cv2.SOLVEPNP_IPPE if planar else cv2.SOLVEPNP_SQPNP
        ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=flags)
        if not ok:
            return np.inf, None
        rv, tv = cv2.solvePnPRefineLM(obj, img, K, None, rv, tv)
        pr, _ = cv2.projectPoints(obj, rv, tv, K, None)
        e = float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - img) ** 2, 1))))
        return e, (K, rv, tv)

    fs = np.geomspace(*f_range, 300)
    errs = [solve(f)[0] for f in fs]
    k = int(np.nanargmin(errs))
    lo, hi = np.log(fs[max(k - 1, 0)]), np.log(fs[min(k + 1, len(fs) - 1)])
    res = minimize_scalar(lambda lf: solve(np.exp(lf))[0], bounds=(lo, hi), method="bounded")
    f = float(np.exp(res.x))
    e, (K, rv, tv) = solve(f)
    R, _ = cv2.Rodrigues(rv)
    cam = Camera(K, R, tv.ravel(), size=size, name="broadcast")
    # how sharply is f determined? (rms at +-5 %)
    sharp = (solve(f * 0.95)[0] + solve(f * 1.05)[0]) / 2 - e
    return cam, e, f, sharp


def _model_lines():
    """Painted lines as 3D segments (x positive = image left, see CREASE_POINTS)."""
    L, P, C, R = PITCH_LENGTH, CREASE_OFFSET, RETURN_CREASE_HALF, 1.83
    segs = []
    for yb, sgn in ((0.0, 1.0), (L, -1.0)):
        yp = yb + sgn * P
        segs.append(((-R, yp, 0.0), (R, yp, 0.0)))               # popping crease
        segs.append(((-C, yb, 0.0), (C, yb, 0.0)))               # bowling crease
        for x in (-C, C):                                        # return creases
            segs.append(((x, yp, 0.0), (x, yb - sgn * 1.0, 0.0)))
    return [(np.array(a, float), np.array(b, float)) for a, b in segs]


def refine_camera_lines(cam, img, anchors=None, iters=3, half=12, step_px=3.0, skip_px=10.0,
                        anchor_sigma_px=3.0):
    """Fit pose + focal length to ALL visible painted-line evidence: for every
    crease segment, find the line's sub-pixel centre perpendicular to its
    projection, then minimise the distances to the projected model lines.
    Much more robust than eight clicked points when players hide parts of
    the lines. `anchors` = (image_points, world_points) keep the solution from
    sliding along the lines (weighted as `anchor_sigma_px` click error)."""
    from scipy.optimize import least_squares
    I = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
    segs = _model_lines()
    h, w = I.shape
    rv, _ = cv2.Rodrigues(cam.R)
    x0 = np.r_[rv.ravel(), cam.t, np.log(cam.K[0, 0])]
    cx, cy = cam.K[0, 2], cam.K[1, 2]

    def make(x):
        K = np.array([[np.exp(x[6]), 0, cx], [0, np.exp(x[6]), cy], [0, 0, 1.0]])
        R, _ = cv2.Rodrigues(x[:3])
        return Camera(K, R, x[3:6], size=cam.size, name=cam.name, fps=cam.fps)

    obs = []
    for _ in range(iters):
        c = make(x0)
        obs = []                       # (segment index, observed image point)
        for si, (A, B) in enumerate(segs):
            uv = c.project(np.stack([A, B]))
            if not np.all(np.isfinite(uv)):
                continue
            d = uv[1] - uv[0]
            Lpx = np.linalg.norm(d)
            if Lpx < 30:
                continue
            d /= Lpx
            n = np.array([-d[1], d[0]])
            ts = np.arange(skip_px, Lpx - skip_px, step_px)
            pts = uv[0][None, :] + ts[:, None] * d[None, :]
            inside = (pts[:, 0] > half + 2) & (pts[:, 0] < w - half - 2) & \
                     (pts[:, 1] > half + 2) & (pts[:, 1] < h - half - 2)
            found = _ridge_profile(I, pts[inside], n, half)
            if len(found) >= 6:          # ignore a segment with too little evidence
                for q in found:
                    obs.append((si, q))
        if len(obs) < 30:
            break

        def res(x, lines_only=False):
            c = make(x)
            out = []
            for si, q in obs:
                A, B = segs[si]
                a, b = c.project(np.stack([A, B]))
                d = b - a
                nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
                out.append((q - a) @ nrm)
            if anchors is not None and not lines_only:
                pr = c.project(np.asarray(anchors[1], float))
                # one anchor = as much weight as ~20 line samples
                out.extend((np.sqrt(20) * (pr - np.asarray(anchors[0], float)) / anchor_sigma_px).ravel())
            return np.array(out)

        r = least_squares(res, x0, loss="soft_l1", f_scale=1.0, x_scale="jac")
        x0 = r.x
    final = make(x0)
    if obs:
        rr = np.abs(res(x0, lines_only=True))
        inl = rr < 2.0
        stats = dict(line_points=int(len(obs)), inliers=int(inl.sum()),
                     line_rms_px=float(np.sqrt(np.mean(rr[inl] ** 2))) if inl.any() else np.inf)
    else:
        stats = dict(line_points=0, inliers=0, line_rms_px=np.inf)
    return final, stats


def calibrate_from_clicks(img, clicks, refine=True):
    """clicks: {name: [u, v]} using CREASE_POINTS names (others ignored)."""
    names = [n for n in CREASE_POINTS if n in clicks]
    if len(names) < 6:
        raise ValueError(f"need at least 6 crease points, got {len(names)}: {names}")
    pts, refined = [], {}
    # rough vanishing direction of the return creases from the clicks themselves
    for n in names:
        c = np.asarray(clicks[n], float)
        partner = n.replace("_L", "_R") if n.endswith("_L") else n.replace("_R", "_L")
        along = (np.asarray(clicks[partner], float) - c) if partner in clicks else np.array([1.0, 0])
        # return crease direction: towards the same-side point at the same end
        other = n.replace("popping", "TMP").replace("bowling", "popping").replace("TMP", "bowling")
        down = (np.asarray(clicks[other], float) - c) if other in clicks else np.array([0, 1.0])
        x, how = (refine_intersection(img, c, along, down) if refine else (c, "click only"))
        refined[n] = dict(click=c.tolist(), refined=np.asarray(x).tolist(), snapped=how)
        pts.append(x)
    world = [CREASE_POINTS[n] for n in names]
    h, w = img.shape[:2]
    cam, rms, f, sharp = calibrate_unknown_focal(np.array(pts), np.array(world), (w, h))
    info = dict(click_rms_px=rms, focal_px_points=f, focal_sharpness=sharp, points=refined)
    if refine:
        cam2, st = refine_camera_lines(cam, img, anchors=(np.array(pts), np.array(world)))
        if st["inliers"] >= 40 and st["line_rms_px"] < 1.5:
            cam = cam2
        info.update(st)
    # final check: reprojection of the click points (after snapping)
    pr = cam.project(np.array(world))
    info["rms_px"] = float(np.sqrt(np.mean(np.sum((pr - np.array(pts)) ** 2, 1))))
    info["focal_px"] = float(cam.K[0, 0])
    info["camera_position_m"] = cam.center.round(2).tolist()
    return cam, info


# ---------------------------------------------------------------- automatic calibration
def white_line_mask(img):
    """Painted lines: thin, brighter than their surroundings and unsaturated."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = max(15, int(round(img.shape[1] / 64)) | 1)        # wider than the widest painted line
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return ((th > 25) & (hsv[..., 1] < 70)).astype(np.uint8) * 255


def auto_crease_points(img):
    """Find the 8 crease corners without clicks, for a camera behind the
    bowler (the usual TV end-on view). Returns ({name: [u, v]}, info) or
    raises RuntimeError if the pitch markings can't be found.

    1. The creases are the near-horizontal painted lines. Their tilt comes from
       the longest straight segment; a tilt-corrected row profile of the white
       pixels then shows one peak per line.
    2. The two popping creases are the strongest pair of peaks far apart; each
       bowling crease is the strongest peak just outside its popping crease.
    3. Between each popping/bowling pair the return creases show as column
       peaks. The left/right pair is chosen so the far end is narrower than the
       near end and both are centred on the same pitch.
    4. Each return crease is fitted as a line and intersected with the creases.
    The snapping and line refinement in calibrate_from_clicks do the rest."""
    from scipy.signal import find_peaks
    H, W = img.shape[:2]
    m = white_line_mask(img)
    L = cv2.HoughLinesP(m, 1, np.pi / 720, 80, minLineLength=W // 12, maxLineGap=25)
    if L is None:
        raise RuntimeError("no painted lines found")
    L = L.reshape(-1, 4).astype(float)
    ang = np.degrees(np.arctan2(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0]))
    hor = L[(np.abs(ang) < 6) | (np.abs(np.abs(ang) - 180) < 6)]
    if not len(hor):
        raise RuntimeError("no crease lines found")
    longest = hor[np.argmax(np.hypot(hor[:, 2] - hor[:, 0], hor[:, 3] - hor[:, 1]))]
    slope = (longest[3] - longest[1]) / (longest[2] - longest[0])
    ys, xs = np.nonzero(m)
    vv = np.round(ys - slope * (xs - W / 2)).astype(int)
    ok = (vv >= 0) & (vv < H)
    prof = np.bincount(vv[ok], minlength=H)[:H].astype(float)
    pk, _ = find_peaks(prof, height=0.03 * W, distance=8)
    if len(pk) < 4:
        raise RuntimeError(f"only {len(pk)} crease lines found")
    # popping creases: strongest pair of lines far apart
    pairs = [(prof[a] + prof[b], a, b) for i, a in enumerate(pk) for b in pk[i + 1:] if b - a > H / 6]
    if not pairs:
        raise RuntimeError("could not find both popping creases")
    _, pf, pn = max(pairs)
    sep = pn - pf
    far_b = [p for p in pk if pf - 0.15 * sep <= p < pf - 3]
    near_b = [p for p in pk if pn + 3 < p <= pn + 0.15 * sep]
    if not far_b or not near_b:
        raise RuntimeError("could not find both bowling creases")
    bf = max(far_b, key=lambda p: prof[p])
    bn = max(near_b, key=lambda p: prof[p])

    def column_peaks(v1, v2):
        lo, hi = min(v1, v2) + 3, max(v1, v2) - 3
        sel = ok & (vv > lo) & (vv < hi)
        cols = np.bincount(xs[sel], minlength=W)[:W].astype(float) / max(hi - lo, 1)
        cs = np.convolve(cols, np.ones(5), "same")
        p, _ = find_peaks(cs, height=1.2, distance=15)
        return p, cs
    fp, fcs = column_peaks(pf, bf)
    np_, ncs = column_peaks(pn, bn)
    best = None
    for i, nl in enumerate(np_):
        for nr in np_[i + 1:]:
            wn = nr - nl
            if wn < 0.15 * W:
                continue
            for j, fl in enumerate(fp):
                for fr in fp[j + 1:]:
                    wf = fr - fl
                    if not 0.3 * wn < wf < 0.98 * wn:
                        continue
                    if abs((fl + fr) / 2 - (nl + nr) / 2) > 0.12 * wn:
                        continue
                    sc = ncs[nl] + ncs[nr] + fcs[fl] + fcs[fr]
                    if best is None or sc > best[0]:
                        best = (sc, nl, nr, fl, fr)
    if best is None:
        raise RuntimeError("could not find the return creases at both ends")
    _, nl, nr, fl, fr = best

    def line_v(v0, u):
        return v0 + slope * (u - W / 2)

    def return_line(u0, v1, v2):
        lo, hi = min(v1, v2) - 2, max(v1, v2) + 2
        sel = (ys >= lo) & (ys <= hi) & (np.abs(xs - u0) <= 12)
        if sel.sum() < 6:
            return 0.0, float(u0)
        a, b = np.polyfit(ys[sel], xs[sel], 1)            # u = a v + b
        return a, b

    def meet(rl, v0):
        a, b = rl                                         # u = a v + b ; v = v0 + slope (u - W/2)
        v = (v0 + slope * (b - W / 2)) / (1 - slope * a)
        return [float(a * v + b), float(v)]
    out = {}
    for end, (vp, vb, ul, ur) in (("bowler", (pn, bn, nl, nr)), ("striker", (pf, bf, fl, fr))):
        for side, u0 in (("L", ul), ("R", ur)):
            rl = return_line(u0, line_v(vp, u0), line_v(vb, u0))
            out[f"{end}_popping_crease_{side}"] = meet(rl, vp)
            out[f"{end}_bowling_crease_{side}"] = meet(rl, vb)
    info = dict(slope=float(slope), lines=dict(far_bowling=int(bf), far_popping=int(pf),
                                               near_popping=int(pn), near_bowling=int(bn)),
                return_creases=dict(near=[int(nl), int(nr)], far=[int(fl), int(fr)]))
    return out, info


def auto_front_foot(video, impact_frame, impact_uv, ref_frames, ball_r=8.0):
    """Estimate where the batter's front foot meets the ground at impact,
    without a click. The batter's leg moved between the stance (ref_frames,
    before release) and impact, so it shows up in the difference image; the
    front foot is the lowest moving point below the ball (the camera is behind
    the bowler, so the foot nearest the bowler is lowest in the image).
    Returns ([u, v], info) or (None, info)."""
    u0, v0 = float(impact_uv[0]), float(impact_uv[1])
    img = dict(read_frames(video, [int(impact_frame)]))[int(impact_frame)]
    ref = np.median(np.stack([f for _, f in read_frames(video, list(ref_frames))]), axis=0).astype(np.uint8)
    H, W = img.shape[:2]
    s = max(1.0, ball_r / 8.0)                            # scale windows with the ball size
    x0, x1 = int(max(0, u0 - 90 * s)), int(min(W, u0 + 90 * s))
    y0, y1 = int(max(0, v0 - 10 * s)), int(min(H, v0 + 220 * s))
    a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[y0:y1, x0:x1]
    b = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)[y0:y1, x0:x1]
    fg = (np.linalg.norm(a - b, axis=2) > 25).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cl, cr = int(u0 - 30 * s) - x0, int(u0 + 40 * s) - x0
    col = fg[:, max(cl, 0):max(cr, 1)]
    rows = col.sum(1)
    start = int(v0) - y0
    last, gap = None, 0
    for r in range(max(start, 0), col.shape[0]):
        if rows[r] >= 3:
            last, gap = r, 0
        elif last is not None:
            gap += 1
            if gap > 25 * s:
                break
    if last is None or last - start < 15 * s:
        return None, dict(found=False)
    xs = np.nonzero(col[max(last - 3, 0):last + 1].any(0))[0]
    foot = [float(x0 + max(cl, 0) + np.median(xs)), float(y0 + last)]
    return foot, dict(found=True, frame=int(impact_frame), method="moving leg below the ball")


# ---------------------------------------------------------------- ball candidates
def corridor_roi(cam, shape, pad_x=1.8, z_top=3.2):
    pts = np.array([[x, y, z] for x in (-pad_x, pad_x) for y in (-0.8, PITCH_LENGTH + 1.0)
                    for z in (0.0, z_top)])
    uv = cam.project(pts)
    roi = np.zeros(shape[:2], np.uint8)
    if np.all(np.isfinite(uv)):
        cv2.fillConvexPoly(roi, cv2.convexHull(uv.astype(np.int32)), 255)
    return roi


def expected_radius_px(cam, uv):
    """Ball radius in pixels for a ball seen at pixel uv, assuming it is over
    the pitch (ray intersected with the plane z = 0.5 m)."""
    p = cam.ray_plane(np.atleast_2d(uv), z=0.5)
    d = np.linalg.norm(p - cam.center, axis=1)
    return cam.K[0, 0] * BALL_RADIUS / d


def _blob_filter(D, r):
    r = max(2, int(round(r)))
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)).astype(np.float32)
    R = int(round(r * 1.9))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1, 2 * R + 1)).astype(np.float32)
    k2[R - r:R + r + 1, R - r:R + r + 1] -= k1
    return cv2.filter2D(D, -1, k1 / k1.sum()) - cv2.filter2D(D, -1, k2 / k2.sum())


def warmth(img):
    """(R - B) / (R + G + B): a red/brown ball is 'warmer' than the tan pitch,
    while shadows keep the pitch's ratio and hair, shoes, blue helmets and
    white kit are cooler."""
    f = img.astype(np.float32)
    return (f[..., 2] - f[..., 0]) / (f.sum(2) + 30.0)


def ball_candidates(frame, bg, roi, r_px, thresh=8.0, top=40, ball="red", bg_warm=None,
                    bg_gray=None):
    """Compact blobs darker than the background and (for a red ball) warmer
    in colour. Returns rows (u, v, score)."""
    if bg_gray is None:
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    D = np.clip(g - bg_gray if ball == "white" else bg_gray - g, 0, None)
    if ball == "white":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        D *= (hsv[..., 1] < 70) & (g >= 155)
    if ball == "red":
        if bg_warm is None:
            bg_warm = warmth(bg)
        W = warmth(frame) - bg_warm
        D = D * np.clip(0.3 + W / 0.08, 0.0, 2.0)   # cool things are suppressed, warm boosted
    D[roi == 0] = 0
    S = _blob_filter(D, 0.7 * r_px)
    mx = cv2.dilate(S, np.ones((9, 9), np.uint8))
    ys, xs = np.where((S >= mx) & (S > thresh))
    if not len(xs) or top <= 0:
        return np.zeros((0, 3))
    sc = S[ys, xs]
    # isolation: a ball in flight is surrounded by background; a shirt number,
    # a shoe or a glove is surrounded by more of the player (foreground).
    F = (np.abs(frame.astype(np.int16) - bg.astype(np.int16)).sum(2) > 45).astype(np.float32)
    rr = max(2, int(round(0.7 * r_px)))
    R = int(round(rr * 2.4))
    ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1, 2 * R + 1)).astype(np.float32)
    inner = int(round(rr * 1.4))
    ring[R - inner:R + inner + 1, R - inner:R + inner + 1] -= cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * inner + 1, 2 * inner + 1))
    ring = np.clip(ring, 0, 1)
    busy = cv2.filter2D(F, -1, ring / ring.sum())[ys, xs]
    sc = sc * (1.15 - busy)
    o = np.argsort(-sc)
    out = []
    for k in o:
        x, y = xs[k], ys[k]
        w = int(max(3, r_px))
        win = D[max(0, y - w):y + w + 1, max(0, x - w):x + w + 1]
        if win.sum() > 0:
            yy, xx = np.mgrid[max(0, y - w):y + w + 1, max(0, x - w):x + w + 1]
            u, v = float((xx * win).sum() / win.sum()), float((yy * win).sum() / win.sum())
            if ball == "white" and any((u - a) ** 2 + (v - b) ** 2 < 4 for a, b, _ in out):
                continue
            out.append((u, v, float(sc[k])))
            if len(out) >= top:
                break
    return np.asarray(out, float).reshape(-1, 3)


# ---------------------------------------------------------------- tracking
def _nodes(cands, held, link_px=3.0):
    """Merge a candidate that stays put over consecutive frames (a held or
    near-duplicate frame) into one node: (f0, f1, u, v, score)."""
    nodes, open_ = [], []
    for f in sorted(cands):
        rows = cands[f]
        new_open = []
        used = np.zeros(len(rows), bool)
        for nd in open_:
            if len(rows):
                d = np.hypot(rows[:, 0] - nd["u"], rows[:, 1] - nd["v"])
                d[used] = np.inf
                k = int(np.argmin(d))
                if d[k] < link_px:
                    used[k] = True
                    nd["f1"] = f
                    nd["score"] = max(nd["score"], rows[k, 2])
                    nd["n"] += 1
                    new_open.append(nd)
                    continue
            nodes.append(nd)
        for k in np.where(~used)[0]:
            new_open.append(dict(f0=f, f1=f, u=rows[k, 0], v=rows[k, 1], score=rows[k, 2], n=1))
        open_ = new_open
    nodes += open_
    return nodes


def _region_mask(cam, shape, xr, yr, zr):
    pts = np.array([[x, y, z] for x in xr for y in yr for z in zr])
    uv = cam.project(pts)
    m = np.zeros(shape[:2], np.uint8)
    if np.all(np.isfinite(uv)):
        cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.int32)), 255)
    return m


def track_delivery(cands, held=None, cam=None, shape=None, max_step=120.0, max_gap=8,
                   max_turn_deg=45.0, node_cost=30.0):
    """Find the ball's path through the candidate nodes.

    Model (no timing assumed): distinct positions that move forward in time,
    steps of at most `max_step` px, turning smoothly except at ONE bounce,
    with the image step length changing gradually. Every node costs
    `node_cost`, so only convincing detections add to a path. If the camera
    is known, the path must START in the release region (over the bowler's
    end, 1.4-2.9 m up) and END at the batter (in front of the striker's
    stumps, below 1.3 m) -- that is what a delivery looks like.
    Dynamic programming over (previous node, current node, bounced?).
    """
    nodes = _nodes(cands, held)
    nodes.sort(key=lambda n: (n["f0"], n["f1"]))
    N = len(nodes)
    if N < 3:
        return []
    P = np.array([[n["u"], n["v"]] for n in nodes])
    f0 = np.array([n["f0"] for n in nodes])
    f1 = np.array([n["f1"] for n in nodes])
    sc = np.array([n["score"] for n in nodes]) - node_cost
    can_start = np.ones(N, bool)
    can_end = np.ones(N, bool)
    if cam is not None and shape is not None:
        rel = _region_mask(cam, shape, (-2.2, 2.2), (16.5, 20.8), (1.4, 2.9))
        bat = _region_mask(cam, shape, (-0.8, 0.8), (0.2, 3.0), (0.0, 1.3))
        ui = np.clip(P[:, 0].astype(int), 0, shape[1] - 1)
        vi = np.clip(P[:, 1].astype(int), 0, shape[0] - 1)
        can_start = rel[vi, ui] > 0
        can_end = bat[vi, ui] > 0
        if not can_start.any() or not can_end.any():
            can_start[:] = True
            can_end[:] = True
    # successors (vectorised)
    succ = []
    for a in range(N):
        gap = f0 - f1[a]
        d = np.hypot(P[:, 0] - P[a, 0], P[:, 1] - P[a, 1])
        ok = (gap >= 1) & (gap <= max_gap) & (d > 2.0) & (d < max_step)
        succ.append(np.where(ok)[0])
    cos_lim = np.cos(np.radians(max_turn_deg))
    best = {}
    by_cur = {}
    for a in np.where(can_start)[0]:
        for b in succ[a]:
            k = (int(a), int(b), 0)
            best[k] = (sc[a] + sc[b], None)
            by_cur.setdefault(int(b), []).append(k)
    for b in sorted(range(N), key=lambda i: (f0[i], f1[i])):
        for key in by_cur.get(b, []):
            s_, _ = best[key]
            a, _, bounced = key
            d1 = P[b] - P[a]
            n1 = np.linalg.norm(d1)
            cs = succ[b]
            if not len(cs):
                continue
            d2 = P[cs] - P[b]
            n2 = np.linalg.norm(d2, axis=1)
            cosang = (d2 @ d1) / (n1 * n2)
            ratio_pen = 2.0 * np.abs(np.log(np.maximum(n2 / n1, 1e-3)))
            for j, c in enumerate(cs):
                for nb in (0, 1):
                    if nb == bounced and cosang[j] >= cos_lim:
                        pen = 0.0
                    elif nb == 1 and bounced == 0:
                        pen = 5.0
                    else:
                        continue
                    nk = (b, int(c), nb)
                    ns = s_ + sc[c] - pen - ratio_pen[j]
                    if nk not in best or ns > best[nk][0]:
                        if nk not in best:
                            by_cur.setdefault(int(c), []).append(nk)
                        best[nk] = (ns, key)
    finals = [k for k in best if can_end[k[1]]]
    if not finals:
        return []
    end = max(finals, key=lambda k: best[k][0])
    path = [end[1]]
    k = end
    while k is not None:
        path.append(k[0])
        k = best[k][1]
    path = path[::-1]
    return [nodes[i] for i in path]


def samples_from_track(track):
    """One observation per distinct ball position. 'k' is the sample's order
    index (the only timing we trust); f0..f1 the frames it was shown in."""
    return [dict(k=i, u=float(n["u"]), v=float(n["v"]), f0=int(n["f0"]), f1=int(n["f1"]),
                 score=float(n["score"])) for i, n in enumerate(track)]


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
