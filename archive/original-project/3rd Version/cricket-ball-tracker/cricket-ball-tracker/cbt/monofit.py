"""3D ball path from ONE camera with NO usable timing (broadcast replays).

Why this works for the usual camera behind the bowler
-----------------------------------------------------
Each image position of the ball is a ray from a camera ~60-80 m behind the
bowler. Along such a ray the ball's sideways position (x) and height (z) barely
change with distance: here dz/dy ~ 0.09 and dx/dy ~ 0.005. So the image fixes x
and z precisely; only the distance down the pitch (y) is loose, and three
physical facts pin that down:
  * the bounce is ON the pitch (z = ball radius)        -> exact, from the image
  * the last tracked position is at the batter's pad    -> front-foot position
  * the first tracked position is near the bowler       -> weak prior

Model (time-free): the path is written as a function of the distance down the
pitch, s = y - y_bounce (s > 0 before the bounce, s < 0 after):
    x(s) = xb + p1 s + p2 s^2           z(s) = r + q1 s + q2 s^2      (s >= 0)
    x(s) = xb + p3 s + p4 s^2           z(s) = r + q3 s + q4 s^2      (s <= 0)
Gravity sets the z-curvature: q2 = -g / (2 vy^2). Each observation gets its own
unknown s_i (ordered in time). Everything is fitted jointly by minimising pixel
error plus weak physical priors; the uncertainty of the stump-plane crossing
comes from the fit's covariance.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares

from .pitch import BALL_RADIUS as R, G
from . import lbw

SPEED_PRIORS = {                    # along-pitch speed before the bounce (m/s): mean, sigma
    "pace": (35.0, 3.5),            # ~120-150 km/h at release
    "medium": (28.0, 3.0),          # ~95-120 km/h
    "spin": (22.0, 3.0),            # ~70-95 km/h
}
PRIOR = dict(
    vy_range=(17.0, 44.0),          # m/s along the pitch, spin .. very fast pace
    retained=(0.87, 0.05),          # vy after / vy before the bounce
    restitution=(0.45, 0.12),       # vz after / vz before
    deviation_sigma=0.05,           # change in x-slope at the bounce (seam / spin)
    swing_sigma=0.0015,             # x-curvature (swing / drift), 1/m
    first_y=(12.0, 19.8),           # first tracked position: between here and the crease
    first_z=(0.6, 2.9),
    pad_offset=(0.10, 0.20),        # impact point vs front-foot toe: m in front, sigma (incl. click error)
    last_y=(0.15, 3.6),             # without a front-foot position
)


def _box(v, lo, hi, soft):
    """0 inside [lo, hi], linear (in units of `soft`) outside."""
    return max(lo - v, 0.0, v - hi) / soft


@dataclass
class MonoFit:
    params: np.ndarray            # 10 path parameters (see module doc)
    s: np.ndarray                 # per-observation position along the pitch (s_i)
    cov: np.ndarray               # covariance of the 10 path parameters
    obs_uv: np.ndarray
    inlier: np.ndarray
    rms_px: float
    vy: float                     # estimated along-pitch speed before the bounce (m/s)
    vy_sigma: float
    impact_prior: str = "none"
    meta: dict = field(default_factory=dict)

    # -- geometry -------------------------------------------------------
    @staticmethod
    def path(params, s):
        yb, xb, p1, p2, q1, q2, p3, p4, q3, q4 = params
        s = np.asarray(s, float)
        pre = s >= 0
        x = xb + np.where(pre, p1 * s + p2 * s * s, p3 * s + p4 * s * s)
        z = R + np.where(pre, q1 * s + q2 * s * s, q3 * s + q4 * s * s)
        return np.stack([x, yb + s, z], -1)

    def at_s(self, s):
        return self.path(self.params, s)

    @property
    def bounce(self):
        return np.array([self.params[1], self.params[0], R])

    @property
    def impact(self):
        return self.at_s(self.s[self.inlier][-1])

    def stumps(self, params=None):
        p = self.params if params is None else params
        P = self.path(p, [-p[0]])[0]
        return P[[0, 2]]

    # -- pseudo time so the replay/console can animate it -----------------
    @property
    def vy_after(self):
        return self.vy * PRIOR["retained"][0]

    def _t_of_s(self, s):
        s = np.asarray(s, float)
        s0 = self.s[self.inlier][0]
        tb = s0 / self.vy
        return np.where(s >= 0, (s0 - s) / self.vy, tb + (-s) / self.vy_after)

    @property
    def t_first(self):
        return 0.0

    @property
    def tb(self):
        return float(self._t_of_s(0.0))

    @property
    def t_last(self):
        return float(self._t_of_s(self.s[self.inlier][-1]))

    def pos(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        s0 = self.s[self.inlier][0]
        tb = self.tb
        s = np.where(t <= tb, s0 - self.vy * t, -(t - tb) * self.vy_after)
        return self.at_s(s)

    @property
    def release_speed_kmh(self):
        return self.vy * 3.6

    # -- review ----------------------------------------------------------
    def sample_params(self, n, rng):
        return rng.multivariate_normal(self.params, self.cov, size=n, check_valid="ignore")

    def problems(self):
        p = []
        b = self.bounce
        if not (-1.6 < b[0] < 1.6 and -0.5 < b[1] < 21.0):
            p.append("bounce point off the pitch")
        n_post = int(np.sum(self.s[self.inlier] < 0))
        if n_post < 2:
            p.append(f"only {n_post} positions after the bounce")
        if self.impact[1] < -0.2 or self.impact[1] > 4.0:
            p.append("impact position implausible")
        st = self.stumps()
        if not np.all(np.isfinite(st)) or st[1] < 0.0 or st[1] > 2.0 or abs(st[0]) > 2.0:
            p.append("projected path is not physical (below the ground or far off)")
        if self.rms_px > 3.0:
            p.append(f"path fits the ball positions poorly ({self.rms_px:.1f}px rms)")
        return p

    def report(self, handed="right", n_mc=500, seed=0, on_field="not out", shot_offered=True):
        rng = np.random.default_rng(seed)
        sxz = np.array([self.stumps(q) for q in self.sample_params(n_mc, rng)])
        bsig = np.sqrt(np.clip(np.diag(self.cov)[[1, 0]], 0, None))
        # No timing in the footage: speed is only ESTIMATED (from how the path
        # curves under gravity), and only shown if that pinned it down.
        speed = self.release_speed_kmh if self.vy_sigma / self.vy < 0.08 else None
        n_in = int(self.inlier.sum())
        return lbw.build_report(
            self.bounce, self.impact, self.stumps(), sxz, bsig, None, handed, on_field,
            shot_offered, self.problems(),
            extra=dict(rms_px=round(self.rms_px, 2), n_obs=len(self.obs_uv), n_inliers=n_in,
                       n_after_bounce=int(np.sum(self.s[self.inlier] < 0)),
                       method="single camera, time-free", impact_prior=self.impact_prior,
                       speed_estimate_kmh=None if speed is None else round(speed, 1),
                       speed_sigma_kmh=round(self.vy_sigma * 3.6, 1)))


def _residuals(theta, n, cams_proj, uv, sig_px, foot_y, speed_prior=None, ordering_w=3.0):
    par, s = theta[:10], theta[10:]
    yb, xb, p1, p2, q1, q2, p3, p4, q3, q4 = par
    X = MonoFit.path(par, s)
    res = [((cams_proj(X) - uv) / sig_px).ravel()]
    # time order: the ball moves towards the batter, so s strictly decreases
    ds = s[1:] - s[:-1]
    res.append(ordering_w * np.maximum(ds + 0.02, 0) / 0.02)
    # gravity <-> speed: q2 = -g / (2 vy^2)
    vlo, vhi = PRIOR["vy_range"]
    q2_lo, q2_hi = -G / (2 * vlo ** 2), -G / (2 * vhi ** 2)
    pri = [_box(q2, q2_lo, q2_hi, 0.0005)]
    if speed_prior is not None:                 # bowler type or a speed-gun reading
        mu, sd = speed_prior
        vy = np.sqrt(G / (2 * max(-q2, 1e-6)))
        pri.append((vy - mu) / sd)
    k, ks = PRIOR["retained"]
    pri.append((q4 * k * k - q2) / (abs(q2) * 2 * ks / k + 1e-4))      # curvature after bounce
    e, es = PRIOR["restitution"]
    pri.append((-q3 * k - e * q1) / (es * abs(q1) + 0.01))             # vz ratio at the bounce
    pri.append((p3 - p1) / PRIOR["deviation_sigma"])
    pri.append(p2 / PRIOR["swing_sigma"])
    pri.append(p4 / PRIOR["swing_sigma"])
    y0, z0 = yb + s[0], X[0, 2]
    pri.append(_box(y0, *PRIOR["first_y"], 0.5))
    pri.append(_box(z0, *PRIOR["first_z"], 0.2))
    yl = yb + s[-1]
    if foot_y is not None:
        mu, sd = PRIOR["pad_offset"]
        pri.append((yl - (foot_y + mu)) / sd)
    else:
        pri.append(_box(yl, *PRIOR["last_y"], 0.3))
    pri.append(_box(q1, 0.0, 0.6, 0.02))                               # descending into the bounce
    pri.append(_box(-q3, 0.0, 0.8, 0.02))                              # rising after it
    res.append(np.array(pri))
    return np.concatenate(res)


def speed_prior_for(bowler=None, speed_kmh=None):
    """Prior on the along-pitch speed: a speed-gun reading if the broadcast
    shows one (release speed; ~2 % is lost to drag on the way), else the
    bowler type, else None (wide box only)."""
    if speed_kmh:
        return (0.97 * speed_kmh / 3.6, 0.8)
    if bowler in SPEED_PRIORS:
        return SPEED_PRIORS[bowler]
    return None


def fit_mono(samples_uv, cam, foot_uv=None, foot_y=None, sig_px=1.5, max_outlier_px=6.0,
             bowler=None, speed_kmh=None):
    """samples_uv: (n, 2) ball positions in time order (one per distinct
    sample). foot_uv: the batter's front-foot ground contact at impact, in
    the image (or foot_y in metres). bowler: "pace" / "spin"; speed_kmh: a
    speed-gun reading if shown. Returns MonoFit."""
    sp = speed_prior_for(bowler, speed_kmh)
    uv = np.asarray(samples_uv, float)
    n = len(uv)
    if n < 6:
        raise ValueError("need at least 6 ball positions")
    if foot_uv is not None and foot_y is None:
        foot_y = float(cam.ray_plane(np.atleast_2d(foot_uv), z=0.0)[0][1])
    proj = cam.project

    # --- initial guesses from the image alone. The fit has local minima (the
    # depth of each sample is free), so several starts are tried and the best
    # kept: different release distances, speeds and ways of spreading the
    # samples along the pitch (evenly, or in proportion to image distance --
    # edited slow motion samples the flight very unevenly).
    kb = int(np.argmax(uv[:, 1]))          # bounce sample: lowest image point
    g = cam.ray_plane(uv[kb:kb + 1], z=R)[0]
    yb0, xb0 = g[1], g[0]
    o, d = cam.ray(uv)

    def spread(a, b, seg_uv, arclen):
        m = len(seg_uv)
        if m == 1:
            return np.array([a])
        if not arclen:
            return np.linspace(a, b, m)
        c = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seg_uv, axis=0), axis=1))]
        return a + (b - a) * c / max(c[-1], 1e-9)

    def start(y_first, vy, arclen):
        y_last = (foot_y + 0.15) if foot_y is not None else max(0.6, yb0 - 2.0)
        y_last = min(y_last, yb0 - 0.3)
        s_init = np.r_[spread(y_first - yb0, 0.0, uv[:kb + 1], arclen)[:-1] if kb > 0 else [],
                       [0.0], spread(0.0, y_last - yb0, uv[kb:], arclen)[1:]]
        tt = (yb0 + s_init - o[1]) / d[:, 1]
        P = o + tt[:, None] * d
        pre, post = s_init > 0, s_init < 0

        def lin(sv, v):
            return np.polyfit(sv, v, 1)[0] if len(sv) >= 2 else 0.0
        p1 = lin(s_init[pre], P[pre, 0] - xb0) if pre.sum() >= 2 else 0.0
        q1 = max(lin(s_init[pre], P[pre, 2] - R), 0.05) if pre.sum() >= 2 else 0.1
        p3 = lin(s_init[post], P[post, 0] - xb0) if post.sum() >= 2 else p1
        q3 = min(lin(s_init[post], P[post, 2] - R), -0.05) if post.sum() >= 2 else -0.2
        q2 = -G / (2 * vy ** 2)
        return np.r_[yb0, xb0, p1, 0.0, q1, q2, p3, 0.0, q3, q2 / 0.76, s_init]

    def solve(th0, idx):
        return least_squares(_residuals, th0, args=(len(idx), proj, uv[idx], sig_px, foot_y, sp),
                             loss="soft_l1", f_scale=2.0, x_scale="jac", max_nfev=4000)

    speeds = [sp[0]] if sp else [24.0, 32.0]
    starts = [start(yf, v, al) for yf in (17.0, 18.5) for v in speeds for al in (True, False)]
    all_idx = np.arange(n)
    x0 = min((solve(x, all_idx) for x in starts), key=lambda r: r.cost).x

    inl = np.ones(n, bool)
    sol = None
    for _ in range(3):
        idx = np.where(inl)[0]
        th0 = np.r_[x0[:10], x0[10:][idx]]
        r = solve(th0, idx)
        # every observation: its best s on the fitted curve (to re-admit / reject)
        par = r.x[:10]
        s_all = x0[10:].copy()
        s_all[idx] = r.x[10:]
        for j in np.where(~inl)[0]:
            grid = np.linspace(-par[0] - 0.5, 21 - par[0], 4000)
            e = np.linalg.norm(proj(MonoFit.path(par, grid)) - uv[j], axis=1)
            s_all[j] = grid[int(np.argmin(e))]
        err = np.linalg.norm(proj(MonoFit.path(par, s_all)) - uv, axis=1)
        new_inl = err < max_outlier_px
        # outliers can only be removed at the ends of the track (spurious
        # detections before release / after the ball reaches the pad)
        sol = (r, idx, s_all, err)
        x0 = np.r_[par, s_all]
        if np.array_equal(new_inl, inl):
            break
        inl = new_inl
        if inl.sum() < 6:
            break
    r, idx, s_all, err = sol
    inl = np.zeros(n, bool)
    inl[idx] = True
    J = r.jac
    m, k = J.shape
    s2 = max(1.0, 2 * r.cost / max(m - k, 1))
    cov_full = np.linalg.pinv(J.T @ J) * s2
    cov = cov_full[:10, :10]
    q2 = r.x[5]
    vy = float(np.sqrt(G / (2 * max(-q2, 1e-6))))
    q2s = float(np.sqrt(max(cov[5, 5], 0)))
    vy_sig = 0.5 * vy * q2s / max(abs(q2), 1e-9)
    rms = float(np.sqrt(np.mean(err[inl] ** 2)))
    return MonoFit(r.x[:10], s_all, cov, uv, inl, rms, vy, vy_sig,
                   impact_prior="front foot" if foot_y is not None else "none",
                   meta=dict(foot_y=foot_y, outlier_err_px=err.round(1).tolist(),
                             speed_prior=None if sp is None else list(sp),
                             speed_source="speed gun" if speed_kmh else (bowler or "none")))


# ---------------------------------------------------------------- track clean-up
def _curve_px(fit, cam, s_from, s_to, n=400):
    s = np.linspace(s_from, s_to, n)
    return s, cam.project(fit.at_s(s))


def reconstruct(nodes, cands, cam, foot_uv=None, bowler=None, speed_kmh=None,
                max_px=4.0, search_px=7.0, max_gap=10):
    """Turn tracker nodes into a trusted set of ball positions and a fit.

    1. Bounce = the lowest image point of the track (camera behind the bowler).
    2. Keep the longest prefix (up to the bounce + k nodes) that a physical
       path fits within `max_px` -- spurious endings after the ball reached
       the pad are dropped.
    3. Model-guided search: look in later frames for candidates that sit on
       the fitted path's continuation towards the batter and add them.
    4. Final fit with the front-foot prior.
    Returns (MonoFit, used_nodes, log)."""
    log = []
    uv_all = np.array([[n["u"], n["v"]] for n in nodes])
    b = int(np.argmax(uv_all[:, 1]))
    best = None
    for end in range(min(b + 1, len(nodes) - 1), len(nodes)):
        sub = nodes[:end + 1]
        uv = np.array([[n["u"], n["v"]] for n in sub])
        if len(uv) < 6:
            continue
        try:
            f = fit_mono(uv, cam, bowler=bowler, speed_kmh=speed_kmh, max_outlier_px=1e9)
        except Exception:
            continue
        err = np.linalg.norm(cam.project(f.at_s(f.s)) - uv, axis=1)
        ok = err.max() < max_px
        log.append(f"prefix to node {end}: max residual {err.max():.1f}px {'ok' if ok else 'rejected'}")
        if ok:
            best = (end, f)
    if best is None:
        raise RuntimeError("no physically consistent ball path found")
    end, f = best
    used = list(nodes[:end + 1])
    # model-guided extension towards the batter
    for _ in range(4):
        last = used[-1]
        s_last = f.s[-1]
        s_stop = -f.params[0] + 0.3                       # just short of the stumps
        if s_last <= s_stop:
            break
        _, curve = _curve_px(f, cam, s_last, s_stop)
        found = None
        for fr in sorted(k for k in cands if last["f1"] < k <= last["f1"] + max_gap):
            c = cands[fr]
            if not len(c):
                continue
            d = np.min(np.linalg.norm(c[:, None, :2] - curve[None, :, :], axis=2), axis=1)
            far = np.linalg.norm(c[:, :2] - np.array([last["u"], last["v"]]), axis=1) > 3.0
            j = np.where((d < search_px) & far)[0]
            if len(j):
                k = j[np.argmax(c[j, 2])]
                found = dict(f0=fr, f1=fr, u=float(c[k, 0]), v=float(c[k, 1]), score=float(c[k, 2]), n=1)
                # extend over the frames it stays put (held frames / ball resting on the pad)
                for fr2 in sorted(k2 for k2 in cands if k2 > fr):
                    c2 = cands[fr2]
                    if len(c2) and np.min(np.hypot(c2[:, 0] - found["u"], c2[:, 1] - found["v"])) < 3.0:
                        found["f1"] = fr2
                        found["n"] += 1
                    else:
                        break
                break
        if found is None:
            break
        used.append(found)
        uv = np.array([[n["u"], n["v"]] for n in used])
        f2 = fit_mono(uv, cam, bowler=bowler, speed_kmh=speed_kmh, max_outlier_px=1e9)
        err = np.linalg.norm(cam.project(f2.at_s(f2.s)) - uv, axis=1)
        if err.max() > max_px:
            used.pop()
            log.append(f"extension at frame {found['f0']} rejected ({err.max():.1f}px)")
            break
        log.append(f"extended with frame {found['f0']} at ({found['u']:.0f},{found['v']:.0f})")
        f = f2
    uv = np.array([[n["u"], n["v"]] for n in used])
    final = fit_mono(uv, cam, foot_uv=foot_uv, bowler=bowler, speed_kmh=speed_kmh)
    return final, used, log
