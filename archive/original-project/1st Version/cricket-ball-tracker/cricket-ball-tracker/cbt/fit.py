"""Physics-constrained 3D trajectory fit from one or more cameras.

Model: two constant-acceleration arcs joined at the bounce point B, which is
constrained to lie on the pitch (z = ball radius):

    p(t) = B + v (t - tb) + a (t - tb)^2 / 2      (v, a) = pre-bounce for t <= tb
                                                   (v, a) = post-bounce for t >  tb

All parameters (tb, Bx, By, v1, a1, v2, a2) are fitted by minimising the
pixel reprojection error over every detection from every camera, each at its
own timestamp. So:
  * cameras do NOT need to be frame-synchronised, only timestamped;
  * a single camera works (the ground constraint + a gravity prior fix depth);
  * every extra camera simply adds observations.
Weak priors keep accelerations physical (gravity, drag, swing).
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

from .pitch import BALL_RADIUS, G
from . import lbw

# Weak physical priors (each adds one residual). Tuned to cover real cricket
# balls generously; they matter most for a single camera, where they are what
# pins down depth/speed. Values to re-check against measured data later.
MAX_STUMPS_SIGMA = 0.05  # m; above this the wickets call is "inconclusive"
K_DRAG = 0.0065          # drag decel ~ K|v|^2 (Cd ~0.35-0.5, m=0.16 kg)
N_PRIORS = 9


def _priors(p14):
    _, v1, a1, v2, a2 = _split(p14)
    s1, s2 = v1 @ v1, v2 @ v2
    return np.array([
        a1[0] / 4.0,                               # swing (lateral accel)
        a2[0] / 4.0,
        (a1[2] + G) / 2.0,                         # gravity (+ small drag/lift)
        (a2[2] + G) / 2.0,
        (a1[1] - K_DRAG * s1) / (0.25 * K_DRAG * s1 + 1.0),   # drag along pitch
        (a2[1] - K_DRAG * s2) / (0.25 * K_DRAG * s2 + 1.0),
        (v2[1] / v1[1] - 0.875) / 0.06 if v1[1] != 0 else 0.0,  # speed kept through bounce
        (v2[2] / -v1[2] - 0.45) / 0.15 if v1[2] != 0 else 0.0,  # vertical restitution
        (v2[0] - 0.875 * v1[0]) / 1.5,             # sideways deviation off the pitch (seam/spin)
    ])


def _split(p):
    B = np.array([p[0], p[1], BALL_RADIUS])
    return B, p[2:5], p[5:8], p[8:11], p[11:14]


def model_pos(p, tb, t):
    B, v1, a1, v2, a2 = _split(p)
    dt = (np.asarray(t) - tb)[:, None]
    pre = dt <= 0
    v = np.where(pre, v1, v2)
    a = np.where(pre, a1, a2)
    return B + v * dt + 0.5 * a * dt ** 2


def model_vel(p, tb, t):
    _, v1, a1, v2, a2 = _split(p)
    dt = (np.asarray(t) - tb)[:, None]
    pre = dt <= 0
    return np.where(pre, v1, v2) + np.where(pre, a1, a2) * dt


@dataclass
class FitResult:
    params: np.ndarray        # 15: tb, Bx, By, v1, a1, v2, a2
    cov: np.ndarray
    rms_px: float
    n_obs: int
    n_inliers: int
    t_first: float
    t_last: float
    t_obs: np.ndarray = None   # timestamps of the observations used

    @property
    def n_after_bounce(self):
        return int(np.sum(self.t_obs > self.tb)) if self.t_obs is not None else -1

    def problems(self):
        """Reasons not to trust this fit (empty list = OK)."""
        p = []
        b, st, v = self.bounce, self.stumps(), self.release_speed_kmh
        if not (-1.6 < b[0] < 1.6 and -0.5 < b[1] < 21.0):
            p.append("bounce point off the pitch")
        if not 40 < v < 170:
            p.append(f"implausible speed {v:.0f} km/h")
        if not np.all(np.isfinite(st)) or abs(st[0]) > 2 or not -0.5 < st[1] < 3:
            p.append("projection to the stumps failed")
        if self.n_after_bounce < 4:
            p.append(f"only {self.n_after_bounce} detections after the bounce")
        return p

    @property
    def tb(self):
        return self.params[0]

    def pos(self, t):
        return model_pos(self.params[1:], self.tb, np.atleast_1d(t))

    @property
    def bounce(self):
        return np.array([self.params[1], self.params[2], BALL_RADIUS])

    @property
    def impact(self):
        return self.pos(self.t_last)[0]

    @property
    def release_speed_kmh(self):
        return float(np.linalg.norm(model_vel(self.params[1:], self.tb, [self.t_first])[0]) * 3.6)

    def stumps(self, params=None):
        """(x, z) where the post-bounce arc crosses the stump plane y = 0."""
        p = self.params if params is None else params
        tb, (B, _, _, v2, a2) = p[0], _split(p[1:])
        a, v, y0 = 0.5 * a2[1], v2[1], B[1]
        if abs(a) < 1e-9:
            dt = -y0 / v
        else:
            disc = v * v - 4 * a * y0
            if disc < 0:
                return np.array([np.nan, np.nan])
            r = np.array([(-v - np.sqrt(disc)) / (2 * a), (-v + np.sqrt(disc)) / (2 * a)])
            r = r[r > 0]
            if not len(r):
                return np.array([np.nan, np.nan])
            dt = r.min()
        P = B + v2 * dt + 0.5 * a2 * dt ** 2
        return P[[0, 2]]

    def sample_params(self, n, rng):
        return rng.multivariate_normal(self.params, self.cov, size=n, check_valid="ignore")

    def report(self, handed="right", n_mc=400, seed=0):
        rng = np.random.default_rng(seed)
        st = self.stumps()
        samples = self.sample_params(n_mc, rng)
        sxz = np.array([self.stumps(s) for s in samples])
        good = np.all(np.isfinite(sxz), axis=1)
        calls = [lbw.wickets(*xz) for xz in sxz[good]]
        probs = {k: calls.count(k) / max(len(calls), 1)
                 for k in ("hitting", "umpires call", "missing")}
        v = lbw.verdict(self.bounce[0], self.impact[0], st, handed)
        problems = self.problems()
        sig = np.nanstd(sxz, axis=0)
        # A call is only as good as the projection. If the ball could be more
        # than ~10 cm (2 sigma) either way at the stumps -- too few frames after
        # the bounce, occlusion, poor geometry -- refuse to call wickets.
        if not np.all(np.isfinite(sig)) or sig.max() > MAX_STUMPS_SIGMA:
            v["wickets"] = "inconclusive"
            problems.append(f"projection too uncertain (+/-{2 * np.nanmax(sig) * 100:.0f} cm)")
        return dict(
            status="ok" if not problems else "NO DECISION: " + "; ".join(problems),
            bounce_xy_m=self.bounce[:2].round(3).tolist(),
            bounce_sigma_m=np.sqrt(np.diag(self.cov)[1:3]).round(3).tolist(),
            impact_xyz_m=self.impact.round(3).tolist(),
            stumps_xz_m=st.round(3).tolist(),
            stumps_sigma_m=sig.round(3).tolist(),
            release_speed_kmh=round(self.release_speed_kmh, 1),
            pitching=v["pitching"], impact=v["impact"], wickets=v["wickets"],
            wickets_confidence={k: round(p, 3) for k, p in probs.items()},
            rms_px=round(self.rms_px, 2), n_obs=self.n_obs, n_inliers=self.n_inliers,
            n_after_bounce=self.n_after_bounce,
        )


# ---------------------------------------------------------------- fitting
def _robust_quadfit(t, y, iters=3):
    keep = np.ones(len(t), bool)
    for _ in range(iters):
        c = np.polyfit(t[keep], y[keep], 2)
        r = y - np.polyval(c, t)
        mad = np.median(np.abs(r[keep])) + 1e-6
        keep = np.abs(r) < 6 * 1.4826 * mad + 1.0
    return c, keep


def guess_bounce_time(t, uv, min_side=4):
    """Bounce = the kink in a 2D image track: the split that best explains
    the track as two separate parabolas."""
    order = np.argsort(t)
    t, uv = t[order], uv[order]
    best, tb = np.inf, None
    for k in range(min_side, len(t) - min_side):
        sse = 0.0
        for seg in (slice(0, k), slice(k, None)):
            for d in range(2):
                c, keep = _robust_quadfit(t[seg], uv[seg, d], iters=2)
                sse += np.sum(np.minimum((uv[seg, d] - np.polyval(c, t[seg])) ** 2, 100.0))
        if sse < best:
            best, tb = sse, 0.5 * (t[k - 1] + t[k])
    return tb


def _residuals(p14, tb, groups, cams, px_sigma):
    res = []
    for ci, (t, uv) in groups.items():
        pred = cams[ci].project(model_pos(p14, tb, t))
        r = (pred - uv) / px_sigma
        res.append(np.nan_to_num(r, nan=1e3).ravel())
    res.append(_priors(p14))
    return np.concatenate([np.ravel(r) for r in res])


def fit(obs, cams, px_sigma=1.0, tb_window=0.024, tb_step=0.004, loss_scale=3.0):
    """obs: iterable of (cam_index, t, (u, v)). Returns FitResult."""
    cam_idx = np.array([o[0] for o in obs])
    t_all = np.array([o[1] for o in obs], float)
    uv_all = np.array([o[2] for o in obs], float)
    groups = {ci: (t_all[cam_idx == ci], uv_all[cam_idx == ci]) for ci in np.unique(cam_idx)}

    # 1. bounce-time guess from the best-covered camera's 2D track
    main = max(groups, key=lambda c: len(groups[c][0]))
    tm, uvm = groups[main]
    tb0 = guess_bounce_time(tm, uvm)

    def init_at(tb, speed):
        i = np.argsort(np.abs(tm - tb))[:2]
        uv_b = uvm[i].mean(0)
        B = cams[main].ray_plane(uv_b[None], z=BALL_RADIUS)[0]
        return np.r_[B[:2], 0, -speed, -0.12 * speed, 0, 0.004 * speed ** 2, -G,
                     0, -0.88 * speed, 0.05 * speed, 0, 0.004 * speed ** 2, -G]

    def solve(tb, x0):
        r = least_squares(_residuals, x0, args=(tb, groups, cams, px_sigma),
                          loss="soft_l1", f_scale=loss_scale, x_scale="jac", max_nfev=300)
        return r

    # 2. multi-start at the guessed bounce time (fast bowling vs spin)
    best = min((solve(tb0, init_at(tb0, s)) for s in (21.0, 29.0, 37.0)), key=lambda r: r.cost)
    # 3. scan bounce time, warm-starting from the best solution so far
    best_tb = tb0
    for tb in np.arange(tb0 - tb_window, tb0 + tb_window + 1e-9, tb_step):
        r = solve(tb, best.x)
        if r.cost < best.cost:
            best, best_tb = r, tb

    # 4. final joint refinement with tb free
    def res_full(p15):
        return _residuals(p15[1:], p15[0], groups, cams, px_sigma)
    final = least_squares(res_full, np.r_[best_tb, best.x], loss="soft_l1",
                          f_scale=loss_scale, x_scale="jac", max_nfev=500)

    J = final.jac
    n, k = J.shape
    s2 = max(1.0, 2 * final.cost / max(n - k, 1))
    cov = np.linalg.pinv(J.T @ J) * s2
    raw = final.fun[:-N_PRIORS] * px_sigma
    pairs = raw.reshape(-1, 2)
    err = np.hypot(pairs[:, 0], pairs[:, 1])
    inl = err < loss_scale * px_sigma * 3
    order = np.concatenate([groups[c][0] for c in groups])   # same order as residuals
    return FitResult(final.x, cov, float(np.sqrt(np.mean(err[inl] ** 2))), len(t_all),
                     int(inl.sum()), float(t_all.min()), float(t_all.max()), order[inl])
