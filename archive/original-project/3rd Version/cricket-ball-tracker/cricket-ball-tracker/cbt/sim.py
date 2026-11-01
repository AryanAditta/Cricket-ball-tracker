"""Synthetic deliveries and synthetic camera observations.

Lets us measure accuracy against known ground truth before any hardware is
bought, and choose camera count / frame rate / placement from numbers.
The physics is deliberately richer than the fitter's model (velocity-
dependent drag, swing, seam/spin deviation at the bounce) so the test is honest.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares

from .pitch import BALL_RADIUS, G, landmarks
from .camera import Camera, calibrate_pose

RHO, MASS, AREA = 1.2, 0.16, np.pi * BALL_RADIUS ** 2


@dataclass
class Delivery:
    t: np.ndarray            # dense time samples (s), release at 0
    pos: np.ndarray          # (N,3) true positions
    t_bounce: float
    bounce: np.ndarray       # true bounce point
    t_impact: float          # when the pad is struck (observations stop here)
    impact: np.ndarray
    stumps: np.ndarray       # true (x, z) where the unobstructed ball crosses y=0
    params: dict = field(default_factory=dict)

    def at(self, t):
        return np.stack([np.interp(t, self.t, self.pos[:, i]) for i in range(3)], -1)


def _accel(v, k, swing):
    s = np.linalg.norm(v)
    return np.array([swing, 0.0, -G]) - k * s * v


def _integrate(p0, v0, k, swing, bounce_fn=None, y_stop=-0.5, dt=2e-4, t_max=2.0):
    """RK4 integrate until y < y_stop. bounce_fn(v) -> new v when z hits radius."""
    ts, ps = [0.0], [p0.copy()]
    p, v, t, tb = p0.copy(), v0.copy(), 0.0, None
    while p[1] > y_stop and t < t_max:
        def f(state):
            return np.r_[state[3:], _accel(state[3:], k, swing)]
        s = np.r_[p, v]
        k1 = f(s); k2 = f(s + dt / 2 * k1); k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        p, v, t = s[:3], s[3:], t + dt
        if p[2] <= BALL_RADIUS and v[2] < 0:
            if bounce_fn is None:
                return np.array(ts + [t]), np.array(ps + [p]), t
            # linear back-off to the contact instant
            if tb is None:
                tb = t
                p = p.copy(); p[2] = BALL_RADIUS
                v = bounce_fn(v)
            else:  # second bounce: stop
                break
        ts.append(t); ps.append(p.copy())
    return np.array(ts), np.array(ps), tb


def random_delivery(rng, kind=None, speed_kmh=None, line=None, length=None, swing=None,
                    turn=None):
    """A random delivery. Any of these can be fixed to "bowl" a specific ball:
    speed_kmh at release; line = bounce x (m, + = off side for a right-hander);
    length = bounce distance from the striker's stumps (m); swing = sideways
    acceleration in the air (m/s^2, + = towards off); turn = sideways speed
    gained off the pitch (m/s, + = towards off)."""
    kind = kind or rng.choice(["pace", "spin"], p=[0.65, 0.35])
    pace = kind == "pace"
    speed = rng.uniform(30, 41) if pace else rng.uniform(19, 25)
    if speed_kmh is not None:
        speed = speed_kmh / 3.6
    release = np.array([rng.uniform(-0.6, 0.6), rng.uniform(18.4, 19.0),
                        rng.uniform(1.95, 2.35) if pace else rng.uniform(1.7, 2.1)])
    target = np.array([rng.uniform(-0.45, 0.45) if line is None else line,
                       (rng.uniform(1.8, 9.0) if pace else rng.uniform(2.0, 6.0))
                       if length is None else length])
    k = 0.5 * RHO * rng.uniform(0.35, 0.5) * AREA / MASS
    if swing is None:
        swing = rng.normal(0, 1.5) if pace else rng.normal(0, 0.4)

    # aim: solve for direction so the first bounce lands on target
    def miss(angles):
        az, el = angles
        v0 = speed * np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
        _, ps, _ = _integrate(release, v0, k, swing, None, dt=1e-3)
        return ps[-1, :2] - target
    sol = least_squares(miss, [0.0, -0.05], diff_step=1e-4)
    az, el = sol.x
    v0 = speed * np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])

    e = rng.uniform(0.35, 0.55)                  # vertical restitution
    fric = rng.uniform(0.82, 0.93)               # speed retained along pitch
    if turn is None:
        turn = rng.normal(0, 0.35) if pace else rng.normal(0, 1.3)   # m/s sideways

    def bounce(v):
        return np.array([v[0] * fric + turn, v[1] * fric, -e * v[2]])

    t, pos, tb = _integrate(release, v0, k, swing, bounce)
    if tb is None:
        return None
    bounce_pt = np.array([np.interp(tb, t, pos[:, i]) for i in range(3)])
    y_imp = min(rng.uniform(0.5, 2.0), bounce_pt[1] - 0.6)
    if y_imp < 0.3:
        return None
    d = Delivery(t, pos, tb, bounce_pt, 0, None, None,
                 dict(kind=kind, speed_kmh=speed * 3.6, swing=swing, turn=turn, e=e))
    # impact = first sample past the pad plane; stumps = crossing of y=0
    i = np.searchsorted(-pos[:, 1], -y_imp)
    j = np.searchsorted(-pos[:, 1], 0.0)
    if j >= len(t):
        return None
    d.t_impact, d.impact = t[i], pos[i]
    d.stumps = pos[j, [0, 2]]
    return d


@dataclass
class Obs:
    cam: int
    t: float      # timestamp on the master clock (after sync)
    uv: np.ndarray


def observe(delivery, cams, rng, px_sigma=1.0, p_miss=0.1, p_outlier=0.02,
            t_sigma=0.3e-3):
    """Per-camera frame sampling with random phase, pixel noise, dropped
    frames, false detections and residual clock-sync error."""
    out = []
    for ci, cam in enumerate(cams):
        phase = rng.uniform(0, 1 / cam.fps)
        tk = np.arange(phase, delivery.t_impact, 1 / cam.fps)
        uv = cam.project(delivery.at(tk))
        ok = cam.in_view(uv, margin=5) & (rng.random(len(tk)) > p_miss)
        uv = uv + rng.normal(0, px_sigma, uv.shape)
        bad = rng.random(len(tk)) < p_outlier
        uv[bad] += rng.uniform(-60, 60, (bad.sum(), 2))
        for t, u, k in zip(tk, uv, ok):
            if k:
                out.append(Obs(ci, t + rng.normal(0, t_sigma), u))
    return out


def calibrated_copy(cam, rng, click_sigma=0.7, focal_err=0.003):
    """Re-estimate a camera's pose the way we would on the ground: click the
    visible pitch landmarks (with error) and run PnP with slightly-wrong
    intrinsics. Returns (estimated camera, number of landmarks used)."""
    lm = landmarks()
    W = np.array(list(lm.values()))
    uv = cam.project(W)
    vis = cam.in_view(uv, margin=10)
    if vis.sum() < 6:
        raise RuntimeError(f"{cam.name}: only {vis.sum()} landmarks in view")
    K = cam.K.copy()
    K[0, 0] *= 1 + rng.normal(0, focal_err)
    K[1, 1] = K[0, 0]
    est, _ = calibrate_pose(K, uv[vis] + rng.normal(0, click_sigma, (vis.sum(), 2)), W[vis],
                            size=cam.size, name=cam.name, fps=cam.fps)
    return est, int(vis.sum())
