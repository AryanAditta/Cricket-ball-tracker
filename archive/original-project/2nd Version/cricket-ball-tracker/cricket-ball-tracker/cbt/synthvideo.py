"""Render synthetic match video from a simulated delivery.

Used to test video -> detection -> tracking -> 3D fit end to end while no
real footage from our own cameras exists.

`hard=True` adds the things that break trackers on real footage:
  * a bowler running through the end-on view (hides the ball at release)
  * a batter who strides forward, with pads at the impact point
  * the ball's shadow on the pitch (a fast, smooth, ball-sized distractor)
  * camera shake, light flicker, longer shutter blur, JPEG-style artefacts
"""
import cv2
import numpy as np
from .pitch import PITCH_HALF_WIDTH, PITCH_LENGTH, BALL_RADIUS

BALL_BGR = (30, 30, 170)            # red ball
WHITES = (232, 236, 238)
SUN = np.array([0.35, 0.25, -1.0])  # direction of sunlight (for shadows)


def _background(cam, rng):
    w, h = cam.size
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (60, 120, 50)                                     # outfield (BGR)
    corners = np.array([[-PITCH_HALF_WIDTH, -2, 0], [PITCH_HALF_WIDTH, -2, 0],
                        [PITCH_HALF_WIDTH, PITCH_LENGTH + 2, 0],
                        [-PITCH_HALF_WIDTH, PITCH_LENGTH + 2, 0]])
    poly = cam.project(corners)
    if np.all(np.isfinite(poly)):
        cv2.fillPoly(img, [poly.astype(np.int32)], (150, 190, 205))
    # crease lines
    for y in (0.0, 1.22, PITCH_LENGTH - 1.22, PITCH_LENGTH):
        seg = cam.project(np.array([[-1.5, y, 0], [1.5, y, 0]]))
        if np.all(np.isfinite(seg)):
            cv2.line(img, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (245, 245, 245), 2)
    tex = rng.normal(0, 6, (h, w, 1))
    img = np.clip(img + tex, 0, 255).astype(np.uint8)
    # stumps at both ends (static, so they become part of the learned background)
    f = cam.K[0, 0]
    for y in (0.0, PITCH_LENGTH):
        for x in (-0.0963, 0.0, 0.0963):
            seg = cam.project(np.array([[x, y, 0.0], [x, y, 0.711]]))
            if np.all(np.isfinite(seg)):
                depth = (cam.R @ np.array([x, y, 0.35]) + cam.t)[2]
                wpx = max(1, int(round(f * 0.036 / depth)))
                cv2.line(img, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)),
                         (205, 225, 235), wpx, cv2.LINE_AA)
    return img


def _box(cam, cx, cy, sx, sy, sz, z0=0.0):
    """Projected silhouette (convex hull) of an axis-aligned box, or None."""
    xs, ys, zs = (cx - sx / 2, cx + sx / 2), (cy - sy / 2, cy + sy / 2), (z0, z0 + sz)
    P = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    uv = cam.project(P)
    if not np.all(np.isfinite(uv)):
        return None
    return cv2.convexHull(uv.astype(np.int32)), float((cam.R @ np.array([cx, cy, z0 + sz / 2]) + cam.t)[2])


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def render(delivery, cam, path, rng, t_start=None, t_pad=0.15, shutter=None,
           noise=3.0, distractor=False, hard=False):
    """Write an mp4; returns the video's t=0 on the master clock."""
    if t_start is None:
        t_start = -0.5 if hard else -0.15
    if shutter is None:
        shutter = 1 / 500 if hard else 1 / 1000
    w, h = cam.size
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), cam.fps, (w, h))
    bg = _background(cam, rng)
    n = int((delivery.t_impact + t_pad - t_start) * cam.fps)
    f = cam.K[0, 0]
    rel = delivery.pos[0]
    imp = delivery.impact
    ti = delivery.t_impact
    shake_phase = rng.uniform(0, 2 * np.pi, 2)
    for k in range(n):
        t = t_start + k / cam.fps
        img = bg.copy()
        layers = []          # (depth, draw_fn)

        if distractor:   # a slow, large "fielder" drifting across the frame
            cx = int(w * 0.15 + 40 * k / cam.fps * 10) % w
            cv2.ellipse(img, (cx, int(h * 0.75)), (14, 30), 0, 0, 360, WHITES, -1)

        ball_visible = 0 <= t <= ti
        if ball_visible:
            P0, P1 = delivery.at(np.array([t, t + shutter]))
            if hard:      # shadow on the pitch, drawn first (it is on the ground)
                S = np.stack([P0 - P0[2] / SUN[2] * SUN, P1 - P1[2] / SUN[2] * SUN])
                S[:, 2] = 0
                suv = cam.project(S)
                if np.all(np.isfinite(suv)):
                    sd = (cam.R @ S[0] + cam.t)[2]
                    rs = max(1, int(round(f * BALL_RADIUS * 1.1 / sd)))
                    m = np.zeros((h, w), np.uint8)
                    cv2.line(m, tuple(suv[0].astype(int)), tuple(suv[1].astype(int)), 255, 2 * rs)
                    img[m > 0] = (img[m > 0] * 0.55).astype(np.uint8)
            uv = cam.project(np.stack([P0, P1]))
            if np.all(np.isfinite(uv)):
                depth = (cam.R @ P0 + cam.t)[2]
                r = max(1, int(round(f * BALL_RADIUS / depth)))
                a, b = (tuple(p) for p in uv.astype(int))

                def draw_ball(img=img, a=a, b=b, r=r):
                    cv2.line(img, a, b, BALL_BGR, 2 * r, cv2.LINE_AA)
                    cv2.circle(img, b, r, BALL_BGR, -1, cv2.LINE_AA)
                layers.append((depth, draw_ball))

        if hard:
            # bowler: runs in at 6 m/s, releases at t=0, follows through
            yb = rel[1] + 0.25 - 6.0 * t + (3.5 * t * t if t > 0 else 0.0)
            box = _box(cam, rel[0] + 0.22, yb, 0.45, 0.3, 1.85)
            if box:
                layers.append((box[1], lambda img=img, hull=box[0]:
                               cv2.fillConvexPoly(img, hull, WHITES)))
            # batter: front-foot stride in the last 0.35 s; pad ends at impact point
            stride = 0.35 * _smooth((t - (ti - 0.35)) / 0.3)
            pad_y = imp[1] - 0.35 + stride
            for (cx, cy, sx, sy, sz) in ((-0.32, pad_y - 0.15, 0.35, 0.5, 1.8),       # body
                                         (imp[0], pad_y + 0.06, 0.16, 0.12, 0.55)):  # front pad
                box = _box(cam, cx, cy, sx, sy, sz)
                if box:
                    layers.append((box[1], lambda img=img, hull=box[0]:
                                   cv2.fillConvexPoly(img, hull, WHITES)))

        for _, draw in sorted(layers, key=lambda L: -L[0]):     # far to near
            draw()

        if hard:
            g = 1 + rng.normal(0, 0.015)                         # light flicker
            img = np.clip(img.astype(np.float32) * g, 0, 255)
            dx = 0.8 * np.sin(2 * np.pi * 3 * t + shake_phase[0]) + rng.normal(0, 0.3)
            dy = 0.6 * np.sin(2 * np.pi * 2 * t + shake_phase[1]) + rng.normal(0, 0.3)
            img = cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                                 borderMode=cv2.BORDER_REPLICATE)
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
        if hard:
            img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1], 1)
        vw.write(img)
    vw.release()
    return t_start
