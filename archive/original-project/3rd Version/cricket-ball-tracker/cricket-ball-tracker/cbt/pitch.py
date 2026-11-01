"""Pitch geometry and world coordinate frame.

World frame (metres):
  origin  = ground point under the middle stump at the STRIKER's end
  +y      = along the pitch towards the bowler's stumps (bowler stumps at y = 20.12)
  +x      = OFF side for a right-handed batter (flip with `handed`)
  +z      = up
A delivery therefore travels towards -y.
"""
import numpy as np

PITCH_LENGTH = 20.12          # stump to stump
CREASE_OFFSET = 1.22          # popping crease in front of stumps
RETURN_CREASE_HALF = 1.32     # return creases are 2.64 m apart
PITCH_HALF_WIDTH = 1.525      # pitch is 3.05 m wide
STUMP_HALF_WIDTH = 0.1143     # outer edges of the three stumps: 22.86 cm
STUMP_HEIGHT = 0.711          # 71.1 cm, top of stumps / bottom of bails
STUMP_RADIUS = 0.018
BALL_RADIUS = 0.036           # men's ball, ~22.4-22.9 cm circumference
G = 9.81


def landmarks():
    """Named 3D points a calibrator can click in an image (both ends).

    Crease markings are painted on the ground (z=0); stump tops give the
    out-of-plane points that make pose estimation well-conditioned.
    """
    pts = {}
    for end, ys in (("striker", 0.0), ("bowler", PITCH_LENGTH)):
        sgn = 1.0 if end == "striker" else -1.0
        pop = ys + sgn * CREASE_OFFSET
        for side, xs in (("off", RETURN_CREASE_HALF), ("leg", -RETURN_CREASE_HALF)):
            pts[f"{end}_bowling_crease_{side}"] = (xs, ys, 0.0)
            pts[f"{end}_popping_crease_{side}"] = (xs, pop, 0.0)
        for name, xs in (("off", STUMP_HALF_WIDTH - STUMP_RADIUS),
                         ("middle", 0.0),
                         ("leg", -(STUMP_HALF_WIDTH - STUMP_RADIUS))):
            pts[f"{end}_{name}_stump_base"] = (xs, ys, 0.0)
            pts[f"{end}_{name}_stump_top"] = (xs, ys, STUMP_HEIGHT)
    return {k: np.array(v, float) for k, v in pts.items()}
