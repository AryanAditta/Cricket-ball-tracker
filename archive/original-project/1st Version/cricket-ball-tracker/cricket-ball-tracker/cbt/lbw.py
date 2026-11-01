"""LBW components in the style of a DRS ball-tracking graphic.

Zone sizes follow the ICC playing conditions as we understand them
(wicket zone = outer edges of the stumps up to the top of the bails;
"umpire's call" when less than half the ball is inside the zone).
CHECK THESE AGAINST THE CURRENT ICC / BCB PLAYING CONDITIONS before use.
"""
import numpy as np
from .pitch import STUMP_HALF_WIDTH, STUMP_HEIGHT, BALL_RADIUS

ZONE_HALF_WIDTH = STUMP_HALF_WIDTH
ZONE_TOP = STUMP_HEIGHT + 0.0127      # top of the bails


def _line(x, handed):
    """'in line' / 'outside off' / 'outside leg' for a ball centre at lateral x.
    Outside means the whole ball is beyond the line of the stumps."""
    x = x if handed == "right" else -x          # +x is off side for a right-hander
    if x > ZONE_HALF_WIDTH + BALL_RADIUS:
        return "outside off"
    if x < -(ZONE_HALF_WIDTH + BALL_RADIUS):
        return "outside leg"
    return "in line"


def wickets(x, z):
    """'hitting' (>= half the ball in the zone), 'umpires call' (clipping),
    'missing'. x, z = projected ball centre at the stump plane."""
    dx = abs(x) - ZONE_HALF_WIDTH
    in_x, in_z = dx <= 0, 0 <= z <= ZONE_TOP
    if in_x and in_z:
        return "hitting"
    # distance from centre to the zone rectangle
    ddx = max(dx, 0.0)
    ddz = max(z - ZONE_TOP, -z, 0.0)
    if np.hypot(ddx, ddz) < BALL_RADIUS:
        return "umpires call"
    return "missing"


def verdict(pitch_x, impact_x, stumps_xz, handed="right"):
    return dict(pitching=_line(pitch_x, handed),
                impact=_line(impact_x, handed),
                wickets=wickets(*stumps_xz))
