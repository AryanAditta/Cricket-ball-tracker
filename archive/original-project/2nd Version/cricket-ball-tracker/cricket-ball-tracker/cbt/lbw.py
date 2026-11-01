"""LBW decision protocol for ball-tracking reviews.

Rules implemented (as published for ICC DRS; see README for sources):

  PITCHING  "outside leg" when MORE than 50% of the ball pitched outside the
            line of leg stump (ball centre beyond the line). No umpire's call.
            Exactly 50% favours the bowler.
  IMPACT    "in line"      >= 50% of the ball inside the line of the stumps
            "umpires call" 1-50% of the ball inside the line
            "outside off/leg" no part of the ball inside the line
  WICKETS   zone = outer edges of the stumps, ground to TOP OF THE BAILS
            (since April 2021).
            "hitting"      >= 50% of the ball inside the zone (centre inside)
            "umpires call" some of the ball inside, centre outside
            "missing"      no part of the ball inside
  DISTANCE  (protects the on-field decision on a projection that is hitting)
            impact >= 3.00 m from the stumps                  -> umpire's call
            2.50 m < impact < 3.00 m and pitch->impact < 0.40 m -> umpire's call
  DECISION  pitched outside leg -> NOT OUT; impact outside leg -> NOT OUT;
            impact outside off -> NOT OUT if a shot was offered (else treated
            as in line); wickets missing -> NOT OUT; any umpire's call ->
            on-field decision stands; otherwise hitting -> OUT.

CHECK THESE AGAINST THE PLAYING CONDITIONS BCB USES before match use: they
are set per competition and change over time.
"""
import numpy as np
from .pitch import STUMP_HALF_WIDTH, STUMP_HEIGHT, BALL_RADIUS

ZONE_HALF_WIDTH = STUMP_HALF_WIDTH
ZONE_TOP = STUMP_HEIGHT + 0.0127      # top of the bails
R = BALL_RADIUS
DIST_FAR = 3.00                       # m, impact to stumps
DIST_MID = 2.50
PITCH_TO_IMPACT_MIN = 0.40


def _off(x, handed):
    return x if handed == "right" else -x        # +x is off side for a right-hander


def pitching(x, handed="right"):
    x = _off(x, handed)
    if x > ZONE_HALF_WIDTH:
        return "outside off"
    if x < -ZONE_HALF_WIDTH:
        return "outside leg"
    return "in line"


def impact(x, handed="right"):
    x = _off(x, handed)
    if abs(x) <= ZONE_HALF_WIDTH:
        return "in line"
    if abs(x) < ZONE_HALF_WIDTH + R:
        return "umpires call"
    return "outside off" if x > 0 else "outside leg"


def wickets(x, z):
    """x, z = projected ball centre at the stump plane."""
    dx = abs(x) - ZONE_HALF_WIDTH
    if dx <= 0 and 0 <= z <= ZONE_TOP:
        return "hitting"
    ddx = max(dx, 0.0)
    ddz = max(z - ZONE_TOP, -z, 0.0)
    if np.hypot(ddx, ddz) < R:
        return "umpires call"
    return "missing"


def distance_rule(impact_to_stumps, pitch_to_impact):
    """Name of the distance rule that applies ("3m", "2.5m/40cm") or None."""
    if impact_to_stumps >= DIST_FAR:
        return "3m"
    if impact_to_stumps > DIST_MID and pitch_to_impact < PITCH_TO_IMPACT_MIN:
        return "2.5m/40cm"
    return None


def verdict(pitch_x, impact_x, stumps_xz, handed="right"):
    return dict(pitching=pitching(pitch_x, handed),
                impact=impact(impact_x, handed),
                wickets=wickets(*stumps_xz))


def decide(pitching, impact, wickets, on_field="not out", shot_offered=True, dist_rule=None):
    """Final decision. Returns dict(decision, umpires_call, reason).

    decision is "out" / "not out"; umpires_call is True when the on-field
    decision stood because of a marginal component."""
    def res(d, uc, why):
        return dict(decision=d, umpires_call=uc, reason=why)
    stands = on_field
    if pitching == "outside leg":
        return res("not out", False, "pitched outside leg")
    if impact == "outside leg":
        return res("not out", False, "impact outside leg")
    if impact == "outside off" and shot_offered:
        return res("not out", False, "impact outside off, shot offered")
    if wickets == "missing":
        return res("not out", False, "missing the stumps")
    if wickets == "inconclusive":
        return res(stands, True, "insufficient tracking data, on-field decision stands")
    if impact == "umpires call":
        return res(stands, True, "umpire's call on impact")
    if wickets == "umpires call":
        return res(stands, True, "umpire's call on wickets")
    if dist_rule:
        return res(stands, True, f"umpire's call ({dist_rule} rule)")
    return res("out", False, "LBW")


def review_outcome(on_field, final, umpires_call, reviewed_by):
    """reviewed_by: "batting" or "fielding". Whether the review was successful
    or lost. On umpire's call the review is retained (current ICC practice)."""
    if umpires_call:
        return "umpire's call (review retained)"
    wanted = "not out" if reviewed_by == "batting" else "out"
    return "successful" if final == wanted else "unsuccessful (review lost)"
