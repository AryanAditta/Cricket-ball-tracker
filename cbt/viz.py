"""DRS-style figure: pitch map (top), side elevation, and view at the stumps."""
import numpy as np


def speed_text(rep):
    if rep.get('release_speed_kmh') is not None:
        return f"release speed {rep['release_speed_kmh']:.0f} km/h"
    if rep.get('speed_estimate_kmh') is not None:
        return f"speed ≈{rep['speed_estimate_kmh']:.0f} km/h (estimate)"
    return "speed —"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse

from .pitch import (PITCH_HALF_WIDTH, CREASE_OFFSET, STUMP_HALF_WIDTH, BALL_RADIUS,
                    RETURN_CREASE_HALF)
from .lbw import ZONE_TOP

INK, MUTED, PITCH, TRACK, PROJ = "#1f2933", "#7b8794", "#e9dcc0", "#c0392b", "#2c6fbb"


def figure(fit, truth=None, path="delivery.png", title=None, rep=None):
    rep = rep or fit.report()
    t = np.linspace(fit.t_first, fit.t_last, 200)
    P = fit.pos(t)
    # projected path: from impact to the stump plane
    st = fit.stumps()
    tp = np.linspace(fit.t_last, fit.t_last + 0.25, 200)
    Q = fit.pos(tp)
    Q = Q[Q[:, 1] >= 0]

    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 2.6, 1.3], height_ratios=[1, 1])
    ax_top = fig.add_subplot(gs[:, 0])
    ax_side = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[1, 1]); ax_txt.axis("off")
    ax_st = fig.add_subplot(gs[:, 2])

    # --- pitch map (top view), batter at the bottom
    ax_top.add_patch(Rectangle((-PITCH_HALF_WIDTH, -1), 2 * PITCH_HALF_WIDTH, 22.5, color=PITCH))
    for y in (0, CREASE_OFFSET, 20.12 - CREASE_OFFSET, 20.12):
        ax_top.plot([-RETURN_CREASE_HALF - .3, RETURN_CREASE_HALF + .3], [y, y], color="white", lw=1.5)
    for x in (-RETURN_CREASE_HALF, RETURN_CREASE_HALF):
        ax_top.plot([x, x], [-0.5, CREASE_OFFSET], color="white", lw=1.5)
        ax_top.plot([x, x], [20.12 - CREASE_OFFSET, 20.62], color="white", lw=1.5)
    ax_top.plot([-STUMP_HALF_WIDTH, STUMP_HALF_WIDTH], [0, 0], color=INK, lw=3)
    ax_top.plot(P[:, 0], P[:, 1], color=TRACK, lw=2)
    ax_top.plot(Q[:, 0], Q[:, 1], color=PROJ, lw=2, ls="--")
    b = fit.bounce
    ax_top.add_patch(Circle(b[:2], BALL_RADIUS * 2.5, color=TRACK, zorder=5))
    if truth is not None:
        ax_top.plot(*truth.bounce[:2], "x", color=INK, ms=8, mew=2, zorder=6)
    ax_top.set_xlim(-PITCH_HALF_WIDTH - .2, PITCH_HALF_WIDTH + .2)
    ax_top.set_ylim(-1, 21)
    ax_top.set_aspect("equal")
    ax_top.set_title("Pitch map", color=INK)
    ax_top.set_xlabel("x (m, + = off side)")
    ax_top.set_ylabel("distance from striker's stumps (m)")

    # --- side elevation
    ax_side.plot(P[:, 1], P[:, 2], color=TRACK, lw=2, label="tracked")
    ax_side.plot(Q[:, 1], Q[:, 2], color=PROJ, lw=2, ls="--", label="projected")
    if truth is not None:
        m = truth.pos[:, 1] >= 0
        ax_side.plot(truth.pos[m, 1], truth.pos[m, 2], color=MUTED, lw=1, label="ground truth")
    ax_side.add_patch(Rectangle((-0.02, 0), 0.04, ZONE_TOP, color=INK))
    ax_side.axhline(0, color=MUTED, lw=1)
    ax_side.set_xlim(21, -0.5)
    ax_side.set_ylim(-0.1, 2.6)
    ax_side.set_xlabel("distance from striker's stumps (m)")
    ax_side.set_ylabel("height (m)")
    ax_side.set_title("Side view", color=INK)
    ax_side.legend(frameon=False, loc="upper right")

    # --- at the stumps (batter's-eye view from the bowler's end)
    ax_st.add_patch(Rectangle((-STUMP_HALF_WIDTH, 0), 2 * STUMP_HALF_WIDTH, ZONE_TOP,
                              fill=False, ec=INK, lw=1.5, ls=":"))
    for x in (-STUMP_HALF_WIDTH + 0.018, 0, STUMP_HALF_WIDTH - 0.018):
        ax_st.add_patch(Rectangle((x - 0.018, 0), 0.036, 0.711, color=INK))
    sx, sz = rep["stumps_sigma_m"]
    if np.all(np.isfinite(st)):
        ax_st.add_patch(Circle(st, BALL_RADIUS, color=PROJ, alpha=0.85, zorder=5))
        if np.isfinite(sx) and np.isfinite(sz):
            ax_st.add_patch(Ellipse(st, 4 * sx, 4 * sz, fill=False, ec=PROJ, lw=1, ls="--"))
    if truth is not None:
        ax_st.plot(*truth.stumps, "x", color=INK, ms=8, mew=2, zorder=6)
    ax_st.set_xlim(-0.6, 0.6)
    ax_st.set_ylim(-0.05, 1.1)
    ax_st.set_aspect("equal")
    ax_st.axhline(0, color=MUTED, lw=1)
    ax_st.set_title("At the stumps (bowler's view)", color=INK)
    ax_st.set_xlabel("x (m)")

    conf = rep["wickets_confidence"]
    lines = [
        f"PITCHING   {rep['pitching'].upper()}   ({rep['bounce_xy_m'][1]:.2f} m from stumps)",
        f"IMPACT     {rep['impact'].upper()}",
        f"WICKETS    {rep['wickets'].upper()}",
        f"DECISION   {rep['decision'].upper()}{'  (UMPIRE' + chr(39) + 'S CALL)' if rep.get('umpires_call') else ''}"
        f"  -- {rep.get('reason', '')}   [on-field: {rep.get('on_field', 'not out')}]",
        f"impact {rep.get('impact_to_stumps_m', float('nan')):.2f} m from stumps · pitch to impact "
        f"{rep.get('pitch_to_impact_m', float('nan')):.2f} m"
        + (f" · {rep['distance_rule']} rule applies" if rep.get('distance_rule') else ""),
        "",
        f"confidence: hitting {conf['hitting']:.0%} · umpire's call {conf['umpires call']:.0%}"
        f" · missing {conf['missing']:.0%}",
        f"{speed_text(rep)} · "
        f"stumps ±{2 * sx * 100:.1f} cm (x), ±{2 * sz * 100:.1f} cm (z) at 2σ",
        f"{rep['n_inliers']}/{rep['n_obs']} detections used · reprojection rms {rep['rms_px']} px",
    ]
    if rep.get("status", "ok") != "ok":
        lines.insert(3, rep["status"])
    ax_txt.text(0.0, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=10.5, color=INK)
    if title:
        fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
