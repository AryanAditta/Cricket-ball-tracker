"""Link per-frame candidates into ball tracks (one per delivery) in 2D."""
import numpy as np


class _Track:
    def __init__(self, row):
        self.rows = [row]
        self.missed = 0

    def predict(self, frame):
        r = np.array(self.rows)
        if len(r) == 1:
            return r[-1, 2:4], None
        # constant velocity in pixels/frame from the last two points
        (f1, _, u1, v1, _), (f2, _, u2, v2, _) = r[-2], r[-1]
        vel = np.array([u2 - u1, v2 - v1]) / max(f2 - f1, 1)
        return r[-1, 2:4] + vel * (frame - f2), np.linalg.norm(vel)


def link(cands, gate_px=40.0, max_gap=3, max_jump_px=250.0):
    """cands: rows (frame, t, u, v, score) -> list of tracks (arrays of rows)."""
    cands = cands[np.argsort(cands[:, 0])]
    active, done = [], []
    for f in np.unique(cands[:, 0]):
        rows = cands[cands[:, 0] == f]
        used = np.zeros(len(rows), bool)
        # longest tracks pick first
        for tr in sorted(active, key=lambda tr: -len(tr.rows)):
            pred, speed = tr.predict(f)
            gate = max_jump_px if speed is None else gate_px + 0.5 * speed
            d = np.linalg.norm(rows[:, 2:4] - pred, axis=1)
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] < gate:
                tr.rows.append(rows[j]); tr.missed = 0; used[j] = True
            else:
                tr.missed += 1
        for j in np.where(~used)[0]:
            active.append(_Track(rows[j]))
        still = []
        for tr in active:
            (done if tr.missed > max_gap else still).append(tr)
        active = still
    return [np.array(tr.rows) for tr in done + active]


def _two_parabola_inliers(tr, tol_px=4.0):
    """Best split of a track into two parabolas (before/after bounce), fitted
    robustly. Returns (inlier mask, inlier rms). Junk picked up at either end
    of a track (the bowler before release, the pad after impact) falls out."""
    t = tr[:, 1] - tr[0, 1]
    best = (np.inf, None)
    for k in range(4, len(tr) - 4):
        keep = np.ones(len(tr), bool)
        res = np.zeros(len(tr))
        for seg in (np.arange(k), np.arange(k, len(tr))):
            m = np.ones(len(seg), bool)
            for _ in range(3):
                r = np.zeros(len(seg))
                for d in (2, 3):
                    c = np.polyfit(t[seg][m], tr[seg, d][m], 2)
                    r += (tr[seg, d] - np.polyval(c, t[seg])) ** 2
                r = np.sqrt(r)
                thr = max(tol_px, 3 * 1.4826 * np.median(r[m]))
                m_new = r < thr
                if m_new.sum() < 4 or np.array_equal(m_new, m):
                    break
                m = m_new
            keep[seg] = m
            res[seg] = r
        score = np.sum(np.minimum(res, 3 * tol_px) ** 2) / len(tr)
        if score < best[0]:
            best = (score, keep)
    if best[1] is None:
        return np.ones(len(tr), bool), np.inf
    keep = best[1]
    return keep, float(np.sqrt(best[0]))


def _drop_static_and_weak(tr, static_px=0.6, rel_score=0.3):
    """A ball in flight never stops: drop points that barely move from their
    neighbour (residual blobs left after impact), and points whose detection
    score is far below the track's typical score."""
    step = np.linalg.norm(np.diff(tr[:, 2:4], axis=0), axis=1) / np.maximum(np.diff(tr[:, 0]), 1)
    slow = step < static_px
    # point i is "still" if it barely moved from the previous point AND
    # (it barely moves to the next point, or it is the last point)
    still = np.r_[False, slow] & np.r_[slow, True]
    weak = tr[:, 4] < rel_score * np.median(tr[:, 4])
    return tr[~(still | weak)]


def ball_tracks(cands, min_len=8, min_speed_px=1.0, min_span_px=30.0, max_quad_rms=6.0):
    """Keep tracks that look like a delivery: long enough, moving, and smooth
    (two parabolas joined at the bounce). Outlying end points are trimmed.
    Sorted by start time."""
    out = []
    for tr in link(cands):
        if len(tr) < min_len:
            continue
        tr = _drop_static_and_weak(tr)
        if len(tr) < min_len:
            continue
        keep, rms = _two_parabola_inliers(tr)
        tr = tr[keep]
        if len(tr) < min_len or rms > max_quad_rms:
            continue
        steps = np.linalg.norm(np.diff(tr[:, 2:4], axis=0), axis=1) / np.maximum(np.diff(tr[:, 0]), 1)
        span = np.linalg.norm(tr[:, 2:4].max(0) - tr[:, 2:4].min(0))
        # end-on cameras see the ball move only ~1-3 px/frame, so be lenient
        if np.median(steps) < min_speed_px or span < min_span_px:
            continue
        out.append(tr)
    return sorted(out, key=lambda tr: tr[0, 1])


def delivery_track(cands, max_gap_frames=20, min_rel_score=0.5, **kw):
    """The ball's track for one delivery.

    Occlusion (bowler at release, batter near impact) can split the ball into
    several track pieces. Take the strongest piece (length x mean score), then
    add any other accepted pieces that do not overlap it in time. Pieces that
    DO overlap it (e.g. the ball's shadow) are rejected.
    """
    tracks = ball_tracks(cands, **kw)
    if not tracks:
        return None
    tracks.sort(key=lambda tr: -len(tr) * np.mean(tr[:, 4]))
    chosen = [tracks[0]]
    ref_score = np.mean(tracks[0][:, 4])
    for tr in tracks[1:]:
        disjoint = all(tr[0, 0] > c[-1, 0] or tr[-1, 0] < c[0, 0] for c in chosen)
        # must look as ball-like as the main piece and sit close to it in time
        gap = min(min(abs(tr[0, 0] - c[-1, 0]), abs(c[0, 0] - tr[-1, 0])) for c in chosen)
        if disjoint and gap <= max_gap_frames and np.mean(tr[:, 4]) >= min_rel_score * ref_score:
            chosen.append(tr)
    out = np.concatenate(chosen)
    return out[np.argsort(out[:, 0])]
