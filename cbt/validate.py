"""Field validation, modelled on the ICC's independent review of ball tracking:
ACCURACY (does what we show match what happened?) and RELIABILITY (on what
share of deliveries do we produce a usable track at all?).

Three tests, all scored against physical ground truth (see VALIDATION.md):

  A  pitching   tape-measured bounce mark vs our bounce point
  B  wickets    bowling at UNGUARDED stumps: did it hit? vs our wickets call
  C  projection same unguarded deliveries, but we cut each track off 1-2.5 m
                before the stumps (as if a pad were there) and compare the
                projection with the ball's own full, observed path
Plus release speed vs a radar gun when available.
"""
import csv
import json
import os
import numpy as np

from . import fit as fitmod, lbw

TARGETS = dict(            # acceptance targets for a match trial (edit to BCB's spec)
    reliability_pct=97.0,
    pitch_err_p95_cm=5.0,
    wickets_agree_pct=98.0,
    projection_err_p95_cm=5.0,
    speed_err_p95_kmh=3.0,
)


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def _f(row, k):
    v = (row.get(k) or "").strip()
    return float(v) if v else None


def score(truth_rows, reports):
    """truth_rows: dicts from truth.csv; reports: {delivery: report dict}."""
    rows, pitch_err, speed_err, agree, decided, marginal = [], [], [], 0, 0, 0
    tracked = 0
    for t in truth_rows:
        d = t["delivery"]
        rep = reports.get(d)
        row = dict(delivery=d, tracked=0)
        if rep is None:
            row["note"] = "no report (not tracked)"
            rows.append(row)
            continue
        ok = rep.get("status", "ok") == "ok"
        tracked += ok
        row["tracked"] = int(ok)
        row["status"] = rep.get("status", "ok")
        px, py = _f(t, "pitch_x"), _f(t, "pitch_y")
        if px is not None and py is not None:
            bx, by = rep["bounce_xy_m"]
            e = float(np.hypot(bx - px, by - py)) * 100
            row.update(pitch_err_cm=round(e, 2), pitch_err_x_cm=round((bx - px) * 100, 2),
                       pitch_err_y_cm=round((by - py) * 100, 2))
            pitch_err.append(e)
            handed = (t.get("handed") or "right").strip()
            row["pitching_truth"] = lbw.pitching(px, handed)
            row["pitching_ours"] = rep["pitching"]
        spd = _f(t, "speed_kmh")
        if spd is not None:
            se = abs(rep["release_speed_kmh"] - spd)
            row["speed_err_kmh"] = round(se, 2)
            speed_err.append(se)
        hit = (t.get("stumps_result") or "").strip().lower()
        if hit in ("hit", "miss"):
            ours = rep["wickets"]
            row["stumps_truth"], row["wickets_ours"] = hit, ours
            if ours in ("umpires call", "inconclusive"):
                marginal += 1                # not an error, but not a decision either
                row["wickets_ok"] = "marginal"
            else:
                decided += 1
                good = (ours == "hitting") == (hit == "hit")
                agree += good
                row["wickets_ok"] = int(good)
        rows.append(row)
    n = len(truth_rows)
    summary = dict(
        deliveries=n,
        reliability_pct=100 * tracked / n if n else float("nan"),
        pitch_err_median_cm=_pct(pitch_err, 50), pitch_err_p95_cm=_pct(pitch_err, 95),
        pitch_err_max_cm=max(pitch_err) if pitch_err else float("nan"),
        pitching_verdict_agree_pct=100 * np.mean([r["pitching_truth"] == r["pitching_ours"]
                                                  for r in rows if "pitching_truth" in r])
        if any("pitching_truth" in r for r in rows) else float("nan"),
        wickets_decided=decided, wickets_marginal=marginal,
        wickets_agree_pct=100 * agree / decided if decided else float("nan"),
        speed_err_median_kmh=_pct(speed_err, 50), speed_err_p95_kmh=_pct(speed_err, 95),
    )
    return summary, rows


def truncation_test(obs, cams, cut_y=(1.0, 1.5, 2.0, 2.5), px_sigma=1.5):
    """Projection accuracy without a batter. Fit the full track of a ball
    bowled at unguarded stumps (it is observed all the way to the stump
    plane), then refit using only detections before the ball reached y=cut
    and compare where each projection crosses the stumps. Returns list of
    dict(cut_m, err_x_cm, err_z_cm, err_cm) or [] if the full track does not
    reach the stumps."""
    full = fitmod.fit(obs, cams, px_sigma=px_sigma)
    if full.pos([full.t_last])[0][1] > 0.3:
        return []                                   # never got near the stumps
    ref = full.stumps()
    out = []
    t = np.array([o[1] for o in obs])
    for c in cut_y:
        # time at which the full fit says the ball was at y = c
        ts = np.linspace(full.tb, full.t_last, 400)
        ys = full.pos(ts)[:, 1]
        if ys.min() > c or ys.max() < c:
            continue
        t_cut = ts[np.argmin(np.abs(ys - c))]
        sub = [o for o, tt in zip(obs, t) if tt <= t_cut]
        try:
            r = fitmod.fit(sub, cams, px_sigma=px_sigma)
        except Exception:
            continue
        st = r.stumps()
        if not np.all(np.isfinite(st)):
            continue
        ex, ez = (st - ref) * 100
        out.append(dict(cut_m=c, err_x_cm=round(float(ex), 2), err_z_cm=round(float(ez), 2),
                        err_cm=round(float(np.hypot(ex, ez)), 2),
                        same_call=lbw.wickets(*st) == lbw.wickets(*ref)))
    return out


def write_report(path, summary, rows, proj=None, targets=TARGETS, title="Validation report"):
    def chk(val, target, lower_is_better=True):
        if val is None or not np.isfinite(val):
            return "n/a"
        ok = val <= target if lower_is_better else val >= target
        return "PASS" if ok else "FAIL"
    L = [f"# {title}", "",
         f"Deliveries: {summary['deliveries']}", "",
         "| Measure | Result | Target | |", "|---|---|---|---|",
         f"| Reliability (usable track) | {summary['reliability_pct']:.1f}% | ≥ {targets['reliability_pct']}% | "
         f"{chk(summary['reliability_pct'], targets['reliability_pct'], False)} |",
         f"| Pitch point error, median | {summary['pitch_err_median_cm']:.1f} cm | | |",
         f"| Pitch point error, 95th pct | {summary['pitch_err_p95_cm']:.1f} cm | ≤ {targets['pitch_err_p95_cm']} cm | "
         f"{chk(summary['pitch_err_p95_cm'], targets['pitch_err_p95_cm'])} |",
         f"| Pitching verdict agrees | {summary['pitching_verdict_agree_pct']:.1f}% | | |",
         f"| Wickets call agrees (decided balls) | {summary['wickets_agree_pct']:.1f}% "
         f"({summary['wickets_decided']} decided, {summary['wickets_marginal']} umpire's call / inconclusive) | "
         f"≥ {targets['wickets_agree_pct']}% | {chk(summary['wickets_agree_pct'], targets['wickets_agree_pct'], False)} |",
         f"| Release speed error, 95th pct | {summary['speed_err_p95_kmh']:.1f} km/h | ≤ {targets['speed_err_p95_kmh']} km/h | "
         f"{chk(summary['speed_err_p95_kmh'], targets['speed_err_p95_kmh'])} |"]
    if proj:
        errs = np.array([p["err_cm"] for p in proj])
        L += [f"| Projection error (truncation test), 95th pct | {_pct(errs, 95):.1f} cm | "
              f"≤ {targets['projection_err_p95_cm']} cm | {chk(_pct(errs, 95), targets['projection_err_p95_cm'])} |"]
        L += ["", "Projection error by distance from the stumps where the track was cut:", "",
              "| Cut at | n | median | 95th pct | same wickets call |", "|---|---|---|---|---|"]
        for c in sorted({p["cut_m"] for p in proj}):
            e = [p["err_cm"] for p in proj if p["cut_m"] == c]
            same = np.mean([p["same_call"] for p in proj if p["cut_m"] == c]) * 100
            L.append(f"| {c:.1f} m | {len(e)} | {_pct(e, 50):.1f} cm | {_pct(e, 95):.1f} cm | {same:.0f}% |")
    bad = [r for r in rows if r.get("wickets_ok") == 0 or not r.get("tracked")]
    if bad:
        L += ["", "Deliveries to look at (untracked or wrong wickets call):", ""]
        L += [f"- {r['delivery']}: {r.get('note') or r.get('status')} "
              f"{'| truth ' + r['stumps_truth'] + ', ours ' + r['wickets_ours'] if 'stumps_truth' in r else ''}"
              for r in bad[:30]]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path


def load_reports(folder):
    reps = {}
    for fn in os.listdir(folder):
        if fn.endswith(".json"):
            with open(os.path.join(folder, fn)) as f:
                rep = json.load(f)
            if "bounce_xy_m" in rep:
                reps[fn[:-5]] = rep
    return reps


def load_truth(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def simulate(n, outdir, rig="2cam_end+side", fps=120.0, seed=0, px_sigma=1.5):
    """Make a synthetic validation set the same shape as a field trial:
    truth.csv (tape-measure noise 5 mm, radar noise 1 km/h, hit/miss at
    unguarded stumps) + reports/<id>.json, plus truncation-test results."""
    from . import sim, rigs
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(outdir, "reports"), exist_ok=True)
    truth, proj = [], []
    i = 0
    while i < n:
        d = sim.random_delivery(rng)
        if d is None:
            continue
        did = f"d{i:03d}"
        cams = rigs.make_rig(rig, fps=fps)
        est = [sim.calibrated_copy(c, rng)[0] for c in cams]
        # unguarded stumps: observe the ball right up to the stump plane
        t_stumps = d.t[np.searchsorted(-d.pos[:, 1], 0.0)]
        d_full = sim.Delivery(d.t, d.pos, d.t_bounce, d.bounce, t_stumps, d.at(t_stumps), d.stumps, d.params)
        obs_all = [(o.cam, o.t, o.uv) for o in sim.observe(d_full, cams, rng, px_sigma=px_sigma)]
        # the "match" fit uses the track only up to a pad position, like a real LBW
        obs = [o for o in obs_all if o[1] <= d.t_impact]
        try:
            r = fitmod.fit(obs, est, px_sigma=px_sigma)
            rep = r.report()
            with open(os.path.join(outdir, "reports", f"{did}.json"), "w") as f:
                json.dump(rep, f)
        except Exception:
            pass
        hit = lbw.wickets(*d.stumps)
        truth.append(dict(delivery=did,
                          pitch_x=round(d.bounce[0] + rng.normal(0, 0.005), 3),
                          pitch_y=round(d.bounce[1] + rng.normal(0, 0.005), 3),
                          speed_kmh=round(d.params["speed_kmh"] + rng.normal(0, 1.0), 1),
                          # a ball that only clips is recorded as a hit only if the bail falls --
                          # here: centre within the stumps' width + half a ball
                          stumps_result="hit" if hit in ("hitting", "umpires call") else "miss",
                          handed="right"))
        for p in truncation_test(obs_all, est, px_sigma=px_sigma):
            proj.append(dict(p, delivery=did))
        i += 1
    with open(os.path.join(outdir, "truth.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(truth[0]))
        w.writeheader()
        w.writerows(truth)
    return truth, proj
