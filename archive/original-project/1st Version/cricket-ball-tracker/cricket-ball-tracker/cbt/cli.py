"""Command line entry point:  python -m cbt <command> ...

  study      accuracy study on simulated deliveries (rigs x fps x noise)
  demo       end-to-end on synthetic VIDEO: render -> detect -> track -> fit -> figure
             + tracked videos to watch  (--hard: players, shadow, shake, flicker)
  videostudy many deliveries through the full video pipeline -> accuracy table
  calibrate  camera pose from clicked pitch landmarks
  detect     video -> ball candidates + ball tracks (CSV)
  fit        tracks from >= 1 calibrated camera -> LBW report (JSON) + figure
"""
import argparse
import json
import os
import time
import numpy as np

from . import sim, rigs, fit as fitmod, viz, lbw
from .camera import Camera, calibrate_pose, load_cameras, save_cameras
from .pitch import landmarks


# ------------------------------------------------------------------ study
def run_study(args):
    rng = np.random.default_rng(args.seed)
    deliveries = []
    while len(deliveries) < args.n:
        d = sim.random_delivery(rng)
        if d is not None:
            deliveries.append(d)
    rows = []
    for rig in args.rigs:
        for fps in args.fps:
            for px in args.px_sigma:
                errs = dict(bounce=[], sx=[], sz=[], speed=[], agree=0, n=0, fail=0, t=0.0)
                for d in deliveries:
                    cams = rigs.make_rig(rig, fps=fps)
                    est = [sim.calibrated_copy(c, rng)[0] for c in cams]
                    obs = [(o.cam, o.t, o.uv) for o in sim.observe(d, cams, rng, px_sigma=px)]
                    t0 = time.time()
                    try:
                        r = fitmod.fit(obs, est, px_sigma=max(px, 0.5))
                    except Exception:
                        errs["fail"] += 1
                        continue
                    errs["t"] += time.time() - t0
                    st = r.stumps()
                    if not np.all(np.isfinite(st)):
                        errs["fail"] += 1
                        continue
                    errs["bounce"].append(np.linalg.norm(r.bounce[:2] - d.bounce[:2]))
                    errs["sx"].append(abs(st[0] - d.stumps[0]))
                    errs["sz"].append(abs(st[1] - d.stumps[1]))
                    errs["speed"].append(abs(r.release_speed_kmh - d.params["speed_kmh"]))
                    errs["agree"] += lbw.wickets(*st) == lbw.wickets(*d.stumps)
                    errs["n"] += 1
                n = max(errs["n"], 1)
                p = lambda k, q: 100 * np.percentile(errs[k], q) if errs[k] else np.nan
                row = dict(rig=rig, fps=fps, px_sigma=px, n=errs["n"], failed=errs["fail"],
                           bounce_med_cm=p("bounce", 50), bounce_p95_cm=p("bounce", 95),
                           stumps_x_med_cm=p("sx", 50), stumps_x_p95_cm=p("sx", 95),
                           stumps_z_med_cm=p("sz", 50), stumps_z_p95_cm=p("sz", 95),
                           speed_med_kmh=np.median(errs["speed"]) if errs["speed"] else np.nan,
                           wickets_agree_pct=100 * errs["agree"] / n,
                           sec_per_fit=errs["t"] / n)
                rows.append(row)
                print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}),
                      flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


# ------------------------------------------------------------------ demo / video study
def _video_run(d, rig, fps, rng, outdir, hard, annotate_video=False):
    """Render every camera's video for delivery d, then run the real pipeline
    on those videos. Returns (fit, report, notes)."""
    from .synthvideo import render
    from .detect import MotionDetector, detect_video
    from .track2d import delivery_track
    cams = rigs.make_rig(rig, fps=fps, size=(1280, 720))
    est, obs, notes, per_cam = [], [], [], []
    for ci, cam in enumerate(cams):
        est.append(sim.calibrated_copy(cam, rng)[0])
        vid = os.path.join(outdir, f"{cam.name}.mp4")
        t0 = render(d, cam, vid, rng, hard=hard)
        cand = detect_video(vid, MotionDetector())
        tr = delivery_track(cand)
        per_cam.append((vid, cam, cand, tr, t0))
        if tr is None:
            notes.append(f"{cam.name}: no ball track found ({len(cand)} candidates)")
            continue
        notes.append(f"{cam.name}: {len(cand)} candidates -> {len(tr)} on the ball track")
        obs += [(ci, t0 + row[1], row[2:4]) for row in tr]
    if not obs:
        raise RuntimeError("no camera produced a ball track")
    r = fitmod.fit(obs, est)
    rep = r.report()
    if annotate_video:
        from .annotate import annotate
        for vid, cam, cand, tr, t0 in per_cam:
            ci = cams.index(cam)
            annotate(vid, vid.replace(".mp4", "_tracked.mp4"), est[ci], r, track=tr,
                     cands=cand, t0=t0, report=rep)
    return r, rep, notes


def run_demo(args):
    rng = np.random.default_rng(args.seed)
    d = None
    for _ in range(50):
        d = sim.random_delivery(rng, kind=args.kind, speed_kmh=args.speed, line=args.line,
                                length=args.length, swing=args.swing, turn=args.turn)
        if d is not None:
            break
    if d is None:
        raise SystemExit("that delivery can't be simulated (e.g. too full: no room for "
                         "the pad after the bounce) -- try a shorter --length")
    os.makedirs(args.outdir, exist_ok=True)
    r, rep, notes = _video_run(d, args.rig, args.fps, rng, args.outdir, args.hard,
                               annotate_video=True)
    print("\n".join(notes))
    rep["truth"] = dict(bounce_xy_m=d.bounce[:2].round(3).tolist(),
                        stumps_xz_m=d.stumps.round(3).tolist(),
                        speed_kmh=round(d.params["speed_kmh"], 1),
                        wickets=lbw.wickets(*d.stumps))
    with open(os.path.join(args.outdir, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    viz.figure(r, truth=d, rep=rep, path=os.path.join(args.outdir, "delivery.png"),
               title=f"Synthetic {d.params['kind']} delivery{' (hard mode)' if args.hard else ''}, "
                     f"{args.rig} @ {args.fps:.0f} fps (x / grey = ground truth)")
    print(json.dumps(rep, indent=2))
    print(f"\nwatch: {args.outdir}/*_tracked.mp4   figure: {args.outdir}/delivery.png")


def run_videostudy(args):
    """Many deliveries, each rendered to video and pushed through the full
    pipeline (detection included) -- the closest thing to a field test."""
    import csv
    import tempfile
    rng = np.random.default_rng(args.seed)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.n):
            d = None
            while d is None:
                d = sim.random_delivery(rng)
            row = dict(i=i, kind=str(d.params["kind"]), speed_kmh=round(d.params["speed_kmh"], 1))
            try:
                r, rep, notes = _video_run(d, args.rig, args.fps, rng, tmp, args.hard)
                st = r.stumps()
                row.update(ok=1, bounce_err_cm=100 * np.linalg.norm(r.bounce[:2] - d.bounce[:2]),
                           stumps_x_err_cm=100 * abs(st[0] - d.stumps[0]),
                           stumps_z_err_cm=100 * abs(st[1] - d.stumps[1]),
                           speed_err_kmh=abs(r.release_speed_kmh - d.params["speed_kmh"]),
                           call=rep["wickets"], true_call=lbw.wickets(*d.stumps),
                           notes=" | ".join(notes))
            except Exception as e:
                row.update(ok=0, notes=str(e))
            rows.append(row)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}),
                  flush=True)
    ok = [r for r in rows if r.get("ok")]
    def pct(k, q):
        v = [r[k] for r in ok if np.isfinite(r[k])]
        return np.percentile(v, q) if v else np.nan
    summary = dict(rig=args.rig, fps=args.fps, hard=args.hard, n=len(rows), tracked=len(ok),
                   bounce_med_cm=pct("bounce_err_cm", 50), bounce_p90_cm=pct("bounce_err_cm", 90),
                   stumps_x_med_cm=pct("stumps_x_err_cm", 50), stumps_x_p90_cm=pct("stumps_x_err_cm", 90),
                   stumps_z_med_cm=pct("stumps_z_err_cm", 50), stumps_z_p90_cm=pct("stumps_z_err_cm", 90),
                   speed_med_kmh=pct("speed_err_kmh", 50),
                   wickets_agree_pct=100 * np.mean([r["call"] == r["true_call"] for r in ok]) if ok else np.nan)
    print("SUMMARY", json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary.items()}))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ real-data path
def run_calibrate(args):
    """points CSV: name,u,v  (names from `python -m cbt landmarks`)."""
    K = np.array(json.load(open(args.intrinsics))["K"])
    lm = landmarks()
    names, uv = [], []
    for line in open(args.points):
        if not line.strip() or line.startswith("name"):
            continue
        n, u, v = line.strip().split(",")
        names.append(n); uv.append((float(u), float(v)))
    W = np.array([lm[n] for n in names])
    cam, rms = calibrate_pose(K, np.array(uv), W, name=args.name, fps=args.fps,
                              time_offset=args.time_offset)
    print(f"{args.name}: {len(names)} landmarks, reprojection rms {rms:.2f} px "
          f"(> ~2 px means a mis-click or wrong landmark name)")
    print("camera position (m):", cam.center.round(2))
    cams = load_cameras(args.out) if os.path.exists(args.out) else []
    cams = [c for c in cams if c.name != args.name] + [cam]
    save_cameras(cams, args.out)


def run_detect(args):
    from .detect import MotionDetector, YoloDetector, detect_video
    from .track2d import ball_tracks
    det = YoloDetector(args.weights) if args.weights else MotionDetector()
    cand = detect_video(args.video, det, args.start, args.end)
    np.savetxt(args.out + ".candidates.csv", cand, delimiter=",",
               header="frame,t,u,v,score", comments="")
    tracks = ball_tracks(cand)
    for i, tr in enumerate(tracks):
        np.savetxt(f"{args.out}.track{i}.csv", tr, delimiter=",",
                   header="frame,t,u,v,score", comments="")
    print(f"{len(cand)} candidates, {len(tracks)} ball tracks -> {args.out}.track*.csv")


def run_fit(args):
    cams = {c.name: c for c in load_cameras(args.cams)}
    obs, used = [], []
    for spec in args.track:              # camname=path
        name, path = spec.split("=", 1)
        cam = cams[name]
        ci = len(used); used.append(cam)
        tr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        obs += [(ci, row[1] - cam.time_offset, row[2:4]) for row in tr]
    r = fitmod.fit(obs, used, px_sigma=args.px_sigma)
    rep = r.report(handed=args.handed)
    print(json.dumps(rep, indent=2))
    with open(args.out + ".json", "w") as f:
        json.dump(rep, f, indent=2)
    viz.figure(r, rep=rep, path=args.out + ".png")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cbt", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("study")
    s.add_argument("--rigs", nargs="+", default=list(rigs.RIGS))
    s.add_argument("--fps", nargs="+", type=float, default=[120.0])
    s.add_argument("--px-sigma", nargs="+", type=float, default=[1.0])
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="results/study.csv")
    s.set_defaults(fn=run_study)

    s = sub.add_parser("demo")
    s.add_argument("--rig", default="2cam_end+side", choices=list(rigs.RIGS))
    s.add_argument("--fps", type=float, default=120.0)
    s.add_argument("--kind", choices=["pace", "spin"], default=None)
    s.add_argument("--seed", type=int, default=3)
    s.add_argument("--outdir", default="results/demo")
    s.add_argument("--hard", action="store_true", help="add players, shadow, shake, flicker")
    g = s.add_argument_group("bowl a specific delivery (anything left out is random)")
    g.add_argument("--speed", type=float, help="release speed, km/h (e.g. 140)")
    g.add_argument("--length", type=float, help="bounce distance from striker's stumps, m "
                   "(yorker ~1.5, full ~3, good ~5-6, short ~8+)")
    g.add_argument("--line", type=float, help="bounce position across the pitch, m: 0 = middle "
                   "stump, + = off side (right-hander), off stump ~ +0.11")
    g.add_argument("--swing", type=float, help="swing in the air, m/s^2 (+ = away to off, "
                   "typical -3..3)")
    g.add_argument("--turn", type=float, help="deviation off the pitch, m/s (+ = to off; "
                   "seam ~0.3, big spin ~2)")
    s.set_defaults(fn=run_demo)

    s = sub.add_parser("videostudy")
    s.add_argument("--rig", default="2cam_end+side", choices=list(rigs.RIGS))
    s.add_argument("--fps", type=float, default=120.0)
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--hard", action="store_true")
    s.add_argument("--out", default="results/videostudy.csv")
    s.set_defaults(fn=run_videostudy)

    s = sub.add_parser("landmarks")
    s.set_defaults(fn=lambda a: [print(f"{k:34s} {v}") for k, v in landmarks().items()])

    s = sub.add_parser("calibrate")
    s.add_argument("--name", required=True)
    s.add_argument("--intrinsics", required=True, help="json with K (from checkerboard)")
    s.add_argument("--points", required=True, help="csv name,u,v of clicked landmarks")
    s.add_argument("--fps", type=float, default=120.0)
    s.add_argument("--time-offset", type=float, default=0.0)
    s.add_argument("--out", default="cameras.json")
    s.set_defaults(fn=run_calibrate)

    s = sub.add_parser("detect")
    s.add_argument("video")
    s.add_argument("--weights", help="YOLO weights; omit to use the motion detector")
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--end", type=int, default=None)
    s.add_argument("--out", default="det")
    s.set_defaults(fn=run_detect)

    s = sub.add_parser("fit")
    s.add_argument("--cams", default="cameras.json")
    s.add_argument("--track", nargs="+", required=True, help="camname=track.csv ...")
    s.add_argument("--handed", choices=["right", "left"], default="right")
    s.add_argument("--px-sigma", type=float, default=1.5)
    s.add_argument("--out", default="delivery")
    s.set_defaults(fn=run_fit)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
