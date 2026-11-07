"""Command line entry point:  python -m cbt <command> ...

  study      accuracy study on simulated deliveries (rigs x fps x noise)
  demo       end-to-end on synthetic VIDEO: render -> detect -> track -> fit -> figure
             + tracked videos to watch  (--hard: players, shadow, shake, flicker)
  videostudy many deliveries through the full video pipeline -> accuracy table
  calibrate  camera pose from clicked pitch landmarks
  detect     video -> ball candidates + ball tracks (CSV)
  fit        tracks from >= 1 calibrated camera -> LBW report (JSON) + figure
  analyse    match video -> DRS package video, pictures, PDF report, console (one command)
  clicktool  broadcast clip -> a page to click the crease corners + front foot (clicks.json)
  broadcast  broadcast / replay clip + clicks.json -> tracking on the footage, 3D replay,
             console and LBW report (single camera)
"""
import argparse
import csv
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
        # what the operator console needs to keep the camera views in sync
        r.video_meta = [dict(name=cam.name, src=vid.replace(".mp4", "_tracked.mp4"), t0=t0,
                             fps=cam.fps, rate=min(cam.fps, 30) / cam.fps)
                        for vid, cam, cand, tr, t0 in per_cam]
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
    rep = r.report(handed=args.handed, on_field=args.on_field.replace("-", " "),
                   shot_offered=not args.no_shot)
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
    html = _write_replay(r, rep, d, args.handed, args.outdir,
                         f"SIMULATED DELIVERY · tracked from {len(rigs.RIGS[args.rig])} synthetic "
                         f"camera feed{'s' if len(rigs.RIGS[args.rig]) > 1 else ''}"
                         f"{' (hard mode)' if args.hard else ''}", args.video)
    from .console import build_console
    con = build_console(args.outdir, getattr(r, "video_meta", []), r, rep,
                        delivery=os.path.basename(os.path.normpath(args.outdir)),
                        source="simulated delivery")
    print(f"\nreplay (open in Chrome/Edge): {html}")
    print(f"operator console:            {con}")
    print(f"camera views: {args.outdir}/*_tracked.mp4   figure: {args.outdir}/delivery.png")


def _write_replay(r, rep, truth, handed, outdir, source, video=False, name="replay"):
    from . import replay
    from .browser import BrowserUnavailable
    data = replay.replay_data(r, rep, truth=truth, handed=handed, source=source,
                              badge="simulation" if truth is not None else "")
    html = replay.write_replay(os.path.join(outdir, f"{name}.html"), data)
    if video:
        try:
            mp4 = replay.render_video(html, os.path.join(outdir, f"{name}.mp4"))
            print(f"replay video: {mp4}")
        except BrowserUnavailable as exc:
            print(f"Replay video skipped: {exc}")
    return html


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
    with open(args.intrinsics, encoding="utf-8") as f:
        intrinsics = json.load(f)
    K = np.array(intrinsics["K"])
    lm = landmarks()
    names, uv = [], []
    for line in open(args.points):
        if not line.strip() or line.startswith("name"):
            continue
        n, u, v = line.strip().split(",")
        names.append(n); uv.append((float(u), float(v)))
    W = np.array([lm[n] for n in names])
    cam, rms = calibrate_pose(K, np.array(uv), W, dist=intrinsics.get("dist"),
                              size=intrinsics.get("size", (1920, 1080)), name=args.name, fps=args.fps,
                              time_offset=args.time_offset)
    cams = load_cameras(args.out) if os.path.exists(args.out) else []
    cams = [c for c in cams if c.name != args.name] + [cam]
    save_cameras(cams, args.out)
    print(f"{args.name}: {len(names)} landmarks, reprojection rms {rms:.2f} px "
          f"(> ~2 px means a mis-click or wrong landmark name)")
    print("camera position (m):", cam.center.round(2))


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
    videos = dict(spec.split("=", 1) for spec in args.camera_video)
    if args.track:
        sources = [spec.split("=", 1) for spec in args.track]
    else:
        sources = list(videos.items())
        if len(sources) < 2:
            raise SystemExit("Provide --track, or at least two calibrated --camera-video feeds.")
        from .detect import MotionDetector, YoloDetector, detect_video
        from .track2d import delivery_track
        from .broadcast import video_info
        detector = YoloDetector(args.weights) if args.weights else None
    seen = set()
    spans = []
    for name, path in sources:
        if name in seen or name not in cams:
            raise SystemExit(f"Camera '{name}' must be distinct and listed in {args.cams}.")
        seen.add(name)
        cam = cams[name]
        ci = len(used)
        used.append(cam)
        if args.track:
            tr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            rate = 1.0
        else:
            active_detector = detector or MotionDetector(ball_colour=args.ball)
            tr = delivery_track(detect_video(path, active_detector))
            if tr is None:
                raise RuntimeError(f"No delivery found for {name}. Supply a checked track using --track.")
            rate = video_info(path)["fps"] / cam.fps
            print(f"{name}: tracked {len(tr)} positions at capture rate {cam.fps:g} fps")
        times = rate * tr[:, 1] - cam.time_offset
        keep = times <= args.impact_time if args.impact_time is not None else np.ones(len(times), bool)
        if not keep.any():
            raise RuntimeError(f"No observations before impact for {name}. Check its clock offset.")
        obs += [(ci, t, uv) for t, uv in zip(times[keep], tr[keep, 2:4])]
        spans.append((times[keep].min(), times[keep].max()))
    if not args.track and max(a for a, _ in spans) >= min(b for _, b in spans):
        raise RuntimeError("The camera tracks do not overlap. Check the capture rates and clock offsets.")
    if max(sum(o[0] == i for o in obs) for i in range(len(used))) < 9:
        raise RuntimeError("At least one camera needs nine detections to fit the bounce.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    r = fitmod.fit(obs, used, px_sigma=args.px_sigma)
    if args.cut_y:          # validation Test B: pretend a pad stopped the ball at y = cut_y
        ts = np.linspace(r.tb, r.t_last, 400)
        ys = r.pos(ts)[:, 1]
        t_cut = ts[np.argmin(np.abs(ys - args.cut_y))]
        obs = [o for o in obs if o[1] <= t_cut]
        r = fitmod.fit(obs, used, px_sigma=args.px_sigma)
    rep = r.report(handed=args.handed, on_field=args.on_field.replace("-", " "),
                   shot_offered=not args.no_shot)
    print(json.dumps(rep, indent=2))
    with open(args.out + ".json", "w") as f:
        json.dump(rep, f, indent=2)
    viz.figure(r, rep=rep, path=args.out + ".png")
    html = _write_replay(r, rep, None, args.handed, os.path.dirname(os.path.abspath(args.out)),
                         f"Tracked from {len(used)} camera feed{'s' if len(used) > 1 else ''}",
                         args.video, name=os.path.basename(args.out) + "_replay")
    print(f"replay (open in Chrome/Edge): {html}")
    if args.camera_video:
        import cv2
        from .console import build_console
        meta = []
        for spec in args.camera_video:            # camname=path
            name, path = spec.split("=", 1)
            cap = cv2.VideoCapture(path)
            container_fps = cap.get(cv2.CAP_PROP_FPS) or cams[name].fps
            cap.release()
            # slow-motion phone files play at e.g. 30 fps but were captured at 240
            meta.append(dict(name=name, src=path, t0=-cams[name].time_offset, fps=cams[name].fps,
                             rate=container_fps / cams[name].fps))
        outdir = os.path.dirname(os.path.abspath(args.out))
        con = build_console(outdir, meta, r, rep, replay_name=os.path.basename(html),
                            delivery=os.path.basename(args.out), source="tracked from camera footage",
                            name=os.path.basename(args.out) + "_console.html")
        print(f"operator console: {con}")


def run_validate(args):
    from . import validate as V
    os.makedirs(args.out, exist_ok=True)
    proj = []
    if args.simulate:
        print(f"simulating a {args.simulate}-delivery field trial ...")
        truth, proj = V.simulate(args.simulate, args.out, seed=args.seed)
        truth_rows = V.load_truth(os.path.join(args.out, "truth.csv"))
        reports = V.load_reports(os.path.join(args.out, "reports"))
        title = f"Validation report (SIMULATED trial, {args.simulate} deliveries)"
    else:
        truth_rows, reports = V.load_truth(args.truth), V.load_reports(args.reports)
        title = "Validation report"
        if args.manifest:                       # truncation test on unguarded-stumps tracks
            cams = {c.name: c for c in load_cameras(args.cams)}
            by = {}
            for row in csv.DictReader(open(args.manifest, newline="")):
                by.setdefault(row["delivery"], []).append(row)
            for did, rows in by.items():
                used, obs = [], []
                for row in rows:
                    cam = cams[row["cam"]]
                    ci = len(used); used.append(cam)
                    tr = np.loadtxt(row["track_csv"], delimiter=",", skiprows=1, ndmin=2)
                    obs += [(ci, r_[1] - cam.time_offset, r_[2:4]) for r_ in tr]
                proj += [dict(p, delivery=did) for p in V.truncation_test(obs, used)]
    summary, rows = V.score(truth_rows, reports)
    md = V.write_report(os.path.join(args.out, "validation.md"), summary, rows, proj, title=title)
    with open(os.path.join(args.out, "per_delivery.csv"), "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(open(md, encoding="utf-8").read())
    print(f"written: {md}")


def run_broadcast(args):
    from .broadcast_run import process, summary
    from .broadcast_run import load_clicks
    cl = load_clicks(args.clicks) if args.clicks else {}   # handed / on-field from the click tool, unless given here
    handed = args.handed or cl.get("handed") or "right"
    on_field = (args.on_field or cl.get("on_field") or "not out").replace("-", " ")
    res = process(args.video, cl, args.outdir, handed=handed, bowler=args.bowler,
                  speed_kmh=args.speed_kmh, on_field=on_field,
                  shot_offered=not args.no_shot, render_video=not args.no_video,
                  shot=args.shot, ball=args.ball, weights=args.weights, stabilize=args.stabilize,
                  speed_roi=args.speed_roi)
    print()
    print(summary(res["report"]))
    print()
    for f in ("tracked_footage.mp4", "console.html", "replay.html", "report.json", "delivery.png",
              "calibration_check.jpg"):
        p = os.path.join(args.outdir, f)
        if os.path.exists(p):
            print(f"  {p}")


def run_analyse(args):
    import time
    from .broadcast_run import process, summary, load_clicks
    from . import package, report
    t0 = time.time()
    cl = load_clicks(args.clicks) if args.clicks else {}
    handed = args.handed or cl.get("handed") or "right"
    on_field = (args.on_field or cl.get("on_field") or "not out").replace("-", " ")
    outdir = args.outdir or os.path.join("results", os.path.splitext(os.path.basename(args.video))[0])
    res = process(args.video, cl, outdir, handed=handed, bowler=args.bowler, speed_kmh=args.speed_kmh,
                  on_field=on_field, shot_offered=not args.no_shot,
                  shot=args.shot, ball=args.ball, weights=args.weights, stabilize=args.stabilize,
                  speed_roi=args.speed_roi, on_result=lambda rep: print("\n" + summary(rep), flush=True))
    print(f"Open {os.path.join(outdir, 'console.html')} to review while exports finish.", flush=True)
    if not args.review_only:
        package.build_package(res, outdir, angles=not args.no_angles, angle_height=args.angle_height)
    html, pdf = report.build(res, outdir)
    print(f"\ndone in {time.time() - t0:.0f} s -> {outdir}")
    outputs = ["tracked_footage.mp4", "console.html", "replay.html", "report.html", "REPORT.txt", "report.json"]
    if not args.review_only:
        outputs += ["DRS_package.mp4", "pictures"]
    if pdf:
        outputs.append("report.pdf")
    for name in outputs:
        print(f"  {os.path.join(outdir, name)}")


def run_clicktool(args):
    from .clicktool import build
    path = build(args.video, args.out, shot=args.shot)
    print(f"open in Chrome/Edge: {path}")


def _broadcast_input_options(parser):
    parser.add_argument("--shot", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="inclusive replay frame range")
    parser.add_argument("--ball", choices=("red", "white"), default="red")
    parser.add_argument("--weights", help="optional cricket-ball YOLO weights")
    parser.add_argument("--stabilize", action="store_true", help="correct small pans and zooms")
    parser.add_argument("--speed-roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="suggest a speed from this km/h graphic region with Tesseract")


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
    s.add_argument("--handed", choices=["right", "left"], default="right", help="batter")
    s.add_argument("--on-field", choices=["not-out", "out"], default="not-out",
                   help="the on-field umpire's decision being reviewed")
    s.add_argument("--no-shot", action="store_true", help="batter offered no shot")
    s.add_argument("--video", action="store_true",
                   help="also render replay.mp4 (needs playwright + chromium)")
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
    s.add_argument("--track", nargs="+", help="camname=track.csv ...")
    s.add_argument("--handed", choices=["right", "left"], default="right")
    s.add_argument("--px-sigma", type=float, default=1.5)
    s.add_argument("--out", default="delivery")
    s.add_argument("--video", action="store_true", help="also render the replay to MP4")
    s.add_argument("--on-field", choices=["not-out", "out"], default="not-out")
    s.add_argument("--no-shot", action="store_true")
    s.add_argument("--cut-y", type=float, default=0.0,
                   help="validation: drop detections after the ball reaches this distance (m) "
                        "from the stumps, as if a pad were there")
    s.add_argument("--camera-video", nargs="+", default=[],
                   help="camname=video.mp4 ... to build the console or detect without --track")
    s.add_argument("--impact-time", type=float, help="last usable timestamp on the common camera clock")
    s.add_argument("--ball", choices=("red", "white"), default="red", help="colour for automatic detection")
    s.add_argument("--weights", help="YOLO cricket-ball weights for automatic detection")
    s.set_defaults(fn=run_fit)

    s = sub.add_parser("analyse", aliases=["analyze"],
                       help="match video -> DRS package video, pictures, PDF report (one command)")
    s.add_argument("video")
    s.add_argument("--outdir", default=None, help="default: results/<video name>")
    s.add_argument("--clicks", default=None,
                   help="clicks.json from `cbt clicktool` (only if automatic calibration fails)")
    s.add_argument("--handed", choices=["right", "left"], default=None, help="batter (default right)")
    s.add_argument("--on-field", choices=["not-out", "out"], default=None,
                   help="the on-field decision being reviewed (default not-out)")
    s.add_argument("--no-shot", action="store_true", help="batter offered no shot")
    s.add_argument("--bowler", choices=["pace", "medium", "spin"], default=None)
    s.add_argument("--speed-kmh", type=float, default=None, help="speed-gun reading if shown")
    s.add_argument("--no-angles", action="store_true", help="skip the 3D angle videos (faster)")
    s.add_argument("--angle-height", type=int, choices=[480, 720, 1080], default=720,
                   help="height of rendered 3D angles (480 is faster)")
    s.add_argument("--review-only", action="store_true", help="skip the presentation video")
    _broadcast_input_options(s)
    s.set_defaults(fn=run_analyse)

    s = sub.add_parser("clicktool", help="make the page for clicking calibration points on a clip")
    s.add_argument("video")
    s.add_argument("--out", default=None, help="html path (default: next to the video)")
    s.add_argument("--shot", type=int, nargs=2, metavar=("FIRST", "LAST"),
                   help="frames of the replay shot (default: picked automatically)")
    s.set_defaults(fn=run_clicktool)

    s = sub.add_parser("broadcast", help="track a delivery in broadcast / replay footage")
    s.add_argument("video")
    s.add_argument("--clicks", default=None, help="clicks.json from `cbt clicktool` (default: automatic)")
    s.add_argument("--outdir", default="results/broadcast")
    s.add_argument("--handed", choices=["right", "left"], default=None,
                   help="batter (default: from clicks.json, else right)")
    s.add_argument("--bowler", choices=["pace", "medium", "spin"], default=None,
                   help="adds a speed prior (leave out unless the fit needs it)")
    s.add_argument("--speed-kmh", type=float, default=None, help="speed-gun reading if shown")
    s.add_argument("--on-field", choices=["not-out", "out"], default=None,
                   help="on-field decision (default: from clicks.json, else not-out)")
    s.add_argument("--no-shot", action="store_true", help="batter offered no shot")
    s.add_argument("--no-video", action="store_true", help="skip rendering tracked_footage.mp4")
    _broadcast_input_options(s)
    s.set_defaults(fn=run_broadcast)

    s = sub.add_parser("validate", help="score the system against field ground truth (VALIDATION.md)")
    s.add_argument("--truth", help="truth.csv: delivery,pitch_x,pitch_y,stumps_result,speed_kmh,handed")
    s.add_argument("--reports", help="folder of <delivery>.json reports from `cbt fit`")
    s.add_argument("--manifest", help="unguarded-stumps tracks for the projection test: "
                                      "delivery,cam,track_csv")
    s.add_argument("--cams", default="cameras.json")
    s.add_argument("--simulate", type=int, default=0,
                   help="instead, simulate an N-delivery trial to see what the report looks like")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="results/validation")
    s.set_defaults(fn=run_validate)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
