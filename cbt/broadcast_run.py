"""One command for a broadcast / replay clip:

    python -m cbt broadcast match.mp4 --clicks clicks.json --handed left --on-field out

scan -> pick the replay shot -> background -> calibrate from the clicked
crease corners -> ball candidates -> track -> single-camera reconstruction
(with the front-foot prior) -> LBW report -> outputs:

    tracked_footage.mp4   the tracking drawn on the original footage
    replay.html           3D replay (orbit, camera presets, record button)
    console.html          third-umpire console (footage + replay + decision)
    report.json           the decision and every number behind it
    delivery.png          plan / side / end view figure
    calibration_check.jpg pitch model drawn over the footage (check this first)
"""
import json
import os
import numpy as np

from . import broadcast as bc
from .monofit import reconstruct


def load_clicks(path):
    """clicks.json from the click tool. Also accepts a flat {name: [u, v]} map."""
    with open(path) as f:
        d = json.load(f)
    if "points" not in d:
        d = dict(points={k: v for k, v in d.items() if k in bc.CREASE_POINTS})
    return d


def candidates_for_shot(video, shot, cam, bg, log=print, ball="red", weights=None):
    import cv2
    a, b = shot
    roi = bc.corridor_roi(cam, bg.shape)
    # ball size mid-pitch (it varies ~1.5x end to end; the blob filter tolerates that)
    mid = cam.project(np.array([[0.0, 10.0, 0.8]]))
    r_px = float(np.clip(bc.expected_radius_px(cam, mid)[0], 2.5, 25))
    bg_w = bc.warmth(bg) if ball == "red" and not weights else None
    bg_g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detector = None
    if weights:
        from .detect import YoloDetector
        detector = YoloDetector(weights)
    h, w = roi.shape
    cands = {}
    for i, fr in bc.read_frames(video, range(a, b + 1)):
        if detector is None:
            cands[i] = bc.ball_candidates(fr, bg, roi, r_px, ball=ball, bg_warm=bg_w, bg_gray=bg_g)
        else:
            rows = [(u, v, 100 * confidence) for u, v, confidence in detector(fr)
                    if 0 <= u < w and 0 <= v < h and roi[int(v), int(u)]]
            cands[i] = np.asarray(rows, float).reshape(-1, 3)
    log(f"ball candidates: frames {a}-{b}, expected ball radius {r_px:.1f}px")
    return cands


def process(video, clicks, outdir, handed="right", bowler=None, speed_kmh=None, on_field="not out",
            shot_offered=True, shot=None, title="BALL TRACKING · ADRS", render_video=True, log=print,
            ball="red", weights=None, stabilize=False, speed_roi=None, on_result=None):
    import cv2
    os.makedirs(outdir, exist_ok=True)
    cl = load_clicks(clicks) if isinstance(clicks, str) else dict(clicks or {})
    source = os.path.basename(video)
    shot = tuple(shot or cl.get("shot") or ()) or None
    if stabilize and (cl.get("points") or cl.get("foot")):
        raise ValueError("Clicks use original video coordinates. Mark the stabilized video in clicktool instead.")
    sc = bc.scan(video)
    if shot is None:
        shot, _ = bc.pick_replay_shot(video, sc, allow_motion=stabilize)
        log(f"replay shot picked automatically: frames {shot[0]}-{shot[1]} "
            "(pass the shot in clicks.json if this is wrong)")
    else:
        log(f"replay shot: frames {shot[0]}-{shot[1]}")
    if len(shot) != 2 or not 0 <= shot[0] <= shot[1] < len(sc["diffs"]) or shot[1] - shot[0] < 5:
        raise ValueError("Choose at least six consecutive frames inside the video")
    original_shot = shot
    speed_suggestion = None
    if speed_roi is not None:
        from .speed_ocr import suggest_speed
        try:
            speed_suggestion = suggest_speed(video, shot, speed_roi)
        except (RuntimeError, ValueError) as exc:
            log(str(exc))
        if speed_suggestion is not None:
            log(f"Speed graphic suggests {speed_suggestion:g} km/h. Confirm before using --speed-kmh.")
        else:
            log("No consistent speed-gun reading found in the selected region.")
    if stabilize:
        from .stabilize import stabilize_shot
        if any(shot[0] < cut <= shot[1] for cut in sc["cuts"]):
            raise ValueError("Stabilization needs one continuous replay shot. Select it with --shot.")
        corrected = os.path.join(outdir, "stabilized_input.mp4")
        stabilize_shot(video, shot, corrected)
        video = corrected
        shot = (0, original_shot[1] - original_shot[0])
        sc = bc.scan(video)
        log(f"stabilized replay written to {corrected}")
    bg = bc.background(video, shot)

    # 1. calibration: clicked crease corners, or found automatically
    if cl.get("points"):
        points, how = cl["points"], f"{len(cl['points'])} clicked crease points"
    else:
        try:
            points, _ = bc.auto_crease_points(bg)
        except RuntimeError as e:
            raise RuntimeError(f"automatic calibration failed ({e}); mark the creases with "
                               "`python -m cbt clicktool` and pass --clicks") from None
        how = "crease corners found automatically"
    cam, info = bc.calibrate_from_clicks(bg, points)
    cam.name = "broadcast"
    info["how"] = how
    info["crease_points"] = points
    log(f"calibration: {how}, rms {info['rms_px']:.2f}px, "
        f"focal {info['focal_px']:.0f}px, camera at {info['camera_position_m']} m")
    if not cl.get("points") and (info["rms_px"] > 3.0 or info.get("inliers", 0) < 40):
        raise RuntimeError("automatic calibration doesn't fit the pitch markings well "
                           f"(rms {info['rms_px']:.1f}px); use `python -m cbt clicktool` and --clicks")
    from .footage import calibration_check, render
    calibration_check(bg, cam, os.path.join(outdir, "calibration_check.jpg"))
    if info["rms_px"] > 3.0:
        log("WARNING: calibration residual is high -- check calibration_check.jpg and the clicks")

    # 2. ball
    cands = candidates_for_shot(video, shot, cam, bg, log, ball=ball, weights=weights)
    held = sc["held"]
    track = bc.track_delivery(cands, held, cam=cam, shape=bg.shape)
    if len(track) < 6:
        raise RuntimeError(f"could not find the ball's path (only {len(track)} positions)")
    log(f"tracker: {len(track)} positions, frames {track[0]['f0']}-{track[-1]['f1']}")

    # 3. reconstruction
    foot = cl.get("foot")
    fit, used, rlog = reconstruct(track, cands, cam, foot_uv=foot, bowler=bowler or cl.get("bowler"),
                                  speed_kmh=speed_kmh or cl.get("speed_kmh"))
    for line in rlog:
        log("  " + line)
    foot_how = "clicked" if foot is not None else None
    if foot is None:
        imp = used[-1]
        ref = range(shot[0], max(shot[0] + 3, min(track[0]["f0"], shot[0] + 8)))
        r_px = float(bc.expected_radius_px(cam, [[imp["u"], imp["v"]]])[0])
        foot, finfo = bc.auto_front_foot(video, imp["f0"], (imp["u"], imp["v"]), ref, r_px)
        if foot is not None:
            foot_how = "found automatically"
            uv = np.array([[n["u"], n["v"]] for n in used])
            from .monofit import fit_mono
            fit = fit_mono(uv, cam, foot_uv=foot, bowler=bowler or cl.get("bowler"),
                           speed_kmh=speed_kmh or cl.get("speed_kmh"))
            log(f"front foot found automatically at ({foot[0]:.0f}, {foot[1]:.0f}) in frame {imp['f0']}")
        else:
            log("NOTE: front foot not found -- the impact point is only constrained by the last "
                "ball position seen; mark it with the click tool for a firmer result")
    extra_problems = ()
    if stabilize and foot is None:
        extra_problems = ("stabilized replay needs a front-foot mark to anchor impact depth",)
        log("Mark the front foot on stabilized_input.mp4 with clicktool and analyse that clip directly.")
    rep = fit.report(handed=handed, on_field=on_field, shot_offered=shot_offered,
                     extra_problems=extra_problems)
    rep.update(source=source, shot_frames=list(map(int, original_shot)),
               track_frame_offset=original_shot[0] if stabilize else 0,
               stabilized=bool(stabilize), ball_colour=ball,
               detector="YOLO" if weights else "foreground",
               speed_gun_kmh=speed_kmh or cl.get("speed_kmh"), speed_ocr_suggestion_kmh=speed_suggestion,
               calibration=dict(rms_px=round(info["rms_px"], 2), focal_px=round(info["focal_px"]),
                                camera_position_m=info["camera_position_m"]),
               front_foot_px=foot, front_foot=foot_how, calibration_source=how)
    samples = bc.samples_from_track(used)
    s_used = fit.s.copy()
    inl = fit.inlier
    samples = [smp for smp, k in zip(samples, inl) if k]
    s_used = s_used[inl]
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if on_result is not None:
        on_result(rep)
    bc.save_json(os.path.join(outdir, "track.json"),
                 dict(samples=samples, s=s_used, camera=dict(K=cam.K, R=cam.R, t=cam.t, size=cam.size)))
    from .camera import save_cameras
    save_cameras([cam], os.path.join(outdir, "camera.json"))

    # 4. outputs
    from . import viz, replay
    viz.figure(fit, rep=rep, path=os.path.join(outdir, "delivery.png"),
               title=f"{source} · single camera")
    src = f"{source} · single camera, " + ("stabilized replay" if stabilize else "broadcast replay")
    html = replay.write_replay(os.path.join(outdir, "replay.html"),
                               replay.replay_data(fit, rep, handed=handed, source=src))
    out_mp4 = os.path.join(outdir, "tracked_footage.mp4")
    if render_video:
        render(video, out_mp4, cam, fit, samples, rep, shot, s_used, title=title,
               sub=f"{rep['handed']}-handed batter · on-field {on_field.upper()}")
        fps = bc.video_info(video)["fps"]
        a = shot[0]
        times = dict(release=(samples[0]["f0"] - a) / fps, bounce=(_bounce_frame(samples, s_used) - a) / fps,
                     impact=(samples[-1]["f0"] - a) / fps)
        from .console import build_console
        build_console(outdir, [dict(name="broadcast", src=out_mp4, t0=0.0, fps=fps, rate=1.0)], fit, rep,
                      replay_name="replay.html", delivery=os.path.splitext(source)[0],
                      source=src, times=times)
    return dict(fit=fit, report=rep, camera=cam, calibration=info, samples=samples, s=s_used,
                shot=shot, outdir=outdir, background=bg, video=video, used=used)


def _bounce_frame(samples, s):
    k = int(np.argmin(np.abs(s)))
    return samples[k]["f0"]


def summary(rep):
    from .footage import final_banner
    big, _, why = final_banner(rep)
    sp = rep.get("release_speed_kmh") or rep.get("speed_estimate_kmh")
    c = rep["wickets_confidence"]
    return "\n".join([
        f"DECISION  {big}  -- {why}",
        f"  pitching {rep['pitching']} | impact {rep['impact']} | wickets {rep['wickets']}",
        f"  pitched {rep['bounce_xy_m'][1]:.2f} m from the stumps; impact {rep['impact_to_stumps_m']:.2f} m out",
        f"  at the stumps: x {rep['stumps_xz_m'][0]:+.3f} m, height {rep['stumps_xz_m'][1]:.3f} m "
        f"(±{200 * rep['stumps_sigma_m'][0]:.1f} / ±{200 * rep['stumps_sigma_m'][1]:.1f} cm, 2σ)",
        f"  paths hitting {c['hitting']:.0%} · umpire's call {c['umpires call']:.0%} · missing {c['missing']:.0%}",
        f"  speed {'≈%.0f km/h (estimate)' % sp if sp else '—'} · {rep['n_inliers']} positions, rms {rep['rms_px']} px",
        f"  status: {rep['status']}" + ("  (low confidence: wide spread of plausible paths)"
                                         if rep.get("low_confidence") else ""),
    ])
