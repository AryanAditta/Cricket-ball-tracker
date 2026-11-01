"""Overlay detections and the fitted 3D path back onto a camera's video."""
import subprocess
import shutil
import cv2
import numpy as np

from .detect import read_frames
from .pitch import BALL_RADIUS

GREEN, RED, BLUE, YELLOW, WHITE = (60, 200, 60), (40, 40, 220), (220, 120, 40), (0, 220, 255), (255, 255, 255)


def annotate(video, out_path, cam, fit, track=None, cands=None, t0=0.0, report=None):
    """cam: the calibrated camera; t0: master-clock time of the video's t=0."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or cam.fps
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    tmp = out_path + ".raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), min(fps, 30), (w, h))

    ts = np.linspace(fit.t_first, fit.t_last, 300)
    path_uv = cam.project(fit.pos(ts))
    tp = np.linspace(fit.t_last, fit.t_last + 0.3, 150)
    proj = fit.pos(tp)
    proj_uv = cam.project(proj[proj[:, 1] >= 0])
    bounce_uv = cam.project(fit.bounce[None])[0]
    used = set() if track is None else set(track[:, 0].astype(int))

    for i, t, frame in read_frames(video):
        T = t0 + t
        if cands is not None:
            for row in cands[cands[:, 0] == i]:
                c = (int(row[2]), int(row[3]))
                cv2.circle(frame, c, 9, GREEN if i in used else YELLOW, 1, cv2.LINE_AA)
        m = ts <= T
        if m.sum() > 1:
            pts = path_uv[m]
            pts = pts[np.all(np.isfinite(pts), 1)].astype(np.int32)
            cv2.polylines(frame, [pts], False, RED, 2, cv2.LINE_AA)
        if T >= fit.tb and np.all(np.isfinite(bounce_uv)):
            cv2.drawMarker(frame, tuple(bounce_uv.astype(int)), RED, cv2.MARKER_CROSS, 18, 2)
        if T >= fit.t_last and len(proj_uv) > 1:
            pts = proj_uv[np.all(np.isfinite(proj_uv), 1)].astype(np.int32)
            for a, b in zip(pts[:-1:2], pts[1::2]):          # dashed
                cv2.line(frame, tuple(a), tuple(b), BLUE, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (w, 34), (20, 20, 20), -1)
        txt = f"{cam.name}  t={T:+.3f}s"
        if report and T >= fit.t_last:
            txt += (f"   PITCHING {report['pitching'].upper()} | IMPACT {report['impact'].upper()}"
                    f" | WICKETS {report['wickets'].upper()}   {report['release_speed_kmh']:.0f} km/h")
        cv2.putText(frame, txt, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, "green=detections used  yellow=rejected  red=fitted path  blue=projection",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        vw.write(frame)
    vw.release()
    # re-encode to H.264 so it plays in browsers/phones, if ffmpeg is available
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "23", out_path], check=True)
        import os
        os.remove(tmp)
    else:
        shutil.move(tmp, out_path)
    return out_path
