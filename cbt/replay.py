"""Broadcast-style 3D replay of a tracked delivery.

`write_replay` produces ONE self-contained HTML file (three.js inlined, works
offline in Chrome/Edge): pitch, stumps and bails at both ends, a batter whose
struck pad sits exactly at the tracked impact point, the ball in slow motion,
bounce mark, projected path into the stumps with its uncertainty, DRS-style
panels and the final decision. It can record itself to video from the browser.

`render_video` turns that page into an MP4 using headless Chromium
(pip install playwright && python -m playwright install chromium) + ffmpeg.
"""
import json
import os
import subprocess
import numpy as np

from .pitch import BALL_RADIUS, STUMP_RADIUS

HERE = os.path.dirname(__file__)


def _proj_end_y(wickets):
    # stop at the stumps if the ball would hit them; otherwise carry on past
    return STUMP_RADIUS + BALL_RADIUS if wickets in ("hitting", "umpires call") else -1.0


def replay_data(fit, rep, truth=None, handed="right", source="", badge=""):
    t0 = fit.t_first
    t = np.arange(fit.t_first, fit.t_last + 1e-9, 0.002)
    P = fit.pos(t)
    # projected path: from impact on, until the stump plane (or past it)
    y_end = _proj_end_y(rep["wickets"])
    tp = np.arange(fit.t_last, fit.t_last + 1.0, 0.002)
    Q = fit.pos(tp)
    keep = np.r_[True, Q[1:, 1] > y_end]
    stop = np.argmin(keep) if not keep.all() else len(keep)
    tp, Q = tp[:max(stop, 2)], Q[:max(stop, 2)]
    d = dict(
        handed=handed,
        path=dict(t=(t - t0).round(4).tolist(), p=P.round(4).tolist()),
        proj=dict(t=(tp - tp[0]).round(4).tolist(), p=Q.round(4).tolist()),
        bounce=dict(t=round(float(fit.tb - t0), 4), p=fit.bounce.round(4).tolist()),
        impact=dict(t=round(float(fit.t_last - t0), 4), p=fit.impact.round(4).tolist()),
        stumps_xz=[float(v) for v in rep["stumps_xz_m"]],
        stumps_sigma=[float(v) for v in rep["stumps_sigma_m"]],
        report=rep,
        meta=dict(source=source, badge=badge),
    )
    if truth is not None:
        m = truth.pos[:, 1] > -0.5
        tt = truth.t[m] - t0
        sel = slice(None, None, max(1, int(round(0.002 / max(np.median(np.diff(truth.t)), 1e-6)))))
        d["truth"] = dict(t=tt[sel].round(4).tolist(), p=truth.pos[m][sel].round(4).tolist())
    return d


def write_replay(path, data):
    with open(os.path.join(HERE, "replay_template.html"), encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(HERE, "vendor", "three.min.js"), encoding="utf-8") as f:
        three = f.read()
    with open(os.path.join(HERE, "vendor", "OrbitControls.js"), encoding="utf-8") as f:
        orbit = f.read()
    blob = json.dumps(data, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    blob = blob.replace("</", "<\\/")
    # use plain replace (no regex) so nothing inside the JS gets interpreted
    html = html.replace("/*__THREE__*/", three.replace("</script", "<\\/script"), 1)
    html = html.replace("/*__ORBIT__*/", orbit.replace("</script", "<\\/script"), 1)
    html = html.replace("__DATA__", blob, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def render_video(html_path, mp4_path, fps=30, size=(1920, 1080), speed=0.25, still=None, cam="auto",
                 t_from=0.0):
    """Render the replay to MP4 frame by frame (deterministic, any machine
    speed). `still=t` writes a single PNG at replay time t instead. cam: a
    camera preset (auto, broadcast, umpire, side, stumps, top); t_from skips
    the start of the timeline."""
    from playwright.sync_api import sync_playwright
    url = "file://" + os.path.abspath(html_path) + "?capture"
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader",
                                    "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        pg = b.new_page(viewport={"width": size[0], "height": size[1]}, device_scale_factor=1)
        pg.goto(url)
        pg.wait_for_function("window.REPLAY !== undefined")
        pg.evaluate(f"REPLAY.setSpeed({speed})")
        pg.evaluate(f"REPLAY.setCam('{cam}')")
        if still is not None:
            pg.evaluate(f"REPLAY.renderAt({still})")
            pg.screenshot(path=mp4_path)
            b.close()
            return mp4_path
        dur = pg.evaluate("REPLAY.duration")
        n = int((dur - t_from) * fps) + 1
        from .media import ffmpeg_exe
        ff = subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate",
                               str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                               "-crf", "18", "-movflags", "+faststart", mp4_path],
                              stdin=subprocess.PIPE)
        for k in range(n):
            pg.evaluate(f"REPLAY.renderAt({t_from + k / fps})")
            ff.stdin.write(pg.screenshot(type="png"))
        ff.stdin.close()
        ff.wait()
        b.close()
    return mp4_path
