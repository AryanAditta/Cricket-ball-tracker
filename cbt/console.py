"""TV-umpire operator console (modelled on the 'Smart Replay' workflow).

One page with: every camera view synced on a common clock with a frame-step
scrubber (release / bounce / impact markers), the 3D replay, the LBW protocol
walked through step by step (with the pitched-outside-leg fast path), the
review outcome, and a review log that can be exported to CSV.
"""
import json
import os
import shutil

from .media import ffmpeg_exe, to_webm

HERE = os.path.dirname(__file__)


def build_console(outdir, video_meta, fit, rep, replay_name="replay.html", delivery="delivery",
                  source="", name="console.html", times=None):
    """video_meta: list of dict(name, src, t0, fps, rate). Video time tau maps to
    master time t0 + tau * rate; rate = real seconds per second of video, so
    footage slowed down 4x (120 fps capture played at 30) has rate 0.25.
    times: override the release/bounce/impact marks (master time), e.g. for
    broadcast footage where the clip itself is the only timeline."""
    media = os.path.join(outdir, "console_media")
    os.makedirs(media, exist_ok=True)
    vids = []
    for m in video_meta:
        if not os.path.exists(m["src"]):
            continue
        if ffmpeg_exe():
            # VP9 with a keyframe every 8 frames: plays everywhere, steps frame by frame quickly
            dst = to_webm(m["src"], os.path.join(media, f"{m['name']}.webm"))
        else:
            dst = os.path.join(media, os.path.basename(m["src"]))
            shutil.copy(m["src"], dst)
        vids.append(dict(m, src=os.path.relpath(dst, outdir).replace(os.sep, "/")))
    data = dict(
        delivery=delivery, meta=dict(source=source), replay=replay_name, report=rep, videos=vids,
        times=times or dict(release=float(fit.t_first), bounce=float(fit.tb), impact=float(fit.t_last)),
    )
    with open(os.path.join(HERE, "console_template.html"), encoding="utf-8") as f:
        html = f.read()
    blob = json.dumps(data, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    html = html.replace("__DATA__", blob.replace("</", "<\\/"), 1)
    path = os.path.join(outdir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
