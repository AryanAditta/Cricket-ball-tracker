"""`python -m cbt clicktool clip.mp4` -> a single HTML page for marking the
calibration points (crease corners) and the batter's front foot on a broadcast
clip. The page saves clicks.json for `python -m cbt broadcast`.

The page is self-contained (the shot's background image and video are
embedded), so it can be emailed or opened offline in Chrome, Edge or Firefox.
"""
import base64
import json
import os
import tempfile

import cv2

from . import broadcast as bc
from .media import to_webm

HERE = os.path.dirname(__file__)

SHORT = {"bowler_popping_crease_L": "near popping L", "bowler_popping_crease_R": "near popping R",
         "bowler_bowling_crease_L": "near stumps-line L", "bowler_bowling_crease_R": "near stumps-line R",
         "striker_popping_crease_L": "far popping L", "striker_popping_crease_R": "far popping R",
         "striker_bowling_crease_L": "far stumps-line L", "striker_bowling_crease_R": "far stumps-line R"}


def _b64(path, mime):
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def build(video, out=None, shot=None, prefill=None, handed=None):
    sc = bc.scan(video)
    if shot is None:
        shot, _ = bc.pick_replay_shot(video, sc)
    shot = (int(shot[0]), int(shot[1]))
    info = bc.video_info(video)
    bg = bc.background(video, shot)
    out = out or os.path.splitext(video)[0] + "_clicktool.html"
    with tempfile.TemporaryDirectory() as tmp:
        jpg = os.path.join(tmp, "bg.jpg")
        cv2.imwrite(jpg, bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        raw = os.path.join(tmp, "shot.mp4")
        w, h = info["size"]
        vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (w, h))
        for _, fr in bc.read_frames(video, range(shot[0], shot[1] + 1)):
            vw.write(fr)
        vw.release()
        webm = to_webm(raw, os.path.join(tmp, "shot.webm"), crf=30, gop=4)
        order = list(bc.CREASE_HELP)
        data = dict(source=os.path.basename(video), size=[w, h], fps=info["fps"], shot=list(shot),
                    background=_b64(jpg, "image/jpeg"), video=_b64(webm, "video/webm"),
                    points=[dict(name=n, label=SHORT[n][0].upper() + SHORT[n][1:], short=SHORT[n], help=bc.CREASE_HELP[n])
                            for n in order],
                    impact_frame_guess=(shot[0] + shot[1]) // 2, prefill=prefill, handed=handed)
    with open(os.path.join(HERE, "clicktool_template.html"), encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__DATA__", json.dumps(data).replace("</", "<\\/"), 1)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out
