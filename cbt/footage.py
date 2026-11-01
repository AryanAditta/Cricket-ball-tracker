"""Draw the ball tracking onto the ORIGINAL broadcast footage.

For every frame of the analysed shot the reconstructed 3D path is projected
back through the calibrated camera and drawn as a trail the width of the ball,
growing with the ball. After impact the projected path into the stumps is
added, the wicket zone is highlighted, and the DRS panels and the decision are
shown; the last frame is held so the verdict can be read.
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pitch import BALL_RADIUS, STUMP_HALF_WIDTH
from .lbw import ZONE_TOP
from .media import ffmpeg_exe

HERE = os.path.dirname(__file__)
RED, BLUE, WHITE = (216, 35, 42), (46, 123, 214), (255, 255, 255)
COL = dict(red=(216, 35, 42), green=(31, 157, 85), amber=(232, 150, 26), grey=(107, 119, 133),
           navy=(10, 22, 36))


def _font(size, bold=True):
    f = os.path.join(HERE, "vendor", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(f, size)


def _verdict_colour(field, v):
    if v == "inconclusive":
        return "grey"
    if v == "umpires call":
        return "amber"
    if field == "pitching":
        return "green" if v == "outside leg" else "red"
    if field == "impact":
        return "red" if v == "in line" else "green"
    return "red" if v == "hitting" else "green"


def final_banner(rep):
    word = rep["decision"].upper()
    if rep["wickets"] == "inconclusive":
        return "NO DECISION", "grey", f"insufficient tracking · on-field {word} stands"
    if rep.get("umpires_call"):
        return "UMPIRE'S CALL", "amber", f"{word} · on-field decision stands · {rep['reason']}"
    return word, "red" if rep["decision"] == "out" else "green", rep["reason"]


def _progress(samples, s_used, frame):
    """Ball position along the path (s) shown in `frame`, interpolated between
    tracked samples by frame index. None before the first sample."""
    if frame < samples[0]["f0"]:
        return None
    for k, smp in enumerate(samples):
        if smp["f0"] <= frame <= smp["f1"]:
            return s_used[k]
        if k + 1 < len(samples) and smp["f1"] < frame < samples[k + 1]["f0"]:
            a, b = smp["f1"], samples[k + 1]["f0"]
            w = (frame - a) / (b - a)
            return s_used[k] + w * (s_used[k + 1] - s_used[k])
    return s_used[-1]


def _draw_trail(img, cam, fit, s_from, s_to, colour, alpha=0.62, n=160):
    if s_from == s_to:
        return
    s = np.linspace(s_from, s_to, n)
    P = fit.at_s(s)
    uv = cam.project(P)
    depth = (P @ cam.R.T + cam.t)[:, 2]
    r = np.maximum(1.5, cam.K[0, 0] * BALL_RADIUS / depth)
    over = img.copy()
    for k in range(len(s) - 1):
        if np.all(np.isfinite(uv[k:k + 2])):
            cv2.line(over, tuple(np.round(uv[k]).astype(int)), tuple(np.round(uv[k + 1]).astype(int)),
                     colour[::-1], int(round(2 * r[k])), cv2.LINE_AA)
    cv2.addWeighted(over, alpha, img, 1 - alpha, 0, dst=img)


def _ground_ring(img, cam, x, y, colour, radius=0.06, width=2):
    a = np.linspace(0, 2 * np.pi, 48)
    uv = cam.project(np.c_[x + radius * np.cos(a), y + radius * np.sin(a), np.zeros_like(a)])
    if np.all(np.isfinite(uv)):
        cv2.polylines(img, [np.round(uv).astype(np.int32)], True, colour[::-1], width, cv2.LINE_AA)
        cv2.fillPoly(img, [np.round(uv).astype(np.int32)], tuple(int(c * 0.9) for c in colour[::-1]))


def _zone(img, cam, colour, alpha=0.35):
    q = np.array([[-STUMP_HALF_WIDTH, 0.0, 0.0], [STUMP_HALF_WIDTH, 0.0, 0.0],
                  [STUMP_HALF_WIDTH, 0.0, ZONE_TOP], [-STUMP_HALF_WIDTH, 0.0, ZONE_TOP]])
    uv = cam.project(q)
    if np.all(np.isfinite(uv)):
        over = img.copy()
        cv2.fillPoly(over, [np.round(uv).astype(np.int32)], colour[::-1])
        cv2.addWeighted(over, alpha, img, 1 - alpha, 0, dst=img)
        cv2.polylines(img, [np.round(uv).astype(np.int32)], True, colour[::-1], 2, cv2.LINE_AA)


def _panels(img, rep, show, u, title, sub):
    """DRS-style panels (drawn with PIL for clean text)."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    W, H = pil.size
    nav = COL["navy"] + (225,)
    # title
    ty = 70 * u                     # leaves the top-left corner free for a player's camera label
    d.rounded_rectangle([24 * u, ty, 640 * u, ty + 86 * u], 10 * u, fill=nav)
    d.rectangle([24 * u, ty, 32 * u, ty + 86 * u], fill=COL["green"] + (255,))
    d.text((48 * u, ty + 10 * u), title, font=_font(int(26 * u)), fill=WHITE)
    d.text((48 * u, ty + 48 * u), sub, font=_font(int(17 * u), bold=False), fill=(190, 204, 216))
    if show >= 1:
        rows = [("PITCHING", rep["pitching"], "pitching"), ("IMPACT", rep["impact"], "impact"),
                ("WICKETS", rep["wickets"], "wickets")]
        px, py, rh = W - 440 * u, 130 * u, 60 * u
        for k, (lab, val, field) in enumerate(rows):
            if k >= show:
                break
            y = py + k * (rh + 10 * u)
            d.rounded_rectangle([px, y, px + 180 * u, y + rh], 8 * u, fill=nav)
            d.text((px + 18 * u, y + rh / 2), lab, font=_font(int(22 * u)), fill=WHITE, anchor="lm")
            c = COL[_verdict_colour(field, val)] + (240,)
            d.rounded_rectangle([px + 188 * u, y, px + 416 * u, y + rh], 8 * u, fill=c)
            d.text((px + 302 * u, y + rh / 2), val.upper().replace("UMPIRES", "UMPIRE'S"),
                   font=_font(int(21 * u)), fill=WHITE, anchor="mm")
        if show >= 3:
            y = py + 3 * (rh + 10 * u)
            d.rounded_rectangle([px, y, px + 416 * u, y + 58 * u], 8 * u, fill=nav)
            d.text((px + 16 * u, y + 8 * u), "PITCHED", font=_font(int(12 * u)), fill=(184, 198, 212))
            d.text((px + 16 * u, y + 26 * u), f"{rep['bounce_xy_m'][1]:.2f} m", font=_font(int(20 * u)),
                   fill=WHITE)
            d.text((px + 145 * u, y + 8 * u), "IMPACT → STUMPS", font=_font(int(12 * u)),
                   fill=(184, 198, 212))
            d.text((px + 145 * u, y + 26 * u), f"{rep['impact_to_stumps_m']:.2f} m",
                   font=_font(int(20 * u)), fill=WHITE)
            sp = rep.get("release_speed_kmh") or rep.get("speed_estimate_kmh")
            d.text((px + 295 * u, y + 8 * u), "SPEED" if rep.get("release_speed_kmh") else "SPEED (EST.)",
                   font=_font(int(12 * u)), fill=(184, 198, 212))
            d.text((px + 295 * u, y + 26 * u), f"{sp:.0f} km/h" if sp else "—", font=_font(int(20 * u)),
                   fill=WHITE)
    if show >= 4:
        big, col, small = final_banner(rep)
        bw, bh = 760 * u, 118 * u
        bx, by = W / 2 - bw / 2, H - 170 * u
        d.rounded_rectangle([bx, by, bx + bw, by + bh], 12 * u, fill=COL[col] + (245,))
        d.text((W / 2, by + 46 * u), big, font=_font(int(56 * u)), fill=WHITE, anchor="mm")
        d.text((W / 2, by + 94 * u), small.upper(), font=_font(int(17 * u)), fill=WHITE, anchor="mm")
        c = rep["wickets_confidence"]
        conf = (f"tracking confidence: hitting {100 * c['hitting']:.0f}% · umpire's call "
                f"{100 * c['umpires call']:.0f}% · missing {100 * c['missing']:.0f}% · "
                f"±{200 * rep['stumps_sigma_m'][0]:.1f} cm sideways at the stumps"
                + (" · LOW CONFIDENCE" if rep.get("low_confidence") else ""))
        f = _font(int(16 * u), bold=False)
        tw = d.textlength(conf, font=f)
        d.rounded_rectangle([W / 2 - tw / 2 - 14 * u, by - 46 * u, W / 2 + tw / 2 + 14 * u, by - 12 * u],
                            8 * u, fill=nav)
        d.text((W / 2, by - 29 * u), conf, font=f, fill=(223, 231, 238), anchor="mm")
    out = Image.alpha_composite(pil, ov).convert("RGB")
    img[:] = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


def render(video, out_path, cam, fit, samples, rep, shot, s_used, title="BALL TRACKING · ADRS",
           sub="single camera · reconstructed from the broadcast replay", lead_in=None, hold_s=4.0,
           slow_hold=2):
    """samples: the tracked samples (dicts with f0, f1) in order; s_used: their
    positions along the path (fit.s for the inliers). Writes an H.264 MP4."""
    from .broadcast import read_frames, video_info
    info = video_info(video)
    fps = info["fps"]
    w, h = info["size"]
    u = h / 1080.0
    a, b = shot
    start = a if lead_in is None else max(0, a - lead_in)
    tmp = out_path + ".raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    s0 = s_used[0]
    s_imp = s_used[-1]
    s_stumps = -fit.params[0]
    f_imp = samples[-1]["f0"]
    proj_frames = 12                                  # frames to grow the projection
    bcol = COL[_verdict_colour("pitching", rep["pitching"])]
    zcol = COL[_verdict_colour("wickets", rep["wickets"])]
    last = None
    for i, frame in read_frames(video, range(start, b + 1)):
        img = frame.copy()
        if i >= a:
            sc = _progress(samples, s_used, i)
            if sc is not None:
                _draw_trail(img, cam, fit, s0, max(sc, s_imp), RED)
                if sc <= 0:
                    _ground_ring(img, cam, fit.bounce[0], fit.bounce[1], bcol)
            if i > f_imp:
                k = min(1.0, (i - f_imp) / proj_frames)
                _draw_trail(img, cam, fit, s_imp, s_imp + k * (s_stumps - s_imp), BLUE, alpha=0.7)
                if k >= 1.0:
                    _zone(img, cam, zcol)
            show = min(4, 1 + (i - f_imp - proj_frames) // 8) if i > f_imp + proj_frames else 0
            _panels(img, rep, show, u, title, sub)
        vw.write(img)
        last = frame
    if last is not None:                                 # hold the verdict on the last frame
        img = last.copy()
        _draw_trail(img, cam, fit, s0, s_imp, RED)
        _ground_ring(img, cam, fit.bounce[0], fit.bounce[1], bcol)
        _draw_trail(img, cam, fit, s_imp, s_stumps, BLUE, alpha=0.7)
        _zone(img, cam, zcol)
        _panels(img, rep, 4, u, title, sub)
        for _ in range(int(hold_s * fps)):
            vw.write(img)
    vw.release()
    exe = ffmpeg_exe()
    if exe:
        import subprocess
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt",
                        "yuv420p", "-crf", "20", "-movflags", "+faststart", out_path], check=True)
        os.remove(tmp)
    else:
        os.replace(tmp, out_path)
    return out_path, img


def calibration_check(bg, cam, path):
    """The pitch model drawn over the background: lines must sit on the
    painted creases and the stumps on the real stumps."""
    from .broadcast import _model_lines
    from .pitch import STUMP_HEIGHT
    img = bg.copy()
    for A, B in _model_lines():
        uv = cam.project(np.stack([A, B]))
        if np.all(np.isfinite(uv)):
            cv2.line(img, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)),
                     (255, 0, 255), 1, cv2.LINE_AA)
    for y in (0.0, 20.12):
        for x in (-0.0963, 0.0, 0.0963):
            uv = cam.project(np.array([[x, y, 0], [x, y, STUMP_HEIGHT]]))
            cv2.line(img, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)),
                     (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)
    return path
