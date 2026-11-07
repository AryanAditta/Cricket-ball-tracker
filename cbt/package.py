"""The DRS package: one broadcast-style video, a folder of pictures and a
PDF report for a reviewed delivery.

    DRS_package.mp4
      1. title card
      2. the delivery as broadcast (replay shot, untouched)
      3. BALL TRACKING on the real footage (path, bounce, projection, panels)
      4. 3D replay - main cinematic view (slow motion, projection, decision)
      5. 3D replay - umpire's view, side-on, stumps close-up
      6. pitch map
      7. decision card
    pictures/   a still from each section
    report.pdf  decision, protocol, measurements, pictures, method, limits
"""
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .footage import _font, COL, _verdict_colour, final_banner
from .media import ffmpeg_exe
from .browser import BrowserUnavailable, chromium_browser, file_url
from .lbw import decision_rows

W, H, FPS = 1920, 1080, 30
NAVY = (10, 22, 36)
ANGLES = [("auto", "3D REPLAY", None), ("umpire", "3D REPLAY · UMPIRE'S VIEW", "seg"),
          ("side", "3D REPLAY · SIDE-ON", "seg"), ("stumps", "3D REPLAY · AT THE STUMPS", "seg")]


# ---------------------------------------------------------------- drawing helpers
def _label(text, sub=None):
    """RGBA strip drawn on every frame of a section (top-left)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f, fs = _font(30), _font(17, bold=False)
    tw = max(d.textlength(text, font=f), d.textlength(sub, font=fs) if sub else 0)
    x, y = W / 2 - tw / 2 - 26, 22                   # top centre: free in both the footage and the 3D views
    hh = 82 if sub else 56
    d.rounded_rectangle([x, y, x + tw + 52, y + hh], 8, fill=(10, 22, 36, 225))
    d.rectangle([x, y + hh - 5, x + tw + 52, y + hh], fill=COL["amber"] + (255,))
    d.text((W / 2, y + 28), text, font=f, fill=(255, 255, 255), anchor="mm")
    if sub:
        d.text((W / 2, y + 60), sub, font=fs, fill=(200, 212, 224), anchor="mm")
    left, top, right, bottom = img.getbbox()
    rgba = np.asarray(img.crop((left, top, right, bottom)), dtype=np.float32)
    return (slice(top, bottom), slice(left, right)), rgba[..., :3][..., ::-1], rgba[..., 3:] / 255


def _apply(frame, lab):
    area, rgb, alpha = lab
    out = frame.copy()
    out[area] = frame[area] * (1 - alpha) + rgb * alpha
    return out


def title_card(rep, path):
    img = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 18, H], fill=COL["green"])
    d.text((120, 300), "BALL TRACKING", font=_font(110), fill=(255, 255, 255))
    d.text((124, 440), "LBW REVIEW  ·  ADRS", font=_font(48), fill=(150, 220, 180))
    rows = [("Clip", rep.get("source", "")),
            ("Batter", f"{rep['handed']}-handed"),
            ("On-field decision", rep["on_field"].upper()),
            ("Tracking", rep.get("method", "multi-camera"))]
    y = 600
    for k, v in rows:
        d.text((124, y), k.upper(), font=_font(24), fill=(140, 160, 180))
        d.text((480, y), str(v), font=_font(28, bold=False), fill=(235, 240, 245))
        y += 52
    img.save(path)
    return path


def decision_card(rep, path):
    img = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(img)
    big, col, small = final_banner(rep)
    d.text((W / 2, 150), "DECISION", font=_font(40), fill=(150, 170, 190), anchor="mm")
    d.rounded_rectangle([W / 2 - 560, 210, W / 2 + 560, 420], 18, fill=COL[col])
    d.text((W / 2, 300), big, font=_font(120), fill=(255, 255, 255), anchor="mm")
    d.text((W / 2, 385), small.upper(), font=_font(28), fill=(255, 255, 255), anchor="mm")
    rows = [("PITCHING", rep["pitching"], "pitching"), ("IMPACT", rep["impact"], "impact"),
            ("WICKETS", rep["wickets"], "wickets")]
    for k, (lab, val, field) in enumerate(rows):
        y = 480 + k * 100
        d.rounded_rectangle([W / 2 - 420, y, W / 2 - 20, y + 84], 10, fill=(20, 38, 56))
        d.text((W / 2 - 390, y + 42), lab, font=_font(38), fill=(255, 255, 255), anchor="lm")
        d.rounded_rectangle([W / 2, y, W / 2 + 420, y + 84], 10, fill=COL[_verdict_colour(field, val)])
        d.text((W / 2 + 210, y + 42), val.upper().replace("UMPIRES", "UMPIRE'S"), font=_font(36),
               fill=(255, 255, 255), anchor="mm")
    c = rep["wickets_confidence"]
    sp = rep.get("release_speed_kmh") or rep.get("speed_estimate_kmh")
    lines = [f"Pitched {rep['bounce_xy_m'][1]:.2f} m from the stumps  ·  impact {rep['impact_to_stumps_m']:.2f} m "
             f"from the stumps" + (f"  ·  speed ≈{sp:.0f} km/h" if sp else ""),
             f"Plausible paths: hitting {100 * c['hitting']:.0f}%  ·  umpire's call {100 * c['umpires call']:.0f}%"
             f"  ·  missing {100 * c['missing']:.0f}%   (±{200 * rep['stumps_sigma_m'][0]:.1f} cm sideways)"
             + ("  ·  LOW CONFIDENCE" if rep.get("low_confidence") else "")]
    for k, t in enumerate(lines):
        d.text((W / 2, 820 + k * 50), t, font=_font(28, bold=False), fill=(215, 225, 235), anchor="mm")
    img.save(path)
    return path


def evidence_card(rep, path):
    img = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(img)
    d.text((112, 88), "DECISION MEASUREMENTS", font=_font(66), fill=(255, 255, 255))
    d.text((116, 179), "Positive sideways values are towards off.",
           font=_font(26, bold=False), fill=(162, 186, 208))
    columns = (116, 785, 1115)
    for x, label in zip(columns, ("MEASUREMENT", "VALUE", "RULE BOUNDARY")):
        d.text((x, 252), label, font=_font(25), fill=(162, 186, 208))
    for i, values in enumerate(decision_rows(rep)):
        y = 305 + 78 * i
        d.rounded_rectangle((92, y, 1828, y + 65), 10, fill=(21, 40, 57))
        for x, value in zip(columns, values):
            d.text((x, y + 32), value, font=_font(29, bold=x == columns[1]),
                   fill=(245, 249, 253), anchor="lm")
    img.save(path)
    return path


# ---------------------------------------------------------------- video writer
class Writer:
    def __init__(self, path):
        self.raw = path + ".raw.avi"
        self.path = path
        self.vw = cv2.VideoWriter(self.raw, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
        self.n = 0

    def frame(self, img, lab=None):
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA if img.shape[1] > W else cv2.INTER_CUBIC)
        if lab is not None:
            img = _apply(img, lab)
        self.vw.write(img)
        self.n += 1
        return img

    def hold(self, img, seconds, lab=None):
        frames = int(round(seconds * FPS))
        if frames <= 0:
            return None
        out = self.frame(img, lab)
        for _ in range(frames - 1):
            self.vw.write(out)
        self.n += frames - 1
        return out

    def video(self, path, lab=None, fps_in=None, start=0, stop=None, every=1):
        cap = cv2.VideoCapture(path)
        k, last = 0, None
        while True:
            ok, img = cap.read()
            if not ok or (stop is not None and k > stop):
                break
            if k >= start and (k - start) % every == 0:
                last = self.frame(img, lab)
            k += 1
        cap.release()
        return last

    def close(self):
        self.vw.release()
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", self.raw, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", "-movflags", "+faststart",
                        self.path], check=True)
        os.remove(self.raw)
        return self.path


# ---------------------------------------------------------------- 3D angles
def render_angles(replay_html, outdir, size=(1280, 720), log=print):
    """Render every angle and the pitch map from a single loaded scene."""
    args = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
    out = {}
    with chromium_browser(args) as browser:
        page = browser.new_page(viewport={"width": size[0], "height": size[1]}, device_scale_factor=1)
        page.goto(file_url(replay_html, capture=True))
        page.wait_for_function("window.REPLAY !== undefined")
        page.evaluate("REPLAY.setSpeed(0.25)")
        tl = page.evaluate("REPLAY.timeline")
        for cam, title, part in ANGLES:
            path = os.path.join(outdir, f"angle_{cam}.mp4")
            log(f"rendering 3D replay: {title.lower()}")
            t0 = tl["t1"] + 0.4 * (tl["t2"] - tl["t1"]) if part == "seg" else 0.0
            t1 = tl["t5"] + 0.8 if part == "seg" else min(tl["end"], tl["t5"] + 1.5)
            page.evaluate("(cam) => REPLAY.setCam(cam)", cam)
            _render_part(page, path, t0, t1)
            out[cam] = path
        still = os.path.join(outdir, "pitch_map.png")
        page.evaluate("(t) => { REPLAY.setCam('top'); REPLAY.renderAt(t); }", tl["end"])
        page.screenshot(path=still)
        out["pitch_map"] = still
    return out, tl


def _render_part(page, mp4, t0, t1):
    from .replay import encode_frames
    encode_frames(page, mp4, FPS, t0, int((t1 - t0) * FPS) + 1)


# ---------------------------------------------------------------- package
def build_package(res, outdir, log=print, angles=True, angle_height=720):
    """res: the dict returned by broadcast_run.process (outputs already in outdir)."""
    from .broadcast import read_frames
    rep, video, shot = res["report"], res["video"], res["shot"]
    pics = os.path.join(outdir, "pictures")
    os.makedirs(pics, exist_ok=True)
    work = tempfile.mkdtemp(prefix="cbt_pkg_")
    try:
        ang = {}
        if angles:
            try:
                ang, _ = render_angles(os.path.join(outdir, "replay.html"), work,
                                       size=(2 * round(angle_height * 8 / 9), angle_height), log=log)
            except BrowserUnavailable as exc:
                log(f"3D angles skipped: {exc}")
        title = title_card(rep, os.path.join(pics, "00_title.png"))
        verdict = decision_card(rep, os.path.join(pics, "90_decision.png"))
        pkg = Writer(os.path.join(outdir, "DRS_package.mp4"))
        log("assembling DRS_package.mp4")
        pkg.hold(cv2.imread(title), 2.5)
        # the delivery as broadcast
        lab = _label("THE DELIVERY", "stabilized broadcast replay" if rep.get("stabilized") else "broadcast replay")
        last_sample = res["samples"][-1]["f1"]
        end = min(shot[1], last_sample + 25)
        first = None
        for i, fr in read_frames(video, range(shot[0], end + 1)):
            out = pkg.frame(fr, lab)
            if first is None:
                first = out
                cv2.imwrite(os.path.join(pics, "01_delivery.jpg"), out)
        # tracking on the real footage
        lab = _label("BALL TRACKING", "reconstructed path on the broadcast footage")
        last = pkg.video(os.path.join(outdir, "tracked_footage.mp4"), lab)
        if last is not None:
            cv2.imwrite(os.path.join(pics, "02_tracking_on_footage.jpg"), last)
        # mid-flight still with the trail (impact frame of the tracked footage)
        cap = cv2.VideoCapture(os.path.join(outdir, "tracked_footage.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, res["samples"][-1]["f0"] - shot[0] + 14))
        ok, im = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(os.path.join(pics, "03_projection_on_footage.jpg"), im)
        # 3D angles
        for k, (cam, title_txt, _) in enumerate(ANGLES):
            if cam not in ang:
                continue
            lab = _label(title_txt, "slow motion · projected path in blue")
            last = pkg.video(ang[cam], lab)
            if last is not None:
                pkg.hold(last, 1.2 if cam != "auto" else 2.0, lab)
                cv2.imwrite(os.path.join(pics, f"{10 + k}_3d_{cam}.jpg"), last)
        if "pitch_map" in ang:
            pm = cv2.imread(ang["pitch_map"])
            out = pkg.hold(pm, 3.0, _label("PITCH MAP", f"pitched {rep['bounce_xy_m'][1]:.2f} m from the stumps"))
            cv2.imwrite(os.path.join(pics, "20_pitch_map.jpg"), out)
        data_card = evidence_card(rep, os.path.join(pics, "89_decision_measurements.png"))
        pkg.hold(cv2.imread(data_card), 5.0)
        pkg.hold(cv2.imread(verdict), 4.0)
        pkg.close()
        for f in ("calibration_check.jpg", "delivery.png"):
            src = os.path.join(outdir, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(pics, "30_" + f if f.startswith("cal") else "31_" + f))
        inputs_check(res, os.path.join(pics, "32_detected_inputs.jpg"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return os.path.join(outdir, "DRS_package.mp4")


def inputs_check(res, path):
    """What the system detected by itself: crease corners on the pitch and the
    front foot at impact (so a person can confirm them at a glance)."""
    from .broadcast import read_frames
    bg = res["background"].copy()
    pts = res["calibration"].get("crease_points", {})
    for n, (u, v) in pts.items():
        cv2.drawMarker(bg, (int(round(u)), int(round(v))), (0, 255, 255), cv2.MARKER_CROSS, 40, 3, cv2.LINE_AA)
        cv2.circle(bg, (int(round(u)), int(round(v))), 16, (0, 255, 255), 2, cv2.LINE_AA)
    rep = res["report"]
    foot = rep.get("front_foot_px")
    imp = res["samples"][-1]
    fr = dict(read_frames(res["video"], [imp["f0"]]))[imp["f0"]]
    if foot:
        cv2.drawMarker(fr, (int(round(foot[0])), int(round(foot[1]))), (255, 120, 0), cv2.MARKER_TILTED_CROSS, 28, 3)
    cv2.circle(fr, (int(round(imp["u"])), int(round(imp["v"]))), 14, (0, 0, 255), 2, cv2.LINE_AA)
    # zoom on the batter for the right half
    u, v = int(imp["u"]), int(imp["v"])
    x0, y0 = max(0, u - 240), max(0, v - 120)
    crop = fr[y0:y0 + 270, x0:x0 + 480]
    crop = cv2.resize(crop, (960, 540), interpolation=cv2.INTER_CUBIC)
    left = cv2.resize(bg, (960, 540), interpolation=cv2.INTER_AREA)
    out = np.hstack([left, crop])
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for x, t in ((14, f"Calibration: {res['calibration'].get('how', '')}"),
                 (974, f"Impact: ball (red circle), front foot (blue cross, {rep.get('front_foot') or 'not found'})")):
        d.text((x, 12), t, font=_font(20), fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    cv2.imwrite(path, cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
    return path
