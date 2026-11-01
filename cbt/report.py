"""Written report for a reviewed delivery: REPORT.txt, report.html and
report.pdf (the PDF is printed from the HTML by headless Chromium)."""
import datetime
import html as H
import os

from .footage import final_banner, _verdict_colour

HEX = dict(red="#d8232a", green="#1f9d55", amber="#e8961a", grey="#6b7785")

RULES = dict(
    pitching="Outside leg only if more than half the ball pitched outside the line of leg stump.",
    impact="In line if at least half the ball is in line; umpire's call if 1–50%; outside if none.",
    wickets="Zone = outer edges of the stumps up to the top of the bails. Hitting if the ball's centre is "
            "inside; umpire's call if only part of the ball is; missing if none of it is.",
    distance="Impact 3 m or more from the stumps, or 2.5–3 m with pitch→impact under 40 cm → umpire's call.",
)


def _fmt_m(v, nd=2):
    return f"{v:.{nd}f} m"


def text_report(rep):
    from .broadcast_run import summary
    c = rep.get("calibration", {})
    sp = rep.get("speed_estimate_kmh") or rep.get("release_speed_kmh")
    return f"""LBW BALL-TRACKING REPORT
Clip: {rep.get('source', '')} (replay shot, frames {rep.get('shot_frames', ['?', '?'])[0]}-{rep.get('shot_frames', ['?', '?'])[1]})
Method: {rep.get('method', '')}
Batter: {rep['handed']}-handed   On-field decision: {rep['on_field'].upper()}   Shot offered: {'yes' if rep['shot_offered'] else 'no'}

{summary(rep)}

PROTOCOL
  1 Pitching : {rep['pitching'].upper():12s} bounce at x {rep['bounce_xy_m'][0]:+.3f} m, {rep['bounce_xy_m'][1]:.2f} m from the stumps
  2 Impact   : {rep['impact'].upper():12s} at x {rep['impact_xyz_m'][0]:+.3f} m, {rep['impact_xyz_m'][1]:.2f} m out, {rep['impact_xyz_m'][2]:.2f} m high
  3 Wickets  : {rep['wickets'].upper():12s} crossing the stumps at x {rep['stumps_xz_m'][0]:+.3f} m, height {rep['stumps_xz_m'][1]:.3f} m
  4 Distance : {rep['distance_rule'] or 'no rule applies'} (impact->stumps {rep['impact_to_stumps_m']:.2f} m, pitch->impact {rep['pitch_to_impact_m']:.2f} m)

INPUTS
  calibration: {rep.get('calibration_source', '')}; rms {c.get('rms_px')} px, focal {c.get('focal_px')} px, camera at {c.get('camera_position_m')} m
  front foot at impact: {rep.get('front_foot') or 'not used'} {rep.get('front_foot_px') or ''}

TRACKING
  {rep['n_inliers']}/{rep['n_obs']} ball positions used, {rep['n_after_bounce']} after the bounce, fit rms {rep['rms_px']} px
  speed {'≈%.0f km/h (estimate from path curvature)' % sp if sp else 'not determined'}
"""


def build(res, outdir, log=print):
    rep = res["report"]
    with open(os.path.join(outdir, "REPORT.txt"), "w", encoding="utf-8") as f:
        f.write(text_report(rep))
    path = os.path.join(outdir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_html(rep, outdir))
    pdf = os.path.join(outdir, "report.pdf")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page()
            pg.goto("file://" + os.path.abspath(path))
            pg.wait_for_load_state("networkidle")
            pg.pdf(path=pdf, format="A4", print_background=True,
                   margin=dict(top="14mm", bottom="14mm", left="12mm", right="12mm"))
            b.close()
    except ImportError:
        log("report.pdf skipped (needs playwright); report.html has the same content")
        pdf = None
    return path, pdf


def _pic(outdir, name, caption):
    p = os.path.join("pictures", name)
    if not os.path.exists(os.path.join(outdir, p)):
        return ""
    return f'<figure><img src="{p}"><figcaption>{H.escape(caption)}</figcaption></figure>'


def _html(rep, outdir):
    big, col, small = final_banner(rep)
    c = rep["wickets_confidence"]
    cal = rep.get("calibration", {})
    sp = rep.get("speed_estimate_kmh") or rep.get("release_speed_kmh")
    bs = rep.get("bounce_sigma_m", [0, 0])
    ss = rep["stumps_sigma_m"]

    def chip(field, v):
        return (f'<span class="chip" style="background:{HEX[_verdict_colour(field, v)]}">'
                f'{H.escape(v.upper().replace("UMPIRES", "UMPIRE’S"))}</span>')
    prot = [
        ("1", "Pitching", chip("pitching", rep["pitching"]),
         f"Bounced {_fmt_m(rep['bounce_xy_m'][1])} from the stumps, {rep['bounce_xy_m'][0] * 100:+.0f} cm across",
         RULES["pitching"]),
        ("2", "Impact", chip("impact", rep["impact"]),
         f"Hit the pad {_fmt_m(rep['impact_to_stumps_m'])} from the stumps, {rep['impact_xyz_m'][0] * 100:+.0f} cm "
         f"across, {rep['impact_xyz_m'][2] * 100:.0f} cm high", RULES["impact"]),
        ("3", "Wickets", chip("wickets", rep["wickets"]),
         f"Projected to cross the stumps {rep['stumps_xz_m'][0] * 100:+.1f} cm across, "
         f"{rep['stumps_xz_m'][1] * 100:.0f} cm high", RULES["wickets"]),
        ("4", "Distance", f'<span class="chip" style="background:{HEX["grey"]}">'
                          f'{H.escape(rep["distance_rule"] or "NO RULE")}</span>',
         f"Impact→stumps {_fmt_m(rep['impact_to_stumps_m'])}, pitch→impact {_fmt_m(rep['pitch_to_impact_m'])}",
         RULES["distance"]),
    ]
    rows = "".join(f"<tr><td class=n>{a}</td><td><b>{b}</b></td><td>{c_}</td><td>{d}<div class=rule>{e}</div></td></tr>"
                   for a, b, c_, d, e in prot)
    meas = [
        ("Bounce point (across, from stumps)", f"{rep['bounce_xy_m'][0]:+.3f} m, {rep['bounce_xy_m'][1]:.2f} m",
         f"±{200 * bs[0]:.1f} cm, ±{200 * bs[1]:.0f} cm"),
        ("Impact (across, from stumps, height)",
         f"{rep['impact_xyz_m'][0]:+.3f} m, {rep['impact_xyz_m'][1]:.2f} m, {rep['impact_xyz_m'][2]:.2f} m", ""),
        ("At the stumps (across, height)", f"{rep['stumps_xz_m'][0]:+.3f} m, {rep['stumps_xz_m'][1]:.3f} m",
         f"±{200 * ss[0]:.1f} cm, ±{200 * ss[1]:.1f} cm"),
        ("Plausible paths", f"hitting {100 * c['hitting']:.0f}% · umpire's call {100 * c['umpires call']:.0f}% · "
                            f"missing {100 * c['missing']:.0f}%", "from the measurement uncertainty"),
        ("Speed", f"≈{sp:.0f} km/h" if sp else "not determined",
         f"±{rep.get('speed_sigma_kmh', 0):.0f} km/h, from path curvature" if sp else ""),
        ("Ball positions used", f"{rep['n_inliers']} of {rep['n_obs']} ({rep['n_after_bounce']} after the bounce)",
         f"fit {rep['rms_px']} px rms"),
        ("Camera calibration", f"{rep.get('calibration_source', '')}",
         f"{cal.get('rms_px')} px rms · camera {cal.get('camera_position_m')} m"),
        ("Front foot at impact", f"{rep.get('front_foot') or 'not used'}",
         f"pixel {[round(v) for v in rep['front_foot_px']]}" if rep.get("front_foot_px") else ""),
    ]
    mrows = "".join(f"<tr><td>{H.escape(a)}</td><td><b>{H.escape(b)}</b></td><td class=mute>{H.escape(c_)}</td></tr>"
                    for a, b, c_ in meas)
    low = ('<p class="warn"><b>Low confidence.</b> The spread of plausible paths is wide for this ball, so the call '
           'rests on the best-estimate path. A second camera angle would narrow it.</p>') if rep.get("low_confidence") else ""
    when = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>LBW report</title>
<style>
@page {{ size: A4; }}
body {{ font: 11pt/1.45 "DejaVu Sans", Arial, sans-serif; color: #18232f; margin: 0; }}
h1 {{ font-size: 20pt; margin: 0; letter-spacing: .02em; }}
h2 {{ font-size: 12.5pt; margin: 22px 0 8px; padding-bottom: 4px; border-bottom: 2px solid #e3e8ee; }}
.head {{ display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 4px solid #0a1624; padding-bottom: 8px; }}
.meta {{ color: #5b6b7b; font-size: 9.5pt; text-align: right; }}
.dec {{ margin: 16px 0; border-radius: 10px; color: #fff; padding: 16px 20px; background: {HEX[col]}; }}
.dec .big {{ font-size: 30pt; font-weight: 800; line-height: 1.1; }}
.dec .why {{ font-size: 11pt; opacity: .95; }}
table {{ width: 100%; border-collapse: collapse; font-size: 10pt; }}
td {{ padding: 7px 8px; border-bottom: 1px solid #e3e8ee; vertical-align: top; }}
td.n {{ width: 18px; color: #8896a5; font-weight: 700; }}
.chip {{ display: inline-block; color: #fff; font-weight: 700; font-size: 9pt; padding: 3px 9px; border-radius: 5px; white-space: nowrap; }}
.rule {{ color: #7a8896; font-size: 8.5pt; margin-top: 3px; }}
.mute {{ color: #6b7785; }}
.warn {{ background: #fff4e0; border-left: 4px solid #e8961a; padding: 8px 12px; font-size: 10pt; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
figure {{ margin: 0; break-inside: avoid; }}
figure img {{ width: 100%; border-radius: 6px; display: block; }}
figcaption {{ font-size: 8.5pt; color: #5b6b7b; margin: 3px 0 6px; }}
.full {{ grid-column: 1 / -1; }}
.pb {{ break-before: page; }}
ul {{ padding-left: 18px; font-size: 10pt; }}
</style></head><body>
<div class="head"><div><h1>LBW BALL-TRACKING REPORT</h1>
<div class="mute">{H.escape(rep.get('source', ''))} · {H.escape(rep['handed'])}-handed batter · on-field {H.escape(rep['on_field'].upper())}
 · shot {'offered' if rep['shot_offered'] else 'not offered'}</div></div>
<div class="meta">Ball tracking · ADRS<br>{when}</div></div>

<div class="dec"><div class="big">{H.escape(big)}</div><div class="why">{H.escape(small)}</div></div>
{low}
<h2>LBW protocol</h2>
<table>{rows}</table>

<div class="grid" style="margin-top:14px">
{_pic(outdir, "02_tracking_on_footage.jpg", "Ball tracking drawn on the broadcast footage: path (red), bounce, projection (blue), stumps zone")}
{_pic(outdir, "11_3d_umpire.jpg", "3D replay · umpire's view")}
</div>

<h2 class="pb">Measurements</h2>
<table>{mrows}</table>

<h2>3D replay</h2>
<div class="grid">
{_pic(outdir, "10_3d_auto.jpg", "Main view at the decision")}
{_pic(outdir, "13_3d_stumps.jpg", "At the stumps: the projected path against the wicket zone")}
{_pic(outdir, "12_3d_side.jpg", "Side-on: height of the ball through bounce, impact and the stumps")}
{_pic(outdir, "20_pitch_map.jpg", "Pitch map: where the ball bounced")}
</div>

<h2 class="pb">How it was measured</h2>
<div class="grid">
<div class="full">{_pic(outdir, "32_detected_inputs.jpg", "What the system found by itself: the crease corners used to calibrate the camera (left), and the ball and front foot at impact (right)")}</div>
{_pic(outdir, "30_calibration_check.jpg", "Calibration check: the pitch model (magenta) and stumps (red) drawn over the footage")}
{_pic(outdir, "31_delivery.png", "Technical view: path from above, from the side and end-on, with the projection and its uncertainty")}
</div>
<ul>
<li><b>Camera.</b> The broadcast camera's position, direction and zoom are solved from the painted creases at both ends.</li>
<li><b>Ball.</b> The ball is found in every frame of the replay as a small dark, red-brown blob that moves along a smooth path; frames repeated by the slow motion are merged.</li>
<li><b>Path.</b> A physical path is fitted: two arcs under gravity joined by a bounce on the pitch. Broadcast slow motion has no usable timing, so only the order of the ball positions is used, and speed comes out as an estimate.</li>
<li><b>Impact.</b> The front foot's position at impact fixes how far down the pitch the ball hit the pad.</li>
<li><b>Decision.</b> The LBW protocol is applied to the best-estimate path. The percentages show how the call could change within the measurement uncertainty.</li>
</ul>
<h2>Limits</h2>
<ul>
<li>One camera behind the bowler measures sideways movement well but distance down the pitch less well, so its projection is wider than a multi-camera system's.</li>
<li>The footage must be from behind the bowler with no zoom or pan during the delivery, and a red ball.</li>
<li>Check the playing conditions (distance rules, wicket zone) against the competition's own before match use.</li>
</ul>
</body></html>"""
