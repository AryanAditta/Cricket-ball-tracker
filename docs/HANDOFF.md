# BCB Cricket Ball Tracker (ADRS / LBW) — Full Handoff

Owner: Mashrafi (ECE, KUET, Kushtia). Project taken over from a friend who won it from BCB (Bangladesh Cricket Board). Goal: a computer-vision system that complements ADRS (Alternate DRS) for LBW review, working from a phone camera / simulator and, now, **real single-camera broadcast replays**.

Status at handoff: **all 46 tests pass** (`python -m pytest -q`, ~75 s). Last user question (which parameters feed the decision) was **answered in chat but not yet implemented in the outputs** — see "Open items".

---

## 1. Chat timeline (what was asked, what was done)

1. Voice-message / .ogg question, garbled transcript about the BCB ball-tracking project, then links (LinkedIn short link, kushagra3204 GitHub, QverLabs, ball-tracking survey, HarshSingh18 README). Asked to go through them (Chrome connected).
2. Scoping answers: (1) research online + BCB content, (2) own camera for now, (3) whichever approach best fits the game, (4) "start work, will figure out in the meantime".
3. "Can I test it?" → simulator/video testing → step-by-step guide.
4. Feedback on glitches/animation (screenshots): wanted batsman with bat, stumps, decisions, BCB-presentable replay → built the 3D replay and console.
5. "Research what can be improved to be like real-life cricket tech" → research done; user chose **Tier 1**: correct LBW protocol, operator console, validation. Implemented.
6. Uploaded `Shamim_LBW.mp4`: "make the full project work on these types of videos" → built the single-camera broadcast pipeline.
7. "One camera → ~1 ball in 5 NO DECISION. Not this." → changed to **decide every ball** on the best-estimate path; uncertainty is shown, not used to withhold.
8. "Where are the images? I need the full reports." → regenerated screenshots, REPORT.txt, results zip.
9. "Give videos as input, get output video like a cricket broadcast: slow motion, projected path, hit/miss, other angles, with pictures, analysis and report." → one command `python -m cbt analyse`.
10. "How to see which parameters are considered (height, speed…)?" → explained (below); not yet put into outputs.
11. Now: this handoff.

---

## 2. How to run

```
pip install -r requirements.txt        # includes pillow; playwright+chromium needed for 3D angles
python -m cbt analyse match.mp4 --handed left --on-field out [--bowler fast|spin...] [--speed-kmh N] [--no-angles]
python -m cbt broadcast match.mp4 [--clicks clicks.json] ...   # lower-level; clicks optional now
python -m cbt clicktool match.mp4                              # offline click page: 8 crease points + front foot
```
- Full `analyse` ≈ 15 min on 2 cores (3D render via swiftshader is the slow part); `--no-angles` ≈ 3 min.
- Calibration is automatic; if it's poor (rms > 3 px) it errors and tells you to use `clicktool`.
- Outputs per clip: `DRS_package.mp4`, `tracked_footage.mp4`, `console.html`, `replay.html`, `report.pdf/html`, `REPORT.txt`, `report.json`, `track.json`, `camera.json`, `delivery.png`, `calibration_check.jpg`, `pictures/` (00_title, 01_delivery, 02_tracking_on_footage, 03_projection_on_footage, 10–13_3d_{auto,umpire,side,stumps}, 20_pitch_map, 30_calibration_check, 31_delivery, 32_detected_inputs, 90_decision).

## 3. Locations

- Project: `/home/claude/cricket-ball-tracker` (zip: `/home/claude/cricket-ball-tracker.zip`, 21 Oct — may be older than later edits; re-zip before sending).
- Scratch: `/tmp/claude-0/-home-claude/eef52db5-10a1-5697-ae0c-9233db0020e5/scratchpad/`
  - `Shamim_LBW.mp4` (input), `shamim/`, `pkg/Shamim_LBW/` (final outputs), `pkg/Shamim_LBW_results.zip` (25.9 MB).
- Earlier demo pack: `/home/claude/BCB_demo_pack.zip`.
- Full transcript: `/root/.claude/projects/-home-claude/eef52db5-10a1-5697-ae0c-9233db0020e5.jsonl`.
- Note: scratch is session-local and may vanish; copy results out.

## 4. Code map (`cbt/`)

| File | Role |
|---|---|
| `pitch.py` | Constants: stump half-width 0.1143 m, height 0.711 m (+0.0127 bails), ball radius 0.036 m |
| `camera.py`, `fit.py`, `detect.py`, `track2d.py`, `sim.py`, `rigs.py`, `synthvideo.py` | Multi-camera/simulator pipeline (original work) |
| `lbw.py` | LBW protocol + `build_report` |
| `monofit.py` | Single-camera, time-free trajectory fit + Monte-Carlo; `fit_mono`, `reconstruct`, `problems()`, `report()` |
| `broadcast.py` | Scan video, pick replay shot, background, calibration (`calibrate_from_clicks`, `auto_crease_points`, `white_line_mask`), ball candidates, DP tracker, `auto_front_foot` |
| `broadcast_run.py` | `process()` orchestrates broadcast clip → fit → report → outputs |
| `footage.py` | Overlays on original footage (panels, banner, confidence line), `calibration_check`, `final_banner` |
| `package.py` | `DRS_package.mp4` + pictures: title/decision cards, 3D angle renders, pitch map, `inputs_check` |
| `report.py` | `REPORT.txt`, `report.html`, `report.pdf` |
| `replay.py` + `replay_template.html` | three.js 3D replay (headless Chromium for video) |
| `console.py` + `console_template.html` | Third-umpire console |
| `clicktool.py` + `clicktool_template.html` | Offline click page |
| `viz.py`, `annotate.py` | Figures; null-safe speed |
| `cli.py` | Commands: `analyse`/`analyze`, `broadcast`, `clicktool`, plus original ones |
| `tests/test_broadcast.py` | 7 broadcast tests (46 total in suite) |

(Several of these files changed on disk after my last read — likely deliberate; I did not revert anything.)

## 5. How the decision is made

**Single-camera fit (`monofit.py`)**: unknowns = bounce (x, y), two polynomial arcs, and a latent along-pitch position `s` per observation (no timing, because broadcast slow-mo is held/blended frames). Priors: gravity↔speed link, retained speed 0.87, restitution 0.45, small deviation/swing, first sample near release, impact near front-foot toe (0.10 ± 0.20 m), optional bowler-type / speed-gun speed. Multi-start (release 17 or 18.5 m; speeds 24 or 32 m/s; even or arc-length spreading) avoids local minima. Monte-Carlo samples of the stump crossing give hitting / umpire's call / missing % and ± bands.

**LBW protocol (`lbw.py`)**
- Pitching: outside leg only if > 50% of ball (centre) beyond leg-stump line.
- Impact: in line if ≥ half the ball in line; umpire's call if 1–50%; outside if none. Outside off protects the batter only if a shot was offered.
- Wickets: zone ±0.1143 m wide, to top of bails (0.7237 m). Centre inside → hitting; ball partly touches → umpire's call; none → missing.
- Distance rule: impact ≥ 3 m from stumps, or 2.5–3 m with pitch→impact < 0.40 m → umpire's call.
- Inputs: handedness, on-field decision, shot offered.

**Policy**: decide every ball on the best-estimate path; `low_confidence` flag if σ > 0.05 m and agreement < 0.85 (informational). Inconclusive (on-field decision stands) only for hard problems: failed projection, bounce off the pitch, < 2 positions after bounce, implausible impact, unphysical stump crossing (z outside 0–2 m or |x| > 2 m), rms > 3 px.

**Speed** is only an estimate from curvature, displayed only if σ_v/v < 0.08. It, and height at impact, are **not** decision criteria.

## 6. Results

Shamim_LBW.mp4, fully automatic (no clicks):
- Calibration rms 1.75 px, focal ≈ 17,810 px, camera ≈ (−0.32, 77.69, 6.98) m.
- Bounce 4.92 m from stumps at x −0.111 m (in line, not outside leg).
- Impact 2.30 m out, x +0.055 m (in line), height 0.46 m; pitch→impact 2.62 m.
- Stumps crossing x +0.196 m (leg side), height 0.733 m; ±6.2 / ±23.0 cm (2σ).
- Paths: hitting 1% · umpire's call 5% · missing 94% → **NOT OUT (missing)**.
- Speed not determined (earlier clicked run: ≈ 94 ± 4 km/h). Auto corners within 2.6 px of hand-clicked; auto front foot within ~4 px.
- Repeatability: 11 jittered runs → NOT OUT in all 11 (some low-confidence).
- Simulated single camera: 117/120 decided, 114 correct (97 %); all 3 errors near the umpire's-call band; no hit/miss swaps; 3 failed to reconstruct.

## 7. Known limitations (told to the user)
- Camera must be behind the bowler (end-on), no zoom/pan during the delivery.
- Red ball only; white-ball detection not done.
- Depth accuracy looser than sideways accuracy.
- Side-on 3D view makes the path look thin.
- 3D package render is slow on 2 cores.
- Single-camera speed is an estimate only.

## 8. Mistakes fixed along the way
Local minima under click jitter → multi-start; unphysical projection still giving a verdict → `problems()` check; NO DECISION on ~1/5 balls → decide-and-flag; `None` speed crashing templates/viz/annotate → null handling; overlapping/clipped text → PIL text + repositioned labels; empty speed box hidden; auto-crease test failure → wider top-hat kernel (`max(15, W/64|1)`); `pitch` camera preset doesn't exist → use `top`.

## 9. Open items / next steps
1. **Decision-parameters panel** (user's last question). Proposal: add a "DECISION DATA" panel to `DRS_package.mp4`, `console.html` and `report.pdf` showing each measurement next to its limit, green/red: bounce x vs leg line; impact x vs ±11.4 cm; stump-crossing x and **height vs 72.4 cm bail line**; ball radius; impact height; impact→stumps and pitch→impact vs 3 m / 2.5 m / 0.40 m; shot offered, on-field call, handedness; speed estimate (marked non-decisive). Currently these are only in `report.pdf` p.2, `REPORT.txt`, `report.json`, and partly the console. Keep change minimal (stored preference: minimal, targeted edits).
2. Faster package render (3D via swiftshader).
3. Better side-on 3D view for a thin path.
4. White-ball detection; second-camera angle.
5. Validate on more real clips (only one real clip so far) and with real BCB footage / ground truth.
6. Re-zip the project and results after any change.

Preference on file: minimal, targeted code changes.
