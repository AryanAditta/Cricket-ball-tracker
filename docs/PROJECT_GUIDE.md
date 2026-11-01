# Cricket Ball Tracker (BCB / ADRS prototype)

Low-cost ball tracking to complement BCB's **Alternate DRS (ADRS)**. Under ADRS, used in domestic cricket when Hawk-Eye isn't hired, the TV umpire has only slow-motion replays and has to judge an LBW's path by eye. This system gives the TV umpire the three standard LBW calls (**pitching**, **impact**, **wickets**) from our own fixed cameras, with a confidence figure attached.

Status: **milestone 0**. Everything runs and is tested on simulated deliveries and synthetic video. It also runs end to end on **real broadcast replay footage** from one camera (see *Broadcast footage* below). No footage from our own cameras exists yet.

## How it works

```
video (each camera) ──► detect ──► 2D track ──┐
                                               ├─► 3D physics fit ──► LBW report + figure
cameras.json (calibrated pose + clock offset) ─┘
```

1. **Calibrate** (`cbt/camera.py`)
   - Lens intrinsics are measured once, from checkerboard photos.
   - Each camera's position and orientation are solved from pitch landmarks clicked in one frame: crease corners and stump tops. `python -m cbt landmarks` lists them.
2. **Detect** (`cbt/detect.py`)
   - Background subtraction picks up the ball as a small round moving blob. It needs no training, and it works because our cameras don't move.
   - A YOLO detector is an optional drop-in. It's the only option for moving or broadcast footage.
3. **Track in 2D** (`cbt/track2d.py`)
   - Per-frame detections are linked into tracks.
   - A track counts as a delivery if it is long enough, moving, and smooth on each side of a single kink (the bounce).
4. **Fit in 3D** (`cbt/fit.py`). This is the core of the system.
   - The ball's path is modelled as two constant-acceleration arcs that meet at a bounce point, and the bounce point is constrained to lie on the pitch.
   - All parameters are fitted at once, by minimising pixel error over every detection from every camera at its own timestamp.
   - Consequences:
     - Cameras only need common timestamps; they don't need frame-level sync.
     - Adding a camera just adds data.
     - The fit gives an uncertainty, which is turned into hitting / umpire's call / missing probabilities.
   - Weak physical priors (gravity, drag, speed lost at the bounce) keep the fit realistic.
5. **LBW decision** (`cbt/lbw.py`): the published DRS protocol.
   - **Pitching:** outside leg only if *more than half* the ball pitched outside the line. There's no umpire's call for pitching.
   - **Impact:** in line if at least half the ball is in line. **Umpire's call** if only 1–50% of it is. Outside if none of it is. If the impact is outside off, it only saves the batter when a shot was offered.
   - **Wickets:** the zone runs up to the **top of the bails** (the April 2021 change). **Hitting** if at least half the ball is in the zone, **umpire's call** if less, **missing** if none.
   - **Distance:** if impact is **3 m or more** from the stumps, or **2.5–3 m** with the pitch-to-impact distance under 40 cm, it's umpire's call.
   - **Decision:** the on-field decision and "shot offered" are inputs. On umpire's call the on-field decision stands and the review is retained.
   - The JavaScript copies of these rules in the replay and the console are tested against the Python on all 384 combinations.
   - **Check the distance rules and zones against the playing conditions BCB uses before match use.**

## Accuracy on simulated deliveries

Test setup:
- 80 random deliveries per configuration: 65% pace (108–148 km/h) and 35% spin (68–90 km/h).
- The simulated physics is deliberately more complex than the fitter's model: drag that depends on speed, swing, and seam/spin deviation at the bounce.
- Detection noise: 1 px, with 10% of detections missed, 2% false detections and 0.3 ms clock error.
- Cameras are calibrated the same way we would on the ground: clicked landmarks with error and a slightly wrong focal length.

| Cameras (120 fps) | Pitch point, median / 95th pct | Stumps sideways (x), median / 95th | Stumps height (z), median / 95th | Wickets call matches truth |
|---|---|---|---|---|
| 1 end-on (behind bowler) | 2.0 cm / **unstable** | 1.7 cm / unstable | 6.4 cm / unstable | 87% |
| 1 side-on | 3.3 / 7.8 cm | 26 / 71 cm | 4.7 / 13.5 cm | 64% |
| 2 side-on (both sides) | 2.3 / 5.1 cm | 2.7 / 9.0 cm | 0.6 / 2.6 cm | 92.5% |
| **2: end-on + side-on** | **1.2 / 3.2 cm** | **0.6 / 3.0 cm** | **0.6 / 2.0 cm** | **98.8%** |
| 4: both ends + both sides | 1.2 / 2.5 cm | 0.4 / 1.7 cm | 0.35 / 1.2 cm | 97.5% |

Frame rate, with the 2-camera end-on + side-on rig (stumps x, 95th percentile): 50 fps: 5.4 cm · 60 fps: 2.8 cm · 120 fps: 3.0 cm · 240 fps: 2.0 cm.

Detection noise at 2.5 px instead of 1 px: stumps x is 1.5 cm median and 5.1 cm at the 95th percentile with 2 cameras, and 0.7 cm / 3.5 cm with 4 cameras.

What this tells us:
- **One camera is not enough for LBW.** End-on alone gets the pitch point but loses the ball's depth and speed. Side-on alone can't see sideways movement.
- **Two cameras at about 90° (end-on plus square) is the minimum viable rig.**
- Four cameras mostly tighten the worst cases.
- The calls that still disagree with the truth are balls within about 1 cm of the zone edge, which is exactly what umpire's call exists for.

Full numbers are in `results/study_*.csv`.

### These numbers are optimistic

The simulation doesn't model several things real footage will have:
- the ball hidden by the bowler at release or by the batter near impact;
- shadows, crowd and sight-screen backgrounds;
- motion blur and rolling-shutter skew from phone or action cameras;
- camera shake on gantries;
- calibration drifting during the day.

The synthetic-video demo has a clean background. Treat the table as a ceiling, and treat milestone 1 as the real test.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests                       # ~2 min
python -m cbt demo --rig 2cam_end+side --seed 4 # renders video, detects, fits -> results/demo/
python -m cbt study --n 80                      # accuracy table above
```

Bowl your own delivery in the simulator. Anything you leave out is random, and `--hard` adds players, shadows, camera shake and flicker:

```bash
# 140 km/h in-swinger, good length, pitching just outside off
python -m cbt demo --kind pace --speed 140 --length 5.5 --line 0.15 --swing -1.5 --turn -0.2
# leg-spinner turning sharply, with real-world clutter
python -m cbt demo --kind spin --speed 85 --length 4 --line -0.05 --turn 1.8 --hard
```

Each run writes the following to `results/demo/`:
- **`replay.html`**: the broadcast-style 3D replay. This is the file to show people (see below);
- `*_tracked.mp4`: what each camera saw, with detections and the fitted path drawn on;
- `delivery.png`: a technical summary graphic;
- `report.json`: the system's estimate next to the ground truth.

### The replay (for presentations)

`replay.html` is a single file that works offline. Open it in Chrome or Edge. It shows:
- the pitch, with stumps and bails at both ends;
- a batter whose struck pad sits exactly where the tracker says the ball hit;
- the ball in slow motion from the release point, the bounce mark and the pitch map;
- the projected path into the stumps, with its uncertainty;
- the DRS panels (pitching, impact, wickets) and the final decision.

Controls: replay, speed (1×, ½×, ¼×), camera presets (Broadcast, Umpire, Side-on, Stumps, Pitch map), and drag to orbit. **Record video** saves the replay as a video file from the browser.

Add `--handed left` for a left-handed batter. Add `--video` to also render `replay.mp4` on the command line; this needs `pip install playwright` and `python -m playwright install chromium`, and takes a few minutes.

`python -m cbt fit ...` on real camera tracks writes the same replay (`<out>_replay.html`), labelled as tracked from camera footage instead of simulated.

`--on-field out|not-out` sets the decision under review and `--no-shot` says no shot was offered. Both can also be changed in the replay and the console.

### The operator console (for the TV umpire)

`console.html` is one screen for the review, modelled on the "Smart Replay" setup, where Hawk-Eye operators feed the TV umpire directly:

- **Synced camera views.** All cameras run on one clock, with a scrubber marked at release, bounce and impact. You can step frame by frame (← →), play at 1×, ¼× or ⅒×, and jump to the release or the impact.
- **The 3D replay** below the cameras.
- **The protocol, step by step:** pitching, impact, wickets, distance. If the ball pitched outside leg, a **fast path** banner says to give NOT OUT immediately, and the remaining steps are greyed out.
- **The inputs:** who reviewed, the on-field decision and shot offered. It shows the decision and the **review outcome** (successful, lost, or retained on umpire's call).
- **A review log.** Kept in that browser; **Export log (CSV)** saves it.

`demo` writes `console.html` automatically. For real footage, add `--camera-video end_bowler=end_bowler.mp4 side_off=side_off.mp4` to `fit`. Slow-motion phone files are handled: the video's frame rate is compared with the capture rate in `cameras.json`.

## Analyse a match video (one command)

```bash
python -m cbt analyse Shamim_LBW.mp4 --handed left --on-field out
```

Give it a match video, the batter's hand and the on-field decision. Everything else is automatic: it picks the replay shot, calibrates the camera from the creases, finds the ball, finds the batter's front foot at impact, and reconstructs the path. Everything goes into `results/<video name>/`:

| Output | What it is |
|---|---|
| **`DRS_package.mp4`** | the broadcast-style review: title → the delivery as broadcast → **ball tracking on the real footage** (path, bounce, projection, stumps zone, panels) → **3D replay in slow motion** (main view, umpire's view, side-on, at the stumps) → pitch map → **decision card** |
| **`report.pdf`** | the full report: decision, the LBW protocol step by step, every measurement with its uncertainty, pictures, how it was measured, limits |
| `pictures/` | stills of every section, the detected inputs, the calibration check, the technical figure |
| `tracked_footage.mp4` | the tracking on the footage on its own |
| `console.html`, `replay.html` | the TV umpire console and the interactive 3D replay (open in Chrome/Edge) |
| `REPORT.txt`, `report.json` | the report as text, and every number in machine-readable form |

`pictures/32_detected_inputs.jpg` shows what the system found by itself: the crease corners and the front foot. If a corner or the foot is wrong, or automatic calibration fails (creases hidden or not visible), mark them by hand with the click tool below and pass `--clicks clicks.json`. `--no-angles` skips the 3D angle videos. Rendering them is the slow part: about 10 minutes on a 2-core machine, less with more cores.

## Broadcast footage (single camera)

This works on the kind of clip that is already available today: the TV replay of an LBW from the camera behind the bowler. It doesn't need our own cameras.

```bash
python -m cbt clicktool Shamim_LBW.mp4                 # writes Shamim_LBW_clicktool.html; open it in Chrome/Edge
python -m cbt analyse Shamim_LBW.mp4 --clicks clicks.json   # or `broadcast` for the outputs without the package
```

**1. The click tool** is one HTML file that works offline. It shows the replay shot with the players removed, so the painted creases are clear.
- Click the 8 places where the popping and bowling creases meet the return creases, 4 at each end. At least 6 are needed. Use **S** to skip one that a player or the umpire hides.
- The software snaps each click to the exact line centres, then fits the camera to every painted crease line it can see.
- Then switch to **Video frame**, step to the frame where the ball hits the pad, and click the **front foot**, at the toe end of the shoe where it meets the ground.
- Set the batter's hand and the on-field decision, then **Download clicks.json**.
- The page has a zoom and a magnifier. Click carefully: see the limits below.

**2. `broadcast`** runs the whole pipeline:
- finds the replay shot (or uses the one saved in clicks.json);
- calibrates the camera (focal length unknown);
- finds the ball, tracks it, and fits a physical path;
- applies the LBW protocol.

It writes:
- **`tracked_footage.mp4`**: the original footage with the tracked path drawn on it, growing with the ball. It adds the bounce mark, the projection into the stumps, the DRS panels and the decision, and holds on the verdict.
- `console.html`: the TV umpire console, with this footage synced to the 3D replay;
- `replay.html`: the 3D replay;
- `report.json` and `delivery.png`;
- **`calibration_check.jpg`**: the pitch model drawn over the footage. **Look at this first.** The magenta lines must sit on the painted creases, and the red lines on the stumps.

**How it copes with broadcast video:**
- **Timing:** slow motion is made of repeated and blended frames, so frame times are useless. The fit is *time-free*. It uses only the order of the ball positions, gravity, a bounce on the pitch, and typical bounce behaviour (speed kept, restitution).
- **Speed:** it comes out as an estimate from how much the path curves, and is labelled as an estimate.
- **Impact:** the front-foot click places the impact along the pitch. That is the one thing a camera behind the bowler can't see well.

**Limits, measured on the Shamim clip:**

| What | Result |
|---|---|
| Calibration | camera found 78 m behind the striker and 7 m up, focal length ≈17,800 px, crease lines fitted to ≈0.8 px |
| Ball path | 21 positions, 0.7 px rms fit; pitched in line 4.9 m from the stumps, hit the pad in line 2.1 m out |
| Projection | passes **≈18 cm leg-side of middle**, i.e. clear of the umpire's-call band (15 cm) by ≈3 cm, at ±5.6 cm sideways (2σ) |
| Fully automatic run (no clicks) | crease corners found within 2.5 px of the hand-clicked ones; front foot found 4 px from the hand-clicked point; **NOT OUT**, ball passing 19.6 cm leg-side of middle, 94% of plausible paths missing |
| Repeatability | in 11 re-runs with different click noise (crease ±1.5 px, foot ±3 px), the path missed leg stump every time, by 1–4.5 cm beyond the band: **NOT OUT in all 11**. 68–93% of plausible paths missed, so some runs are flagged low confidence |

So:
- **The front-foot click matters most.** One pixel up or down moves the impact about 5 cm along the pitch, and that changes the angle into the stumps. Zoom in and click carefully.
- **Every ball gets a decision** unless the reconstruction itself fails. On 120 simulated single-camera deliveries (30 fps, edited slow motion, foot click off by up to ±15 cm), 117 were decided and 114 of those matched the truth (97%). All 3 misses were hitting or missing called as umpire's call, or the reverse, for balls within about 3 cm of the band; none was hitting called missing. The other 3 failed to reconstruct (NO DECISION).
- **One camera behind the bowler sees sideways movement well but depth poorly**, so the projection is wider than Hawk-Eye's. A second, side-on camera tightens it (see the accuracy table above).
- The shot must be static (no zooming or panning during the delivery), from behind the bowler, with the creases at both ends visible.
- **Red ball only for now.** The detector looks for a dark, warm-coloured ball. White-ball footage needs a brightness-based detector, which isn't written yet.

## Validation

`VALIDATION.md` is the protocol for proving accuracy to BCB, modelled on the ICC's independent review of ball tracking: accuracy against tape-measured bounce marks, hits and misses at unguarded stumps, a track-truncation test of the projection, radar speeds, and reliability in match conditions. `python -m cbt validate` scores a trial with PASS/FAIL against the targets. `python -m cbt validate --simulate 40` shows what a report looks like.

Like Hawk-Eye, the call is made on the best-estimate path, and the umpire's-call band is the margin for error. The spread of plausible paths is shown with every decision, and flagged *low confidence* when it is wide. Only a failed reconstruction (no ball path found, or a path that isn't physical) gives NO DECISION.

On real footage:

```bash
python -m cbt calibrate --name end_bowler --intrinsics end_bowler_K.json --points end_bowler_clicks.csv
python -m cbt detect end_bowler.mp4 --out det/end   # add --weights best.pt for YOLO
python -m cbt fit --track end_bowler=det/end.track0.csv side_off=det/side.track0.csv --out ball1 \
    --on-field not-out --camera-video end_bowler=end_bowler.mp4 side_off=side_off.mp4
```

## Plan

| Milestone | What | How we know it worked |
|---|---|---|
| **M0 ✔** | Simulator, physics fitter, video pipeline, accuracy study | This README |
| **M1: nets trial** | 2 cameras (end-on + square) at nets or an academy ground. Calibration procedure, clock sync, real footage. Label 2–3k frames and fine-tune YOLO. | Chalk-dusted ball marks on the pitch measured with a tape, compared with our pitch point (target ≤ 3 cm) |
| **M2: projection check** | Bowl at unguarded stumps | Predicted hit/miss vs what actually happened, and miss distance vs a tape |
| **M3: operator tool** | One screen for the TV umpire: pick the delivery, get the graphic within 30 s | Dry run with a BCB match official |
| **M4: match trial** | Run alongside ADRS in a domestic match | Agreement with on-field and TV-umpire decisions; turnaround time |

## Hardware guidance (to confirm in M1)

- **Placement:**
  - Camera 1: end-on, behind the bowler's arm, raised about 5–8 m on the sight-screen gantry.
  - Camera 2: square of the wicket, about 40 m out, raised about 8 m.
  - Later, add the mirror pair at the other end and other side, so the rig works from both ends.
- **Cameras:**
  - 1080p at 120 fps or more, global shutter if the budget allows, fixed or lockable zoom, short exposure (about 1/1000 s).
  - Phones and action cameras are fine for the M1 prototype. Expect rolling-shutter error.
- **Clock sync:** a hardware trigger or PTP on machine-vision cameras. For a prototype, use a flash or LED visible to both cameras and compute the offset. The fitter only needs common timestamps.
- **Compute:** any laptop. A fit takes 0.4–3 s on 2 CPU cores.

## Open questions for BCB

- Their baseline or requirements document: what accuracy and turnaround do they expect?
- Which grounds, and can we mount cameras permanently there?
- Is the output just the TV-umpire graphic, or also a broadcast feed?
- Which playing conditions should the zone rules follow?

## Sources and prior work we reviewed

- [kushagra3204/Cricket-Ball-Trajectory-Prediction](https://github.com/kushagra3204/Cricket-Ball-Trajectory-Prediction): YOLOv8 ball detector and a 1,778-image Kaggle dataset. 2D only. No licence file, so we haven't copied anything from it.
- [HarshSingh18/CV_Project](https://github.com/HarshSingh18/CV_Project): colour-threshold and YOLO ball detection on red-ball footage.
- [shoaib1522/Cricket-Ball-Tracker](https://github.com/shoaib1522/Cricket-Ball-Tracker) (MIT): YOLOv5 + 2D Kalman filter + ground-plane bounce mapping. Our pipeline has the same overall structure, but the fit is 3D and physics-based.
- Kamble, Keskar & Bhurchandi, *Ball tracking in sports: a survey*, Artificial Intelligence Review 52 (2019).

LBW protocol and DRS practice:
- [cricket.com.au: umpire's call explained (zones, 50% rules, 3 m and 2.5 m/40 cm rules)](https://www.cricket.com.au/news/3319658/umpires-call-explained-why-is-that-not-out-lbw-cricket-law-drs-decision-review-system-pitching-outside-leg-impact-in-line)
- [ESPNcricinfo: wicket zone extended to the top of the bails (April 2021)](https://www.espncricinfo.com/print/story/1257516)
- [Wisden: the 3 m rule](https://wisden.com/series/indian-premier-league-2023/cricket-news/explained-3m-three-metre-drs-lbw-impact-rule-rohit-sharma-mi-rcb-review)
- [ESPNcricinfo: IPL Smart Replay System](https://aus.cricinfo.com/story/ipl-2024-introduces-smart-replay-system-for-quicker-more-accurate-reviews-1425406)
- [ESPNcricinfo: ICC's independent review of ball tracking (accuracy and reliability)](https://www.espncricinfo.com/print/story/553449)
