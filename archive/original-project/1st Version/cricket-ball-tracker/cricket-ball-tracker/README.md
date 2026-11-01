# Cricket Ball Tracker (BCB / ADRS prototype)

Low-cost ball tracking to complement BCB's **Alternate DRS (ADRS)**. Under ADRS, used in domestic cricket when Hawk-Eye isn't hired, the TV umpire has only slow-motion replays and has to judge an LBW's path by eye. This system gives the TV umpire the three standard LBW calls (**pitching**, **impact**, **wickets**) from our own fixed cameras, with a confidence figure attached.

Status: **milestone 0**. Everything below runs and is tested, but only on simulated deliveries and synthetic video. No real footage from our cameras exists yet.

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
5. **LBW call** (`cbt/lbw.py`)
   - Calls pitching, impact and wickets with ball-width margins.
   - **The zone sizes must be checked against the current ICC/BCB playing conditions.**

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
python -m pytest -q tests                       # ~40 s
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
- `*_tracked.mp4`: what each camera saw, with detections and the fitted path drawn on;
- `delivery.png`: the DRS-style graphic;
- `report.json`: the system's estimate next to the ground truth.

When the projection to the stumps is too uncertain (more than ±10 cm at 2σ), the wickets call is **inconclusive** rather than a guess.

On real footage:

```bash
python -m cbt calibrate --name end_bowler --intrinsics end_bowler_K.json --points end_bowler_clicks.csv
python -m cbt detect end_bowler.mp4 --out det/end   # add --weights best.pt for YOLO
python -m cbt fit --track end_bowler=det/end.track0.csv side_off=det/side.track0.csv --out ball1
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
