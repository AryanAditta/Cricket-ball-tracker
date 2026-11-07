# BCB Cricket Ball Tracker

Computer vision prototype for cricket LBW review: video input, ball tracking, reconstructed trajectory, and pitching / impact / wickets decisions with uncertainty.

## Current state

The active code builds on the supplied fifth-version archive. It supports a simulator, synchronized calibrated cameras, and end-on broadcast replays. It generates annotated footage, an interactive replay, an umpire console, decision measurements and written reports. White-ball detection and small-motion stabilization are experimental.

This is a development prototype. Only one real broadcast example is supplied, and measured field validation is still pending. The project name describes its intended ADRS use; it does not establish BCB approval or certification.

## Start

Use Python 3.13 from the repository root on Windows:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 -m cbt analyse "archive/original-project/Shamim LBW.mp4" --handed left --on-field out --no-angles
```

After setup, double-click `Run Cricket Ball Tracker.cmd` for a clip picker with handedness, on-field call, ball colour and output options. Results are saved under `results/`. The console includes a **Measurements** button beside the verdict.

For PDF and rendered 3D video angles, install the optional browser runtime:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-render.txt
.\.venv\Scripts\python.exe -m playwright install chromium
```

Use `--review-only` for the console and reports without the presentation video. Use `--angle-height 480` for faster 3D export, or `--no-angles` to skip the rendered angles. The offline 3D replay works in a browser in every mode.

For manual correction, generate a click tool with `python -m cbt clicktool "path/to/clip.mp4"`, then pass the downloaded file as `--clicks clicks.json`.

```sh
python -m pytest -q tests
```

Windows CI installs Chromium and runs the complete suite on pushes and pull requests. The [import verification](docs/REPOSITORY_STATUS.md) records the original archive checks.

## Camera inputs

For broadcast replays, start with a camera behind the bowler and both ends of the pitch visible:

- `--ball white` uses bright moving objects as candidates. `--weights best.pt` uses an optional Ultralytics cricket-ball model instead. Install `ultralytics` separately and supply your own weights.
- `--stabilize --shot FIRST LAST` corrects a small pan or zoom within one continuous replay shot. Frame numbers are zero-based and inclusive. Inspect `calibration_check.jpg` and the stabilized clip before relying on the result. For manual correction, generate a click tool from `stabilized_input.mp4` and run analysis on that clip directly.
- `--speed-roi X Y WIDTH HEIGHT` uses Tesseract on PATH to suggest a speed from a numeric km/h graphic. Coordinates are pixels in the original video. Confirm the reading before passing it as `--speed-kmh`.

For two synchronized cameras, first create `cameras.json` with the capture rate and clock offset for each camera. Use one delivery per clip and trim it at impact or supply its timestamp on the common camera clock using `--impact-time`.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m cbt fit --cams cameras.json --camera-video "end_bowler=end.mp4" "side_off=side.mp4" --out results/paired/delivery
```

Use `--track` if you already have checked tracks. The synchronized workflow needs common capture timing. Edited replay angles with different slow-motion timing need separate alignment.

## Files

| Location | Contents |
|---|---|
| `cbt/` | Active Python code, HTML templates, and bundled rendering assets |
| `tests/` | Core, LBW, broadcast, and validation tests |
| [Project guide](docs/PROJECT_GUIDE.md) | Full original instructions and simulated performance claims |
| [Validation plan](VALIDATION.md) | Proposed physical ground-truth trials and targets |
| [Handoff](docs/HANDOFF.md) | Historical project context and recorded next steps |
| [Repository status](docs/REPOSITORY_STATUS.md) | Import provenance, limitations, and remaining work |
| [Original project archive](archive/README.md) | All supplied versions, footage, studies, and generated outputs |

The latest saved report and presentation video are in [`archive/original-project/5th version/`](archive/original-project/5th%20version/). The full result bundle is `Shamim_LBW_results.zip`; it includes the console, replay, measurements and pictures.

## Decision policy

The best-estimate trajectory determines the call. Wide uncertainty can add a low-confidence flag; failed or physically implausible reconstruction produces no decision. Handedness, shot offered, and the on-field decision are inputs. Check the competition's current playing conditions before operational use.

## Next work

- Validate on varied real clips and measured bounce / stump-crossing trials.
- Calibrate the reported path uncertainty against independent outcomes.
- Check white-ball and small-motion modes on varied match footage.
- Confirm the target competition's LBW playing conditions and the rights to distribute the code and archived media.

No project-wide license was included in the supplied material. Bundled third-party notices are retained under `cbt/vendor/`.
