# BCB Cricket Ball Tracker

Computer vision prototype for cricket LBW review: video input, ball tracking, reconstructed trajectory, and pitching / impact / wickets decisions with uncertainty.

## Current state

The active code is imported unchanged from the supplied fifth-version source ZIP. It supports a simulator and calibrated multi-camera pipeline, plus static end-on **red-ball broadcast replays** from one camera. It generates a review video, interactive replay and operator console, pictures, and PDF / text / JSON reports.

This is a development prototype. Only one real broadcast example is supplied, and measured field validation is still pending. The project name describes its intended ADRS use; it does not establish BCB approval or certification.

## Start

Use Python 3.11 from the repository root:

```sh
python -m pip install -r requirements.txt
python -m pip install playwright
python -m playwright install chromium
python -m cbt analyse "archive/original-project/Shamim LBW.mp4" --handed left --on-field out
```

Outputs are written to `results/<video name>/`. Add `--no-angles` to skip the additional 3D angle videos. The camera must be static, behind the bowler, with both ends of the pitch visible. White-ball detection is not implemented.

For manual correction, generate a click tool with `python -m cbt clicktool "path/to/clip.mp4"`, then pass the downloaded file as `--clicks clicks.json`.

```sh
python -m pytest -q tests
```

The handoff records 46 passing tests from the previous environment. See [verification notes](docs/REPOSITORY_STATUS.md) for the checks actually performed during this import.

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

- Show decision measurements beside their thresholds in the video, console, and report.
- Validate on varied real clips and measured bounce / stump-crossing trials.
- Improve render time and the side-on view.
- Add white-ball detection and a second-camera workflow.

No project-wide license was included in the supplied material. Bundled third-party notices are retained under `cbt/vendor/`.
