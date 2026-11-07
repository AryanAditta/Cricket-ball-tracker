# Validation protocol

**Question BCB will ask: how do you know it's accurate?** This document is the answer. It mirrors what the ICC did when it commissioned an independent review of Hawk-Eye and Virtual Eye ([ESPNcricinfo](https://www.espncricinfo.com/print/story/553449)). That review measured two things:

- **Accuracy:** do the pitching point, the impact point and the predicted stump hit match what really happened?
- **Reliability:** on what share of deliveries does the system produce a usable track at all? The ICC's own example was that 50 good tracks out of 60 appeals is unacceptable, and 97 out of 100 is acceptable.

Every test below compares the system with **physical ground truth**: a tape measure, the stumps themselves, or a radar gun. The system is never checked against itself.

`python -m cbt validate` scores a trial and writes `validation.md` with PASS/FAIL against the targets. To see what a report looks like before any fieldwork, run:

```bash
python -m cbt validate --simulate 40 --out results/validation_demo
```

## Before every session

1. **Calibrate each camera**: `python -m cbt calibrate ...`. Accept only if the reprojection error is **≤ 1.5 px**. More usually means a mis-click or a landmark with the wrong name.
2. **Check the cameras stayed put.** At the end of the session, re-click 4 landmarks on a frame from the last delivery. If any point has moved more than 2 px, a camera was knocked; mark that session's deliveries and re-calibrate.
3. **Check sync.** Fire a flash (or clap in front of all cameras) at the start and the end. The time offsets must agree to within a third of a frame (≈3 ms at 120 fps).
4. **Record the conditions**: date, time, light, ball (red or white, new or old), camera positions, fps, shutter speed.

## Test A: pitching accuracy (the pitch map)

- **Setup:** dust the ball lightly with chalk powder, or bowl onto a dark mat taped to the pitch, so every bounce leaves a mark. Bowl 50 or more deliveries covering yorker to short, both lines, pace and spin.
- **Measure:** after each ball, tape-measure the centre of the mark:
  - `pitch_y`: distance from the line of the striker's stumps (m);
  - `pitch_x`: sideways distance from the middle-stump line (m, **+ = off side** for a right-hander).
  - Clean the mark before the next ball. Tape error is about 5 mm.
- **Pass:** 95% of balls within **5 cm**. The pitching verdict ("in line / outside leg") should agree except on balls within about 1 cm of the line.

## Test B: wickets at unguarded stumps

- **Setup:** no batter. Bowl at the stumps with the bails on. Aim many deliveries close to the edges of the stumps, because those are the hard ones.
- **Record:** `stumps_result` = `hit` if a bail is dislodged or a stump visibly moves (check the stump camera), otherwise `miss`.
- **How the system is scored:** fit each delivery with `python -m cbt fit ... --cut-y 1.5`. That drops every detection after the ball is 1.5 m from the stumps, as if a pad had stopped it there, so this tests the *projection*, not just the tracking.
- **Pass:** among balls the system calls **hitting** or **missing**, at least **98%** agree with what happened. **Umpire's call** and **inconclusive** results are counted separately: they aren't errors, but too many of them means the system isn't useful.

## Test C: projection accuracy (truncation test)

This test needs no batter, and it's the strongest check of the prediction.

- Take Test B's deliveries, which are tracked **all the way to the stump plane**. The full track tells us where each ball actually crossed the stumps.
- Then cut each track off at 1.0, 1.5, 2.0 and 2.5 m before the stumps, as if a pad had stopped it there, refit, and compare the projection with where the ball really crossed.
- Provide a manifest of tracks (`delivery,cam,track_csv`) with `--manifest` and `--cams`.
- **Pass:** 95% within **5 cm** when cut 1.5 m out. The report also shows how fast the error grows with distance, which is the physical reason the DRS **3 m rule** exists.

## Test D: release speed

Stand a radar gun behind the bowler's arm and record `speed_kmh` for each ball. **Pass:** 95% within **3 km/h**. Radar guns themselves are typically ±1–2 km/h.

## Test E: reliability in real conditions

Run the system through a full nets session and then a practice match, with batters, fielders, shadows and changing light.

- **Reliability** = the share of deliveries with a usable track (`status: ok`).
- **Pass:** at least **97%**.
- Every failure is listed in the report. Look at each one in the operator console and note why it failed (bowler hiding the ball at release, batter hiding it after the bounce, ball lost against the background, and so on).

## truth.csv format

```
delivery,pitch_x,pitch_y,stumps_result,speed_kmh,handed
b001,0.04,5.62,hit,132.5,right
b002,-0.21,3.10,miss,,right
```

Leave a column empty when it wasn't measured. Each delivery's report is `<reports folder>/<delivery>.json`; `python -m cbt fit --out reports/b001 ...` writes `reports/b001.json`.

```bash
python -m cbt validate --truth truth.csv --reports reports --manifest tracks.csv --cams cameras.json --out results/trial1
```

## Wicket probability and missing speed

The validation report groups predicted miss probabilities into bands and compares them with observed misses. For this comparison, use a stump camera or independent crossing measurement to distinguish a true miss from contact that did not dislodge a bail. Use a held-out field session to evaluate calibration. The simulated report exercises the scoring code.

Speed errors are reported with the number of radar deliveries that could be scored. Single-camera estimates can be included when available, with their source shown in the per-delivery CSV. A speed estimate fitted using an entered speed-gun reading is excluded from independent speed scoring.

## What the targets mean

The targets in `cbt/validate.py` (`TARGETS`) are a starting point for a **domestic ADRS trial**, not a published ICC standard. Professional systems quote average errors of a few millimetres on the observed path. Agree the final numbers with BCB before the trial, and publish the full validation report with the results, including the failures. Critics of ball tracking have long argued that hiding the margin of error is the real problem ([The Conversation](https://theconversation.com/out-goal-the-ball-was-in-but-could-hawk-eye-get-it-wrong-20741)), so showing it openly is a strength.
