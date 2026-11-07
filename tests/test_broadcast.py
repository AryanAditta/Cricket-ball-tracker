"""Single-camera broadcast path: calibration from clicks on a pitch image,
the time-free fit, and the outputs (no real footage needed)."""
import numpy as np
import pytest
import cv2

from cbt import sim, lbw
from cbt.camera import Camera
from cbt import broadcast as bc
from cbt.monofit import fit_mono


def broadcast_cam():
    """Like a TV end-on camera: ~78 m behind the striker, 7 m up, long lens."""
    return Camera.look_at((-0.3, 77.8, 7.0), (0, 9.5, 0.4),
                          field_width=1920 / 17850 * np.linalg.norm([0.3, 68.3, 6.6]),
                          size=(1920, 1080), name="tv")


def synthetic_pitch_image(cam):
    """Tan pitch on green grass with the painted creases (5 cm wide lines)."""
    img = np.full((1080, 1920, 3), (60, 120, 70), np.uint8)
    corners = cam.project(np.array([[1.5, -1, 0], [-1.5, -1, 0], [-1.5, 21.1, 0], [1.5, 21.1, 0]]))
    cv2.fillConvexPoly(img, np.round(corners).astype(np.int32), (120, 170, 200))
    for A, B in bc._model_lines():
        n = 60
        P = np.linspace(A, B, n)
        for half in np.linspace(-0.025, 0.025, 6):          # thickness across the line
            d = B - A
            nrm = np.array([-d[1], d[0], 0.0]) / np.linalg.norm(d[:2])
            uv = cam.project(P + half * nrm)
            cv2.polylines(img, [np.round(uv * 4).astype(np.int32)], False, (245, 245, 245), 1,
                          cv2.LINE_AA, shift=2)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_calibration_from_noisy_clicks():
    cam = broadcast_cam()
    img = synthetic_pitch_image(cam)
    rng = np.random.default_rng(1)
    clicks = {n: (cam.project(np.array([p]))[0] + rng.normal(0, 4, 2)).tolist()
              for n, p in bc.CREASE_POINTS.items()}
    est, info = bc.calibrate_from_clicks(img, clicks)
    # the camera is found to within a couple of metres along the line of sight (that is
    # what an unknown focal length allows) and the pitch reprojects to ~a pixel
    assert np.linalg.norm(est.center - cam.center) < 3.0
    assert abs(est.K[0, 0] / cam.K[0, 0] - 1) < 0.05
    test = np.array([[x, y, z] for x in (-0.11, 0.11) for y in (0.0, 2.0, 20.12) for z in (0.0, 0.7)])
    err = np.linalg.norm(est.project(test) - cam.project(test), axis=1)
    assert err.max() < 3.0, err


def test_monofit_recovers_stumps_crossing():
    """Irregular, untimed samples from one TV camera + a front-foot hint."""
    cam = broadcast_cam()
    rng = np.random.default_rng(7)
    good, n = 0, 0
    errs = []
    for _ in range(8):
        d = None
        while d is None:
            d = sim.random_delivery(rng)
        t = [rng.uniform(0, 0.03)]
        while t[-1] < d.t_impact:
            t.append(t[-1] + rng.choice([1, 1, 2]) / 30.0)
        t = np.array([x for x in t if x <= d.t_impact] + [d.t_impact])
        uv = cam.project(d.at(t)) + rng.normal(0, 1.0, (len(t), 2))
        f = fit_mono(uv, cam, foot_y=d.impact[1] - 0.10 + rng.normal(0, 0.1))
        rep = f.report(n_mc=200)
        st = f.stumps()
        errs.append(abs(st[0] - d.stumps[0]))
        if rep["wickets"] != "inconclusive":
            n += 1
            good += rep["wickets"] == lbw.wickets(*d.stumps)
    assert np.median(errs) < 0.05, errs        # lateral error at the stumps, metres
    assert n == 8 and good >= 7        # always decided; at most a boundary case differs


def test_progress_interpolates_between_samples():
    from cbt.footage import _progress
    samples = [dict(f0=10, f1=12), dict(f0=16, f1=16), dict(f0=20, f1=22)]
    s = np.array([5.0, 3.0, -1.0])
    assert _progress(samples, s, 9) is None
    assert _progress(samples, s, 11) == 5.0
    assert _progress(samples, s, 14) == pytest.approx(4.0)
    assert _progress(samples, s, 18) == pytest.approx(1.0)
    assert _progress(samples, s, 30) == -1.0


def test_report_without_measured_speed_renders(tmp_path):
    """Single-camera reports carry speed=None; every output must cope."""
    import os
    from cbt.browser import BrowserUnavailable, chromium_browser, file_url
    from cbt import replay, viz
    cam = broadcast_cam()
    rng = np.random.default_rng(3)
    d = None
    while d is None:
        d = sim.random_delivery(rng)
    t = np.linspace(0.01, d.t_impact, 18)
    f = fit_mono(cam.project(d.at(t)), cam, foot_y=d.impact[1] - 0.1)
    rep = f.report(n_mc=100)
    rep["release_speed_kmh"] = None
    viz.figure(f, rep=rep, path=str(tmp_path / "d.png"))
    html = replay.write_replay(str(tmp_path / "r.html"), replay.replay_data(f, rep, source="test"))
    errors = []
    try:
        with chromium_browser() as b:
            pg = b.new_page()
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(file_url(html))
            pg.wait_for_function("window.REPLAY !== undefined", timeout=20000)
            pg.evaluate("REPLAY.renderAt(REPLAY.duration)")
    except BrowserUnavailable as e:
        if os.environ.get("CBT_REQUIRE_BROWSER"):
            raise
        pytest.skip(str(e))
    assert not errors, errors


def test_auto_crease_points_without_clicks():
    """The crease corners are found from the painted lines alone."""
    cam = broadcast_cam()
    img = synthetic_pitch_image(cam)
    pts, info = bc.auto_crease_points(img)
    for n, p in bc.CREASE_POINTS.items():
        truth = cam.project(np.array([p]))[0]
        assert np.linalg.norm(np.array(pts[n]) - truth) < 4.0, (n, pts[n], truth)
    est, ci = bc.calibrate_from_clicks(img, pts)
    assert np.linalg.norm(est.center - cam.center) < 3.0


def test_package_cards_and_writer(tmp_path):
    """Decision card, title card and the package video writer (no browser needed)."""
    from cbt import package
    cam = broadcast_cam()
    rng = np.random.default_rng(5)
    d = None
    while d is None:
        d = sim.random_delivery(rng)
    t = np.linspace(0.01, d.t_impact, 18)
    rep = fit_mono(cam.project(d.at(t)), cam, foot_y=d.impact[1] - 0.1).report(n_mc=100)
    rep["source"] = "test.mp4"
    title = package.title_card(rep, str(tmp_path / "t.png"))
    card = package.decision_card(rep, str(tmp_path / "d.png"))
    w = package.Writer(str(tmp_path / "p.mp4"))
    w.hold(cv2.imread(title), 0.5, package._label("TITLE", "sub"))
    w.hold(cv2.imread(card), 0.5)
    w.close()
    cap = cv2.VideoCapture(str(tmp_path / "p.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
