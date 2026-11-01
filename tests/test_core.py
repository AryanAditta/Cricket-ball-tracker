import numpy as np
import pytest

from cbt import sim, rigs, fit, lbw
from cbt.camera import Camera
from cbt.pitch import landmarks, STUMP_HALF_WIDTH, BALL_RADIUS


def test_project_ray_roundtrip():
    cam = Camera.look_at((0, 55, 6), (0, 10, 0.3), 8.0)
    X = np.array([[0.2, 5.0, BALL_RADIUS], [-0.4, 12.0, BALL_RADIUS]])
    back = cam.ray_plane(cam.project(X), z=BALL_RADIUS)
    assert np.allclose(back, X, atol=1e-6)


def test_calibration_recovers_pose():
    rng = np.random.default_rng(0)
    for cam in rigs.make_rig("4cam"):
        est, n = sim.calibrated_copy(cam, rng, click_sigma=0.0, focal_err=0.0)
        assert n >= 6
        assert np.linalg.norm(est.center - cam.center) < 1e-3


def test_lbw_zones():
    assert lbw.wickets(0.0, 0.3) == "hitting"
    assert lbw.wickets(STUMP_HALF_WIDTH + 0.02, 0.3) == "umpires call"
    assert lbw.wickets(STUMP_HALF_WIDTH + 0.05, 0.3) == "missing"
    assert lbw.verdict(-0.3, 0.0, (0, 0.3))["pitching"] == "outside leg"
    assert lbw.verdict(-0.3, 0.0, (0, 0.3), handed="left")["pitching"] == "outside off"


@pytest.mark.parametrize("rig", ["2cam_end+side", "4cam"])
def test_fit_accuracy_multicam(rig):
    rng = np.random.default_rng(42)
    errs = []
    for _ in range(5):
        d = None
        while d is None:
            d = sim.random_delivery(rng)
        cams = rigs.make_rig(rig)
        est = [sim.calibrated_copy(c, rng)[0] for c in cams]
        obs = [(o.cam, o.t, o.uv) for o in sim.observe(d, cams, rng)]
        r = fit.fit(obs, est)
        errs.append(np.abs(r.stumps() - d.stumps).max())
        assert np.linalg.norm(r.bounce[:2] - d.bounce[:2]) < 0.10
    assert np.median(errs) < 0.03       # 3 cm at the stumps
