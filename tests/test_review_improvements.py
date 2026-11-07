import cv2
import numpy as np

from cbt.browser import file_url
from cbt.broadcast import ball_candidates
from cbt.stabilize import _registration
from cbt.validate import score, write_report


def test_local_file_url_preserves_reserved_filename_chars(tmp_path):
    uri = file_url(tmp_path / "review #1%20.html", capture=True)
    assert uri.endswith("/review%20%231%2520.html?capture")


def test_radar_validation_keeps_missing_and_assisted_speeds_out_of_accuracy(tmp_path):
    truth = [{"delivery": key, "speed_kmh": "90", "stumps_result": result}
             for key, result in (("missing", "miss"), ("estimate", "miss"), ("assisted", "hit"))]
    base = {"release_speed_kmh": None, "wickets": "missing", "wickets_confidence": {"missing": 0.8}}
    reports = {
        "missing": base,
        "estimate": dict(base, speed_estimate_kmh=92),
        "assisted": dict(base, speed_estimate_kmh=90, speed_gun_kmh=90),
    }
    summary, rows = score(truth, reports)
    assert [r["speed_source"] for r in rows] == ["unavailable", "estimated", "speed-gun assisted"]
    assert (summary["speed_scored"], summary["speed_measured"], summary["speed_err_p95_kmh"]) == (1, 3, 2)
    report = tmp_path / "validation.md"
    write_report(report, summary, rows)
    assert "| 80–100% | 3 | 80% | 67% |" in report.read_text(encoding="utf-8")


def test_white_ball_candidates_reject_dark_and_saturated_objects():
    bg = np.full((180, 320, 3), (80, 115, 130), np.uint8)
    frame = bg.copy()
    cv2.circle(frame, (70, 75), 7, (248, 249, 251), -1)
    cv2.circle(frame, (140, 75), 7, (20, 20, 30), -1)
    cv2.circle(frame, (210, 75), 7, (10, 240, 255), -1)
    candidates = ball_candidates(frame, bg, np.full(bg.shape[:2], 255, np.uint8), 8, ball="white")
    assert len(candidates) == 1
    assert np.allclose(candidates[0, :2], [70, 75], atol=1)


def test_stabilization_recovers_pan_and_zoom():
    rng = np.random.default_rng(4)
    reference = np.full((360, 640), 80, np.uint8)
    for x, y in rng.integers([30, 30], [610, 330], size=(160, 2)):
        cv2.rectangle(reference, (x, y), (x + 6, y + 5), 240, -1)
    pan_zoom = cv2.getRotationMatrix2D((320, 180), 1.0, 1.035)
    pan_zoom[:, 2] += [18, -9]
    moved = cv2.warpAffine(reference, pan_zoom, (640, 360))
    features = cv2.goodFeaturesToTrack(reference, 300, 0.012, 6)
    correction = _registration(reference, moved, features)
    corners = np.array([[120.0, 80.0], [500.0, 300.0]])
    moved_corners = corners @ pan_zoom[:, :2].T + pan_zoom[:, 2]
    assert np.allclose(moved_corners @ correction[:, :2].T + correction[:, 2], corners, atol=0.7)
