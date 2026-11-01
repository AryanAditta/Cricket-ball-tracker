from cbt import validate as V


def _rep(bx, by, wickets, status="ok", speed=130.0):
    return dict(status=status, bounce_xy_m=[bx, by], pitching="in line", wickets=wickets,
                release_speed_kmh=speed)


def test_score_counts_and_errors():
    truth = [
        dict(delivery="a", pitch_x="0.00", pitch_y="5.00", stumps_result="hit", speed_kmh="131", handed="right"),
        dict(delivery="b", pitch_x="0.10", pitch_y="3.00", stumps_result="miss", speed_kmh="", handed="right"),
        dict(delivery="c", pitch_x="", pitch_y="", stumps_result="hit", speed_kmh="", handed="right"),
        dict(delivery="d", pitch_x="0", pitch_y="4", stumps_result="miss", speed_kmh="", handed="right"),
    ]
    reports = {"a": _rep(0.03, 5.04, "hitting"),          # 5 cm error, correct
               "b": _rep(0.10, 3.00, "hitting"),          # wrong call
               "c": _rep(0.0, 4.0, "umpires call")}       # marginal; d untracked
    s, rows = V.score(truth, reports)
    assert s["deliveries"] == 4
    assert abs(s["reliability_pct"] - 75.0) < 1e-9
    assert s["wickets_decided"] == 2 and s["wickets_marginal"] == 1
    assert abs(s["wickets_agree_pct"] - 50.0) < 1e-9
    assert abs(s["pitch_err_median_cm"] - 2.5) < 1e-6         # errors 5.0 and 0.0
    assert abs(s["speed_err_median_kmh"] - 1.0) < 1e-9
