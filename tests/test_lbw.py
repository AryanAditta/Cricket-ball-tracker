import pytest
from cbt import lbw
from cbt.lbw import ZONE_HALF_WIDTH as HW, ZONE_TOP, R


@pytest.mark.parametrize("x,handed,expected", [
    (0.0, "right", "in line"),
    (-HW + 0.001, "right", "in line"),
    (-HW - 0.001, "right", "outside leg"),      # centre just outside: >50% of ball outside
    (-HW, "right", "in line"),                  # exactly 50%: favours the bowler
    (HW + 0.02, "right", "outside off"),
    (-HW - 0.02, "left", "outside off"),        # mirrored for a left-hander
    (HW + 0.02, "left", "outside leg"),
])
def test_pitching_uses_ball_centre(x, handed, expected):
    assert lbw.pitching(x, handed) == expected


@pytest.mark.parametrize("x,expected", [
    (0.05, "in line"),
    (HW, "in line"),                 # half the ball in line
    (HW + 0.01, "umpires call"),     # less than half, but some of it
    (HW + R - 0.001, "umpires call"),
    (HW + R + 0.001, "outside off"),
    (-(HW + R + 0.001), "outside leg"),
])
def test_impact_umpires_call(x, expected):
    assert lbw.impact(x) == expected


@pytest.mark.parametrize("x,z,expected", [
    (0.0, 0.4, "hitting"),
    (0.0, ZONE_TOP - 0.001, "hitting"),          # top of bails counts (2021 change)
    (0.0, ZONE_TOP + 0.01, "umpires call"),
    (HW + 0.02, 0.4, "umpires call"),
    (HW + R + 0.01, 0.4, "missing"),
    (0.0, ZONE_TOP + R + 0.01, "missing"),
])
def test_wickets(x, z, expected):
    assert lbw.wickets(x, z) == expected


def test_distance_rules():
    assert lbw.distance_rule(3.0, 2.0) == "3m"
    assert lbw.distance_rule(3.4, 0.1) == "3m"
    assert lbw.distance_rule(2.7, 0.3) == "2.5m/40cm"
    assert lbw.distance_rule(2.7, 0.5) is None
    assert lbw.distance_rule(2.5, 0.3) is None          # must be MORE than 2.5 m
    assert lbw.distance_rule(1.0, 0.1) is None


@pytest.mark.parametrize("p,i,w,on_field,shot,drule,decision,uc", [
    ("in line", "in line", "hitting", "not out", True, None, "out", False),
    ("outside leg", "in line", "hitting", "out", True, None, "not out", False),
    ("in line", "outside leg", "hitting", "out", True, None, "not out", False),
    ("in line", "outside off", "hitting", "out", True, None, "not out", False),
    ("in line", "outside off", "hitting", "not out", False, None, "out", False),  # no shot
    ("in line", "in line", "missing", "out", True, None, "not out", False),
    ("in line", "umpires call", "hitting", "not out", True, None, "not out", True),
    ("in line", "umpires call", "hitting", "out", True, None, "out", True),
    ("outside off", "in line", "umpires call", "out", True, None, "out", True),
    ("in line", "in line", "hitting", "not out", True, "3m", "not out", True),
    ("in line", "in line", "inconclusive", "out", True, None, "out", True),
    ("in line", "umpires call", "missing", "out", True, None, "not out", False),
])
def test_decide(p, i, w, on_field, shot, drule, decision, uc):
    d = lbw.decide(p, i, w, on_field, shot, drule)
    assert (d["decision"], d["umpires_call"]) == (decision, uc)


def test_review_outcome():
    assert lbw.review_outcome("not out", "out", False, "fielding") == "successful"
    assert lbw.review_outcome("out", "out", False, "batting").startswith("unsuccessful")
    assert lbw.review_outcome("out", "out", True, "batting").startswith("umpire's call")


def test_replay_js_matches_python(tmp_path):
    """Verify the offline decision table and failed-track handling in Chromium."""
    import itertools
    import numpy as np
    import os
    from cbt.browser import BrowserUnavailable, chromium_browser, file_url
    from cbt import sim, rigs, fit, replay
    rng = np.random.default_rng(3)
    d = None
    while d is None:
        d = sim.random_delivery(rng)
    cams = rigs.make_rig("2cam_end+side")
    r = fit.fit([(o.cam, o.t, o.uv) for o in sim.observe(d, cams, rng)], cams)
    html = replay.write_replay(str(tmp_path / "r.html"), replay.replay_data(r, r.report()))
    cases = [list(c) for c in itertools.product(
        ["in line", "outside off", "outside leg"], ["in line", "umpires call", "outside off", "outside leg"],
        ["hitting", "umpires call", "missing", "inconclusive"], ["out", "not out"], [True, False], [None, "3m", "2.5m/40cm"])]
    try:
        with chromium_browser(["--use-gl=angle", "--use-angle=swiftshader",
                               "--enable-unsafe-swiftshader"]) as b:
            pg = b.new_page()
            pg.goto(file_url(html, capture=True))
            pg.wait_for_function("window.REPLAY !== undefined")
            js = pg.evaluate("cs => cs.map(c => { const d = REPLAY.decide(...c); return [d.decision, d.umpiresCall]; })", cases)
            from cbt.console import build_console
            rep = r.report()
            rep.update(status="NO DECISION: incomplete track", pitching="outside leg",
                       wickets="inconclusive", decision="out", on_field="out")
            con = build_console(str(tmp_path), [], r, rep)
            pg.goto(file_url(con))
            pg.wait_for_function("window.CONSOLE !== undefined")
            js2 = pg.evaluate("cs => cs.map(c => { const d = CONSOLE.decide(...c); return [d.decision, d.umpiresCall]; })", cases)
            assert pg.locator("#verdict .big").inner_text() == "NO DECISION"
            note = pg.locator("#verdict .small").first.text_content()
            assert note.startswith("OUT · ") and "on-field decision stands" in note
            assert not pg.locator("#fast").is_visible()
    except BrowserUnavailable as e:
        if os.environ.get("CBT_REQUIRE_BROWSER"):
            raise
        pytest.skip(str(e))
    assert js == js2
    for c, j in zip(cases, js):
        d = lbw.decide(*c)
        assert [d["decision"], d["umpires_call"]] == j, c
