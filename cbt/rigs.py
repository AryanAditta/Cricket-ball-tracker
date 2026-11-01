"""Candidate camera layouts for an own-camera installation.

Positions are in the world frame of cbt.pitch (metres). Heights assume
cameras on the sight-screen gantry / roof of the stands.
`field_width` is how many metres the frame spans horizontally at the aim
point -- i.e. the zoom setting.
"""
from .camera import Camera
from .pitch import PITCH_LENGTH

L = PITCH_LENGTH


def _cams(fps, size):
    return {
        # behind the bowler, elevated -- the classic "end-on" view
        "end_bowler": Camera.look_at((0, L + 35, 6), (0, 10, 0.3), 8.0, size=size,
                                     name="end_bowler", fps=fps),
        # behind the striker
        "end_striker": Camera.look_at((0, -35, 6), (0, 10, 0.3), 8.0, size=size,
                                      name="end_striker", fps=fps),
        # square of the wicket, off side and leg side
        "side_off": Camera.look_at((40, L / 2, 8), (0, L / 2, 0.5), 24.0, size=size,
                                   name="side_off", fps=fps),
        "side_leg": Camera.look_at((-40, L / 2, 8), (0, L / 2, 0.5), 24.0, size=size,
                                   name="side_leg", fps=fps),
    }


RIGS = {
    "1cam_end": ["end_bowler"],
    "1cam_side": ["side_off"],
    "2cam_end+side": ["end_bowler", "side_off"],
    "2cam_sides": ["side_off", "side_leg"],
    "4cam": ["end_bowler", "end_striker", "side_off", "side_leg"],
}


def make_rig(name, fps=120.0, size=(1920, 1080)):
    cams = _cams(fps, size)
    return [cams[n] for n in RIGS[name]]
