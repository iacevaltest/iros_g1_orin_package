"""The Peval boundary — the only contract between a team and the robot.

Three endpoints on the Orin, and nothing else:

    CameraStream   SUB  :5555   images in
    StateStream    SUB  :5557   proprioception in
    ActionSink     BIND :5556   actions out   <- the only lane-aware piece

Everything upstream of these (how your policy server talks to your policy
client, what runs inside your containers, which framework you use) is yours.
Everything downstream (the whole-body controller, the robot, the e-stop) is
the organizer's.

DO NOT MODIFY THIS PACKAGE. The bench runs against these exact formats; a
local edit here will pass your tests and fail on the robot.
"""

from __future__ import annotations

from .actions import ActionSink, ActionError, DecoupledSink, SonicSink
from .cameras import CameraStream, CAMERA_KEYS
from .states import StateStream, RobotState

LANES = ("sonic", "decoupled")

__all__ = [
    "ActionSink",
    "ActionError",
    "SonicSink",
    "DecoupledSink",
    "CameraStream",
    "CAMERA_KEYS",
    "StateStream",
    "RobotState",
    "LANES",
]
