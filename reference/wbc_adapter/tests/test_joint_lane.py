"""Joint-lane tests for the WBC adapter -- no robot, no ROS 2, no WBC process.

Run from the repository root with either runner:

    python -m pytest reference/wbc_adapter/tests
    python -m unittest discover -s reference/wbc_adapter/tests

Needs the same Python deps the adapter needs (numpy, pinocchio, pink,
qpsolvers, msgpack, zmq, scipy) plus `decoupled_wbc` importable -- the WBC
checkout on PYTHONPATH -- because the UpperBodyMapper reads the WBC's own
robot model, and that model's URDF doubles as the IK's URDF here when
ik.py's default (~/g1_bridge/...) is absent.

The taskspace-lane REGRESSION test is the interference proof for the joint
lane: the same chunks through the same fake backend and state, once
through this checkout's wbc_driver.py and once through the last revision
before the joint lane existed (BASELINE_REV, loaded from git into a temp
file), must yield the same goals to 1e-9.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

import msgpack
import numpy as np

HERE = Path(__file__).resolve().parent
ADAPTER_DIR = HERE.parent
REPO_ROOT = ADAPTER_DIR.parent.parent
for _p in (str(ADAPTER_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import boundary_wire  # noqa: E402
import ik  # noqa: E402
import wbc_driver  # noqa: E402
import wbc_goal  # noqa: E402
from upper_body_map import UpperBodyMapper  # noqa: E402

# Last commit before the joint lane. Override with WBC_DRIVER_BASELINE_REV.
BASELINE_REV = "47f4e1b"

CHUNK_HZ = 20.0
DT = 1.0 / CHUNK_HZ

# A fixed 43-wide WBC state: legs/waist zero, both arms in a plausible
# pose (shoulder_roll inside the WBC model's +-0.19 narrowing), hands at
# distinct non-zero values so a copied hand slot is distinguishable from
# a zeroed one.
LEFT_ARM = np.array([0.20, 0.30, -0.10, 0.50, 0.10, -0.20, 0.05])
RIGHT_ARM = np.array([0.20, -0.30, 0.10, 0.50, -0.10, -0.20, -0.05])
Q43 = np.zeros(43)
Q43[15:22] = LEFT_ARM
Q43[22:29] = 0.01 * np.arange(1, 8)       # left hand
Q43[29:36] = RIGHT_ARM
Q43[36:43] = -0.01 * np.arange(1, 8)      # right hand


def _model_data_dir() -> Path:
    from decoupled_wbc.control.robot_model.instantiation import g1
    root = Path(g1.__file__).resolve().parents[4]
    return root / "decoupled_wbc" / "control" / "robot_model" / "model_data" / "g1"


def _ik_urdf() -> tuple[Path, Path]:
    if ik.DEFAULT_URDF.exists():
        return ik.DEFAULT_URDF, ik.DEFAULT_URDF_DIR
    d = _model_data_dir()
    return d / "g1_29dof_with_hand.urdf", d


_MAPPER = None
_SOLVER = None


def mapper() -> UpperBodyMapper:
    global _MAPPER
    if _MAPPER is None:
        _MAPPER = UpperBodyMapper(waist_location="lower_body")
    return _MAPPER


def make_solver() -> ik.UpperBodyIK:
    urdf, assets = _ik_urdf()
    return ik.UpperBodyIK(ik.IKSettings(max_err=1e-3), include_waist=False,
                          urdf=urdf, assets=assets, warm_start="current")


def solver() -> ik.UpperBodyIK:
    global _SOLVER
    if _SOLVER is None:
        _SOLVER = make_solver()
    return _SOLVER


class FakeBackend:
    """publish_goal records; get_robot_q returns a fixed 43-wide q."""
    name = "fake"

    def __init__(self, q43=Q43):
        self.q = np.asarray(q43, dtype=np.float64).copy()
        self.goals: list[dict] = []
        self.toggles = 0

    def publish_goal(self, goal):
        self.goals.append(_deep_copy_goal(goal))

    def get_robot_q(self):
        return self.q.copy()

    def toggle_policy_action(self):
        self.toggles += 1

    def health(self):
        return {"backend": self.name, "state_stale": False, "state_staleness_s": 0.0}


class FakeDex1:
    def __init__(self):
        self.sent = []

    def send(self, blob: bytes):
        assert blob.startswith(b"dex1")
        self.sent.append(msgpack.unpackb(blob[4:], raw=False))


def _deep_copy_goal(goal):
    out = {}
    for k, v in goal.items():
        if isinstance(v, list):
            out[k] = [np.array(x, dtype=np.float64) if isinstance(x, np.ndarray) else x for x in v]
        elif isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v
    return out


def make_args(**over) -> types.SimpleNamespace:
    a = dict(
        lane="decoupled", live=True, wbc_backend="zmq", upper_body_from_model=True,
        engage_policy=False, verbose=False, state_source="wbc",
        orin_host="127.0.0.1", state_port=5557, enable_waist=False,
        chunk_hz=CHUNK_HZ, max_waypoints=16, max_joint_vel=1.0, max_ik_err=1e-3,
        ik_warm_start="current", dex1_port=0, dex1_host="127.0.0.1",
        max_chunk_age_s=1.0, joint_lane="on", joint_lane_limits="urdf",
        goto_max_speed=0.45,
    )
    a.update(over)
    return types.SimpleNamespace(**a)


def make_ctx(backend=None, dex1=None, limits="urdf", **args_over):
    args = make_args(joint_lane_limits=limits, **args_over)
    backend = FakeBackend() if backend is None else backend
    lim = wbc_driver._load_arm_limits(limits, mapper(), solver().settings)
    return wbc_driver._DecoupledContext(args, wbc_driver.Stats(), backend, None,
                                        solver(), mapper(), dex1, lim)


def joint_msg(rows, issued_at=None) -> bytes:
    from boundary.actions import _pack_joint_chunk
    return _pack_joint_chunk(np.asarray(rows, dtype=np.float32),
                             time.time() if issued_at is None else issued_at)


def goto_msg(left, right, max_speed=0.3, hands=None, issued_at=None) -> bytes:
    from boundary.actions import _pack_goto
    return _pack_goto(left, right, max_speed, hands,
                      time.time() if issued_at is None else issued_at)


def joint_rows(T=3, left=None, right=None):
    """(T,22): hands +-0.5, arms = measured + small per-row offsets, distinct
    navigate/base-height per row."""
    rows = np.zeros((T, 22), dtype=np.float32)
    rows[:, 0:2] = 0.5
    rows[:, 2:4] = -0.5
    for t in range(T):
        rows[t, 4:11] = LEFT_ARM + 0.01 * (t + 1) if left is None else left[t]
        rows[t, 11:18] = RIGHT_ARM - 0.01 * (t + 1) if right is None else right[t]
        rows[t, 18:21] = [0.1 * (t + 1), -0.05 * (t + 1), 0.02 * (t + 1)]
        rows[t, 21] = 0.70 + 0.01 * t
    return rows


class _StopLoop(Exception):
    pass


class FakeSub:
    """Feeds run_decoupled a scripted sequence: bytes -> a message, a
    callable -> the message it builds AT RECV TIME (so its issued_at is
    fresh even though run_decoupled spends seconds loading models before
    its first recv), 'again' -> zmq.Again (keepalive path), end -> _StopLoop."""

    def __init__(self, script):
        self.script = list(script)

    def recv(self):
        import zmq
        if not self.script:
            raise _StopLoop()
        item = self.script.pop(0)
        if item == "again":
            raise zmq.Again()
        return item() if callable(item) else item


@contextlib.contextmanager
def _patched_factories(module, backend):
    """Route the module's run_decoupled to the fake backend and to an IK
    built against a URDF that exists here."""
    urdf, assets = _ik_urdf()
    real_ik_cls = ik.UpperBodyIK

    def ik_factory(settings, include_waist=False, warm_start="current", **kw):
        return real_ik_cls(settings, include_waist=include_waist, urdf=urdf,
                           assets=assets, warm_start=warm_start)

    saved_backend = wbc_goal.make_backend
    wbc_goal.make_backend = lambda kind, **kw: backend
    ik.UpperBodyIK = ik_factory
    try:
        yield
    finally:
        wbc_goal.make_backend = saved_backend
        ik.UpperBodyIK = real_ik_cls


def drive(module, args, script, backend=None):
    """Run `module.run_decoupled` over a scripted socket; returns (backend, stats)."""
    backend = FakeBackend() if backend is None else backend
    stats = module.Stats()
    with _patched_factories(module, backend), contextlib.redirect_stdout(io.StringIO()):
        try:
            module.run_decoupled(args, FakeSub(script), stats)
        except _StopLoop:
            pass
    return backend, stats


# ---------------------------------------------------------------------------
# 1. mapping into the WBC's upper-body slots
# ---------------------------------------------------------------------------


class JointChunkMapping(unittest.TestCase):
    def test_rows_land_in_the_right_slots(self):
        dex1 = FakeDex1()
        ctx = make_ctx(dex1=dex1)
        rows = joint_rows(T=3)
        t0 = time.monotonic()
        wbc_driver._handle_joint(ctx, joint_msg(rows))

        self.assertEqual(len(ctx.backend.goals), 1)
        goal = ctx.backend.goals[0]
        self.assertEqual(set(goal), {"target_upper_body_pose", "base_height_command",
                                     "navigate_cmd", "target_time"})
        wps = goal["target_upper_body_pose"]
        self.assertEqual(len(wps), 3)
        for t, wp in enumerate(wps):
            self.assertEqual(wp.shape, (28,))
            np.testing.assert_allclose(wp[0:7], rows[t, 4:11], atol=1e-6)     # left arm
            np.testing.assert_allclose(wp[14:21], rows[t, 11:18], atol=1e-6)  # right arm
            np.testing.assert_allclose(wp[7:14], Q43[22:29])                  # left hand: state
            np.testing.assert_allclose(wp[21:28], Q43[36:43])                 # right hand: state
            np.testing.assert_allclose(goal["base_height_command"][t], [rows[t, 21]], atol=1e-6)
            np.testing.assert_allclose(goal["navigate_cmd"][t], rows[t, 18:21], atol=1e-6)
        # same schedule as the taskspace path: t_base + (i+1)/chunk_hz
        times = np.asarray(goal["target_time"])
        np.testing.assert_allclose(np.diff(times), DT, atol=1e-9)
        self.assertGreater(times[0], t0 + DT - 1e-6)
        self.assertLess(times[0], time.monotonic() + DT + 1e-6)
        # keepalive template holds the last waypoint
        np.testing.assert_allclose(ctx.last_goal_template["upper_body"], wps[-1])
        self.assertIsNone(ctx.last_goal_template["wrist_pose"])
        # gripper relay: rows[0][0], rows[0][2], same as taskspace
        self.assertEqual(dex1.sent, [{"left": 0.5, "right": -0.5}])
        self.assertEqual(ctx.stats.joint_accepted, 1)
        self.assertEqual(ctx.stats.published, 1)
        self.assertEqual(ctx.stats.gripper_sent, 1)
        self.assertEqual(ctx.stats.joints_clamped, 0)

    def test_schedule_matches_taskspace_lane(self):
        ctx = make_ctx()
        wbc_driver._handle_joint(ctx, joint_msg(joint_rows(T=4)))
        wbc_driver._handle_taskspace(ctx, taskspace_msg(taskspace_rows(T=4)))
        tj = np.asarray(ctx.backend.goals[0]["target_time"])
        tt = np.asarray(ctx.backend.goals[1]["target_time"])
        self.assertEqual(len(tj), len(tt))
        np.testing.assert_allclose(np.diff(tj), np.diff(tt), atol=1e-9)

    def test_max_waypoints_truncates_like_taskspace(self):
        ctx = make_ctx(max_waypoints=2)
        wbc_driver._handle_joint(ctx, joint_msg(joint_rows(T=5)))
        self.assertEqual(len(ctx.backend.goals[0]["target_upper_body_pose"]), 2)


# ---------------------------------------------------------------------------
# 2. the existing step clamp, anchored on the measured arms
# ---------------------------------------------------------------------------


class JointStepClamp(unittest.TestCase):
    def test_first_waypoint_within_max_step_of_measured(self):
        ctx = make_ctx(max_joint_vel=1.0)
        max_step = 1.0 * DT
        far_left = np.tile(LEFT_ARM + np.array([0, 0, 0, 1.0, 0, 0, 0]), (4, 1))
        far_right = np.tile(RIGHT_ARM + np.array([-0.8, 0, 0, 0, 0, 0, 0]), (4, 1))
        wbc_driver._handle_joint(ctx, joint_msg(joint_rows(T=4, left=far_left, right=far_right)))
        wps = ctx.backend.goals[0]["target_upper_body_pose"]
        arms = [np.concatenate([wp[0:7], wp[14:21]]) for wp in wps]
        measured = np.concatenate([LEFT_ARM, RIGHT_ARM])
        self.assertLessEqual(np.max(np.abs(arms[0] - measured)), max_step + 1e-9)
        for a, b in zip(arms, arms[1:]):
            self.assertLessEqual(np.max(np.abs(b - a)), max_step + 1e-9)
        # the ramp actually moves toward the target, one max_step per row
        self.assertAlmostEqual(arms[0][3], LEFT_ARM[3] + max_step, places=9)
        self.assertAlmostEqual(arms[3][3], LEFT_ARM[3] + 4 * max_step, places=9)
        self.assertAlmostEqual(arms[0][7], RIGHT_ARM[0] - max_step, places=9)

    def test_disabled_clamp_passes_rows_through(self):
        ctx = make_ctx(max_joint_vel=0)
        rows = joint_rows(T=2)
        rows[:, 7] = LEFT_ARM[3] + 0.9
        wbc_driver._handle_joint(ctx, joint_msg(rows))
        wps = ctx.backend.goals[0]["target_upper_body_pose"]
        self.assertAlmostEqual(wps[0][3], LEFT_ARM[3] + 0.9, places=6)


# ---------------------------------------------------------------------------
# 3. position clamp from the robot model's limits
# ---------------------------------------------------------------------------


class JointPositionClamp(unittest.TestCase):
    def test_limits_come_from_the_robot_model(self):
        lim = wbc_driver._load_arm_limits("urdf", mapper(), solver().settings)
        m = mapper().model
        for j, name in enumerate(lim.names):
            idx = m.joint_to_dof_index[name]
            self.assertEqual(lim.lower[j], m.lower_joint_limits[idx])
            self.assertEqual(lim.upper[j], m.upper_joint_limits[idx])
        self.assertEqual(lim.names[3], "left_elbow_joint")
        self.assertEqual(lim.names[10], "right_elbow_joint")

    def test_out_of_range_joint_is_clamped_counted_and_logged_once(self):
        ctx = make_ctx(max_joint_vel=0)
        lim = ctx.arm_limits
        rows = joint_rows(T=3)
        rows[:, 7] = 3.0          # left elbow, well past its upper limit
        rows[:, 14] = -3.0        # right elbow, past its lower limit
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            wbc_driver._handle_joint(ctx, joint_msg(rows))
            wbc_driver._handle_joint(ctx, joint_msg(rows))
        for goal in ctx.backend.goals:
            for wp in goal["target_upper_body_pose"]:
                self.assertAlmostEqual(wp[3], lim.upper[3] - lim.MARGIN, places=12)
                self.assertAlmostEqual(wp[17], lim.lower[10] + lim.MARGIN, places=12)
        self.assertEqual(ctx.stats.joints_clamped, 2 * 3 * 2)
        log = err.getvalue()
        self.assertEqual(log.count("left_elbow_joint commanded"), 1)
        self.assertEqual(log.count("right_elbow_joint commanded"), 1)

    def test_wbc_model_narrowing_of_shoulder_roll_applies(self):
        ctx = make_ctx(max_joint_vel=0)
        rows = joint_rows(T=1)
        rows[0, 5] = 0.0          # left shoulder_roll: URDF allows it, WBC model does not
        with contextlib.redirect_stderr(io.StringIO()):
            wbc_driver._handle_joint(ctx, joint_msg(rows))
        wp = ctx.backend.goals[0]["target_upper_body_pose"][0]
        self.assertAlmostEqual(wp[1], ctx.arm_limits.lower[1] + ctx.arm_limits.MARGIN, places=12)
        self.assertGreater(ctx.arm_limits.lower[1], 0.0)

    def test_ik_mode_applies_the_solver_overrides(self):
        s = solver().settings
        urdf = wbc_driver._load_arm_limits("urdf", mapper(), s)
        ikm = wbc_driver._load_arm_limits("ik", mapper(), s)
        for off in (0, 7):
            self.assertEqual(ikm.upper[off + 3], s.elbow_upper_limit_override)
            self.assertEqual(ikm.upper[off + 4], s.wrist_roll_limit_override)
            self.assertEqual(ikm.lower[off + 4], -s.wrist_roll_limit_override)
            self.assertGreater(urdf.upper[off + 3], ikm.upper[off + 3])
        ctx = make_ctx(limits="ik", max_joint_vel=0)
        rows = joint_rows(T=1)
        rows[0, 7] = 1.9      # left elbow: inside the URDF, outside the IK override
        rows[0, 8] = 1.5      # left wrist_roll: same
        with contextlib.redirect_stderr(io.StringIO()):
            wbc_driver._handle_joint(ctx, joint_msg(rows))
        wp = ctx.backend.goals[0]["target_upper_body_pose"][0]
        self.assertAlmostEqual(wp[3], s.elbow_upper_limit_override - ikm.MARGIN, places=12)
        self.assertAlmostEqual(wp[4], s.wrist_roll_limit_override - ikm.MARGIN, places=12)


# ---------------------------------------------------------------------------
# 4. validation
# ---------------------------------------------------------------------------


class JointValidation(unittest.TestCase):
    def _reject(self, rows, expect_substr):
        ctx = make_ctx()
        with contextlib.redirect_stderr(io.StringIO()) as err:
            wbc_driver._handle_joint(ctx, joint_msg(rows))
        self.assertEqual(ctx.backend.goals, [])
        self.assertEqual(ctx.stats.joint_rejected, 1)
        self.assertEqual(ctx.stats.rejected, 1)
        self.assertIn(expect_substr, err.getvalue())

    def test_nan(self):
        rows = joint_rows(T=2)
        rows[1, 9] = np.nan
        self._reject(rows, "NaN")

    def test_wrong_width(self):
        self._reject(np.zeros((2, 21), dtype=np.float32), "expected (T, 22)")

    def test_hand_out_of_range(self):
        rows = joint_rows(T=2)
        rows[0, 2] = 1.5
        self._reject(rows, "hand commands must lie in [-1, 1]")

    def test_hand_tolerance(self):
        rows = joint_rows(T=1)
        rows[0, 0:2] = 1.0005     # inside the 1e-3 tolerance, accepted
        self.assertEqual(boundary_wire.validate_joint(
            boundary_wire.decode_joint(joint_msg(rows))), [])

    def test_too_long(self):
        self._reject(joint_rows(T=65), "outside 1..64")

    def test_empty(self):
        self._reject(np.zeros((0, 22), dtype=np.float32), "outside 1..64")

    def test_stale_chunk_dropped(self):
        ctx = make_ctx()
        wbc_driver._handle_joint(ctx, joint_msg(joint_rows(T=2), issued_at=time.time() - 5.0))
        self.assertEqual(ctx.backend.goals, [])
        self.assertEqual(ctx.stats.stale, 1)
        self.assertEqual(ctx.stats.joint_rejected, 0)

    def test_no_robot_state_refuses(self):
        class NoState(FakeBackend):
            def get_robot_q(self):
                return None
        ctx = make_ctx(backend=NoState())
        with contextlib.redirect_stderr(io.StringIO()) as err:
            wbc_driver._handle_joint(ctx, joint_msg(joint_rows(T=2)))
        self.assertEqual(ctx.backend.goals, [])
        self.assertIn("no robot state", err.getvalue())

    def test_undecodable_frame(self):
        ctx = make_ctx()
        with contextlib.redirect_stderr(io.StringIO()):
            wbc_driver._handle_joint(ctx, b"joint" + b"\xc1garbage")
        self.assertEqual(ctx.stats.joint_rejected, 1)


# ---------------------------------------------------------------------------
# 5. goto
# ---------------------------------------------------------------------------


class Goto(unittest.TestCase):
    TARGET_LEFT = LEFT_ARM + np.array([0.0, 0.0, 0.0, 0.9, 0.0, -0.3, 0.0])
    TARGET_RIGHT = RIGHT_ARM + np.array([0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def _arms(self, goal):
        return np.asarray([np.concatenate([wp[0:7], wp[14:21]])
                           for wp in goal["target_upper_body_pose"]])

    def test_monotone_interpolation_at_requested_speed(self):
        dex1 = FakeDex1()
        ctx = make_ctx(dex1=dex1)
        with contextlib.redirect_stdout(io.StringIO()):
            wbc_driver._handle_goto(ctx, goto_msg(self.TARGET_LEFT, self.TARGET_RIGHT,
                                                  max_speed=0.3, hands=(-1.0, 1.0)))
        self.assertEqual(len(ctx.backend.goals), 1)
        goal = ctx.backend.goals[0]
        arms = self._arms(goal)
        # 0.9 rad at 0.3 rad/s = 3.0 s = 60 waypoints at 20 Hz
        self.assertEqual(len(arms), 60)
        times = np.asarray(goal["target_time"])
        self.assertEqual(len(times), 60)
        np.testing.assert_allclose(np.diff(times), DT, atol=1e-9)
        measured = np.concatenate([LEFT_ARM, RIGHT_ARM])
        target = np.concatenate([self.TARGET_LEFT, self.TARGET_RIGHT])
        path = np.vstack([measured, arms])
        steps = np.diff(path, axis=0)
        # never faster than the requested speed, on any joint
        self.assertLessEqual(np.max(np.abs(steps)), 0.3 * DT + 1e-9)
        # straight line: every joint moves monotonically, in equal steps
        for j in range(14):
            sgn = np.sign(target[j] - measured[j])
            self.assertTrue(np.all(sgn * steps[:, j] >= -1e-12))
            np.testing.assert_allclose(steps[:, j], steps[0, j], atol=1e-9)
        np.testing.assert_allclose(arms[-1], target, atol=1e-12)
        # hands ride on the goal's other slots, not on the arms
        wp = goal["target_upper_body_pose"][0]
        np.testing.assert_allclose(wp[7:14], Q43[22:29])
        # base commands: nothing published before -> WBC defaults
        np.testing.assert_allclose(goal["base_height_command"][0], [wbc_goal.DEFAULT_BASE_HEIGHT])
        np.testing.assert_allclose(goal["navigate_cmd"][-1], [0.0, 0.0, 0.0])
        # template: the keepalive will hold the final pose
        np.testing.assert_allclose(ctx.last_goal_template["upper_body"],
                                   goal["target_upper_body_pose"][-1])
        self.assertIsNotNone(ctx.goto_in_flight)
        self.assertEqual(dex1.sent, [{"left": -1.0, "right": 1.0}])
        self.assertEqual(ctx.stats.goto_accepted, 1)
        self.assertEqual(ctx.stats.published, 1)

    def test_speed_capped_by_goto_max_speed(self):
        ctx = make_ctx(goto_max_speed=0.45, max_joint_vel=1.0)
        with contextlib.redirect_stdout(io.StringIO()):
            wbc_driver._handle_goto(ctx, goto_msg(self.TARGET_LEFT, self.TARGET_RIGHT, max_speed=5.0))
        arms = self._arms(ctx.backend.goals[0])
        self.assertEqual(len(arms), 40)   # 0.9 / (0.45 * 0.05)
        steps = np.diff(np.vstack([np.concatenate([LEFT_ARM, RIGHT_ARM]), arms]), axis=0)
        self.assertLessEqual(np.max(np.abs(steps)), 0.45 * DT + 1e-9)
        np.testing.assert_allclose(arms[-1], np.concatenate([self.TARGET_LEFT, self.TARGET_RIGHT]),
                                   atol=1e-12)

    def test_speed_capped_by_max_joint_vel(self):
        ctx = make_ctx(goto_max_speed=2.0, max_joint_vel=0.5)
        with contextlib.redirect_stdout(io.StringIO()):
            wbc_driver._handle_goto(ctx, goto_msg(self.TARGET_LEFT, self.TARGET_RIGHT, max_speed=5.0))
        arms = self._arms(ctx.backend.goals[0])
        self.assertEqual(len(arms), 36)   # 0.9 / (0.5 * 0.05)
        np.testing.assert_allclose(arms[-1], np.concatenate([self.TARGET_LEFT, self.TARGET_RIGHT]),
                                   atol=1e-12)

    def test_no_hands_means_no_gripper_command(self):
        dex1 = FakeDex1()
        ctx = make_ctx(dex1=dex1)
        with contextlib.redirect_stdout(io.StringIO()):
            wbc_driver._handle_goto(ctx, goto_msg(self.TARGET_LEFT, self.TARGET_RIGHT))
        self.assertEqual(dex1.sent, [])
        self.assertEqual(ctx.stats.gripper_sent, 0)

    def test_target_is_position_clamped(self):
        ctx = make_ctx()
        bad_left = LEFT_ARM.copy()
        bad_left[3] = 3.0
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            wbc_driver._handle_goto(ctx, goto_msg(bad_left, RIGHT_ARM, max_speed=0.45))
        arms = self._arms(ctx.backend.goals[0])
        self.assertAlmostEqual(arms[-1][3], ctx.arm_limits.upper[3] - ctx.arm_limits.MARGIN, places=12)
        self.assertEqual(ctx.stats.joints_clamped, 1)

    def test_already_there_is_one_waypoint(self):
        ctx = make_ctx()
        with contextlib.redirect_stdout(io.StringIO()):
            wbc_driver._handle_goto(ctx, goto_msg(LEFT_ARM, RIGHT_ARM))
        arms = self._arms(ctx.backend.goals[0])
        self.assertEqual(len(arms), 1)
        np.testing.assert_allclose(arms[0], np.concatenate([LEFT_ARM, RIGHT_ARM]), atol=1e-12)

    def test_stale_goto_dropped(self):
        ctx = make_ctx()
        wbc_driver._handle_goto(ctx, goto_msg(self.TARGET_LEFT, self.TARGET_RIGHT,
                                              issued_at=time.time() - 5.0))
        self.assertEqual(ctx.backend.goals, [])
        self.assertEqual(ctx.stats.stale, 1)
        self.assertIsNone(ctx.last_goal_template)

    def test_invalid_goto_rejected(self):
        for left, right, speed, hands, what in (
            (np.zeros(6), RIGHT_ARM, 0.3, None, "left_arm has shape"),
            (LEFT_ARM, RIGHT_ARM, 0.0, None, "max_speed"),
            (LEFT_ARM, RIGHT_ARM, 0.3, (2.0, 0.0), "hands"),
        ):
            ctx = make_ctx()
            msg = b"goto" + msgpack.packb({
                "left_arm": [float(v) for v in left], "right_arm": [float(v) for v in right],
                "max_speed": speed, "hands": hands, "issued_at": time.time()},
                use_bin_type=True)
            with contextlib.redirect_stderr(io.StringIO()) as err:
                wbc_driver._handle_goto(ctx, msg)
            self.assertEqual(ctx.backend.goals, [], what)
            self.assertEqual(ctx.stats.goto_rejected, 1, what)
            self.assertIn(what, err.getvalue())

    def test_keepalive_resends_the_remaining_trajectory_then_holds(self):
        ctx = make_ctx()
        with contextlib.redirect_stdout(io.StringIO()):
            wbc_driver._handle_goto(ctx, goto_msg(self.TARGET_LEFT, self.TARGET_RIGHT, max_speed=0.3))
        goto_goal = ctx.backend.goals[0]
        # keepalive while the whole trajectory is still ahead: same waypoints, same times
        self.assertTrue(wbc_driver._publish_keepalive(ctx))
        ka = ctx.backend.goals[1]
        np.testing.assert_allclose(np.asarray(ka["target_time"]), np.asarray(goto_goal["target_time"]))
        np.testing.assert_allclose(np.asarray(ka["target_upper_body_pose"]),
                                   np.asarray(goto_goal["target_upper_body_pose"]))
        self.assertEqual(ctx.keepalives_sent, 1)
        # half way through: only the future half is re-sent
        now = time.monotonic()
        ctx.goto_in_flight["times"] = [now - 1.0 + i * DT for i in range(60)]
        wbc_driver._publish_keepalive(ctx)
        ka2 = ctx.backend.goals[2]
        self.assertTrue(all(t > now for t in ka2["target_time"]))
        self.assertLess(len(ka2["target_time"]), 60)
        self.assertGreater(len(ka2["target_time"]), 0)
        np.testing.assert_allclose(ka2["target_upper_body_pose"][-1],
                                   goto_goal["target_upper_body_pose"][-1])
        # trajectory done: plain single-row hold of the final pose
        ctx.goto_in_flight["times"] = [now - 10.0 + i * DT for i in range(60)]
        wbc_driver._publish_keepalive(ctx)
        ka3 = ctx.backend.goals[3]
        self.assertIsNone(ctx.goto_in_flight)
        # the pre-existing hold form: a one-waypoint trajectory at t+1/chunk_hz
        self.assertEqual(len(ka3["target_time"]), 1)
        self.assertGreater(ka3["target_time"][0], now)
        np.testing.assert_allclose(ka3["target_upper_body_pose"][0],
                                   goto_goal["target_upper_body_pose"][-1])
        # a later chunk supersedes whatever goto was in flight
        with contextlib.redirect_stdout(io.StringIO()):
            wbc_driver._handle_goto(ctx, goto_msg(self.TARGET_LEFT, self.TARGET_RIGHT, max_speed=0.3))
        self.assertIsNotNone(ctx.goto_in_flight)
        wbc_driver._handle_joint(ctx, joint_msg(joint_rows(T=2)))
        self.assertIsNone(ctx.goto_in_flight)


# ---------------------------------------------------------------------------
# 6. REGRESSION: the taskspace lane is unchanged
# ---------------------------------------------------------------------------


def _wrist_pose_in_pelvis(arm_ik, q29):
    """(pos xyz, quat wxyz) of the arm's wrist frame in the pelvis frame, by
    FK on the IK's own model -- the inverse of what ik.py solves, so a
    target built this way is reachable and IK really iterates on it."""
    import pinocchio as pin
    from scipy.spatial.transform import Rotation
    model = arm_ik.model
    q_full = pin.neutral(model)
    for name, idx in zip(ik._BODY_Q_NAMES, range(29)):
        if name in arm_ik.q_index:
            q_full[arm_ik.q_index[name]] = q29[idx]
    data = model.createData()
    pin.framesForwardKinematics(model, data, q_full)
    T_p = data.oMf[model.getFrameId(ik.PELVIS_FRAME)]
    T_w = data.oMf[model.getFrameId(arm_ik.wrist_frame)]
    T = T_p.inverse() * T_w
    quat_xyzw = Rotation.from_matrix(T.rotation).as_quat()
    return np.asarray(T.translation), np.roll(quat_xyzw, 1)


def taskspace_rows(T=4):
    """(T,25) rows whose wrist targets are FK of arm poses a little away
    from the measured ones, so each row is a real (reachable) IK problem."""
    q29 = wbc_driver._q29(Q43)
    s = solver()
    rows = np.zeros((T, 25), dtype=np.float32)
    for t in range(T):
        q = q29.copy()
        q[15:22] = LEFT_ARM + (t + 1) * np.array([0.03, 0.0, 0.02, 0.04, 0.0, 0.01, 0.0])
        q[22:29] = RIGHT_ARM + (t + 1) * np.array([-0.03, 0.0, -0.02, 0.04, 0.0, 0.0, 0.01])
        lp, lq = _wrist_pose_in_pelvis(s.left, q)
        rp, rq = _wrist_pose_in_pelvis(s.right, q)
        rows[t, 0:2] = 0.3
        rows[t, 2:4] = -0.7
        rows[t, 4:7] = lp
        rows[t, 7:11] = lq
        rows[t, 11:14] = rp
        rows[t, 14:18] = rq
        rows[t, 18:21] = [0.05 * t, 0.0, 0.01]
        rows[t, 21] = 0.74
    return rows


def taskspace_msg(rows, issued_at=None) -> bytes:
    rows = np.asarray(rows, dtype=np.float32)
    return b"taskspace" + msgpack.packb({
        "actions": rows.tobytes(), "shape": list(rows.shape), "dtype": "f32",
        "issued_at": time.time() if issued_at is None else issued_at,
    }, use_bin_type=True)


def _load_baseline_module():
    import os
    rev = os.environ.get("WBC_DRIVER_BASELINE_REV", BASELINE_REV)
    try:
        src = subprocess.run(
            ["git", "show", f"{rev}:reference/wbc_adapter/wbc_driver.py"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise unittest.SkipTest(
            f"baseline wbc_driver.py at {rev} not available from git ({exc}); "
            "the taskspace regression proof cannot run") from exc
    if "_handle_joint" in src:
        raise AssertionError(f"{rev} already contains the joint lane; the regression "
                             "test would be comparing the new code with itself")
    tmp = tempfile.mkdtemp(prefix="wbc_driver_baseline_")
    path = Path(tmp) / "wbc_driver_baseline.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("wbc_driver_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_goals_equal(tc, new, old, label):
    tc.assertEqual(len(new), len(old), f"{label}: goal count")
    for k, (gn, go) in enumerate(zip(new, old)):
        tc.assertEqual(set(gn), set(go), f"{label}: goal {k} keys")
        for key in gn:
            if key == "target_time":
                if isinstance(gn[key], list):
                    tc.assertIsInstance(go[key], list)
                    tc.assertEqual(len(gn[key]), len(go[key]))
                    np.testing.assert_allclose(np.diff(gn[key]), np.diff(go[key]), atol=1e-9,
                                               err_msg=f"{label}: goal {k} target_time spacing")
                else:
                    tc.assertNotIsInstance(go[key], list)
                continue
            a, b = np.asarray(gn[key], dtype=np.float64), np.asarray(go[key], dtype=np.float64)
            tc.assertEqual(a.shape, b.shape, f"{label}: goal {k} {key} shape")
            np.testing.assert_allclose(a, b, atol=1e-9, rtol=0,
                                       err_msg=f"{label}: goal {k} {key}")


class TaskspaceRegression(unittest.TestCase):
    """Same taskspace chunks, same state, same settings: the goals this
    checkout produces must equal the pre-joint-lane driver's to 1e-9."""

    @classmethod
    def setUpClass(cls):
        cls.baseline = _load_baseline_module()
        cls.chunk_a = taskspace_rows(T=4)
        cls.chunk_b = taskspace_rows(T=3)
        cls.chunk_b[:, 4:7] += 0.01     # a second, different chunk
        cls.chunk_b[:, 0:4] = [0.9, 0.9, -0.2, -0.2]

    def _script(self):
        return [lambda: taskspace_msg(self.chunk_a), "again",
                lambda: taskspace_msg(self.chunk_b), "again"]

    def test_goals_identical_to_baseline(self):
        args = make_args()
        new_backend, new_stats = drive(wbc_driver, args, self._script())
        old_backend, old_stats = drive(self.baseline, args, self._script())
        # 2 chunks + 2 keepalives each; IK really ran and accepted
        self.assertEqual(len(old_backend.goals), 4)
        self.assertEqual(old_stats.waypoints, 7)
        self.assertGreater(old_stats.left_ok + old_stats.right_ok, 0)
        self.assertEqual(len(old_backend.goals[0]["target_upper_body_pose"]), 4)
        self.assertIn("wrist_pose", old_backend.goals[0])
        _assert_goals_equal(self, new_backend.goals, old_backend.goals, "taskspace")
        for field in ("messages", "rejected", "stale", "waypoints", "left_ok",
                      "right_ok", "published", "gripper_sent"):
            self.assertEqual(getattr(new_stats, field), getattr(old_stats, field), field)

    def test_goals_identical_with_clamp_off_and_in_bench_mode(self):
        for over in (dict(max_joint_vel=0.0),
                     dict(upper_body_from_model=False, max_joint_vel=1.0)):
            args = make_args(**over)
            new_backend, _ = drive(wbc_driver, args, self._script())
            old_backend, _ = drive(self.baseline, args, self._script())
            self.assertEqual(len(old_backend.goals), 4)
            _assert_goals_equal(self, new_backend.goals, old_backend.goals, f"taskspace {over}")

    def test_rejections_identical_to_baseline(self):
        bad = taskspace_rows(T=2)
        bad[0, 7:11] = [0.5, 0.5, 0.0, 0.0]      # non-unit quaternion
        nan = taskspace_rows(T=2)
        nan[1, 4] = np.nan
        script = [lambda: taskspace_msg(bad), lambda: taskspace_msg(nan),
                  lambda: taskspace_msg(self.chunk_a, issued_at=time.time() - 5.0),
                  b"pose" + b"\x00" * 8,          # not ours, ignored
                  lambda: taskspace_msg(self.chunk_a)]
        with contextlib.redirect_stderr(io.StringIO()):
            new_backend, new_stats = drive(wbc_driver, make_args(), list(script))
            old_backend, old_stats = drive(self.baseline, make_args(), list(script))
        self.assertEqual(old_stats.rejected, 2)
        self.assertEqual(old_stats.stale, 1)
        self.assertEqual(len(old_backend.goals), 1)
        _assert_goals_equal(self, new_backend.goals, old_backend.goals, "rejections")
        for field in ("messages", "rejected", "stale", "published"):
            self.assertEqual(getattr(new_stats, field), getattr(old_stats, field), field)

    def test_joint_topics_are_ignored_by_baseline_and_by_lane_off(self):
        script = [lambda: joint_msg(joint_rows(T=2)),
                  lambda: goto_msg(LEFT_ARM + 0.1, RIGHT_ARM),
                  lambda: taskspace_msg(self.chunk_a)]
        old_backend, _ = drive(self.baseline, make_args(), list(script))
        off_backend, off_stats = drive(wbc_driver, make_args(joint_lane="off"), list(script))
        on_backend, on_stats = drive(wbc_driver, make_args(), list(script))
        self.assertEqual(len(old_backend.goals), 1)
        self.assertEqual(len(off_backend.goals), 1)
        self.assertEqual(off_stats.joint_accepted, 0)
        _assert_goals_equal(self, off_backend.goals, old_backend.goals, "lane off")
        self.assertEqual(len(on_backend.goals), 3)
        self.assertEqual(on_stats.joint_accepted, 1)
        self.assertEqual(on_stats.goto_accepted, 1)
        # the taskspace goal is the same whether or not joint traffic preceded it
        _assert_goals_equal(self, [on_backend.goals[2]], old_backend.goals, "lane on")


class TaskspaceGripperRelay(unittest.TestCase):
    def test_taskspace_relays_row0_hands(self):
        dex1 = FakeDex1()
        ctx = make_ctx(dex1=dex1)
        wbc_driver._handle_taskspace(ctx, taskspace_msg(taskspace_rows(T=2)))
        self.assertEqual(len(dex1.sent), 1)
        self.assertAlmostEqual(dex1.sent[0]["left"], 0.3, places=6)
        self.assertAlmostEqual(dex1.sent[0]["right"], -0.7, places=6)


# ---------------------------------------------------------------------------
# 7. client library
# ---------------------------------------------------------------------------


class ClientLibrary(unittest.TestCase):
    def test_make_rows_layout(self):
        from boundary.actions import JOINT_SLICES, JointSink, TASKSPACE_SLICES
        T = 3
        left = np.tile(LEFT_ARM, (T, 1))
        right = np.tile(RIGHT_ARM, (T, 1))
        lh = np.full((T, 2), 0.25)
        rh = np.full((T, 2), -0.75)
        nav = np.tile([0.1, 0.2, 0.3], (T, 1))
        bh = np.array([0.7, 0.71, 0.72])
        rows = JointSink.make_rows(left, right, lh, rh, nav, bh)
        self.assertEqual(rows.shape, (T, 22))
        self.assertEqual(rows.dtype, np.float32)
        np.testing.assert_allclose(rows[:, JOINT_SLICES["left_hand"]], lh)
        np.testing.assert_allclose(rows[:, JOINT_SLICES["right_hand"]], rh)
        np.testing.assert_allclose(rows[:, JOINT_SLICES["left_arm"]], left, atol=1e-6)
        np.testing.assert_allclose(rows[:, JOINT_SLICES["right_arm"]], right, atol=1e-6)
        np.testing.assert_allclose(rows[:, JOINT_SLICES["navigate_cmd"]], nav, atol=1e-6)
        np.testing.assert_allclose(rows[:, JOINT_SLICES["base_height_cmd"]][:, 0], bh, atol=1e-6)
        # explicit column positions, and the hand/nav/base slots the taskspace lane uses
        np.testing.assert_allclose(rows[0, 4:11], LEFT_ARM, atol=1e-6)
        np.testing.assert_allclose(rows[0, 11:18], RIGHT_ARM, atol=1e-6)
        self.assertEqual(JOINT_SLICES["left_hand"], TASKSPACE_SLICES["left_hand"])
        self.assertEqual(JOINT_SLICES["navigate_cmd"], TASKSPACE_SLICES["navigate_cmd"])
        self.assertEqual(JOINT_SLICES["base_height_cmd"], TASKSPACE_SLICES["base_height_cmd"])
        # defaults: hands open, base commands zero; single-row inputs allowed
        one = JointSink.make_rows(LEFT_ARM, RIGHT_ARM)
        self.assertEqual(one.shape, (1, 22))
        np.testing.assert_allclose(one[0, 0:4], -1.0)
        np.testing.assert_allclose(one[0, 18:22], 0.0)
        self.assertEqual(JointSink.validate_chunk(rows).shape, (T, 22))

    def test_make_rows_rejects_mismatch(self):
        from boundary.actions import ActionError, JointSink
        with self.assertRaises(ActionError):
            JointSink.make_rows(np.zeros((2, 7)), np.zeros((3, 7)))
        with self.assertRaises(ActionError):
            JointSink.make_rows(np.zeros((2, 6)), np.zeros((2, 7)))
        with self.assertRaises(ActionError):
            JointSink.make_rows(np.zeros((2, 7)), np.zeros((2, 7)), left_hand=np.zeros((1, 2)))

    def test_validate_chunk_mirrors_adapter(self):
        from boundary.actions import ActionError, JointSink
        rows = joint_rows(T=2)
        for mutate in (lambda r: r.__setitem__((0, 5), np.nan),
                       lambda r: r.__setitem__((1, 0), 1.5)):
            bad = rows.copy()
            mutate(bad)
            with self.assertRaises(ActionError):
                JointSink.validate_chunk(bad)
        with self.assertRaises(ActionError):
            JointSink.validate_chunk(np.zeros((65, 22), dtype=np.float32))
        with self.assertRaises(ActionError):
            JointSink.validate_chunk(np.zeros((2, 25), dtype=np.float32))

    def test_client_bytes_decode_in_adapter(self):
        from boundary import actions
        rows = joint_rows(T=3)
        chunk = boundary_wire.decode_joint(actions._pack_joint_chunk(rows, 123.5))
        np.testing.assert_array_equal(chunk.actions, rows)
        self.assertEqual(chunk.issued_at, 123.5)
        self.assertEqual(boundary_wire.validate_joint(chunk), [])
        req = boundary_wire.decode_goto(actions._pack_goto(LEFT_ARM, RIGHT_ARM, 0.3, (0.5, -0.5), 7.0))
        np.testing.assert_allclose(req.left_arm, LEFT_ARM)
        np.testing.assert_allclose(req.right_arm, RIGHT_ARM)
        self.assertEqual(req.max_speed, 0.3)
        np.testing.assert_allclose(req.hands, [0.5, -0.5])
        self.assertEqual(boundary_wire.validate_goto(req), [])
        req = boundary_wire.decode_goto(actions._pack_goto(LEFT_ARM, RIGHT_ARM, 0.3, None, 7.0))
        self.assertIsNone(req.hands)
        with self.assertRaises(actions.ActionError):
            actions._pack_goto(LEFT_ARM[:6], RIGHT_ARM, 0.3, None, None)
        with self.assertRaises(actions.ActionError):
            actions._pack_goto(LEFT_ARM, RIGHT_ARM, 0.0, None, None)

    def test_for_lane_knows_joint(self):
        import random
        from boundary.actions import ActionError, ActionSink, JointSink
        with self.assertRaises(ActionError):
            ActionSink.for_lane("nonsense")
        sink = ActionSink.for_lane("joint", port=random.randint(20000, 40000), host="127.0.0.1")
        try:
            self.assertIsInstance(sink, JointSink)
            self.assertEqual(sink.lane, "joint")
            sink.send_chunk(joint_rows(T=2))              # validates and publishes
            sink.send_goto(LEFT_ARM, RIGHT_ARM, max_speed=0.3, hands=(-1.0, -1.0))
            self.assertEqual(sink.published, 2)
            with self.assertRaises(ActionError):
                sink.send_goto(LEFT_ARM, RIGHT_ARM, max_speed=-1.0)
        finally:
            sink.close()

    def test_arms_reached(self):
        from boundary.actions import ActionError, arms_reached
        q29 = wbc_driver._q29(Q43)
        self.assertTrue(arms_reached(q29, LEFT_ARM, RIGHT_ARM))
        self.assertTrue(arms_reached(q29, LEFT_ARM + 0.04, RIGHT_ARM - 0.04, tol_rad=0.05))
        self.assertFalse(arms_reached(q29, LEFT_ARM + 0.06, RIGHT_ARM, tol_rad=0.05))
        self.assertFalse(arms_reached(q29, LEFT_ARM, RIGHT_ARM + np.eye(7)[6] * 0.1))
        with self.assertRaises(ActionError):
            arms_reached(Q43, LEFT_ARM, RIGHT_ARM)     # 43-wide is the WBC's q, not body_q
        with self.assertRaises(ActionError):
            arms_reached(q29, LEFT_ARM[:6], RIGHT_ARM)


if __name__ == "__main__":
    unittest.main()
