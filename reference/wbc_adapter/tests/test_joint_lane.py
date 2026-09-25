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
        item = item() if callable(item) else item
        if item == "again":
            raise zmq.Again()
        return item


def _again_after(seconds: float):
    """Script item: wait, then time out -- a keepalive fired that much later."""
    def _item():
        time.sleep(seconds)
        return "again"
    return _item


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
    def test_limits_are_the_raw_urdf_values(self):
        import pinocchio as pin
        lim = wbc_driver._load_arm_limits("urdf", mapper(), solver().settings)
        # the numbers must be the URDF's: re-read the file itself, independently
        urdf, _ = _ik_urdf()
        raw = pin.buildModelFromUrdf(str(urdf))
        for j, name in enumerate(lim.names):
            q = raw.joints[raw.getJointId(name)].idx_q
            self.assertEqual(lim.lower[j], raw.lowerPositionLimit[q], name)
            self.assertEqual(lim.upper[j], raw.upperPositionLimit[q], name)
        self.assertEqual(lim.names[3], "left_elbow_joint")
        self.assertEqual(lim.names[10], "right_elbow_joint")
        # NOT the WBC RobotModel's narrowed arrays: left shoulder_roll goes
        # well below 0.19, right shoulder_roll is the mirrored URDF value
        m = mapper().model
        self.assertLess(lim.lower[1], 0.0)
        self.assertGreater(m.lower_joint_limits[m.joint_to_dof_index["left_shoulder_roll_joint"]], 0.0)
        np.testing.assert_allclose([lim.lower[1], lim.upper[1]], [-1.5882, 2.2515], atol=1e-4)
        np.testing.assert_allclose([lim.lower[8], lim.upper[8]], [-2.2515, 1.5882], atol=1e-4)

    def test_shoulder_roll_near_the_torso_passes_in_both_modes(self):
        for mode in ("urdf", "ik"):
            ctx = make_ctx(limits=mode, max_joint_vel=0)
            rows = joint_rows(T=3)
            rows[0, 5] = 0.10          # left shoulder_roll, inside the URDF, under the 0.19 narrowing
            rows[1, 5] = 0.0
            rows[2, 5] = 0.175         # the raised-arm measurement echoed back
            rows[:, 12] = -0.10        # right shoulder_roll
            with contextlib.redirect_stderr(io.StringIO()) as err:
                wbc_driver._handle_joint(ctx, joint_msg(rows))
            wps = ctx.backend.goals[0]["target_upper_body_pose"]
            for t, wp in enumerate(wps):
                self.assertAlmostEqual(wp[1], rows[t, 5], places=6, msg=mode)
                self.assertAlmostEqual(wp[15], -0.10, places=6, msg=mode)
            self.assertEqual(ctx.stats.joints_clamped, 0, mode)
            self.assertEqual(err.getvalue(), "", mode)

    def test_beyond_the_urdf_clamps_in_both_modes(self):
        for mode in ("urdf", "ik"):
            ctx = make_ctx(limits=mode, max_joint_vel=0)
            rows = joint_rows(T=1)
            rows[0, 5] = 2.4           # left shoulder_roll past 2.2515
            rows[0, 12] = 1.7          # right shoulder_roll past its mirrored 1.5882
            with contextlib.redirect_stderr(io.StringIO()):
                wbc_driver._handle_joint(ctx, joint_msg(rows))
            wp = ctx.backend.goals[0]["target_upper_body_pose"][0]
            lim = ctx.arm_limits
            self.assertAlmostEqual(wp[1], lim.upper[1] - lim.MARGIN, places=12, msg=mode)
            self.assertAlmostEqual(wp[15], lim.upper[8] - lim.MARGIN, places=12, msg=mode)
            self.assertAlmostEqual(lim.upper[1], 2.2515, places=4)
            self.assertAlmostEqual(lim.upper[8], 1.5882, places=4)
            self.assertEqual(ctx.stats.joints_clamped, 2, mode)

    def test_elbow_between_urdf_and_ik_override(self):
        s = solver().settings
        for mode, expect in (("urdf", 1.6), ("ik", s.elbow_upper_limit_override - 1e-3)):
            ctx = make_ctx(limits=mode, max_joint_vel=0)
            rows = joint_rows(T=1)
            rows[0, 7] = 1.6           # left elbow: inside the URDF (2.0944), over the IK override (1.4)
            with contextlib.redirect_stderr(io.StringIO()):
                wbc_driver._handle_joint(ctx, joint_msg(rows))
            wp = ctx.backend.goals[0]["target_upper_body_pose"][0]
            self.assertAlmostEqual(wp[3], expect, places=6, msg=mode)
            self.assertEqual(ctx.stats.joints_clamped, 0 if mode == "urdf" else 1, mode)

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

    def test_small_trims_are_applied_but_not_reported(self):
        ctx = make_ctx(max_joint_vel=0)
        lim = ctx.arm_limits
        lo = lim.lower[1] + lim.MARGIN                 # left shoulder_roll floor, 0.191
        rows = joint_rows(T=3)
        rows[:, 5] = lo - 0.0025                       # the rest pose: 2.5 mrad under
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            wbc_driver._handle_joint(ctx, joint_msg(rows))
        for wp in ctx.backend.goals[0]["target_upper_body_pose"]:
            self.assertAlmostEqual(wp[1], lo, places=12)   # clamped exactly as before
        self.assertEqual(ctx.stats.joints_clamped, 0)      # but not counted
        self.assertEqual(err.getvalue(), "")               # nor logged
        # a 0.1 rad trim on the same joint is intent: counted, logged once
        rows[:, 5] = lo - 0.1
        with contextlib.redirect_stderr(err):
            wbc_driver._handle_joint(ctx, joint_msg(rows))
            wbc_driver._handle_joint(ctx, joint_msg(rows))
        for goal in ctx.backend.goals[1:]:
            for wp in goal["target_upper_body_pose"]:
                self.assertAlmostEqual(wp[1], lo, places=12)
        self.assertEqual(ctx.stats.joints_clamped, 6)
        self.assertEqual(err.getvalue().count("left_shoulder_roll_joint commanded"), 1)
        self.assertGreater(0.0025, 0.0)
        self.assertLess(0.0025, wbc_driver.CLAMP_REPORT_THRESHOLD_RAD)

    def test_ik_mode_applies_the_solver_overrides(self):
        """With the default IKSettings, `ik` mode differs from `urdf` by the
        elbow bound only: wrist_roll_limit_override is None (a posture
        weight steers the IK instead), so wrist_roll keeps the raw URDF
        +-1.9722 on both lanes. An explicit float restores the old cap."""
        s = solver().settings
        self.assertIsNone(s.wrist_roll_limit_override)
        urdf = wbc_driver._load_arm_limits("urdf", mapper(), s)
        ikm = wbc_driver._load_arm_limits("ik", mapper(), s)
        for off in (0, 7):
            self.assertEqual(ikm.upper[off + 3], s.elbow_upper_limit_override)
            self.assertEqual(ikm.upper[off + 3], 1.4)
            self.assertGreater(urdf.upper[off + 3], ikm.upper[off + 3])
            self.assertEqual(ikm.upper[off + 4], urdf.upper[off + 4])
            self.assertEqual(ikm.lower[off + 4], urdf.lower[off + 4])
            np.testing.assert_allclose([ikm.lower[off + 4], ikm.upper[off + 4]],
                                       [-1.9722, 1.9722], atol=1e-4)
        # every other joint is identical between the two modes
        diff = np.flatnonzero((ikm.upper != urdf.upper) | (ikm.lower != urdf.lower))
        self.assertEqual(list(diff), [3, 10])
        ctx = make_ctx(limits="ik", max_joint_vel=0)
        rows = joint_rows(T=1)
        rows[0, 7] = 1.9      # left elbow: inside the URDF, outside the IK override
        rows[0, 8] = 1.5      # left wrist_roll: inside the URDF, over the OLD 0.9 cap
        with contextlib.redirect_stderr(io.StringIO()):
            wbc_driver._handle_joint(ctx, joint_msg(rows))
        wp = ctx.backend.goals[0]["target_upper_body_pose"][0]
        self.assertAlmostEqual(wp[3], s.elbow_upper_limit_override - ikm.MARGIN, places=12)
        self.assertAlmostEqual(wp[4], 1.5, places=6)     # passes through untouched
        self.assertEqual(ctx.stats.joints_clamped, 1)

    def test_ik_mode_with_an_explicit_wrist_roll_override(self):
        s = ik.IKSettings(wrist_roll_limit_override=0.9)
        ikm = wbc_driver._load_arm_limits("ik", mapper(), s)
        urdf = wbc_driver._load_arm_limits("urdf", mapper(), s)
        for off in (0, 7):
            self.assertEqual(ikm.upper[off + 3], 1.4)
            self.assertEqual(ikm.upper[off + 4], 0.9)
            self.assertEqual(ikm.lower[off + 4], -0.9)
        # the override never touches `urdf` mode
        np.testing.assert_allclose([urdf.lower[4], urdf.upper[4]], [-1.9722, 1.9722], atol=1e-4)
        self.assertIn("ik.py solver overrides", ikm.source)
        self.assertEqual(ikm.clamp(np.r_[LEFT_ARM[:4], 1.5, LEFT_ARM[5:], RIGHT_ARM])[0][4],
                         0.9 - ikm.MARGIN)


class WristRollPostureWeight(unittest.TestCase):
    """2026-09-23: `IKSettings.wrist_roll_limit_override` defaults to None.
    The old 0.9 rad value was written into the solver model's position
    limits, i.e. a hard QP constraint that made every pose needing more
    wrist roll infeasible; `wrist_roll_posture_weight` (4.0) now steers the
    redundant DOF toward the measured seed instead of forbidding it.

    The targets are FK of a joint configuration with wrist_roll = 1.3 rad
    on both arms (beyond the old cap, inside the URDF's 1.9722) on the same
    model the other tests use. The seed is the MEASURED pose the driver
    hands the solver every call. With the seed already past 0.9 -- what the
    robot reports once it has tracked there -- the old cap projects the
    seed back to 0.9 and the constraint keeps the solution there; the new
    default must track the intended roll. From a seed far below the cap
    (the fixed test pose, wrist_roll 0.10) the 7-DOF arm can reach the same
    6-DOF pose with much less wrist roll, so only acceptance is asserted
    there and the solved value is reported."""

    WR = 1.3
    TOL = 0.3

    @staticmethod
    def _fresh(**over) -> ik.UpperBodyIK:
        urdf, assets = _ik_urdf()
        return ik.UpperBodyIK(ik.IKSettings(max_err=1e-3, **over), include_waist=False,
                              urdf=urdf, assets=assets, warm_start="current")

    @staticmethod
    def _seed(wrist_roll):
        q29 = wbc_driver._q29(Q43)
        q29[19] = wrist_roll      # left_wrist_roll
        q29[26] = wrist_roll      # right_wrist_roll
        return q29

    @classmethod
    def _row(cls, s, wrist_roll):
        q = cls._seed(wrist_roll)
        lp, lq = _wrist_pose_in_pelvis(s.left, q)
        rp, rq = _wrist_pose_in_pelvis(s.right, q)
        row = np.zeros(25)
        row[4:7], row[7:11], row[11:14], row[14:18] = lp, lq, rp, rq
        return row

    def test_default_has_no_hard_wrist_roll_limit(self):
        s = self._fresh()
        self.assertIsNone(s.settings.wrist_roll_limit_override)
        self.assertEqual(s.settings.wrist_roll_posture_weight, 4.0)
        for arm in (s.left, s.right):
            j = arm.q_index[f"{arm.side}_wrist_roll_joint"]
            np.testing.assert_allclose([arm.model.lowerPositionLimit[j], arm.model.upperPositionLimit[j]],
                                       [-1.9722, 1.9722], atol=1e-4)
            e = arm.q_index[f"{arm.side}_elbow_joint"]
            self.assertEqual(arm.model.upperPositionLimit[e], 1.4)      # elbow override unchanged

    def test_default_accepts_and_tracks_wrist_roll_beyond_the_old_cap(self):
        s = self._fresh()
        row = self._row(s, self.WR)
        res = s.solve_row(row, self._seed(1.25))
        self.assertTrue(res.left_ok and res.right_ok, (res.left_err, res.right_err))
        got = (res.upper_body[4], res.upper_body[11])
        print(f"\n[wrist_roll posture weight] seed 1.25 -> target {self.WR}: "
              f"solved L={got[0]:+.3f} R={got[1]:+.3f}")
        for v in got:
            self.assertGreater(v, 0.9)                     # past the old cap
            self.assertLess(abs(v - self.WR), self.TOL)

    def test_default_accepts_from_a_far_seed_and_reports_the_solved_roll(self):
        s = self._fresh()
        row = self._row(s, self.WR)
        res = s.solve_row(row, wbc_driver._q29(Q43))       # measured wrist_roll 0.10 / -0.10
        self.assertTrue(res.left_ok and res.right_ok, (res.left_err, res.right_err))
        got = (res.upper_body[4], res.upper_body[11])
        print(f"\n[wrist_roll posture weight] seed 0.10 -> target {self.WR}: "
              f"solved L={got[0]:+.3f} R={got[1]:+.3f} (redundant DOF; accept only)")
        for v in got:
            self.assertGreater(v, 0.10)                    # moved toward the target, not held at the seed

    def test_explicit_override_restores_the_hard_cap(self):
        s = self._fresh(wrist_roll_limit_override=0.9)
        for arm in (s.left, s.right):
            j = arm.q_index[f"{arm.side}_wrist_roll_joint"]
            self.assertEqual(arm.model.upperPositionLimit[j], 0.9)
            self.assertEqual(arm.model.lowerPositionLimit[j], -0.9)
        row = self._row(s, self.WR)
        res = s.solve_row(row, self._seed(1.25))
        print(f"\n[wrist_roll hard cap 0.9] seed 1.25 -> target {self.WR}: "
              f"L ok={res.left_ok} q={res.upper_body[4]:+.3f} R ok={res.right_ok} q={res.upper_body[11]:+.3f}")
        # old behaviour: the solution never leaves +-0.9 -- either the arm
        # is rejected (held at the measured 1.25) or accepted inside the cap
        for ok, v in ((res.left_ok, res.upper_body[4]), (res.right_ok, res.upper_body[11])):
            if ok:
                self.assertLessEqual(abs(v), 0.9 + 1e-6)
            else:
                self.assertAlmostEqual(v, 1.25)             # held: measured pose passed through


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
        self.assertIsNotNone(ctx.in_flight)
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
        # keepalive while the whole trajectory is still ahead: the next 2 s
        # of it, as a prefix -- same waypoints, same times
        self.assertTrue(wbc_driver._publish_keepalive(ctx))
        ka = ctx.backend.goals[1]
        k = len(ka["target_time"])
        self.assertGreaterEqual(k, 1)
        self.assertLessEqual(k, int(wbc_driver.KEEPALIVE_WINDOW_S * CHUNK_HZ) + 1)
        np.testing.assert_allclose(np.asarray(ka["target_time"]),
                                   np.asarray(goto_goal["target_time"][:k]))
        np.testing.assert_allclose(np.asarray(ka["target_upper_body_pose"]),
                                   np.asarray(goto_goal["target_upper_body_pose"][:k]))
        self.assertEqual(ctx.keepalives_sent, 1)
        # half way through: only the future half is re-sent
        now = time.monotonic()
        ctx.in_flight["times"] = [now - 1.0 + i * DT for i in range(60)]
        wbc_driver._publish_keepalive(ctx)
        ka2 = ctx.backend.goals[2]
        self.assertTrue(all(now < t <= now + wbc_driver.KEEPALIVE_WINDOW_S + 1e-6
                            for t in ka2["target_time"]))
        self.assertLess(len(ka2["target_time"]), 60)
        self.assertGreater(len(ka2["target_time"]), 0)
        # trajectory done: plain single-row hold of the final pose
        ctx.in_flight["times"] = [now - 10.0 + i * DT for i in range(60)]
        wbc_driver._publish_keepalive(ctx)
        ka3 = ctx.backend.goals[3]
        self.assertIsNone(ctx.in_flight)
        # the pre-existing hold form: a one-waypoint trajectory at t+1/chunk_hz
        self.assertEqual(len(ka3["target_time"]), 1)
        self.assertGreater(ka3["target_time"][0], now)
        np.testing.assert_allclose(ka3["target_upper_body_pose"][0],
                                   goto_goal["target_upper_body_pose"][-1])
        # a later chunk supersedes whatever goto was in flight
        with contextlib.redirect_stdout(io.StringIO()):
            wbc_driver._handle_goto(ctx, goto_msg(self.TARGET_LEFT, self.TARGET_RIGHT, max_speed=0.3))
        self.assertEqual(len(ctx.in_flight["times"]), 60)
        wbc_driver._handle_joint(ctx, joint_msg(joint_rows(T=2)))
        self.assertEqual(len(ctx.in_flight["times"]), 2)     # the chunk's own trajectory now


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
        # chunk, immediate keepalive, chunk, immediate keepalive, then a
        # keepalive after the last chunk (3 rows = 0.15 s) has played out
        return [lambda: taskspace_msg(self.chunk_a), "again",
                lambda: taskspace_msg(self.chunk_b), "again", _again_after(0.3)]

    def _check_keepalives(self, new_goals, old_goals, label):
        """Chunk goals identical old vs new; keepalives differ by design:
        old = one-row hold, new = the chunk's future tail with original
        times, then the hold once the tail is exhausted."""
        self.assertEqual(len(new_goals), 5, label)
        self.assertEqual(len(old_goals), 5, label)
        _assert_goals_equal(self, [new_goals[0], new_goals[2]],
                            [old_goals[0], old_goals[2]], label + " chunks")
        for k in (1, 3):
            self.assertEqual(len(old_goals[k]["target_time"]), 1, label)   # old: hold
            chunk, ka = new_goals[k - 1], new_goals[k]
            n = len(ka["target_time"])
            self.assertGreaterEqual(n, 1, label)
            self.assertLessEqual(n, len(chunk["target_time"]), label)
            # the tail: last n waypoints of the chunk, ORIGINAL times
            np.testing.assert_allclose(ka["target_time"], chunk["target_time"][-n:])
            np.testing.assert_allclose(np.asarray(ka["target_upper_body_pose"]),
                                       np.asarray(chunk["target_upper_body_pose"][-n:]))
            np.testing.assert_allclose(np.asarray(ka["base_height_command"]),
                                       np.asarray(chunk["base_height_command"][-n:]))
            np.testing.assert_allclose(np.asarray(ka["navigate_cmd"]),
                                       np.asarray(chunk["navigate_cmd"][-n:]))
        # after chunk_b has played out: the hold form, final pose, fresh time
        hold, old_hold = new_goals[4], old_goals[4]
        self.assertEqual(len(hold["target_time"]), 1, label)
        self.assertGreater(hold["target_time"][0], new_goals[2]["target_time"][-1])
        for key in ("target_upper_body_pose", "base_height_command", "navigate_cmd", "wrist_pose"):
            np.testing.assert_allclose(np.asarray(hold[key]), np.asarray(old_hold[key]),
                                       atol=1e-9, err_msg=f"{label}: hold {key}")

    def test_goals_identical_to_baseline(self):
        args = make_args()
        new_backend, new_stats = drive(wbc_driver, args, self._script())
        old_backend, old_stats = drive(self.baseline, args, self._script())
        # 2 chunks + 3 keepalives each; IK really ran and accepted
        self.assertEqual(len(old_backend.goals), 5)
        self.assertEqual(old_stats.waypoints, 7)
        self.assertGreater(old_stats.left_ok + old_stats.right_ok, 0)
        self.assertEqual(len(old_backend.goals[0]["target_upper_body_pose"]), 4)
        self.assertIn("wrist_pose", old_backend.goals[0])
        self._check_keepalives(new_backend.goals, old_backend.goals, "taskspace")
        for field in ("messages", "rejected", "stale", "waypoints", "left_ok",
                      "right_ok", "published", "gripper_sent"):
            self.assertEqual(getattr(new_stats, field), getattr(old_stats, field), field)

    def test_goals_identical_with_clamp_off_and_in_bench_mode(self):
        for over in (dict(max_joint_vel=0.0),
                     dict(upper_body_from_model=False, max_joint_vel=1.0)):
            args = make_args(**over)
            new_backend, _ = drive(wbc_driver, args, self._script())
            old_backend, _ = drive(self.baseline, args, self._script())
            self._check_keepalives(new_backend.goals, old_backend.goals, f"taskspace {over}")

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


class ChunkKeepalive(unittest.TestCase):
    """The keepalive re-sends a chunk's not-yet-due waypoints with their
    original times; only after the last one is due does it hold."""

    def _run(self, ctx, send):
        send()
        chunk = ctx.backend.goals[-1]
        times = list(chunk["target_time"])
        T = len(times)
        # immediately: everything is still ahead -> the whole chunk, same times
        now = time.monotonic()
        self.assertTrue(wbc_driver._publish_keepalive(ctx))
        ka = ctx.backend.goals[-1]
        self.assertEqual(len(ka["target_time"]), T)
        np.testing.assert_allclose(ka["target_time"], times)
        np.testing.assert_allclose(np.asarray(ka["target_upper_body_pose"]),
                                   np.asarray(chunk["target_upper_body_pose"]))
        self.assertTrue(all(t > now for t in ka["target_time"]))
        # mid-chunk: rows 0 and 1 are past, row 2 is due exactly now (not
        # future), so only rows 3.. come back, with their original times
        now = time.monotonic()
        ctx.in_flight["times"] = [now - 2 * DT + i * DT for i in range(T)]
        wbc_driver._publish_keepalive(ctx)
        ka = ctx.backend.goals[-1]
        self.assertEqual(len(ka["target_time"]), T - 3)
        np.testing.assert_allclose(ka["target_time"], ctx.in_flight["times"][3:])
        np.testing.assert_allclose(np.asarray(ka["target_upper_body_pose"]),
                                   np.asarray(chunk["target_upper_body_pose"][3:]))
        np.testing.assert_allclose(np.asarray(ka["base_height_command"]),
                                   np.asarray(chunk["base_height_command"][3:]))
        np.testing.assert_allclose(np.asarray(ka["navigate_cmd"]),
                                   np.asarray(chunk["navigate_cmd"][3:]))
        self.assertTrue(all(t > now for t in ka["target_time"]))
        # exactly-now is not future
        ctx.in_flight["times"] = [time.monotonic() - 1.0] * (T - 1) + [time.monotonic()]
        n_before = len(ctx.backend.goals)
        wbc_driver._publish_keepalive(ctx)
        self.assertIsNone(ctx.in_flight)
        hold = ctx.backend.goals[-1]
        self.assertEqual(len(ctx.backend.goals), n_before + 1)
        # after the last time: the hold form -- one row, final pose, now + dt
        self.assertEqual(len(hold["target_time"]), 1)
        self.assertGreater(hold["target_time"][0], time.monotonic())
        np.testing.assert_allclose(hold["target_upper_body_pose"][0],
                                   chunk["target_upper_body_pose"][-1])
        np.testing.assert_allclose(hold["base_height_command"][0], chunk["base_height_command"][-1])
        np.testing.assert_allclose(hold["navigate_cmd"][0], chunk["navigate_cmd"][-1])
        # stays the hold from now on
        wbc_driver._publish_keepalive(ctx)
        self.assertEqual(len(ctx.backend.goals[-1]["target_time"]), 1)
        for g in ctx.backend.goals[1:]:
            self.assertNotIn("goto", g)
        return chunk

    def test_joint_chunk(self):
        ctx = make_ctx()
        chunk = self._run(ctx, lambda: wbc_driver._handle_joint(ctx, joint_msg(joint_rows(T=6))))
        self.assertEqual(len(chunk["target_time"]), 6)
        self.assertNotIn("wrist_pose", ctx.backend.goals[1])

    def test_taskspace_chunk(self):
        ctx = make_ctx()
        chunk = self._run(ctx, lambda: wbc_driver._handle_taskspace(
            ctx, taskspace_msg(taskspace_rows(T=4))))
        self.assertEqual(len(chunk["target_time"]), 4)
        # the tail carries the chunk's wrist_pose, like the hold does
        np.testing.assert_allclose(ctx.backend.goals[1]["wrist_pose"], chunk["wrist_pose"])

    def test_never_publishes_a_waypoint_at_or_before_now(self):
        ctx = make_ctx()
        wbc_driver._handle_joint(ctx, joint_msg(joint_rows(T=8)))
        base = ctx.in_flight["times"]
        for shift in (0.0, 0.05, 0.12, 0.3, 0.39, 0.4, 1.0):
            now = time.monotonic()
            ctx.in_flight = {"upper_body": ctx.backend.goals[0]["target_upper_body_pose"],
                             "times": [t - (base[0] - now) - shift for t in base],
                             "base_heights": ctx.backend.goals[0]["base_height_command"],
                             "nav_cmds": ctx.backend.goals[0]["navigate_cmd"]}
            wbc_driver._publish_keepalive(ctx)
            ka = ctx.backend.goals[-1]
            self.assertTrue(all(t > now for t in ka["target_time"]), shift)
            if ctx.in_flight is None:
                self.assertEqual(len(ka["target_time"]), 1)


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
        # base commands default to zero; single-row inputs allowed; hands required
        one = JointSink.make_rows(LEFT_ARM, RIGHT_ARM, [1.0, 1.0], [-1.0, -1.0])
        self.assertEqual(one.shape, (1, 22))
        np.testing.assert_allclose(one[0, 0:4], [1.0, 1.0, -1.0, -1.0])
        np.testing.assert_allclose(one[0, 18:22], 0.0)
        with self.assertRaises(ValueError) as cm:      # ActionError is a ValueError
            JointSink.make_rows(LEFT_ARM, RIGHT_ARM, None, [-1.0, -1.0])
        self.assertIn("-1 = open, +1 = closed", str(cm.exception))
        with self.assertRaises(ValueError):
            JointSink.make_rows(LEFT_ARM, RIGHT_ARM, [1.0, 1.0], None)
        self.assertEqual(JointSink.validate_chunk(rows).shape, (T, 22))

    def test_make_rows_rejects_mismatch(self):
        from boundary.actions import ActionError, JointSink
        h = np.zeros((2, 2))
        with self.assertRaises(ActionError):
            JointSink.make_rows(np.zeros((2, 7)), np.zeros((3, 7)), h, h)
        with self.assertRaises(ActionError):
            JointSink.make_rows(np.zeros((2, 6)), np.zeros((2, 7)), h, h)
        with self.assertRaises(ActionError):
            JointSink.make_rows(np.zeros((2, 7)), np.zeros((2, 7)), np.zeros((1, 2)), h)

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
        # default tolerance 0.10: a PD steady-state offset of 0.066 counts as reached
        self.assertTrue(arms_reached(q29, LEFT_ARM + 0.066, RIGHT_ARM - 0.066))
        self.assertFalse(arms_reached(q29, LEFT_ARM, RIGHT_ARM + np.eye(7)[6] * 0.11))
        with self.assertRaises(ActionError):
            arms_reached(Q43, LEFT_ARM, RIGHT_ARM)     # 43-wide is the WBC's q, not body_q
        with self.assertRaises(ActionError):
            arms_reached(q29, LEFT_ARM[:6], RIGHT_ARM)


# ---------------------------------------------------------------------------
# 8. review fixes
# ---------------------------------------------------------------------------


def raw_goto_msg(left, right, max_speed, hands=None, issued_at="now") -> bytes:
    """A goto frame built without the client's validation, to hit the
    adapter's own checks. issued_at=None omits the key."""
    msg = {"left_arm": [float(v) for v in left], "right_arm": [float(v) for v in right],
           "max_speed": max_speed, "hands": hands}
    if issued_at == "now":
        msg["issued_at"] = time.time()
    elif issued_at is not None:
        msg["issued_at"] = issued_at
    return b"goto" + msgpack.packb(msg, use_bin_type=True)


def raw_joint_msg(rows, issued_at="now") -> bytes:
    rows = np.asarray(rows, dtype=np.float32)
    msg = {"actions": rows.tobytes(), "shape": list(rows.shape), "dtype": "f32"}
    if issued_at == "now":
        msg["issued_at"] = time.time()
    elif issued_at is not None:
        msg["issued_at"] = issued_at
    return b"joint" + msgpack.packb(msg, use_bin_type=True)


class ReviewFixes(unittest.TestCase):
    FAR_LEFT = LEFT_ARM + np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0])

    def test_1_tiny_speed_and_long_moves_are_refused_and_loop_survives(self):
        # handler level: each is rejected and counted, none raises
        for speed, what in ((1e-9, "max_speed"), (0.005, "max_speed"), (0.01, "cap")):
            ctx = make_ctx()
            with contextlib.redirect_stderr(io.StringIO()) as err:
                wbc_driver._handle_goto(ctx, raw_goto_msg(self.FAR_LEFT, RIGHT_ARM, speed))
            self.assertEqual(ctx.backend.goals, [], speed)
            self.assertEqual(ctx.stats.goto_rejected, 1, speed)
            self.assertEqual(ctx.stats.rejected, 1, speed)
            self.assertIn(what, err.getvalue())
        # 0.9 rad at 0.01 rad/s = 90 s: over the 15 s cap even though the speed is legal
        self.assertEqual(boundary_wire.validate_goto(boundary_wire.decode_goto(
            raw_goto_msg(self.FAR_LEFT, RIGHT_ARM, 0.01))), [])
        # a legal, in-cap move right at the floor still works
        ctx = make_ctx()
        with contextlib.redirect_stdout(io.StringIO()):
            wbc_driver._handle_goto(ctx, raw_goto_msg(LEFT_ARM + 0.1, RIGHT_ARM, 0.01))
        self.assertEqual(len(ctx.backend.goals), 1)
        self.assertEqual(len(ctx.backend.goals[0]["target_time"]), 200)   # 10 s
        # loop level: three bad gotos, then a taskspace chunk still publishes
        script = [lambda: raw_goto_msg(self.FAR_LEFT, RIGHT_ARM, 1e-9),
                  lambda: raw_goto_msg(self.FAR_LEFT, RIGHT_ARM, 0.005),
                  lambda: raw_goto_msg(self.FAR_LEFT, RIGHT_ARM, 0.01),
                  lambda: taskspace_msg(taskspace_rows(T=2))]
        with contextlib.redirect_stderr(io.StringIO()):
            backend, stats = drive(wbc_driver, make_args(), script)
        self.assertEqual(len(backend.goals), 1)
        self.assertEqual(stats.goto_rejected, 3)
        self.assertEqual(stats.published, 1)

    def test_1_injected_exception_fails_closed(self):
        real = wbc_driver._position_clamp

        def boom(ctx, arms):
            raise RuntimeError("injected")
        wbc_driver._position_clamp = boom
        try:
            ctx = make_ctx()
            with contextlib.redirect_stderr(io.StringIO()) as err:
                wbc_driver._handle_joint(ctx, joint_msg(joint_rows(T=2)))
                wbc_driver._handle_goto(ctx, goto_msg(self.FAR_LEFT, RIGHT_ARM))
            self.assertEqual(ctx.backend.goals, [])
            self.assertEqual(ctx.stats.joint_rejected, 1)
            self.assertEqual(ctx.stats.goto_rejected, 1)
            self.assertEqual(ctx.stats.rejected, 2)
            self.assertEqual(err.getvalue().count("injected"), 2)
            self.assertIn("holding last safe goal", err.getvalue())
        finally:
            wbc_driver._position_clamp = real
        # and the lane works again afterwards
        wbc_driver._handle_joint(ctx, joint_msg(joint_rows(T=2)))
        self.assertEqual(len(ctx.backend.goals), 1)

    def test_2_goto_max_speed_must_be_positive(self):
        parser = wbc_driver.build_parser()
        for bad in ("0", "-1", "0.0", "nan", "x"):
            with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
                parser.parse_args(["--lane", "decoupled", "--goto-max-speed", bad])
            self.assertIn("goto-max-speed", err.getvalue())
        args = parser.parse_args(["--lane", "decoupled", "--goto-max-speed", "0.5"])
        self.assertEqual(args.goto_max_speed, 0.5)
        self.assertEqual(parser.parse_args(["--lane", "decoupled"]).goto_max_speed, 0.45)

    def test_3_goto_keepalive_resends_a_bounded_window(self):
        ctx = make_ctx()
        with contextlib.redirect_stdout(io.StringIO()):
            wbc_driver._handle_goto(ctx, goto_msg(self.FAR_LEFT, RIGHT_ARM, max_speed=0.075))
        goal = ctx.backend.goals[0]
        self.assertEqual(len(goal["target_time"]), 240)          # 12 s, inside the 15 s cap
        for _ in range(3):
            now = time.monotonic()
            self.assertTrue(wbc_driver._publish_keepalive(ctx))
            ka = ctx.backend.goals[-1]
            self.assertLessEqual(len(ka["target_time"]), int(2.0 * CHUNK_HZ) + 1)
            self.assertGreaterEqual(len(ka["target_time"]), 1)
            self.assertTrue(all(t > now for t in ka["target_time"]))
            np.testing.assert_allclose(np.diff(ka["target_time"]), DT, atol=1e-9)
        # the plain hold (no goto in flight) is the pre-existing one-row form
        ctx.in_flight = None
        wbc_driver._publish_keepalive(ctx)
        self.assertEqual(len(ctx.backend.goals[-1]["target_time"]), 1)

    def test_4_goto_stands_still_at_the_last_height(self):
        ctx = make_ctx()
        rows = taskspace_rows(T=3)
        rows[:, 18:21] = [0.3, 0.0, 0.2]       # walking
        rows[:, 21] = 0.68
        wbc_driver._handle_taskspace(ctx, taskspace_msg(rows))
        np.testing.assert_allclose(ctx.last_goal_template["navigate_cmd"], [0.3, 0.0, 0.2])
        with contextlib.redirect_stdout(io.StringIO()):
            wbc_driver._handle_goto(ctx, goto_msg(self.FAR_LEFT, RIGHT_ARM, max_speed=0.3))
        goal = ctx.backend.goals[1]
        for nav in goal["navigate_cmd"]:
            np.testing.assert_allclose(nav, [0.0, 0.0, 0.0])
        for bh in goal["base_height_command"]:
            np.testing.assert_allclose(bh, [0.68], atol=1e-6)
        # ...and the keepalive after it holds the same
        wbc_driver._publish_keepalive(ctx)
        np.testing.assert_allclose(ctx.backend.goals[2]["navigate_cmd"][0], [0.0, 0.0, 0.0])

    def test_5_make_rows_requires_hands(self):
        from boundary.actions import ActionError, JointSink
        with self.assertRaises(ActionError):
            JointSink.make_rows(LEFT_ARM, RIGHT_ARM, None, None)
        with self.assertRaises(TypeError):
            JointSink.make_rows(LEFT_ARM, RIGHT_ARM)        # no longer optional

    def test_6_missing_or_nan_issued_at_rejected(self):
        for bad in (None, float("nan"), float("inf")):
            ctx = make_ctx()
            with contextlib.redirect_stderr(io.StringIO()) as err:
                wbc_driver._handle_joint(ctx, raw_joint_msg(joint_rows(T=2), issued_at=bad))
                wbc_driver._handle_goto(ctx, raw_goto_msg(LEFT_ARM + 0.1, RIGHT_ARM, 0.3, issued_at=bad))
            self.assertEqual(ctx.backend.goals, [], bad)
            self.assertEqual(ctx.stats.joint_rejected, 1, bad)
            self.assertEqual(ctx.stats.goto_rejected, 1, bad)
            self.assertEqual(ctx.stats.stale, 0, bad)
            self.assertEqual(err.getvalue().count("issued_at missing or not finite"), 2)
        # taskspace keeps its pre-existing behaviour: a missing stamp is age 0
        msg = b"taskspace" + msgpack.packb({"actions": taskspace_rows(T=1).tobytes(),
                                            "shape": [1, 25]}, use_bin_type=True)
        ctx = make_ctx()
        wbc_driver._handle_taskspace(ctx, msg)
        self.assertEqual(len(ctx.backend.goals), 1)

    def test_7_zero_max_waypoints_does_not_raise(self):
        ctx = make_ctx(max_waypoints=0)
        wbc_driver._handle_joint(ctx, joint_msg(joint_rows(T=3)))
        wbc_driver._handle_taskspace(ctx, taskspace_msg(taskspace_rows(T=2)))
        self.assertEqual(ctx.backend.goals, [])
        self.assertEqual(ctx.stats.joint_accepted, 0)
        self.assertEqual(ctx.stats.joint_rejected, 0)

    def test_8_stale_state_refuses_goto_only(self):
        class Stale(FakeBackend):
            def health(self):
                return {"backend": "fake", "state_stale": True}
        ctx = make_ctx(backend=Stale())
        with contextlib.redirect_stderr(io.StringIO()) as err:
            wbc_driver._handle_goto(ctx, goto_msg(self.FAR_LEFT, RIGHT_ARM))
        self.assertEqual(ctx.backend.goals, [])
        self.assertEqual(ctx.stats.goto_rejected, 1)
        self.assertIn("state stale, goto refused", err.getvalue())
        wbc_driver._handle_joint(ctx, joint_msg(joint_rows(T=2)))
        self.assertEqual(len(ctx.backend.goals), 1)

    def test_9_boundary_package_exports_the_lane(self):
        import boundary
        self.assertIn("joint", boundary.LANES)
        self.assertIn("JointSink", boundary.__all__)
        self.assertIs(boundary.JointSink, boundary.actions.JointSink)
        self.assertIn("arms_reached", boundary.__all__)

    def test_10_limits_failure_disables_the_lane_but_not_the_run(self):
        real = wbc_driver._load_arm_limits

        def boom(mode, mapper, settings):
            raise RuntimeError("no limits today")
        wbc_driver._load_arm_limits = boom
        try:
            script = [lambda: joint_msg(joint_rows(T=2)),
                      lambda: goto_msg(self.FAR_LEFT, RIGHT_ARM),
                      lambda: joint_msg(joint_rows(T=2)),
                      lambda: taskspace_msg(taskspace_rows(T=2))]
            with contextlib.redirect_stderr(io.StringIO()) as err:
                backend, stats = drive(wbc_driver, make_args(), script)
        finally:
            wbc_driver._load_arm_limits = real
        self.assertEqual(len(backend.goals), 1)          # the taskspace chunk
        self.assertIn("wrist_pose", backend.goals[0])
        self.assertEqual(stats.joint_rejected, 2)
        self.assertEqual(stats.goto_rejected, 1)
        self.assertEqual(stats.rejected, 3)
        self.assertEqual(stats.published, 1)
        log = err.getvalue()
        self.assertIn("joint lane DISABLED", log)
        self.assertIn("no limits today", log)
        self.assertEqual(log.count("joint lane disabled this run"), 1)   # logged once


if __name__ == "__main__":
    unittest.main()
