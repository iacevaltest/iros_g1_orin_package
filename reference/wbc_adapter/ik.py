"""(T,25) end-effector poses -> the joint-space upper-body vector the WBC wants.

This step exists because NVIDIA's Decoupled WBC does NOT do IK inside its
50 Hz loop -- `G1DecoupledWholeBodyPolicy.set_goal` consumes
`target_upper_body_pose` in joint space, and its upper-body "policy" is
interpolation over those joint targets
(`decoupled_wbc/control/policy/wbc_policy_factory.py`). NVIDIA's own IK
(`control/teleop/solver/body/body_ik_solver.py`, pink + pinocchio +
qpsolvers) runs client-side, upstream of the WBC. So does ours.

--------------------------------------------------------------------------
THE FRAME CONVENTION IS THE WHOLE BALLGAME -- READ THIS
--------------------------------------------------------------------------
The organizer's `(T,25)` contract says cols [4:7]/[11:14] are "end-effector
position (xyz, metres)" and [7:11]/[14:18] the orientation quaternion, but
it does NOT define WHERE on the hand that point is, or in what frame. That
definition is set, in practice, by whatever IK consumes it -- i.e. by this
module. Whatever we target here becomes the contract every team is
implicitly being graded against.

Two consequences, both important:

  1. Use the `decoupled_wbc` backend on the robot whenever possible. It is
     NVIDIA's own solver against NVIDIA's own robot model, so the frame is
     defined by the same stack that executes the motion -- no third
     convention invented by us in the middle.
  2. A team whose checkpoint was trained against a DIFFERENT tool frame
     (e.g. their own kinematics measured the wrist joint while we drive a
     point 4cm further out) will be systematically biased, every step, in a
     way that looks like a bad policy rather than a units mismatch. That is
     a per-team finding to raise with them -- it is NOT something to "fix"
     by quietly bending this module per team. Keep this module
     team-agnostic.

The `pink` backend exists so the pipeline is testable off-robot; it is a
faithful re-implementation, not the authority. Cross-check it against the
`decoupled_wbc` backend on the Orin before trusting it for anything live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Shared with the g1_bridge package -- Unitree's own G1 29-DoF URDF.
DEFAULT_URDF_DIR = Path.home() / "g1_bridge" / "robot" / "assets" / "g1_urdf"
DEFAULT_URDF = DEFAULT_URDF_DIR / "g1_29dof_with_hand.urdf"

PELVIS_FRAME = "pelvis"
WRIST_FRAMES = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}

ARM_JOINTS = {
    "left": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
             "left_shoulder_yaw_joint", "left_elbow_joint",
             "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"],
    "right": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint",
              "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"],
}
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

# Per-joint posture weight, matched to NVIDIA's own
# body_ik_solver_settings.py posture_weight dict (unlisted joints there
# default to 1.0 -- reproduced here for wrist_roll, which NVIDIA doesn't
# list). Keys are joint names with the left_/right_ prefix and _joint
# suffix stripped, e.g. "left_elbow_joint" -> "elbow".
_ARM_POSTURE_WEIGHTS = {
    "shoulder_pitch": 4.0,
    "shoulder_roll": 3.0,
    "shoulder_yaw": 0.1,
    "elbow": 3.0,
    "wrist_roll": 1.0,
    "wrist_pitch": 1.0,
    "wrist_yaw": 0.1,
}

_WeightedPostureTaskCls = None


def _get_weighted_posture_task_cls(pink_module):
    """Lazily build a PostureTask subclass that scales error/jacobian by a
    per-joint weight vector -- reproduces NVIDIA's own WeightedPostureTask
    in body_ik_solver.py exactly (down to the per-row jacobian scaling),
    since it isn't exported as a reusable class from that module."""
    global _WeightedPostureTaskCls
    if _WeightedPostureTaskCls is None:
        class WeightedPostureTask(pink_module.tasks.PostureTask):
            def __init__(self, cost, weights, lm_damping=0.0, gain=1.0):
                super().__init__(cost=cost, lm_damping=lm_damping, gain=gain)
                self.weights = weights

            def compute_error(self, configuration):
                return self.weights * super().compute_error(configuration)

            def compute_jacobian(self, configuration):
                J = super().compute_jacobian(configuration)
                return self.weights[:, np.newaxis] * J

        _WeightedPostureTaskCls = WeightedPostureTask
    return _WeightedPostureTaskCls


@dataclass
class IKResult:
    upper_body: np.ndarray        # (N,) joint targets, WBC upper-body group order
    left_err: float
    right_err: float
    left_ok: bool
    right_ok: bool
    left_iters: int = 0
    right_iters: int = 0


@dataclass
class IKSettings:
    max_err: float = 1e-3         # reject a per-arm target above this residual
    # 2026-08-25: was 200 (the "50 was too few" note below predates
    # -- from a different team's checkpoint, not
    # necessarily applicable to this one's kinematics). Measured live: a
    # solve grinding toward 200 iterations directly delays the next chunk
    # (this adapter is single-threaded through solve -> schedule ->
    # publish), producing exactly the "long interval, then jerk" pattern --
    # message cadence collapsed from ~11/s to ~0.5/s with solve time
    # climbing 5.1->36.9 ms/waypoint right before a safety violation.
    # Replayed the actual captured target rows from that window: accept
    # count and the accepted solutions themselves are BYTE-IDENTICAL at
    # max_iters=50 vs 200 (solves that converge do so well under 50 iters;
    # solves that don't converge plateau early and gain nothing from the
    # other 150). Set to 100 -- 2x margin above the value confirmed
    # sufficient on this policy's own data, not matching the historically-
    # flagged-insufficient 50 exactly. Cuts average solve time roughly in
    # half (34ms -> 18ms) with zero measured accuracy regression.
    max_iters: int = 100          # 50 was measured too few on a DIFFERENT team's checkpoint -- non-converged, jumpy solutions
    # 2026-09-03: was 1e-6 -- three orders of magnitude tighter than max_err
    # (1e-3), the threshold that actually decides accept/reject. Every solve
    # that already cleared max_err kept grinding toward near-machine-precision
    # anyway, since the loop's own break condition never checked max_err at
    # all. Measured against real captured (target, seed) pairs,
    # offline: solves that converge were pegged at max_iters=100 the entire
    # time under the old value; matching converged_err to max_err instead
    # dropped that to a median of 1 iteration and cut wall time per pair
    # 20.0ms -> 2.0ms (10.2x) -- with accept/reject decisions BYTE-IDENTICAL,
    # since that gate is applied after the loop exits regardless of why. Only
    # speeds up solves that were already going to be accepted; a target that
    # can't converge still burns the full max_iters either way. Global
    # default -- applies the same to every team, not tuned for one.
    converged_err: float = 1e-3
    # Extra offset (metres, in the wrist link's own frame) from the URDF
    # wrist_yaw_link to the point the (T,25) pose is taken to describe.
    # ZERO by default and deliberately so: inventing an offset here would
    # silently redefine the contract for every team at once. Only change
    # this if the organizer pins the convention explicitly -- and then it
    # is still one global number, never per team.
    tool_offset_xyz: dict = field(
        default_factory=lambda: {"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)}
    )
    # 2026-08-25: weight on staying close to the seed configuration
    # (pink.PostureTask), relative to position_cost=orientation_cost=1.0 on
    # the primary Cartesian task. Briefly raised to 1e-2 (10x) same day to
    # try to suppress a measured elbow null-space jump (6.9 / -9.285 rad/s,
    # past the WBC's hard +-6.0 limit) -- reverted back down after offline
    # testing showed it doesn't trade off predictably: it sometimes
    # improved convergence and sometimes caused otherwise-easy, near-static
    # targets to get stuck just above max_err with NO further progress even
    # given 2.5x the iteration budget (a real local-minimum, not an
    # iteration-count problem) -- i.e. it can turn "occasionally too fast"
    # into "chronically stuck, holds forever, does nothing". A small bump
    # (3x, not 10x) is kept as a mild bias with less observed downside; the
    # real protection against the elbow jump is the clamp-resync fix in
    # wbc_driver.py (bounds worst-case commanded jump to real measured
    # position every chunk) plus the hold-on-reject fix below (refuses to
    # act on a failed solve at all) -- both unambiguous wins, unlike this
    # knob. Revisit only with a wider, more representative offline
    # benchmark, not one difficult trajectory segment.
    #
    # 2026-08-25 SUPERSEDED: the uniform 3e-3 above was a blind guess at a
    # knob NVIDIA's own solver doesn't use uniformly. `ik.py`'s header has
    # said since day one that this module is a bench stand-in and must be
    # cross-checked against the real `decoupled_wbc` backend before being
    # trusted live -- that check was never actually done. It has now been
    # done: read against
    # ~/GR00T-WholeBodyControl/decoupled_wbc/control/teleop/solver/body/
    # body_ik_solver.py + body_ik_solver_settings.py (NVIDIA's own IK,
    # same pink+pinocchio stack, running against the same URDF). Four
    # concrete divergences found, all now matched below: (1) equal 1:1
    # position/orientation task weight vs NVIDIA's 4:1 (8.0/2.0), biased
    # toward nailing position; (2) lm_damping essentially off (1e-6) vs
    # NVIDIA's 3.0 -- LM damping is specifically the cushion that helps a
    # solve converge gracefully near an infeasible/boundary target instead
    # of grinding or landing on a wild solution, i.e. exactly our observed
    # near-limit accept decay; (3) a uniform posture cost across all 7 arm
    # joints vs NVIDIA's per-joint WEIGHTED posture task, which steers the
    # null space toward wrist_yaw/shoulder_yaw (weight 0.1, "spend the
    # redundant DOF here") and away from shoulder_pitch/elbow (weight 3-4,
    # "stay put") -- directly relevant, since the elbow is the joint that
    # jumped past the WBC's hard velocity limit earlier this session; (4)
    # raw URDF elbow limit (-1.0472..2.0944) vs NVIDIA's own solver-side
    # override capping the upper bound at 1.4 rad -- meaning our solver was
    # searching a region of elbow flexion NVIDIA's real controller doesn't
    # actually range into. `posture_cost` restored to NVIDIA's own base
    # value (0.01); the earlier 3e-3 mild-bias reasoning above no longer
    # applies once real per-joint weighting is doing the steering instead
    # of a uniform scalar.
    posture_cost: float = 0.01
    posture_lm_damping: float = 1.0          # NVIDIA body_ik_solver_settings.py
    hand_position_cost: float = 8.0          # NVIDIA: link_costs["hand"]["position_cost"]
    hand_orientation_cost: float = 2.0       # NVIDIA: link_costs["hand"]["orientation_cost"]
    hand_lm_damping: float = 3.0             # NVIDIA: link_costs["hand"]["lm_damping"]
    elbow_upper_limit_override: float = 1.4  # NVIDIA: ik_joint_limits["elbow_pitch"][1]
    # 2026-08-31: found live on a team's dry run -- a held
    # target reproduced offline (same target, same settled seed, same solver
    # code) converged to a DIFFERENT null-space configuration than what the
    # real robot actually settled at and held steady for 13s (max diff 1.1
    # rad across the 7 joints). The dominant divergence was wrist_roll: our
    # solver pins it exactly at the raw URDF limit (+-1.9722) while the real
    # robot held 0.872. Swept a symmetric wrist_roll cap against this exact
    # captured (target, seed) pair plus 15 genuinely-easy cases from the same
    # run's known-high-accept window: cap has ZERO measurable effect on the
    # easy cases (73.3% accept unchanged across every value tested, 1.9722
    # down to 0.8) while steadily pulling the hard case's solution toward
    # what the real robot did, bottoming out at cap=0.9 (diff 1.10 -> 0.073
    # rad), residual still far under max_err (0.000245 vs 0.001). Same
    # pattern as the elbow override above -- the solver's appetite for this
    # joint is effectively unbounded near this target, so it just uses
    # whatever range is available; capping it steers toward the same
    # solution the real WBC's own execution converges to instead.
    wrist_roll_limit_override: float = 0.9


class PinkArmIK:
    """One reduced pinocchio model + pink solver per arm. Bench/reference
    backend -- same construction validated at 7e-13 round-trip residual in
    the g1_bridge package."""

    def __init__(self, side: str, settings: IKSettings, urdf: Path = DEFAULT_URDF,
                 assets: Path = DEFAULT_URDF_DIR, warm_start: str = "current"):
        import pinocchio as pin
        import pink
        import qpsolvers

        self._pin, self._pink, self._qpsolvers = pin, pink, qpsolvers
        self.side = side
        self.settings = settings

        full = pin.RobotWrapper.BuildFromURDF(
            filename=str(urdf), package_dirs=[str(assets)], root_joint=None)
        self.model = full.model
        self.q_index = {}
        for name in self.model.names:
            if name == "universe":
                continue
            j = self.model.joints[self.model.getJointId(name)]
            if j.nq == 1:
                self.q_index[name] = j.idx_q

        arm = set(ARM_JOINTS[side])
        self._lock_ids = [self.model.getJointId(n) for n in self.model.names
                          if n != "universe" and n not in arm]
        self.wrist_frame = WRIST_FRAMES[side]
        ox, oy, oz = settings.tool_offset_xyz[side]
        self.offset = pin.SE3(np.eye(3), np.array([ox, oy, oz]))
        self.warm_start = warm_start
        self._last_q = None
        self.last_err = float("inf")

        # NVIDIA solver-side elbow limit override (see IKSettings.posture_cost
        # comment). Set once on the full model, before any buildReducedModel
        # call in solve() -- pinocchio's reduced model carries position
        # limits over from the source model for retained joints, so every
        # future per-call reduced model inherits this automatically.
        elbow_joint = f"{side}_elbow_joint"
        self.model.upperPositionLimit[self.q_index[elbow_joint]] = \
            settings.elbow_upper_limit_override

        wrist_roll_joint = f"{side}_wrist_roll_joint"
        wc = settings.wrist_roll_limit_override
        self.model.upperPositionLimit[self.q_index[wrist_roll_joint]] = wc
        self.model.lowerPositionLimit[self.q_index[wrist_roll_joint]] = -wc
        self.last_iters = 0
        self._cached_target_pos: np.ndarray | None = None
        self._cached_target_quat: np.ndarray | None = None
        self._cached_output: np.ndarray | None = None

    def solve(self, body_q29: np.ndarray, target_pos: np.ndarray,
              target_quat_wxyz: np.ndarray) -> np.ndarray:
        pin, pink, qpsolvers = self._pin, self._pink, self._qpsolvers

        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_quat_wxyz = np.asarray(target_quat_wxyz, dtype=np.float64)

        # 2026-08-24, found live on a team's dry run:
        # `warm_start="current"` reseeds every solve from the arm's
        # MEASURED pose, which is exactly what makes the metric
        # deterministic (see the comment below). But when a policy holds a
        # target fixed for many consecutive messages (any static
        # idle/arm_waypoints/gripper stage), re-solving an UNCHANGED target
        # from a fresh measured seed every time lets tiny measurement noise
        # in that seed get pushed a little further along the arm's null
        # space each call (7 joints for a 6-DOF Cartesian target -- one
        # redundant DOF, unconstrained by the primary task). Individually
        # negligible; compounding over ~80 solves in ~30s it walked two
        # right-arm joints 0.087 rad and 0.131 rad with the Cartesian
        # target never moving and IK reporting 100% accept the whole time
        # (accept/reject is Cartesian-only, so it's blind to this) -- well
        # under --max-joint-vel too, so nothing else catches it either.
        # Fix: if the target is the same one we solved last call, skip
        # solving again and reuse that exact joint solution -- no re-solve,
        # no fresh seed, no drift. Only short-circuits on a genuinely
        # unchanged target; a real target change always re-solves.
        if (self._cached_output is not None
                and self._cached_target_pos is not None
                and self._cached_target_quat is not None
                and np.allclose(target_pos, self._cached_target_pos, atol=1e-9)
                and np.allclose(target_quat_wxyz, self._cached_target_quat, atol=1e-9)):
            return self._cached_output.copy()

        q_full = pin.neutral(self.model)
        for name, idx in zip(_BODY_Q_NAMES, range(29)):
            if name in self.q_index:
                q_full[self.q_index[name]] = body_q29[idx]

        data = self.model.createData()
        pin.framesForwardKinematics(self.model, data, q_full)
        T_pelvis_world = data.oMf[self.model.getFrameId(PELVIS_FRAME)].copy()

        rot = Rotation.from_quat(np.roll(np.asarray(target_quat_wxyz), -1)).as_matrix()
        T_tool_pelvis = pin.SE3(rot, np.asarray(target_pos, dtype=np.float64))
        T_wrist_world = T_pelvis_world * (T_tool_pelvis * self.offset.inverse())

        reduced = pin.buildReducedModel(self.model, self._lock_ids, q_full)
        r_data = reduced.createData()
        r_index = {}
        for name in reduced.names:
            if name == "universe":
                continue
            j = reduced.joints[reduced.getJointId(name)]
            if j.nq == 1:
                r_index[name] = j.idx_q

        # Seed. "current" seeds from the arm's MEASURED configuration every
        # call, so a solve depends only on (target, body_q) -- deterministic
        # and reproducible. "last" seeds from the previous solution, which is
        # faster but makes the result depend on message ORDER and timing:
        # measured on real data that made the per-arm accept rate swing
        # 67%-100% across runs with identical input. That is evaluator-side
        # nondeterminism and it cannot be allowed to decide a team's score.
        if self.warm_start == "last" and self._last_q is not None:
            q_red = self._last_q.copy()
        else:
            q_red = pin.neutral(reduced)
            for name in ARM_JOINTS[self.side]:
                q_red[r_index[name]] = q_full[self.q_index[name]]

        cfg = pink.Configuration(reduced, r_data, q_red)
        task = pink.FrameTask(self.wrist_frame,
                              position_cost=self.settings.hand_position_cost,
                              orientation_cost=self.settings.hand_orientation_cost,
                              lm_damping=self.settings.hand_lm_damping)
        task.set_target(T_wrist_world)

        # weights indexed by r_index[name], not ARM_JOINTS order -- the
        # reduced model's internal q-vector order isn't guaranteed to match
        # ARM_JOINTS[side] (see the r_index-based result[] lookup below,
        # which exists for the same reason).
        weights = np.ones(reduced.nq)
        for name in ARM_JOINTS[self.side]:
            key = name.replace(f"{self.side}_", "", 1).replace("_joint", "")
            weights[r_index[name]] = _ARM_POSTURE_WEIGHTS[key]
        WeightedPostureTask = _get_weighted_posture_task_cls(pink)
        posture = WeightedPostureTask(cost=self.settings.posture_cost,
                                       weights=weights,
                                       lm_damping=self.settings.posture_lm_damping)
        posture.set_target(q_red.copy())

        solver = "quadprog" if "quadprog" in qpsolvers.available_solvers \
            else qpsolvers.available_solvers[0]
        n_iter, err = 0, None
        for n_iter in range(1, self.settings.max_iters + 1):
            v = pink.solve_ik(cfg, [task, posture], 0.02, solver=solver)
            cfg.integrate_inplace(v, 0.02)
            err = float(np.linalg.norm(task.compute_error(cfg)))
            if err < self.settings.converged_err:
                break

        self._last_q = cfg.q.copy()
        self.last_err = err
        self.last_iters = n_iter
        result = np.array([cfg.q[r_index[n]] for n in ARM_JOINTS[self.side]])

        self._cached_target_pos = target_pos
        self._cached_target_quat = target_quat_wxyz
        self._cached_output = result.copy()
        return result


_BODY_Q_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
assert len(_BODY_Q_NAMES) == 29


class UpperBodyIK:
    """Turns one (25,) action row into an upper-body joint vector.

    `include_waist` must match how `run_g1_control_loop.py` was launched
    (`--enable-waist` -> waist_location="lower_and_upper_body" -> the
    upper_body joint group includes the 3 waist joints, width 17; otherwise
    width 14). Getting this wrong misaligns every joint in the vector, so
    the driver reads it from the robot model where it can rather than
    trusting a default.
    """

    def __init__(self, settings: IKSettings | None = None, include_waist: bool = False,
                 urdf: Path = DEFAULT_URDF, assets: Path = DEFAULT_URDF_DIR,
                 warm_start: str = "current"):
        self.settings = settings or IKSettings()
        self.include_waist = include_waist
        self.left = PinkArmIK("left", self.settings, urdf, assets, warm_start)
        self.right = PinkArmIK("right", self.settings, urdf, assets, warm_start)
        self._last_good: np.ndarray | None = None

    @property
    def width(self) -> int:
        return 17 if self.include_waist else 14

    def solve_row(self, row25: np.ndarray, body_q29: np.ndarray) -> IKResult:
        row = np.asarray(row25, dtype=np.float64).reshape(-1)
        q_left = self.left.solve(body_q29, row[4:7], row[7:11])
        q_right = self.right.solve(body_q29, row[11:14], row[14:18])

        left_ok = self.left.last_err < self.settings.max_err
        right_ok = self.right.last_err < self.settings.max_err

        # Hold the last known-good target per arm rather than chasing an
        # unreachable one. An unreachable target is not a small error --
        # the solver's best effort for it can be a wildly different arm
        # configuration.
        #
        # 2026-08-25: the "no prior good solution yet" fallback used to be
        # np.concatenate([q_left, q_right, ...]) -- THIS call's own solve,
        # the exact thing left_ok/right_ok just said not to trust. If both
        # arms fail on EVERY call from message 1, self._last_good never
        # gets set (it's only assigned when left_ok or right_ok), so this
        # branch was hit on every single call, not just the first -- the
        # "hold" protection never actually engaged, every failed solve
        # passed straight through as if it had succeeded. Measured live:
        # L=0.0% R=0.0% for an entire 62-message run, arm moving
        # continuously backwards the whole time. Correct safe fallback
        # with no prior good solution is the robot's ACTUAL MEASURED
        # current pose (already available as body_q29), not a solve that
        # just failed -- holding real position is always safe; holding an
        # unvalidated one is not.
        prev = self._last_good
        if prev is None:
            prev = np.asarray(body_q29, dtype=np.float64)[15:29]
        out_l = q_left if left_ok else prev[0:7]
        out_r = q_right if right_ok else prev[7:14]

        if self.include_waist:
            waist = np.asarray(body_q29, dtype=np.float64)[12:15]   # hold measured waist
            upper = np.concatenate([out_l, out_r, waist])
        else:
            upper = np.concatenate([out_l, out_r])

        if left_ok or right_ok:
            self._last_good = upper.copy()

        return IKResult(upper_body=upper,
                        left_err=self.left.last_err, right_err=self.right.last_err,
                        left_ok=left_ok, right_ok=right_ok,
                        left_iters=self.left.last_iters, right_iters=self.right.last_iters)
