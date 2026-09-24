#!/usr/bin/env python3
"""The organizer-side WBC adapter: subscribe a team's :5556, drive the robot.

This is the piece the boundary contract always assumed existed and that no
bench run has ever actually had -- "the organizer's WBC dials in". With it,
a team's own Thor + Orin containers run COMPLETELY UNMODIFIED, exactly per
their own INSTRUCTIONS.md, and nothing team-specific lives here.

    team Thor container  --:8765-->  team Orin container
                                          | binds :5556  (boundary/actions.py)
                                          v
                                   THIS ADAPTER (subscriber)
                                          |
             decoupled: IK -> joint-space goal -> ROS2 ControlPolicy/upper_body_pose
             sonic:     relay protocol-v4 pose frames -> gear_sonic_deploy
                                          |
                                          v
                                    real G1 motors

Lane handling, both verified against NVlabs/GR00T-WholeBodyControl v1.1:

  decoupled  The WBC consumes JOINT SPACE (`target_upper_body_pose`); it
             runs no IK in its 50 Hz loop. So we IK here (see ik.py -- read
             its frame-convention warning). `target_time` accepts a list,
             so a whole (T,25) chunk goes over as an interpolated waypoint
             trajectory rather than just its first row.

  sonic      The organizer's `boundary/actions.py` pose frame is already
             byte-identical to the protocol-v4 "pose" message
             `gear_sonic_deploy` consumes -- same `pose` topic, same 1280B
             header, same token_state(1,64)/frame_index(1,)/
             left+right_hand_joints(1,7) fields. So this lane is a RELAY,
             not a translation. We decode only to validate and report.

  joint      Rides on the decoupled stack (same :5556 socket, same loop,
             same WBC), for policies trained on joint-space actions. A
             (T,22) chunk carries hands + 7+7 arm angles + base commands;
             the arm angles go into `target_upper_body_pose` directly --
             position-clamped to the robot model's limits, step-clamped
             exactly like the IK output, mapped by UpperBodyMapper, never
             re-solved. A `goto` request interpolates from the measured
             arms to a target pose at a bounded speed as one goal. Off
             with --joint-lane off. The WBC still owns rt/lowcmd; this is
             the ONLY joint channel into it, and the e-stop overrides all.

DRY-RUN BY DEFAULT. `--live` is required to actually publish anything
robot-ward.

SAFETY -- this adapter is not a safety system. It re-validates the contract
floor (`boundary_wire.validate_*`) and holds the last known-good arm target
when IK says a pose is unreachable, but well-formed is not the same as
sane. The independent e-stop is what stops the robot, and for the sonic
lane specifically the stop that actually works is `gear_sonic_deploy`'s own
(gamepad Select/O, or its command-topic stop flag) -- whoever owns
rt/lowcmd wins, and that is the deploy binary, not this process.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from boundary_wire import (  # noqa: E402
    GOTO_TOPIC, JOINT_TOPIC, POSE_TOPIC, TASKSPACE_TOPIC, BoundaryStateSubscriber,
    decode_goto, decode_joint, decode_pose, decode_taskspace, validate_goto,
    validate_joint, validate_pose, validate_taskspace,
)
import wbc_goal  # noqa: E402

LANES = ("decoupled", "sonic")


class Stats:
    def __init__(self):
        self.messages = 0
        self.rejected = 0
        self.stale = 0
        self.waypoints = 0
        self.left_ok = 0
        self.right_ok = 0
        self.published = 0
        self.gripper_sent = 0
        self.solve_ms = 0.0
        # joint lane
        self.joint_accepted = 0
        self.joint_rejected = 0
        self.goto_accepted = 0
        self.goto_rejected = 0
        self.joints_clamped = 0     # individual joint values pulled inside a limit
        self.t0 = time.monotonic()

    def report(self) -> str:
        el = max(time.monotonic() - self.t0, 1e-6)
        line = (f"[stats] {self.messages} msgs ({self.messages / el:.1f}/s), "
                f"{self.rejected} rejected, {self.stale} stale, "
                f"{self.published} published, {self.gripper_sent} gripper")
        if self.waypoints:
            # Per-waypoint, which is what IK is actually solved per. Anything
            # over 100% here means the denominator is wrong, not that the
            # policy is unusually good.
            line += (f" | IK accept over {self.waypoints} waypoints: "
                     f"L={100.0 * self.left_ok / self.waypoints:.1f}% "
                     f"R={100.0 * self.right_ok / self.waypoints:.1f}%"
                     f", {self.solve_ms / self.waypoints:.1f} ms/waypoint")
        if (self.joint_accepted or self.joint_rejected or self.goto_accepted
                or self.goto_rejected or self.joints_clamped):
            line += (f" | joint lane: {self.joint_accepted} chunks ok/"
                     f"{self.joint_rejected} rejected, goto {self.goto_accepted} ok/"
                     f"{self.goto_rejected} rejected, {self.joints_clamped} joint "
                     f"values clamped")
        return line


def make_action_subscriber(host: str, port: int, conflate: bool) -> zmq.Socket:
    """Dial into the team's bound :5556. We are the subscriber -- the team's
    client is the long-lived-side PUB, per boundary/actions.py.

    `conflate` is lane-dependent and matters:
      decoupled -- CONFLATE on. Each (T,25) chunk is a complete fresh plan
        from a fresh observation, so a newer chunk supersedes an older one
        outright. Without this, any moment where our IK is slower than the
        team's publish rate silently builds an unbounded backlog and every
        chunk we pull is progressively staler -- the robot ends up tracking
        the past. Same reasoning as boundary/cameras.py's own CONFLATE.
      sonic -- CONFLATE off. That lane is a sequential 50 Hz stream of
        individual latent rows; dropping rows is dropping motion.
    """
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")     # both topics; we dispatch on prefix
    sock.setsockopt(zmq.CONFLATE, 1 if conflate else 0)
    sock.setsockopt(zmq.LINGER, 0)
    # 2026-08-25: was 1000ms. The WBC has its OWN internal watchdog --
    # "Teleop mode timeout after 1.0s, injecting safe goal" -- that fires
    # if IT goes >1.0s without receiving a fresh goal FROM US, and that
    # injected transition does not go through our own IK/clamp pipeline at
    # all (it's the WBC's own code, not ours), so nothing in wbc_driver.py
    # or ik.py can bound it. Measured live: this fired immediately before
    # the worst violation of the session (right_elbow -13.234 rad/s) on a
    # run where --max-joint-vel had just been LOWERED (2.0, from 4.0) --
    # our own clamp getting more conservative had no effect, because the
    # violation wasn't coming through our clamp's path at all. A 1000ms
    # RCVTIMEO meant this loop itself could silently wait the entire 1.0s
    # doing nothing between chunks -- racing the WBC's own timeout and
    # sometimes losing. Shortened so the loop wakes up often enough to
    # publish a keepalive (see the zmq.Again handler below) well before
    # the WBC's deadline, regardless of how sparse the team's actual data
    # is.
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://{host}:{port}")
    return sock


def _q29(q):
    """Reduce a robot state vector to ik.py's 29-joint `_BODY_Q_NAMES` layout.

    `body_q` is NOT one shape. `--state-source boundary` (:5557) delivers the
    29-wide Unitree body convention -- legs 0-11, waist 12-14, left arm 15-21,
    right arm 22-28 -- which is exactly what ik.py expects. `--state-source
    wbc` (the DEFAULT, and what every live run has actually used) delivers the
    WBC's own model q, built from g1_29dof_with_hand.urdf: 43 wide, with the
    Dex3 hand joints interleaved -- legs 0-11, waist 12-14, left arm 15-21,
    LEFT HAND 22-28, right arm 29-35, right hand 36-42.

    Slicing that 43-wide vector with [:29] silently hands ik.py the left hand
    where the right arm should be, so the right arm's IK seed, its
    velocity-clamp anchor and its on-reject "hold measured pose" fallback all
    read constant hand values instead of the arm. The left arm is 15-21 in
    both layouts, so only the right arm is affected -- that is the left/right
    asymmetry seen live from 2026-09-04 onward.

    Confirmed against the loaded robot model (nq=43, right_arm=[29..35]),
    against this adapter's own startup banner (upper_body width=28, right_arm
    slots=[14..20] == full-q 29..35 minus 15), and on 1963 captured
    env_state_act records where q[22:29] is identically zero in every one
    while the true right arm q[29:36] moved 0.4558 rad. Fixed 2026-09-21.

    Shape-based so the boundary (29) and zeros (29) paths are untouched.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    return np.concatenate([q[0:22], q[29:36]]) if q.shape[0] == 43 else q[:29]


class ArmLimits:
    """Position limits for the 14 commanded arm joints, [left 7, right 7] in
    Unitree G1JointIndex order -- the order the joint lane's rows, ik.py's
    `_BODY_Q_NAMES[15:29]` and `UpperBodyMapper.build_waypoint` all share.

    Read from the robot model, never typed in. `--joint-lane-limits urdf`
    uses the limits exactly as the WBC's own RobotModel loads them: the
    URDF, plus the WBC's own supplemental narrowing (shoulder_roll is kept
    away from the torso, +-0.19 rad, in g1_supplemental_info.py). Those are
    the same arrays the WBC's JointSafetyMonitor enforces on the real robot,
    so a command that passes this clamp is one the WBC will not trip on.
    `ik` additionally applies ik.py's solver-side overrides (elbow upper
    bound, symmetric wrist_roll cap) so both lanes range over the same
    joint space -- off by default because those overrides exist to steer a
    redundant IK solution, not to protect hardware; see docs/CONTRACT.md.
    """

    MARGIN = 1e-3   # rad inside the limit, so the WBC never sees an exact edge

    def __init__(self, names, lower, upper, source: str):
        self.names = list(names)
        self.lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        self.upper = np.asarray(upper, dtype=np.float64).reshape(-1)
        self.source = source
        assert len(self.names) == self.lower.shape[0] == self.upper.shape[0] == 14

    def clamp(self, arms14):
        """-> (clamped copy, boolean mask of the entries that moved)."""
        arms = np.asarray(arms14, dtype=np.float64).reshape(-1)
        clamped = np.clip(arms, self.lower + self.MARGIN, self.upper - self.MARGIN)
        return clamped, clamped != arms

    def describe(self) -> str:
        span = ", ".join(f"{n.replace('_joint', '')}=[{lo:+.3f},{hi:+.3f}]"
                         for n, lo, hi in zip(self.names, self.lower, self.upper))
        return f"{self.source}: {span}"


def _load_arm_limits(mode: str, mapper, ik_settings) -> ArmLimits:
    """Arm joint limits for the joint lane's position clamp -- see ArmLimits.

    With a mapper the WBC's own RobotModel is the source (that is the model
    the goal is ultimately executed against). Without one (bench dry-run,
    no decoupled_wbc installed) the same URDF ik.py solves against is read
    with pinocchio directly, which carries no supplemental narrowing.
    """
    from ik import ARM_JOINTS, DEFAULT_URDF
    names = list(ARM_JOINTS["left"]) + list(ARM_JOINTS["right"])
    if mapper is not None:
        model = mapper.model
        idx = [model.joint_to_dof_index[n] for n in names]
        lower = np.asarray(model.lower_joint_limits, dtype=np.float64)[idx]
        upper = np.asarray(model.upper_joint_limits, dtype=np.float64)[idx]
        source = "WBC robot model"
    else:
        import pinocchio as pin
        model = pin.buildModelFromUrdf(str(DEFAULT_URDF))
        idx = [model.joints[model.getJointId(n)].idx_q for n in names]
        lower = np.asarray(model.lowerPositionLimit, dtype=np.float64)[idx]
        upper = np.asarray(model.upperPositionLimit, dtype=np.float64)[idx]
        source = f"URDF {DEFAULT_URDF.name}"
    if mode == "ik":
        for side_offset in (0, 7):
            # same two overrides ik.py applies to its solver model, taken
            # from the same settings object rather than retyped here
            upper[side_offset + 3] = ik_settings.elbow_upper_limit_override
            lower[side_offset + 4] = -ik_settings.wrist_roll_limit_override
            upper[side_offset + 4] = ik_settings.wrist_roll_limit_override
        source += " + ik.py solver overrides"
    return ArmLimits(names, lower, upper, source)


class _DecoupledContext:
    """State shared by the per-message handlers of `run_decoupled`.

    These used to be locals of one monolithic loop. They live here so the
    handlers (`_handle_taskspace`, `_handle_joint`, `_handle_goto`) can be
    driven one message at a time from a test with a fake backend; the
    control flow inside each handler is otherwise the loop's, unchanged.
    """

    # Margin under the WBC's ~1.0s watchdog -- fire before it does, not
    # after. Checked between individual waypoint solves (see
    # _handle_taskspace), not just between chunks, since the solve loop
    # itself is where a slow cycle actually accumulates.
    KEEPALIVE_DEADLINE_S = 0.7

    def __init__(self, args, stats: Stats, backend, state_sub, solver, mapper,
                 dex1_pub, arm_limits: ArmLimits | None):
        self.args = args
        self.stats = stats
        self.backend = backend
        self.state_sub = state_sub
        self.solver = solver
        self.mapper = mapper
        self.dex1_pub = dex1_pub
        self.arm_limits = arm_limits
        # Rate-limits the commanded arm joints against the last waypoint this
        # adapter actually scheduled. IK has no notion of how far away in TIME
        # its target is -- chunk_hz alone schedules the first waypoint of every
        # chunk only 1/chunk_hz out, so a solved pose far from the current one
        # implies whatever velocity that delta needs, with nothing capping it.
        # Measured tripping the WBC's own real-hardware-only joint safety
        # monitor (joint_safety.py, +-6 rad/s): first at ~7.3-7.4 rad/s on the
        # first live goal (fixed by clamping), then again later at -6.570 rad/s
        # on right_elbow_joint on a run where the clamp was active the whole
        # time -- see _step_clamp's i==0 branch for why (the clamp's reference
        # point can silently drift from the robot's real position if it's only
        # ever updated from what THIS process last commanded). Reset every
        # chunk to ground-truth body_q now, not just once at startup.
        self.last_commanded_arms = None
        # Last successfully-published waypoint + its accompanying fields, kept
        # around purely so a keepalive (see _publish_keepalive) can hold this
        # exact position rather than reconstructing one from parts.
        self.last_goal_template: dict | None = None
        self.keepalives_sent = 0
        # 2026-09-03: wall-clock time of the last thing actually published to
        # the WBC, by ANY path (real chunk or keepalive). See the mid-solve
        # watchdog check in _handle_taskspace for why this exists separately
        # from the zmq.Again-triggered keepalive -- that one only fires when
        # the loop is BLOCKED waiting for a message; it does nothing if the
        # loop is instead BUSY solving one. Traced a real live violation
        # (right_elbow_joint -8.992 rad/s) to exactly that gap: the team's
        # client published on a steady ~300ms cadence the whole time (confirmed
        # independently via capture_evidence.py's own separate subscription,
        # zero gaps >330ms anywhere near the violation) and the adapter's own
        # [stats] line showed 0 keepalives for the entire run right up to the
        # trip -- so `zmq.Again` never fired even though the WBC's own >1.0s
        # "no fresh goal" watchdog still did, and injected the unclamped
        # transition that caused it (confirmed: a "Teleop mode timeout" line
        # sits directly before that violation in the WBC log). CONFLATE=1 on
        # this socket means as long as a message is EVER waiting, `sub.recv()`
        # returns immediately rather than timing out -- so a slow processing
        # cycle (not a slow client) can silently eat the whole 1.0s budget with
        # this loop technically busy the entire time, never once blocked long
        # enough to hit the existing keepalive path.
        self.last_publish_time = time.monotonic()
        self.last_report = time.monotonic()
        # A goto trajectory still ahead of the WBC's clock, if any. A goto is
        # one goal spanning seconds, while the keepalive fires after 200ms of
        # client silence; the WBC's InterpolationPolicy.schedule_waypoint
        # TRIMS everything after the garbage-collection time when a waypoint
        # arrives earlier than its last scheduled one, so a single-row hold
        # at t+1/chunk_hz would cut the goto short and demand its final pose
        # almost at once. While this is set the keepalive re-sends the
        # remaining future waypoints instead (see _publish_keepalive).
        self.goto_in_flight: dict | None = None
        # Joints already reported as clamped -- the log line fires once per
        # joint; stats.joints_clamped carries the running count.
        self.clamp_logged: set[int] = set()


def _publish_keepalive(ctx: _DecoupledContext) -> bool:
    """Hold the last commanded position with a refreshed target_time.

    2026-08-25: publish a keepalive rather than doing nothing. See the
    RCVTIMEO comment in make_action_subscriber: the WBC has its own ~1.0s
    "no fresh goal" watchdog that injects an UNCLAMPED transition of its own
    if we go quiet too long. A short RCVTIMEO alone only helps if we
    actually USE the extra wakeups to publish something -- this is that.

    Returns False when there is nothing to hold yet (dry-run, no backend,
    or nothing published so far), in which case the caller says so.
    """
    args, backend = ctx.args, ctx.backend
    if not (args.live and backend is not None and ctx.last_goal_template is not None):
        return False
    g = ctx.goto_in_flight
    if g is not None:
        now = time.monotonic()
        ahead = [i for i, t in enumerate(g["times"]) if t > now]
        if ahead:
            keep_goal = wbc_goal.build_goal(
                upper_body_waypoints=np.asarray([g["upper_body"][i] for i in ahead]),
                target_time=[g["times"][i] for i in ahead],
                base_height_command=[g["base_height"]] * len(ahead),
                navigate_cmd=[g["navigate_cmd"]] * len(ahead),
            )
            backend.publish_goal(keep_goal)
            ctx.keepalives_sent += 1
            ctx.last_publish_time = time.monotonic()
            return True
        ctx.goto_in_flight = None   # arrived; fall through to the plain hold
    tpl = ctx.last_goal_template
    keep_goal = wbc_goal.build_goal(
        upper_body_waypoints=tpl["upper_body"],
        target_time=[time.monotonic() + 1.0 / args.chunk_hz],
        base_height_command=tpl["base_height"],
        navigate_cmd=tpl["navigate_cmd"],
        wrist_pose=tpl["wrist_pose"],
    )
    backend.publish_goal(keep_goal)
    ctx.keepalives_sent += 1
    ctx.last_publish_time = time.monotonic()
    return True


def _read_body_q(ctx: _DecoupledContext):
    """The robot state this message is acted on, or None (already reported)."""
    args = ctx.args
    body_q = None
    if ctx.state_sub is not None:
        body_q = ctx.state_sub.get_body_q()
    elif ctx.backend is not None:
        body_q = ctx.backend.get_robot_q()
    if body_q is None:
        if args.live or args.state_source != "zeros":
            # No state means no valid IK seed and no waist hold. Refusing
            # is the safe branch: publishing a goal solved against a
            # guessed configuration is worse than publishing nothing, and
            # measuring against one is worse than not measuring.
            print("[adapter] no robot state yet -- skipping chunk", file=sys.stderr)
            return None
        body_q = np.zeros(29)
    return body_q


def _fresh_body_q(ctx: _DecoupledContext, body_q):
    """Re-read state as close to publish time as this process can get."""
    fresh_body_q = body_q
    if ctx.state_sub is not None:
        fresh_q = ctx.state_sub.get_body_q()
        if fresh_q is not None:
            fresh_body_q = fresh_q
    elif ctx.backend is not None:
        fresh_q = ctx.backend.get_robot_q()
        if fresh_q is not None:
            fresh_body_q = fresh_q
    return fresh_body_q


def _chunk_age(ctx: _DecoupledContext, issued_at: float, what: str = "chunk"):
    """Seconds since the sender issued this message, or None if it is too
    old to act on (counted as stale).

    DO NOT re-apply the sender's latency compensation. The reference
    client already drops the rows its own inference latency consumed
    AND backdates issued_at to when that inference started
    (components/client.py: send_chunk(chunk[skip:], issued_at=now-L)).
    Skipping again on that backdated stamp double-counts the same
    latency -- measured here against a real team container it ate 11
    of 16 rows on top of the client's own 4, leaving one usable row.
    issued_at is still the right thing to judge STALENESS with; it is
    just not an amount to re-skip by.
    """
    args = ctx.args
    age = max(time.time() - issued_at, 0.0) if issued_at else 0.0
    if args.max_chunk_age_s and age > args.max_chunk_age_s:
        ctx.stats.stale += 1
        if args.verbose:
            print(f"[adapter] dropping {what} {age * 1000:.0f}ms old "
                  f"(> {args.max_chunk_age_s * 1000:.0f}ms)", file=sys.stderr)
        return None
    return age


def _step_clamp(ctx: _DecoupledContext, arms, i: int, fresh_body_q, max_step):
    """Bound the per-waypoint arm step to max_step, anchored on the FRESH
    measured arms for the first waypoint of every chunk.

    2026-08-25: clamp against a FRESH state read, taken AFTER solving,
    not the body_q read before the solve loop started. The solve loop
    can legitimately take tens of ms (both arms, up to max_iters each,
    across up to --max-waypoints rows) -- lowering max_iters (200->100,
    same day) shrank this but could not zero it, because it was never
    really an iteration-count problem: ANY nonzero solve time means the
    real robot keeps moving (tracking whatever the PREVIOUS goal was)
    while our clamp anchor sits frozen at a pose that's now stale by
    exactly that amount. The clamp then bounds our own commanded sequence
    to small steps relative to that stale anchor -- but says nothing
    about the jump from wherever the robot ACTUALLY is (by publish time)
    to wherever our first "clamped" waypoint claims to start from. That
    jump is completely unbounded, and grows with solve time -- exactly
    why violations correlated with slow solves (5ms/waypoint -> 35-40ms
    right before each of the last two) without max_iters alone fixing
    it. Re-fetching state, as close to publish time as this process can
    get, and anchoring the clamp to THAT instead closes the actual gap
    rather than the solve-time symptom of it. Lane/policy-agnostic --
    this touches only the clamp, not IK itself, so it applies identically
    to every lane through this same adapter, not just one team.

    `arms` is always [left arm(7), right arm(7)], in the same order as
    _q29(body_q)[15:29] -- see ik.py's _BODY_Q_NAMES.
    """
    if max_step is not None:
        if i == 0:
            ctx.last_commanded_arms = np.asarray(_q29(fresh_body_q)[15:29], dtype=np.float64)
        arms = ctx.last_commanded_arms + np.clip(
            arms - ctx.last_commanded_arms, -max_step, max_step)
        ctx.last_commanded_arms = arms
    return arms


def _position_clamp(ctx: _DecoupledContext, arms):
    """Joint lane only: pull each arm joint inside the robot model's limits.

    Logs once per joint the first time it happens; every clamped value is
    counted in stats.joints_clamped. The taskspace lane never comes here
    -- its IK is bounded by the solver's own limits.
    """
    lim = ctx.arm_limits
    if lim is None:
        return np.asarray(arms, dtype=np.float64)
    clamped, hit = lim.clamp(arms)
    if hit.any():
        ctx.stats.joints_clamped += int(hit.sum())
        for j in np.flatnonzero(hit):
            j = int(j)
            if j in ctx.clamp_logged:
                continue
            ctx.clamp_logged.add(j)
            print(f"[adapter] joint lane: {lim.names[j]} commanded {arms[j]:+.4f} rad, "
                  f"outside [{lim.lower[j]:+.4f}, {lim.upper[j]:+.4f}] -- clamped to "
                  f"{clamped[j]:+.4f}. Reported once per joint; the [stats] line "
                  f"counts every occurrence.", file=sys.stderr)
    return clamped


def _arm_waypoint(ctx: _DecoupledContext, fresh_body_q, arms, fallback):
    """One WBC upper-body waypoint from 14 arm angles: the mapper replaces
    only the arm slots (hands/waist pass through as measured); in bench
    mode (no robot model) `fallback` supplies the rest of the vector."""
    if ctx.mapper is not None:
        return ctx.mapper.build_waypoint(fresh_body_q, arms[0:7], arms[7:14])
    out = np.asarray(fallback, dtype=np.float64).copy()
    out[:14] = arms
    return out


def _bench_upper_body(ctx: _DecoupledContext, fresh_body_q, arms):
    """Bench-mode stand-in for the mapper on the joint lane: the solver's
    width, arms first, measured waist appended iff --enable-waist -- the
    same layout ik.py emits."""
    if ctx.args.enable_waist:
        return np.concatenate([arms, np.asarray(_q29(fresh_body_q)[12:15], dtype=np.float64)])
    return np.asarray(arms, dtype=np.float64)


def _publish_trajectory(ctx: _DecoupledContext, waypoints, base_heights, nav_cmds,
                        wrist_pose, hands):
    """Schedule `waypoints` from now at chunk_hz, publish, update the
    keepalive template, relay the gripper. Shared by all three handlers."""
    args, stats = ctx.args, ctx.stats
    # Schedule from now, after our own solve cost -- the only latency
    # this adapter is entitled to compensate for is the one it adds.
    t_base = time.monotonic()
    times = [t_base + (i + 1) / args.chunk_hz for i in range(len(waypoints))]

    goal = wbc_goal.build_goal(
        upper_body_waypoints=np.asarray(waypoints),
        target_time=times,
        # per-waypoint, straight off each row -- not just row 0
        base_height_command=base_heights,
        navigate_cmd=nav_cmds,
        wrist_pose=wrist_pose,
    )

    if ctx.backend is not None:
        ctx.backend.publish_goal(goal)
        stats.published += 1
        ctx.last_publish_time = time.monotonic()
        # Hold-position template for a keepalive if the next real
        # chunk is slow to arrive -- last waypoint reached, single-row.
        ctx.last_goal_template = {
            "upper_body": waypoints[-1],
            "base_height": base_heights[-1],
            "navigate_cmd": nav_cmds[-1],
            "wrist_pose": wrist_pose,
        }
        # Whatever was in flight before is superseded by this goal.
        ctx.goto_in_flight = None

    # boundary/actions.py: cols [0:2] left hand, [2:4] right hand,
    # -1 open .. +1 closed, both columns of a pair duplicated.
    if ctx.dex1_pub is not None and hands is not None:
        ctx.dex1_pub.send(b"dex1" + msgpack.packb(
            {"left": float(hands[0]), "right": float(hands[1])},
            use_bin_type=True))
        stats.gripper_sent += 1
    return goal, times


def _maybe_report(ctx: _DecoupledContext):
    if time.monotonic() - ctx.last_report > 2.0:
        print(ctx.stats.report() + f" | {ctx.keepalives_sent} keepalives")
        ctx.last_report = time.monotonic()


def _handle_taskspace(ctx: _DecoupledContext, msg: bytes):
    """One `taskspace` message: validate, IK per row, clamp, publish."""
    args, stats = ctx.args, ctx.stats
    try:
        chunk = decode_taskspace(msg)
    except Exception as exc:
        stats.rejected += 1
        print(f"[adapter] undecodable frame: {exc}", file=sys.stderr)
        return

    stats.messages += 1
    problems = validate_taskspace(chunk)
    if problems:
        stats.rejected += 1
        print(f"[adapter] REJECTED chunk: {'; '.join(problems)}", file=sys.stderr)
        return

    body_q = _read_body_q(ctx)
    if body_q is None:
        return

    age = _chunk_age(ctx, chunk.issued_at)
    if age is None:
        return

    rows = chunk.actions
    waypoints = []
    t_solve = time.monotonic()
    dt = 1.0 / args.chunk_hz
    max_step = args.max_joint_vel * dt if args.max_joint_vel else None
    res0 = None
    raw_results = []
    solve_failed = False
    for i, row in enumerate(rows[:args.max_waypoints]):
        try:
            res = ctx.solver.solve_row(row, _q29(body_q))
        except Exception as exc:
            # Fail closed. The WBC keeps its last published goal; never
            # terminate the adapter or publish a partially solved chunk.
            #
            # Reachable since the 43->29 indexing fix: before it, the
            # right arm's seed was a constant zero vector, inside every
            # limit, so pink's check_limits could not fire on that side.
            # The seed is now the real measured pose, which can sit
            # microscopically outside the limits ik.py narrows below the
            # URDF -- 2026-09-21 aborted the process on 0.904509 against
            # a +/-0.9 wrist_roll override, 0.26 degrees over.
            stats.rejected += 1
            solve_failed = True
            print(
                f"[adapter] IK solve failed; holding last safe goal: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            break
        if i == 0:
            res0 = res
        stats.waypoints += 1
        stats.left_ok += int(res.left_ok)
        stats.right_ok += int(res.right_ok)
        raw_results.append(res)
        # Mid-solve watchdog (2026-09-03) -- see KEEPALIVE_DEADLINE_S.
        # Checked after every waypoint, not just between chunks, because a
        # slow chunk is exactly the case the zmq.Again-only keepalive
        # can't see.
        if (args.live and ctx.backend is not None and ctx.last_goal_template is not None
                and time.monotonic() - ctx.last_publish_time > ctx.KEEPALIVE_DEADLINE_S):
            _publish_keepalive(ctx)
    solve_s = time.monotonic() - t_solve
    stats.solve_ms += solve_s * 1000.0
    if solve_failed:
        return

    fresh_body_q = _fresh_body_q(ctx, body_q)

    for i, res in enumerate(raw_results):
        # res.upper_body[:14] is always [left arm(7), right arm(7)], in
        # the same order as _q29(body_q)[15:29] -- see ik.py's _BODY_Q_NAMES.
        # Waist/hand slots are already held at measured values (ik.py's
        # own waist passthrough; mapper.build_waypoint's for hands), so
        # only the arm portion can ever jump and needs clamping here.
        arms = _step_clamp(ctx, res.upper_body[:14].copy(), i, fresh_body_q, max_step)
        # arm slots replaced; hands/waist pass through as measured
        waypoints.append(_arm_waypoint(ctx, fresh_body_q, arms, res.upper_body))

    if not waypoints:
        return

    n = len(waypoints)
    wrist_pose = np.concatenate([rows[0][4:7], rows[0][7:11],
                                 rows[0][11:14], rows[0][14:18]])
    goal, _ = _publish_trajectory(
        ctx, waypoints,
        [[float(r[21])] for r in rows[:n]],
        [np.asarray(r[18:21], dtype=np.float64) for r in rows[:n]],
        wrist_pose, (rows[0][0], rows[0][2]))

    if args.verbose:
        # Diagnostic for the frame/offset question (ik.py's own header):
        # the (T,25) contract never states which point on the hand the
        # position means or in what frame -- these are the RAW targets
        # the policy is actually outputting, right off the wire,
        # so their physical plausibility (reach length, not embedded in
        # the robot, tracking the scene over time) can be eyeballed
        # directly rather than only inferred from IK's own residual.
        print(f"[adapter] would publish {len(waypoints)} waypoints in "
              f"{solve_s * 1000:.0f}ms (chunk age {age * 1000:.0f}ms), "
              f"base_h={goal['base_height_command'][0][0]:.3f}, "
              f"upper_body[0]={np.round(waypoints[0], 3)}")
        if res0 is not None:
            print(f"[adapter] row0 RAW targets (pelvis frame, ik.py's "
                  f"assumption) -- "
                  f"left_pos={np.round(rows[0][4:7], 3)} "
                  f"left_quat={np.round(rows[0][7:11], 3)} "
                  f"left_err={res0.left_err:.4f} left_ok={res0.left_ok} | "
                  f"right_pos={np.round(rows[0][11:14], 3)} "
                  f"right_quat={np.round(rows[0][14:18], 3)} "
                  f"right_err={res0.right_err:.4f} right_ok={res0.right_ok} | "
                  f"left_hand={rows[0][0]:.3f} right_hand={rows[0][2]:.3f} "
                  f"(-1=open, +1=closed)")

    _maybe_report(ctx)


def _handle_joint(ctx: _DecoupledContext, msg: bytes):
    """One `joint` message: (T,22) rows of hands + arm angles + base cmds.

    Same gates as the taskspace path (decode, validate, robot state,
    staleness), then position clamp -> the SAME step clamp -> mapper ->
    build_goal -> publish -> gripper relay. No IK is involved: the WBC's
    native input is joint space, so the row's arm angles go into the goal
    as they are, bounded but never re-solved.
    """
    args, stats = ctx.args, ctx.stats
    try:
        chunk = decode_joint(msg)
    except Exception as exc:
        stats.rejected += 1
        stats.joint_rejected += 1
        print(f"[adapter] undecodable joint frame: {exc}", file=sys.stderr)
        return

    stats.messages += 1
    problems = validate_joint(chunk)
    if problems:
        stats.rejected += 1
        stats.joint_rejected += 1
        print(f"[adapter] REJECTED joint chunk: {'; '.join(problems)}", file=sys.stderr)
        return

    body_q = _read_body_q(ctx)
    if body_q is None:
        return

    age = _chunk_age(ctx, chunk.issued_at, "joint chunk")
    if age is None:
        return

    rows = chunk.actions
    dt = 1.0 / args.chunk_hz
    max_step = args.max_joint_vel * dt if args.max_joint_vel else None
    # No solve stands between reading the state and clamping against it,
    # so the read above IS the fresh anchor the taskspace path re-fetches.
    waypoints = []
    for i, row in enumerate(rows[:args.max_waypoints]):
        arms = np.concatenate([row[4:11], row[11:18]]).astype(np.float64)
        arms = _position_clamp(ctx, arms)
        arms = _step_clamp(ctx, arms, i, body_q, max_step)
        waypoints.append(_arm_waypoint(ctx, body_q, arms,
                                       _bench_upper_body(ctx, body_q, arms)))

    n = len(waypoints)
    stats.joint_accepted += 1
    goal, _ = _publish_trajectory(
        ctx, waypoints,
        [[float(r[21])] for r in rows[:n]],
        [np.asarray(r[18:21], dtype=np.float64) for r in rows[:n]],
        None, (rows[0][0], rows[0][2]))

    if args.verbose:
        print(f"[adapter] joint chunk: {n} waypoints (chunk age {age * 1000:.0f}ms), "
              f"base_h={goal['base_height_command'][0][0]:.3f}, "
              f"upper_body[0]={np.round(waypoints[0], 3)} | "
              f"left_hand={rows[0][0]:.3f} right_hand={rows[0][2]:.3f} "
              f"(-1=open, +1=closed)")

    _maybe_report(ctx)


def _handle_goto(ctx: _DecoupledContext, msg: bytes):
    """One `goto` message: straight-line joint interpolation from the
    measured arms to the requested ones, published as ONE multi-waypoint
    goal through the same clamp -> mapper -> build_goal -> publish path.

    Speed is min(request, --goto-max-speed, --max-joint-vel): the last term
    makes the step clamp a no-op by construction, so the final waypoint IS
    the target rather than wherever a clipped ramp happened to end. Does
    not block -- the client watches body_q on :5557 to decide arrival.
    """
    args, stats = ctx.args, ctx.stats
    try:
        req = decode_goto(msg)
    except Exception as exc:
        stats.rejected += 1
        stats.goto_rejected += 1
        print(f"[adapter] undecodable goto frame: {exc}", file=sys.stderr)
        return

    stats.messages += 1
    problems = validate_goto(req)
    if problems:
        stats.rejected += 1
        stats.goto_rejected += 1
        print(f"[adapter] REJECTED goto: {'; '.join(problems)}", file=sys.stderr)
        return

    body_q = _read_body_q(ctx)
    if body_q is None:
        return

    age = _chunk_age(ctx, req.issued_at, "goto")
    if age is None:
        return

    target = _position_clamp(ctx, np.concatenate([req.left_arm, req.right_arm]))
    measured = np.asarray(_q29(body_q)[15:29], dtype=np.float64)
    speed = min(req.max_speed, args.goto_max_speed)
    if args.max_joint_vel:
        speed = min(speed, args.max_joint_vel)
    dt = 1.0 / args.chunk_hz
    max_step = args.max_joint_vel * dt if args.max_joint_vel else None
    delta = float(np.max(np.abs(target - measured)))
    steps = max(1, int(math.ceil(delta / (speed * dt) - 1e-9)))
    path = np.linspace(measured, target, steps + 1)[1:]

    waypoints = []
    for i, arms in enumerate(path):
        arms = _step_clamp(ctx, arms, i, body_q, max_step)
        waypoints.append(_arm_waypoint(ctx, body_q, arms,
                                       _bench_upper_body(ctx, body_q, arms)))

    # A goto moves the arms and nothing else: the base keeps whatever was
    # last commanded (or the WBC's own default when nothing was).
    tpl = ctx.last_goal_template
    base_height = [wbc_goal.DEFAULT_BASE_HEIGHT] if tpl is None else list(tpl["base_height"])
    navigate = (np.asarray(wbc_goal.DEFAULT_NAV_CMD, dtype=np.float64) if tpl is None
                else np.asarray(tpl["navigate_cmd"], dtype=np.float64))
    n = len(waypoints)
    stats.goto_accepted += 1
    hands = None if req.hands is None else (req.hands[0], req.hands[1])
    _, times = _publish_trajectory(ctx, waypoints, [base_height] * n, [navigate] * n,
                                   None, hands)
    if ctx.backend is not None:
        ctx.goto_in_flight = {
            "upper_body": waypoints, "times": times,
            "base_height": base_height, "navigate_cmd": navigate,
        }
    print(f"[adapter] goto: {n} waypoints over {n * dt:.2f}s at {speed:.2f} rad/s "
          f"(max |delta| {delta:.3f} rad, request {req.max_speed:.2f} rad/s, "
          f"age {age * 1000:.0f}ms)")

    _maybe_report(ctx)


def run_decoupled(args, sub: zmq.Socket, stats: Stats):
    from ik import IKSettings, UpperBodyIK

    backend = None
    if args.live:
        backend = wbc_goal.make_backend(args.wbc_backend)
        print(f"[adapter] WBC backend: {backend.health()}")
        if args.engage_policy:
            # Wait for the WBC's own state topic to actually be live before
            # sending the toggle -- sending it before rclpy has received a
            # single state message risks it landing before the subscriber
            # side of the control loop is fully up. Bounded wait; if state
            # never arrives this is the same "no robot state" situation the
            # rest of the loop already refuses to act on, so fail loud here
            # too rather than silently skipping the toggle.
            print("[adapter] --engage-policy: waiting for WBC state before "
                  "sending toggle_policy_action...")
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and backend.health()["state_stale"]:
                time.sleep(0.05)
            if backend.health()["state_stale"]:
                print("[adapter] --engage-policy: WBC state never came up in 5s "
                      "-- NOT sending the toggle. use_policy_action stays False "
                      "(safe default: hold current position).", file=sys.stderr)
            else:
                backend.toggle_policy_action()
                print("[adapter] --engage-policy: sent toggle_policy_action=True "
                      "once. Confirm engagement independently -- decode "
                      "/ControlPolicy/lower_body_policy_status, don't just trust "
                      "this log line.")
    else:
        print("[adapter] DRY-RUN -- decoding, validating and solving IK, "
              "publishing nothing to the WBC.")

    # Where real body_q comes from. The WBC's own state topic is
    # authoritative when it is running; :5557 lets a DRY RUN measure against
    # the real arm configuration with no WBC and no ROS 2 at all, which is
    # the cheapest way to get a meaningful reachability number.
    state_sub = None
    if args.state_source == "boundary":
        state_sub = BoundaryStateSubscriber(args.orin_host, args.state_port)
        print(f"[adapter] body_q from the organizer's state endpoint "
              f"{state_sub.endpoint}")
    elif args.state_source == "wbc":
        print("[adapter] body_q from the WBC's own state topic")
    else:
        print("[adapter] WARNING: --state-source zeros -- IK is solved against a "
              "ZERO joint vector, not the real arm. Shapes and plumbing only; "
              "any reachability number from this run is meaningless.",
              file=sys.stderr)

    solver = UpperBodyIK(IKSettings(max_err=args.max_ik_err),
                         include_waist=args.enable_waist,
                         warm_start=args.ik_warm_start)

    # Never guess the upper-body width. Query the WBC's own robot model and
    # overwrite only the arm slots -- see upper_body_map.py. A mismatch here
    # CRASHES the control loop, which on the real robot is the balance
    # controller.
    mapper = None
    # Only reach for decoupled_wbc's robot model when we are actually about
    # to publish to a real ros2 control loop -- NOT just because that is the
    # default --wbc-backend value. A pure dry-run (no --live) needs none of
    # this, and must keep working in g1_control_venv, which has no
    # decoupled_wbc installed (that only exists in the separate g1_wbc conda
    # env). Getting this wrong broke Phase A: `python3 wbc_driver.py --lane
    # decoupled --state-source boundary --verbose` (no --live) started
    # importing decoupled_wbc anyway and crashed with ModuleNotFoundError.
    if (args.live and args.wbc_backend == "ros2") or args.upper_body_from_model:
        from upper_body_map import UpperBodyMapper
        waist_loc = "lower_and_upper_body" if args.enable_waist else "lower_body"
        mapper = UpperBodyMapper(waist_location=waist_loc)
        print(f"[adapter] {mapper.describe()}")
    else:
        print(f"[adapter] bench mode: emitting {solver.width}-wide vectors "
              f"(no robot model available to map against)")

    # Gripper. The WBC cannot drive Dex1-1 (it is Dex3-only), so the hand
    # columns of (T,25) would otherwise be silently discarded and no grasp
    # could occur. run_wbc_with_dex1.py injects these into the same
    # rt/lowcmd message the body already goes out on -- a second publisher
    # is not possible, see that file's header.
    dex1_pub = None
    if args.dex1_port:
        _ctx = zmq.Context.instance()
        dex1_pub = _ctx.socket(zmq.PUB)
        dex1_pub.setsockopt(zmq.LINGER, 0)
        dex1_pub.connect(f"tcp://{args.dex1_host}:{args.dex1_port}")
        print(f"[adapter] gripper -> tcp://{args.dex1_host}:{args.dex1_port} "
              f"(requires the WBC be launched via run_wbc_with_dex1.py)")
    else:
        print("[adapter] --dex1-port 0: gripper commands DISCARDED, no grasp possible")

    # Joint lane: same socket, same loop, joint angles instead of poses.
    # Its position clamp reads the robot model's limits; it never types
    # them in. Off means the two topics are ignored exactly like any other
    # unknown prefix.
    joint_lane = args.joint_lane == "on"
    arm_limits = None
    if joint_lane:
        try:
            arm_limits = _load_arm_limits(args.joint_lane_limits, mapper, solver.settings)
        except Exception as exc:
            if mapper is not None:
                raise
            # Bench mode with no robot model to read from: nothing reaches
            # a robot here, so run unclamped rather than refuse to start.
            print(f"[adapter] joint lane: no position limits available in bench "
                  f"mode ({type(exc).__name__}: {exc}); position clamp OFF",
                  file=sys.stderr)
        print(f"[adapter] joint lane: ON -- topics {JOINT_TOPIC!r}/{GOTO_TOPIC!r}, "
              f"limits={args.joint_lane_limits}, goto max speed "
              f"{args.goto_max_speed:.2f} rad/s, position clamp "
              f"{'ON' if arm_limits is not None else 'OFF'}")
        if arm_limits is not None:
            print(f"[adapter] joint lane limits ({arm_limits.describe()})")
    else:
        print(f"[adapter] joint lane: OFF -- {JOINT_TOPIC!r}/{GOTO_TOPIC!r} ignored")

    ctx = _DecoupledContext(args, stats, backend, state_sub, solver, mapper,
                            dex1_pub, arm_limits)
    while True:
        try:
            msg = sub.recv()
        except zmq.Again:
            if not _publish_keepalive(ctx):
                print("[adapter] no actions on :5556 yet (is the team's Orin client up?)")
            continue
        if msg.startswith(TASKSPACE_TOPIC):
            _handle_taskspace(ctx, msg)
        elif joint_lane and msg.startswith(JOINT_TOPIC):
            _handle_joint(ctx, msg)
        elif joint_lane and msg.startswith(GOTO_TOPIC):
            _handle_goto(ctx, msg)
        # anything else: not ours, ignored


_COMMAND_HEADER_SIZE = 1280  # must match ZMQPackedMessageSubscriber::HEADER_SIZE


def _pack_command(*, start: bool, stop: bool, planner: bool) -> bytes:
    """Build a gear_sonic_deploy 'command'-topic message.

    Byte-for-byte the same wire format as the deploy's own reference
    publisher (tests/test_zmq_manager.py ZMQPublisher.send_command) --
    topic + a JSON header padded to _COMMAND_HEADER_SIZE + packed u8 data,
    all as one message. Needed because operator_state.start is the only
    way g1_deploy_onnx_ref's Control() state machine ever leaves
    WAIT_FOR_CONTROL (see InitControl()/Control() switch in
    g1_deploy_onnx_ref.cpp) -- without sending this, the deploy sits
    healthy-looking and idle forever, never reaching CONTROL, never
    running CreatePolicyCommand, no matter how many pose frames we relay.
    Found and fixed 2026-09-17, after an earlier sonic-lane sim session
    proved everything up to this point but ran out of time before wiring
    this in.
    """
    header = {
        "v": 1, "endian": "le", "count": 1,
        "fields": [
            {"name": "start", "dtype": "u8", "shape": [1]},
            {"name": "stop", "dtype": "u8", "shape": [1]},
            {"name": "planner", "dtype": "u8", "shape": [1]},
        ],
    }
    header_bytes = json.dumps(header).encode("utf-8")
    header_bytes += b"\x00" * (_COMMAND_HEADER_SIZE - len(header_bytes))
    data = struct.pack("BBB", int(start), int(stop), int(planner))
    return b"command" + header_bytes + data


def run_sonic(args, sub: zmq.Socket, stats: Stats):
    """Relay protocol-v4 pose frames straight through to gear_sonic_deploy.

    The organizer's sonic frame and the deploy's input frame are the same
    bytes (verified: `pose` topic, 1280B header, identical field set), so
    this forwards the ORIGINAL message rather than re-packing it -- no
    chance of a re-encode drifting from what the team actually published.
    """
    out = None
    if args.live:
        ctx = zmq.Context.instance()
        # Socket type CONFIRMED 2026-09-04: PUB is right. The deploy's own
        # reference publisher (tests/test_zmq_manager.py:26) is a zmq.PUB,
        # and its subscriber is a SUB, so PUB/SUB is the matched pair.
        out = ctx.socket(zmq.PUB if args.sonic_socket == "pub" else zmq.PUSH)
        out.setsockopt(zmq.LINGER, 0)
        # WE BIND, the deploy dials in. Verified 2026-09-04 against
        # gear_sonic_deploy's own code, not assumed: its
        # ZMQPackedMessageSubscriber::Connect() calls socket_->connect()
        # (zmq_packed_message_subscriber.hpp:202), and BOTH of its reference
        # publishers bind -- tests/test_zmq_manager.py:28 (host default "*")
        # and tests/zmq_pose_subscriber_test.cpp:28. This previously called
        # connect() on both ends, which ZMQ accepts silently while delivering
        # nothing: no error, no warning, just zero frames through.
        out.bind(f"tcp://*:{args.sonic_port}")
        print(f"[adapter] relaying pose frames on "
              f"tcp://*:{args.sonic_port} ({args.sonic_socket}, we bind; "
              f"start gear_sonic_deploy with --zmq-port {args.sonic_port})")

        # ZMQ's "slow joiner" behavior: messages sent on a PUB immediately
        # after bind can be dropped if no SUB has connected yet. The
        # deploy's own reference publisher sleeps 0.5s after bind for the
        # same reason (tests/test_zmq_manager.py:32) -- match it here
        # since the start command below is exactly the kind of one-shot
        # message slow-joiner drops would silently eat.
        time.sleep(0.5)

        # planner=False: sonic-lane teams stream pose frames from
        # their OWN external policy (this relay's whole job); gear_sonic_
        # deploy's internal planner is a different, unused input mode
        # (--input-type zmq_manager --planner-file ..., not --input-type
        # zmq). Sending planner=True here would ask the deploy to run a
        # planner nobody configured.
        out.send(_pack_command(start=True, stop=False, planner=False))
        print("[adapter] sent command(start=1, stop=0, planner=0) -- "
              "this is the ONLY thing that moves gear_sonic_deploy out of "
              "WAIT_FOR_CONTROL. Without it, everything looks healthy "
              "(connected, 0 rejected) but CreatePolicyCommand() never runs.")
    else:
        print("[adapter] DRY-RUN -- validating pose frames, relaying nothing.")

    last_report = time.monotonic()
    try:
        while True:
            try:
                msg = sub.recv()
            except zmq.Again:
                print("[adapter] no actions on :5556 yet (is the team's Orin client up?)")
                continue
            if not msg.startswith(POSE_TOPIC):
                continue

            stats.messages += 1
            try:
                problems = validate_pose(decode_pose(msg))
            except Exception as exc:
                stats.rejected += 1
                print(f"[adapter] undecodable pose frame: {exc}", file=sys.stderr)
                continue
            if problems:
                stats.rejected += 1
                print(f"[adapter] REJECTED pose: {'; '.join(problems)}", file=sys.stderr)
                continue

            if out is not None:
                out.send(msg)          # original bytes, deliberately not re-packed
                stats.published += 1

            if time.monotonic() - last_report > 2.0:
                print(stats.report())
                last_report = time.monotonic()
    except KeyboardInterrupt:
        if out is not None:
            # Graceful stop signal on clean shutdown. NOT a safety
            # mechanism -- the independent e-stop is what actually stops
            # the robot regardless of whether this fires (see --live's own
            # banner above). This just tells the deploy to stop cleanly
            # rather than leaving it mid-CONTROL when this process exits.
            out.send(_pack_command(start=False, stop=True, planner=False))
            print("[adapter] sent command(stop=1) before exiting")
        raise


def _refuse_incompatible_state_source(args):
    """Refuse to launch on a --state-source the upper-body mapper cannot use.

    Two consumers want different widths out of the same vector. ik.py wants
    the 29-wide Unitree body layout. UpperBodyMapper indexes the WBC's own
    model, whose upper_body group runs up to model index 42, so it needs the
    full 43-wide q. --state-source boundary (:5557) and --state-source zeros
    both deliver 29.

    Left unchecked this is not a startup failure -- it is accepted, the robot
    is engaged, and then the FIRST chunk dies inside
    UpperBodyMapper.current_upper_body() with

        ValueError: robot state q has 29 entries but upper_body indexes up to 42

    which is correct but arrives a second too late and reads like a mystery.
    Everything needed to rule it out is present the moment argv is parsed, so
    rule it out here, loudly, before anything connects to a robot.
    """
    if args.lane != "decoupled":
        return
    if args.state_source not in ("boundary", "zeros"):
        return
    if args.upper_body_from_model:
        why = "--upper-body-from-model was passed explicitly"
    elif args.live and args.wbc_backend == "ros2":
        why = "--live --wbc-backend ros2 implies --upper-body-from-model"
    else:
        return

    print("=" * 70, file=sys.stderr)
    print("LAUNCH DENIED -- incompatible options. Nothing was started and the",
          file=sys.stderr)
    print("robot was not contacted.", file=sys.stderr)
    print("", file=sys.stderr)
    print(f"  You asked for : --state-source {args.state_source}", file=sys.stderr)
    print(f"  Which implies : the robot state arrives 29 joints wide",
          file=sys.stderr)
    print(f"  But also      : {why}", file=sys.stderr)
    print(f"  Which needs   : the WBC's own 43-wide model vector, because the",
          file=sys.stderr)
    print(f"                  upper_body group indexes up to model index 42",
          file=sys.stderr)
    print("", file=sys.stderr)
    print("  Run it one of these two ways instead:", file=sys.stderr)
    print("", file=sys.stderr)
    print("    Drive the robot for real -- drop --state-source and let it",
          file=sys.stderr)
    print("    default to 'wbc', which is the 43-wide source the mapper needs:",
          file=sys.stderr)
    print(f"      --lane decoupled --live --wbc-backend ros2", file=sys.stderr)
    print("", file=sys.stderr)
    print(f"    Measure against the real arm with no WBC -- keep",
          file=sys.stderr)
    print(f"    --state-source {args.state_source} and drop --live and",
          file=sys.stderr)
    print("    --upper-body-from-model, so no robot model is loaded at all:",
          file=sys.stderr)
    print(f"      --lane decoupled --state-source {args.state_source} --verbose",
          file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lane", required=True, choices=LANES,
                   help="must match the team's manifest.yaml")
    p.add_argument("--actions-host", default="127.0.0.1",
                   help="where the team's client bound :5556 (its own host)")
    p.add_argument("--actions-port", type=int, default=5556)
    p.add_argument("--live", action="store_true",
                   help="actually drive the robot. Default: dry-run.")
    p.add_argument("--engage-policy", action="store_true",
                   help="2026-09-14: send {'toggle_policy_action': True} once, "
                        "as soon as the WBC's own state topic is confirmed live. "
                        "Without this, G1GearWbcPolicy.use_policy_action stays at "
                        "its constructor default (False) for the entire run -- "
                        "NVIDIA's own teleop safe-mode, which holds the robot's "
                        "current measured joint position every tick rather than "
                        "running the trained RL balance policy. Looks completely "
                        "healthy (WBC up, topics ticking at 50Hz, this adapter "
                        "publishing with 0 rejected) while never actually engaging "
                        "balance -- found by decoding "
                        "/ControlPolicy/lower_body_policy_status directly, since "
                        "nothing else in the stack surfaces this. Only meaningful "
                        "with --live --wbc-backend ros2; ignored otherwise. This "
                        "is a TOGGLE: sent exactly once per run, never repeated, "
                        "since a second send would disengage it again.")
    p.add_argument("--verbose", action="store_true")
    # decoupled
    p.add_argument("--wbc-backend", default="ros2", choices=("ros2", "zmq"),
                   help="ros2 = the real Decoupled WBC; zmq = bench loopback")
    p.add_argument("--upper-body-from-model", action="store_true",
                   help="force querying decoupled_wbc's robot model for the "
                        "upper-body layout (implied by --live --wbc-backend "
                        "ros2). Requires --state-source wbc (the default): "
                        "the mapper indexes up to model index 42, which the "
                        "29-wide 'boundary' and 'zeros' sources cannot "
                        "provide. The combination is refused at startup.")
    p.add_argument("--enable-waist", action="store_true",
                   help="set iff run_g1_control_loop.py runs with waist in the "
                        "upper-body group (width 17 vs 14)")
    p.add_argument("--chunk-hz", type=float, default=20.0,
                   help="cadence the (T,25) rows are meant to play out at")
    p.add_argument("--max-waypoints", type=int, default=16)
    p.add_argument("--max-joint-vel", type=float, default=1.0,
                   help="rad/s cap on how fast any commanded arm joint may move "
                        "between scheduled waypoints, regardless of what the raw "
                        "IK solution implies. Margin under the WBC's own real-"
                        "hardware joint safety monitor (+-6 rad/s, joint_safety.py) "
                        "-- IK has no notion of the schedule's timing, so an "
                        "unthrottled solve far from the current pose can exceed "
                        "that limit and trip a hard shutdown. 0 disables. "
                        "2026-08-25: lowered from 4.0 -- across four live "
                        "violations this session, ACTUAL measured joint velocity "
                        "reached up to 3.0x this commanded cap (12.03 rad/s "
                        "actual vs a 4.0 cap), a real gap between commanded and "
                        "realized motion this adapter doesn't fully explain yet "
                        "(clamp-reference staleness fixes reduced but did not "
                        "eliminate it). "
                        "2026-09-03: lowered again, 2.0 -> 1.0. One team's "
                        "session tripped right_elbow_joint at -7.153 (WBC's own "
                        "reported figure) to -7.706 rad/s (independently "
                        "recomputed from capture_evidence.py's raw body_q "
                        "samples, ~20ms apart) against a 2.0 commanded cap -- "
                        "~3.6-3.9x amplification, WORSE than the 3.0x this "
                        "comment already flagged as unexplained, not better. "
                        "Ruled out: the already-documented unclamped path (WBC's "
                        "own >1.0s teleop-timeout injecting an unclamped safe "
                        "goal) -- no 'Teleop mode timeout' line anywhere near "
                        "this violation in the WBC log, so this went through "
                        "our own solve->clamp->publish path, not around it. "
                        "target_time spacing was also checked and matches the "
                        "dt this clamp assumes (times = t_base + (i+1)/chunk_hz, "
                        "same chunk_hz used for max_step), so it isn't a simple "
                        "units/timing mismatch either. Real body_q samples show "
                        "the joint smoothly RISING for ~360ms right before the "
                        "trip, then reversing hard within one ~23ms sample -- "
                        "consistent with (not proven as) a position-only clamp "
                        "saying nothing about the arm's existing momentum when a "
                        "reversal is commanded, so tracking a same-magnitude "
                        "position step in the opposite direction of travel can "
                        "demand more real velocity than the step size alone "
                        "implies. Since the amplification factor itself is "
                        "trending worse with each measurement, not converging, "
                        "1.0 buys real margin against that uncertainty rather "
                        "than assuming 3x again: even at this session's ~3.9x, "
                        "worst case lands ~3.9 rad/s, clear of the 6.0 limit. "
                        "The amplification mechanism is still not understood -- "
                        "this is a mitigation, not a fix for the root cause.")
    p.add_argument("--max-ik-err", type=float, default=1e-3)
    p.add_argument("--ik-warm-start", default="current", choices=("current", "last"),
                   help="'current' (default) seeds IK from the measured arm pose "
                        "every solve -- deterministic, reproducible, required for "
                        "scored attempts. 'last' seeds from the previous solution: "
                        "faster, but makes results depend on message order/timing.")
    p.add_argument("--state-source", default="wbc",
                   choices=("wbc", "boundary", "zeros"),
                   help="where real body_q comes from. 'wbc' = the WBC's own "
                        "state topic (authoritative, needs --live/ros2). "
                        "'boundary' = the organizer's :5557 endpoint, which "
                        "lets a dry run measure against the real arm with no "
                        "WBC at all. 'zeros' = plumbing smoke test only. "
                        "INCOMPATIBLE with the upper-body mapper: 'boundary' "
                        "and 'zeros' are 29 joints wide, while the mapper "
                        "indexes the WBC's model up to index 42 and needs the "
                        "43-wide 'wbc' source. Combining them is refused at "
                        "startup -- see --upper-body-from-model.")
    p.add_argument("--orin-host", default="127.0.0.1",
                   help="host serving the organizer's :5555/:5557 endpoints")
    p.add_argument("--state-port", type=int, default=5557)
    p.add_argument("--dex1-port", type=int, default=5599,
                   help="where run_wbc_with_dex1.py listens for gripper "
                        "targets. 0 disables (no grasp possible).")
    p.add_argument("--dex1-host", default="127.0.0.1")
    p.add_argument("--max-chunk-age-s", type=float, default=1.0,
                   help="drop a chunk older than this (0 disables). Guards "
                        "against acting on a stale plan after a stall.")
    # joint lane (rides on the decoupled stack; see the module docstring)
    p.add_argument("--joint-lane", default="on", choices=("on", "off"),
                   help="accept the b'joint' (T,22) joint-angle chunks and b'goto' "
                        "pose requests on the same :5556 socket, alongside "
                        "b'taskspace'. The taskspace path is unaffected either "
                        "way. 'off' ignores both topics like any unknown prefix.")
    p.add_argument("--joint-lane-limits", default="urdf", choices=("urdf", "ik"),
                   help="which position limits the joint lane clamps arm angles "
                        "to (never typed in; read from the robot model). 'urdf' "
                        "(default) = the limits exactly as the WBC's own robot "
                        "model loads them -- the URDF plus the WBC's supplemental "
                        "narrowing (shoulder_roll kept +-0.19 rad from the "
                        "torso), i.e. the same table its JointSafetyMonitor "
                        "enforces on the real robot. 'ik' additionally applies "
                        "ik.py's solver-side overrides (elbow upper bound 1.4, "
                        "wrist_roll +-0.9) so the joint lane ranges over the "
                        "same space the taskspace lane's IK does; those exist "
                        "to steer a redundant solution, not to protect hardware, "
                        "so they are not the default. Switch if ruled.")
    p.add_argument("--goto-max-speed", type=float, default=0.45,
                   help="rad/s ceiling on a b'goto' request's own max_speed. "
                        "Also capped by --max-joint-vel so the step clamp never "
                        "shortens the ramp. 0.45 is the speed the joint-space "
                        "pre-motion that preceded this lane ran at live.")
    # sonic
    p.add_argument("--sonic-host", default="127.0.0.1")
    p.add_argument("--sonic-port", type=int, default=5580,
                   help="gear_sonic_deploy's zmq input endpoint (--input-type "
                        "zmq, NOT the default --input-type zmq_manager -- that "
                        "one runs an internal planner nobody wants here). "
                        "CONFIRMED 2026-09-04: g1_deploy_onnx_ref.cpp's own "
                        "--zmq-port compiles in a default of 5556, same as "
                        "the boundary's action port -- deploy.sh can't even "
                        "pass --zmq-port through, so reaching this requires "
                        "calling `just run g1_deploy_onnx_ref` directly with "
                        "--zmq-port matching this flag. 5580 isn't special, "
                        "just deliberately not 5555/5556/5557 (the organizer's "
                        "camera/action/state ports) -- confirm whatever port "
                        "gear_sonic_deploy is actually launched with matches "
                        "this value exactly.")
    p.add_argument("--sonic-socket", default="pub", choices=("pub", "push"))
    args = p.parse_args()
    _refuse_incompatible_state_source(args)

    print(f"[adapter] lane={args.lane} "
          f"actions=tcp://{args.actions_host}:{args.actions_port} "
          f"{'LIVE' if args.live else 'DRY-RUN'}")
    if args.live:
        print("=" * 70)
        print("LIVE -- this drives the real robot. Confirm the robot is")
        print("supported as your setup requires and the independent e-stop is")
        print("staffed. This adapter is not a safety system; the e-stop is.")
        print("=" * 70)
        input("Press Enter to proceed, Ctrl+C to abort...")

    sub = make_action_subscriber(args.actions_host, args.actions_port,
                                 conflate=(args.lane == "decoupled"))
    stats = Stats()
    try:
        if args.lane == "decoupled":
            run_decoupled(args, sub, stats)
        else:
            run_sonic(args, sub, stats)
    except KeyboardInterrupt:
        print(f"\n[adapter] stopped. {stats.report()}")


if __name__ == "__main__":
    main()
