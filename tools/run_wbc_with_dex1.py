#!/usr/bin/env python3
"""Launch NVIDIA's Decoupled WBC with Dex1-1 gripper actuation added.

Drop-in replacement for running `run_g1_control_loop.py` directly. Takes
the SAME arguments and passes them straight through:

    python3 run_wbc_with_dex1.py --interface $IFACE --no-with-hands

--------------------------------------------------------------------------
WHY THIS EXISTS
--------------------------------------------------------------------------
NVIDIA's stack is Dex3-only. Hand commands go through
`G1ThreeFingerHand`/`HandCommandSender` on `rt/dex3/{left,right}/cmd`,
which this rig has no hardware for. With `--no-with-hands` (required here)
NOTHING actuates the Dex1-1 gripper, so a team's `(T,25)` hand columns are
computed and discarded and a grasp cannot physically happen.

The Dex1-1 on this robot sits at motor indices **31 and 33 of the same
35-slot `rt/lowcmd` array** as the body (empirically confirmed by the
a prior bench session: those slots tracked the real gripper as it was opened and
closed by hand, and driving them worked). So the gripper cannot be driven
by a separate process -- `LowCmd_` is a whole-message publish at 50 Hz and
two publishers would race, dropping the body into mode 0 every other
message. It has to go out on the SAME message the body controller already
sends.

--------------------------------------------------------------------------
WHAT IT DOES
--------------------------------------------------------------------------
Wraps `BodyCommandSender.send_command`. That method mutates a persistent
`self.low_cmd`, computes the CRC, and publishes. We set the gripper slots
on that same object *before* delegating, so the original's CRC covers them
and the message stays valid. NVIDIA's repo is not modified at all -- remove
this launcher and the behaviour is exactly stock.

Gripper targets arrive over a local ZMQ SUB (default `:5599`), published by
`wbc_driver.py --dex1-port 5599`. Local socket, same host, no DDS
involvement.

Safety properties, deliberate:
  * Until a command arrives, the gripper slots are NOT touched at all.
  * Commands are rate-limited (`--dex1-max-speed`, rad/s of raw motor q),
    so a step change in the policy's output cannot slam the gripper.
  * Targets are clamped to the calibrated open/closed range.
  * kp/kd are the soft gripper gains, never the arms' stiff ones.

--------------------------------------------------------------------------
START-UP POSE (--seed-from-measured, default ON)
--------------------------------------------------------------------------
Stock, `get_wbc_policy` seeds the upper-body `InterpolationPolicy` with
`robot_model.get_initial_upper_body_pose()`, which is the model's constant
rest pose (shoulder_roll +-0.2 rad, every other arm joint 0), and
`JointSafetyMonitor.get_safe_action` then ramps the arm command linearly
from the first measured q to that pose over 100 steps (2 s) at full kp. So
at every launch the arms swing to the rest pose before any client connects.

`RobotModel.set_initial_body_pose(q)` exists for exactly this (NVIDIA's own
`run_sync_sim_data_collection.py` calls it with `obs["q"]`) but the real
control loop never does. This wrapper wraps `get_wbc_policy` as seen from
`run_g1_control_loop`'s namespace: it waits for a valid observation from the
`G1Env` the loop just built, copies the 43-wide measured q with the 14 hand
slots zeroed (Dex1 has no hand state; the model default is zero anyway),
calls `robot_model.set_initial_body_pose(q)`, then hands off to the stock
factory with identical arguments. The interpolator therefore starts AT the
measured pose, the 2 s ramp becomes a no-op, and the arms hold where they
are. `--enable-waist` is covered automatically: the seeded vector is the
full model q, so the waist joints in the upper-body group get their
measured values too.

If no valid state arrives within 5 s (observe() raising before the first
rt/lowstate, or a 29-joint body vector that is all exactly zero, which a
fresh process can report before the first message) it warns loudly and
falls back to stock behaviour. `--no-seed-from-measured` restores stock
behaviour outright.

CALIBRATION (measured on one specific unit -- re-measure for your rig):
    index 31 = left, 33 = right
    q =  0.00  CLOSED
    q = -5.30  OPEN          <- sign is FLIPPED vs Unitree's reference,
    kp = 5.0, kd = 0.05         which uses 0..+5.4. Verified by watching
                                motor_state while moving it by hand.
Re-verify with `diagnose_dex1.py` before trusting them on a different unit.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

GRIPPER_INDEX = {"left": 31, "right": 33}
DEX1_CLOSED_Q = 0.0
DEX1_OPEN_Q = -5.30
GRIPPER_KP = 5.0
GRIPPER_KD = 0.05
DEX1_TOPIC = b"dex1"


def hand_norm_to_dex1_q(norm: float) -> float:
    """boundary/actions.py convention: -1 = open, +1 = closed."""
    norm = float(np.clip(norm, -1.0, 1.0))
    return DEX1_CLOSED_Q + (DEX1_OPEN_Q - DEX1_CLOSED_Q) * (1.0 - norm) / 2.0


class Dex1Injector:
    def __init__(self, port: int, max_speed: float, verbose: bool = False):
        import zmq
        self._zmq = zmq
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, DEX1_TOPIC)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(f"tcp://127.0.0.1:{port}")
        self.max_speed = max_speed
        self.verbose = verbose

        self.target = {"left": None, "right": None}
        self.current = {"left": None, "right": None}
        self._lock = threading.Lock()
        self._last_t = None
        self.commands_seen = 0
        threading.Thread(target=self._rx, daemon=True).start()
        print(f"[dex1] listening on tcp://127.0.0.1:{port}; "
              f"max speed {max_speed} rad/s; NOT commanding until a target arrives")

    def _rx(self):
        import msgpack
        while True:
            try:
                blob = self.sock.recv()
            except Exception:
                time.sleep(0.05)
                continue
            try:
                msg = msgpack.unpackb(blob[len(DEX1_TOPIC):], raw=False)
                with self._lock:
                    for side in ("left", "right"):
                        if side in msg:
                            self.target[side] = hand_norm_to_dex1_q(msg[side])
                    self.commands_seen += 1
            except Exception:
                pass

    def apply(self, low_cmd):
        """Mutate low_cmd's gripper slots. Called before the CRC is taken."""
        with self._lock:
            target = dict(self.target)
        if target["left"] is None and target["right"] is None:
            return  # nothing commanded yet -- leave the slots untouched

        now = time.monotonic()
        dt = 0.02 if self._last_t is None else min(max(now - self._last_t, 1e-3), 0.1)
        self._last_t = now
        step = self.max_speed * dt

        lo, hi = min(DEX1_OPEN_Q, DEX1_CLOSED_Q), max(DEX1_OPEN_Q, DEX1_CLOSED_Q)
        for side, idx in GRIPPER_INDEX.items():
            tgt = target[side]
            if tgt is None:
                continue
            tgt = float(np.clip(tgt, lo, hi))
            cur = self.current[side]
            if cur is None:
                cur = tgt          # first command: adopt it, then rate-limit
            cur += float(np.clip(tgt - cur, -step, step))
            self.current[side] = cur

            m = low_cmd.motor_cmd[idx]
            m.mode = 1
            m.q = cur
            m.dq = 0.0
            m.tau = 0.0
            m.kp = GRIPPER_KP
            m.kd = GRIPPER_KD


def install(port: int, max_speed: float, verbose: bool):
    from decoupled_wbc.control.envs.g1.utils import command_sender as cs

    injector = Dex1Injector(port, max_speed, verbose)
    original = cs.BodyCommandSender.send_command

    def send_command_with_dex1(self, cmd_q, cmd_dq, cmd_tau):
        try:
            # low_cmd is persistent on the sender; setting the gripper slots
            # here means the original's CRC covers them.
            injector.apply(self.low_cmd)
        except Exception as exc:            # never take the body down for a gripper
            print(f"[dex1] injector error (body unaffected): {exc}", file=sys.stderr)
        return original(self, cmd_q, cmd_dq, cmd_tau)

    cs.BodyCommandSender.send_command = send_command_with_dex1
    print("[dex1] BodyCommandSender.send_command wrapped "
          f"(motors {GRIPPER_INDEX['left']}/{GRIPPER_INDEX['right']}, "
          f"kp={GRIPPER_KP}, kd={GRIPPER_KD})")
    return injector


SEED_TIMEOUT_S = 5.0


def wait_for_measured_q(env, robot_model, timeout_s: float = SEED_TIMEOUT_S,
                        poll_s: float = 0.02):
    """Poll env.observe() until it yields a usable model-space q, or time out.

    Usable means: observe() did not raise (on the real robot the state
    processor returns None before the first rt/lowstate and G1Body.observe
    then raises), obs["q"] has the model's width, and the body joints (all
    non-hand slots) are not all exactly zero. Returns a float64 copy of q
    with the hand slots zeroed, or None on timeout.
    """
    hand_idx = list(robot_model.get_joint_group_indices("hands"))
    width = int(getattr(robot_model, "num_dofs", 43))
    body_mask = np.ones(width, dtype=bool)
    body_mask[hand_idx] = False
    deadline = time.monotonic() + timeout_s
    last_reason = "observe() never returned"
    while True:
        try:
            obs = env.observe()
        except Exception as exc:              # no low state yet
            obs, last_reason = None, f"observe() raised {type(exc).__name__}: {exc}"
        q = None if not isinstance(obs, dict) else obs.get("q")
        if q is not None:
            q = np.array(q, dtype=np.float64)
            if q.shape != (width,):
                last_reason = f"obs['q'] has shape {q.shape}, expected ({width},)"
            elif not np.any(q[body_mask] != 0.0):
                last_reason = "body joints all exactly zero (no real rt/lowstate yet)"
            else:
                q[hand_idx] = 0.0
                return q
        if time.monotonic() >= deadline:
            print(f"[seed] no valid observation within {timeout_s:.1f}s: {last_reason}",
                  file=sys.stderr)
            return None
        time.sleep(poll_s)


def install_seed_from_measured_patch(loop_mod=None, timeout_s: float = SEED_TIMEOUT_S):
    """Make the control loop seed its upper-body interpolator from measured q.

    Wraps `G1Env` and `get_wbc_policy` AS REFERENCED FROM run_g1_control_loop's
    module namespace (both are from-imports there, so patching the source
    modules would not reach `main`). The G1Env wrapper only records the env
    instance the loop builds; the get_wbc_policy wrapper waits for that env's
    first valid observation, calls `robot_model.set_initial_body_pose(q)`,
    and then calls the stock factory with the caller's arguments forwarded
    untouched (the stock caller passes upper_body_joint_speed as the 4th
    POSITIONAL, which lands on the factory's `init_time`; that quirk is
    preserved, not fixed, here).

    `loop_mod` defaults to the real run_g1_control_loop module; tests pass a
    namespace with fake `G1Env` / `get_wbc_policy` attributes. Returns a dict
    with the captured env and the seeded q for inspection.
    """
    if loop_mod is None:
        from decoupled_wbc.control.main.teleop import run_g1_control_loop as loop_mod

    original_env_cls = loop_mod.G1Env
    original_get_wbc_policy = loop_mod.get_wbc_policy
    state = {"env": None, "seeded_q": None, "fell_back": False}

    def G1Env_capturing(*args, **kwargs):
        env = original_env_cls(*args, **kwargs)
        state["env"] = env
        return env

    def get_wbc_policy_seeded(robot_type, robot_model, wbc_config, *args, **kwargs):
        env = state["env"]
        q = None
        if env is None:
            print("[seed] WARNING: no G1Env was constructed through the patched name; "
                  "cannot read the measured pose", file=sys.stderr)
        else:
            q = wait_for_measured_q(env, robot_model, timeout_s=timeout_s)
        if q is None:
            state["fell_back"] = True
            print("[seed] WARNING: falling back to STOCK start-up: the interpolator is "
                  "seeded with the model rest pose and the arms WILL ramp to it "
                  "(shoulder_roll +-0.2, else 0) over 2 s at full kp", file=sys.stderr)
        else:
            robot_model.set_initial_body_pose(q)
            state["seeded_q"] = q
            arm_idx = list(robot_model.get_joint_group_indices("arms"))
            arms = " ".join(f"{v:+.3f}" for v in q[arm_idx])
            print(f"[seed] upper-body interpolator seeded from MEASURED q; "
                  f"14 arm joints (L sp sr sy el wr wp wy | R same) = {arms}")
        return original_get_wbc_policy(robot_type, robot_model, wbc_config, *args, **kwargs)

    loop_mod.G1Env = G1Env_capturing
    loop_mod.get_wbc_policy = get_wbc_policy_seeded
    print("[seed] run_g1_control_loop.get_wbc_policy wrapped: start-up pose will be the "
          f"measured pose (waits up to {timeout_s:.0f}s for a valid state)")
    return state


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dex1-port", type=int, default=5599)
    pre.add_argument("--dex1-max-speed", type=float, default=2.0,
                     help="rad/s of raw Dex1 motor q; full stroke is ~5.3")
    pre.add_argument("--dex1-verbose", action="store_true")
    pre.add_argument("--no-dex1", action="store_true",
                     help="run completely stock, no gripper injection")
    pre.add_argument("--seed-from-measured", action=argparse.BooleanOptionalAction,
                     default=True,
                     help="at launch, seed the WBC's upper-body interpolator with the "
                          "MEASURED joint angles so the arms hold where they are instead "
                          "of swinging to the model's rest pose over 2 s "
                          "(--no-seed-from-measured restores stock behaviour)")
    args, passthrough = pre.parse_known_args()

    if not args.no_dex1:
        install(args.dex1_port, args.dex1_max_speed, args.dex1_verbose)
    else:
        print("[dex1] --no-dex1: running stock, gripper will NOT actuate")

    # Independent of the Dex1 patch: seed the start-up pose from the measured
    # joints (see START-UP POSE in the module docstring).
    if args.seed_from_measured:
        install_seed_from_measured_patch()
    else:
        print("[seed] --no-seed-from-measured: stock start-up, the arms will ramp to "
              "the model rest pose (shoulder_roll +-0.2, else 0) over 2 s")

    # 2026-09-03: default NVIDIA's own preventive upper-body velocity limit
    # to something real. ControlLoopConfig.upper_body_joint_speed defaults to
    # 1000 rad/s, which feeds PoseTrajectoryInterpolator.schedule_waypoint()
    # as max_change_rate:
    #     pose_min_duration = max(|end_pose - pose| / max_change_rate)
    #     duration = max(duration, pose_min_duration)
    # i.e. it STRETCHES a waypoint's arrival time so the interpolated
    # trajectory never demands more than max_change_rate. At 1000 rad/s
    # pose_min_duration is ~0 for any real joint delta, so that clamp never
    # engages and the interpolator will demand literally any velocity the
    # waypoint spacing implies. It is dead code at the default.
    #
    # Why this matters more than wbc_driver.py's own --max-joint-vel: that
    # one bounds what we ASK for, between waypoints we publish. This one
    # bounds what the WBC actually DEMANDS OF THE MOTORS between them.
    # Measured across three live violations on this rig, all right_elbow:
    #     --max-joint-vel 4.0 -> 12.03 rad/s realized (3.0x)
    #     --max-joint-vel 2.0 ->  7.15 rad/s realized (3.6x)
    #     --max-joint-vel 1.0 ->  7.01 rad/s realized (7.0x)
    # Halving our own cap 2.0 -> 1.0 barely moved the realized figure --
    # it plateaus around 7 rad/s, i.e. our clamp had stopped being the
    # binding constraint at all and the remaining motion was generated
    # downstream, here.
    #
    # 3.0 rad/s chosen from measured real motion on this rig, not by
    # analogy: across 37404 samples x 14 arm joints of a real run,
    # p99.9 = 0.349 rad/s and only 0.0015% of samples exceed 3.0 -- so this
    # is inert for essentially all genuine motion while sitting 2x under
    # the WBC's own hard joint_safety.py limit (+-6.0 rad/s). Margin is
    # deliberately conservative: the commanded-vs-realized gap has been
    # underestimated three times running, so this does not assume the
    # interpolator's bound is perfectly realized either.
    #
    # Team-agnostic and applied identically to every team, same as every
    # other value in this pipeline. Override by passing the flag yourself.
    if not any(a.startswith("--upper-body-joint-speed") for a in passthrough):
        passthrough += ["--upper-body-joint-speed", "3.0"]
        print("[wbc] defaulting --upper-body-joint-speed 3.0 "
              "(NVIDIA's own default of 1000 rad/s disables its preventive "
              "interpolator velocity limit entirely)")

    # Hand the remaining args to NVIDIA's own entrypoint, untouched.
    sys.argv = [sys.argv[0]] + passthrough
    import tyro
    from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
    from decoupled_wbc.control.main.teleop.run_g1_control_loop import main as wbc_main
    wbc_main(tyro.cli(ControlLoopConfig))


if __name__ == "__main__":
    main()
