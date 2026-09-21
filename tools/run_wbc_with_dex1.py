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


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dex1-port", type=int, default=5599)
    pre.add_argument("--dex1-max-speed", type=float, default=2.0,
                     help="rad/s of raw Dex1 motor q; full stroke is ~5.3")
    pre.add_argument("--dex1-verbose", action="store_true")
    pre.add_argument("--no-dex1", action="store_true",
                     help="run completely stock, no gripper injection")
    args, passthrough = pre.parse_known_args()

    if not args.no_dex1:
        install(args.dex1_port, args.dex1_max_speed, args.dex1_verbose)
    else:
        print("[dex1] --no-dex1: running stock, gripper will NOT actuate")

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
