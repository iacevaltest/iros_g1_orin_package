#!/usr/bin/env python3
"""Diagnose how the Dex1-1 gripper is actually wired on this G1.

READ-ONLY. Subscribes and observes; never publishes, never commands a
motor. Safe to run with the robot powered and standing.

    python3 diagnose_dex1.py --iface <the 192.168.123.x interface>

Run it in an env that has unitree_sdk2py (e.g. a dedicated venv
works; so does the WBC conda env).

--------------------------------------------------------------------------
WHY: two hypotheses, opposite fixes
--------------------------------------------------------------------------
H1  Dex1 is just two extra motors on the MAIN hg motor bus, at indices
    31/33 of the same 35-slot array as the body. There is then NO separate
    gripper service to revive, and `rt/lowcmd` is the only way to drive it
    -- which conflicts with whoever owns the body (the WBC, or
    gear_sonic_deploy). Fix = coordinate through the body controller.

H2  Dex1 has its own device/service like Dex3 does (NVIDIA drives Dex3 on
    `rt/dex3/{left,right}/cmd` with HandCmd_), and it is simply not
    running. Fix = start it, then implement the sidecar's UnitreeDriver
    and drive the gripper independently of rt/lowcmd -- which also works
    for the sonic lane.

Existing evidence leans H1: a prior bench session watched `motor_state[31]`/`[33]` track
the real gripper as it was opened and closed by hand, and drove it
successfully over `rt/lowcmd`. This script is here to settle it.

The distinguishing test is section 3: move the gripper BY HAND and see
which interface reports the movement.
"""
from __future__ import annotations

import argparse
import time

CANDIDATE_HAND_TOPICS = [
    # Dex3 naming, confirmed used by NVIDIA's stack -- listed so we can see
    # whether the pattern exists at all on this robot.
    ("rt/dex3/left/state", "HandState_"),
    ("rt/dex3/right/state", "HandState_"),
    # Plausible Dex1 analogues. None of these appear in NVIDIA's code or in
    # unitree_sdk2py; they are guesses precisely because the naming is
    # undocumented for this gripper.
    ("rt/dex1/left/state", "HandState_"),
    ("rt/dex1/right/state", "HandState_"),
    ("rt/dex1/state", "HandState_"),
    ("rt/gripper/state", "HandState_"),
]


def section(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--iface", default=None, help="robot DDS interface (192.168.123.x side)")
    p.add_argument("--seconds", type=float, default=6.0)
    args = p.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    # ---------------------------------------------------------------- 1
    section("1. Is rt/lowstate alive, and how many motor slots does it carry?")
    box = {"msg": None, "n": 0}

    def on_low(m):
        box["msg"] = m
        box["n"] += 1

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_low, 10)
    t0 = time.time()
    while box["msg"] is None and time.time() - t0 < 5:
        time.sleep(0.1)

    if box["msg"] is None:
        print("  FAIL: no rt/lowstate. Wrong --iface, or the robot's low-level")
        print("        service is not running. Nothing below will work.")
        return 1

    ms = box["msg"].motor_state
    print(f"  ok: rt/lowstate live, {len(ms)} motor slots")
    print("  slots 28..34 (Dex1 is expected at 31 and 33):")
    for i in range(28, min(35, len(ms))):
        tag = "  <-- Dex1 LEFT?" if i == 31 else ("  <-- Dex1 RIGHT?" if i == 33 else "")
        print(f"    [{i:2d}] q={ms[i].q:+8.4f}  dq={ms[i].dq:+7.3f}  "
              f"tau={ms[i].tau_est:+7.3f}{tag}")

    # ---------------------------------------------------------------- 2
    section("2. Does any hand/gripper DDS topic publish state?")
    print("  (Dex3 names are what NVIDIA's stack uses; the rt/dex1* names are")
    print("   guesses -- no such topic appears in their code or in the SDK.)")
    found = {}
    subs = []
    for topic, _ in CANDIDATE_HAND_TOPICS:
        cnt = {"n": 0}

        def mk(c):
            def cb(_m):
                c["n"] += 1
            return cb
        try:
            s = ChannelSubscriber(topic, HandState_)
            s.Init(mk(cnt), 10)
            subs.append(s)
            found[topic] = cnt
        except Exception as exc:
            print(f"  {topic:28s} subscribe failed: {type(exc).__name__}")
    time.sleep(args.seconds)
    for topic, cnt in found.items():
        verdict = f"{cnt['n']} msgs" if cnt["n"] else "SILENT"
        print(f"  {topic:28s} {verdict}")

    if not any(c["n"] for c in found.values()):
        print("\n  -> No hand-style topic publishes anything. Consistent with H1")
        print("     (Dex1 is on the main motor bus, no separate service), and")
        print("     also with H2 (service exists but is stopped). Section 3")
        print("     distinguishes them.")

    # ---------------------------------------------------------------- 3
    section("3. THE DECIDING TEST -- move the gripper BY HAND now")
    print("  Open and close each gripper by hand for the next "
          f"{args.seconds:.0f}s.")
    print("  If slots 31/33 change, Dex1 reports through rt/lowstate -> H1:")
    print("  it is on the main motor bus and rt/lowcmd is the only way in.\n")
    start = {i: ms[i].q for i in range(28, min(35, len(ms)))}
    spread = {i: [ms[i].q, ms[i].q] for i in start}
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        m = box["msg"]
        if m is not None:
            for i in spread:
                q = m.motor_state[i].q
                spread[i][0] = min(spread[i][0], q)
                spread[i][1] = max(spread[i][1], q)
        time.sleep(0.02)

    print("  slot   min       max       travel")
    moved = []
    for i in sorted(spread):
        lo, hi = spread[i]
        tr = hi - lo
        if tr > 0.05:
            moved.append(i)
        print(f"   [{i:2d}] {lo:+8.4f}  {hi:+8.4f}  {tr:7.4f}"
              + ("   <-- MOVED" if tr > 0.05 else ""))

    section("VERDICT")
    if moved:
        print(f"  Slots {moved} moved in rt/lowstate.")
        print("  => H1 CONFIRMED: Dex1 sits on the main hg motor array.")
        print("     There is no separate gripper service to revive, so the")
        print("     sidecar's UnitreeDriver has nothing independent to talk")
        print("     to. The gripper can only be driven via rt/lowcmd, which")
        print("     the body controller owns -- so the command must go")
        print("     THROUGH that controller, not alongside it.")
        print("     For sonic (a C++ binary) that means the deploy binary")
        print("     needs to carry the gripper, or the gripper is out of")
        print("     scope for that lane.")
    elif any(c["n"] for c in found.values()):
        live = [t for t, c in found.items() if c["n"]]
        print(f"  Hand topics publishing: {live}")
        print("  => H2: a hand-style service IS alive. Implement the sidecar's")
        print("     UnitreeDriver against it -- that path is independent of")
        print("     rt/lowcmd and therefore works for BOTH lanes.")
    else:
        print("  Nothing moved and no hand topic published.")
        print("  Either the gripper was not actually moved during section 3,")
        print("  or it is unpowered/disconnected. Re-run and make sure the")
        print("  grippers physically move. If slots 31/33 stay flat while the")
        print("  gripper clearly moves, the feedback path itself is broken and")
        print("  that is a hardware/firmware question for Unitree.")
    print("\n  (Reference calibration for one unit, if H1: indices 31/33,")
    print("   q=0.0 closed, q=-5.30 open, kp=5.0, kd=0.05 -- sign-flipped")
    print("   from Unitree's reference, so trust these, not the docs.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
