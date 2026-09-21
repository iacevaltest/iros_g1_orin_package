#!/usr/bin/env python3
"""Preflight for running NVIDIA's Decoupled WBC on the G1's Orin.

Run this ON THE ORIN, inside the conda env you built for the WBC, BEFORE
trying to start `run_g1_control_loop.py`. Every check maps to a real
requirement read out of GR00T-WholeBodyControl v1.1 (commit a0732b6), and
several map to landmines this rig has actually hit before.

    python3 preflight_orin.py --repo ~/GR00T-WholeBodyControl --interface eth0

Exit code 0 = everything required passed. Nothing here touches the robot;
it is all read-only inspection.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
_results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = ""):
    _results.append((status, name, detail))
    mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
    print(f"[{mark}] {name}" + (f"\n           {detail}" if detail else ""))


def check_python():
    v = sys.version_info
    if (v.major, v.minor) >= (3, 10):
        record(OK, f"Python {v.major}.{v.minor}.{v.micro} (>=3.10 required)")
    else:
        record(FAIL, f"Python {v.major}.{v.minor}.{v.micro} is too old",
               "decoupled_wbc requires >=3.10, but JetPack 5.1.1 ships 3.8 as the "
               "system Python. Build a conda env with python=3.10 and run "
               "everything (including this script) inside it -- do NOT try to "
               "upgrade the system Python on the robot.")


def check_imports():
    required = {
        "numpy": "numpy==1.26.4 is pinned by decoupled_wbc",
        "scipy": "scipy==1.15.3 is pinned by decoupled_wbc",
        "torch": "lower-body policy runtime",
        "onnxruntime": "runs the Balance/Walk ONNX policies",
        "pinocchio": "provided by the `pin` package",
        "pink": "provided by `pin-pink`; upper-body IK",
        "qpsolvers": "IK QP backend (osqp/quadprog)",
        "msgpack": "goal/state wire format",
        "msgpack_numpy": "goal/state wire format",
        "zmq": "provided by `pyzmq`",
        "yaml": "provided by PyYAML; WBC config files",
        "tyro": "CLI parsing for run_g1_control_loop.py",
    }
    for mod, why in required.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            record(OK, f"import {mod} ({ver})")
        except Exception as exc:
            record(FAIL, f"import {mod} failed", f"{why}. {type(exc).__name__}: {exc}")

    # numpy 2.x breaks things compiled against 1.x, and decoupled_wbc pins 1.26.4.
    try:
        import numpy as np
        if not np.__version__.startswith("1.26"):
            record(WARN, f"numpy is {np.__version__}, not the pinned 1.26.4",
                   "Anything in this env compiled against numpy 1.x can fail in "
                   "confusing ways under 2.x rather than failing cleanly.")
    except Exception:
        pass


def check_ros2():
    try:
        import rclpy  # noqa: F401
        record(OK, "import rclpy")
    except Exception as exc:
        record(FAIL, "import rclpy failed",
               f"{type(exc).__name__}: {exc}. ROS 2 Humble is required. Official "
               "Humble binaries target Ubuntu 22.04, but JetPack 5.1.1 is 20.04 -- "
               "install ROS 2 into the conda env from conda-forge/RoboStack "
               "instead of apt (this is the same approach NVIDIA's own "
               "install_scripts/install_ros.sh takes).")
        return
    try:
        from std_msgs.msg import ByteMultiArray  # noqa: F401
        record(OK, "import std_msgs.msg.ByteMultiArray (goal/state transport)")
    except Exception as exc:
        record(FAIL, "import std_msgs failed", f"{type(exc).__name__}: {exc}")

    distro = os.environ.get("ROS_DISTRO")
    if distro:
        record(OK, f"ROS_DISTRO={distro}")
    else:
        record(WARN, "ROS_DISTRO is not set",
               "Usually fine inside a RoboStack conda env, but if rclpy imported "
               "from somewhere unexpected, check which install you actually got.")

    rmw = os.environ.get("RMW_IMPLEMENTATION")
    record(OK if rmw else WARN,
           f"RMW_IMPLEMENTATION={rmw or '(unset -> default)'}",
           "" if rmw else "Only matters if the WBC and this adapter end up on "
                          "different RMW implementations -- they must match to see "
                          "each other.")


def check_decoupled_wbc(repo: Path):
    if not repo.exists():
        record(FAIL, f"repo not found at {repo}",
               "Clone NVlabs/GR00T-WholeBodyControl (with git-lfs installed and "
               "pulled -- the policy ONNX files are LFS objects).")
        return
    record(OK, f"repo present at {repo}")

    loop = repo / "decoupled_wbc/control/main/teleop/run_g1_control_loop.py"
    record(OK if loop.exists() else FAIL,
           f"run_g1_control_loop.py {'found' if loop.exists() else 'MISSING'}",
           "" if loop.exists() else f"expected at {loop}")

    policy_dir = repo / "decoupled_wbc/sim2mujoco/resources/robots/g1/policy"
    for name in ("GR00T-WholeBodyControl-Balance.onnx", "GR00T-WholeBodyControl-Walk.onnx"):
        f = policy_dir / name
        if not f.exists():
            record(FAIL, f"{name} missing", f"expected at {f}")
            continue
        size = f.stat().st_size
        head = f.open("rb").read(64)
        if head.startswith(b"version https://git-lfs"):
            record(FAIL, f"{name} is an unpulled git-lfs pointer",
                   "Run `git lfs install && git lfs pull` in the repo. A pointer "
                   "file is a few hundred bytes of text, not a model.")
        elif size < 500_000:
            record(WARN, f"{name} is only {size} bytes", "suspiciously small")
        else:
            record(OK, f"{name} present ({size / 1e6:.1f} MB)")

    try:
        importlib.import_module("decoupled_wbc")
        record(OK, "import decoupled_wbc (package installed)")
    except Exception as exc:
        record(FAIL, "import decoupled_wbc failed",
               f"{type(exc).__name__}: {exc}. Run `pip install -e decoupled_wbc/` "
               "from the repo root, inside the conda env.")


def check_interface(iface: str | None):
    if not iface:
        record(WARN, "no --interface given",
               "run_g1_control_loop.py defaults to interface='sim'. For the real "
               "robot you MUST pass the actual DDS network interface, or it will "
               "quietly come up in simulation mode instead of talking to motors.")
        return
    ifaces = os.listdir("/sys/class/net") if Path("/sys/class/net").exists() else []
    if iface in ifaces:
        record(OK, f"network interface {iface!r} exists", f"available: {', '.join(ifaces)}")
    else:
        record(FAIL, f"network interface {iface!r} not found",
               f"available: {', '.join(ifaces) or '(none)'}")


def check_conflicts():
    """Whoever owns rt/lowcmd wins. Two controllers = a fight over the motors."""
    try:
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:
        record(WARN, "could not list processes"); return

    patterns = {
        "g1_policy_bridge.py": "raw rt/lowcmd bridge (alternative style)",
        "bridge.py --team": "g1_bridge (raw rt/lowcmd takeover)",
        "g1_deploy_onnx_ref": "gear_sonic_deploy (owns rt/lowcmd at 50 Hz)",
        "run_g1_control_loop": "an existing Decoupled WBC instance",
    }
    hits = [(pat, why) for pat, why in patterns.items()
            if any(pat in ln for ln in out.splitlines())]
    if hits:
        for pat, why in hits:
            record(FAIL, f"conflicting controller running: {pat}",
                   f"{why}. Only ONE thing may own rt/lowcmd. Stop it before "
                   "starting the WBC, or they will fight over the motors.")
    else:
        record(OK, "no conflicting low-level controller detected")


def check_action_socket(host: str, port: int):
    """The team's client BINDS :5556; the WBC side subscribes."""
    s = socket.socket()
    s.settimeout(1.0)
    try:
        s.connect((host, port))
        record(OK, f"team action socket {host}:{port} is bound",
               "their Orin client is up and publishing")
    except Exception:
        record(WARN, f"nothing bound on {host}:{port} yet",
               "Expected until the team's Orin container is started. The WBC "
               "and adapter can start first -- the adapter retries.")
    finally:
        s.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default="~/GR00T-WholeBodyControl")
    p.add_argument("--interface", default=None,
                   help="the real DDS network interface, e.g. eth0")
    p.add_argument("--actions-host", default="127.0.0.1")
    p.add_argument("--actions-port", type=int, default=5556)
    args = p.parse_args()

    print("=" * 68)
    print("Decoupled WBC preflight -- run this ON THE ORIN, in the WBC conda env")
    print("=" * 68)

    print("\n-- interpreter & packages --")
    check_python()
    check_imports()
    print("\n-- ROS 2 --")
    check_ros2()
    print(f"\n-- repo & policy weights --")
    check_decoupled_wbc(Path(os.path.expanduser(args.repo)))
    print("\n-- robot interface --")
    check_interface(args.interface)
    print("\n-- exclusivity (who owns rt/lowcmd) --")
    check_conflicts()
    print("\n-- team submission --")
    check_action_socket(args.actions_host, args.actions_port)

    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    print("\n" + "=" * 68)
    print(f"{len(_results)} checks: {len(_results) - len(fails) - len(warns)} pass, "
          f"{len(warns)} warn, {len(fails)} FAIL")
    if fails:
        print("\nBlocking:")
        for _, name, _d in fails:
            print(f"  - {name}")
        print("\nDo not start the WBC until these are resolved.")
    print("=" * 68)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
