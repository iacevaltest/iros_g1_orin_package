#!/usr/bin/env python3
"""Gate the sensor stream BEFORE scoring an attempt.

Run on PC2, in the venv running real_orin.py, once real_orin.py is up and
before the team's client container starts mattering:

    python3 preflight_sensors.py --require-wrists

Why this exists: `real_orin.py` degrades SILENTLY. If a RealSense fails to
open, its thread prints a warning and returns; the camera publisher keeps
publishing whatever keys it does have. A team whose policy declares
`camera_keys: [ego_view, left_wrist, right_wrist]` -- some policies do -- then
has its own server substitute the head frame for the missing wrists
(`server.py`: `if self._last_left_wrist is None: self._last_left_wrist =
ego.copy()`). The policy receives the head camera in all three slots and
behaves badly, and that is an EVALUATOR fault, not the team's.

Note a checkpoint consuming only `ego_view` will not hit this failure mode
did not exist for them. Do not carry that assumption forward.

Exit 0 = the stream is fit to score against. Read-only; publishes nothing.
"""
from __future__ import annotations

import argparse
import sys
import time

import msgpack
import numpy as np

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
_results = []


def record(status, name, detail=""):
    _results.append((status, name, detail))
    mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
    print(f"[{mark}] {name}" + (f"\n           {detail}" if detail else ""))


def collect_camera(host, port, seconds):
    import zmq
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.SUB)
    s.setsockopt_string(zmq.SUBSCRIBE, "")
    s.setsockopt(zmq.RCVTIMEO, 2000)
    s.connect("tcp://{}:{}".format(host, port))
    frames = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        try:
            frames.append(msgpack.unpackb(s.recv(), raw=False))
        except Exception:
            break
    s.close(linger=0)
    return frames


def collect_state(host, port, seconds):
    import zmq
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
    s.setsockopt(zmq.RCVTIMEO, 2000)
    s.connect("tcp://{}:{}".format(host, port))
    msgs = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        try:
            blob = s.recv()
            msgs.append((time.time(), msgpack.unpackb(blob[len(b"g1_debug"):], raw=False)))
        except Exception:
            break
    s.close(linger=0)
    return msgs


def check_cameras(frames, require_wrists, require_stereo=False, save_dir=None):
    import cv2
    if not frames:
        record(FAIL, "no camera messages on :5555",
               "is real_orin.py running? the team's client will block forever "
               "on 'waiting for the organizer's endpoints'")
        return
    record(OK, "camera stream live ({} msgs)".format(len(frames)))

    keys = set()
    for f in frames:
        keys |= set(f.get("images", {}).keys())
    record(OK, "camera keys present: {}".format(sorted(keys)))

    needed = ["ego_view", "left_wrist", "right_wrist"]
    for k in needed:
        if k in keys:
            record(OK, "{} present".format(k))
        elif k == "ego_view":
            record(FAIL, "ego_view MISSING", "boundary requires it; nothing works without it")
        elif require_wrists:
            record(FAIL, "{} MISSING".format(k),
                   "A policy declaring this key will silently get the HEAD frame "
                   "substituted for it. Any result scored in this state is an "
                   "evaluator fault -> no-contest. Fix the RealSense before scoring.")
        else:
            record(WARN, "{} missing (--require-wrists not set)".format(k))

    # Some policies declare camera_keys requiring
    # BOTH ego_view_left and ego_view_right instead of (or in addition to)
    # mono ego_view. boundary/cameras.py holds a black placeholder forever
    # for a missing key rather than erroring -- a policy fed permanent black
    # stereo input can still run and produce output, so this failure mode is
    # invisible unless checked for explicitly. --require-stereo turns a
    # missing stereo key into a hard FAIL, same treatment as --require-wrists.
    stereo_keys = ["ego_view_left", "ego_view_right"]
    for k in stereo_keys:
        if k in keys:
            record(OK, "{} present".format(k))
        elif require_stereo:
            record(FAIL, "{} MISSING".format(k),
                   "A policy declaring this key gets a black placeholder held "
                   "forever (boundary/cameras.py's own tolerant-missing-key "
                   "behavior) -- confident, wrong visual input, not 'no camera'. "
                   "Any result scored in this state is an evaluator fault -> "
                   "no-contest. Fix real_orin.py's stereo publish before scoring.")
        else:
            record(WARN, "{} missing (--require-stereo not set)".format(k))

    # decode + shape + distinctness
    last = frames[-1]["images"]
    decoded = {}
    for k, jpg in last.items():
        arr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            record(FAIL, "{}: JPEG failed to decode".format(k)); continue
        decoded[k] = arr
        if save_dir:
            import os
            path = os.path.join(save_dir, "{}.jpg".format(k))
            cv2.imwrite(path, arr)
            record(OK, "  saved {}".format(path))
        if arr.shape != (480, 640, 3):
            record(FAIL, "{}: shape {}, expected (480, 640, 3)".format(k, arr.shape))
        else:
            record(OK, "{}: decodes to (480, 640, 3), mean brightness {:.1f}".format(
                k, float(arr.mean())))

    # Catch the head-frame-substituted-for-wrist case directly.
    if "ego_view" in decoded:
        for k in ("left_wrist", "right_wrist"):
            if k in decoded and decoded[k].shape == decoded["ego_view"].shape:
                if np.array_equal(decoded[k], decoded["ego_view"]):
                    record(FAIL, "{} is byte-identical to ego_view".format(k),
                           "the head frame is being substituted for a wrist camera")
    if "left_wrist" in decoded and "right_wrist" in decoded:
        if decoded["left_wrist"].shape == decoded["right_wrist"].shape and \
           np.array_equal(decoded["left_wrist"], decoded["right_wrist"]):
            record(FAIL, "left_wrist and right_wrist are byte-identical",
                   "both keys are being fed from one camera")

    # Same distinctness checks for the stereo pair: byte-identical eyes means
    # real_orin.py is publishing the same frame under both keys (a copy-paste
    # bug), not genuine binocular capture.
    if "ego_view_left" in decoded and "ego_view_right" in decoded:
        if decoded["ego_view_left"].shape == decoded["ego_view_right"].shape and \
           np.array_equal(decoded["ego_view_left"], decoded["ego_view_right"]):
            record(FAIL, "ego_view_left and ego_view_right are byte-identical",
                   "both keys are being fed from one eye, not genuine stereo")

    # Frozen-frame check across the capture window.
    if len(frames) >= 2:
        for k in sorted(keys):
            a = frames[0]["images"].get(k)
            b = frames[-1]["images"].get(k)
            if a is not None and b is not None and a == b:
                record(WARN, "{}: first and last frame identical".format(k),
                       "camera may be frozen (or the scene is perfectly static)")


def check_state(msgs):
    if not msgs:
        record(FAIL, "no state messages on :5557",
               "is real_orin.py running, and is rt/lowstate alive?")
        return
    record(OK, "state stream live ({} msgs)".format(len(msgs)))

    ts = [t for t, _ in msgs]
    if len(ts) > 1:
        rate = (len(ts) - 1) / max(ts[-1] - ts[0], 1e-6)
        record(OK if rate > 20 else WARN, "state rate ~{:.0f} Hz".format(rate),
               "" if rate > 20 else "real_orin.py targets 50 Hz")

    _, last = msgs[-1]
    body_q = np.asarray(last.get("body_q", []), dtype=float)
    if body_q.shape != (29,):
        record(FAIL, "body_q shape {}, expected (29,)".format(body_q.shape)); return
    if not np.isfinite(body_q).all():
        record(FAIL, "body_q contains NaN/Inf"); return
    record(OK, "body_q (29,) finite, range [{:.3f}, {:.3f}]".format(body_q.min(), body_q.max()))

    bq = np.asarray(last.get("base_quat", []), dtype=float)
    if bq.shape != (4,):
        record(FAIL, "base_quat shape {}, expected (4,)".format(bq.shape))
    else:
        n = float(np.linalg.norm(bq))
        if abs(n - 1.0) > 1e-2:
            record(FAIL, "base_quat norm {:.4f}, not unit".format(n))
        else:
            record(OK, "base_quat unit (norm {:.4f}), w-first per "
                       "imu_state.quaternion = w,x,y,z".format(n))

    # STALE STATE: real_orin.py republishes its last cached value forever if
    # rt/lowstate dies. Frozen proprioception is invisible downstream -- the
    # policy and our IK both keep running on a pose that stopped being true.
    if len(msgs) > 10:
        first_q = np.asarray(msgs[0][1]["body_q"], dtype=float)
        if np.array_equal(first_q, body_q):
            record(WARN, "body_q identical across the whole window",
                   "either the robot is perfectly still, or rt/lowstate has "
                   "stopped and real_orin.py is republishing a cached value. "
                   "Move an arm by hand and re-run to tell the difference.")
        else:
            record(OK, "body_q is changing (proprioception is live, not cached)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--state-port", type=int, default=5557)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--require-wrists", action="store_true",
                   help="REQUIRED for any team whose server declares wrist "
                        "camera_keys. Turns a missing wrist "
                        "camera into a hard FAIL instead of a warning.")
    p.add_argument("--require-stereo", action="store_true",
                   help="REQUIRED for any team whose server declares "
                        "ego_view_left/ego_view_right camera_keys "
                        "Turns a missing "
                        "stereo key into a hard FAIL instead of a warning.")
    p.add_argument("--save-frames", default=None, metavar="DIR",
                   help="also write the actual decoded ego_view/left_wrist/"
                        "right_wrist JPEGs to this dir, for visual inspection "
                        "-- pass/fail on shape and brightness cannot catch a "
                        "bad crop; looking at the pixels can.")
    args = p.parse_args()

    if args.save_frames:
        import os
        os.makedirs(args.save_frames, exist_ok=True)

    print("=" * 68)
    print("Sensor preflight -- run in the venv running real_orin.py, on PC2")
    print("=" * 68)
    print("\n-- cameras :{} --".format(args.camera_port))
    check_cameras(collect_camera(args.host, args.camera_port, args.seconds),
                  args.require_wrists, require_stereo=args.require_stereo,
                  save_dir=args.save_frames)
    print("\n-- state :{} --".format(args.state_port))
    check_state(collect_state(args.host, args.state_port, args.seconds))

    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    print("\n" + "=" * 68)
    print("{} checks: {} pass, {} warn, {} FAIL".format(
        len(_results), len(_results) - len(fails) - len(warns), len(warns), len(fails)))
    if fails:
        print("\nBlocking -- do not score an attempt against this stream:")
        for _, name, _d in fails:
            print("  - {}".format(name))
    print("=" * 68)
    print("\nStill do the wrist cover test by hand: cover the physical LEFT")
    print("wrist camera and confirm left_wrist's mean brightness drops (not")
    print("right_wrist's). WRIST_SERIALS was mapped backwards once already,")
    print("and a silent L/R swap degrades the policy without erroring.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
