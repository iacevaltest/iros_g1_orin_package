"""Real camera + real state publisher for the G1 -- drop-in replacement for
mocks/mock_orin.py. Speaks the exact same wire protocol (boundary/cameras.py,
boundary/states.py), so components/client.py and boundary/ need zero changes.

Cameras :5555  -- msgpack {"timestamps": {...}, "images": {key: bgr_jpeg}}
State   :5557  -- b"g1_debug" + msgpack {body_q, base_quat}

Read-only on the robot side (rt/lowstate subscribe only). Never publishes to
rt/arm_sdk or anything that could move the robot -- this script only feeds
observations upstream to the Thor policy.
"""
from __future__ import annotations

import ast
import os
import sys
import threading
import time

import cv2
import msgpack
import numpy as np
import zmq

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

try:
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False

# 2026-08-24: HEAD_DEVICE used to be a hardcoded /dev/videoN path, but that
# path has now drifted THREE times in one session (video12 -> video0 ->
# video16) just from normal USB re-enumeration after a power cycle -- V4L2
# device numbers are not stable identities. Resolve by name instead, at
# every startup, via /sys/class/video4linux/videoN/name (no extra deps).
# The head camera enumerates as v4l2 device "USB Camera: USB Camera", but
# note that name is NOT unique -- this specific unit exposes TWO nodes under
# that same name (e.g. video16 AND video17), one real capture node and one
# metadata/still node with no actual formats. Disambiguate by trying to
# actually open+read a frame from each candidate and keeping the first one
# that produces a real, wide frame -- same probe-and-verify pattern already
# used below for confirming the negotiated resolution.
HEAD_DEVICE_NAME = os.environ.get("HEAD_DEVICE_NAME", "USB Camera")


def _find_head_device() -> str | None:
    override = os.environ.get("HEAD_DEVICE")
    if override:
        return override
    import glob
    candidates = []
    for path in sorted(glob.glob("/dev/video*"), key=lambda p: int(p[len("/dev/video"):])):
        name_path = f"/sys/class/video4linux/{path.split('/')[-1]}/name"
        try:
            name = open(name_path).read().strip()
        except OSError:
            continue
        if HEAD_DEVICE_NAME in name:
            candidates.append(path)
    for path in candidates:
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok and frame is not None and frame.shape[1] >= 1280:
            print(f"[real_orin] head camera resolved to {path} (name={HEAD_DEVICE_NAME!r}, "
                  f"probe frame {frame.shape[1]}x{frame.shape[0]}) -- {len(candidates)} "
                  f"candidate node(s) shared this name", flush=True)
            return path
    if candidates:
        print(f"[real_orin] WARNING: found {len(candidates)} node(s) named "
              f"{HEAD_DEVICE_NAME!r} ({candidates}) but none produced a real capture "
              f"frame", file=sys.stderr)
    return None


# 2026-08-24: HEAD_WIDTH/HEIGHT had drifted to 640x480, a mode this camera's
# own `v4l2-ctl --list-formats-ext` does NOT list (only 3840x1080, 2160x1080,
# 3040x1520, 3840x1520 are discrete MJPG sizes) -- V4L2 was silently
# substituting its own nearest mode instead of the documented one. Restored
# to 3840x1080, the exact resolution EVALUATION_PROTOCOL.md's own "ego_view
# crop convention" section documents and freezes: left half of the 3840x1080
# stereo frame is 1920x1080 (16:9), then resized (not cropped) to 640x480
# (4:3). That resize is now applied per-eye below, independently, so
# ego_view_left/ego_view_right get the same treatment as the legacy mono
# ego_view rather than a from-scratch convention.
HEAD_WIDTH = 3840
HEAD_HEIGHT = 1080
# Wrist RealSense serials are PER-ROBOT. Set these for your rig; there is no
# sensible default. Find them with: tools/diagnose_cameras.py
WRIST_SERIALS = {
    "left_wrist":  os.environ.get("LEFT_WRIST_SERIAL", ""),
    "right_wrist": os.environ.get("RIGHT_WRIST_SERIAL", ""),
}

FRAME_SHAPE = (480, 640)  # (h, w) -- matches boundary's FRAME_SHAPE

# Eye selection confirmed via televuer.py's own left/right display code (left-half
# of frame = left eye). Rectification confirmed OFF via xr_teleoperate's actual
# recording code (teleop_hand_and_arm.py): colors["color_0"]/["color_1"] are raw
# head_img.bgr[:, :w//2] / [:, w//2:] slices -- no undistort/remap call anywhere.
# The training data used the raw, distorted, unrectified half-frame directly.
# Still switchable via env var in case this needs revisiting empirically.
EGO_VIEW_EYE = os.environ.get("EGO_VIEW_EYE", "left")          # "left" or "right"
EGO_VIEW_RECTIFY = os.environ.get("EGO_VIEW_RECTIFY", "0") == "1"

# 2026-08-24: some teams' checkpoints declare
# camera_keys requiring BOTH ego_view_left and ego_view_right -- the mono
# EGO_VIEW_EYE-selected `ego_view` key alone leaves those two permanently
# absent, which boundary/cameras.py tolerates by holding a black placeholder
# forever (see that module's docstring). A black placeholder is not "no
# camera" from the policy's point of view -- it is confident, wrong visual
# input. Publish both eyes unconditionally alongside the legacy mono
# `ego_view` key (kept for teams that only ever declared
# `ego_view`), rather than gating stereo behind a flag that could be left
# off by accident before a scored attempt.
PUBLISH_STEREO = os.environ.get("PUBLISH_STEREO", "1") != "0"

# Stereo calibration for the head camera, loaded from config/.
# PER-RIG DATA -- see config/head_camera_calibration.yaml. Only consulted when
# EGO_VIEW_RECTIFY=1; the default is rectification OFF, matching the raw frames
# the reference training data was collected on.
_CALIB_PATH = os.environ.get(
    "HEAD_CAMERA_CALIBRATION",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "config", "head_camera_calibration.yaml"))


def _load_calibration(path):
    """Minimal loader -- avoids a PyYAML dependency in the camera env."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, _, rest = line.partition(":")
            out[key.strip()] = ast.literal_eval(rest.strip())
    return out


_CALIB = _load_calibration(_CALIB_PATH)
_CAM_MATRIX_LEFT = np.array(_CALIB["cam_matrix_left"])
_CAM_MATRIX_RIGHT = np.array(_CALIB["cam_matrix_right"])
_DIST_LEFT = np.array(_CALIB["dist_left"])
_DIST_RIGHT = np.array(_CALIB["dist_right"])
_R1 = np.array(_CALIB["r1"])
_R2 = np.array(_CALIB["r2"])
_P1 = np.array(_CALIB["p1"])
_P2 = np.array(_CALIB["p2"])

_RECTIFY_SIZE = (FRAME_SHAPE[1], FRAME_SHAPE[0])  # (w, h) = (640, 480)

_map1_left, _map2_left = cv2.initUndistortRectifyMap(
    _CAM_MATRIX_LEFT, _DIST_LEFT, _R1, _P1, _RECTIFY_SIZE, cv2.CV_16SC2
)
_map1_right, _map2_right = cv2.initUndistortRectifyMap(
    _CAM_MATRIX_RIGHT, _DIST_RIGHT, _R2, _P2, _RECTIFY_SIZE, cv2.CV_16SC2
)

_latest_lock = threading.Lock()
_latest_frames: dict[str, np.ndarray] = {}   # key -> BGR uint8 (480,640,3)
_latest_state = {"body_q": None, "base_quat": None}


def _set_frame(key, bgr):
    with _latest_lock:
        _latest_frames[key] = bgr


def _prep_eye(raw_eye, map1, map2):
    """Resize (not crop) to FRAME_SHAPE, then optionally rectify -- same
    treatment as the legacy mono ego_view path, applied per-eye."""
    if raw_eye.shape[1] != FRAME_SHAPE[1] or raw_eye.shape[0] != FRAME_SHAPE[0]:
        raw_eye = cv2.resize(raw_eye, (FRAME_SHAPE[1], FRAME_SHAPE[0]))
    if EGO_VIEW_RECTIFY:
        return cv2.remap(raw_eye, map1, map2, cv2.INTER_LINEAR)
    return raw_eye


def head_camera_thread():
    head_device = _find_head_device()
    if head_device is None:
        print(f"[real_orin] WARNING: no head camera found (looked for a v4l2 device "
              f"named {HEAD_DEVICE_NAME!r}) -- ego_view* keys will never publish", file=sys.stderr)
        return

    cap = cv2.VideoCapture(head_device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, HEAD_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEAD_HEIGHT)
    if not cap.isOpened():
        print(f"[real_orin] WARNING: could not open head camera {head_device}", file=sys.stderr)
        return

    # Confirm what we actually got -- V4L2 backends sometimes silently substitute
    # the nearest supported mode instead of erroring on an unsupported request.
    ok, probe = cap.read()
    if not ok:
        print("[real_orin] WARNING: head camera opened but produced no frame", file=sys.stderr)
        return
    actual_h, actual_w = probe.shape[:2]
    if (actual_w, actual_h) != (HEAD_WIDTH, HEAD_HEIGHT):
        print(f"[real_orin] WARNING: requested {HEAD_WIDTH}x{HEAD_HEIGHT} but camera gave "
              f"{actual_w}x{actual_h} -- eye split below assumes a side-by-side stereo "
              f"frame and will be wrong if this mode isn't genuinely binocular", file=sys.stderr)
    print(f"[real_orin] head camera live at {actual_w}x{actual_h}, "
          f"eye={EGO_VIEW_EYE} rectify={EGO_VIEW_RECTIFY} stereo_publish={PUBLISH_STEREO}")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        half_w = frame.shape[1] // 2
        raw_left = frame[:, :half_w]
        raw_right = frame[:, half_w:]

        mono_raw = raw_left if EGO_VIEW_EYE == "left" else raw_right
        mono_map1, mono_map2 = (_map1_left, _map2_left) if EGO_VIEW_EYE == "left" \
            else (_map1_right, _map2_right)
        _set_frame("ego_view", _prep_eye(mono_raw, mono_map1, mono_map2))

        if PUBLISH_STEREO:
            _set_frame("ego_view_left", _prep_eye(raw_left, _map1_left, _map2_left))
            _set_frame("ego_view_right", _prep_eye(raw_right, _map1_right, _map2_right))


def wrist_camera_thread(key: str, serial: str):
    if not HAVE_REALSENSE:
        print(f"[real_orin] WARNING: pyrealsense2 not available, skipping {key}", file=sys.stderr)
        return
    pipeline = rs.pipeline()
    config = rs.config()
    try:
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, FRAME_SHAPE[1], FRAME_SHAPE[0], rs.format.bgr8, 30)
        pipeline.start(config)
    except Exception as exc:
        print(f"[real_orin] WARNING: could not start RealSense {serial} for {key}: {exc}", file=sys.stderr)
        return
    print(f"[real_orin] {key} (RealSense {serial}) live")
    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
        except Exception:
            continue
        color = frames.get_color_frame()
        if not color:
            continue
        img = np.asanyarray(color.get_data())  # already BGR (format.bgr8)
        _set_frame(key, img)


def state_thread(iface):
    def handler(msg: LowState_):
        with _latest_lock:
            _latest_state["body_q"] = [msg.motor_state[i].q for i in range(29)]
            _latest_state["base_quat"] = list(msg.imu_state.quaternion)

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(handler, 10)
    print("[real_orin] state (rt/lowstate) subscribed")


def camera_publish_loop(port: int, fps: float):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{port}")
    interval = 1.0 / fps
    print(f"[real_orin] publishing cameras on :{port}")
    while True:
        t0 = time.time()
        with _latest_lock:
            frames = dict(_latest_frames)
        if "ego_view" in frames:
            images = {}
            timestamps = {}
            now = time.time()
            for key, bgr in frames.items():
                ok, jpeg = cv2.imencode(".jpg", bgr)
                if ok:
                    images[key] = jpeg.tobytes()
                    timestamps[key] = now
            sock.send(msgpack.packb({"timestamps": timestamps, "images": images}, use_bin_type=True))
        elapsed = time.time() - t0
        time.sleep(max(0.0, interval - elapsed))


def state_publish_loop(port: int, hz: float):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{port}")
    interval = 1.0 / hz
    print(f"[real_orin] publishing state on :{port}")
    prefix = b"g1_debug"
    while True:
        t0 = time.time()
        with _latest_lock:
            body_q = _latest_state["body_q"]
            base_quat = _latest_state["base_quat"]
        if body_q is not None:
            payload = msgpack.packb({"body_q": body_q, "base_quat": base_quat}, use_bin_type=True)
            sock.send(prefix + payload)
        elapsed = time.time() - t0
        time.sleep(max(0.0, interval - elapsed))


if __name__ == "__main__":
    iface = sys.argv[1] if len(sys.argv) > 1 else None
    if iface:
        ChannelFactoryInitialize(0, iface)
    else:
        ChannelFactoryInitialize(0)

    state_thread(iface)
    threading.Thread(target=head_camera_thread, daemon=True).start()
    for key, serial in WRIST_SERIALS.items():
        threading.Thread(target=wrist_camera_thread, args=(key, serial), daemon=True).start()
    threading.Thread(target=camera_publish_loop, args=(5555, 30.0), daemon=True).start()

    # state loop runs in main thread
    state_publish_loop(5557, 50.0)
