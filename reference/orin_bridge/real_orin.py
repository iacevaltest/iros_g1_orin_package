"""Real camera + real state publisher for the G1 -- drop-in replacement for
mocks/mock_orin.py. Speaks the exact same wire protocol (boundary/cameras.py,
boundary/states.py), so components/client.py and boundary/ need zero changes.

Cameras :5555  -- msgpack {"timestamps": {...}, "images": {key: bgr_jpeg}}
State   :5557  -- b"g1_debug" + msgpack {body_q, base_quat}

Read-only on the robot side (rt/lowstate subscribe only). Never publishes to
rt/arm_sdk or anything that could move the robot -- this script only feeds
observations upstream to the Thor policy.

NOTE: the documented launch path (docs/RUNBOOK.md, Step 1) is the split pair
real_orin_cameras.py (teleimager env) + real_orin_state.py (g1_wbc env); this
combined script is kept for rigs where one env has both dependency sets. Its
head-camera path is kept identical to real_orin_cameras.py -- see that file's
docstring for the head-camera geometry (native 1280x480 side-by-side, each
half a 640x480 eye, no resize, matching the training dataset) and for why the
3840x1080 per-eye resize used from 2026-08-24 is gone.
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
_TAG = "[real_orin]"
HEAD_DEVICE_NAME = os.environ.get("HEAD_DEVICE_NAME", "USB Camera")


def _find_head_device() -> str | None:
    """Resolve the head camera by V4L2 name, then confirm each candidate by
    actually capturing one frame in the native mode.

    A node is accepted only if it returns a frame EXACTLY HEAD_WIDTH wide. The
    2026-08-24 incident probed a RealSense node under the wrong name and got
    848x480 / 640x480 frames; a ">= 1280" test or a "whatever it gives" test
    would have accepted such a node, this one rejects it.
    """
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
    rejected = []
    for path in candidates:
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, HEAD_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEAD_HEIGHT)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok and frame is not None and probe_frame_is_head_camera(frame, HEAD_WIDTH):
            print(f"{_TAG} head camera resolved to {path} (name={HEAD_DEVICE_NAME!r}, "
                  f"probe frame {frame.shape[1]}x{frame.shape[0]}) -- {len(candidates)} "
                  f"candidate node(s) shared this name", flush=True)
            return path
        if ok and frame is not None:
            rejected.append(f"{path}={frame.shape[1]}x{frame.shape[0]}")
    if candidates:
        print(f"{_TAG} WARNING: found {len(candidates)} node(s) named "
              f"{HEAD_DEVICE_NAME!r} ({candidates}) but none produced a "
              f"{HEAD_WIDTH}-wide capture frame"
              + (f" (rejected: {', '.join(rejected)})" if rejected else ""),
              file=sys.stderr)
    return None


def probe_frame_is_head_camera(frame: np.ndarray, expected_width: int = 1280) -> bool:
    """True iff a probe frame is exactly `expected_width` wide.

    Rejects the RealSense-sized frames (848x480, 640x480) that the wrong node
    produced on 2026-08-24, and anything else that is not the requested mode.
    """
    return frame is not None and frame.ndim >= 2 and frame.shape[1] == expected_width


# Head camera geometry -- see real_orin_cameras.py's docstring. The head camera
# is opened in its NATIVE 1280x480 side-by-side MJPG mode; each half IS a
# 640x480 eye, no resize, no crop, exactly how the training dataset
# (HF BitRobot/G1_WBT_Dex1_Building-Children-Table, 480x640x3 per cam key) was
# captured -- verified on raw MCAP frames (77/77 head frames 1280x480, halves
# differ by a horizontal-only disparity). The calibration intrinsics
# (fx ~337, cx ~316, cy ~232) are those of a 640x480 eye.
#
# History: the native mode was used successfully on 2026-08-19/20. On
# 2026-08-24 a wrong /dev/video node (a RealSense, 848x480 / 640x480 frames)
# was probed, the session concluded that 1280x480 "was not a listed mode",
# and the bridge moved to 3840x1080 with each 1920x1080 eye cv2.resize'd to
# 640x480 -- a 16:9 -> 4:3 horizontal squash by 0.75 the dataset never had.
# That squash is gone. Anything other than 1280x480 is refused unless
# HEAD_ALLOW_RESIZE_FALLBACK=1 (diagnostics only).
DATASET_HEAD_SIZE = (1280, 480)   # (w, h) side-by-side stereo frame
HEAD_WIDTH = int(os.environ.get("HEAD_WIDTH", str(DATASET_HEAD_SIZE[0])))
HEAD_HEIGHT = int(os.environ.get("HEAD_HEIGHT", str(DATASET_HEAD_SIZE[1])))
HEAD_ALLOW_RESIZE_FALLBACK = os.environ.get("HEAD_ALLOW_RESIZE_FALLBACK", "0") == "1"
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

# The rectify maps are built for a 640x480 eye -- the intrinsics are 640x480
# intrinsics (cx ~316, cy ~232). They are therefore only geometrically valid
# in the native 1280x480 mode, where each raw half already is 640x480.
# Remapping a resized (squashed) eye with them would be silently wrong.
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


def split_stereo_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a side-by-side stereo frame at its middle column -- (left, right)
    views, no copy, no resize. In the native 1280x480 mode each half is a
    640x480 eye, exactly as the dataset's cam_left / cam_right were sliced."""
    half_w = frame.shape[1] // 2
    return frame[:, :half_w], frame[:, half_w:]


def head_geometry_mismatch(actual_size, expected_size=DATASET_HEAD_SIZE):
    """None if the negotiated (w, h) is the expected side-by-side geometry,
    otherwise a one-line reason."""
    aw, ah = int(actual_size[0]), int(actual_size[1])
    ew, eh = int(expected_size[0]), int(expected_size[1])
    if (aw, ah) == (ew, eh):
        return None
    return (f"negotiated {aw}x{ah}, expected {ew}x{eh}: each half would be "
            f"{aw // 2}x{ah}, not the dataset's {ew // 2}x{eh} eye")


def decide_head_geometry(actual_size, expected_size=DATASET_HEAD_SIZE,
                         allow_resize_fallback=False, tag=_TAG):
    """(mode, lines): "native" + banner / "fallback" + WARNING lines /
    "refuse" + ERROR lines. Same logic as real_orin_cameras.py."""
    aw, ah = int(actual_size[0]), int(actual_size[1])
    reason = head_geometry_mismatch(actual_size, expected_size)
    eye_w, eye_h = FRAME_SHAPE[1], FRAME_SHAPE[0]
    if reason is None:
        return "native", [
            f"{tag} head camera live at {aw}x{ah} (native side-by-side; each eye "
            f"{eye_w}x{eye_h}, no resize, matches dataset)",
        ]
    dataset_w, dataset_h = DATASET_HEAD_SIZE
    ew, eh = int(expected_size[0]), int(expected_size[1])
    body = [
        f"head camera negotiated {aw}x{ah} (requested {ew}x{eh}) -- this stream would NOT match "
        f"the dataset geometry.",
        f"  The training dataset (HF BitRobot/G1_WBT_Dex1_Building-Children-Table) was captured in the",
        f"  camera's native {dataset_w}x{dataset_h} side-by-side MJPG mode: each half IS a {eye_w}x{eye_h} eye,",
        f"  no resize, no crop. Splitting {aw}x{ah} gives {aw // 2}x{ah} halves; resizing those to",
        f"  {eye_w}x{eye_h} changes the aspect ratio and the field of view the policy was trained on.",
        f"  Check the /dev/video node (HEAD_DEVICE / HEAD_DEVICE_NAME -- a RealSense node returns",
        f"  848x480 or 640x480) and the camera's MJPG modes (v4l2-ctl --list-formats-ext).",
    ]
    if allow_resize_fallback:
        lines = [f"{tag} WARNING: HEAD_ALLOW_RESIZE_FALLBACK=1 -- " + body[0]]
        lines += [f"{tag} WARNING:{line}" for line in body[1:]]
        lines.append(f"{tag} WARNING: publishing head frames anyway with a per-eye cv2.resize to "
                     f"{eye_w}x{eye_h}. DIAGNOSTICS ONLY -- never for a scored run.")
        return "fallback", lines
    lines = [f"{tag} ERROR: " + body[0]]
    lines += [f"{tag} ERROR:{line}" for line in body[1:]]
    lines.append(f"{tag} ERROR: REFUSING to publish head frames -- ego_view* keys will not appear on :5555.")
    lines.append(f"{tag} ERROR: Fix the camera mode. Set HEAD_ALLOW_RESIZE_FALLBACK=1 only to force the old "
                 f"per-eye resize for diagnostics.")
    return "refuse", lines


_resize_logged = False


def prepare_eye(raw_eye, rectify_maps=None):
    """One raw half-frame -> published eye. In the native mode the half is
    already 640x480 and is returned unchanged. The resize guard is kept for
    the HEAD_ALLOW_RESIZE_FALLBACK path only and logs the first time it
    fires. `rectify_maps` is (map1, map2) built for 640x480, or None (raw,
    EGO_VIEW_RECTIFY=0, the dataset's own treatment)."""
    global _resize_logged
    if raw_eye.shape[1] != FRAME_SHAPE[1] or raw_eye.shape[0] != FRAME_SHAPE[0]:
        if not _resize_logged:
            _resize_logged = True
            print(f"{_TAG} WARNING: per-eye resize {raw_eye.shape[1]}x{raw_eye.shape[0]} -> "
                  f"{FRAME_SHAPE[1]}x{FRAME_SHAPE[0]} was needed -- this stream does NOT match "
                  f"the dataset geometry (logged once)", file=sys.stderr, flush=True)
        raw_eye = cv2.resize(raw_eye, (FRAME_SHAPE[1], FRAME_SHAPE[0]))
    if rectify_maps is not None:
        return cv2.remap(raw_eye, rectify_maps[0], rectify_maps[1], cv2.INTER_LINEAR)
    return raw_eye


def head_camera_thread():
    if (HEAD_WIDTH, HEAD_HEIGHT) != DATASET_HEAD_SIZE:
        print(f"{_TAG} WARNING: HEAD_WIDTH/HEAD_HEIGHT overridden to {HEAD_WIDTH}x{HEAD_HEIGHT}; the "
              f"dataset geometry is {DATASET_HEAD_SIZE[0]}x{DATASET_HEAD_SIZE[1]} -- diagnostics only",
              file=sys.stderr, flush=True)

    head_device = _find_head_device()
    if head_device is None:
        print(f"{_TAG} WARNING: no head camera found (looked for a v4l2 device "
              f"named {HEAD_DEVICE_NAME!r}) -- ego_view* keys will never publish", file=sys.stderr)
        return

    cap = cv2.VideoCapture(head_device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, HEAD_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEAD_HEIGHT)
    if not cap.isOpened():
        print(f"{_TAG} WARNING: could not open head camera {head_device}", file=sys.stderr)
        return

    # Confirm what we actually got -- V4L2 backends silently substitute the
    # nearest supported mode instead of erroring on an unsupported request.
    ok, probe = cap.read()
    if not ok:
        print(f"{_TAG} WARNING: head camera opened but produced no frame", file=sys.stderr)
        return
    actual_h, actual_w = probe.shape[:2]
    mode, lines = decide_head_geometry((actual_w, actual_h), (HEAD_WIDTH, HEAD_HEIGHT),
                                       HEAD_ALLOW_RESIZE_FALLBACK)
    for line in lines:
        print(line, file=sys.stderr if mode != "native" else sys.stdout, flush=True)
    if mode == "refuse":
        cap.release()
        return
    print(f"{_TAG} head camera settings: eye={EGO_VIEW_EYE} rectify={EGO_VIEW_RECTIFY} "
          f"stereo_publish={PUBLISH_STEREO}", flush=True)

    maps_left = (_map1_left, _map2_left) if EGO_VIEW_RECTIFY else None
    maps_right = (_map1_right, _map2_right) if EGO_VIEW_RECTIFY else None
    mono_is_left = EGO_VIEW_EYE == "left"

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        raw_left, raw_right = split_stereo_frame(frame)
        left = prepare_eye(raw_left, maps_left)
        right = prepare_eye(raw_right, maps_right)

        _set_frame("ego_view", left if mono_is_left else right)
        if PUBLISH_STEREO:
            _set_frame("ego_view_left", left)
            _set_frame("ego_view_right", right)


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
