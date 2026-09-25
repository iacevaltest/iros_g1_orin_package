"""Real camera publisher for the G1 -- camera-only half of real_orin.py, split
out to run in the teleimager conda env (has pyrealsense2/cv2) while the state
half runs separately in g1_wbc (has unitree_sdk2py/cyclonedds).

Cameras :5555 -- msgpack {"timestamps": {...}, "images": {key: bgr_jpeg}}

No robot SDK import at all -- pure V4L2/RealSense capture + ZMQ publish.

Head camera geometry
--------------------
The head camera is opened in its NATIVE 1280x480 side-by-side MJPG mode.
Each half of that frame IS a 640x480 eye: no resize, no crop, no
rectification. That is exactly how the training dataset
(HF BitRobot/G1_WBT_Dex1_Building-Children-Table, meta/info.json: every
observation.images.cam_* is 480x640x3 at 30 fps) was captured -- verified on
the raw MCAP frames: 77/77 head frames are 1280x480 and the two halves differ
by a horizontal-only disparity. The calibration intrinsics in
config/head_camera_calibration.yaml (fx ~337, cx ~316, cy ~232) are those of
a 640x480 eye, i.e. they belong to this mode.

History, so nobody re-derives the wrong answer: the native 1280x480 mode was
used successfully on 2026-08-19/20. On 2026-08-24 a wrong /dev/video node
(a RealSense, 848x480 / 640x480 frames) was probed, the session inferred that
1280x480 "was not a listed mode", and the bridge was moved to 3840x1080 with
each 1920x1080 eye cv2.resize'd to 640x480 -- a 16:9 -> 4:3 horizontal squash
by 0.75 that the dataset never had. That squash is gone as of this revision.
The bridge now refuses to publish head frames from any other geometry unless
HEAD_ALLOW_RESIZE_FALLBACK=1 is set explicitly (diagnostics only).
"""
from __future__ import annotations

import os
import sys
import threading
import time

import cv2
import msgpack
import numpy as np
import zmq

try:
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False

_TAG = "[real_orin_cameras]"

HEAD_DEVICE_NAME = os.environ.get("HEAD_DEVICE_NAME", "USB Camera")

# The dataset's head-camera geometry. This is a fact about the training data,
# not a tunable.
DATASET_HEAD_SIZE = (1280, 480)   # (w, h) side-by-side stereo frame
FRAME_SHAPE = (480, 640)          # (h, w) -- one eye; matches boundary's FRAME_SHAPE

# Requested capture size. Env-overridable for DIAGNOSTICS ONLY: anything other
# than 1280x480 does not match the dataset and the bridge says so at startup.
HEAD_WIDTH = int(os.environ.get("HEAD_WIDTH", str(DATASET_HEAD_SIZE[0])))
HEAD_HEIGHT = int(os.environ.get("HEAD_HEIGHT", str(DATASET_HEAD_SIZE[1])))

# Escape hatch for the geometry check in head_camera_thread. When the camera
# negotiates something other than HEAD_WIDTHxHEAD_HEIGHT the bridge REFUSES to
# publish head frames -- unless this is "1", in which case it falls back to the
# pre-2026-09 per-eye resize and warns on every startup line. Never set this
# for a scored run.
HEAD_ALLOW_RESIZE_FALLBACK = os.environ.get("HEAD_ALLOW_RESIZE_FALLBACK", "0") == "1"

# Wrist RealSense serials are PER-ROBOT. Set these for your rig; there is no
# sensible default. Find them with: tools/diagnose_cameras.py
WRIST_SERIALS = {
    "left_wrist":  os.environ.get("LEFT_WRIST_SERIAL", ""),
    "right_wrist": os.environ.get("RIGHT_WRIST_SERIAL", ""),
}

EGO_VIEW_EYE = os.environ.get("EGO_VIEW_EYE", "left")          # "left" or "right"
EGO_VIEW_RECTIFY = os.environ.get("EGO_VIEW_RECTIFY", "0") == "1"
PUBLISH_STEREO = os.environ.get("PUBLISH_STEREO", "1") != "0"


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


def probe_frame_is_head_camera(frame: np.ndarray, expected_width: int = DATASET_HEAD_SIZE[0]) -> bool:
    """True iff a probe frame is exactly `expected_width` wide.

    Rejects the RealSense-sized frames (848x480, 640x480) that the wrong node
    produced on 2026-08-24, and anything else that is not the requested mode.
    """
    return frame is not None and frame.ndim >= 2 and frame.shape[1] == expected_width


# Stereo calibration for THIS physical head camera (confirmed same unit used for the
# original 512-episode data collection). Values as given, not recomputed --
# R1/R2/P1/P2 are the calibration's own rectification outputs, used as-is so the
# rectified image matches whatever was actually used at data-collection time.
_CAM_MATRIX_LEFT = np.array([
    [337.5311318539417, 0.0, 316.5285046932812],
    [0.0, 336.61378142923456, 232.50620475777816],
    [0.0, 0.0, 1.0],
])
_CAM_MATRIX_RIGHT = np.array([
    [336.30012498108425, 0.0, 321.60051380995424],
    [0.0, 335.47329565297144, 231.69425545320323],
    [0.0, 0.0, 1.0],
])
_DIST_LEFT = np.array([0.06635329597971165, -0.07841619072258442,
                        -0.0032837567734969727, -0.0010816865229956933, 0.021030073866954904])
_DIST_RIGHT = np.array([0.06366431884731834, -0.08229830690155956,
                         -0.0031845859537499963, 0.0017675102141209843, 0.027381390668112876])
_R1 = np.array([
    [0.9992450760979893, -0.006580452854965362, 0.03828805994232576],
    [0.006547478029861057, 0.9999780783764134, 0.0009865586977590912],
    [-0.038293712608887094, -0.0007351236897390344, 0.9992662563940548],
])
_R2 = np.array([
    [0.9970325836991178, -0.0070000624646499926, 0.07666176470544075],
    [0.007066114951150908, 0.9999748604134095, -0.0005903902768392253],
    [-0.07665570469155238, 0.001130339184874579, 0.997056981958187],
])
_P1 = np.array([
    [362.7751607273391, 0.0, 282.4990463256836, 0.0],
    [0.0, 362.7751607273391, 229.0226879119873, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])
_P2 = np.array([
    [362.7751607273391, 0.0, 282.4990463256836, -21875.510224633603],
    [0.0, 362.7751607273391, 229.0226879119873, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])
# The rectify maps are built for a 640x480 eye -- the intrinsics above are
# 640x480 intrinsics (cx ~316, cy ~232). They are therefore only geometrically
# valid in the native 1280x480 mode, where each raw half already is 640x480.
# Remapping a resized (squashed) eye with them would be silently wrong.
_RECTIFY_SIZE = (FRAME_SHAPE[1], FRAME_SHAPE[0])  # (w, h) = (640, 480)

_map1_left, _map2_left = cv2.initUndistortRectifyMap(
    _CAM_MATRIX_LEFT, _DIST_LEFT, _R1, _P1, _RECTIFY_SIZE, cv2.CV_16SC2
)
_map1_right, _map2_right = cv2.initUndistortRectifyMap(
    _CAM_MATRIX_RIGHT, _DIST_RIGHT, _R2, _P2, _RECTIFY_SIZE, cv2.CV_16SC2
)
RECTIFY_MAPS = {
    "left": (_map1_left, _map2_left),
    "right": (_map1_right, _map2_right),
}

_latest_lock = threading.Lock()
_latest_frames: dict[str, np.ndarray] = {}   # key -> BGR uint8 (480,640,3)


def _set_frame(key, bgr):
    with _latest_lock:
        _latest_frames[key] = bgr


# ---------------------------------------------------------------------------
# Pure geometry helpers (no camera, no threads) -- unit-tested in
# tests/test_head_geometry.py.
# ---------------------------------------------------------------------------

def split_stereo_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a side-by-side stereo frame at its middle column.

    Returns (left, right) as views into `frame`: no copy, no resize. In the
    native 1280x480 mode each half is a 640x480 eye, exactly as the dataset's
    cam_left / cam_right were sliced at collection time.
    """
    half_w = frame.shape[1] // 2
    return frame[:, :half_w], frame[:, half_w:]


def head_geometry_mismatch(actual_size: tuple[int, int],
                           expected_size: tuple[int, int] = DATASET_HEAD_SIZE) -> str | None:
    """None if the negotiated (w, h) is the expected side-by-side geometry,
    otherwise a one-line reason. 3840x1080 -> mismatch (16:9 eyes, would need
    a 0.75 horizontal squash to reach 640x480); 848x480 / 640x480 -> mismatch
    (RealSense-sized, not the head camera)."""
    aw, ah = int(actual_size[0]), int(actual_size[1])
    ew, eh = int(expected_size[0]), int(expected_size[1])
    if (aw, ah) == (ew, eh):
        return None
    eye_w, eye_h = aw // 2, ah
    return (f"negotiated {aw}x{ah}, expected {ew}x{eh}: each half would be "
            f"{eye_w}x{eye_h}, not the dataset's {ew // 2}x{eh} eye")


def decide_head_geometry(actual_size: tuple[int, int],
                         expected_size: tuple[int, int] = DATASET_HEAD_SIZE,
                         allow_resize_fallback: bool = False,
                         tag: str = _TAG) -> tuple[str, list[str]]:
    """Decide what head_camera_thread does with a negotiated frame size.

    Returns (mode, lines) where mode is one of
      "native"   -- geometry matches; lines is the startup banner
      "fallback" -- mismatch but HEAD_ALLOW_RESIZE_FALLBACK=1; publish with the
                    old per-eye resize; every line carries WARNING
      "refuse"   -- mismatch; do NOT publish head frames; lines is the error
    """
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


def prepare_eye(raw_eye: np.ndarray,
                rectify_maps: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Turn one raw half-frame into a published eye.

    In the native 1280x480 mode the half is already 640x480 and this returns
    it unchanged (same buffer, no resize). The resize guard is kept for the
    HEAD_ALLOW_RESIZE_FALLBACK path only, and logs the first time it fires so
    a silent squash can never come back unnoticed.

    `rectify_maps` is (map1, map2) from cv2.initUndistortRectifyMap built for
    640x480, or None for raw output (EGO_VIEW_RECTIFY=0, the dataset's own
    treatment).
    """
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


def resize_was_needed() -> bool:
    """True once prepare_eye has ever had to resize (test / diagnostics hook)."""
    return _resize_logged


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

    maps_left = RECTIFY_MAPS["left"] if EGO_VIEW_RECTIFY else None
    maps_right = RECTIFY_MAPS["right"] if EGO_VIEW_RECTIFY else None
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
        print(f"{_TAG} WARNING: pyrealsense2 not available, skipping {key}", file=sys.stderr)
        return
    pipeline = rs.pipeline()
    config = rs.config()
    try:
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, FRAME_SHAPE[1], FRAME_SHAPE[0], rs.format.bgr8, 30)
        pipeline.start(config)
    except Exception as exc:
        print(f"{_TAG} WARNING: could not start RealSense {serial} for {key}: {exc}", file=sys.stderr)
        return
    print(f"{_TAG} {key} (RealSense {serial}) live")
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


def camera_publish_loop(port: int, fps: float):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{port}")
    interval = 1.0 / fps
    print(f"{_TAG} publishing cameras on :{port}")
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


if __name__ == "__main__":
    threading.Thread(target=head_camera_thread, daemon=True).start()
    for key, serial in WRIST_SERIALS.items():
        threading.Thread(target=wrist_camera_thread, args=(key, serial), daemon=True).start()
    camera_publish_loop(5555, 30.0)
