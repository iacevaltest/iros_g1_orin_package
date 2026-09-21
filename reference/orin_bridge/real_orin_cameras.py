"""Real camera publisher for the G1 -- camera-only half of real_orin.py, split
out to run in the teleimager conda env (has pyrealsense2/cv2) while the state
half runs separately in g1_wbc (has unitree_sdk2py/cyclonedds). Verbatim logic
from real_orin.py's camera threads, unchanged (calibration, eye-selection,
device-resolution comments all preserved as-is).

Cameras :5555 -- msgpack {"timestamps": {...}, "images": {key: bgr_jpeg}}

No robot SDK import at all -- pure V4L2/RealSense capture + ZMQ publish.
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
            print(f"[real_orin_cameras] head camera resolved to {path} (name={HEAD_DEVICE_NAME!r}, "
                  f"probe frame {frame.shape[1]}x{frame.shape[0]}) -- {len(candidates)} "
                  f"candidate node(s) shared this name", flush=True)
            return path
    if candidates:
        print(f"[real_orin_cameras] WARNING: found {len(candidates)} node(s) named "
              f"{HEAD_DEVICE_NAME!r} ({candidates}) but none produced a real capture "
              f"frame", file=sys.stderr)
    return None


HEAD_WIDTH = 3840
HEAD_HEIGHT = 1080
# Wrist RealSense serials are PER-ROBOT. Set these for your rig; there is no
# sensible default. Find them with: tools/diagnose_cameras.py
WRIST_SERIALS = {
    "left_wrist":  os.environ.get("LEFT_WRIST_SERIAL", ""),
    "right_wrist": os.environ.get("RIGHT_WRIST_SERIAL", ""),
}

FRAME_SHAPE = (480, 640)  # (h, w) -- matches boundary's FRAME_SHAPE

EGO_VIEW_EYE = os.environ.get("EGO_VIEW_EYE", "left")          # "left" or "right"
EGO_VIEW_RECTIFY = os.environ.get("EGO_VIEW_RECTIFY", "0") == "1"
PUBLISH_STEREO = os.environ.get("PUBLISH_STEREO", "1") != "0"

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
_RECTIFY_SIZE = (FRAME_SHAPE[1], FRAME_SHAPE[0])  # (w, h) = (640, 480)

_map1_left, _map2_left = cv2.initUndistortRectifyMap(
    _CAM_MATRIX_LEFT, _DIST_LEFT, _R1, _P1, _RECTIFY_SIZE, cv2.CV_16SC2
)
_map1_right, _map2_right = cv2.initUndistortRectifyMap(
    _CAM_MATRIX_RIGHT, _DIST_RIGHT, _R2, _P2, _RECTIFY_SIZE, cv2.CV_16SC2
)

_latest_lock = threading.Lock()
_latest_frames: dict[str, np.ndarray] = {}   # key -> BGR uint8 (480,640,3)


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
        print(f"[real_orin_cameras] WARNING: no head camera found (looked for a v4l2 device "
              f"named {HEAD_DEVICE_NAME!r}) -- ego_view* keys will never publish", file=sys.stderr)
        return

    cap = cv2.VideoCapture(head_device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, HEAD_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEAD_HEIGHT)
    if not cap.isOpened():
        print(f"[real_orin_cameras] WARNING: could not open head camera {head_device}", file=sys.stderr)
        return

    ok, probe = cap.read()
    if not ok:
        print("[real_orin_cameras] WARNING: head camera opened but produced no frame", file=sys.stderr)
        return
    actual_h, actual_w = probe.shape[:2]
    if (actual_w, actual_h) != (HEAD_WIDTH, HEAD_HEIGHT):
        print(f"[real_orin_cameras] WARNING: requested {HEAD_WIDTH}x{HEAD_HEIGHT} but camera gave "
              f"{actual_w}x{actual_h} -- eye split below assumes a side-by-side stereo "
              f"frame and will be wrong if this mode isn't genuinely binocular", file=sys.stderr)
    print(f"[real_orin_cameras] head camera live at {actual_w}x{actual_h}, "
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
        print(f"[real_orin_cameras] WARNING: pyrealsense2 not available, skipping {key}", file=sys.stderr)
        return
    pipeline = rs.pipeline()
    config = rs.config()
    try:
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, FRAME_SHAPE[1], FRAME_SHAPE[0], rs.format.bgr8, 30)
        pipeline.start(config)
    except Exception as exc:
        print(f"[real_orin_cameras] WARNING: could not start RealSense {serial} for {key}: {exc}", file=sys.stderr)
        return
    print(f"[real_orin_cameras] {key} (RealSense {serial}) live")
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
    print(f"[real_orin_cameras] publishing cameras on :{port}")
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
