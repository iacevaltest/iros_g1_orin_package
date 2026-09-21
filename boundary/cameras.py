"""Camera input — SUB on the Orin's camera server (:5555).

Wire format published by the organizer's camera server:

    ZMQ PUB, one msgpack frame per tick:
        {"timestamps": {key: float}, "images": {key: jpeg_bytes}}

    keys: "ego_view"        head camera, mono          (always present)
          "ego_view_left"   head camera, stereo left   (may be absent)
          "ego_view_right"  head camera, stereo right  (may be absent)
          "left_wrist"      RealSense D405             (may be absent)
          "right_wrist"     RealSense D405             (may be absent)

JPEGs are encoded BGR; :meth:`CameraStream.read` decodes and flips to RGB,
so what you get back is HWC uint8 RGB at 480x640x3 — the layout every
VLA stack expects.

Only ``ego_view`` is guaranteed. The server does not publish at all until
the ego camera is live, and it drops individual wrist keys when those
cameras are absent or failing. Write your client so a missing wrist key is
survivable.

If your policy wants stereo input instead of the mono ``ego_view``, declare
``ego_view_left``/``ego_view_right`` in your server's ``camera_keys``
metadata (see ``components/server.py``) — the mono ``ego_view`` keeps
publishing unchanged either way, so this is purely opt-in.

The socket is CONFLATE: you always get the newest frame, never a backlog.
If your policy is slower than 30 Hz you skip frames rather than fall behind,
which is what you want.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import msgpack
import numpy as np
import zmq

CAMERA_KEYS = ("ego_view", "ego_view_left", "ego_view_right", "left_wrist", "right_wrist")
REQUIRED_KEY = "ego_view"
FRAME_SHAPE = (480, 640, 3)

DEFAULT_PORT = 5555


@dataclass
class CameraFrame:
    """One synchronized publish from the camera server."""

    images: dict[str, np.ndarray]     # key -> (480, 640, 3) uint8 RGB
    timestamps: dict[str, float]      # key -> capture time (Orin clock)
    received_at: float                # local receive time

    @property
    def ego(self) -> np.ndarray:
        return self.images[REQUIRED_KEY]

    def age_s(self) -> float:
        return time.time() - self.received_at


class CameraStream:
    """Newest-frame-wins subscriber for the Orin camera server."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
        self.endpoint = f"tcp://{host}:{port}"
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.endpoint)
        self._latest: CameraFrame | None = None

    def read(self, timeout_ms: int = 1000) -> CameraFrame | None:
        """Block for the next frame. Returns None on timeout."""
        if not self._socket.poll(timeout_ms):
            return None
        frame = self._decode(self._socket.recv())
        self._latest = frame
        return frame

    def latest(self) -> CameraFrame | None:
        """The most recent frame read, without blocking for a new one."""
        return self._latest

    def wait_until_live(self, timeout_s: float = 30.0) -> CameraFrame:
        """Block until the ego camera is publishing. Raises on timeout."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            frame = self.read(timeout_ms=500)
            if frame is not None and REQUIRED_KEY in frame.images:
                return frame
        raise TimeoutError(
            f"no '{REQUIRED_KEY}' frame from {self.endpoint} within {timeout_s}s — "
            "is the organizer's camera server running?"
        )

    @staticmethod
    def _decode(blob: bytes) -> CameraFrame:
        msg = msgpack.unpackb(blob, raw=False)
        images = {}
        for key, jpeg in msg.get("images", {}).items():
            bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            images[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return CameraFrame(
            images=images,
            timestamps=dict(msg.get("timestamps", {})),
            received_at=time.time(),
        )

    def close(self):
        self._socket.close(linger=0)
