"""Action output — BIND on :5556, the one lane-aware module.

    THE CLIENT BINDS. The whole-body controller connects to you.

That is backwards from the usual ZMQ habit and it trips everyone up once.
Your client is the PUB server on :5556; the organizer's WBC is the
subscriber that dials in. It follows from the WBC being the long-lived
process and your client being the thing that comes and goes.

--------------------------------------------------------------------------
Lane selection
--------------------------------------------------------------------------
Your lane is decided by your model, not by preference:

    sonic      GR00T N1.7 with the UNITREE_G1_SONIC embodiment. The only
               thing that emits a 64-dim motion token. Actions are named
               arrays framed for gear_sonic_deploy, streamed one row at a
               time at 50 Hz.

    decoupled  Everything else — Pi 0.5, GR00T N1.6, MolmoAct2, non-VLA
               methods. Actions are (T, 25) task-space chunks; the
               organizer's adapter runs inverse kinematics and drives the
               Decoupled WBC.

Declare it in your manifest. The bench brings up a different controller for
each, so a wrong declaration is caught before you get robot time, not during.

--------------------------------------------------------------------------
Safety
--------------------------------------------------------------------------
Both sinks validate before publishing and raise ActionError on violation —
they never publish a malformed action. This is the last in-band check before
29 DoF of humanoid moves. It is not a substitute for the organizer's e-stop,
which is independent of your code and always wins.

The SONIC latent bound (|motion_token| <= 1.25) is a training-range check,
not a semantic one. A latent inside the bound can still be nonsense; nothing
outside the paired decoder can tell. Treat it as a floor, not a guarantee.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod

import msgpack
import numpy as np
import zmq

DEFAULT_PORT = 5556

# --- SONIC lane ------------------------------------------------------------
LATENT_DIM = 64
HAND_DOF = 7
LATENT_ABS_BOUND = 1.25
POSE_TOPIC = b"pose"
POSE_PROTOCOL_VERSION = 4
HEADER_SIZE = 1280

# --- Decoupled lane --------------------------------------------------------
TASKSPACE_DIM = 25
TASKSPACE_TOPIC = b"taskspace"
MAX_CHUNK_LENGTH = 64

# (T, 25) row layout — fixed, do not reorder:
#   [0:2]   left hand, 2 finger joints, -1 = open, +1 = closed
#   [2:4]   right hand, same convention
#   [4:7]   left end-effector position (xyz, metres)
#   [7:11]  left end-effector quaternion (w, x, y, z)
#   [11:14] right end-effector position
#   [14:18] right end-effector quaternion (w, x, y, z)
#   [18:21] navigate_cmd (vx, vy, yaw_rate)
#   [21]    base_height_cmd
#   [22:25] torso_orientation_rpy_cmd (roll, pitch, yaw)
TASKSPACE_SLICES = {
    "left_hand": slice(0, 2),
    "right_hand": slice(2, 4),
    "left_ee_pos": slice(4, 7),
    "left_ee_quat": slice(7, 11),
    "right_ee_pos": slice(11, 14),
    "right_ee_quat": slice(14, 18),
    "navigate_cmd": slice(18, 21),
    "base_height_cmd": slice(21, 22),
    "torso_rpy": slice(22, 25),
}

_DTYPE_TAG = {
    np.dtype(np.float32): "f32",
    np.dtype(np.int64): "i64",
}


class ActionError(ValueError):
    """An action that violates the boundary contract. Never published."""


class ActionSink(ABC):
    """Base for both lanes: owns the bound PUB socket, nothing else."""

    def __init__(self, port: int = DEFAULT_PORT, host: str = "*"):
        self.endpoint = f"tcp://{host}:{port}"
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 20)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(self.endpoint)   # we bind; the WBC connects
        self.published = 0

    @staticmethod
    def for_lane(lane: str, port: int = DEFAULT_PORT, host: str = "*") -> "ActionSink":
        """Build the sink for a declared lane."""
        sinks = {"sonic": SonicSink, "decoupled": DecoupledSink}
        if lane not in sinks:
            raise ActionError(
                f"unknown lane {lane!r}; expected one of {sorted(sinks)}"
            )
        return sinks[lane](port=port, host=host)

    @property
    @abstractmethod
    def lane(self) -> str: ...

    def _publish(self, blob: bytes):
        self._socket.send(blob)
        self.published += 1

    def close(self):
        self._socket.close(linger=0)


# ---------------------------------------------------------------------------
# SONIC lane
# ---------------------------------------------------------------------------


class SonicSink(ActionSink):
    """Streams latent actions to gear_sonic_deploy, one row per call.

    The deploy binary consumes single poses at its own 50 Hz cadence, so a
    policy chunk of shape (T, 64) is sent as T successive calls to
    :meth:`send_step` — not as one message. Validate the whole chunk first
    with :meth:`validate_chunk`, then stream it.
    """

    def __init__(self, port: int = DEFAULT_PORT, host: str = "*"):
        super().__init__(port=port, host=host)
        self._frame_index = 0

    @property
    def lane(self) -> str:
        return "sonic"

    @staticmethod
    def validate_chunk(
        motion_token: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Check one policy chunk end to end. Returns the coerced arrays."""
        token = _coerce_2d(motion_token, LATENT_DIM, "motion_token")
        left = _coerce_2d(left_hand, HAND_DOF, "left_hand_joints")
        right = _coerce_2d(right_hand, HAND_DOF, "right_hand_joints")
        if not (len(token) == len(left) == len(right)):
            raise ActionError(
                f"chunk lengths disagree: motion_token={len(token)}, "
                f"left_hand={len(left)}, right_hand={len(right)}"
            )
        if len(token) > MAX_CHUNK_LENGTH:
            raise ActionError(f"chunk length {len(token)} exceeds max {MAX_CHUNK_LENGTH}")
        peak = float(np.max(np.abs(token)))
        if peak > LATENT_ABS_BOUND:
            raise ActionError(
                f"max|motion_token| = {peak:.3f} exceeds the training-range bound "
                f"{LATENT_ABS_BOUND} — chunk rejected"
            )
        return token, left, right

    def send_step(
        self,
        motion_token: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ):
        """Validate and publish one 50 Hz row."""
        token, left, right = self.validate_chunk(motion_token, left_hand, right_hand)
        if len(token) != 1:
            raise ActionError(
                f"send_step takes one row, got {len(token)}; iterate the chunk yourself"
            )
        self._publish(
            _frame_named_arrays(
                {
                    "token_state": token,
                    "frame_index": np.array([self._frame_index], dtype=np.int64),
                    "left_hand_joints": left,
                    "right_hand_joints": right,
                },
                topic=POSE_TOPIC,
                version=POSE_PROTOCOL_VERSION,
            )
        )
        self._frame_index += 1

    def send_chunk(
        self,
        motion_token: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        rate_hz: float = 50.0,
    ):
        """Validate a whole chunk, then stream its rows at ``rate_hz``.

        Blocking. Most clients want their own loop instead, so they can
        interleave a fresh observation or abort mid-chunk.
        """
        token, left, right = self.validate_chunk(motion_token, left_hand, right_hand)
        period = 1.0 / rate_hz
        for i in range(len(token)):
            tick = time.monotonic()
            self.send_step(token[i:i + 1], left[i:i + 1], right[i:i + 1])
            remaining = period - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)


# ---------------------------------------------------------------------------
# Decoupled lane
# ---------------------------------------------------------------------------


class DecoupledSink(ActionSink):
    """Publishes (T, 25) task-space chunks to the organizer's IK adapter.

    Unlike the SONIC lane, the whole chunk goes out as one message: the
    adapter owns the interpolation and the 50 Hz cadence, so it wants the
    trajectory rather than a stream of rows.
    """

    @property
    def lane(self) -> str:
        return "decoupled"

    @staticmethod
    def validate_chunk(actions: np.ndarray) -> np.ndarray:
        """Check one (T, 25) chunk. Returns the coerced float32 array."""
        arr = np.asarray(actions)
        if arr.ndim == 1 and arr.shape[0] == TASKSPACE_DIM:
            arr = arr.reshape(1, TASKSPACE_DIM)   # single-step chunks are legal
        if arr.ndim != 2 or arr.shape[1] != TASKSPACE_DIM:
            raise ActionError(
                f"actions have shape {arr.shape}, expected (T, {TASKSPACE_DIM})"
            )
        if arr.shape[0] < 1:
            raise ActionError("action chunk is empty")
        if arr.shape[0] > MAX_CHUNK_LENGTH:
            raise ActionError(
                f"chunk length T={arr.shape[0]} exceeds max {MAX_CHUNK_LENGTH}"
            )
        if not np.issubdtype(arr.dtype, np.floating):
            raise ActionError(f"actions dtype {arr.dtype} is not floating")
        arr = arr.astype(np.float32, copy=False)
        if not np.all(np.isfinite(arr)):
            raise ActionError("actions contain non-finite values")

        hands = np.concatenate(
            [arr[:, TASKSPACE_SLICES["left_hand"]], arr[:, TASKSPACE_SLICES["right_hand"]]],
            axis=1,
        )
        if np.max(np.abs(hands)) > 1.0 + 1e-3:
            raise ActionError(
                f"hand commands must lie in [-1, 1] (-1 open, +1 closed); "
                f"peak |value| = {float(np.max(np.abs(hands))):.3f}"
            )
        for side in ("left", "right"):
            quat = arr[:, TASKSPACE_SLICES[f"{side}_ee_quat"]]
            norms = np.linalg.norm(quat, axis=1)
            if np.any(np.abs(norms - 1.0) > 1e-2):
                worst = float(norms[int(np.argmax(np.abs(norms - 1.0)))])
                raise ActionError(
                    f"{side} end-effector quaternions are not unit length "
                    f"(worst |q| = {worst:.4f}, expected 1.0). Note the layout is "
                    "(w, x, y, z) — but a wrongly-ordered unit quaternion still "
                    "passes this check, so verify your ordering by hand."
                )
        return arr

    def send_chunk(self, actions: np.ndarray, issued_at: float | None = None):
        """Validate and publish one task-space chunk."""
        arr = self.validate_chunk(actions)
        payload = msgpack.packb(
            {
                "actions": arr.tobytes(),
                "shape": list(arr.shape),
                "dtype": "f32",
                "issued_at": issued_at if issued_at is not None else time.time(),
            },
            use_bin_type=True,
        )
        self._publish(TASKSPACE_TOPIC + payload)


# ---------------------------------------------------------------------------
# Framing helpers
# ---------------------------------------------------------------------------


def _coerce_2d(value, dim: int, name: str) -> np.ndarray:
    """Accept (B,T,D) with B==1, (T,D) or (D,); return float32 (T, D)."""
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.floating):
        raise ActionError(f"{name} has non-float dtype {arr.dtype}")
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ActionError(f"{name} has batch size {arr.shape[0]}, expected 1")
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ActionError(f"{name} has {arr.ndim} dims after squeeze, expected 2")
    if arr.shape[1] != dim:
        raise ActionError(f"{name} last dim is {arr.shape[1]}, expected {dim}")
    if not np.all(np.isfinite(arr)):
        raise ActionError(f"{name} contains non-finite values")
    return arr


def _frame_named_arrays(arrays: dict[str, np.ndarray], topic: bytes, version: int) -> bytes:
    """Pack named arrays as [topic][zero-padded JSON header][little-endian data].

    The header describes each field's name, dtype tag and shape in order;
    the payload is the raw C-contiguous buffers concatenated in that same
    order. This is the framing gear_sonic_deploy expects.
    """
    fields, buffers = [], []
    for name, value in arrays.items():
        arr = np.ascontiguousarray(value)
        if arr.dtype not in _DTYPE_TAG:
            arr = arr.astype(np.float32)
        if arr.dtype.byteorder == ">":
            arr = arr.astype(arr.dtype.newbyteorder("<"))
        fields.append({"name": name, "dtype": _DTYPE_TAG[arr.dtype], "shape": list(arr.shape)})
        buffers.append(arr.tobytes())

    header = json.dumps(
        {"v": version, "endian": "le", "count": 1, "fields": fields},
        separators=(",", ":"),
    ).encode("utf-8")
    if len(header) > HEADER_SIZE:
        raise ActionError(f"header is {len(header)} bytes, over the {HEADER_SIZE} limit")
    return topic + header.ljust(HEADER_SIZE, b"\x00") + b"".join(buffers)
