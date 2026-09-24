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

    joint      A policy trained on joint-space actions (``robot_q_desired``
               style: the 7+7 arm angles themselves). Actions are (T, 22)
               chunks of hands + arm angles + base commands, sent on the
               SAME socket and consumed by the SAME adapter and WBC as
               ``decoupled`` — no IK anywhere, the angles go to the
               controller as published (bounded, never re-solved). Also
               carries a ``goto`` request to move the arms to a start
               pose at a bounded speed.

Declare it in your manifest. The bench brings up a different controller for
each, so a wrong declaration is caught before you get robot time, not during.
(``joint`` and ``decoupled`` share a controller; the declaration still tells
the organizer what your rows mean.)

--------------------------------------------------------------------------
Safety
--------------------------------------------------------------------------
All sinks validate before publishing and raise ActionError on violation —
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

# --- Joint lane ------------------------------------------------------------
JOINT_DIM = 22
JOINT_TOPIC = b"joint"
GOTO_TOPIC = b"goto"
ARM_DOF = 7
BODY_DOF = 29
GOTO_MIN_SPEED = 0.01   # rad/s; the adapter refuses slower goto requests

# (T, 22) row layout — fixed, do not reorder. Arm angles are radians in
# Unitree's canonical G1JointIndex order: shoulder_pitch, shoulder_roll,
# shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw — the same order
# ``body_q[15:22]`` / ``body_q[22:29]`` on :5557 use, so a policy that
# consumes body_q can publish in the frame it observed.
#   [0:2]   left hand, 2 finger joints, -1 = open, +1 = closed
#   [2:4]   right hand, same convention
#   [4:11]  left arm, 7 joint angles (rad)
#   [11:18] right arm, 7 joint angles (rad)
#   [18:21] navigate_cmd (vx, vy, yaw_rate)
#   [21]    base_height_cmd
JOINT_SLICES = {
    "left_hand": slice(0, 2),
    "right_hand": slice(2, 4),
    "left_arm": slice(4, 11),
    "right_arm": slice(11, 18),
    "navigate_cmd": slice(18, 21),
    "base_height_cmd": slice(21, 22),
}
# Where the arms sit in the (29,) ``body_q`` the state stream publishes.
BODY_Q_LEFT_ARM = slice(15, 22)
BODY_Q_RIGHT_ARM = slice(22, 29)

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
        sinks = {"sonic": SonicSink, "decoupled": DecoupledSink, "joint": JointSink}
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
# Joint lane
# ---------------------------------------------------------------------------


class JointSink(ActionSink):
    """Publishes (T, 22) joint-space chunks, plus ``goto`` start-pose requests.

    Same transport and cadence model as :class:`DecoupledSink` — one whole
    chunk per message, the adapter schedules it at its own rate — but the
    rows carry arm joint angles instead of wrist poses, so the adapter runs
    no IK on them. Its position clamp (robot-model limits) and step clamp
    (``--max-joint-vel``) still apply; the controller's own safety monitor
    and the e-stop still win.

    ``send_goto`` asks the adapter to interpolate the arms from where they
    are to a target at a bounded speed. It returns immediately; watch
    ``body_q`` on :5557 and use :func:`arms_reached` to decide arrival. Do
    not publish chunks while a goto is under way — a newer message
    supersedes it (the socket is newest-wins), which is by design.
    """

    @property
    def lane(self) -> str:
        return "joint"

    @staticmethod
    def make_rows(
        left_arm: np.ndarray,
        right_arm: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        navigate: np.ndarray | None = None,
        base_height: np.ndarray | None = None,
    ) -> np.ndarray:
        """Assemble a (T, 22) chunk from its parts.

        ``left_arm``/``right_arm`` are (T, 7) radians (or (7,) for T=1).
        ``left_hand``/``right_hand`` are (T, 2) in [-1, 1] (-1 open, +1
        closed) and are REQUIRED: there is no "hold" value on this wire, and
        defaulting them would silently open or close a gripper mid-task.
        ``navigate`` (T, 3) and ``base_height`` (T,) default to zero.
        """
        if left_hand is None or right_hand is None:
            raise ActionError(
                "left_hand and right_hand are required: pass (T, 2) commands in "
                "[-1, 1] (-1 = open, +1 = closed) for every row; there is no "
                "default that is safe mid-task"
            )
        left = _rows(left_arm, ARM_DOF, "left_arm")
        right = _rows(right_arm, ARM_DOF, "right_arm")
        T = left.shape[0]
        if right.shape[0] != T:
            raise ActionError(f"left_arm has {T} rows, right_arm has {right.shape[0]}")
        rows = np.zeros((T, JOINT_DIM), dtype=np.float32)
        rows[:, JOINT_SLICES["left_arm"]] = left
        rows[:, JOINT_SLICES["right_arm"]] = right
        rows[:, JOINT_SLICES["left_hand"]] = _rows(left_hand, 2, "left_hand", T)
        rows[:, JOINT_SLICES["right_hand"]] = _rows(right_hand, 2, "right_hand", T)
        if navigate is not None:
            rows[:, JOINT_SLICES["navigate_cmd"]] = _rows(navigate, 3, "navigate", T)
        if base_height is not None:
            rows[:, JOINT_SLICES["base_height_cmd"]] = _rows(base_height, 1, "base_height", T)
        return rows

    @staticmethod
    def validate_chunk(actions: np.ndarray) -> np.ndarray:
        """Check one (T, 22) chunk. Returns the coerced float32 array."""
        arr = np.asarray(actions)
        if arr.ndim == 1 and arr.shape[0] == JOINT_DIM:
            arr = arr.reshape(1, JOINT_DIM)
        if arr.ndim != 2 or arr.shape[1] != JOINT_DIM:
            raise ActionError(
                f"actions have shape {arr.shape}, expected (T, {JOINT_DIM})"
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
            [arr[:, JOINT_SLICES["left_hand"]], arr[:, JOINT_SLICES["right_hand"]]],
            axis=1,
        )
        if np.max(np.abs(hands)) > 1.0 + 1e-3:
            raise ActionError(
                f"hand commands must lie in [-1, 1] (-1 open, +1 closed); "
                f"peak |value| = {float(np.max(np.abs(hands))):.3f}"
            )
        return arr

    def send_chunk(self, actions: np.ndarray, issued_at: float | None = None):
        """Validate and publish one joint-space chunk."""
        arr = self.validate_chunk(actions)
        self._publish(_pack_joint_chunk(arr, issued_at))

    def send_goto(
        self,
        left_arm: np.ndarray,
        right_arm: np.ndarray,
        max_speed: float = 0.3,
        hands: tuple[float, float] | None = None,
        issued_at: float | None = None,
    ):
        """Ask the adapter to move the arms to (``left_arm``, ``right_arm``).

        ``max_speed`` is rad/s on the fastest joint, at least 0.01; the
        adapter caps it at its own ``--goto-max-speed`` (0.45 by default)
        and refuses a move that would take longer than 15 s at the
        resulting speed. ``hands`` is an optional (left, right) pair in
        [-1, 1]; omitted means the grippers are left as they are.
        Non-blocking.
        """
        self._publish(_pack_goto(left_arm, right_arm, max_speed, hands, issued_at))


def arms_reached(
    body_q29: np.ndarray,
    left_arm: np.ndarray,
    right_arm: np.ndarray,
    tol_rad: float = 0.05,
) -> bool:
    """True when every arm joint in ``body_q`` (the (29,) state vector from
    :5557) is within ``tol_rad`` of the target. Pure; pair it with
    ``StateStream`` to wait for a ``send_goto`` to land."""
    q = np.asarray(body_q29, dtype=np.float64).reshape(-1)
    if q.shape[0] != BODY_DOF:
        raise ActionError(f"body_q has {q.shape[0]} entries, expected {BODY_DOF}")
    target = np.concatenate([
        np.asarray(left_arm, dtype=np.float64).reshape(-1),
        np.asarray(right_arm, dtype=np.float64).reshape(-1),
    ])
    if target.shape[0] != 2 * ARM_DOF:
        raise ActionError(f"left_arm + right_arm must be 2 x {ARM_DOF} values")
    measured = np.concatenate([q[BODY_Q_LEFT_ARM], q[BODY_Q_RIGHT_ARM]])
    return bool(np.max(np.abs(measured - target)) <= tol_rad)


def _rows(value, dim: int, name: str, expect_T: int | None = None) -> np.ndarray:
    """Coerce (T, dim), (dim,) or, for dim == 1, (T,) to float32 (T, dim)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1) if dim == 1 else arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ActionError(f"{name} has shape {arr.shape}, expected (T, {dim})")
    if expect_T is not None and arr.shape[0] != expect_T:
        raise ActionError(f"{name} has {arr.shape[0]} rows, expected {expect_T}")
    if not np.all(np.isfinite(arr)):
        raise ActionError(f"{name} contains non-finite values")
    return arr.astype(np.float32)


def _pack_joint_chunk(arr: np.ndarray, issued_at: float | None = None) -> bytes:
    """The exact bytes ``JointSink.send_chunk`` publishes for a validated chunk."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    payload = msgpack.packb(
        {
            "actions": arr.tobytes(),
            "shape": list(arr.shape),
            "dtype": "f32",
            "issued_at": issued_at if issued_at is not None else time.time(),
        },
        use_bin_type=True,
    )
    return JOINT_TOPIC + payload


def _pack_goto(left_arm, right_arm, max_speed: float, hands, issued_at: float | None) -> bytes:
    """The exact bytes ``JointSink.send_goto`` publishes (validated here)."""
    left = _rows(left_arm, ARM_DOF, "left_arm", 1)[0]
    right = _rows(right_arm, ARM_DOF, "right_arm", 1)[0]
    if not (np.isfinite(max_speed) and max_speed >= GOTO_MIN_SPEED):
        raise ActionError(
            f"max_speed must be a finite rad/s >= {GOTO_MIN_SPEED}, got {max_speed!r}")
    msg = {
        "left_arm": [float(v) for v in left],
        "right_arm": [float(v) for v in right],
        "max_speed": float(max_speed),
        "issued_at": issued_at if issued_at is not None else time.time(),
    }
    if hands is not None:
        h = _rows(hands, 2, "hands", 1)[0]
        if np.max(np.abs(h)) > 1.0 + 1e-3:
            raise ActionError("hands must lie in [-1, 1] (-1 open, +1 closed)")
        msg["hands"] = [float(h[0]), float(h[1])]
    return GOTO_TOPIC + msgpack.packb(msg, use_bin_type=True)


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
