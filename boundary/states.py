"""Proprioception input — SUB on the Orin's state endpoint (:5557).

Wire format, identical for both lanes:

    ZMQ PUB, topic "g1_debug" (a string prefix on the same bytes),
    payload msgpack:
        body_q       (29,)  float   canonical G1 body joint positions, radians
        base_quat    (4,)   float   base orientation, wxyz
        left_hand_q  (7,)   float   OPTIONAL — absent on Dex1-1 rigs
        right_hand_q (7,)   float   OPTIONAL — absent on Dex1-1 rigs

This schema is guaranteed on :5557 whichever whole-body controller the
organizer is running, so your client never has to know which WBC it is
talking to and never needs ROS 2.

HAND STATE IS USUALLY ABSENT. The competition G1 carries Dex1-1 two-finger
grippers, not Dex3 hands, so ``left_hand_q`` / ``right_hand_q`` are normally
missing. ``RobotState.hands_present`` tells you; synthesize whatever your
model expects if it needs a hand vector.

JOINT ORDER: ``body_q`` follows Unitree's canonical G1 29-DoF ordering
(``G1JointIndex`` in ``unitree_sdk2py``). The per-index name table is in the
README. Slice it however your model was trained — the organizer does not
impose a subset or a reordering.

The socket is CONFLATE: newest state wins, no backlog.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import msgpack
import numpy as np
import zmq

STATE_TOPIC = "g1_debug"
BODY_DOF = 29
HAND_DOF = 7

DEFAULT_PORT = 5557

REQUIRED = {"body_q": (BODY_DOF,), "base_quat": (4,)}
OPTIONAL = {"left_hand_q": (HAND_DOF,), "right_hand_q": (HAND_DOF,)}


class StateError(ValueError):
    """A state message that does not match the published schema."""


@dataclass
class RobotState:
    """One decoded ``g1_debug`` message."""

    body_q: np.ndarray                      # (29,) float32, radians
    base_quat: np.ndarray                   # (4,) float32, wxyz
    left_hand_q: np.ndarray | None = None   # (7,) or None
    right_hand_q: np.ndarray | None = None  # (7,) or None
    received_at: float = field(default_factory=time.time)

    @property
    def hands_present(self) -> bool:
        return self.left_hand_q is not None and self.right_hand_q is not None

    def age_s(self) -> float:
        return time.time() - self.received_at


class StateStream:
    """Newest-state-wins subscriber for the Orin state endpoint."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
        self.endpoint = f"tcp://{host}:{port}"
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, STATE_TOPIC)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.endpoint)
        self._latest: RobotState | None = None

    def read(self, timeout_ms: int = 1000) -> RobotState | None:
        """Block for the next state. Returns None on timeout."""
        if not self._socket.poll(timeout_ms):
            return None
        state = self._decode(self._socket.recv())
        self._latest = state
        return state

    def latest(self) -> RobotState | None:
        return self._latest

    def wait_until_live(self, timeout_s: float = 30.0) -> RobotState:
        """Block until valid state arrives. Raises on timeout."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            state = self.read(timeout_ms=500)
            if state is not None:
                return state
        raise TimeoutError(
            f"no '{STATE_TOPIC}' message from {self.endpoint} within {timeout_s}s — "
            "is the organizer's state endpoint up?"
        )

    @staticmethod
    def _decode(blob: bytes) -> RobotState:
        prefix = STATE_TOPIC.encode("utf-8")
        if not blob.startswith(prefix):
            raise StateError(f"message does not carry the {STATE_TOPIC!r} topic prefix")
        msg = msgpack.unpackb(blob[len(prefix):], raw=False)
        if not isinstance(msg, dict):
            raise StateError(f"payload is {type(msg).__name__}, expected a dict")

        decoded = {}
        for name, shape in REQUIRED.items():
            if name not in msg:
                raise StateError(f"missing required key {name!r}")
            decoded[name] = _as_vector(msg[name], shape, name)
        for name, shape in OPTIONAL.items():
            decoded[name] = (
                _as_vector(msg[name], shape, name) if msg.get(name) is not None else None
            )
        return RobotState(**decoded)

    def close(self):
        self._socket.close(linger=0)


def _as_vector(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise StateError(f"{name} has shape {arr.shape}, expected {shape}")
    if not np.all(np.isfinite(arr)):
        raise StateError(f"{name} contains non-finite values")
    return arr
