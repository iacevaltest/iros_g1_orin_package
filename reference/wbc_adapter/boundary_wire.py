"""Decode what a team's client publishes on :5556.

The team's `components/client.py` BINDS :5556 and publishes through the
organizer's `boundary/actions.py`; the WBC side is a SUBSCRIBER that dials
in. That is the organizer's documented design ("THE CLIENT BINDS. The
whole-body controller connects to you") and it is why this adapter can run
every team's containers completely unmodified -- we are simply the
subscriber the contract always expected on the other end.

Wire formats are organizer-owned and identical for every team. Decoders
below mirror `mocks/mock_wbc.py`'s (the organizer's own reference
subscriber), verified against `boundary/actions.py` in the upstream
template:

  decoupled : TASKSPACE_TOPIC + msgpack({"actions": <float32 bytes>,
                                         "shape": [T, 25],
                                         "issued_at": float})
  sonic     : POSE_TOPIC + <1280B JSON header, NUL-padded> + <raw buffers>

Nothing here is team-specific. Do not add team-specific handling to this
module -- if a team needs special treatment at this layer, they are off
contract and that is a finding, not an adapter feature.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import msgpack
import numpy as np

# Mirrors boundary/actions.py. Kept as literals rather than imported so this
# adapter does not depend on any one team's checkout of the template.
POSE_TOPIC = b"pose"
TASKSPACE_TOPIC = b"taskspace"
HEADER_SIZE = 1280
TASKSPACE_DIM = 25
LATENT_ABS_BOUND = 1.25

_TAG_DTYPE = {"f32": np.float32, "f64": np.float64, "i32": np.int32,
              "i64": np.int64, "u8": np.uint8, "bool": np.bool_}


@dataclass
class TaskspaceChunk:
    """One `decoupled`-lane message: a (T, 25) task-space action chunk."""
    actions: np.ndarray   # (T, 25) float32
    issued_at: float      # sender's wall clock when inference was issued


@dataclass
class PoseStep:
    """One `sonic`-lane message: a single 50 Hz latent row + hand joints."""
    fields: dict[str, np.ndarray]


def decode_taskspace(message: bytes) -> TaskspaceChunk:
    if not message.startswith(TASKSPACE_TOPIC):
        raise ValueError(f"frame does not start with topic {TASKSPACE_TOPIC!r}")
    msg = msgpack.unpackb(message[len(TASKSPACE_TOPIC):], raw=False)
    arr = np.frombuffer(msg["actions"], dtype=np.float32).reshape(msg["shape"])
    return TaskspaceChunk(actions=arr, issued_at=float(msg.get("issued_at", 0.0)))


def decode_pose(message: bytes) -> PoseStep:
    if not message.startswith(POSE_TOPIC):
        raise ValueError(f"frame does not start with topic {POSE_TOPIC!r}")
    start = len(POSE_TOPIC)
    header = json.loads(message[start:start + HEADER_SIZE].rstrip(b"\x00").decode("utf-8"))
    payload = message[start + HEADER_SIZE:]

    out, offset = {}, 0
    for field in header["fields"]:
        dtype = np.dtype(_TAG_DTYPE[field["dtype"]]).newbyteorder("<")
        count = int(np.prod(field["shape"])) if field["shape"] else 1
        nbytes = dtype.itemsize * count
        out[field["name"]] = np.frombuffer(
            payload[offset:offset + nbytes], dtype=dtype
        ).reshape(field["shape"])
        offset += nbytes
    return PoseStep(fields=out)


STATE_TOPIC = b"g1_debug"
STATE_PORT = 5557
BODY_DOF = 29


class BoundaryStateSubscriber:
    """Read real `body_q` off the organizer's state endpoint (:5557).

    Same schema `boundary/states.py` guarantees -- msgpack under a
    `g1_debug` topic prefix, CONFLATE so newest state wins. Decoded here
    rather than importing the team's `boundary` package, so the adapter
    stays independent of any team checkout.

    This matters for dry runs: without it the IK is solved against a zero
    joint vector, which is not the arm's real configuration and makes any
    reachability number meaningless.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = STATE_PORT):
        import zmq
        self._zmq = zmq
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.SUB)
        self._sock.setsockopt(self._zmq.SUBSCRIBE, STATE_TOPIC)
        self._sock.setsockopt(self._zmq.CONFLATE, 1)
        self._sock.setsockopt(self._zmq.LINGER, 0)
        self._sock.connect(f"tcp://{host}:{port}")
        self.endpoint = f"tcp://{host}:{port}"
        self._latest = None

    def get_body_q(self):
        """Newest (29,) body_q, or None until the first message arrives."""
        import numpy as np
        try:
            blob = self._sock.recv(self._zmq.NOBLOCK)
        except self._zmq.Again:
            return self._latest
        try:
            msg = msgpack.unpackb(blob[len(STATE_TOPIC):], raw=False)
            q = np.asarray(msg["body_q"], dtype=np.float64).reshape(-1)
            if q.shape[0] == BODY_DOF:
                self._latest = q
        except Exception:
            pass
        return self._latest


def validate_taskspace(chunk: TaskspaceChunk) -> list[str]:
    """Contract checks. Same floor `boundary/actions.py` applies before it
    publishes -- re-checked here because we are the last thing between a
    team's output and 29 DoF of humanoid, and because a team could always
    have replaced their transport (components/ is team-owned).

    Well-formed is NOT the same as good. A unit quaternion pointing
    somewhere absurd passes every check here.
    """
    problems = []
    arr = chunk.actions
    if arr.ndim != 2 or arr.shape[1] != TASKSPACE_DIM:
        problems.append(f"actions have shape {arr.shape}, expected (T, {TASKSPACE_DIM})")
        return problems
    if not np.isfinite(arr).all():
        problems.append("actions contain NaN or Inf")
        return problems
    for label, cols in (("left", slice(7, 11)), ("right", slice(14, 18))):
        norms = np.linalg.norm(arr[:, cols], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            problems.append(
                f"{label} EE quaternion not unit length "
                f"(min {norms.min():.4f}, max {norms.max():.4f})"
            )
    return problems


def validate_pose(step: PoseStep) -> list[str]:
    """`sonic`-lane contract checks, mirroring mock_wbc.py's."""
    problems = []
    for name, shape in (("token_state", (1, 64)), ("frame_index", (1,)),
                        ("left_hand_joints", (1, 7)), ("right_hand_joints", (1, 7))):
        if name not in step.fields:
            problems.append(f"missing field {name!r}")
        elif step.fields[name].shape != shape:
            problems.append(f"{name} has shape {step.fields[name].shape}, expected {shape}")
    if "token_state" in step.fields:
        peak = float(np.max(np.abs(step.fields["token_state"])))
        if peak > LATENT_ABS_BOUND:
            problems.append(f"max|motion_token| = {peak:.3f} > {LATENT_ABS_BOUND}")
    return problems
