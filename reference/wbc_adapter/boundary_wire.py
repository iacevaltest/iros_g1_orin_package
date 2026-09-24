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
  joint     : JOINT_TOPIC + msgpack({"actions": <float32 bytes>,
                                     "shape": [T, 22], "dtype": "f32",
                                     "issued_at": float})
              GOTO_TOPIC + msgpack({"left_arm": [7], "right_arm": [7],
                                    "max_speed": float, "hands": [2] (opt),
                                    "issued_at": float})
  sonic     : POSE_TOPIC + <1280B JSON header, NUL-padded> + <raw buffers>

The joint lane is carried on the same :5556 socket as the decoupled lane
and consumed by the same adapter process (wbc_driver.py --lane decoupled);
it differs only in what a row means -- arm joint angles instead of wrist
poses, so no IK is involved. See docs/CONTRACT.md, "The joint lane".

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
JOINT_TOPIC = b"joint"
GOTO_TOPIC = b"goto"
HEADER_SIZE = 1280
TASKSPACE_DIM = 25
JOINT_DIM = 22
ARM_DOF = 7
MAX_CHUNK_LENGTH = 64
LATENT_ABS_BOUND = 1.25
HAND_ABS_TOL = 1e-3

# (T, 22) joint-lane row layout -- fixed, do not reorder. Arm angles are
# radians in Unitree G1JointIndex order: shoulder_pitch, shoulder_roll,
# shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw.
JOINT_SLICES = {
    "left_hand": slice(0, 2),
    "right_hand": slice(2, 4),
    "left_arm": slice(4, 11),
    "right_arm": slice(11, 18),
    "navigate_cmd": slice(18, 21),
    "base_height_cmd": slice(21, 22),
}

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


@dataclass
class JointChunk:
    """One `joint`-lane chunk: (T, 22) hands + arm joint angles + base cmds."""
    actions: np.ndarray   # (T, 22) float32
    issued_at: float


@dataclass
class GotoRequest:
    """One `joint`-lane "go to pose" request: a target arm configuration the
    adapter interpolates to from the measured arms, at a bounded speed."""
    left_arm: np.ndarray            # (7,) radians, G1JointIndex order
    right_arm: np.ndarray           # (7,)
    max_speed: float                # rad/s, further capped by the adapter
    hands: np.ndarray | None        # (2,) -1 open .. +1 closed, or None = hold
    issued_at: float


def decode_taskspace(message: bytes) -> TaskspaceChunk:
    if not message.startswith(TASKSPACE_TOPIC):
        raise ValueError(f"frame does not start with topic {TASKSPACE_TOPIC!r}")
    msg = msgpack.unpackb(message[len(TASKSPACE_TOPIC):], raw=False)
    arr = np.frombuffer(msg["actions"], dtype=np.float32).reshape(msg["shape"])
    return TaskspaceChunk(actions=arr, issued_at=float(msg.get("issued_at", 0.0)))


def decode_joint(message: bytes) -> JointChunk:
    if not message.startswith(JOINT_TOPIC):
        raise ValueError(f"frame does not start with topic {JOINT_TOPIC!r}")
    msg = msgpack.unpackb(message[len(JOINT_TOPIC):], raw=False)
    dtype = msg.get("dtype", "f32")
    if dtype != "f32":
        raise ValueError(f"joint chunk dtype {dtype!r}, expected 'f32'")
    arr = np.frombuffer(msg["actions"], dtype=np.float32).reshape(msg["shape"])
    return JointChunk(actions=arr, issued_at=float(msg.get("issued_at", 0.0)))


def decode_goto(message: bytes) -> GotoRequest:
    if not message.startswith(GOTO_TOPIC):
        raise ValueError(f"frame does not start with topic {GOTO_TOPIC!r}")
    msg = msgpack.unpackb(message[len(GOTO_TOPIC):], raw=False)
    hands = msg.get("hands")
    return GotoRequest(
        left_arm=np.asarray(msg["left_arm"], dtype=np.float64).reshape(-1),
        right_arm=np.asarray(msg["right_arm"], dtype=np.float64).reshape(-1),
        max_speed=float(msg["max_speed"]),
        hands=None if hands is None else np.asarray(hands, dtype=np.float64).reshape(-1),
        issued_at=float(msg.get("issued_at", 0.0)),
    )


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


def validate_joint(chunk: JointChunk) -> list[str]:
    """`joint`-lane contract checks: shape, chunk length, finiteness and
    the hand range. Joint angles are NOT range-checked here -- the adapter
    clamps them to the robot model's limits (and counts every clamp), which
    is the safer response to a policy that overshoots a limit by a hair.
    """
    problems = []
    arr = chunk.actions
    if arr.ndim != 2 or arr.shape[1] != JOINT_DIM:
        problems.append(f"actions have shape {arr.shape}, expected (T, {JOINT_DIM})")
        return problems
    if not 1 <= arr.shape[0] <= MAX_CHUNK_LENGTH:
        problems.append(f"chunk length T={arr.shape[0]} outside 1..{MAX_CHUNK_LENGTH}")
        return problems
    if not np.isfinite(arr).all():
        problems.append("actions contain NaN or Inf")
        return problems
    hands = np.concatenate([arr[:, JOINT_SLICES["left_hand"]],
                            arr[:, JOINT_SLICES["right_hand"]]], axis=1)
    peak = float(np.max(np.abs(hands)))
    if peak > 1.0 + HAND_ABS_TOL:
        problems.append(f"hand commands must lie in [-1, 1]; peak |value| = {peak:.3f}")
    return problems


def validate_goto(req: GotoRequest) -> list[str]:
    problems = []
    for label, arm in (("left_arm", req.left_arm), ("right_arm", req.right_arm)):
        if arm.shape != (ARM_DOF,):
            problems.append(f"{label} has shape {arm.shape}, expected ({ARM_DOF},)")
        elif not np.isfinite(arm).all():
            problems.append(f"{label} contains NaN or Inf")
    if not (np.isfinite(req.max_speed) and req.max_speed > 0.0):
        problems.append(f"max_speed must be a positive finite rad/s, got {req.max_speed!r}")
    if req.hands is not None:
        if req.hands.shape != (2,):
            problems.append(f"hands has shape {req.hands.shape}, expected (2,)")
        elif not np.isfinite(req.hands).all():
            problems.append("hands contain NaN or Inf")
        elif float(np.max(np.abs(req.hands))) > 1.0 + HAND_ABS_TOL:
            problems.append("hands must lie in [-1, 1]")
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
