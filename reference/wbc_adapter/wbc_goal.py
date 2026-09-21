"""Publish goal dicts into NVIDIA's Decoupled WBC, and read robot state back.

Verified against `NVlabs/GR00T-WholeBodyControl` v1.1 (commit a0732b6,
2026-08-20), not against second-hand notes:

  * `decoupled_wbc/control/main/constants.py`
        CONTROL_GOAL_TOPIC = "ControlPolicy/upper_body_pose"
        STATE_TOPIC_NAME   = "G1Env/env_state_act"
        DEFAULT_BASE_HEIGHT = 0.74      # matches the (T,25) col [21] convention
        DEFAULT_NAV_CMD     = [0, 0, 0]
  * `decoupled_wbc/control/policy/g1_decoupled_whole_body_policy.py::set_goal`
        consumes target_upper_body_pose / base_height_command / target_time /
        interpolation_garbage_collection_time / navigate_cmd. `target_time`
        may be a LIST -- the upper-body policy is interpolation over
        waypoints, which is what lets us hand it a whole (T,25) chunk
        instead of just its first row.
  * `decoupled_wbc/control/main/teleop/run_g1_control_loop.py`
        sets `interpolation_garbage_collection_time` itself on every tick,
        so we deliberately do NOT set it here.
  * `decoupled_wbc/control/utils/ros_utils.py`
        ROSMsgPublisher/Subscriber = msgpack (+msgpack_numpy) packed into a
        ByteMultiArray whose `data` is a tuple of single bytes.

`wrist_pose` (14 = 2 x [xyz + wxyz]) rides along because the control loop
exports it (`action.eef`) for logging/replay parity. It is NOT consumed by
set_goal -- the WBC runs on joint space. Do not mistake it for the thing
that moves the arms.

The "zmq" backend exists so the whole pipeline can be exercised off-robot
(no ROS 2, no G1) with byte-identical payloads. Modelled on the same
two-backend split in the operator harness at ~/peval_v1-main.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np

CONTROL_GOAL_TOPIC = "ControlPolicy/upper_body_pose"
STATE_TOPIC_NAME = "G1Env/env_state_act"
DEFAULT_NAV_CMD = [0.0, 0.0, 0.0]
DEFAULT_BASE_HEIGHT = 0.74

ZMQ_GOAL_TOPIC = b"wbc_goal"
ZMQ_STATE_TOPIC = b"wbc_state"


def pack_payload(msg: dict) -> bytes:
    return msgpack.packb(msg, default=mnp.encode)


def unpack_payload(data: bytes):
    return msgpack.unpackb(data, object_hook=mnp.decode, raw=False)


def build_goal(
    upper_body_waypoints: np.ndarray,
    target_time: list[float] | float | None = None,
    base_height_command=DEFAULT_BASE_HEIGHT,
    navigate_cmd=None,
    wrist_pose=None,
) -> dict:
    """Assemble one WBC goal.

    `upper_body_waypoints` is either a single (N,) joint vector or a (T, N)
    trajectory, where N is the WBC's own upper-body joint-group width (14
    without waist, 17 with -- depends on how the control loop was launched;
    never hardcode it, read it from the robot model. See ik.py).

    --------------------------------------------------------------------
    TRAJECTORY MODE HAS A STRICT FORM -- GET IT WRONG AND THE WBC DIES
    --------------------------------------------------------------------
    When `target_time` is a LIST, `InterpolationPolicy.set_goal` requires
    EVERY other key to be a `list` of exactly that length:

        if isinstance(target_time, list):
            for key, vec in goal.items():
                assert isinstance(vec, list)
                assert len(vec) == len(target_time)

    A numpy (T, N) array is NOT a list and trips that assert. Found the
    hard way: against a live control loop the adapter happily reported
    "79 published, 0 rejected" while the WBC crashed on every single goal
    with `AssertionError` in `interpolation_policy.py`. Publishing
    successfully says nothing about the WBC accepting it.

    Per-key widths, from the policy's own `init_values`
    (`wbc_policy_factory.py`): `base_height_command` is **1**,
    `navigate_cmd` is **3**, `target_upper_body_pose` is 14 or 17. Each
    key must end up as a (T, D) array after `np.array(...)`, so scalars
    have to be wrapped per waypoint -- `[[h], [h], ...]`, not `[h, h, ...]`
    (the latter becomes (T,) and gets mis-tiled to (T, T)).

    NVIDIA's own safety injection confirms the shape:
    `[np.array(DEFAULT_NAV_CMD)] * len(target_time)`.

    `wrist_pose` is carried for logging parity only -- the whole-body
    policy filters it out before the interpolation policy sees it, so it
    is exempt from the list rule.

    `interpolation_garbage_collection_time` is deliberately NOT set here:
    `run_g1_control_loop.py` adds it itself every tick.

    NOTE `target_time` is compared against the WBC's own
    `time.monotonic()`. That is only meaningful if this adapter runs on
    the SAME HOST as the control loop. It does (both on PC2) -- do not
    move it off-box without switching to a shared clock.
    """
    wp = np.asarray(upper_body_waypoints, dtype=np.float64)
    if wp.ndim == 1:
        wp = wp[None, :]
    T = wp.shape[0]

    def _per_waypoint(value, width, default):
        """-> list of T (width,) arrays, whatever shape came in."""
        if value is None:
            value = default
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim == 1:
            arr = np.tile(arr.reshape(1, -1), (T, 1)) if arr.shape[0] == width \
                else np.tile(arr.reshape(-1, 1), (1, width))[:T]
        if arr.shape[0] != T:
            arr = np.tile(arr[:1], (T, 1))
        return [np.asarray(row, dtype=np.float64) for row in arr]

    trajectory = isinstance(target_time, (list, tuple))

    if trajectory:
        goal: dict[str, Any] = {
            "target_upper_body_pose": [np.asarray(r, dtype=np.float64) for r in wp],
            "base_height_command": _per_waypoint(base_height_command, 1, DEFAULT_BASE_HEIGHT),
            "navigate_cmd": _per_waypoint(navigate_cmd, 3, DEFAULT_NAV_CMD),
            "target_time": list(target_time),
        }
    else:
        goal = {
            "target_upper_body_pose": wp[0],
            "base_height_command": np.asarray(
                [float(np.ravel(base_height_command)[0])], dtype=np.float64),
            "navigate_cmd": np.asarray(
                DEFAULT_NAV_CMD if navigate_cmd is None else np.ravel(navigate_cmd)[:3],
                dtype=np.float64),
        }
        if target_time is not None:
            goal["target_time"] = float(target_time)

    if wrist_pose is not None:
        goal["wrist_pose"] = np.asarray(wrist_pose, dtype=np.float64)
    return goal


class ZmqWbcBackend:
    """Bench loopback: goals out on a bound PUB, state in from a SUB.

    Same payload bytes as the ROS 2 backend, different carrier -- so an
    off-robot test exercises the real encoding, not a stand-in for it.
    """

    name = "zmq"

    def __init__(self, goal_port: int = 5610, state_port: int = 5611,
                 state_stale_after_s: float = 0.5):
        import zmq

        self._zmq = zmq
        self.context = zmq.Context()
        self.goal_socket = self.context.socket(zmq.PUB)
        self.goal_socket.setsockopt(zmq.LINGER, 0)
        self.goal_socket.bind(f"tcp://*:{goal_port}")
        self.state_socket = self.context.socket(zmq.SUB)
        self.state_socket.setsockopt(zmq.SUBSCRIBE, ZMQ_STATE_TOPIC)
        self.state_socket.setsockopt(zmq.CONFLATE, 1)
        self.state_socket.setsockopt(zmq.LINGER, 0)
        self.state_socket.connect(f"tcp://localhost:{state_port}")
        self.endpoint = f"goal PUB tcp://*:{goal_port}, state SUB tcp://localhost:{state_port}"

        self.stale_after_s = state_stale_after_s
        self._latest_q: np.ndarray | None = None
        self._last_state_time: float | None = None
        self.goals_published = 0

    def publish_goal(self, goal: dict):
        self.goal_socket.send(ZMQ_GOAL_TOPIC + pack_payload(goal))
        self.goals_published += 1

    def toggle_policy_action(self):
        """Bench loopback has no real g1_gear_wbc_policy on the other end --
        nothing to toggle. Kept as a no-op for interface parity with
        Ros2WbcBackend so callers don't need to branch on backend type."""
        pass

    def get_robot_q(self) -> np.ndarray | None:
        try:
            raw = self.state_socket.recv(self._zmq.NOBLOCK)
        except self._zmq.Again:
            raw = None
        if raw is not None:
            try:
                msg = unpack_payload(raw[len(ZMQ_STATE_TOPIC):])
                self._latest_q = np.asarray(msg["q"], dtype=np.float64).reshape(-1)
                self._last_state_time = time.monotonic()
            except Exception:
                pass
        return self._latest_q

    def state_staleness_s(self) -> float | None:
        if self._last_state_time is None:
            return None
        return time.monotonic() - self._last_state_time

    def health(self) -> dict[str, Any]:
        staleness = self.state_staleness_s()
        return {"backend": self.name, "endpoint": self.endpoint,
                "goals_published": self.goals_published,
                "state_staleness_s": staleness,
                "state_stale": staleness is None or staleness > self.stale_after_s}

    def close(self):
        self.goal_socket.close(linger=0)
        self.state_socket.close(linger=0)
        self.context.term()


class Ros2WbcBackend:
    """Real backend. Requires ROS 2 + rclpy and the `decoupled_wbc` package
    -- i.e. the same environment `run_g1_control_loop.py` itself runs in,
    on the G1's Orin.
    """

    name = "ros2"

    def __init__(self, state_stale_after_s: float = 0.5):
        try:
            import rclpy
            from std_msgs.msg import ByteMultiArray
        except ImportError as e:
            raise RuntimeError(
                "rclpy not available -- use --wbc-backend zmq for bench testing, "
                "or run this on the Orin with ROS 2 sourced"
            ) from e

        self._ByteMultiArray = ByteMultiArray
        rclpy.init(args=None)
        self.node = rclpy.create_node("ikea_wbc_adapter")
        self.publisher = self.node.create_publisher(ByteMultiArray, CONTROL_GOAL_TOPIC, 1)
        self.subscription = self.node.create_subscription(
            ByteMultiArray, STATE_TOPIC_NAME, self._on_state, 1
        )
        self.endpoint = f"ros2 {CONTROL_GOAL_TOPIC} / {STATE_TOPIC_NAME}"
        self.stale_after_s = state_stale_after_s
        self._last_state_time: float | None = None
        self._latest_q: np.ndarray | None = None
        self.goals_published = 0

        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._spin_thread.start()
        self._rclpy = rclpy

    def _on_state(self, msg):
        try:
            payload = bytes(b for chunk in msg.data for b in chunk)
            state = unpack_payload(payload)
            self._latest_q = np.asarray(state["q"], dtype=np.float64).reshape(-1)
            self._last_state_time = time.monotonic()
        except Exception as e:  # noqa: BLE001
            self.node.get_logger().warning(f"state decode failed: {e}")

    def publish_goal(self, goal: dict):
        payload = pack_payload(goal)
        msg = self._ByteMultiArray()
        msg.data = tuple(bytes([b]) for b in payload)
        self.publisher.publish(msg)
        self.goals_published += 1

    def toggle_policy_action(self):
        """Flip G1GearWbcPolicy.use_policy_action (default False at boot --
        NVIDIA's own teleop safe-mode, holding the current measured pose
        rather than running the trained RL balance policy). Verified by
        reading g1_decoupled_whole_body_policy.py::set_goal and
        g1_gear_wbc_policy.py::set_goal directly: this key is filtered into
        `lower_body_goal` independently of any upper-body keys, and
        InterpolationPolicy.set_goal no-ops cleanly (`if "target_time" not
        in goal: return`) when none are present -- so a standalone goal
        containing only this key is safe and does not disturb whatever
        upper-body trajectory is already in flight.

        This is a TOGGLE, not a set -- call it exactly once per intended
        transition. use_policy_action always starts False fresh on every
        control-loop launch (it's a constructor default, not persisted
        state), so exactly one call here reliably engages it.
        """
        self.publish_goal({"toggle_policy_action": True})

    def get_robot_q(self) -> np.ndarray | None:
        return self._latest_q

    def state_staleness_s(self) -> float | None:
        if self._last_state_time is None:
            return None
        return time.monotonic() - self._last_state_time

    def health(self) -> dict[str, Any]:
        staleness = self.state_staleness_s()
        return {"backend": self.name, "endpoint": self.endpoint,
                "goals_published": self.goals_published,
                "state_staleness_s": staleness,
                "state_stale": staleness is None or staleness > self.stale_after_s}

    def close(self):
        try:
            self.node.destroy_node()
            self._rclpy.shutdown()
        except Exception:
            pass


def make_backend(kind: str, **kwargs):
    if kind == "zmq":
        return ZmqWbcBackend(**kwargs)
    if kind == "ros2":
        return Ros2WbcBackend(**{k: v for k, v in kwargs.items()
                                 if k == "state_stale_after_s"})
    raise ValueError(f"unknown wbc backend {kind!r}")
