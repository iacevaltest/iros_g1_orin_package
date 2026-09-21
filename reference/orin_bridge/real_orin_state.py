"""Real state publisher for the G1 -- state-only half of real_orin.py, split
out to run in the g1_wbc conda env (has unitree_sdk2py/cyclonedds) while the
camera half runs separately in teleimager (has pyrealsense2/cv2). Verbatim
logic from real_orin.py's state_thread/state_publish_loop, unchanged.

Read-only on the robot side (rt/lowstate subscribe only). Never publishes to
rt/arm_sdk or anything that could move the robot.

State :5557 -- b"g1_debug" + msgpack {body_q, base_quat}
"""
from __future__ import annotations

import sys
import threading
import time

import msgpack
import zmq

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

_latest_lock = threading.Lock()
_latest_state = {"body_q": None, "base_quat": None, "gripper_q": None}

# Dex1 gripper motors. They are NOT part of the 29-joint body vector, so the
# original real_orin.py never captured them and "did the gripper physically
# move?" was unanswerable from our logs (a team asked for exactly this on
# 2026-09-10). motor_state is 35 long on this rig and both indices report.
# Mapping used by run_wbc_with_dex1.py: q=0.0 closed, q=-5.30 open.
GRIPPER_INDEX = {"left": 31, "right": 33}


def state_thread(iface):
    def handler(msg: LowState_):
        with _latest_lock:
            _latest_state["body_q"] = [msg.motor_state[i].q for i in range(29)]
            _latest_state["base_quat"] = list(msg.imu_state.quaternion)
            n = len(msg.motor_state)
            if n > max(GRIPPER_INDEX.values()):
                _latest_state["gripper_q"] = {
                    side: {"q": msg.motor_state[i].q,
                           "dq": msg.motor_state[i].dq,
                           "tau_est": msg.motor_state[i].tau_est}
                    for side, i in GRIPPER_INDEX.items()
                }

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(handler, 10)
    print("[real_orin_state] state (rt/lowstate) subscribed")


def state_publish_loop(port: int, hz: float):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{port}")
    interval = 1.0 / hz
    print(f"[real_orin_state] publishing state on :{port}")
    prefix = b"g1_debug"
    while True:
        t0 = time.time()
        with _latest_lock:
            body_q = _latest_state["body_q"]
            base_quat = _latest_state["base_quat"]
            gripper_q = _latest_state["gripper_q"]
        if body_q is not None:
            payload = msgpack.packb({"body_q": body_q, "base_quat": base_quat,
                                     "gripper_q": gripper_q}, use_bin_type=True)
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
    state_publish_loop(5557, 50.0)
