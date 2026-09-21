"""Map IK arm solutions into the WBC's actual upper-body joint vector.

Do not guess this width. Queried from NVIDIA's own robot model:

    waist_location="lower_body"            upper_body = 28   <- default
    waist_location="upper_body"            upper_body = 31
    waist_location="lower_and_upper_body"  upper_body = 31

    left_arm = 7, right_arm = 7, waist = 3

28 is **14 arm joints + 14 hand joints** — the model is
`g1_29dof_with_hand.urdf`, so the upper-body group always carries Dex3-style
hand joints. `--no-with-hands` does NOT change this; it was tried and the
width stayed 28.

Getting this wrong is not a soft failure. A width mismatch propagates into
`InterpolationPolicy.schedule_waypoint` and raises

    ValueError: operands could not be broadcast together with shapes (32,) (18,)

which **kills the control loop process**. On the real robot the control loop
is the balance controller, so crashing it drops the robot. Verified against
a live loop in sim: sending a 14-wide vector took the WBC down every time.

The safe construction, implemented here: read the WBC's CURRENT upper-body
pose out of its own published state, copy it, and overwrite ONLY the arm
slots with the IK solution. Hand joints (and waist, when it lives in the
upper body) pass through at their measured values. That is correct for a
Dex1-1 rig, which has no Dex3 joints to command, and it is automatically
right for any width the WBC happens to be configured for.
"""
from __future__ import annotations

import numpy as np


class UpperBodyMapper:
    """Builds WBC-shaped upper-body waypoints from per-arm IK solutions."""

    def __init__(self, waist_location: str = "lower_body"):
        from decoupled_wbc.control.robot_model.instantiation.g1 import (
            instantiate_g1_robot_model,
        )
        self.model = instantiate_g1_robot_model(waist_location=waist_location)
        self.upper_idx = np.asarray(self.model.get_joint_group_indices("upper_body"))
        self.width = len(self.upper_idx)

        pos = {int(q): i for i, q in enumerate(self.upper_idx)}
        self.arm_slots = {}
        for side in ("left", "right"):
            arm_q = self.model.get_joint_group_indices(f"{side}_arm")
            missing = [int(q) for q in arm_q if int(q) not in pos]
            if missing:
                raise RuntimeError(
                    f"{side}_arm joints {missing} are not inside the upper_body "
                    "group; cannot map IK output safely."
                )
            self.arm_slots[side] = np.asarray([pos[int(q)] for q in arm_q])

        self.waist_slots = None
        waist_q = self.model.get_joint_group_indices("waist")
        if all(int(q) in pos for q in waist_q):
            self.waist_slots = np.asarray([pos[int(q)] for q in waist_q])

    def describe(self) -> str:
        w = "yes" if self.waist_slots is not None else "no (waist is lower-body)"
        return (f"upper_body width={self.width}, left_arm slots="
                f"{self.arm_slots['left'].tolist()}, right_arm slots="
                f"{self.arm_slots['right'].tolist()}, waist in upper_body: {w}")

    def current_upper_body(self, q_full: np.ndarray) -> np.ndarray:
        """Slice the WBC's own upper-body pose out of its full q vector."""
        q_full = np.asarray(q_full, dtype=np.float64).reshape(-1)
        if q_full.shape[0] <= int(self.upper_idx.max()):
            raise ValueError(
                f"robot state q has {q_full.shape[0]} entries but upper_body "
                f"indexes up to {int(self.upper_idx.max())}"
            )
        return q_full[self.upper_idx].copy()

    def build_waypoint(self, q_full: np.ndarray, q_left_arm: np.ndarray,
                       q_right_arm: np.ndarray) -> np.ndarray:
        """Current upper-body pose with ONLY the arm slots replaced.

        Hands (and waist, if it lives here) keep their measured values --
        we never invent commands for joints this rig does not have.
        """
        out = self.current_upper_body(q_full)
        out[self.arm_slots["left"]] = np.asarray(q_left_arm, dtype=np.float64)
        out[self.arm_slots["right"]] = np.asarray(q_right_arm, dtype=np.float64)
        return out
