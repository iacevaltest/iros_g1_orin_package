"""Tests for the start-up-pose seeding patch in tools/run_wbc_with_dex1.py.

No robot, no ROS 2, no WBC process, no MuJoCo: the patch is installed
against a fake `run_g1_control_loop` namespace (fake G1Env, fake
get_wbc_policy, fake robot_model) and its observable contract is checked:

  * the q handed to `robot_model.set_initial_body_pose` is the observed
    43-wide q with the 14 hand slots zeroed, taken from the FIRST valid
    observation (observe() raising, then all-zero body joints, are skipped);
  * the stock factory is called exactly once with the caller's arguments
    forwarded untouched, including the positional-argument quirk (the 4th
    positional `upper_body_joint_speed` lands on the factory's `init_time`);
  * if no valid observation arrives before the timeout the patch warns on
    stderr and falls back to stock (no set_initial_body_pose, factory still
    called with identical args);
  * --enable-waist needs no special handling: the full model q is seeded,
    so the waist slots carry their measured values.

Run from the repository root:

    python -m pytest reference/wbc_adapter/tests
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
WRAPPER_PATH = REPO_ROOT / "tools" / "run_wbc_with_dex1.py"


def load_wrapper():
    spec = importlib.util.spec_from_file_location("run_wbc_with_dex1_under_test", WRAPPER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 43-wide model layout: legs 0-11, waist 12-14, L arm 15-21, L hand 22-28,
# R arm 29-35, R hand 36-42.
ARM_IDX = list(range(15, 22)) + list(range(29, 36))
HAND_IDX = list(range(22, 29)) + list(range(36, 43))
WAIST_IDX = [12, 13, 14]


def real_looking_q() -> np.ndarray:
    q = np.zeros(43)
    q[0:12] = np.linspace(-0.3, 0.3, 12)               # legs, non-zero
    q[WAIST_IDX] = [0.05, -0.02, 0.10]                 # waist, non-zero
    q[15:22] = [-0.6, 0.4, 0.0, 1.0, 0.0, 0.3, 0.0]    # left arm
    q[29:36] = [-0.6, -0.4, 0.0, 1.0, 0.0, 0.3, 0.0]   # right arm
    q[HAND_IDX] = 0.7                                  # hands: deliberately NON-zero
    return q


class FakeRobotModel:
    num_dofs = 43

    def __init__(self):
        self.set_calls: list[np.ndarray] = []

    def get_joint_group_indices(self, name):
        return {"hands": HAND_IDX, "arms": ARM_IDX}[name]

    def set_initial_body_pose(self, q, q_idx=None):
        assert q_idx is None
        self.set_calls.append(np.array(q, dtype=np.float64))


class FakeEnv:
    """observe() replays a script: 'raise', 'zeros', or an array."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def observe(self):
        self.calls += 1
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, str) and item == "raise":
            raise AttributeError("'NoneType' object has no attribute 'shape'")
        if isinstance(item, str) and item == "zeros":
            return {"q": np.zeros(43), "dq": np.zeros(43)}
        return {"q": np.array(item, dtype=np.float64), "dq": np.zeros(43)}


def make_fake_loop_module(env_script):
    """A stand-in for run_g1_control_loop's namespace."""
    record = {"env_ctor_kwargs": None, "factory_calls": []}
    sentinel_policy = object()

    def FakeG1Env(**kwargs):
        record["env_ctor_kwargs"] = kwargs
        return FakeEnv(env_script)

    def fake_get_wbc_policy(*args, **kwargs):
        record["factory_calls"].append((args, kwargs))
        return sentinel_policy

    mod = types.SimpleNamespace(G1Env=FakeG1Env, get_wbc_policy=fake_get_wbc_policy)
    return mod, record, sentinel_policy


def drive_loop(mod, robot_model, upper_body_joint_speed=3.0):
    """Do what run_g1_control_loop.main does with these two names."""
    env = mod.G1Env(env_name="default", robot_model=robot_model, config={"x": 1}, wbc_version="v2")
    policy = mod.get_wbc_policy("g1", robot_model, {"x": 1}, upper_body_joint_speed)
    return env, policy


class SeedFromMeasuredTest(unittest.TestCase):
    def setUp(self):
        self.wrapper = load_wrapper()

    def test_seeds_first_valid_q_with_hands_zeroed_and_forwards_args(self):
        q_real = real_looking_q()
        mod, record, sentinel = make_fake_loop_module(["raise", "zeros", "zeros", q_real])
        rm = FakeRobotModel()

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            state = self.wrapper.install_seed_from_measured_patch(loop_mod=mod, timeout_s=2.0)
            env, policy = drive_loop(mod, rm, upper_body_joint_speed=3.0)

        # env captured; observe() polled past the raise and the two all-zero frames
        self.assertIs(state["env"], env)
        self.assertEqual(env.calls, 4)
        self.assertFalse(state["fell_back"])

        # seeded q: observed q with the 14 hand slots zeroed, everything else identical
        self.assertEqual(len(rm.set_calls), 1)
        expected = q_real.copy()
        expected[HAND_IDX] = 0.0
        np.testing.assert_array_equal(rm.set_calls[0], expected)
        self.assertTrue(np.all(rm.set_calls[0][HAND_IDX] == 0.0))
        np.testing.assert_array_equal(rm.set_calls[0][ARM_IDX], q_real[ARM_IDX])
        np.testing.assert_array_equal(rm.set_calls[0][WAIST_IDX], q_real[WAIST_IDX])  # waist covered
        # the seeded array is a copy, not the env's buffer
        self.assertIsNot(rm.set_calls[0], q_real)

        # stock factory called once, positional quirk preserved, nothing added
        self.assertEqual(len(record["factory_calls"]), 1)
        args, kwargs = record["factory_calls"][0]
        self.assertEqual(len(args), 4)
        self.assertEqual(args[0], "g1")
        self.assertIs(args[1], rm)
        self.assertEqual(args[2], {"x": 1})
        self.assertEqual(args[3], 3.0)          # 4th positional -> factory's init_time, untouched
        self.assertEqual(kwargs, {})
        self.assertIs(policy, sentinel)

        # one log line with the 14 arm values
        seed_lines = [l for l in out.getvalue().splitlines() if "seeded from MEASURED" in l]
        self.assertEqual(len(seed_lines), 1)
        for v in q_real[ARM_IDX]:
            self.assertIn(f"{v:+.3f}", seed_lines[0])

    def test_all_zeros_times_out_and_falls_back_to_stock_with_warning(self):
        mod, record, sentinel = make_fake_loop_module(["zeros"])
        rm = FakeRobotModel()

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            state = self.wrapper.install_seed_from_measured_patch(loop_mod=mod, timeout_s=0.1)
            env, policy = drive_loop(mod, rm, upper_body_joint_speed=3.0)

        self.assertTrue(state["fell_back"])
        self.assertIsNone(state["seeded_q"])
        self.assertEqual(rm.set_calls, [])                     # stock seed untouched
        self.assertGreaterEqual(env.calls, 2)                  # it did keep polling
        self.assertIn("WARNING", err.getvalue())
        self.assertIn("STOCK", err.getvalue())
        self.assertIn("all exactly zero", err.getvalue())

        # stock factory still called exactly once with identical args
        self.assertEqual(len(record["factory_calls"]), 1)
        args, kwargs = record["factory_calls"][0]
        self.assertEqual(args, ("g1", rm, {"x": 1}, 3.0))
        self.assertEqual(kwargs, {})
        self.assertIs(policy, sentinel)

    def test_observe_raising_forever_falls_back(self):
        mod, record, _ = make_fake_loop_module(["raise"])
        rm = FakeRobotModel()
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            state = self.wrapper.install_seed_from_measured_patch(loop_mod=mod, timeout_s=0.1)
            drive_loop(mod, rm)
        self.assertTrue(state["fell_back"])
        self.assertEqual(rm.set_calls, [])
        self.assertEqual(len(record["factory_calls"]), 1)
        self.assertIn("observe() raised AttributeError", err.getvalue())

    def test_wrong_width_q_is_not_seeded(self):
        mod, record, _ = make_fake_loop_module([np.ones(29)])
        rm = FakeRobotModel()
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.wrapper.install_seed_from_measured_patch(loop_mod=mod, timeout_s=0.1)
            drive_loop(mod, rm)
        self.assertEqual(rm.set_calls, [])
        self.assertEqual(len(record["factory_calls"]), 1)
        self.assertIn("expected (43,)", err.getvalue())

    def test_env_constructed_outside_patched_name_falls_back(self):
        mod, record, _ = make_fake_loop_module([real_looking_q()])
        rm = FakeRobotModel()
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.wrapper.install_seed_from_measured_patch(loop_mod=mod, timeout_s=0.1)
            mod.get_wbc_policy("g1", rm, {}, 3.0)          # no G1Env(...) call first
        self.assertEqual(rm.set_calls, [])
        self.assertEqual(len(record["factory_calls"]), 1)
        self.assertIn("no G1Env", err.getvalue())

    def test_wrapper_flag_defaults_on(self):
        """The pre-parser exposes --seed-from-measured / --no-seed-from-measured, default ON."""
        import argparse
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--seed-from-measured", action=argparse.BooleanOptionalAction, default=True)
        self.assertTrue(pre.parse_known_args([])[0].seed_from_measured)
        self.assertFalse(pre.parse_known_args(["--no-seed-from-measured"])[0].seed_from_measured)
        src = WRAPPER_PATH.read_text()
        self.assertIn('"--seed-from-measured", action=argparse.BooleanOptionalAction', src)
        self.assertIn("default=True", src)


if __name__ == "__main__":
    unittest.main()
