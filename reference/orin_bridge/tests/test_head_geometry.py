"""Head-camera geometry tests for the camera bridge -- no camera, no ZMQ.

Run from the repository root:

    PYTHONPATH=reference/orin_bridge python -m pytest reference/orin_bridge/tests -q
    PYTHONPATH=reference/orin_bridge python -m unittest discover -s reference/orin_bridge/tests

Needs cv2, numpy, msgpack and zmq (the bridge imports them at module load);
pyrealsense2 is optional, the bridge guards that import.

What these pin down: the training dataset's head frames are 1280x480
side-by-side, each half a 640x480 eye, no resize. The bridge must reproduce
that byte-for-byte, detect the 3840x1080 geometry that produced a horizontal
squash between 2026-08-24 and this fix, and only fall back to the resize
when HEAD_ALLOW_RESIZE_FALLBACK=1 is set explicitly.
"""
from __future__ import annotations

import ast
import contextlib
import io
import math
import os
import unittest

import numpy as np

import real_orin_cameras as bridge

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
_CALIB_PATH = os.path.join(_REPO_ROOT, "config", "head_camera_calibration.yaml")


def _synthetic_sbs_frame(width: int, height: int) -> np.ndarray:
    """Side-by-side frame whose left and right halves are distinct and
    non-uniform, so an accidental swap, crop or resample is detectable."""
    rng = np.random.default_rng(1234)
    frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    half = width // 2
    frame[:, :half, 0] = 200          # left half: blue-ish tint
    frame[:, half:, 2] = 200          # right half: red-ish tint
    return frame


def _load_calibration(path: str) -> dict:
    """Tiny parser for the flat `key: <python literal>` yaml used in config/."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, _, rest = line.partition(":")
            out[key.strip()] = ast.literal_eval(rest.strip())
    return out


class TestConstants(unittest.TestCase):
    def test_defaults_are_the_dataset_geometry(self):
        self.assertEqual(bridge.DATASET_HEAD_SIZE, (1280, 480))
        self.assertEqual(bridge.FRAME_SHAPE, (480, 640))
        # Without HEAD_WIDTH/HEAD_HEIGHT in the environment the bridge asks
        # for exactly the dataset mode.
        if "HEAD_WIDTH" not in os.environ and "HEAD_HEIGHT" not in os.environ:
            self.assertEqual((bridge.HEAD_WIDTH, bridge.HEAD_HEIGHT), (1280, 480))

    def test_rectify_defaults_to_raw(self):
        if "EGO_VIEW_RECTIFY" not in os.environ:
            self.assertFalse(bridge.EGO_VIEW_RECTIFY)


class TestSplitStereoFrame(unittest.TestCase):
    def test_native_frame_splits_into_two_640x480_halves_byte_identical(self):
        frame = _synthetic_sbs_frame(1280, 480)
        left, right = bridge.split_stereo_frame(frame)
        self.assertEqual(left.shape, (480, 640, 3))
        self.assertEqual(right.shape, (480, 640, 3))
        np.testing.assert_array_equal(left, frame[:, :640])
        np.testing.assert_array_equal(right, frame[:, 640:])
        self.assertEqual(left.tobytes(), frame[:, :640].tobytes())
        self.assertEqual(right.tobytes(), frame[:, 640:].tobytes())
        # distinct content on each side, so the halves are not confused
        self.assertFalse(np.array_equal(left, right))
        # views, not copies: no resample happened
        self.assertTrue(np.shares_memory(left, frame))
        self.assertTrue(np.shares_memory(right, frame))

    def test_prepare_eye_is_a_no_op_in_native_mode(self):
        frame = _synthetic_sbs_frame(1280, 480)
        left, right = bridge.split_stereo_frame(frame)
        bridge._resize_logged = False
        out_l = bridge.prepare_eye(left, None)
        out_r = bridge.prepare_eye(right, None)
        self.assertIs(out_l, left)
        self.assertIs(out_r, right)
        self.assertEqual(out_l.tobytes(), frame[:, :640].tobytes())
        self.assertEqual(out_r.tobytes(), frame[:, 640:].tobytes())
        self.assertFalse(bridge.resize_was_needed())


class TestGeometryMismatch(unittest.TestCase):
    def test_native_size_is_not_a_mismatch(self):
        self.assertIsNone(bridge.head_geometry_mismatch((1280, 480)))

    def test_3840x1080_is_a_mismatch(self):
        frame = _synthetic_sbs_frame(3840, 1080)
        h, w = frame.shape[:2]
        reason = bridge.head_geometry_mismatch((w, h))
        self.assertIsNotNone(reason)
        self.assertIn("3840x1080", reason)
        self.assertIn("1920x1080", reason)

    def test_realsense_sized_frames_are_rejected_as_head_camera(self):
        # the 2026-08-24 wrong-node incident
        for w, h in ((848, 480), (640, 480)):
            self.assertIsNotNone(bridge.head_geometry_mismatch((w, h)))
            self.assertFalse(bridge.probe_frame_is_head_camera(np.zeros((h, w, 3), np.uint8)))
        self.assertTrue(bridge.probe_frame_is_head_camera(np.zeros((480, 1280, 3), np.uint8)))
        # wider-than-expected is not accepted either: exactly 1280 wide
        self.assertFalse(bridge.probe_frame_is_head_camera(np.zeros((1080, 3840, 3), np.uint8)))


class TestFallbackDecision(unittest.TestCase):
    def test_native_gives_banner(self):
        mode, lines = bridge.decide_head_geometry((1280, 480), (1280, 480), allow_resize_fallback=False)
        self.assertEqual(mode, "native")
        self.assertEqual(len(lines), 1)
        self.assertIn("head camera live at 1280x480 (native side-by-side; each eye 640x480, "
                      "no resize, matches dataset)", lines[0])

    def test_mismatch_refuses_without_fallback(self):
        mode, lines = bridge.decide_head_geometry((3840, 1080), (1280, 480), allow_resize_fallback=False)
        self.assertEqual(mode, "refuse")
        text = "\n".join(lines)
        self.assertGreater(len(lines), 3, "error must be loud and multi-line")
        self.assertIn("would NOT match the dataset geometry", text)
        self.assertIn("REFUSING to publish head frames", text)
        self.assertIn("HEAD_ALLOW_RESIZE_FALLBACK=1", text)
        self.assertTrue(all("ERROR" in line for line in lines))

    def test_mismatch_falls_back_with_warning_on_every_line(self):
        mode, lines = bridge.decide_head_geometry((3840, 1080), (1280, 480), allow_resize_fallback=True)
        self.assertEqual(mode, "fallback")
        self.assertTrue(all("WARNING" in line for line in lines), lines)
        text = "\n".join(lines)
        self.assertIn("HEAD_ALLOW_RESIZE_FALLBACK=1", text)
        self.assertIn("would NOT match the dataset geometry", text)
        self.assertIn("DIAGNOSTICS ONLY", text)

    def test_fallback_path_resizes_and_logs_once(self):
        frame = _synthetic_sbs_frame(3840, 1080)
        left, right = bridge.split_stereo_frame(frame)
        self.assertEqual(left.shape, (1080, 1920, 3))
        bridge._resize_logged = False
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            out_l = bridge.prepare_eye(left, None)
            out_r = bridge.prepare_eye(right, None)
        self.assertEqual(out_l.shape, (480, 640, 3))
        self.assertEqual(out_r.shape, (480, 640, 3))
        self.assertTrue(bridge.resize_was_needed())
        self.assertEqual(err.getvalue().count("per-eye resize"), 1, "must log exactly once")
        bridge._resize_logged = False


class TestRectifyMaps(unittest.TestCase):
    def test_maps_are_built_for_a_640x480_eye(self):
        for side in ("left", "right"):
            map1, map2 = bridge.RECTIFY_MAPS[side]
            self.assertEqual(map1.shape[:2], (480, 640), side)
            self.assertEqual(map2.shape[:2], (480, 640), side)

    def test_rectified_eye_keeps_640x480(self):
        frame = _synthetic_sbs_frame(1280, 480)
        left, _ = bridge.split_stereo_frame(frame)
        out = bridge.prepare_eye(left, bridge.RECTIFY_MAPS["left"])
        self.assertEqual(out.shape, (480, 640, 3))


class TestFieldOfViewFromCalibration(unittest.TestCase):
    """Keeps the HFOV/VFOV figures quoted in docs/CONTRACT.md honest."""

    def test_yaml_intrinsics_give_about_87_by_71_degrees(self):
        calib = _load_calibration(_CALIB_PATH)
        self.assertEqual(calib["rectify_size"], [640, 480])
        k = calib["cam_matrix_left"]
        fx, fy = k[0][0], k[1][1]
        self.assertAlmostEqual(fx, 337.53, places=2)
        self.assertAlmostEqual(fy, 336.61, places=2)
        hfov = math.degrees(2 * math.atan(320 / fx))
        vfov = math.degrees(2 * math.atan(240 / fy))
        self.assertAlmostEqual(hfov, 87.0, delta=0.6)
        self.assertAlmostEqual(vfov, 71.0, delta=0.6)
        # and the bridge loaded this same file
        self.assertAlmostEqual(bridge._CAM_MATRIX_LEFT[0, 0], fx)
        self.assertAlmostEqual(bridge._CAM_MATRIX_LEFT[1, 1], fy)


if __name__ == "__main__":
    unittest.main()
