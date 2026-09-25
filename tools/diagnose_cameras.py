#!/usr/bin/env python3
"""Re-derive the camera hardware ground truth on PC2 -- serials, USB ports,
video device numbers, actual resolutions -- rather than trusting whatever
constants happen to be hardcoded in the copy of the camera bridge on disk.

Written because the copy of the bridge pasted into the evaluation chat and
the copy actually running on PC2 have already been found to disagree
(different WRIST_SERIALS mapping, different head-camera resolution
assumptions) -- so nothing about this hardware should be taken on faith
from a script's source code. This probes the hardware directly.

Step 5 probes the head camera's native 1280x480 side-by-side mode -- the
one the dataset was captured in and the only one real_orin_cameras.py will
publish from. On 2026-08-24 a RealSense node was probed under the head
camera's name, answered 848x480 / 640x480, and the session wrongly concluded
that 1280x480 was not a native mode; this step tells the two cases apart.

READ-ONLY except for one thing: it opens each camera briefly to negotiate
a stream and grab ONE frame, same as the bridge itself does at startup.
It does not touch the robot, DDS, or any motor.

    python3 diagnose_cameras.py --out ~/camera_diag

Run in the venv that has pyrealsense2 and opencv -- the camera bridge's
own env (teleimager) already has both, since it needs them itself.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

# The head camera's native side-by-side mode -- the geometry the training
# dataset was captured in (each half a 640x480 eye). Must agree with
# real_orin_cameras.py's DATASET_HEAD_SIZE.
HEAD_W, HEAD_H = 1280, 480


def section(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


def diagnose_realsense(out_dir: str):
    section("1. RealSense devices -- serial, USB port, physical path")
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("  pyrealsense2 not importable in this env -- wrong venv?")
        return {}

    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        print("  NO RealSense devices detected at all.")
        print("  Check: lsusb | grep -i intel   and the physical cable seating.")
        return {}

    info = {}
    for d in devices:
        name = d.get_info(rs.camera_info.name)
        serial = d.get_info(rs.camera_info.serial_number)
        try:
            port = d.get_info(rs.camera_info.physical_port)
        except Exception:
            port = "(unavailable)"
        try:
            usb_type = d.get_info(rs.camera_info.usb_type_descriptor)
        except Exception:
            usb_type = "?"
        try:
            fw = d.get_info(rs.camera_info.firmware_version)
        except Exception:
            fw = "?"
        print(f"  serial {serial}")
        print(f"    name       : {name}")
        print(f"    USB        : {usb_type}")
        print(f"    firmware   : {fw}")
        print(f"    phys. port : {port}")
        info[serial] = {"name": name, "port": port}

    print(f"\n  {len(devices)} RealSense device(s) found. Serials above are the")
    print("  ONLY thing that should ever populate WRIST_SERIALS -- copy them")
    print("  directly, do not retype from memory or from old chat logs.")

    section("2. Grab one frame from EACH RealSense, save it, so you can look")
    for d in devices:
        serial = d.get_info(rs.camera_info.serial_number)
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            pipeline.start(cfg)
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            color = frames.get_color_frame()
            if color:
                import cv2
                import numpy as np
                img = np.asanyarray(color.get_data())
                path = os.path.join(out_dir, f"realsense_{serial}.jpg")
                cv2.imwrite(path, img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                texture = float(gray.std())
                # RealSense auto-exposure compensates for a covered lens by
                # raising gain, so mean brightness alone can stay flat or
                # even rise while covered -- confirmed empirically on this
                # rig (both serials' brightness went UP ~1 point when one
                # lens was physically covered). Texture (std dev) is a
                # better signal: a covered lens is flat/low-detail even
                # after gain compensation; an uncovered one shows real
                # scene texture regardless of exposure level.
                flag = "  <-- LOW TEXTURE, likely COVERED" if texture < 15 else ""
                print(f"  serial {serial}: saved {path}  "
                      f"(mean brightness {img.mean():.1f}, texture/std {texture:.1f}){flag}")
            else:
                print(f"  serial {serial}: pipeline started but no color frame")
            pipeline.stop()
        except Exception as exc:
            print(f"  serial {serial}: FAILED to grab a frame -- {type(exc).__name__}: {exc}")
        time.sleep(0.3)  # let the USB bus settle before the next device

    return info


def diagnose_v4l2(out_dir: str):
    section("3. /dev/video* nodes -- name, and what actually opens")
    nodes = sorted(glob.glob("/dev/video*"),
                    key=lambda p: int("".join(filter(str.isdigit, p)) or 0))
    if not nodes:
        print("  No /dev/video* nodes at all.")
        return

    for node in nodes:
        idx = "".join(filter(str.isdigit, node))
        name_path = f"/sys/class/video4linux/video{idx}/name"
        name = open(name_path).read().strip() if os.path.exists(name_path) else "?"
        print(f"  {node}  ({name})")

    # v4l2-ctl gives real supported-format info if it's installed; skip
    # cleanly if not rather than failing the whole script over it.
    try:
        out = subprocess.run(["v4l2-ctl", "--list-devices"],
                             capture_output=True, text=True, timeout=5).stdout
        if out.strip():
            print("\n  v4l2-ctl --list-devices (groups nodes by physical camera):")
            for line in out.splitlines():
                print("   ", line)
    except FileNotFoundError:
        print("\n  (v4l2-ctl not installed -- `sudo apt install v4l-utils` for")
        print("   per-node supported-resolution listings. Not required below.)")

    section("4. Actually open each video node and see what resolution comes out")
    import cv2
    for node in nodes:
        idx = "".join(filter(str.isdigit, node))
        cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"  {node}: could not open (likely a metadata/control node, not a capture node)")
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # Ask for something absurdly large; V4L2 clamps to the actual max.
        # This is INFORMATIONAL: the max mode is not the mode the bridge
        # uses. The head camera's max is 3840x1080, and running it there is
        # exactly the mistake this tool exists to prevent (see step 5).
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, frame = cap.read()
        if ok and frame is not None:
            path = os.path.join(out_dir, f"video{idx}.jpg")
            cv2.imwrite(path, frame)
            print(f"  {node}: max negotiated {w}x{h}, actual frame {frame.shape}, "
                  f"saved {path}")
        else:
            print(f"  {node}: opened but no frame ({w}x{h} negotiated)")
        cap.release()
        time.sleep(0.2)

    section(f"5. Probe the DATASET head-camera mode ({HEAD_W}x{HEAD_H} side-by-side) on each node")
    print("  The bridge (real_orin_cameras.py) opens the head camera at exactly this")
    print("  mode and accepts a node only if the frame comes back exactly")
    print(f"  {HEAD_W} wide: each half is then a 640x480 eye, no resize, which is how")
    print("  the training dataset was captured. A RealSense node answers this")
    print("  request with 848x480 or 640x480 -- that is the wrong node, not a")
    print("  missing mode (the 2026-08-24 mistake).")
    matches = []
    for node in nodes:
        idx = "".join(filter(str.isdigit, node))
        cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, HEAD_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEAD_H)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"  {node}: no frame at {HEAD_W}x{HEAD_H}")
        elif frame.shape[1] == HEAD_W and frame.shape[0] == HEAD_H:
            path = os.path.join(out_dir, f"video{idx}_{HEAD_W}x{HEAD_H}.jpg")
            cv2.imwrite(path, frame)
            half = frame.shape[1] // 2
            diff = float(abs(frame[:, :half].astype(int) - frame[:, half:].astype(int)).mean())
            print(f"  {node}: {frame.shape[1]}x{frame.shape[0]}  <-- MATCHES the dataset mode; "
                  f"left/right halves differ by mean |diff| {diff:.1f} "
                  f"(> 0 = genuinely two eyes), saved {path}")
            matches.append(node)
        else:
            print(f"  {node}: asked {HEAD_W}x{HEAD_H}, got {frame.shape[1]}x{frame.shape[0]} "
                  f"-- NOT the head camera in its native mode (RealSense-sized = wrong node)")
        time.sleep(0.2)
    if not matches:
        print(f"\n  No node produced {HEAD_W}x{HEAD_H}. The bridge will refuse to publish head")
        print("  frames rather than resize another mode -- fix the camera before running it.")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=os.path.expanduser("~/camera_diag"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 68)
    print("Camera hardware ground truth -- PC2")
    print(f"Saving sample frames to {args.out}")
    print("=" * 68)

    rs_info = diagnose_realsense(args.out)
    diagnose_v4l2(args.out)

    section("WHAT TO DO WITH THIS")
    print(f"  1. Frames saved under {args.out} -- view them (scp back to the")
    print("     Thor, or open directly if PC2 has a display) to confirm:")
    print("       - which serial is physically the LEFT vs RIGHT wrist")
    print("         (cover one camera by hand, re-run, see which brightness drops")
    print("         -- same test preflight_sensors.py automates on the live stream)")
    print("       - the head camera: exactly one node must answer the step-5 probe")
    print(f"         with {HEAD_W}x{HEAD_H} and visibly two eyes side by side. That is")
    print("         the node the bridge will pick. Its max mode (step 4) is NOT")
    print("         what the bridge uses.")
    if rs_info:
        print(f"\n  2. Set LEFT_WRIST_SERIAL / RIGHT_WRIST_SERIAL for real_orin_cameras.py")
        print("     from the serials found in step 1 above, matched to physical side")
        print("     by the cover test -- never by guessing which one 'should' be which.")
    print("\n  3. Re-run real_orin_cameras.py -- expect the line")
    print(f"     'head camera live at {HEAD_W}x{HEAD_H} (native side-by-side; each eye 640x480,")
    print("     no resize, matches dataset)' -- then preflight_sensors.py --require-wrists.")


if __name__ == "__main__":
    main()
