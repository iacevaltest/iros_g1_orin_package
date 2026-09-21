#!/usr/bin/env python3
"""Capture correlated observation+action evidence for a team's policy.

READ-ONLY. Three plain SUB sockets, nothing published anywhere -- the
organizer's camera stream (:5555), the organizer's state stream (:5557),
and the team's own bound action stream (:5556, boundary/actions.py). Drives
nothing, safe to run any time, with or without the WBC or robot live, and
safe to run *alongside* a live session since it never publishes.

Written because "the gripper doesn't open" is a claim; a team can reasonably
ask for the actual numbers. This gives a team concrete, timestamped evidence
of what their policy actually received and actually output, rather than a
verbal description of behavior -- useful to attach to a bug report.

Decoders are the same ones `wbc_driver.py` already uses and that were
verified against the organizer's own `boundary/actions.py` contract source
(TASKSPACE_SLICES) -- nothing new to trust here.

Every run gets its own labeled, timestamped directory --
<outdir-base>/<team>_<YYYYmmdd_HHMMSS>/ -- so repeated runs, and different
teams, never overwrite each other's evidence. Under that:
    frames/<camera-key>/<unix-ts>.jpg   every frame, one subdir per camera
    <camera-key>.mp4                   compiled from those frames at the
                                        actual achieved FPS, one per camera
                                        (only with --video)
    log.jsonl                          one line per captured event, timestamped
    summary.txt                        sample counts + per-hand-column stats,
                                        printed at exit

Ctrl+C stops the capture early but still compiles whatever was captured into
video + summary -- you do not have to let the full --seconds elapse to get
usable output.

    python3 capture_evidence.py --team yourteam --video   # 180s default
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boundary_wire import TASKSPACE_TOPIC, decode_taskspace  # noqa: E402

STATE_TOPIC = b"g1_debug"


def open_sub(host: str, port: int, topic: bytes = b"", conflate: bool = False) -> zmq.Socket:
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.SUBSCRIBE, topic)
    s.setsockopt(zmq.CONFLATE, 1 if conflate else 0)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(f"tcp://{host}:{port}")
    return s


def compile_videos(frames_dir: Path, outdir: Path):
    """Stitch every camera's saved frames into an .mp4 at the actual
    achieved FPS (from real timestamps, not a guessed frame rate)."""
    import cv2
    videos_written = []
    if not frames_dir.exists():
        return videos_written
    for key_dir in sorted(p for p in frames_dir.iterdir() if p.is_dir()):
        frame_paths = sorted(key_dir.glob("*.jpg"), key=lambda p: float(p.stem))
        if len(frame_paths) < 2:
            continue
        timestamps = [float(p.stem) for p in frame_paths]
        span = timestamps[-1] - timestamps[0]
        fps = (len(frame_paths) - 1) / span if span > 0 else 10.0
        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            print(f"[capture] {key_dir.name}: first frame failed to decode, "
                  f"skipping video", file=sys.stderr)
            continue
        h, w = first.shape[:2]
        out_path = outdir / f"{key_dir.name}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        written = 0
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is not None and img.shape[:2] == (h, w):
                writer.write(img)
                written += 1
        writer.release()
        videos_written.append((key_dir.name, out_path, written, fps))
        print(f"[capture] wrote {out_path} ({written} frames @ {fps:.1f} fps)")
    return videos_written


def write_summary(outdir: Path, seconds: float, n_actions: int, n_state: int,
                   n_frames_saved: int, videos_written: list,
                   left_hand_vals: list, right_hand_vals: list):
    lines = [
        f"Capture window: {seconds:.0f}s",
        f"Action chunks captured: {n_actions}",
        f"State messages captured: {n_state}",
        f"Camera frames saved: {n_frames_saved}",
    ]
    for name, out_path, written, fps in videos_written:
        lines.append(f"Video: {out_path.name} -- {name}, {written} frames @ {fps:.1f} fps")
    if right_hand_vals:
        rh, lh = np.asarray(right_hand_vals), np.asarray(left_hand_vals)
        lines.append(f"right_hand: min={rh.min():.3f} max={rh.max():.3f} "
                     f"mean={rh.mean():.3f} std={rh.std():.4f} "
                     f"(-1=open, +1=closed, over {len(rh)} samples)")
        lines.append(f"left_hand:  min={lh.min():.3f} max={lh.max():.3f} "
                     f"mean={lh.mean():.3f} std={lh.std():.4f} "
                     f"(-1=open, +1=closed, over {len(lh)} samples)")
        if rh.std() < 1e-3:
            lines.append(f"right_hand NEVER VARIED across {len(rh)} samples -- "
                         f"pinned at {rh.mean():.3f} for the entire capture.")
        if lh.std() < 1e-3:
            lines.append(f"left_hand NEVER VARIED across {len(lh)} samples -- "
                         f"pinned at {lh.mean():.3f} for the entire capture.")
    else:
        lines.append("No action chunks captured at all -- is the team's "
                      "client actually publishing on :5556?")
    summary = "\n".join(lines)
    print("\n" + summary)
    (outdir / "summary.txt").write_text(summary + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--state-port", type=int, default=5557)
    p.add_argument("--actions-port", type=int, default=5556)
    p.add_argument("--seconds", type=float, default=180.0)
    p.add_argument("--team", required=True,
                   help="team name -- every run is written to its own "
                        "<team>_<timestamp> subdir under --outdir-base, so "
                        "repeated runs (and different teams) never overwrite "
                        "each other's evidence.")
    p.add_argument("--outdir-base", default="~/policy_evidence_capture",
                   help="base directory; the actual per-run output goes to "
                        "<outdir-base>/<team>_<YYYYmmdd_HHMMSS>/")
    p.add_argument("--frame-every-s", type=float, default=1.0,
                   help="save a camera snapshot at most this often. Ignored "
                        "when --video is set, which saves every frame so "
                        "the compiled video has no gaps.")
    p.add_argument("--video", action="store_true",
                   help="save every camera frame (not throttled) and compile "
                        "one .mp4 per camera at the end, at the actual "
                        "achieved FPS -- lets you actually watch what each "
                        "camera saw during the run, not just spot-check "
                        "still frames.")
    args = p.parse_args()

    run_id = f"{args.team}_{time.strftime('%Y%m%d_%H%M%S')}"
    outdir = Path(os.path.expanduser(args.outdir_base)) / run_id
    frames_dir = outdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "log.jsonl"

    # Camera frames carry no topic prefix (unlike state/actions) -- same
    # convention preflight_sensors.py and diagnose_cameras.py already rely on.
    cam_sub = open_sub(args.host, args.camera_port, topic=b"", conflate=True)
    state_sub = open_sub(args.host, args.state_port, topic=STATE_TOPIC, conflate=True)
    # NOT conflated: for evidence we want every chunk the policy actually
    # emitted, not just whatever's newest when we happen to poll -- unlike
    # wbc_driver.py's live control path, this capture has no real-time
    # deadline to protect, so there's no reason to drop any of them.
    act_sub = open_sub(args.host, args.actions_port, topic=b"", conflate=False)

    print("[capture] READ-ONLY -- three SUB sockets, publishes nothing, drives nothing")
    print(f"[capture] camera :{args.camera_port}  state :{args.state_port}  "
          f"actions :{args.actions_port}")
    print(f"[capture] writing to {outdir}")

    right_hand_vals, left_hand_vals = [], []
    gripper_seen = [0]
    n_actions = n_state = n_frames_saved = 0
    last_frame_saved = 0.0
    t_end = time.time() + args.seconds
    interrupted = False

    logf = open(log_path, "w")
    try:
        while time.time() < t_end:
            now = time.time()

            try:
                blob = cam_sub.recv(zmq.NOBLOCK)
                if args.video or now - last_frame_saved >= args.frame_every_s:
                    msg = msgpack.unpackb(blob, raw=False)
                    images = msg.get("images", {})
                    for key, jpg in images.items():
                        key_dir = frames_dir / key
                        key_dir.mkdir(parents=True, exist_ok=True)
                        (key_dir / f"{now:.6f}.jpg").write_bytes(jpg)
                        n_frames_saved += 1
                    if images:
                        last_frame_saved = now
                        logf.write(json.dumps(
                            {"t": now, "type": "camera", "keys": sorted(images.keys())}
                        ) + "\n")
                        print(f"[{now:.3f}] CAMERA  keys={sorted(images.keys())}")
            except zmq.Again:
                pass
            except Exception as exc:
                print(f"[capture] camera decode error: {exc}", file=sys.stderr)

            try:
                blob = state_sub.recv(zmq.NOBLOCK)
                msg = msgpack.unpackb(blob[len(STATE_TOPIC):], raw=False)
                n_state += 1
                body_q = msg.get("body_q")
                # 2026-09-10: also record the Dex1 finger angles (motors 31/33).
                # They are not in the 29-joint body vector, so "was the gripper
                # commanded to close, and did it physically move?" could not be
                # answered from an earlier capture at all. real_orin_state.py
                # now publishes them under "gripper_q" (absent -> None, so an
                # older publisher still works).
                gripper_q = msg.get("gripper_q")
                if gripper_q is not None:
                    gripper_seen[0] += 1
                logf.write(json.dumps({"t": now, "type": "state", "body_q": body_q,
                                       "gripper_q": gripper_q}) + "\n")
                arms = (np.round(np.asarray(body_q[15:29], dtype=float), 3)
                        if body_q is not None and len(body_q) >= 29 else None)
                print(f"[{now:.3f}] STATE   arms(L+R, 14)={list(arms) if arms is not None else None}")
            except zmq.Again:
                pass
            except Exception as exc:
                print(f"[capture] state decode error: {exc}", file=sys.stderr)

            try:
                blob = act_sub.recv(zmq.NOBLOCK)
                if blob.startswith(TASKSPACE_TOPIC):
                    chunk = decode_taskspace(blob)
                    row0 = chunk.actions[0]
                    left_hand_vals.append(float(row0[0]))
                    right_hand_vals.append(float(row0[2]))
                    n_actions += 1
                    logf.write(json.dumps({
                        "t": now, "type": "action", "issued_at": chunk.issued_at,
                        "shape": list(chunk.actions.shape),
                        "row0": chunk.actions[0].tolist(),
                        # 2026-09-03: full chunk, not just row0. The adapter
                        # IK-solves every row individually (~15-17/chunk), and
                        # a real safety violation traced to this run couldn't
                        # be fully diagnosed offline afterward because only
                        # row0 was ever captured -- the intra-chunk trigger
                        # (if that's where it was) was invisible. Logging the
                        # whole array so the next one is actually replayable
                        # at the resolution the adapter itself operates at.
                        "rows": chunk.actions.tolist(),
                    }) + "\n")
                    print(f"[{now:.3f}] ACTION  shape={list(chunk.actions.shape)} "
                          f"issued_at={chunk.issued_at:.3f} | "
                          f"left_hand={row0[0]:.3f} right_hand={row0[2]:.3f} "
                          f"(-1=open,+1=closed) | "
                          f"left_pos={np.round(row0[4:7], 3)} "
                          f"right_pos={np.round(row0[11:14], 3)} | "
                          f"left_quat={np.round(row0[7:11], 3)} "
                          f"right_quat={np.round(row0[14:18], 3)}")
            except zmq.Again:
                pass
            except Exception as exc:
                print(f"[capture] action decode error: {exc}", file=sys.stderr)

            time.sleep(0.005)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[capture] stopped early (Ctrl+C) -- compiling whatever was "
              "captured so far, not discarding it.")
    finally:
        logf.close()

    # Always compile + summarize, whether the run finished naturally or was
    # Ctrl+C'd early -- partial evidence is still evidence.
    videos_written = compile_videos(frames_dir, outdir) if args.video else []
    write_summary(outdir, args.seconds, n_actions, n_state, n_frames_saved,
                  videos_written, left_hand_vals, right_hand_vals)
    if gripper_seen[0]:
        print(f"[capture] measured Dex1 finger angle recorded in {gripper_seen[0]} "
              f"state sample(s) (motors 31/33)")
    else:
        print("[capture] WARNING: no 'gripper_q' in any state message -- the "
              "state publisher predates the 2026-09-10 change, so measured "
              "finger angle will NOT be in this capture", file=sys.stderr)
    print(f"\n[capture] wrote {log_path}, {outdir / 'summary.txt'}, "
          f"and {n_frames_saved} frame(s) under {frames_dir}")


if __name__ == "__main__":
    main()
