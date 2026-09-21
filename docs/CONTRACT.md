# The boundary contract

This is the only interface between your code and the robot. `boundary/` in
this repo is the authoritative implementation — **if this document and the
code disagree, the code wins.** Do not modify `boundary/`.

## Sockets

| Port | Direction | Payload |
|---|---|---|
| `:5555` | organizer PUB → you SUB | cameras |
| `:5557` | organizer PUB → you SUB | robot state |
| `:5556` | **you BIND** → organizer connects | your actions |

All three live on the robot's onboard computer. `:5556` is the inversion that
catches everyone: **you are the server for your own actions.** The whole-body
controller dials in to you.

`:5555` and `:5557` are CONFLATE sockets — newest message wins, no backlog.
You cannot fall behind on them; you can only miss frames.

## Lanes

Declare one in `manifest.yaml`. It must match what your code actually
publishes.

**`decoupled`** — you emit task-space end-effector poses; the organizer runs
inverse kinematics. One whole `(T,25)` chunk per message, `T ≤ 64`, topic
prefix `b"taskspace"`.

**`sonic`** — you emit a 64-dim latent motion token at up to 50 Hz, one row
per call, relayed byte-for-byte into the controller. No IK involved. Topic
prefix `b"pose"`, 1280-byte zero-padded JSON header then raw
little-endian buffers. `|token| ≤ 1.25`.

## The `(T,25)` row layout — fixed, do not reorder

```
[0:2]    left hand, 2 finger joints,  -1 = open, +1 = closed
[2:4]    right hand, same convention
[4:7]    left  end-effector position  (x, y, z) metres
[7:11]   left  end-effector quaternion (w, x, y, z)
[11:14]  right end-effector position
[14:18]  right end-effector quaternion (w, x, y, z)
[18:21]  navigate_cmd (vx, vy, yaw_rate)
[21]     base_height_cmd
[22:25]  torso_orientation_rpy_cmd (roll, pitch, yaw)
```

Validated before publish: dtype floating (cast to float32), all finite,
`1 ≤ T ≤ 64`, hand commands within `[-1, 1]` (1e-3 tolerance), both EE
quaternions unit-norm (1e-2 tolerance). A malformed action raises
`ActionError` and is **never published**.

> **Quaternion ordering is not checked, and this is the single most common
> silent failure.** The code verifies the norm, not the order. A wrongly
> ordered unit quaternion passes validation and produces confidently wrong
> motion. Verify `w`-first by hand.

**End-effector frame:** `left_wrist_yaw_link` / `right_wrist_yaw_link`, with
**zero tool offset**. This matches NVIDIA's reference implementation on a
byte-identical URDF. If your policy assumes a tool offset, apply it in your
own code before publishing — the organizer's value is global and will not be
changed per team.

## State — `:5557`

msgpack dict, topic `"g1_debug"`:

- `body_q` `(29,)` float32 radians, Unitree canonical G1 29-DoF order
- `base_quat` `(4,)` w-first
- `left_hand_q` / `right_hand_q` `(7,)` — **absent** on this rig. The
  competition G1 has 2-finger grippers, not 3-finger hands. Do not require
  these keys.

## Cameras — `:5555`

msgpack frame: `{"timestamps": {key: float}, "images": {key: jpeg_bytes}}`.
JPEG is BGR-encoded on the wire; `boundary/cameras.py` decodes to **RGB**.
Every image is `(480, 640, 3)` uint8.

| Key | Required | Source |
|---|---|---|
| `ego_view` | **yes** | head stereo camera, one eye |
| `ego_view_left` / `ego_view_right` | no | the two halves separately |
| `left_wrist` / `right_wrist` | no | wrist depth cameras, colour stream |

`ego_view` is the only guaranteed key — the bridge does not publish at all
until the head camera is live. **Wrist keys can be silently absent** if a
camera fails to open. If you declare them, check for them; do not assume a
missing key means anything sensible.

### How `ego_view` is produced

The head camera delivers a `3840×1080` side-by-side stereo frame. The bridge
takes the **left half** (`1920×1080`, 16:9) and **resizes** it to `640×480`
(4:3). That is an aspect-ratio squash, not a crop.

This matters if your training data was prepared differently. The contract
fixes the `480×640` output but says nothing about how to get there, so it is
stated explicitly here. Rectification is **off** by default, matching the raw
frames the reference dataset was collected on.

## Evaluator settings that affect your result

These are the organizer's choices, not inherited from any reference
implementation. They are frozen and published in advance because each one can
change an outcome. They will not change mid-series.

| Setting | Value | Why it matters |
|---|---|---|
| `--max-ik-err` | `1e-3` | Sets IK accept/reject directly |
| IK iteration cap | `200` | Too few makes converging solves look unreachable |
| `--ik-warm-start` | `current` | Seeds from measured state, so a solve depends only on `(target, body_q)` |
| Reject behaviour | hold last known-good | Rather than dropping the chunk |
| Chunk backlog | `CONFLATE` | Superseded chunks are dropped |
| `--max-chunk-age-s` | `1.0` | When a plan is too stale to act on |
| `--max-waypoints` | `16` | How much of each chunk executes |
| `--chunk-hz` | `20.0` | Scheduling rate |

`--ik-warm-start current` is not a tuning preference. Warm-starting from the
*previous solution* made results depend on message arrival order — the same 8
targets scored 7/8 accepted in one order and 3/8 reversed. Seeding from
measured state makes the metric reproducible.

## Fairness standard

If a run fails, the cause must be demonstrably yours. Failures traceable to
the organizer's stack are **no-contest**: re-run, and they do not consume an
attempt. That includes adapter crashes, network drops between the two
machines, controller faults, an e-stop fired for a non-policy reason, and any
evaluator-side configuration changing mid-series.

What does count as your failure: contract violations at `:5556`, your server
failing to load its own checkpoint or silently falling back to hold-still,
your client crashing or failing to connect for reasons inside your code, and
a policy that runs correctly but produces unreachable or task-incorrect
targets.

## Self-testing before you ship

`conformance.py` in the submission template runs your real `server.py`,
`client.py` and `transport.py` against mock cameras, mock state and a mock
controller as four separate processes, and requires a run of clean messages
with zero rejections.

A pass means your wiring is correct. It says nothing about whether your
policy is any good, and it does not exercise real hardware, real latency, or
the e-stop.
