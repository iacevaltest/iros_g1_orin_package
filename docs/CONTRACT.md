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

**`joint`** — you emit arm **joint angles** (the 7+7 values a
`robot_q_desired`-style policy predicts). One whole `(T,22)` chunk per
message, `T ≤ 64`, topic prefix `b"joint"`, plus a `b"goto"` request to
move the arms to a start pose. Same socket, same organizer adapter and same
whole-body controller as `decoupled`; the only difference is that no IK is
run — see [The `joint` lane](#the-joint-lane) below.

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

## The `joint` lane

Use it when your policy was trained on joint-space actions. Sending those
through `decoupled` means running your joints through forward kinematics,
publishing wrist poses, and having the organizer's IK regenerate joints —
and that round trip is not faithful (a 7-DoF arm on a 6-DoF target; the
solver's posture task pulls toward its seed). The controller's native
input is already joint space, so `joint` hands your angles to it directly.

Declare `joint` in `manifest.yaml`. The bench brings up the same controller
stack as for `decoupled`; the declaration tells the organizer what your
rows mean.

### The `(T,22)` row layout — fixed, do not reorder

```
[0:2]    left hand, 2 finger joints,  -1 = open, +1 = closed   (as (T,25))
[2:4]    right hand, same convention
[4:11]   left arm, 7 joint angles, radians
[11:18]  right arm, 7 joint angles, radians
[18:21]  navigate_cmd (vx, vy, yaw_rate)                       (as (T,25))
[21]     base_height_cmd                                        (as (T,25))
```

Arm order is Unitree's canonical `G1JointIndex` order — `shoulder_pitch,
shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw` —
which is exactly `body_q[15:22]` (left) and `body_q[22:29]` (right) on the
state stream. Publish in the frame you observe. There is no torso column:
the adapter never forwards one on either lane.

Wire: `b"joint"` + msgpack `{"actions": <float32 bytes>, "shape": [T, 22],
"dtype": "f32", "issued_at": <float, sender wall clock>}` — the same
envelope as `taskspace`. `boundary/actions.py`'s `JointSink` builds it;
`JointSink.make_rows(left_arm, right_arm, left_hand, right_hand, navigate,
base_height)` assembles the rows and `JOINT_SLICES` names the columns.

Validated before publish, and again by the adapter (a bad chunk is dropped
and counted, like a bad `(T,25)` chunk): dtype floating (cast to float32),
all finite, `1 ≤ T ≤ 64`, hand commands within `[-1, 1]` (1e-3
tolerance). **Joint angles are not range-checked on the wire**; the adapter
clamps them (next section) rather than dropping a chunk over a hair past a
limit.

### What the adapter does with a chunk

In order, and nothing else:

1. decode, validate; require fresh robot state (no state, no motion —
   same refusal as `decoupled`); drop the chunk if `issued_at` is older
   than `--max-chunk-age-s`;
2. **position clamp** each arm joint to the robot model's limits (below);
3. **step clamp**: the same `--max-joint-vel` / `--chunk-hz` rate limit
   the IK output gets, anchored on the freshly measured arms for the first
   row of every chunk — a row far from where the arm is becomes a bounded
   step toward it;
4. write the 14 angles into the controller's upper-body vector. That
   vector is 28 wide on this rig (7+7 arm joints plus 7+7 three-finger
   hand joints the controller's model carries); the hand slots are copied
   from the measured state and the two-finger grippers ignore them. You
   never see this vector;
5. publish the chunk as one interpolated waypoint trajectory scheduled
   from now at `--chunk-hz`, and relay row 0's hand commands to the
   grippers — identical to the `decoupled` path from this point on.

**No IK runs anywhere on this lane**, and nothing on it changes what the
`decoupled` path does with a `(T,25)` chunk.

### Limits and clamps

| Clamp | Source | Setting |
|---|---|---|
| position | the controller's own robot model: the URDF limits plus the WBC's supplemental narrowing (`shoulder_roll` is kept 0.19 rad from the torso) — the same table its joint safety monitor enforces on the real robot. Read from the model, never typed in; a 1e-3 rad margin keeps commands off the exact edge. | `--joint-lane-limits urdf` (default) |
| position, IK-parity | the above plus the two overrides the organizer's IK solver applies to itself (`elbow ≤ 1.4`, `wrist_roll` within `±0.9`). Those exist to steer a redundant IK solution, not to protect hardware, so they are **not** applied to joint-space policies unless ruled. | `--joint-lane-limits ik` |
| step | `--max-joint-vel` (1.0 rad/s) at `--chunk-hz` (20 Hz): 0.05 rad per row | as `decoupled` |

Every clamped value is counted in the adapter's `[stats]` line; the first
clamp of each joint is also logged with the value and the limit. A policy
that is clamped often is a policy commanding outside the robot's range —
that is visible in your own published rows, and it is yours.

### `goto` — move to a start pose

Wire: `b"goto"` + msgpack `{"left_arm": [7 floats], "right_arm": [7 floats],
"max_speed": <rad/s>, "hands": [left, right] (optional), "issued_at":
<float>}`. `JointSink.send_goto(left_arm, right_arm, max_speed=0.3,
hands=None)` sends it.

The adapter reads the measured arms, position-clamps the target, builds a
straight-line joint interpolation from measured to target at
`min(max_speed, --goto-max-speed [0.45], --max-joint-vel)` sampled at
`--chunk-hz`, and publishes it as **one** multi-waypoint goal through the
same step clamp → mapper → publish path. The final waypoint is exactly the
(clamped) target. `hands`, if given, is relayed to the grippers once;
omitted, the grippers stay as they are. The base keeps the last commanded
`navigate_cmd` / `base_height_cmd` (the controller's defaults if nothing was
commanded yet). A stale `goto` is dropped like a stale chunk.

It **does not block**. The adapter keeps the trajectory alive (its
keepalive re-sends the part still ahead, then holds the final pose); you
decide arrival by watching `body_q` on `:5557` —
`boundary.actions.arms_reached(body_q, left_arm, right_arm, tol_rad=0.05)`
is the check. Do not publish chunks while a `goto` is under way: `:5556` is
newest-wins, so a later chunk supersedes it, and a chunk sent in the same
instant may cause the `goto` to be dropped. That precedence is deliberate —
your policy's chunk always wins over a positioning request.

### Ownership

The whole-body controller owns `rt/lowcmd`; nothing on this lane talks to
the motors. The `joint` lane is the **only** joint-space channel into the
controller, and it enters through the same adapter, clamps and keepalive as
every other lane. The independent e-stop overrides everything on it.

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
| `--joint-lane` | `on` | Whether `b"joint"` / `b"goto"` are accepted at all |
| `--joint-lane-limits` | `urdf` | Which position limits the `joint` lane clamps to (see [Limits and clamps](#limits-and-clamps)) |
| `--goto-max-speed` | `0.45` | Ceiling on a `goto` request's speed, rad/s |

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
