# Running Your Submission Against the Real Whole-Body Controller (WBC)

**Who this is for:** teams whose submission is just a policy — a Thor
policy-server container + a PC2/Orin-side policy-client container, built
from the organizer's `ikea_iros_submit` template, with **no whole-body or
lower-body controller of your own.** For you, balance and motor control
are entirely the organizer's real WBC (NVIDIA's `GR00T-WholeBodyControl`)
— your job stops at publishing task-space actions onto the boundary
contract.

**If your submission ships its own WBC/LBC** (a full custom whole-body or
lower-body controller replacing NVIDIA's), this doc doesn't describe your
setup — talk to the organizers directly about how your pipeline connects
to the real robot; nothing below applies as written.

This is a consolidation of everything learned running teams live so far
this program — see §9 for where each piece came from if you want the full
detail behind a given claim.

---

## 1 · The big picture

```
 YOUR Thor container ──:8765──▶ YOUR PC2-side client container
  (policy server)                (policy client, binds :5556 — boundary/actions.py)
                                        │
                                        ▼
                              wbc_driver.py  ← YOU run this, on PC2
                                        │
     decoupled lane: IK ─▶ joint-space goal ─▶ ROS2 ControlPolicy/upper_body_pose
     sonic lane:      relay protocol-v4 pose frames ─▶ gear_sonic_deploy
                                        ▼
                                 real G1 motors
```

Two more things feed **your** client, published by a camera/state bridge
you also run yourself on PC2 (not mocks, once you're live):
- `:5555` — real head/wrist camera streams, in `boundary/cameras.py`'s wire
  format.
- `:5557` — real robot state (`body_q`, `base_quat`, optional hand state),
  in `boundary/states.py`'s wire format.

**There is no organizer intervention at runtime — you run the whole
pipeline yourself, on both machines.** The organizer provides the code
(`wbc_driver.py`, the camera/state bridge, NVIDIA's WBC) and the machines
(Thor + the robot's onboard computer); you SSH in and
drive every step in §4 yourself, end to end, including the real WBC and
the adapter — not just your own two containers. The point of this
architecture is that **your own containers don't change** between bench
onboarding and the real robot — same images, same run commands, same
ports. Only what's on the other end of `:5555`/`:5556`/`:5557` changes
(mocks → real WBC/cameras/state), and standing that up is now also your
job, using the provided scripts.

## 2 · Which lane are you on?

Declared in your `manifest.yaml` — this determines what `wbc_driver.py`
does with your output once you run it:

| Lane | Your output | What happens to it |
|---|---|---|
| **`decoupled`** | one `(T,25)` task-space pose chunk per message | the adapter's IK solves it into joint targets, sent to NVIDIA's Decoupled WBC over ROS2 |
| **`joint`** | one `(T,22)` chunk of hand commands + 7+7 arm joint angles per message (plus `goto` start-pose requests) | no IK: the angles are position- and rate-clamped and sent to the same Decoupled WBC as `decoupled`, through the same `wbc_driver.py --lane decoupled` run |
| **`sonic`** | `motion_token (T,64)` + hand joints at 50Hz | relayed byte-for-byte (no translation) into `gear_sonic_deploy`, NVIDIA's C++/TensorRT deploy binary |

If you're `decoupled`: the WBC runs **no IK of its own** in its control
loop — it just interpolates between the joint targets it's handed. All the
IK happens upstream, in `wbc_driver.py`, using your published
end-effector pose.

**Which lane for which policy.** A policy trained on wrist poses
(end-effector position + quaternion actions) is `decoupled`. A policy
trained on `action.robot_q_desired`-style joint actions — the arm joint
angles themselves — is `joint`: publish the angles you predict, in the
`body_q` order you observed them, and nothing re-solves them. Pushing
joint-space actions through `decoupled` (FK on your side, IK on ours)
is not faithful: a 7-DoF arm on a 6-DoF target, with the solver's posture
task pulling toward its seed. Both lanes run through the same
`wbc_driver.py --lane decoupled` process and the same WBC; `joint` is on by
default there (`--joint-lane on`) and adds `--joint-lane-limits` and
`--goto-max-speed` (§9). The `joint` lane's contract, clamps and the `goto`
request are in `docs/CONTRACT.md`, "The `joint` lane".

## 3 · Pre-flight — things that are YOUR responsibility to get right

These have each been found wrong at least once this program, always
discovered live rather than caught in advance. Check every one of these
against your own submission before your slot, not after something looks
off in the video:

- **End-effector tool-offset convention.** The organizer's IK is pinned to
  **zero tool offset** (raw `wrist_yaw_link` origin) by default — this is
  a fixed, global setting, never changed per team. If your checkpoint was
  trained against a different EE point (e.g. `wrist_yaw` + some offset
  along an axis), **you must account for that in what you publish** — the
  point your `(T,25)` action represents must be the zero-offset point,
  even if your model's *state input* used the real offset internally. At
  least one team's IK accept rate went from ~43% to 100% purely by fixing
  this on their side; it will look exactly like "bad IK" or "policy not
  tracking well" from the outside if you don't.
- **Camera key → role mapping.** Confirm what your own `camera_keys` code
  actually declares (not just your docs) matches what your model expects
  for each slot (head stereo/mono, left wrist, right wrist). A mismatch
  here produces "the arm tracked fine but the task failed" — nothing in
  IK or safety would ever catch a vision-input swap.
- **Gripper starting pose.** If your policy expects a specific (possibly
  asymmetric) starting hand state matching your training data's frame 0,
  say so — don't assume default open/open is fine.
- **`base_height_cmd` / `torso_rpy` convention.** Most teams publish `0`
  for both, meaning "no override, the WBC keeps authority" — **not** "hold
  a literal zero target." Confirm which one your policy means.
- **Row cadence.** The adapter owns interpolation at a fixed 50Hz
  regardless of your own chunk-generation rate — don't assume the
  reference `client.py`'s internal re-query rate (20Hz) is the wire
  cadence; it isn't.
- **Quaternion order.** `(T,25)` quaternions must be **unit-norm AND
  w-first**. The boundary checks the norm but not the ordering — a
  scalar-last quaternion passes contract validation silently and then
  produces wrong poses downstream.
- **Python-version gotchas if your Orin/PC2-side image runs genuine Python
  3.8** (the real G1 base, `l4t-base:35.3.1`): the reference
  `components/transport.py`'s `ping_interval`/`ping_timeout` kwargs and
  `client.py`'s `ThreadPoolExecutor(cancel_futures=...)` both require
  Python 3.9+ and will crash on cp38 — guard both behind an
  `inspect.signature()` feature-check. Several teams have hit this; it's
  template-inherited, not a bug you introduced, but it's still yours to
  patch (`components/` is team-owned).

## 4 · The exact command sequence

**You run every step below yourself** — SSH into PC2 for steps 0–3 and 7
using the access details issued to your team, and launch your own two
containers for steps 4–5. Nobody else drives any part of this. **Order
matters — the PC2 pieces (steps 0–3) come up first, your Thor container
comes up last, immediately before going live** (see the explanation after
the command blocks for why).

### Step 0 — environment **[PC2]**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate g1_wbc
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export CYCLONEDDS_HOME=~/cyclonedds/install
export CYCLONEDDS_URI=~/cyclonedds_ws/cyclonedds.xml
export PYTHONPATH=$HOME/GR00T-WholeBodyControl
export LD_PRELOAD=$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib/libgomp.so.1:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib/libc10.so:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so

# confirm nothing already owns rt/lowcmd before starting anything
ps -eo pid,args | grep -E "g1_policy_bridge|run_g1_control_loop|g1_deploy_onnx_ref|gear_sonic_deploy" | grep -v grep
```

### Step 1 — camera/state bridge **[PC2, two terminals]**

```bash
# Terminal A1 — camera half (needs pyrealsense2/cv2 — different env)
conda activate teleimager
python ~/real_orin_cameras.py &

# Terminal A2 — state half (needs unitree_sdk2py/cyclonedds — different env)
conda activate g1_wbc
python ~/real_orin_state.py &
```
This is what starts publishing real data on `:5555`/`:5557` — before this,
your client would just be talking to mocks.

### Step 2 — the real WBC **[PC2, Terminal B]**

```bash
conda activate g1_wbc
cd ~/GR00T-WholeBodyControl
python3 ~/wbc_adapter/deploy/run_wbc_with_dex1.py \
  --interface eth0 --no-with-hands   # add --enable-waist here iff your policy needs the 31-wide (waist) upper-body vector
```
`run_wbc_with_dex1.py` (`tools/run_wbc_with_dex1.py` in this repo) is a
thin wrapper around NVIDIA's `run_g1_control_loop.py`: it passes every
flag through, adds the Dex1 gripper writes (motors 31/33) that the stock
controller does not know about, and caps the controller's upper-body
interpolator at `--upper-body-joint-speed 3.0` rad/s (stock default is
1000, i.e. off). Running the stock entrypoint directly leaves the grippers
dead and the interpolator uncapped. `--interface` defaults to `sim` — set
the real network interface explicitly here, or nothing reaches the motors.

**The controller moves the arms when it starts.** Its upper-body
interpolator is seeded with a fixed rest pose (`shoulder_roll` ±0.2 rad,
every other arm joint 0), and a 2 s ramp carries each arm joint from
wherever it is to that pose at full stiffness, before the adapter or any
team code is connected. Launch it with the arms clear of the table and
anything else within reach, wait for the ramp to finish, and only then
position the robot for a stage. (Seeding the start pose from the measured
joints instead is an organizer change tracked separately.)

The controller must reach a stable idle/hold state (confirmable via
`ros2 topic hz /G1Env/env_state_act`) before Step 3 starts.

### Step 3 — the adapter, dry-run first **[PC2, Terminal C]**

```bash
conda activate g1_wbc
cd ~/wbc_adapter
python wbc_driver.py --lane <decoupled|sonic> --state-source boundary
  # NOT --live yet — this is what will dial into YOUR bound :5556 once you're up
```

### Step 4 — YOUR Thor container **[YOU]**

```bash
docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --network host \
  <your Thor image> <your documented args>
```
Exactly your own `INSTRUCTIONS.md` run command — nothing organizer-specific
added here.

### Step 5 — YOUR PC2/Orin-side client container **[YOU]**

```bash
docker run --network host <your Orin/client image> <your documented args>
```
This binds `:5556`, which Step 3's adapter is already listening on, and
subscribes to the now-real `:5555`/`:5557` from Step 1.

### Step 6 — going live **[PC2]**

```bash
conda activate g1_wbc
cd ~/wbc_adapter
python wbc_driver.py --lane <decoupled|sonic> --state-source boundary \
  --live --engage-policy
```
**Run this yourself, as a human at the keyboard — not from a script or
agent — with your e-stop operator staffed and ready.** This has been
established practice throughout the program: real robot, real risk, a
person's hand on the actual command. `--engage-policy` (added 2026-09-15)
is required on the `ros2` WBC backend: without it, the WBC's lower body
silently stays in a **safe teleop hold** (current measured joint position,
not the trained balance policy) while every other signal — WBC up, 50Hz
topics, adapter reporting 0 rejected — looks completely healthy. Confirm
engagement independently by decoding
`/ControlPolicy/lower_body_policy_status` yourself, not just by trusting
the adapter's log line.

### Why the order matters

Your Thor container runs its own internal stage sequence on a fixed clock
the moment it starts, and it does **not** wait for a client to connect. If
you start it too early relative to PC2 being ready, your useful policy
stages can burn through unsupervised before your own client ever connects
— you'd see misleadingly low accept numbers from message 1 that look like
a dead policy but aren't. This is also why, if you restart anything
mid-session on either machine, the other side's live connection may drop
ungracefully — expect to need a restart on that side too, not an automatic
reconnect.

## 5 · Reading your own results correctly

- **"Published: N, rejected: 0" from the adapter's logs does not mean the
  WBC executed your actions.** This exact phrasing has looked completely
  healthy while the WBC was doing nothing with it, more than once. If
  your run doesn't look right, ask what the WBC's own log/state shows, not
  just the adapter's publish count.
- **A safety violation during your run is the safety layer working, not
  a bug in your policy.** The WBC enforces a hard joint-velocity limit
  independent of anything you publish. If it trips, that's expected
  behavior, not something to be alarmed about — `wbc_driver.py`'s own
  clamp (`--max-joint-vel`, documented default 2.0 rad/s as of this
  program) sits well under it as a first line of defense.
- **A `decoupled`-lane submission has zero visibility into or control over
  joint-space timing, velocity, or IK feasibility once it leaves your
  client** — that entire translation happens inside `wbc_driver.py`
  between your client and the WBC. If something looks like erratic
  motion, the first question is always whether your own raw `(T,25)`
  output was actually changing frame-to-frame during that window
  (checkable directly from your own published values) before assuming
  it's your policy's fault.
- **Task success is a separate axis from safety and IK tracking.** A run
  can complete every stage with zero safety violations and high IK accept
  and still not complete the task — accept-rate only measures "did we
  solve the IK for the pose you asked for," never "was that pose useful."
  Check your own raw gripper channel (`(T,25)` columns `[0:2]`/`[2:4]`)
  directly against the relevant stage — a gripper that never closes during
  a pick stage is visible straight from your own published data, no video
  needed to detect it.

## 6 · Common submission-side pitfalls, seen across multiple teams

- **Orin/PC2 base image built on R36 instead of the real G1's R35.3.1**
  (`nvcr.io/nvidia/l4t-jetpack:r35.3.1` / `l4t-base:35.3.1`). An R36 image
  is untested on real hardware and must be rebuilt.
- **Gated Hugging Face backbones** (e.g. checkpoints that load a separate
  `nvidia/Cosmos-Reason2-2B`-class backbone) need to be bundled as a
  second read-only-mounted weight dir + `HF_HUB_OFFLINE=1`, or you need to
  arrange authenticated access at every server startup — not just build
  time. If your `config.json` names a separate backbone repo, check this
  before your slot.
- **`decoupled` lane needs `scipy`** for FK — confirm it's baked into your
  Thor image, not just present on a dev machine.
- **`--gpus all` is often unimplemented on Jetson** — use `--runtime
  nvidia` in your documented run command.
- **`NVIDIA_DISABLE_REQUIRE=1`** is usually needed to work around a
  driver-version guard when your container is built against newer CUDA
  than the flashed L4T reports — include it explicitly in your documented
  launch command, don't assume it's implied.

## 7 · If something looks wrong during your slot

- Check the WBC's own log (Terminal B) for the expected stage/state — not
  just the adapter's stats in Terminal C.
- Work out whether it was a safety-layer trip (expected, not a bug) vs. a
  genuine connection issue — check the WBC log for a joint-limit message
  around the same moment.
- If accept rate is near-zero from the very first message, check whether
  your Thor container came up before PC2 was ready (§4) — that alone
  produces this symptom.
- If the gripper never seems to close, check your own raw published
  gripper channel first — `wbc_driver.py`'s gripper path is a verified
  pure relay (raw values in, raw values out), so if your own output never
  commands a close, that's on your side, not the adapter's.

## 8 · Machine access and network

You need direct SSH access to **both** Thor and the robot's onboard
computer to run everything in §4 yourself.

**Hostnames, addresses and login details are issued to your team
separately** — they are specific to the event rig and are not in this
repository. Ask for them before your slot, not during it.

Two things worth knowing in advance:

- No `sudo` is required for anything in this runbook. If you think you
  need it, ask first — it usually means something else is wrong.
- The robot's onboard computer may have a hostname unrelated to its
  login account. Use the username you were given, not the hostname.

Everything else in §4 is `${THOR_HOST}` / `${ROBOT_HOST}` substitution.

---

## 9 · CLI reference

Generated from the source of each script. Run any of them with
`--help` for the authoritative, current version.

### `tools/camera_live_view.py`

| Flag | Default | Description |
|---|---|---|
| `--camera-host` | `(dynamic)` | Robot onboard computer IP -- where real_orin.py publishes :5555 (or set ROBOT_HOST) |
| `--camera-port` | `5555` |  |
| `--port` | `8081` | local HTTP port to serve on |
| `--bind` | `'0.0.0.0'` |  |

### `tools/capture_evidence.py`

| Flag | Default | Description |
|---|---|---|
| `--host` | `'127.0.0.1'` |  |
| `--camera-port` | `5555` |  |
| `--state-port` | `5557` |  |
| `--actions-port` | `5556` |  |
| `--seconds` | `180.0` |  |
| `--team` |  | team name -- every run is written to its own <team>_<timestamp> subdir under --outdir-base, so repeated runs (and different teams) never overwrite each other's evidence. |
| `--outdir-base` | `'~/policy_evidence_capture'` | base directory; the actual per-run output goes to <outdir-base>/<team>_<YYYYmmdd_HHMMSS>/ |
| `--frame-every-s` | `1.0` | save a camera snapshot at most this often. Ignored when --video is set, which saves every frame so the compiled video has no gaps. |
| `--video` | `flag` | save every camera frame (not throttled) and compile one .mp4 per camera at the end, at the actual achieved FPS -- lets you actually watch what each camera saw during the run, not just spot-check still frames. |

### `tools/diagnose_cameras.py`

| Flag | Default | Description |
|---|---|---|
| `--out` | `(dynamic)` |  |

### `tools/diagnose_dex1.py`

| Flag | Default | Description |
|---|---|---|
| `--iface` | `None` | robot DDS interface (192.168.123.x side) |
| `--seconds` | `6.0` |  |

### `tools/preflight_orin.py`

| Flag | Default | Description |
|---|---|---|
| `--repo` | `'~/GR00T-WholeBodyControl'` |  |
| `--interface` | `None` | the real DDS network interface, e.g. eth0 |
| `--actions-host` | `'127.0.0.1'` |  |
| `--actions-port` | `5556` |  |

### `tools/preflight_sensors.py`

| Flag | Default | Description |
|---|---|---|
| `--host` | `'127.0.0.1'` |  |
| `--camera-port` | `5555` |  |
| `--state-port` | `5557` |  |
| `--seconds` | `3.0` |  |
| `--require-wrists` | `flag` | REQUIRED for any team whose server declares wrist camera_keys. Turns a missing wrist camera into a hard FAIL instead of a warning. |
| `--require-stereo` | `flag` | REQUIRED for any team whose server declares ego_view_left/ego_view_right camera_keys Turns a missing stereo key into a hard FAIL instead of a warning. |
| `--save-frames` | `None` | also write the actual decoded ego_view/left_wrist/right_wrist JPEGs to this dir, for visual inspection -- pass/fail on shape and brightness cannot catch a bad crop; looking at the pixels can. |

### `tools/run_wbc_with_dex1.py`

| Flag | Default | Description |
|---|---|---|
| `--dex1-port` | `5599` |  |
| `--dex1-max-speed` | `2.0` | rad/s of raw Dex1 motor q; full stroke is ~5.3 |
| `--dex1-verbose` | `flag` |  |
| `--no-dex1` | `flag` | run completely stock, no gripper injection |

### `reference/wbc_adapter/wbc_driver.py`

| Flag | Default | Description |
|---|---|---|
| `--lane` |  | must match the team's manifest.yaml |
| `--actions-host` | `'127.0.0.1'` | where the team's client bound :5556 (its own host) |
| `--actions-port` | `5556` |  |
| `--live` | `flag` | actually drive the robot. Default: dry-run. |
| `--engage-policy` | `flag` | 2026-09-14: send {'toggle_policy_action': True} once, as soon as the WBC's own state topic is confirmed live. Without this, G1GearWbcPolicy.use_policy_action stays at its constructor default (False) for the entire run -- NVIDIA's own teleop safe-mode, which holds the robot's current measured joint position every tick rather than running the trained RL balance policy. Looks completely healthy (WBC up, topics ticking at 50Hz, this adapter publishing with 0 rejected) while never actually engaging balance -- found by decoding /ControlPolicy/lower_body_policy_status directly, since nothing else in the stack surfaces this. Only meaningful with --live --wbc-backend ros2; ignored otherwise. This is a TOGGLE: sent exactly once per run, never repeated, since a second send would disengage it again. |
| `--verbose` | `flag` |  |
| `--wbc-backend` | `'ros2'` | ros2 = the real Decoupled WBC; zmq = bench loopback |
| `--upper-body-from-model` | `flag` | force querying decoupled_wbc's robot model for the upper-body layout (implied by --wbc-backend ros2) |
| `--enable-waist` | `flag` | set iff run_g1_control_loop.py runs with waist in the upper-body group (upper-body vector width 31 vs 28: 7+7 arm joints, 7+7 hand-model slots, plus 3 waist joints when enabled) |
| `--chunk-hz` | `20.0` | cadence the (T,25) rows are meant to play out at |
| `--max-waypoints` | `16` |  |
| `--max-joint-vel` | `1.0` | rad/s cap on how fast any commanded arm joint may move between scheduled waypoints, regardless of what the raw IK solution implies. Margin under the WBC's own real-hardware joint safety monitor (+-6 rad/s, joint_safety.py) -- IK has no notion of the schedule's timing, so an unthrottled solve far from the current pose can exceed that limit and trip a hard shutdown. 0 disables. 2026-08-25: lowered from 4.0 -- across four live violations this session, ACTUAL measured joint velocity reached up to 3.0x this commanded cap (12.03 rad/s actual vs a 4.0 cap), a real gap between commanded and realized motion this adapter doesn't fully explain yet (clamp-reference staleness fixes reduced but did not eliminate it). 2026-09-03: lowered again, 2.0 -> 1.0. One team's session tripped right_elbow_joint at -7.153 (WBC's own reported figure) to -7.706 rad/s (independently recomputed from capture_evidence.py's raw body_q samples, ~20ms apart) against a 2.0 commanded cap -- ~3.6-3.9x amplification, WORSE than the 3.0x this comment already flagged as unexplained, not better. Ruled out: the already-documented unclamped path (WBC's own >1.0s teleop-timeout injecting an unclamped safe goal) -- no 'Teleop mode timeout' line anywhere near this violation in the WBC log, so this went through our own solve->clamp->publish path, not around it. target_time spacing was also checked and matches the dt this clamp assumes (times = t_base + (i+1)/chunk_hz, same chunk_hz used for max_step), so it isn't a simple units/timing mismatch either. Real body_q samples show the joint smoothly RISING for ~360ms right before the trip, then reversing hard within one ~23ms sample -- consistent with (not proven as) a position-only clamp saying nothing about the arm's existing momentum when a reversal is commanded, so tracking a same-magnitude position step in the opposite direction of travel can demand more real velocity than the step size alone implies. Since the amplification factor itself is trending worse with each measurement, not converging, 1.0 buys real margin against that uncertainty rather than assuming 3x again: even at this session's ~3.9x, worst case lands ~3.9 rad/s, clear of the 6.0 limit. The amplification mechanism is still not understood -- this is a mitigation, not a fix for the root cause. |
| `--max-ik-err` | `0.001` |  |
| `--ik-warm-start` | `'current'` | 'current' (default) seeds IK from the measured arm pose every solve -- deterministic, reproducible, required for scored attempts. 'last' seeds from the previous solution: faster, but makes results depend on message order/timing. |
| `--state-source` | `'wbc'` | where real body_q comes from. 'wbc' = the WBC's own state topic (authoritative, needs --live/ros2). 'boundary' = the organizer's :5557 endpoint, which lets a dry run measure against the real arm with no WBC at all. 'zeros' = plumbing smoke test only. |
| `--orin-host` | `'127.0.0.1'` | host serving the organizer's :5555/:5557 endpoints |
| `--state-port` | `5557` |  |
| `--dex1-port` | `5599` | where run_wbc_with_dex1.py listens for gripper targets. 0 disables (no grasp possible). |
| `--dex1-host` | `'127.0.0.1'` |  |
| `--max-chunk-age-s` | `1.0` | drop a chunk older than this (0 disables). Guards against acting on a stale plan after a stall. |
| `--joint-lane` | `'on'` | accept the b'joint' (T,22) joint-angle chunks and b'goto' pose requests on the same :5556 socket, alongside b'taskspace'. The taskspace path is unaffected either way. 'off' ignores both topics like any unknown prefix. |
| `--joint-lane-limits` | `'urdf'` | which position limits the joint lane clamps arm angles to (never typed in; read from the robot model file). 'urdf' (default) = the RAW URDF limits of g1_29dof_with_hand.urdf, the model both the WBC's robot model and ik.py load -- NOT the WBC RobotModel's supplemental-narrowed arrays (its 0.19 rad shoulder_roll narrowing is enforced by nothing on the robot: its JointSafetyMonitor treats position violations as warnings, and the pose lane's IK ranges over the raw URDF too). 'ik' = the same raw URDF limits plus ik.py's solver-side position overrides, read from IKSettings -- exactly what PinkArmIK enforces -- so the joint lane ranges over the same space the taskspace lane's IK does. With the defaults that is the elbow upper bound (1.4) only: wrist_roll is no longer hard-limited on either lane (IKSettings.wrist_roll_limit_override is None; a posture weight steers the IK toward the seed instead), so 'ik' differs from 'urdf' only by the elbow bound unless a wrist_roll override is set. Those overrides exist to steer a redundant solution, not to protect hardware, so they are not the default. Switch if ruled. |
| `--goto-max-speed` | `0.45` | rad/s ceiling on a b'goto' request's own max_speed; must be > 0 (--joint-lane off is the switch, not 0). Also capped by --max-joint-vel so the step clamp never shortens the ramp. 0.45 is the speed the joint-space pre-motion that preceded this lane ran at live. |
| `--sonic-host` | `'127.0.0.1'` |  |
| `--sonic-port` | `5580` | gear_sonic_deploy's zmq input endpoint (--input-type zmq, NOT the default --input-type zmq_manager -- that one runs an internal planner nobody wants here). CONFIRMED 2026-09-04: g1_deploy_onnx_ref.cpp's own --zmq-port compiles in a default of 5556, same as the boundary's action port -- deploy.sh can't even pass --zmq-port through, so reaching this requires calling `just run g1_deploy_onnx_ref` directly with --zmq-port matching this flag. 5580 isn't special, just deliberately not 5555/5556/5557 (the organizer's camera/action/state ports) -- confirm whatever port gear_sonic_deploy is actually launched with matches this value exactly. |
| `--sonic-socket` | `'pub'` |  |

### Robot-side bridge scripts

`reference/orin_bridge/*.py` take **no command-line arguments** — they are
configured entirely by environment variables, so `--help` does nothing.

| Variable | Default | Meaning |
|---|---|---|
| `HEAD_DEVICE_NAME` | `USB Camera` | V4L2 name substring used to find the head camera |
| `HEAD_DEVICE` | *(unset)* | Explicit `/dev/videoN` override; skips name matching |
| `LEFT_WRIST_SERIAL` | *(unset, required)* | RealSense serial for `left_wrist` |
| `RIGHT_WRIST_SERIAL` | *(unset, required)* | RealSense serial for `right_wrist` |
| `EGO_VIEW_EYE` | `left` | Which half of the stereo frame becomes `ego_view` |
| `EGO_VIEW_RECTIFY` | `0` | Set `1` to rectify. Off by default — reference data is unrectified |
| `PUBLISH_STEREO` | `1` | Also publish `ego_view_left` / `ego_view_right` |
| `HEAD_CAMERA_CALIBRATION` | `config/head_camera_calibration.yaml` | Calibration file path |

Wrist serials are **per-robot and have no default**. Find yours with
`tools/diagnose_cameras.py`.
