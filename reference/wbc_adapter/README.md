# wbc_adapter — run any team's submission on the real G1, unmodified

The missing piece the boundary contract always assumed existed: the thing
that **dials into a team's bound `:5556`** and actually drives the robot.
`boundary/actions.py` says it outright — *"THE CLIENT BINDS. The whole-body
controller connects to you."* Every bench run so far has had `mock_wbc.py`
on that socket, which validates shape and then throws the actions away.
This puts NVIDIA's real whole-body control there instead.

The design goal is that **teams change nothing**. Their Thor container and
their Orin container run exactly per their own `INSTRUCTIONS.md`. Nothing
in this package is team-specific, and nothing should become team-specific —
if a team needs special handling at this layer they are off-contract, and
that is an evaluation finding, not an adapter feature.

```
 team Thor container ──:8765──▶ team Orin container
                                     │ binds :5556  (boundary/actions.py)
                                     ▼
                              wbc_driver.py  ← this package
                                     │
        decoupled: IK ─▶ joint-space goal ─▶ ROS2 ControlPolicy/upper_body_pose
        sonic:     relay protocol-v4 pose frames ─▶ gear_sonic_deploy
                                     ▼
                               real G1 motors
```

## What runs where

| Where | What | Notes |
|---|---|---|
| Thor | team's policy-server container | unmodified, per their manifest |
| G1 Orin | team's policy-client container | unmodified; binds `:5556` |
| G1 Orin | `run_g1_control_loop.py` (Decoupled WBC) | `decoupled` lane |
| G1 Orin | `gear_sonic_deploy` binary | `sonic` lane |
| G1 Orin | **`wbc_driver.py`** (this) | bridges the two |
| G1 Orin | real camera/state publisher on `:5555`/`:5557` | **still to build — see Gaps** |

## Verified against source, not notes

Everything below was read out of `NVlabs/GR00T-WholeBodyControl` **v1.1**
(commit `a0732b6`, 2026-08-20), cloned to `~/GR00T-WholeBodyControl`. An
earlier operator harness (`~/peval_v1-main`) documented the same contract
from a 2026-07-05 clone; where they agreed I kept it, but v1.1 is the
authority here and it is six weeks newer.

- `constants.py` → `CONTROL_GOAL_TOPIC = "ControlPolicy/upper_body_pose"`,
  `STATE_TOPIC_NAME = "G1Env/env_state_act"`, `DEFAULT_BASE_HEIGHT = 0.74`
  (which is exactly the `(T,25)` col `[21]` value teams send — the
  conventions line up).
- `G1DecoupledWholeBodyPolicy.set_goal` consumes
  `target_upper_body_pose` / `base_height_command` / `target_time` /
  `navigate_cmd`. **The WBC runs no IK in its 50 Hz loop** — its upper-body
  "policy" is interpolation over joint targets. So IK happens here,
  upstream, exactly as NVIDIA's own teleop stack does it.
- `target_time` accepts a **list**, so a whole `(T,25)` chunk goes over as
  an interpolated waypoint trajectory rather than only its first row.
- `run_g1_control_loop.py` sets `interpolation_garbage_collection_time`
  itself every tick — so this package deliberately does not.
- `ros_utils.py` → goals are msgpack(+msgpack_numpy) packed into a
  `ByteMultiArray` whose `data` is a tuple of single bytes.

**The `sonic` lane turned out to be a relay, not a translation.** The
organizer's `boundary/actions.py` pose frame is byte-identical to the
protocol-v4 `pose` message `gear_sonic_deploy` consumes: same `pose` topic,
same 1280-byte header, same `token_state (1,64)` / `frame_index (1,)` /
`left|right_hand_joints (1,7)` fields. The driver forwards the **original
bytes** rather than re-packing, so a re-encode can't drift from what the
team actually published.

## Tested end-to-end on the Thor bench

Run against a team's real containers with a real trained checkpoint
(not mocks, not a stub): team Thor server → team Orin client → this
adapter → goal payloads decoded by a listener.

```
goal: keys=[base_height_command, navigate_cmd, target_time,
            target_upper_body_pose, wrist_pose]
      target_upper_body_pose shape=(12, 14) float64
      target_time list len=12, span 0.550s
      base_height_command=0.74   navigate_cmd=[0 0 0]
```

Three real bugs were found and fixed by that test, all of which would have
mattered on hardware:

1. **Double latency compensation.** The reference `client.py` already drops
   the rows its inference latency consumed *and* backdates `issued_at` to
   when that inference started. Re-skipping on that stamp double-counts:
   measured live, it ate **11 of 16 rows on top of the client's own 4**,
   leaving a single usable waypoint. `issued_at` is now used only to judge
   staleness, never to re-skip.
2. **Unbounded backlog.** With a plain `SUB`, any moment where IK is slower
   than the team's publish rate silently queues, and every chunk pulled is
   staler than the last — the robot ends up tracking the past. `decoupled`
   now uses `CONFLATE` (a newer chunk is a complete fresh plan and
   supersedes an older one); `sonic` deliberately does not (dropping rows
   there is dropping motion).
3. **Accept-rate denominator.** IK is solved per waypoint but the rate was
   divided by messages, reporting >100%.

## The `joint` lane

Policies trained on joint-space actions do not need the IK at all: the
Decoupled WBC's native input is a joint-space `target_upper_body_pose`.
The same `wbc_driver.py --lane decoupled` process therefore also accepts a
`b"joint"` topic — `(T,22)` rows of hand commands, 7+7 arm joint angles in
`G1JointIndex` order, `navigate_cmd` and `base_height_cmd` — and a
`b"goto"` request that interpolates the arms from the measured pose to a
target at a bounded speed as one multi-waypoint goal. Rows are
position-clamped to the robot model's own limits (read from the WBC's
model, `--joint-lane-limits`), rate-clamped by the existing
`--max-joint-vel` step clamp anchored on the fresh measured arms, mapped
into the 28-wide vector by `UpperBodyMapper` (hand slots copied from state)
and published exactly as an IK result would be. `--joint-lane off` disables
both topics. The taskspace path is untouched; `tests/test_joint_lane.py`
proves it by replaying the same chunks through the pre-lane driver (pulled
from git) and asserting identical goals. Wire spec: `docs/CONTRACT.md`,
"The `joint` lane".

## Safety

This adapter is **not** a safety system. It re-checks the contract floor
and holds the last known-good arm target when IK says a pose is
unreachable, but well-formed is not the same as sane.

- **Dry-run is the default.** `--live` is required to publish anything
  robot-ward, and prompts before proceeding.
- **It refuses to publish without robot state.** Solving IK against a
  guessed configuration is worse than publishing nothing. (Verified: with
  no state source it correctly published zero goals.)
- **Stale-chunk guard** (`--max-chunk-age-s`, default 1.0s) so a stall
  can't be followed by acting on an old plan.
- The stop that actually works is the independent e-stop. For `sonic`
  specifically it is `gear_sonic_deploy`'s own stop (gamepad `Select`/`O`,
  or its command-topic stop flag) — whoever owns `rt/lowcmd` wins, and that
  is the deploy binary, not this process.

## Running

```bash
# Bench / dry-run (no robot, no ROS 2) — decode, validate, solve IK, print
python3 wbc_driver.py --lane decoupled --actions-host 127.0.0.1 --verbose

# Bench loopback with real goal encoding over ZMQ instead of ROS 2
python3 wbc_driver.py --lane decoupled --live --wbc-backend zmq

# On the G1, against the real Decoupled WBC
python3 wbc_driver.py --lane decoupled --live --wbc-backend ros2 \
    --actions-host 127.0.0.1 [--enable-waist]

# On the G1, sonic lane (relay to the deploy binary)
python3 wbc_driver.py --lane sonic --live \
    --sonic-host 127.0.0.1 --sonic-port <the deploy's real zmq input port>
```

`--enable-waist` **must** match how `run_g1_control_loop.py` was launched:
waist in the upper-body group means width 17, otherwise 14. Get it wrong
and every joint in the vector is misaligned. Read it from the robot model
rather than trusting the flag where you can.

## Gaps — what is NOT done

1. **Real camera/state publisher for `:5555`/`:5557`.** The team's client
   needs these, and on the bench `mock_orin.py` provides them. On the robot
   something must publish *real* head/wrist cameras and real robot state in
   `boundary/cameras.py`'s and `boundary/states.py`'s exact wire format.
   An earlier bench session built exactly this (`real_orin.py`, on the G1 at
   `/home/unitree/`) and it is team-agnostic by construction — it should be
   lifted into this package rather than rewritten. Note the state half gets
   easier once the WBC is up: `G1Env/env_state_act` already publishes `q`,
   and `wbc_goal.py` already subscribes to it.
2. **Neither NVIDIA component has been built or run.** `gear_sonic_deploy`
   is a C++/TensorRT project (`just build`); `decoupled_wbc` needs ROS 2 +
   its own Python env on the Orin. Nothing here has been tested against
   either — the ROS 2 backend is written against the verified message
   contract but has never round-tripped with a live `run_g1_control_loop`.
3. **`--sonic-port` is a placeholder (5580), not a confirmed value.** Read
   the real port off the deploy's launch config. It is deliberately not
   5555/5556/5557 — those are the organizer's camera/action/state ports.
   The socket type (`pub` vs `push`) is likewise unconfirmed against the
   deploy's `zmq_endpoint_interface`.
4. **The EE frame convention is undefined by the contract** — see the long
   warning at the top of `ik.py`. The `(T,25)` spec says "end-effector
   position" without pinning where on the hand that is. Whatever IK
   consumes it defines it in practice. Use the `decoupled_wbc` backend on
   the robot so NVIDIA's own solver and robot model define the frame,
   rather than a third convention invented here.
5. **peval's own open items still stand** (`~/peval_v1-main`,
   `FINDINGS_ADDENDUM.md` §8): the real 28-index joint mapping is an
   unconfirmed placeholder, timing limits are placeholders, checklists are
   DRAFT and hard-block real sessions.

## Observed IK accept rate — read this before drawing conclusions

Against one real team checkpoint the per-waypoint IK accept rate came out
**L=100%, R=67–100%** depending on the run (the right arm varies with
solver warm-start sequencing). That is dramatically better than an earlier
measured 1–3%, but the two numbers **are not comparable**: this was against
`mock_orin.py`'s synthetic images, whereas the earlier one was against real
cameras on the real robot. A policy fed synthetic input may sit near a
neutral, trivially-reachable pose and tell you nothing about its behaviour
on real perception. Treat this as "the plumbing and IK work", not as
evidence about the policy.
