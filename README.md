# IROS 2026 — G1 / Orin interface package

Everything a team needs to build a policy that drives the Unitree G1 in the
IKEA IROS Assembly Challenge: the wire contract you publish onto, the
organizer-side code that consumes it, the environments it runs in, and the
tools to check your rig before a slot.

**Start with [`docs/CONTRACT.md`](docs/CONTRACT.md).** It is the only document
you strictly have to read.

## The split — who runs what

```
   YOUR POLICY, ON THOR             THE ROBOT'S ONBOARD COMPUTER
   (GPU, your inference)            (organizer's code, you run it)

   policy server  ──BINDS :8765──<  your client  ──BINDS :5556──<  wbc_adapter
                                         │                             │
                                         ├──SUB :5555 cameras ─────────┤
                                         └──SUB :5557 state   ─────────┘
```

Two things about that diagram trip people up:

1. **Your client BINDs `:5556`; the organizer's controller dials in to you.**
   That is backwards from the usual ZMQ habit and it catches everyone once.
2. **The Orin dials out to Thor's `:8765`**, not the other way round. That
   link is *yours* — `components/transport.py` is team-owned and you may
   replace it entirely. The organizer never speaks it.

You build the two policy components. You do not build or modify anything in
`boundary/` or `reference/`.

Note the two halves are packaged differently. Your policy runs on Thor,
however you choose to package it. The robot-side stack it drives
(`wbc_adapter`, the camera/state bridge, and the sonic-lane control binary)
runs natively on the robot's onboard computer — not in a container.
`reference/` is that native half, published so the layer consuming your
output is not a black box.

## What is in here

| Path | What it is |
|---|---|
| `docs/CONTRACT.md` | The wire contract, and the evaluator settings that affect your score |
| `docs/RUNBOOK.md` | Operating procedure, plus a full CLI reference |
| `docs/TROUBLESHOOTING.md` | Failures that have already cost people a bench slot |
| `boundary/` | The contract package. Identical for every team. **Do not modify.** |
| `reference/wbc_adapter/` | Organizer code that consumes your actions. Read it; don't run it. |
| `reference/orin_bridge/` | Publishes the cameras and state you subscribe to |
| `tools/` | Preflight and diagnostic scripts |
| `env/` | Conda environments for the robot-side stack |
| `config/` | DDS config, ROS environment, camera calibration |

## Target platform

The Thor box your policy runs on:

```
NVIDIA Jetson AGX Thor          Ubuntu 24.04 (noble), L4T R39 rev 2.1
GPU: sm_110 (Blackwell)         CUDA 13.2, driver 595.78
122 GiB unified RAM, 14 cores   Docker 29.7.2, default runtime: runc
```

Everything must be `linux/arm64`. **sm_110 is new** — prebuilt CUDA wheels often have
no kernels for it. See `docs/TROUBLESHOOTING.md` before assuming a CUDA
library works.

## Running your policy

**This repository does not prescribe how you package or launch your policy.**
What it specifies is the contract your code must meet at the sockets in the
diagram above. How you get there is yours.

In practice teams do this two ways, and both are fine:

- **A container image** — the common path. Digest-pinned, declared in your
  `manifest.yaml`, launched with `docker run`.
- **A conda environment plus your own scripts** — no container at all.

If you containerise, a few Thor-specific details save time:

- **`--runtime nvidia`.** Docker's default runtime on this box is plain
  `runc`, so the GPU must be requested explicitly. `--gpus all` alone is not
  reliable on Jetson; passing both is harmless.
- **`-e NVIDIA_DISABLE_REQUIRE=1`.** NGC CUDA base images bake in a
  driver-compatibility gate that fails here. Without it you get a container
  that starts cleanly and has **no GPU at all**.
- **`--network host`.** The boundary is raw TCP, not a Docker bridge network.
- **Mount weights read-only**; don't bake them into the image.

If you run natively, the same constraints apply minus the Docker flags: the
boundary ports must be reachable, and your environment needs whatever your
model requires. `linux/arm64` either way.

## The robot side runs natively, on ROS 2

Nothing in `reference/` is containerised. On the robot's onboard computer:

- `real_orin*.py` publishes cameras and state over ZMQ
- `wbc_adapter/wbc_driver.py` consumes your actions from `:5556` (pure ZMQ)
- `wbc_goal.py` publishes goals onto the **ROS 2** topic
  `ControlPolicy/upper_body_pose` and reads state from `G1Env/env_state_act`
- the sonic-lane control binary is native C++/TensorRT

You do not run any of this — it is published so the layer consuming your
output is not a black box. One consequence worth knowing: because ROS 2 is in
the path, `RMW_IMPLEMENTATION` must match across every process on that side.
A mismatch is silent — everything starts, topics never connect. See
`config/ros_dds_env.sh`.

## Robot-specific values

Camera serials, calibration, hostnames and network addresses are **per-rig**
and are placeholders here. The event robot is not the development robot. Set
them via environment variables or `config/`; do not hardcode.
