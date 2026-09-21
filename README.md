# IROS 2026 — G1 / Orin interface package

Everything a team needs to build a policy that drives the Unitree G1 in the
IKEA IROS Assembly Challenge: the wire contract you publish onto, the
organizer-side code that consumes it, the environments it runs in, and the
tools to check your rig before a slot.

**Start with [`docs/CONTRACT.md`](docs/CONTRACT.md).** It is the only document
you strictly have to read.

## The split — who runs what

```
   YOUR THOR CONTAINER              THE ROBOT'S ONBOARD COMPUTER
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

You build the two containers. You do not build or modify anything in
`boundary/` or `reference/`.

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

The Thor box your container runs on:

```
NVIDIA Jetson AGX Thor          Ubuntu 24.04 (noble), L4T R39 rev 2.1
GPU: sm_110 (Blackwell)         CUDA 13.2, driver 595.78
122 GiB unified RAM, 14 cores   Docker 29.7.2, default runtime: runc
```

`linux/arm64` images only. **sm_110 is new** — prebuilt CUDA wheels often have
no kernels for it. See `docs/TROUBLESHOOTING.md` before assuming a CUDA
library works.

## Launching your Thor container

```bash
docker run -d --name ikea-iros-thor \
  --runtime nvidia --network host \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -v /path/to/weights:/model:ro \
  <your-registry>/<your-image>@sha256:<digest>
```

Non-negotiable parts:

- `--runtime nvidia`, **not** `--gpus all` — Jetson does not support the latter.
- `-e NVIDIA_DISABLE_REQUIRE=1` — without it NGC CUDA bases fail their
  driver-compat gate and you silently get **no GPU at all**.
- `--network host` — the boundary is raw TCP, not a Docker bridge network.
- Weights mounted **read-only**, never baked into the image.
- Pin by `@sha256:` digest, never `:latest`.

## Robot-specific values

Camera serials, calibration, hostnames and network addresses are **per-rig**
and are placeholders here. The event robot is not the development robot. Set
them via environment variables or `config/`; do not hardcode.
