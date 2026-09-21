# Troubleshooting

Failures that have already cost real bench time. Most of these are silent —
they do not crash, they just produce wrong behaviour or no behaviour.

## GPU and containers

**No GPU at all, despite `--runtime nvidia`.**
NGC CUDA base images bake in a driver-compatibility gate that fails on this
Thor. Add `-e NVIDIA_DISABLE_REQUIRE=1`. Without it you get a container that
starts fine and sees no GPU.

**`--gpus all` does nothing.** Jetson does not support it. Use
`--runtime nvidia`. Note the Docker default runtime on this box is plain
`runc`, so the flag is required, not optional.

**`cudaErrorNoKernelImageForDevice`.**
Thor is **sm_110 (Blackwell)**. Prebuilt CUDA wheels — ONNX Runtime CUDA
builds especially — frequently ship no kernels for it. Your PyTorch model can
work perfectly while an ONNX Runtime component in the same pipeline fails.
Fix is a build targeting sm_110; it cannot be patched at the bench.

**`RuntimeError: Numpy is not available` on first inference.**
The nastiest one here. Some NGC PyTorch bases ship a torch compiled against
the NumPy 1.x ABI. Pinning `numpy==2.x` on top leaves torch importable, CUDA
working, the container starting, `:8765` binding and conformance passing —
then `torch.from_numpy()` fails on the first real inference, during your slot.
`torch.as_tensor()` still works, which makes it look partially fine.

Check before you ship:

```bash
docker run --rm --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 <your-image> \
  python3 -c "import numpy,torch; torch.from_numpy(numpy.ones(3,dtype='float32'))"
```

**Base image mismatch.** The Orin-side image must match the robot's L4T
release. Building on a newer base can mask bugs that only appear on the real
Python version the robot actually has.

## The Thor↔Orin link

**`TypeError: create_connection() got an unexpected keyword argument
'ping_interval'`.**
The reference `transport.py` passes `ping_interval=None, ping_timeout=None`
to `websockets.sync.client.connect()`. The last `websockets` release built
for older Python versions has no such kwargs on that call. Thor is unaffected;
the Orin side dies on first connect. This is inherited from the template and
has hit multiple teams. Guard both kwargs:

```python
import inspect
_params = inspect.signature(_ws_connect).parameters
kwargs = {k: v for k, v in (("ping_interval", None), ("ping_timeout", None))
          if k in _params}
```

Two settings in that file are load-bearing on real hardware and worth keeping
if you rewrite it: disabled keepalive (cold first inference can exceed the
default timeout) and `max_size=None` (three 480×640×3 images per observation).

## Timing

**Your policy stages run before anyone is watching.**
If your server runs a scripted stage sequence on its own clock from container
start, it does not wait for a client to connect. Start it too early and the
useful stages burn through unsupervised. Bring the robot side up first and
your Thor container **last**, immediately before going live. If your sequence
does not restart itself, `docker rm -f` and re-run before each attempt.

## Cameras

**`/dev/videoN` numbering drifts** — observed moving three times in a single
session from ordinary USB re-enumeration. Never hardcode a path. The bridge
matches the head camera by V4L2 *name* and then verifies by reading a probe
frame; wrist cameras are matched by RealSense serial.

**A missing wrist camera does not stop the stream.** The key is simply absent.
If your server substitutes the head frame for a missing wrist, your policy
receives the head camera in three slots and fails for a reason that looks like
yours but is not. Check for the keys you declared.

**Left/right wrist assignment cannot be guessed.** Confirm it empirically by
covering one lens and watching which stream darkens. `tools/diagnose_cameras.py`
exists for this.

## DDS and ROS

**Topics never connect, but nothing errors.**
`RMW_IMPLEMENTATION` must be identical across every ROS 2 process. A mismatch
is silent: both sides start cleanly and simply never see each other.

**`undefined symbol: shm_set_data_state`.**
Two incompatible builds of `libddsc` can coexist — one from conda-forge (has
the symbol, needed by `librmw_cyclonedds_cpp.so`) and one from a source build
(lacks it, needed by the cyclonedds Python package). Because the Python
binding loads first via `ctypes`, `rmw_cyclonedds_cpp` dies. Use
`rmw_fastrtps_cpp`, which sidesteps `libddsc` entirely. See
`config/ros_dds_env.sh`.

**`*** buffer overflow detected ***` from `ChannelFactoryInitialize`.**
Passing an explicit interface name can stack-smash inside Cyclone DDS's
verbose interface-logging path. The culprit is a `<Tracing>` block in the SDK's
config template that is present only in the named-interface variant. Removing
that block fixes it. Auto-detect mode does not crash. This is
platform-specific — it does not reproduce everywhere.

**Activating the conda environment is not enough.** It gives you a ROS distro
with `CYCLONEDDS_URI`, `CYCLONEDDS_HOME` and `PYTHONPATH` unset. Source
`config/ros_dds_env.sh` as well.

**A login shell may hang.** The robot's shell profile can contain an
interactive prompt with no default case, which blocks forever on a
non-interactive `bash -lc`. Prefer plain non-login commands.

## Weights

**Gated backbones.** If your checkpoint depends on a gated model repository,
the backbone is not bundled with your delivery and the organizer may not have
access. Confirm the whole load path works from a clean environment.

**Weights are never baked into images.** Mount them read-only at runtime.

## Grippers

The gripper is a 2-finger unit. It is **not** exposed on the topic name you
might expect from the SDK docs, and a silent gripper topic is not evidence of
missing hardware — the joints ride in the main command array. Do not conclude
the hardware is absent from topic silence alone.
