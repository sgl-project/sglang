# pyxccl + oneCCL Setup for SGLang XPU (Intel GPU)

`PyXcclCommunicator` is the XPU counterpart of `PyNcclCommunicator` — a
direct-binding communicator for Intel's oneCCL library that bypasses
`torch.distributed` for collective operations (AllReduce, AllGather,
ReduceScatter, Broadcast, Send/Recv). It lives at
`python/sglang/srt/distributed/device_communicators/pyxccl.py`. The ctypes layer
that binds oneCCL's `oneccl*` C API is vendored in
`pyxccl_wrapper.py` (the XPU analogue of `pynccl_wrapper.py`) — no third-party
Python binding is required, only the oneCCL runtime library `libccl.so`.

By default, tensor-parallel (`--tp > 1`) collectives on Intel GPU go through
torch's XCCL backend. **pyxccl is opt-in**: set `SGLANG_ENABLE_PYXCCL=1` to route
them through oneCCL directly instead. This is the path to use when your torch
build has no usable XCCL backend. When `SGLANG_ENABLE_PYXCCL=1` is set but pyxccl
cannot initialize, SGLang raises at startup for `tp>1` rather than silently
falling back — since you opted in precisely because torch.distributed may be
unavailable.

---

## Quick Checklist

```text
[ ] 0. Install one-time system prerequisites
[ ] 1. Source Intel oneAPI compiler environment
[ ] 2. Build oneCCL-v2 XPU library (libccl.so.1)
[ ] 3. Point SGLANG_PYXCCL_SO_PATH / LD_LIBRARY_PATH at libccl.so.1
[ ] 4. Source the SGLang-XPU activation script before every run
```

---

## Step 0 — One-time system prerequisites

Install these packages once on the machine:

```bash
apt-get install -y gcc g++ libopenmpi-dev
```

Set up a dedicated Python environment for build tools (cmake, ninja). This
environment is used only during the oneCCL build, not at runtime:

```bash
uv venv /work/.build-env --python 3.12
uv pip install --python /work/.build-env/bin/python cmake ninja
```

---

## Step 1 — Source the Intel oneAPI compiler

```bash
source /work/compiler/setvars.sh
```

This must be run **in the same shell** before building or running anything.
Verify it works:

```bash
icpx -v   # should print: Intel(R) oneAPI DPC++/C++ Compiler 2026.x.x
```

> **Note:** Sourcing oneAPI gives you the compiler and the `libccl.so` runtime.
> It does **not** install the Python `pyxccl` binding (Step 3), and a stock
> oneCCL runtime may predate the NCCL-compatible `oneccl*` C shim or be built
> without SYCL/XPU device code. Build oneCCL-v2 from source (Step 2) to
> guarantee a `libccl.so.1` that has both.

---

## Step 2 — Build oneCCL-v2 XPU library

oneCCL must be built from source with SYCL/dpcpp support to produce
`libccl.so.1` (the XPU-capable shim used by pyxccl).

### 2a. Clone and patch the repo

Two source patches are required for the SYCL build to work correctly:

**Patch 1** — `CMakeLists.txt` (top-level): pass `-DCMAKE_CXX_STANDARD=17` to
the inner libccl ExternalProject so that `FindIntelSYCL_level_zero.cmake` can
detect `-fsycl` support (SYCL requires C++17).

**Patch 2** — `deps/libccl/CMakeLists.txt`: guard `set(CMAKE_CXX_STANDARD 11)`
with `if(NOT CMAKE_CXX_STANDARD)` so the C++17 override is not clobbered by the
libccl default.

### 2b. Run the build script

```bash
# Standard Release build (recommended):
bash scripts/build_oneccl_xpu.sh --install

# Clean rebuild from scratch:
bash scripts/build_oneccl_xpu.sh --install --clean

# Debug build:
bash scripts/build_oneccl_xpu.sh --install --debug

# Also build tests and examples:
bash scripts/build_oneccl_xpu.sh --install --with-tests --with-examples

# Faster device-code for inner-loop development (larger binary, not for deploy):
bash scripts/build_oneccl_xpu.sh --install --device-code per_source
```

The script automatically sources the compiler, locates cmake/ninja, initializes
git submodules, configures with `-DCOMPUTE_BACKEND=dpcpp
-DCMAKE_CXX_COMPILER=icpx`, and installs into
`<repo>/build-xpu-release/_install/`.

### 2c. Verify the build

```bash
CCL_INSTALL=/work/libraries.performance.communication.oneccl-v2/build-xpu-release/_install
ls -lh $CCL_INSTALL/lib/libccl.so*
# Expected output:
#   libccl.so  -> libccl.so.2
#   libccl.so.1.0   (~258 MB, XPU SYCL device code)
#   libccl.so.2.0   (~128 KB, oneCCL v2 core)
```

---

## Step 3 — Point SGLang at the oneCCL runtime

The ctypes binding is vendored in SGLang (`pyxccl_wrapper.py`), so there is no
separate Python package to install — you only need the oneCCL runtime library
(`libccl.so`) on the loader path. Use the SYCL library you built in Step 2 by
either putting its directory on `LD_LIBRARY_PATH` or pointing
`SGLANG_PYXCCL_SO_PATH` directly at `libccl.so.1`:

```bash
CCL_INSTALL=/work/libraries.performance.communication.oneccl-v2/build-xpu-release/_install
export LD_LIBRARY_PATH=${CCL_INSTALL}/lib:$LD_LIBRARY_PATH
export SGLANG_PYXCCL_SO_PATH=${CCL_INSTALL}/lib/libccl.so.1
```

Verify (the binding is importable directly from SGLang):

```bash
source /work/compiler/setvars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/compiler/lib/
CCL_INSTALL=/work/libraries.performance.communication.oneccl-v2/build-xpu-release/_install
python -c "
from sglang.srt.distributed.device_communicators.pyxccl_wrapper import ONECCLLibrary
lib = ONECCLLibrary('$CCL_INSTALL/lib/libccl.so.1')
print('oneCCL version:', lib.onecclGetVersion())
"
```

---

## Step 4 — Activation script

Source `scripts/activate-xpu-sglang.sh` before every SGLang run. Its contents:

```bash
#!/bin/bash
# activate-xpu-sglang.sh — source this before running SGLang on XPU with pyxccl
#
# Usage:  source scripts/activate-xpu-sglang.sh

# 1. Intel oneAPI compiler & SYCL runtime
source /work/compiler/setvars.sh

# 2. Runtime shared libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/compiler/lib/

_CCL=/work/libraries.performance.communication.oneccl-v2/build-xpu-release/_install
export LD_LIBRARY_PATH=${_CCL}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${_CCL}/opt/mpi/libfabric/lib:$LD_LIBRARY_PATH

# 3. oneCCL / OFI transport
export FI_PROVIDER_PATH=${_CCL}/opt/mpi/libfabric/lib/prov
export CCL_ATL_TRANSPORT=ofi
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0

# 4. SGLang pyxccl integration
# Opt into the direct-oneCCL path (pyxccl is off by default).
export SGLANG_ENABLE_PYXCCL=1
export SGLANG_PYXCCL_SO_PATH=${_CCL}/lib/libccl.so.1

unset _CCL
```

Run SGLang with tensor parallelism:

```bash
source scripts/activate-xpu-sglang.sh
python -m sglang.launch_server \
    --model-path facebook/opt-125m --device xpu --tp 2 \
    --attention-backend intel_xpu
```

---

## OFI transport notes

oneCCL needs a transport backend for inter-process communication. The bundled
libfabric providers shipped with oneCCL support:

| Provider | Use case |
| --- | --- |
| `shm` | Single-node, shared-memory (lowest latency) |
| `tcp` | Single/multi-node, TCP fallback |
| `verbs` | InfiniBand / RoCE (HPC clusters) |

Without `CCL_ATL_TRANSPORT=ofi` and `FI_PROVIDER_PATH`, oneCCL cannot initialize
any transport. `PyXcclCommunicator` logs a warning, marks itself disabled, and
SGLang raises a `RuntimeError` at `tp>1` — set these variables before starting
SGLang workers.

---

## Diagnosing tp>1 failures

### libccl.so.1 not found

```text
Failed to load oneCCL library from libccl.so.1
```

**Fix**: Check `SGLANG_PYXCCL_SO_PATH` and `LD_LIBRARY_PATH`.

### OFI transport not configured

```text
CCL_ERROR: could not initialize any transport library
oneCCL communicator initialization failed
```

**Fix**: Set `CCL_ATL_TRANSPORT=ofi` and `FI_PROVIDER_PATH`.

### oneCCL runtime not installed

```text
PyXcclCommunicator failed to initialize
Failed to load oneCCL library from libccl.so.1
```

**Fix**: Build the oneCCL runtime (Step 2) and set `SGLANG_PYXCCL_SO_PATH` to a
valid `libccl.so.1` (or put its directory on `LD_LIBRARY_PATH`).

---

## Environment variable reference

| Variable | Default | Description |
| --- | --- | --- |
| `SGLANG_ENABLE_PYXCCL` | `0` | Set to `1` to route XPU `tp>1` collectives through oneCCL directly; otherwise torch's XCCL backend is used |
| `SGLANG_PYXCCL_SO_PATH` | (unset) | Explicit path to `libccl.so.1` |
| `CCL_ATL_TRANSPORT` | (auto) | Must be `ofi` for XPU |
| `FI_PROVIDER_PATH` | (system) | Path to OFI provider `.so` files |
| `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK` | `1` | Set to `0` to suppress PCIe topology warnings |
