# Building sglang-sxm2 from source

Everything here has been executed end-to-end on the reference host. Where a step
has a common failure mode, it is written down with the exact error text, because
most of these fail *quietly* — a missing sm70 kernel does not raise, it returns
"unavailable" and silently falls back to a path that produces wrong output.

`scripts/install_v100.sh` automates the whole sequence. This document explains
what it does, so you can run the steps individually when one fails.

---

## 0. Prerequisites

### Hardware

- V100 32 GB: eight for GLM-5.3-Flash and DeepSeek-V4.1-Flash, four for
  Qwen3.8-Flash-Next and MiniMax-H3. SXM2 with NVLink is the measured setup;
  PCIe cards work, slower (see each model page).
- Host RAM and disk depend on the model. The README's Requirements section
  and each model page under `docs/v100/` list them.

### CUDA

**CUDA 12.8 or 12.9. Not 13.x** — CUDA 13 removed support for compute
capabilities below 7.5, which includes Volta. There is no workaround.

```bash
/usr/local/cuda/bin/nvcc --version   # expect 12.8 or 12.9
```

### Host compiler — the one that catches everyone

CUDA 12.9 caps host-compiler support at **GCC 14**, and its own header says so:

```c
#if __GNUC__ > 14
#error -- unsupported GNU version! gcc versions later than 14 are not supported!
```

Many current distributions default `gcc` to 15. Worse, some install `gcc-15`
without `g++-15`, so the default driver has no `cc1plus` at all and every CUDA
compile dies with:

```
gcc: fatal error: cannot execute 'cc1plus': posix_spawnp: No such file or directory
nvcc fatal   : Failed to preprocess host compiler properties.
```

Check and pin:

```bash
ls /usr/libexec/gcc/x86_64-linux-gnu/*/cc1plus     # which GCCs are complete
g++-14 --version                                   # install if absent

export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export CUDAHOSTCXX=/usr/bin/g++-14
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-14"
```

`NVCC_PREPEND_FLAGS` matters separately from `CUDAHOSTCXX`: the runtime JIT
kernels invoke a bare `nvcc`, which reads the former but not the latter. The
serve script exports both.

---

## 1. Clone

```bash
git clone https://github.com/dg1kjd/sglang-sxm2.git
cd sglang-sxm2
```

## 2. Python environment

Python 3.12. Any venv tool works; the reference host uses `uv`.

```bash
uv venv --python 3.12 ~/sglang-v100-venv
source ~/sglang-v100-venv/bin/activate
```

Keep the venv's `bin` on `PATH` for everything below — the JIT kernels shell out
to `ninja`, which is a venv-local binary with no system copy.

## 3. Python dependencies

```bash
pip install --no-deps -r requirements.txt
```

`--no-deps` is mandatory. `python/pyproject.toml` carries upstream's pins, which
target the CUDA 13 stack; letting pip resolve freely pulls cu13 wheels and
breaks Volta support. `requirements.txt` pins the exact validated cu12 set.

Then install SGLang itself, still without dependency resolution. This builds
SGLang's Rust extensions, so it needs Rust 1.85 or newer (`rustup` if the
distro's `cargo` is older). The Rust radix tree requires torch 2.11 or newer
and is optional (the Python tree is the default), so leave it out:

```bash
SGLANG_BUILD_RUST_EXTS=grpc,multimodal,server \
  pip install --no-deps --no-build-isolation -e python/
```

## 4. FlashInfer for sm70

Upstream FlashInfer does not build for Volta. The port carries a patch:

```bash
export SGLANG_V100_DEPS_DIR=~/.cache/sglang-v100-sources
# clone at the pinned revision, apply patches/flashinfer-sm70.patch, then:
pip install --no-deps --no-build-isolation -e "$SGLANG_V100_DEPS_DIR/flashinfer-sm70"
```

Pinned revision: `c3c40a7b90b792fc59f90f8f55c9e2de9c1b6833`.
`scripts/install_v100.sh` does the clone-and-patch for you.

## 5. sglang-kernel, sm70 only

```bash
export TORCH_CUDA_ARCH_LIST=7.0
export CMAKE_ARGS="-DSGL_KERNEL_V100_ONLY=ON -DSGL_KERNEL_COMPILE_THREADS=2"
pip install --no-deps --no-build-isolation python/sglang/kernels/aot
```

Two things to know:

- **`--no-build-isolation` is required, not an optimisation.** The AOT
  `pyproject.toml` declares `torch==2.14.1` as a *build* requirement. An
  isolated build would fetch that and produce an extension whose ABI does not
  match the runtime's torch 2.9.1.
- **`SGL_KERNEL_V100_ONLY=ON` matters a lot.** With it, the build is 38 objects
  and about 11 minutes. Without it, the Hopper FlashAttention-3 target is also
  compiled — 374 more objects of sm90a kernels that cannot run on Volta,
  turning an 11-minute build into an hour.

Confirm the arch in the configure output:

```
-- Added CUDA NVCC flags for: -gencode;arch=compute_70,code=sm_70
```

(The CMake target is named `common_ops_sm100_build` regardless — that is a
label, not the architecture.)

## 6. TurboMind sm70 and Marlin V100

```bash
python scripts/build_sm70_turbomind.py     # block-FP8 + FP16 MoE backend
bash   scripts/setup_v100_marlin.sh        # GPTQ/AWQ repack + NVFP4 MoE
```

Both install their `.so` into `python/sglang/kernels/prebuilt/`, which is where
`sglang.kernels.sm70_paths` looks. That module is the single place that knows
the location; if you relocate the artifacts, change it there and nowhere else.

**This step is not optional.** The stock upstream Marlin MoE kernel is an empty
stub below sm80: it writes nothing. If the V100 kernels are missing, the server
starts, answers, and returns **zero-valued expert output** — plausible-looking
garbage rather than an error. You will see this warning at startup:

```
SM70 (V100) detected but the marlin_v100 MoE kernel was not found.
... routed-expert output will be ZERO (incorrect).
```

Treat it as fatal.

## 7. Restore the CUDA 12 NCCL

Some wheels pull the cu13 NCCL, which torch 2.9.1 cannot use:

```bash
pip uninstall -y nvidia-nccl-cu13
pip install --force-reinstall --no-deps nvidia-nccl-cu12==2.27.5
```

## 8. Model weights

Each model page has its download command and checkpoint notes:
[GLM-5.3-Flash](GLM-5.3-Flash.md), [Qwen3.8-Flash-Next](Qwen3.8-Flash-Next.md),
[DeepSeek-V4.1-Flash](DeepSeek-V4.1-Flash.md), [MiniMax-H3](MiniMax-H3.md).

```bash
pip install -U "huggingface_hub[cli]"
hf download RadixArk/GLM-5.3-Flash-NVFP4 --local-dir ~/models/GLM-5.3-Flash-NVFP4
```

## 9. Verify

```bash
bash scripts/smoke_v100.sh
```

Expect every line to report a registration:

```
SGLang V100 environment is ready: 2.9.1+cu128
FlashInfer SM70 sampling: ...
Attention: SGLang TileLang SM70 package
SM70 kernel: .../sgl_kernel/sm70/common_ops.abi3.so
SM70 Marlin repack: registered
SM70 TurboMind FP8: registered
SM70 TurboMind FP16 MoE: registered
SM70 TurboMind exact AWQ dequantizer: registered
NCCL: 2.27.5
```

If `SGLANG_V100_PYTHON` is not on `PATH`, set it explicitly:

```bash
SGLANG_V100_PYTHON=$(which python) bash scripts/smoke_v100.sh
```

## 10. Serve

```bash
GLM53_MODEL=~/models/GLM-5.3-Flash-NVFP4 bash scripts/serve_glm53_flash_nvfp4_v100.sh
```

Each launcher carries the tuned serving config for its model, with the
*reasons* for each memory-sensitive value in its comments. Read them before
changing `--mem-fraction-static`, the prefill chunk size, or the KV dtype. The
scripts take Python from `SGLANG_V100_VENV` (default `~/sglang-v100-venv`); if
that has none, from the active venv or conda env.

First launch compiles the sm70 JIT kernels; expect several minutes, with every
rank compiling at once. Later launches reuse `~/.cache/sglang/jit/sm70`.

Ready when the log says:

```
The server is fired up and ready to roll!
```

## 11. Run it as a service (recommended)

The serve script derives its repo root from its own location and takes the venv
from `SGLANG_V100_VENV`. That is convenient interactively and a trap over time:
run a stale checkout's copy of the script, or leave a different venv active, and
you silently serve the wrong tree. A unit file pins both.

```bash
sudo cp scripts/sglang-v100.service.example /etc/systemd/system/sglang-v100.service
sudo cp scripts/sglang-v100.env.example     /etc/sglang-v100.env
$EDITOR /etc/sglang-v100.env      # venv, model path, port, GPUs
$EDITOR /etc/systemd/system/sglang-v100.service   # User=, WorkingDirectory=, ExecStart= paths

sudo systemctl daemon-reload
sudo systemctl enable --now sglang-v100
journalctl -u sglang-v100 -f
```

`TimeoutStartSec=infinity` is deliberate: a cold JIT cache compiles on every
rank at once, the cache is only written on success, and a start timeout would
kill it mid-compile every time, so it would never converge.

---

## Docker

The container runs the same steps as this document. It is built from source on
your machine; no pre-built image is published.

### Host requirements

- An NVIDIA driver from the R570 series or newer (the image uses CUDA 12.8.1).
- Docker with Compose v2 and Buildx, and the NVIDIA Container Toolkit
  registered as a Docker runtime:

  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
  ```

- About 100 GB of disk while building: the image is 32 GB, and
  `docker builder prune` frees the ~55 GB build cache afterwards.
- For DeepSeek-V4.1-Flash, the 1 GiB hugepages reserved on the host, as in its
  model page. A container cannot reserve them.

### How it is built

`docker/v100.Dockerfile` has three stages:

| stage | contents |
|---|---|
| `base` | CUDA 12.8.1 devel on Ubuntu 24.04, g++-12 and g++-14, Python 3.12 venv in `/opt/venv` |
| `builder` | step 3's `requirements.txt` with `--no-deps`, then FlashInfer sm70, TurboMind sm70, sglang-kernel, Marlin V100 and SGLang with its Rust extensions, each its own layer |
| `runtime` | the venv, FlashInfer and the `python/` tree from `builder`, plus the serve scripts and the entrypoint |

The native kernels come before the application code, so a change under
`python/` rebuilds only the last layers. The last builder step checks that
every kernel library contains sm70 code and the expected entry points. A cold
build takes about an hour; `MAX_JOBS` caps the compiler parallelism.

### How it is composed

`docker/v100-compose.yaml` defines one service per model on the same image:

| service | model | GPUs | port | checkpoint under `MODELS_DIR` |
|---|---|---|---|---|
| `glm53` | GLM-5.3-Flash | 8 | 11435 | `GLM-5.3-Flash-NVFP4` |
| `qwen38` | Qwen3.8-Flash-Next, MTP | 4 | 11435 | `Qwen3.8-Flash-Next-NVFP4` |
| `dsv41` | DeepSeek-V4.1-Flash | 8 | 11435 | `DeepSeek-V4.1-Flash` |
| `h3` | MiniMax-H3 video | 4 | 30010 | `MiniMax-H3` |

Every service sits behind a profile of its own name, so nothing starts by
accident; naming the service on the command line selects it. All services use
host networking and host IPC (NCCL and the custom all-reduce need both), an
unlimited `memlock` for pinned host memory, and all GPUs.

| mount | in the container | holds |
|---|---|---|
| `${MODELS_DIR:-~/models}` | `/models`, read-only | checkpoints |
| volume `sglang-sxm2-cache` | `/root/.cache` | Hugging Face cache, Qwen's host KV tier, DeepSeek's session store |
| volume `sglang-sxm2-jit` | `/root/sglang-v100-jit` | FlashInfer, TileLang, Triton and Inductor JIT kernels |

### Using it

From the repository root:

```bash
docker compose -f docker/v100-compose.yaml build glm53     # builds the one shared image
MODELS_DIR=/data/models docker compose -f docker/v100-compose.yaml up glm53
```

A bare `docker compose build` builds nothing, because every service is behind
a profile. Start one language model at a time; they share port 11435. The first
launch of each model compiles its JIT kernels into the `sglang-sxm2-jit`
volume, about 10 minutes for GLM; later launches start from the cache.

The serve scripts' variables pass through from the shell, for example
`SGLANG_V100_HOST=127.0.0.1` to keep the server off the network, or
`SGLANG_V100_PORT`, `GLM53_GPUS`, `FLASH_NEXT_GPUS`, `H3_GPUS`. On PCIe cards
without NVLink set `SGLANG_CUSTOM_AR_ALLOW_PCIE=1` for Qwen (see its page).

The entrypoint, `docker/v100-entrypoint.sh`, takes the model as its first
argument and runs the kernel check from step 9 before each model unless
`SGLANG_V100_SKIP_STARTUP_CHECK=1`:

```bash
docker compose -f docker/v100-compose.yaml run --rm glm53 smoke         # kernel check only
docker compose -f docker/v100-compose.yaml run --rm qwen38 qwen38 target  # Qwen without MTP
docker compose -f docker/v100-compose.yaml run --rm h3 h3 ref2va        # H3 reference mode
docker compose -f docker/v100-compose.yaml run --rm glm53 bash          # a shell in the image
```

Arguments starting with `--` go to `python -m sglang.launch_server` directly.

---

## Troubleshooting

Failures observed during bring-up, with their causes.

| Symptom | Cause |
|---|---|
| `cannot execute 'cc1plus'` | Default `gcc` is 15 (or has no C++ backend). Set `CC`/`CXX`/`CUDAHOSTCXX`/`NVCC_PREPEND_FLAGS` to GCC 14 — see §0. |
| `No such file or directory: 'ninja'` | The venv's `bin` is not on `PATH`. The JIT shells out to `ninja`, and there is usually no system copy. |
| `sglang-kernel is installed with version 0.4.3, which is less than 0.4.6.post1` | Rebuild step 5 from `python/sglang/kernels/aot` (the package moved out of the repo root). |
| `cannot import name 'AnyTokensFormat' from 'xgrammar.structural_tag'` | xgrammar is older than 0.2.1. |
| `SM70 NVFP4 decode extension is unavailable` | A relocated `.cu` was not found. All sm70 sources live under `python/sglang/kernels/jit/csrc/`; consumers must resolve them via `sglang.kernels.sm70_paths.sm70_csrc()`. |
| `ModuleNotFoundError: No module named 'deep_gemm'` during graph capture | An sm90+ backend was imported behind a bare `_is_cuda` gate — and V100 *is* CUDA. Optional backends must load through `_optional_graph_backend_types()`. |
| `AssertionError: Unsupported layout: layer_first` | `--hicache-mem-layout` must be `page_first`. `MambaPoolHost` accepts only `page_first`/`page_first_direct`. |
| `slice [0, N) escapes the M-byte pull workspace` | An all-reduce larger than the workspace was forced into the custom path. Oversized reduces must fall back to NCCL. |
| Correct-looking but nonsensical answers | Almost certainly the Marlin MoE stub — see §6. Check the startup log for the ZERO-output warning. |
| ~100% CPU per rank while idle | `--sleep-on-idle` is not in effect. Healthy idle is ~4% per rank, with the scheduler blocked in `zmq.poll`. |

### Build takes an hour

You forgot `-DSGL_KERNEL_V100_ONLY=ON`, and are compiling the Hopper FA3 kernel
set. See §5.
