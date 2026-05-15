# Distributed Export Runtime Wiring

This note documents the second step of the `@sgl_compile` export work: making
exported callsites aware of distributed execution requirements before they are
scheduled under TP, DP, PP, EP, or CUDA graph capture.

## Design

The export path now has three separate decisions:

- The platform chooses the high-level policy with `torch_compile_strategy()`:
  `compile`, `noop`, or `export`.
- The runtime declares capabilities with `ExportRuntimeCapabilities`.
- Each callsite can require properties such as CUDA graph safety or distributed
  compatibility through `TorchCompileConfig`.

`DistributedExportContext` records the execution topology seen by the first
callsite invocation:

- TP rank and size
- PP rank and size
- DP size from global server args
- EP rank and size
- tensor device type and index

The context is written into export metadata with the input schema and artifact
configuration. This gives cloud artifact builders and serving workers a shared
description of the graph environment.

The important behavior change is that incompatibility is resolved before export:

```text
@sgl_compile callsite
  -> platform strategy says export
  -> runtime capabilities are checked against callsite requirements
  -> incompatible export falls back to compile/noop/error
  -> compatible export builds or loads artifacts
```

This avoids exporting model-forward callsites into runtimes that cannot safely
participate in CUDA graph capture or distributed execution. It is deliberately
different from dynamically bypassing a runtime inside capture. If a callsite is
performance critical and requires graph capture, the runtime must advertise and
provide that property up front.

## Current Policy

`OnnxExportRuntime` currently advertises TP/PP/DP/EP compatibility, but not CUDA
graph capture safety. Its CUDA I/O binding path is safe for the migrated
mutation-style sampling callsite, `apply_scaling_penalties`, but it does not yet
cover arbitrary model-forward outputs.

Model-forward callsites that may execute inside CUDA graph capture are marked
with `requires_cuda_graph_safe=True`. Under `SGLANG_PLATFORM=cuda_onnx`, those
callsites fall back to `torch.compile` today instead of producing ONNX runtime
callables. This keeps the scheduler path valid for TP and DP while still
allowing compatible sampling-side ONNX artifacts.

## Validation Results

Hardware and model:

- GPU: NVIDIA H100 80GB HBM3
- Model: `Qwen/Qwen3-14B`
- Platform: `SGLANG_PLATFORM=cuda_onnx`
- Artifact mode: `SGLANG_EXPORT_ARTIFACT_MODE=build_if_missing`
- Server flags: `--cuda-graph-max-bs 4 --disable-piecewise-cuda-graph`
- Sampling: `temperature=0.0`, `repetition_penalty=1.12`

### TP=2

Command shape:

```bash
CUDA_VISIBLE_DEVICES=1,2 \
SGLANG_PLATFORM=cuda_onnx \
SGLANG_EXPORT_DIR=/tmp/sglang-qwen14-tp2-onnx-runtime-wired \
SGLANG_EXPORT_ARTIFACT_MODE=build_if_missing \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-14B \
  --host 127.0.0.1 \
  --port 30000 \
  --tp-size 2 \
  --cuda-graph-max-bs 4 \
  --disable-piecewise-cuda-graph \
  --disable-custom-all-reduce \
  --trust-remote-code
```

Outcome:

- Server started with TP=2.
- CUDA graph capture completed for batch sizes `[1, 2, 4]`.
- Decode logs showed `cuda graph: True`.
- Only `apply_scaling_penalties` produced ONNX artifacts.

Measured smoke workload:

| Mode | Requests | Median latency | Completion tok/s |
| --- | ---: | ---: | ---: |
| Sequential | 8 | `0.480s` | `130.8` |
| Concurrency 4 | 12 | `0.495s` | `449.8` |

The remaining TP-specific caveat is custom all-reduce. This run used
`--disable-custom-all-reduce` because the earlier TP ONNX smoke hit a custom
all-reduce device mismatch before this runtime policy layer.

### DP=2

Command shape:

```bash
CUDA_VISIBLE_DEVICES=1,2 \
SGLANG_PLATFORM=cuda_onnx \
SGLANG_EXPORT_DIR=/tmp/sglang-qwen14-dp2-onnx-runtime-wired-seq \
SGLANG_EXPORT_ARTIFACT_MODE=build_if_missing \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-14B \
  --host 127.0.0.1 \
  --port 30002 \
  --dp-size 2 \
  --cuda-graph-max-bs 4 \
  --disable-piecewise-cuda-graph \
  --trust-remote-code
```

Outcome:

- Server started with DP=2.
- Both DP ranks captured CUDA graphs for batch sizes `[1, 2, 4]`.
- Sequential requests exercised both DP ranks.
- Decode logs showed `cuda graph: True` on DP0 and DP1.
- Only `apply_scaling_penalties` produced ONNX artifacts.

Measured sequential smoke workload:

| Mode | Requests | Median latency | Completion tok/s |
| --- | ---: | ---: | ---: |
| Sequential | 6 | `0.530s` | `78.9` |

Concurrent DP stress is not yet clean. Concurrency 4 and 8 both reproduced a
CUDA device failure in the model path after several successful responses:

```text
DP1 Scheduler hit an exception
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

The earlier concurrency-8 run also reported an index-select device assert before
the scheduler shutdown. The trace landed in Qwen3 model/layernorm execution, not
inside ONNX Runtime export or artifact loading. This should be tracked as a
separate DP/concurrency investigation.

## Artifacts

Local outputs from the validation runs:

- `/tmp/qwen14_tp2_cuda_onnx_cuda_graph_wired_bench.json`
- `/tmp/qwen14_dp2_cuda_onnx_cuda_graph_wired_seq_bench.json`
- `/tmp/sglang-qwen14-tp2-onnx-runtime-wired/sglang.srt.sampling.penaltylib.repetition_penalty.apply_scaling_penalties.onnx`
- `/tmp/sglang-qwen14-tp2-onnx-runtime-wired/sglang.srt.sampling.penaltylib.repetition_penalty.apply_scaling_penalties.metadata.json`
- `/tmp/sglang-qwen14-dp2-onnx-runtime-wired-seq/sglang.srt.sampling.penaltylib.repetition_penalty.apply_scaling_penalties.onnx`
- `/tmp/sglang-qwen14-dp2-onnx-runtime-wired-seq/sglang.srt.sampling.penaltylib.repetition_penalty.apply_scaling_penalties.metadata.json`

Generated `.pt2`, `.onnx`, and `.metadata.json` files are intentionally not
checked into the repository.

## Reproduction Notes

The local environment used `uv` without installing the editable project. First
generate a dependency file from `python/pyproject.toml`:

```bash
python - <<'PY'
import tomllib

project = tomllib.load(open("python/pyproject.toml", "rb"))["project"]
deps = list(project.get("dependencies", []))
deps.extend(project.get("optional-dependencies", {}).get("test", []))
open("/tmp/sglang-pyproject-reqs.txt", "w").write("\n".join(deps) + "\n")
PY
```

Then build the CUDA library path used by ONNX Runtime:

```bash
LIB_PATHS=$(PYTHONPATH=python:. uv run --no-project \
  --with-requirements /tmp/sglang-pyproject-reqs.txt python - <<'PY'
import pathlib
import sys

paths = []
for path in sys.path:
    root = pathlib.Path(path) / "nvidia"
    if root.exists():
        paths.extend(str(lib) for lib in root.glob("*/lib"))
print(":".join(paths))
PY
)
```

Run ONNX-enabled servers with:

```bash
PYTHONPATH=python:. \
LD_LIBRARY_PATH="$LIB_PATHS:${LD_LIBRARY_PATH:-}" \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
SGLANG_PLATFORM=cuda_onnx \
SGLANG_EXPORT_ARTIFACT_MODE=build_if_missing \
uv run --no-project --with-requirements /tmp/sglang-pyproject-reqs.txt \
  --with onnxruntime-gpu \
  --with onnxscript \
  --with nvidia-cublas-cu12 \
  --with nvidia-cuda-runtime-cu12 \
  --with nvidia-curand-cu12 \
  --with nvidia-cufft-cu12 \
  python -m sglang.launch_server ...
```

## Future Directions

- Make ONNX Runtime advertise CUDA graph safety only after non-mutating outputs
  are backed by true CUDA I/O binding with preallocated device outputs.
- Add runtime requirements for collective behavior, such as rank-local only,
  collective-aware, or host-side synchronization requirements.
- Store enough distributed context in metadata for load-time validation across
  TP/DP/PP/EP topology changes.
- Split artifact keys by topology when a graph is rank-local but shape or
  constants differ by rank.
- Fix the TP custom all-reduce device mismatch so TP ONNX smoke does not need
  `--disable-custom-all-reduce`.
- Investigate the DP concurrent decode device assert separately from ONNX export
  policy. A minimal non-ONNX DP reproduction should determine whether this is a
  general DP/CUDA graph issue or introduced by the platform path.
