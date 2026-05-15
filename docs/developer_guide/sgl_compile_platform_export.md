# Platform-Controlled Torch Compile And Export

This note documents the platform compile/export prototype added for SRT
platforms. It introduces a single decorator, `@sgl_compile`, so individual
callsites can be compiled, skipped, or exported according to the active
`SRTPlatform`.

## Design

The platform contract now includes three compile/export hooks:

- `torch_compile_strategy()`: returns `compile`, `noop`, or `export`.
- `torch_compile_defaults()`: returns default `TorchCompileConfig` values for
  the platform.
- `make_exported_program_callable()`: converts a `torch.export.ExportedProgram`
  into a runtime callable for platforms that run exported artifacts.

`@sgl_compile` is lazy. The wrapped callable is resolved on first use, after the
active platform is known. The decorator merges library defaults, platform
defaults, and callsite options, then either:

- returns the original function for `noop`,
- calls `torch.compile` for `compile`, or
- captures a `torch.export.ExportedProgram` for `export`.

In-tree platforms are now represented by explicit `SRTPlatform` subclasses in
`sglang.srt.platforms.builtin`. `SGLANG_PLATFORM` can select either an in-tree
platform such as `cuda` or `cuda_onnx`, or an out-of-tree plugin registered via
the existing platform entry point group.

`CudaOnnxPlatform` demonstrates the export path. It exports decorated callsites
to ONNX, loads them with ONNX Runtime, and executes them through
`CUDAExecutionProvider` when available. It is intentionally scoped to decorated
callsite kernels, not full-model ONNX export.

## In-Place Outputs

Some migrated callsites mutate inputs and return `None`. For example,
`apply_scaling_penalties(logits, scaling_penalties)` updates `logits` in place.
`TorchCompileConfig.copy_output_to_arg_index` handles this by returning the
mutated argument from the export wrapper and copying the runtime output back
into the original argument.

## Dynamic Shapes

The Qwen3-14B benchmark exposed a dynamic-shape issue: the first ONNX export of
`apply_scaling_penalties` could capture batch size 1, then fail when the SGLang
scheduler batched later requests at batch size 3.

For dynamic exports, `@sgl_compile` now infers dynamic tensor shape specs for
positional tensor inputs. Batch-1 tensor examples are promoted during export so
PyTorch does not specialize that dimension to a constant. The runtime still uses
the real request tensors.

The exported Qwen3-14B ONNX artifact was verified to have symbolic input and
output dimensions:

```text
args_0 ['s33', 's50']
args_1 ['s33', 's50']
out slice_scatter ['s33', 's50']
```

## Qwen3-14B Benchmark Results

Hardware and model:

- GPU: NVIDIA H100 80GB HBM3
- Model: `Qwen/Qwen3-14B`
- Backend comparison: default CUDA platform vs `SGLANG_PLATFORM=cuda_onnx`
- Server flags: `--cuda-graph-max-bs 4 --disable-piecewise-cuda-graph`
- Request shape: prompt length about 27 tokens, `max_new_tokens=64`
- Sampling: `temperature=0.0`, `repetition_penalty=1.12`, `ignore_eos=true`
- Workload: 3 warmups, 12 sequential requests, 16 requests at concurrency 4

| Mode | Seq median latency | Seq completion tok/s | C4 median latency | C4 completion tok/s |
| --- | ---: | ---: | ---: | ---: |
| CUDA | `0.705s` | `90.8` | `0.718s` | `312.9` |
| `cuda_onnx` | `0.784s` | `81.6` | `0.865s` | `290.8` |

The ONNX path is slower in this benchmark: about 10% lower sequential throughput
and 7% lower concurrency-4 throughput. This is expected for the current
prototype because ONNX is only running a small sampling-side penalty kernel and
the runtime path copies tensors through CPU/NumPy around ONNX Runtime. This is
not measuring a full-model ONNX deployment.

## Artifacts

Local benchmark outputs from the H100 run:

- `/tmp/qwen14_cuda_bench.json`
- `/tmp/qwen14_cuda_onnx_bench.json`
- `/tmp/sglang-qwen14-onnx-artifacts-dyn2/sglang.srt.sampling.penaltylib.repetition_penalty.apply_scaling_penalties.onnx`

The ONNX artifact is generated output and is not checked into the repository.

## Reproduction

The commands below assume the repository root as the working directory. They use
`uv` and the dependencies declared by `python/pyproject.toml`, then add ONNX
Runtime packages only for the `cuda_onnx` run.

If the local environment cannot build the editable package because Rust is not
installed, generate a temporary requirements file from the project metadata and
use `PYTHONPATH=python:.`:

```bash
python - <<'PY'
import tomllib

project = tomllib.load(open("python/pyproject.toml", "rb"))["project"]
deps = list(project.get("dependencies", []))
deps.extend(project.get("optional-dependencies", {}).get("test", []))
open("/tmp/sglang-pyproject-reqs.txt", "w").write("\n".join(deps) + "\n")
PY
```

Build a CUDA library path from the `uv` overlay:

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

Run the default CUDA server:

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=python:. \
LD_LIBRARY_PATH="$LIB_PATHS:${LD_LIBRARY_PATH:-}" \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
uv run --no-project --with-requirements /tmp/sglang-pyproject-reqs.txt \
  python -m sglang.launch_server \
  --model-path Qwen/Qwen3-14B \
  --host 127.0.0.1 \
  --port 30000 \
  --cuda-graph-max-bs 4 \
  --disable-piecewise-cuda-graph \
  --trust-remote-code
```

Run the ONNX platform server:

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=python:. \
LD_LIBRARY_PATH="$LIB_PATHS:${LD_LIBRARY_PATH:-}" \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
SGLANG_PLATFORM=cuda_onnx \
SGLANG_EXPORT_DIR=/tmp/sglang-qwen14-onnx-artifacts \
uv run --no-project --with-requirements /tmp/sglang-pyproject-reqs.txt \
  --with onnxruntime-gpu \
  --with onnxscript \
  --with nvidia-cublas-cu12 \
  --with nvidia-cuda-runtime-cu12 \
  --with nvidia-curand-cu12 \
  --with nvidia-cufft-cu12 \
  python -m sglang.launch_server \
  --model-path Qwen/Qwen3-14B \
  --host 127.0.0.1 \
  --port 30001 \
  --cuda-graph-max-bs 4 \
  --disable-piecewise-cuda-graph \
  --trust-remote-code
```

Run the client benchmark against either port by changing `URL`:

```bash
uv run --no-project --with requests python - <<'PY'
import concurrent.futures
import json
import statistics
import time
import urllib.request

URL = "http://127.0.0.1:30000/generate"
PROMPT = (
    "Write a concise technical explanation of why benchmark methodology matters "
    "for comparing inference backends. Include one concrete example."
)
PARAMS = {
    "temperature": 0.0,
    "max_new_tokens": 64,
    "repetition_penalty": 1.12,
    "ignore_eos": True,
}


def call(i):
    payload = {
        "text": PROMPT + f"\nRequest id: {i}",
        "sampling_params": PARAMS,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=240) as response:
        body = response.read().decode()
    elapsed = time.perf_counter() - start
    obj = json.loads(body)
    meta = obj.get("meta_info", {})
    return {
        "latency_s": elapsed,
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
    }


def summarize(name, rows, wall=None):
    latencies = [row["latency_s"] for row in rows]
    completion_tokens = sum(row["completion_tokens"] or 0 for row in rows)
    total = wall if wall is not None else sum(latencies)
    return {
        "name": name,
        "requests": len(rows),
        "mean_latency_s": statistics.mean(latencies),
        "median_latency_s": statistics.median(latencies),
        "min_latency_s": min(latencies),
        "max_latency_s": max(latencies),
        "completion_tokens": completion_tokens,
        "wall_s": total,
        "completion_tok_per_s": completion_tokens / total,
    }


warmup = [call(f"warm-{i}") for i in range(3)]
sequential = [call(f"seq-{i}") for i in range(12)]
start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    concurrent_rows = list(executor.map(call, [f"conc-{i}" for i in range(16)]))
wall = time.perf_counter() - start

print(
    json.dumps(
        {
            "warmup": summarize("warmup", warmup),
            "sequential": summarize("sequential", sequential),
            "concurrency4": summarize("concurrency4", concurrent_rows, wall),
        },
        indent=2,
    )
)
PY
```

Run targeted unit coverage:

```bash
PYTHONPATH=python:. uv run --no-project \
  --with-requirements /tmp/sglang-pyproject-reqs.txt \
  python -m pytest \
  test/registered/unit/platforms/test_platform_interface.py \
  test/registered/unit/test_torch_compile_decorator.py \
  test/registered/unit/test_torch_compile_onnx.py
```
