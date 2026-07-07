# Running SGLang locally on macOS (arm64) — dev loop

**Status:** dev-only, not for upstream. **Last updated:** 2026-07-06.

This branch adds small, Linux-only-path fallbacks so `sglang.launch_server` can
boot on an Apple-silicon Mac (CPU device) for local smoke-testing / benching
against MLX, without a Linux box or the CUDA-only `sgl-kernel` wheel. Every code
change is a no-op on Linux — it only fires when a Linux-only API raises.

## What this branch changes

`python/sglang/srt/utils/common.py`
- `parse_lscpu_topology`: on `FileNotFoundError` (no `lscpu`), synthesize a
  single-node, one-CPU-per-core topology.
- `get_physical_cpus_by_numa`: `psutil.Process().cpu_affinity()` is not
  implemented on macOS; fall back to "all CPUs".

`python/sglang/srt/model_executor/model_runner.py`
- `init_torch_distributed`: wrap the `sgl_kernel` CPU ops
  (`init_cpu_threads_env`, SHM-AllReduce `initialize`) in `try/except
  AttributeError`, so a missing `sgl-kernel` wheel doesn't kill the CPU path.
  (Reconciled 2026-07-06 against current `main`, which removed the
  `sgl_kernel::shm_allgather` `register_fake` block — the fallback now wraps
  only the ops that still exist.)

## Prerequisites

- A venv with a macOS `torch` build (MPS is fine; the CPU path is used):
  `python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"`
- A **standard HF-format** model cached locally (not MLX format). Small is best,
  e.g. `Qwen/Qwen2.5-0.5B-Instruct`.

## Run the HTTP server on CPU

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TORCH_COMPILE_DISABLE=1 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu \
  --host 127.0.0.1 --port 30020 \
  --attention-backend torch_native \
  --mem-fraction-static 0.3 \
  --context-length 4096 \
  --max-running-requests 1
```

Then: `curl -s localhost:30020/health` → `200`, and `/v1/chat/completions` works.

### Why `TORCH_COMPILE_DISABLE=1` is required

Without it, the **scheduler crashes on the first real request** with:

```
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
TypeError: 'module' object is not callable
```

`apply_scaling_penalties` (in `sampling/sampling_batch_info.py`) is
`@torch.compile`-decorated; the inductor backend isn't usable on macOS.
`TORCH_COMPILE_DISABLE=1` forces eager and the crash goes away. Use
`--attention-backend torch_native` too (flashinfer/triton are CUDA-only).

## Run the native gRPC server (`--grpc-port`)

The native gRPC server is a PyO3 Rust extension (`sglang.srt.grpc._core`) built
from `rust/sglang-grpc/`. A normal macOS install won't have it, so build it once:

```bash
# 1. Build the extension against YOUR venv's Python, with the macOS
#    dynamic-lookup linker flag (a plain `cargo build` fails to link the
#    undefined _Py* symbols because pyo3's `extension-module` doesn't link libpython).
cd rust/sglang-grpc
PYO3_PYTHON=/path/to/.venv/bin/python \
RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" \
cargo build --release
#    (needs protoc on PATH for the tonic build.rs proto codegen)

# 2. Install the built dylib as the importable extension module.
cp target/release/lib_core.dylib \
   ../../python/sglang/srt/grpc/_core.so
```

Verify: `python -c "from sglang.srt.grpc import _core; print(_core.start_server.__text_signature__)"`
→ `(host, port, runtime_handle, worker_threads=4, response_channel_capacity=64, response_timeout_secs=300)`

Launch (native gRPC runs alongside HTTP):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_COMPILE_DISABLE=1 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu \
  --host 127.0.0.1 --port 30020 \
  --grpc-port 30021 \
  --attention-backend torch_native \
  --mem-fraction-static 0.3 --context-length 4096 --max-running-requests 1
```

Look for `Native gRPC server started on 127.0.0.1:30021` and
`_core::server: gRPC server listening on 127.0.0.1:30021`.

### Smoke-test the gRPC endpoint

```bash
# Generate Python stubs from the proto (into a temp dir to avoid the real
# `sglang` package colliding with the generated `sglang.runtime.v1` module).
python -m grpc_tools.protoc -I proto \
  --python_out=/tmp/grpc_client --grpc_python_out=/tmp/grpc_client \
  proto/sglang/runtime/v1/sglang.proto
```

Then a minimal client (run with `PYTHONPATH=/tmp/grpc_client` only):

```python
import grpc
from sglang.runtime.v1 import sglang_pb2 as pb, sglang_pb2_grpc as pbg
stub = pbg.SglangServiceStub(grpc.insecure_channel("127.0.0.1:30021"))
print(stub.HealthCheck(pb.HealthCheckRequest()).healthy)          # True
print(stub.GetModelInfo(pb.GetModelInfoRequest()).model_path)
req = pb.TextGenerateRequest(
    text="The capital of France is",
    sampling_params=pb.SamplingParams(temperature=0.0, max_new_tokens=16),
    stream=True,
)
for r in stub.TextGenerate(req, timeout=120):
    if r.finished:
        print(repr(r.text)); break
```

Verified 2026-07-06: `HealthCheck` → `True`, `GetModelInfo` →
`Qwen/Qwen2.5-0.5B-Instruct`, `TextGenerate` streamed
`" Paris. It is the largest city in Europe..."`.

> Note: `_core.so` is gitignored; rebuild it after any change to the proto or the
> `rust/sglang-grpc/` crate.
