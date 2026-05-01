# Test-failure report: `test_nvidia_nemotron_3_nano.py`

This is a running report. Each time the test is re-run with a change, a new
section is appended. The "Fix log" section at the end tracks every change
applied to the test or codebase, in order.

---

## Run 1 (initial) — `AssertionError: Page size must be 1 for MambaRadixCache v1, got 128`

### Exit path

`popen_launch_server` timed out waiting for the server to come up healthy
because every TP worker died during `Scheduler.__init__`. The assertion comes
from `sglang/srt/mem_cache/mamba_radix_cache.py:436-438`:

```
AssertionError: Page size must be 1 for MambaRadixCache v1, got 128
```

That assertion is raised during `init_cache_with_memory_pool()`
(`sglang/srt/managers/scheduler.py:856`).

### The underlying conflict (seen clearly in the server log)

Three log lines make the root cause unambiguous — they happen in this order
during `ServerArgs` post-processing:

```
NemotronHForCausalLM with radix cache requires page_size=1 in the current
Mamba scheduling mode (no_buffer), but got 64. Automatically setting page_size=1.

Disabling overlap schedule since mamba no_buffer is not compatible with overlap
schedule, try to use --disable-radix-cache if overlap schedule is necessary

Intel XPU attention backend only supports page_size of 32, 64 or 128,
changing page_size from 1 to 128.
```

So the code first coerces `page_size = 1` to satisfy the Mamba no-buffer
radix-cache path (`sglang/srt/server_args.py:2275-2281`), then *later* coerces
`page_size = 128` to satisfy the `intel_xpu` attention backend
(`sglang/srt/server_args.py:2531-2536`). The two coercions silently cancel each
other, and `MambaRadixCache.__init__` rejects the final `page_size=128`
(`mamba_radix_cache.py:435-438`).

### Why this model and why XPU specifically

- **Model**: `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` uses the
  `NemotronHForCausalLM` architecture (hybrid Mamba + attention + MoE).
  Hybrid-SSM models go down the `self.is_hybrid_ssm → MambaRadixCache` branch
  (`scheduler.py:853-856`) rather than the normal `RadixCache`. The
  `MambaRadixCache v1` path requires `page_size == 1` when
  `enable_mamba_extra_buffer` is false, which is the default outside CUDA
  (the extra-buffer mode is explicitly CUDA-only, `server_args.py:2257-2259`).
- **XPU**: The `intel_xpu` attention backend hard-refuses `page_size=1`. So on
  XPU the registered CUDA config ("just use the defaults") cannot be copied
  verbatim — the CUDA test works because on CUDA the backend doesn't force
  page_size≠1.

### Secondary observations (not the cause of the failure)

- `avail mem=1.80 GB` after weights (14.76 GB) + Mamba cache (~2.9 GB) + KV
  cache (~3.24 GB) on a 24 GB partition. `--mem-fraction-static 0.92` is
  aggressive for a hybrid-SSM model because the Mamba SSM state
  (`2.86 GB per rank`) is counted separately from weights. If the radix-cache
  issue is fixed, the process could still OOM on CUDA-graph capture.
- XPU auto-disables piecewise CUDA graph (`XPU platform does not support
  piecewise CUDA graph`) — fine, but the `--cuda-graph-max-bs 8` setting
  should still be honored.
- `BaseImageProcessorFast` deprecation + `vllm`/`tvm_ffi` import-error
  warnings are benign.

### Suggested fixes (pick one; listed in order of preference)

#### Fix A — Disable radix cache (minimal, recommended)

Disabling radix cache takes the scheduler down the `ChunkCache` path
(`scheduler.py:799-807`), which does not require `page_size=1`. This is also
the remedy suggested by SGLang's own warning: *"try to use
`--disable-radix-cache` if overlap schedule is necessary"*.

Proposed diff to `XPU_SERVER_ARGS`:

```python
XPU_SERVER_ARGS = [
    "--device", "xpu",
    "--tp=4",
    "--trust-remote-code",
    "--disable-overlap-schedule",
    "--disable-radix-cache",           # NEW: avoid MambaRadixCache's page_size==1 requirement
    "--page-size", "64",                # still legal for intel_xpu (32/64/128)
    "--attention-backend", "intel_xpu",
    "--model-impl", "sglang",
    "--tool-call-parser", "qwen3_coder",
    "--reasoning-parser", "deepseek-r1",
    "--mem-fraction-static", "0.85",    # NEW: from 0.92 — more headroom for Mamba SSM + KV
    "--context-length", "8192",
    "--chunked-prefill-size", "1024",
    "--max-running-requests", "8",
    "--cuda-graph-max-bs", "8",
]
```

Behavior: `disable_radix_cache=True` + `chunked_prefill_size=1024` routes the
scheduler to `ChunkCache` (non-hybrid-SWA branch). `intel_xpu` then gets its
required `page_size=64`. This mirrors the pattern used for the `trtllm_mha`
fallback already living in `server_args.py:2288-2294`.

#### Fix B — Switch to `triton` attention backend + `page_size=1`

If Option A causes throughput issues, `triton` tolerates `page_size=1`, which
keeps the `MambaRadixCache` but drops the XMX attention kernel — so this
defeats the point of smoke-testing on XPU. Fallback only.

#### Fix C — Same as A but pin `--mamba-scheduler-strategy no_buffer` explicitly

Functionally identical to A; just more self-documenting.

### Full log

`/tmp/nemotron_test_output.log`

---

## Run 2 (after applying Fix A on a loaded shared node) — `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` during weight load

### Changes applied

```python
XPU_SERVER_ARGS = [
    "--device", "xpu",
    "--tp=4",
    "--trust-remote-code",
    "--disable-overlap-schedule",
    "--disable-radix-cache",           # NEW
    "--page-size", "64",
    "--attention-backend", "intel_xpu",
    "--model-impl", "sglang",
    "--tool-call-parser", "qwen3_coder",
    "--reasoning-parser", "deepseek-r1",
    "--mem-fraction-static", "0.85",    # CHANGED: 0.92 → 0.85
    "--context-length", "8192",
    "--chunked-prefill-size", "1024",
    "--max-running-requests", "8",
    "--cuda-graph-max-bs", "8",
]
```

### Result

The MambaRadixCache `page_size` assertion was no longer hit (Fix A confirmed
effective for the original bug). However, the server now died during weight
loading at shard 1/13:

```
File "sglang/srt/layers/vocab_parallel_embedding.py", line 468, in weight_loader
    param[: loaded_weight.shape[0]].data.copy_(loaded_weight)
RuntimeError: Native API failed. Native API returns: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
```

### Root cause (environment, not the test)

This run happened while another user's `gemma-4-31B` server was already
running on the same node (PID 349337), holding all 4 XPUs at ~95% utilization
with `--mem-fraction-static 0.92`. At weight-load time only a small slice of
per-device HBM was free on each XPU, so even our first embedding shard
couldn't be copied in. This was confirmed by `ps -ef` — four
`sglang::scheduler_TP{0..3}` processes from a different PID tree were active
when our schedulers started.

This is **not** an issue with the test config itself; it is an environmental
OOM from device contention. No test-level change was necessary after
confirming the conflicting run was the only consumer.

### Full log

`/tmp/nemotron_test_output_fixA.log`

---

## Run 3 (clean XPUs, Fix A still applied) — `NameError: causal_conv1d_fn_triton is not defined` in the Mamba forward path

### Result

With the node free of other workloads, the server fully boots this time:

```
[TP0..3] Load weight end. elapsed=36.9 s, type=NemotronHForCausalLM,
         avail mem=7.95 GB, mem usage=14.76 GB
[TP0..3] Mamba Cache is allocated. max_mamba_cache_size: 8,
         conv_state size: 0.00 GB, ssm_state size: 0.10 GB
[TP0..3] KV Cache is allocated. #tokens: 1557504,
         K size: 2.23 GB, V size: 2.23 GB
[TP0..3] Memory pool end. avail mem=3.39 GB
[TP0]    max_total_num_tokens=1557504, chunked_prefill_size=1024,
         max_prefill_tokens=16384, max_running_requests=8,
         context_len=8192, available_gpu_mem=3.39 GB
[INFO]   Uvicorn running on http://127.0.0.1:21000
[INFO]   127.0.0.1:44034 - "GET /model_info HTTP/1.1" 200 OK
```

So Fix A fully resolves the radix-cache conflict, and the Fix-A memory
tuning is adequate (3.39 GB free per rank post allocation — headroom is
tight but sufficient). The server then fails during the health-probe warm-up
forward pass:

```
File "sglang/srt/model_executor/model_runner.py", line 2809, in forward_extend
    self.model.forward(...)
File "sglang/srt/models/nemotron_h.py", line 437, in forward
    output = self._forward_mamba(hidden_states, forward_batch)
File "sglang/srt/models/nemotron_h.py", line 410, in _forward_mamba
    attn_backend.linear_attn_backend.forward(...)
File "sglang/srt/layers/attention/hybrid_linear_attn_backend.py", line 705, in forward
    return mixer.forward(...)
File "sglang/srt/layers/attention/mamba/mamba.py", line 507, in forward
    else causal_conv1d_fn_triton
NameError: name 'causal_conv1d_fn_triton' is not defined
```

### Root cause

`sglang/srt/layers/attention/mamba/mamba.py:34-51` gates the import of
`causal_conv1d_fn` and `causal_conv1d_fn_triton` behind device guards:

```python
from sglang.srt.utils import is_cpu, is_cuda, is_npu, set_weight_attrs
...
if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn, ...
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_fn as causal_conv1d_fn_triton, ...
    )
elif is_npu():
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu as causal_conv1d_fn, ...
    )
# NB: no `elif is_xpu(): ...` branch
```

On XPU *neither* branch runs, so `causal_conv1d_fn` and
`causal_conv1d_fn_triton` are never bound. When the Mamba mixer's first
prefill hits `mamba.py:503-507`:

```python
ccfn = (
    causal_conv1d_fn
    if not use_triton_causal_conv
    else causal_conv1d_fn_triton
)
```

Python raises `NameError` for whichever branch the runtime selects
(`causal_conv1d_fn_triton` in this run, because the Mamba backend is the
Triton implementation).

Checking the two Python modules the CUDA branch imports:

- `sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` is **pure
  Triton** (the header explicitly says "adapted from vllm/…"). It does not
  depend on CUDA-only bindings. Triton already runs on XPU in this repo (see
  `linear_attn_backend: decode=triton, prefill=triton` in the log and the
  `--mamba-backend triton` default).
- `sglang/srt/layers/attention/mamba/causal_conv1d.py` is the *native* CUDA
  kernel wrapper; it gates on `_HAS_SGL_KERNEL` and silently falls back to
  the triton implementation when the sgl-kernel CUDA binding is missing
  (see the `_causal_conv1d_fn_triton` fallback at line 80 of that file).

So the fix should be to extend the device gate so XPU imports both names
from the Triton module (the Triton fallback is the same path that CUDA takes
when its native kernel is unavailable).

### Suggested fix (code change, *not* a test-config change)

This one needs a small change to the SGLang source, not just the XPU test.

Apply to `sglang/python/sglang/srt/layers/attention/mamba/mamba.py` around
lines 32-51:

```python
from sglang.srt.utils import is_cpu, is_cuda, is_npu, is_xpu, set_weight_attrs

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_fn as causal_conv1d_fn_triton,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_update as causal_conv1d_update_triton,
    )
elif is_npu():
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu as causal_conv1d_fn,
    )
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_update_npu as causal_conv1d_update,
    )
elif is_xpu():
    # NEW: XPU has no native causal_conv1d kernel yet; use the portable Triton
    # implementation for both the "native" and the "_triton" entry points so
    # `causal_conv1d_fn` / `causal_conv1d_fn_triton` are always bound on XPU.
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_fn as causal_conv1d_fn,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_fn as causal_conv1d_fn_triton,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_update as causal_conv1d_update,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_update as causal_conv1d_update_triton,
    )
```

Rationale:

- `causal_conv1d_triton.py` is pure Triton/PyTorch and has no CUDA-only
  dependency, so it is the natural fallback for XPU.
- Binding both `causal_conv1d_fn` and `causal_conv1d_fn_triton` to the same
  Triton implementation matches the semantics on CUDA when the native
  sgl-kernel binding is missing (see `causal_conv1d.py:80`).
- This keeps the test config exactly as in Fix A — no further
  test-side changes needed.

A tiny test-side guard could also be added: select
`--mamba-backend triton` explicitly if it isn't already implied, so the
Triton path is deterministic on XPU. In the current log the server already
prints `Linear attention kernel backend: decode=triton, prefill=triton`, so
the effective default is already Triton. Still, being explicit is cheap:

```python
    "--mamba-backend", "triton",        # optional but explicit
```

### Full log

`/tmp/nemotron_test_output_fixA_retry.log`

---

## Run 4 (Fix 2 applied) — `RuntimeError: PyTorch was compiled without CUDA support` inside `mamba_chunk_scan_combined`

### Result

With the `elif is_xpu():` branch added to
`sglang/srt/layers/attention/mamba/mamba.py`, Run 4 *no longer* hits the
Run 3 `NameError`. The server reaches the same healthy startup state as
Run 3 (weights loaded, Mamba cache + KV cache allocated, Uvicorn up and
`GET /model_info → 200 OK`) and progresses further into the warm-up
forward pass. It then dies inside the Mamba SSD op itself:

```
File "sglang/srt/model_executor/model_runner.py", line 2882, in forward_extend
    self.model.forward(...)
File "sglang/srt/models/nemotron_h.py", line 758, in forward
File "sglang/srt/models/nemotron_h.py", line 642, in forward
File "sglang/srt/models/nemotron_h.py", line 448, in forward
File "sglang/srt/models/nemotron_h.py", line 416, in _forward_mamba
    attn_backend.linear_attn_backend.forward(...)
File "sglang/srt/layers/attention/hybrid_linear_attn_backend.py", line 707, in forward
    return mixer.forward(...)
File "sglang/srt/layers/attention/mamba/mamba.py", line 566, in forward
    varlen_state = mamba_chunk_scan_combined(...)
File "sglang/srt/layers/attention/mamba/ops/ssd_combined.py", line 230, in mamba_chunk_scan_combined
    _mamba_chunk_scan_combined_fwd(...)
File "sglang/srt/layers/attention/mamba/ops/ssd_combined.py", line 98, in _mamba_chunk_scan_combined_fwd
    dA_cumsum, dt = _chunk_cumsum_fwd(...)
File "sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py", line 463, in _chunk_cumsum_fwd
    with torch.cuda.device(dt.device.index):
File ".../torch/cuda/__init__.py", line 533, in __enter__
    self.prev_idx = torch.cuda._exchange_device(self.idx)
File ".../torch/cuda/__init__.py", line 135, in _exchange_device
    raise RuntimeError("PyTorch was compiled without CUDA support")
RuntimeError: PyTorch was compiled without CUDA support
```

All four TP ranks hit the same exception at `ssd_chunk_state.py:463`,
the scheduler SIGQUITs its children, and `popen_launch_server` reports
the server process exited with code `-9`, surfacing as
`setUpClass` `ERROR` — not a `NameError` anymore.

### Root cause

`sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py:463` (and very
likely sibling call sites in `ssd_combined.py` / `ssd_state.py`) opens a
`torch.cuda.device(dt.device.index)` context *unconditionally*, even when
the tensors live on `xpu:*`. On this host the env is
`sgl-xpu-d` (XPU-only torch build), so `torch.cuda._exchange_device` is
unavailable and raises immediately. This guard exists so that Triton
kernels in the SSD ops get launched on the right device context — the
correct XPU-aware replacement is to dispatch through
`torch.xpu.device(...)` when the tensor lives on XPU (or use
`torch.get_device_module(dt).device(dt.device.index)` /
`torch.device(dt.device)` + a no-op context when neither is needed —
Triton on XPU already honours the tensor's device).

Fix 2 is confirmed effective (the NameError is gone, `decode=triton,
prefill=triton` backend is picked, the mixer forward starts). The
remaining failure is a *different* defect in the Mamba SSD ops and is
out of scope for Fix 2 — it needs a Fix 3 that replaces the hardcoded
`torch.cuda.device(...)` with a device-aware context manager in the
`ssd_*` modules.

### Full log

`/tmp/nemotron_test_output_fix2.log`

---

## Run 5 (Fix 3 applied) — `AssertionError: activation = relu2 is not supported.` in the XPU fused-MoE FFN

### Result

With the device-aware `_device_context(...)` helper replacing the
hard-coded `torch.cuda.device(...)` in `ssd_chunk_state.py`,
`ssd_state_passing.py`, and `ssd_bmm.py`, Run 5 progresses past the Run 4
failure site. The Mamba prefill forward pass now runs end-to-end:
`_chunk_cumsum_fwd`, `_chunk_state_fwd`, `_state_passing_fwd`,
`_bmm_chunk_fwd`, `_chunk_scan_fwd`, and `chunk_state_varlen` all
complete — no "PyTorch was compiled without CUDA support" exception
anywhere in the log. Control leaves `mamba_chunk_scan_combined` and
re-enters the decoder layer stack, where the next (non-Mamba) layer —
the MoE feed-forward — raises a new, distinct error:

```
File "sglang/srt/models/nemotron_h.py", line 379, in forward
    hidden_states = self.mixer.forward(hidden_states)
File "sglang/srt/models/nemotron_h.py", line 278, in forward
    final_hidden_states, shared_output = self._forward_core(hidden_states)
File "sglang/srt/models/nemotron_h.py", line 234, in _forward_core
    return self._forward_core_normal(hidden_states)
File "sglang/srt/models/nemotron_h.py", line 249, in _forward_core_normal
    final_hidden_states = self.experts(hidden_states, topk_output)
File "sglang/srt/layers/moe/fused_moe_triton/layer.py", line 1055, in forward
File "sglang/srt/layers/moe/fused_moe_triton/layer.py", line 1065, in forward_impl
File "sglang/srt/layers/moe/fused_moe_triton/layer.py", line 1086, in run_moe_core
File "sglang/srt/layers/quantization/unquant.py", line 393, in apply
File "sglang/srt/layers/utils/multi_platform.py", line 83, in forward
File "sglang/srt/layers/quantization/unquant.py", line 576, in forward_xpu
    assert moe_runner_config.activation in [
AssertionError: activation = relu2 is not supported.
```

All four TP ranks SIGQUIT on the same assertion; `popen_launch_server`
times out with `Server process exited with code -9`.

### Root cause

`UnquantizedFusedMoEMethod.forward_xpu` in
`sglang/python/sglang/srt/layers/quantization/unquant.py:565-579`
hard-asserts that the MoE activation is one of `{"silu", "gelu"}`:

```python
def forward_xpu(self, layer, dispatch_output):
    ...
    assert moe_runner_config.activation in [
        "silu",
        "gelu",
    ], f"activation = {moe_runner_config.activation} is not supported."
```

The Nemotron-H MoE uses `relu2` (squared-ReLU). Other platforms do
support this — the same file, lines 437-439, shows the CUDA path
branches on `activation == "relu2"` and calls into a cutlass MoE runner
with `activation_type=relu2`. The XPU `forward_xpu` was written for
Qwen/DeepSeek-class MoEs that stuck to `silu`/`gelu` and the
Nemotron-H pattern simply wasn't contemplated.

Fix 3 is confirmed effective for its intended bug: the Run 4
`torch.cuda.device(...)` → `CUDA support` RuntimeError is gone, and the
first end-to-end Mamba prefill completes for all four TP ranks. The
remaining failure is in a *different* subsystem (fused-MoE on XPU, not
Mamba SSD) and will need a Fix 4 that either:

1. Teaches `UnquantizedFusedMoEMethod.forward_xpu` to accept `relu2` by
   dispatching it to the sgl-kernel-xpu `fused_experts` entry point (if
   that kernel exposes a relu2 path), or
2. Falls back to a Triton / manual relu2 MoE path on XPU, or
3. Fuses it pre-/post-kernel (apply `x*x*(x>0)` on the activations
   around a silu-less matmul — adequate for correctness though not
   throughput-optimal).

This is out of scope for Fix 3 and is intentionally deferred.

### Full log

`/tmp/nemotron_test_output_fix3.log`

---

## Fix log (chronological)

### Fix 1 — `Run 1 → Run 2`: disable radix cache + loosen memory fraction
- **When applied**: between Run 1 and Run 2.
- **What changed**: edited `test/srt/xpu/test_nvidia_nemotron_3_nano.py`
  `XPU_SERVER_ARGS`:
  - **Added** `--disable-radix-cache` to route the scheduler to
    `ChunkCache` instead of `MambaRadixCache`, side-stepping the
    `page_size==1` assertion that is mutually exclusive with the
    `intel_xpu` attention backend's required page sizes (32/64/128).
  - **Lowered** `--mem-fraction-static` from `0.92` → `0.85` to give the
    Mamba SSM state + KV pool + activation buffers headroom on a 24 GB
    Arc Pro B60 partition (Run 1 landed with only `1.80 GB` free post
    allocation — below the watermark needed for a stable prefill).
- **Outcome**: Fixes the original `MambaRadixCache` assertion
  (confirmed by the log — `disable_radix_cache=True` and no MambaRadixCache
  instantiation in Run 2/3). Run 2 then hit a device-OOM because another
  user's server was holding the XPUs; Run 3 confirmed the config works on
  clean XPUs up through server startup.
- **Status**: ✅ Applied and effective for its intended bug.
- **File**: `test/srt/xpu/test_nvidia_nemotron_3_nano.py`

### Fix 2 — `Run 3 → Run 4`: XPU branch for `causal_conv1d_fn` imports
- **Motivation**: Run 3 failure — `NameError: causal_conv1d_fn_triton is
  not defined` at `mamba.py:507` on the first Mamba prefill.
- **What changed**: edited
  `sglang/python/sglang/srt/layers/attention/mamba/mamba.py`. Added
  `is_xpu` to the `sglang.srt.utils` import block (now lines 33-39) and
  inserted a new `elif is_xpu():` branch (now lines 59-74) after the
  existing `is_npu()` branch. The new branch binds
  `causal_conv1d_fn`, `causal_conv1d_fn_triton`, `causal_conv1d_update`,
  and `causal_conv1d_update_triton` — all four — from
  `sglang.srt.layers.attention.mamba.causal_conv1d_triton` (pure Triton,
  device-agnostic). Both the "native" and "_triton" names resolve to the
  same Triton function, matching what CUDA does when the native sgl-kernel
  binding is absent (see `causal_conv1d.py:80`).
- **Why it's the right fix**: the Triton implementation is already the
  CUDA-side fallback when the sgl-kernel native binding is missing, and
  Triton is already in use for the rest of the Mamba/linear-attention path
  on XPU (the log shows `Linear attention kernel backend:
  decode=triton, prefill=triton`). The absence of an XPU import branch was
  a pure oversight — no semantic change is required, only the symbol
  bindings.
- **When applied**: 2026-05-01 UTC.
- **File**: `sglang/python/sglang/srt/layers/attention/mamba/mamba.py`,
  lines 33-39 (imports) and lines 59-74 (new `elif is_xpu()` branch).
- **Status**: ✅ Applied. The `NameError` is gone — Run 4 progresses past
  `mamba.py:507` into `mamba_chunk_scan_combined` and fails later on an
  unrelated hard-coded `torch.cuda.device(...)` guard. See Run 4 below.
- **Outcome**: see Run 4 below.

### Fix 3 — `Run 4 → Run 5`: device-aware context manager in the Mamba SSD ops
- **Motivation**: Run 4 failure — `RuntimeError: PyTorch was compiled without
  CUDA support` raised by `torch.cuda._exchange_device` at
  `ssd_chunk_state.py:463` during the first Mamba prefill. On XPU the CUDA
  module's device-switching context manager cannot be entered, but the
  SSD-op Triton launchers opened it unconditionally on the tensor's device
  index.
- **What changed**: replaced every unconditional
  `with torch.cuda.device(<tensor>.device.index):` on the Mamba prefill path
  with a new module-level helper `_device_context(tensor)` that dispatches
  through `torch.get_device_module(tensor.device).device(tensor.device.index)`
  and falls back to `contextlib.nullcontext()` when the device has no
  `.device` context manager (CPU and misc. backends). The helper is defined
  once per file so each SSD op module stays self-contained (no cross-file
  import for three identical one-paragraph functions). Files and real line
  ranges after the edit:
  - `sglang/python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py`
    - helper defined at lines 11-38 (imports + `_device_context`)
    - call sites: line 484 (`_chunk_cumsum_fwd`, was 463), line 544
      (`_chunk_state_fwd`, was 523), line 620 (`chunk_state_varlen`, was 599)
  - `sglang/python/sglang/srt/layers/attention/mamba/ops/ssd_state_passing.py`
    - helper defined at lines 11-29 (`import contextlib` + `_device_context`)
    - call site: line 237 (`_state_passing_fwd`, was 217)
  - `sglang/python/sglang/srt/layers/attention/mamba/ops/ssd_bmm.py`
    - helper defined at lines 11-30 (`import contextlib` + `_device_context`)
    - call site: line 201 (`_bmm_chunk_fwd`, was 182)

  Canonical replacement pattern:

  ```python
  def _device_context(tensor: torch.Tensor):
      dev = tensor.device
      if dev.type == "cpu" or dev.index is None:
          return contextlib.nullcontext()
      dev_module = torch.get_device_module(dev)
      dev_cm = getattr(dev_module, "device", None)
      if dev_cm is None:
          return contextlib.nullcontext()
      return dev_cm(dev.index)

  # before:
  with torch.cuda.device(x.device.index):
      kernel[grid](...)

  # after:
  with _device_context(x):
      kernel[grid](...)
  ```

- **Why it's the right fix**: PyTorch already exposes
  `torch.get_device_module(device)` precisely so generic code can pick the
  right namespace (`torch.cuda` / `torch.xpu` / etc.). The old code
  hard-coded the CUDA namespace. On CUDA the new path resolves to
  `torch.cuda.device(idx)` — byte-for-byte identical behavior. On XPU it
  resolves to `torch.xpu.device(idx)` — which exists and is exactly the
  device-switching CM the Triton launcher needs. This is additive: we only
  change what happens on non-CUDA. It also mirrors the `is_arch_support_pdl`
  style guard that the DSv4 enablement used to remove the same class of
  "assume-CUDA" bug in other SGLang kernel wrappers.
- **Scope decision**: surveyed all `torch.cuda.device(` sites under
  `python/sglang/srt/layers/attention/mamba/ops/`. The prefill path that
  the Nemotron test exercises goes through
  `_mamba_chunk_scan_combined_fwd` (`ssd_combined.py:98-162`) and hits
  exactly the 5 sites fixed above (`_chunk_cumsum_fwd`, `_chunk_state_fwd`,
  `chunk_state_varlen`, `_state_passing_fwd`, `_bmm_chunk_fwd`). The
  Nemotron-H config uses `cu_seqlens` for varlen prefill, so
  `chunk_state_varlen` is on the first-prefill path (the Run 4 trace shows
  `varlen_state = mamba_chunk_scan_combined(...)` at `mamba.py:566`). All
  5 fixed preemptively so Run 5 does not regress on the next sibling.
- **Known follow-ups** (not on the prefill path, deliberately left for a
  later fix once decode is exercised):
  - `ops/mamba_ssm.py:430` — `selective_state_update` (decode only).
  - `ops/layernorm_gated.py:122` — orphaned; `Mixer2RMSNormGated` uses
    `sglang/srt/layers/attention/fla/layernorm_gated.py` instead of this
    file, so the site is dead code for the current call graph.
- **When applied**: 2026-05-01 UTC.
- **Status**: ✅ Applied. Run 5 progresses past the Run 4 failure site —
  the Mamba prefill forward pass completes for all four TP ranks — and
  fails in the downstream MoE FFN (`relu2` activation not supported by
  `UnquantizedFusedMoEMethod.forward_xpu`). See Run 5 above.
- **Outcome**: see Run 5 above.

### Summary of current status

| Aspect                            | Status |
| --------------------------------- | ------ |
| MambaRadixCache page_size clash   | ✅ Fixed by Fix 1 |
| Memory headroom                   | ✅ Fixed by Fix 1 (3.39 GB free per rank post-alloc) |
| Server boots, routes, KV cache    | ✅ Working in Run 3 |
| Mamba prefill import gate         | ✅ Fixed by Fix 2 (Run 4 no longer hits the `NameError`) |
| Mamba prefill forward pass        | ✅ Fixed by Fix 3 (device-aware `_device_context` replaces hard-coded `torch.cuda.device(...)`; Run 5 completes `_chunk_cumsum_fwd`, `_chunk_state_fwd`, `_state_passing_fwd`, `_bmm_chunk_fwd`, and `chunk_state_varlen`) |
| MoE FFN (`relu2` activation) on XPU | ❌ New block: `UnquantizedFusedMoEMethod.forward_xpu` at `unquant.py:576-579` only accepts `silu`/`gelu` — Nemotron-H uses `relu2`. Needs Fix 4. |
| Test end-to-end                   | ❌ Not yet passing (needs Fix 4 for fused-MoE `relu2`) |

### Referenced source locations

- `sglang/test/srt/xpu/test_nvidia_nemotron_3_nano.py` — the failing test
- `sglang/python/sglang/srt/mem_cache/mamba_radix_cache.py:435-438` — Run 1 assertion
- `sglang/python/sglang/srt/server_args.py:2274-2287` — first page_size coercion → 1
- `sglang/python/sglang/srt/server_args.py:2531-2536` — second page_size coercion → 128
- `sglang/python/sglang/srt/managers/scheduler.py:763-856` — cache-class selection
- `sglang/python/sglang/srt/layers/attention/mamba/mamba.py:32-51` — **Run 3 import gate (needs XPU branch)**
- `sglang/python/sglang/srt/layers/attention/mamba/mamba.py:503-507` — Run 3 NameError site
- `sglang/python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` — pure-Triton impl to use on XPU
- `sglang/python/sglang/srt/layers/attention/mamba/causal_conv1d.py:80` — precedent for Triton fallback on CUDA
- `sglang/python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py:11-38, 484, 544, 620` — **Fix 3 sites + `_device_context` helper**
- `sglang/python/sglang/srt/layers/attention/mamba/ops/ssd_state_passing.py:11-29, 237` — **Fix 3 site + helper**
- `sglang/python/sglang/srt/layers/attention/mamba/ops/ssd_bmm.py:11-30, 201` — **Fix 3 site + helper**
- `sglang/python/sglang/srt/layers/attention/mamba/ops/ssd_combined.py:98-162` — Mamba prefill call graph (`_chunk_cumsum_fwd` → `_chunk_state_fwd` → `_state_passing_fwd` → `_bmm_chunk_fwd` → `_chunk_scan_fwd` → `chunk_state_varlen`)
- `sglang/python/sglang/srt/layers/attention/mamba/ops/mamba_ssm.py:430` — deferred `torch.cuda.device(...)` site (decode-only `selective_state_update`; not on Run 5 prefill path)
- `sglang/python/sglang/srt/layers/attention/mamba/ops/layernorm_gated.py:122` — deferred `torch.cuda.device(...)` site (orphaned; `Mixer2RMSNormGated` uses `fla/layernorm_gated.py` instead)
- `sglang/python/sglang/srt/layers/quantization/unquant.py:565-579` — **Run 5 failure site: `forward_xpu` rejects `relu2` activation**
- `sglang/python/sglang/srt/layers/quantization/unquant.py:437-439` — CUDA precedent for `relu2` dispatch in the same method
- `sglang/test/registered/models/test_nvidia_nemotron_3_nano.py` — CUDA reference config

### Full logs

- Run 1: `/tmp/nemotron_test_output.log`
- Run 2: `/tmp/nemotron_test_output_fixA.log`
- Run 3: `/tmp/nemotron_test_output_fixA_retry.log`
- Run 4: `/tmp/nemotron_test_output_fix2.log`
- Run 5: `/tmp/nemotron_test_output_fix3.log`
