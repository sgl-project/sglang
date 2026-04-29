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

### Fix 2 — (proposed, not yet applied): XPU branch for `causal_conv1d_fn` imports
- **Motivation**: Run 3 failure — `NameError: causal_conv1d_fn_triton is
  not defined` at `mamba.py:507` on the first Mamba prefill.
- **What to change**: `sglang/python/sglang/srt/layers/attention/mamba/mamba.py`
  lines 32-51. Add an `elif is_xpu():` branch that binds
  `causal_conv1d_fn`, `causal_conv1d_fn_triton`, `causal_conv1d_update`,
  and `causal_conv1d_update_triton` from
  `sglang.srt.layers.attention.mamba.causal_conv1d_triton` (pure Triton,
  device-agnostic).
- **Why it's the right fix**: the Triton implementation is already the
  CUDA-side fallback when the sgl-kernel native binding is missing, and
  Triton is already in use for the rest of the Mamba/linear-attention path
  on XPU (the log shows `Linear attention kernel backend:
  decode=triton, prefill=triton`). The absence of an XPU import branch is a
  pure oversight — no semantic change is required, only the symbol
  bindings.
- **Status**: ⏸️ Not yet applied per the user's "report; do not modify"
  instruction. Awaiting authorization to edit
  `sglang/python/sglang/srt/layers/attention/mamba/mamba.py`.
- **Alternative workaround** (test-side only): none clean. The only way to
  bypass this inside the test is to disable the Mamba layers entirely,
  which would break the model. The real fix has to go into the SGLang
  source.

### Summary of current status

| Aspect                            | Status |
| --------------------------------- | ------ |
| MambaRadixCache page_size clash   | ✅ Fixed by Fix 1 |
| Memory headroom                   | ✅ Fixed by Fix 1 (3.39 GB free per rank post-alloc) |
| Server boots, routes, KV cache    | ✅ Working in Run 3 |
| Mamba prefill forward pass        | ❌ Blocked by missing XPU import branch (Fix 2 required) |
| Test end-to-end                   | ❌ Not yet passing (needs Fix 2) |

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
- `sglang/test/registered/models/test_nvidia_nemotron_3_nano.py` — CUDA reference config

### Full logs

- Run 1: `/tmp/nemotron_test_output.log`
- Run 2: `/tmp/nemotron_test_output_fixA.log`
- Run 3: `/tmp/nemotron_test_output_fixA_retry.log`
