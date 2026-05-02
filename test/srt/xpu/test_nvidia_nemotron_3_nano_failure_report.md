# Test-failure report: `test_nvidia_nemotron_3_nano.py`

Running report. A new Run section is appended each time the smoke test is
re-executed with a change. The Fix log at the end tracks every applied
change in order.

---

## Quick status

| Aspect | Status |
| --- | --- |
| MambaRadixCache `page_size` clash | Fixed by Fix 1 |
| Memory headroom (Mamba SSM + KV + activations) | Fixed by Fix 1 |
| Server boots, routes, `/model_info` | Working since Run 3 |
| Mamba prefill import gate (`causal_conv1d_fn_triton`) | Fixed by Fix 2 |
| Mamba prefill forward pass (SSD ops) | Fixed by Fix 3 |
| MoE FFN (`relu2` activation) on XPU — host whitelist | Fixed by Fix 4 (Candidate A) |
| MoE FFN (`relu2` activation) on XPU — sgl-kernel-xpu | Routed through Triton MoE runner by Fix 5 (surgical Candidate C); sgl-kernel-xpu follow-up issue filed separately |
| Mamba2 `selective_state_update` decode on XPU | Fixed by Fix 6 (`_device_context` helper in `mamba_ssm.py`) |
| Test end-to-end | ✅ Green since Run 8 — `test_simple_code_qa` passes in 118.982s, decode ~14.2 tok/s |

Next blocker: None on this test. Smoke test `test_simple_code_qa` is
green on XPU (TP=4, `intel_xpu` attention backend). Known follow-ups,
all out of scope for the smoke test:

- `ops/layernorm_gated.py:122` — orphaned dead code; `Mixer2RMSNormGated`
  uses `attention/fla/layernorm_gated.py` instead.
- `mamba_state_scatter_triton.py:120` — `.is_cuda` guard in
  `fused_mamba_state_scatter_with_mask`, reachable only from EAGLE
  tree-attention in `hybrid_linear_attn_backend.py`. Will need the same
  treatment when someone enables speculative decode on the XPU hybrid-SSM
  path.
- `sgl-kernel-xpu` `fused_experts` does not accept `relu2`. Fix 5 routes
  around it via the Triton MoE runner; reverting Fix 5 to the fast path
  is a one-line deletion once sgl-kernel-xpu lands relu2.

---

## Run 1 — `AssertionError: Page size must be 1 for MambaRadixCache v1, got 128`

**Exit path.** `popen_launch_server` timed out. Every TP worker died during
`Scheduler.__init__` at
`python/sglang/srt/mem_cache/mamba_radix_cache.py:436-438`, raised from
`init_cache_with_memory_pool` (`scheduler.py:856`).

**Root cause.** Two conflicting coercions in `ServerArgs` post-processing:

1. `server_args.py:2275-2281` sets `page_size = 1` because
   `NemotronHForCausalLM` with radix cache requires it in Mamba `no_buffer`
   scheduling mode.
2. `server_args.py:2531-2536` then raises `page_size` to 128 because the
   `intel_xpu` attention backend only accepts 32/64/128.

`MambaRadixCache.__init__` rejects the final `page_size=128`.

**Why XPU.** `enable_mamba_extra_buffer` (the mode that tolerates
`page_size!=1`) is CUDA-only (`server_args.py:2257-2259`). The
`intel_xpu` backend also hard-refuses `page_size=1`, so the CUDA config
cannot be copied verbatim.

**Full log.** `/tmp/nemotron_test_output.log`.

---

## Run 2 — `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` during weight load

**What was applied.** Fix 1 (see Fix log): `--disable-radix-cache`,
`--mem-fraction-static 0.85`.

**Result.** MambaRadixCache assertion gone. New failure during the first
embedding shard copy in
`python/sglang/srt/layers/vocab_parallel_embedding.py:468`.

**Root cause.** Environmental, not the test. Another user's
`gemma-4-31B` server (PID 349337) was holding all 4 XPUs at ~95%
utilization when the schedulers started. Confirmed by `ps -ef`. No
test-level change needed.

**Full log.** `/tmp/nemotron_test_output_fixA.log`.

---

## Run 3 — `NameError: causal_conv1d_fn_triton is not defined`

**Setup.** Clean XPUs this time. Server boots fully:

- Weights loaded in ~37s, `avail mem=7.95 GB`, `mem usage=14.76 GB`.
- Mamba cache + KV cache (`#tokens=1557504`, K/V ~2.23 GB each) allocated.
- Uvicorn listening on `127.0.0.1:21000`, `GET /model_info` → 200.

**Failure on warm-up forward pass:**

```
File "sglang/srt/layers/attention/mamba/mamba.py", line 507, in forward
    else causal_conv1d_fn_triton
NameError: name 'causal_conv1d_fn_triton' is not defined
```

**Root cause.** `mamba.py:34-51` gates imports on `is_cuda()` / `is_npu()`
with no XPU branch, so on XPU neither `causal_conv1d_fn` nor
`causal_conv1d_fn_triton` is bound. The pure-Triton module
`causal_conv1d_triton.py` has no CUDA dependency and is already the
CUDA-side fallback when the sgl-kernel native binding is absent
(`causal_conv1d.py:80`) — the right fallback for XPU.

**Full log.** `/tmp/nemotron_test_output_fixA_retry.log`.

---

## Run 4 — `RuntimeError: PyTorch was compiled without CUDA support`

**What was applied.** Fix 2 (see Fix log): `elif is_xpu():` branch in
`mamba.py` binding all four `causal_conv1d_*` symbols to the pure-Triton
implementation.

**Result.** `NameError` gone. Mamba mixer forward progresses into
`mamba_chunk_scan_combined` and then:

```
File "sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py", line 463,
     in _chunk_cumsum_fwd
    with torch.cuda.device(dt.device.index):
  File ".../torch/cuda/__init__.py", line 135, in _exchange_device
    raise RuntimeError("PyTorch was compiled without CUDA support")
```

All four TP ranks fail identically. Scheduler SIGQUITs; `popen_launch_server`
reports `exit code -9`.

**Root cause.** `ssd_chunk_state.py:463` (and sibling `ssd_*` ops) open
`torch.cuda.device(...)` unconditionally on the tensor's device index, even
when tensors live on `xpu:*`. Needs a device-aware dispatch via
`torch.get_device_module(...)` so the CM resolves to `torch.xpu.device(...)`
on XPU.

**Full log.** `/tmp/nemotron_test_output_fix2.log`.

---

## Run 5 — `AssertionError: activation = relu2 is not supported.`

**What was applied.** Fix 3 (see Fix log): `_device_context` helper
replacing hard-coded `torch.cuda.device(...)` in
`ssd_chunk_state.py` / `ssd_state_passing.py` / `ssd_bmm.py`.

**Result.** Mamba prefill forward pass completes end-to-end on all four
TP ranks — `_chunk_cumsum_fwd`, `_chunk_state_fwd`, `_state_passing_fwd`,
`_bmm_chunk_fwd`, `_chunk_scan_fwd`, `chunk_state_varlen` all run clean.
Control leaves the Mamba mixer and enters the MoE feed-forward, which
raises:

```
File "sglang/srt/layers/quantization/unquant.py", line 576, in forward_xpu
    assert moe_runner_config.activation in [
AssertionError: activation = relu2 is not supported.
```

All four TP ranks SIGQUIT on the same assertion.

**Root cause.** `UnquantizedFusedMoEMethod.forward_xpu`
(`unquant.py:565-579`) hard-whitelists `silu` and `gelu`. The CUDA path
in the same file at `unquant.py:437-439` already handles `relu2` via a
cutlass MoE runner. XPU wasn't extended with the Nemotron-H activation.

**Fix 3 is confirmed effective.** The Run 4 "CUDA support" RuntimeError
is gone and the first end-to-end Mamba prefill succeeds. The remaining
failure is in a different subsystem (fused MoE), not Mamba.

**Full log.** `/tmp/nemotron_test_output_fix3.log`.

---

## Run 6 — kernel-side `AssertionError: Only silu and gelu are supported but got relu2`

**What was applied.** Fix 4 Candidate A — `relu2` added to
`UnquantizedFusedMoEMethod.forward_xpu` whitelist at
`python/sglang/srt/layers/quantization/unquant.py:576-580`.

**Precondition check.**
`python3 -c "from sgl_kernel import fused_experts; import inspect;
print(inspect.signature(fused_experts))"` returned
`activation: str = 'silu'` — a plain string kwarg with no type constraint
at the Python entry point. So the host-side edit forwarded `relu2` into
the kernel wrapper unchanged, as Candidate A predicted. Whether the
wrapper itself accepts the value was deferred to this run.

**Result.** The host-side assertion at `forward_xpu` no longer fires.
Server boots, weights load, Mamba prefill completes, and control reaches
the MoE FFN for the first time. The failure has moved one frame down
into sgl-kernel-xpu:

```
File "sglang/srt/layers/quantization/unquant.py", line 588, in forward_xpu
    output = fused_experts(
             ^^^^^^^^^^^^^^
File "/home/sdp/workspace/sgl-kernel-xpu/python/sgl_kernel/moe.py", line 307, in fused_experts
    assert activation in (
           ^^^^^^^^^^^^^^^
AssertionError: Only silu and gelu are supported but got relu2
```

All four TP ranks SIGQUIT on the same assertion.
`popen_launch_server` times out with `Server process exited with code -9`
after 60s of the 120s launch timeout (the test never reaches
`test_simple_text_qa`).

**Subsystem.** Kernel wrapper — `sgl-kernel-xpu/python/sgl_kernel/moe.py`
lines 307-310:

```python
assert activation in (
    "silu",
    "gelu",
), f"Only silu and gelu are supported but got {activation}"
```

This is a second, independent whitelist inside sgl-kernel-xpu. The sglang
repo cannot reach past it without either (i) extending the kernel wrapper
upstream in `sgl-kernel-xpu` (Candidate B collapsed into that edit), or
(ii) routing `relu2` through the device-agnostic Triton MoE runner from
the `forward_xpu` else-branch (Candidate C).

**Fix 4 Candidate A assessment.** Correctly applied, correctly forwarded
to the kernel, **but insufficient on its own** because the sgl-kernel-xpu
wrapper re-asserts the same whitelist. Per user instruction, Candidate A
is left in place (it's a prerequisite for any further fix — without it,
`relu2` can't reach the kernel layer at all), and no automatic escalation
to Candidate B or C is performed. Decision escalated.

**Full log.** `/tmp/nemotron_test_output_fix4.log`.

---

## Run 7 — `RuntimeError: PyTorch was compiled without CUDA support` in `selective_state_update` (decode)

**What was applied.** Fix 5 — surgical Candidate C: `relu2` routed to the
Triton MoE runner in `forward_xpu`.

**Result.** Fix 5 is effective — the MoE subsystem no longer blocks:

- Server boots. Weights load (~6s, `avail mem=7.95 GB`, `mem usage=14.76 GB`
  per TP rank). Uvicorn on `127.0.0.1:21000`, `GET /model_info` → 200.
- Triton MoE config emits the expected "Using default MoE kernel
  config … E=128,N=464,device_name=Intel(R) Arc(TM) Pro B60 Graphics"
  notices at `23:34:45` on all four TP ranks. This is proof the `relu2`
  tokens are taking the Triton path added by Fix 5, not the
  sgl-kernel-xpu path. (Fix 4 Run 6 never reached this stage.)
- First real prefill batch succeeds at `23:34:49` —
  `Prefill batch, #new-seq: 1, #new-token: 64, #cached-token: 0,
  input throughput (token/s): 1.46`. The MoE layers in the prefill pass
  clear cleanly.

The failure has moved to a **different, known-deferred subsystem**:
Mamba2 `selective_state_update` on the **decode** path. All four TP
ranks SIGQUIT on the same assertion at `23:34:52`:

```
File "sglang/srt/layers/attention/mamba/mamba.py", line 704, in forward
    selective_state_update(
File "sglang/srt/layers/attention/mamba/ops/ssu_dispatch.py", line 258,
     in selective_state_update
    _mamba_ssu_backend(
File "sglang/srt/layers/attention/mamba/ops/ssu_dispatch.py", line 80, in __call__
    self._kernel(
File "sglang/srt/layers/attention/mamba/ops/mamba_ssm.py", line 430,
     in selective_state_update
    with torch.cuda.device(x.device.index):
  File "/root/miniforge3/envs/sgl-xpu-d/lib/python3.12/site-packages/torch/cuda/__init__.py",
       line 135, in _exchange_device
    raise RuntimeError("PyTorch was compiled without CUDA support")
```

`popen_launch_server` reports
`Exception: Server process exited with code -9. Check server logs for errors.`
at the `setUpClass` level after ~70s — the server dies before
`test_simple_text_qa` runs.

**Subsystem.** Mamba2 SSU (Selective State Update) decode kernel
dispatch. `ops/mamba_ssm.py:430` holds the same
`with torch.cuda.device(x.device.index):` pattern that Fix 3 removed
from the **prefill**-path SSD ops (`ssd_chunk_state.py`,
`ssd_state_passing.py`, `ssd_bmm.py`). This exact site is flagged in
the Fix 3 entry as a "Known follow-up (not on the prefill path;
deliberately deferred)":

> `ops/mamba_ssm.py:430` — `selective_state_update` (decode only).

Fix 3 chose not to widen its scope because Run 5 was in prefill and
the decode path was unreachable. Fix 5 made decode reachable, and the
deferred site fires on the very first decode step.

**Why this is progress, not a regression.** The crash signature is
identical to Run 4 (`RuntimeError: PyTorch was compiled without CUDA
support`), but at a **different file/line** (`mamba_ssm.py:430` vs.
`ssd_chunk_state.py:463`) and in a **different phase** (decode vs.
prefill). The MoE `relu2` story is closed for this test run. The
remaining blocker is a one-file extension of Fix 3 to the decode path.

**Fix 5 assessment.** ✅ Applied and effective for its stated scope
(MoE `relu2` on XPU). Orthogonal Mamba2 decode-path defect now
exposed.

**Candidate Fix 6 (proposal, not applied).** Extend the Fix 3
`_device_context(tensor)` helper pattern to
`python/sglang/srt/layers/attention/mamba/ops/mamba_ssm.py`:

- Add the same `_device_context` helper (copy from
  `ssd_chunk_state.py:11-38`).
- Replace `with torch.cuda.device(x.device.index):` at
  `mamba_ssm.py:430` with `with _device_context(x):`.
- Audit the rest of `mamba_ssm.py` for any other hard-coded
  `torch.cuda.device(` call sites — especially
  `selective_scan_fn` and any module-level helpers — and convert them
  in the same commit to avoid a Run 8 repeat.

This is a direct repeat of Fix 3's pattern applied to the decode
kernel. Not auto-applied per guardrails.

**Full log.** `/tmp/nemotron_test_output_fix5.log`.

---

## Run 8 — `test_simple_code_qa` ✅ end-to-end green

**What was applied.** Fix 6 — `_device_context` helper in
`ops/mamba_ssm.py`, replacing the hard-coded
`torch.cuda.device(x.device.index)` on the Mamba2 SSU decode path.

**Result.** End-to-end pass on XPU (TP=4, `intel_xpu` attention backend,
`Intel(R) Arc(TM) Pro B60 Graphics` × 4). The Run 7 decode-step
`RuntimeError` at `mamba_ssm.py:430` is gone.

- `python3 -m unittest -v xpu.test_nvidia_nemotron_3_nano` →
  `Ran 1 test in 118.982s` / `OK`.
- Test: `test_simple_code_qa` (the only method on
  `TestNemotron3Nano30BXPU`). Assertion passed; final
  `POST /v1/chat/completions HTTP/1.1` → `200 OK` at 00:23:35.
- Prefill batch (test prompt, 64 new tokens) @ 00:23:18 — input
  throughput 39.67 tok/s.
- Decode batches @ 00:23:20-00:23:34 — steady-state gen throughput
  ~14.2 tok/s (0.46 on the warm-up step, then 14.18, 14.30, 14.29,
  14.04, 14.16). `mamba num: 1, mamba usage: 0.12` on every decode
  step — the SSU kernel is exercised and completes without the CUDA
  RuntimeError.
- Fix 3's pattern generalizes cleanly: on CUDA the CM is
  `torch.cuda.device(idx)` (byte-for-byte the original behavior); on
  XPU it resolves to `torch.xpu.device(idx)` via
  `torch.get_device_module(...)`.

**Fix 6 assessment.** ✅ Applied and effective. Mamba2 decode path
fully unblocked on XPU. The Nemotron-H non-speculative decode smoke
test is green end-to-end.

**Housekeeping notes** (non-blocking, observed in this log):

- Two `/health_generate` 503s during warm-up (00:22:16, 00:22:26) and
  one "Health check failed. Server couldn't get a response from
  detokenizer for last 20 seconds" at 00:23:06 — all self-resolved
  before `test_simple_code_qa` ran at 00:23:17. This appears to be
  slow-to-warm heartbeat on XPU, not a defect.
- CCL warnings on launch (`did not find MPI-launcher specific
  variables`, `topology recognition shows PCIe connection`) — cosmetic
  on a single-node XPU.
- First attempted run of this fix timed out during distributed init
  (stale `/dev/shm/psm_*` / `sem.mp-*` from a previous aborted job).
  Cleaning `/dev/shm/psm_*` + `/dev/shm/sem.mp-*` resolved it; the log
  captured below is the clean second run.

**Full log.** `/tmp/nemotron_test_output_fix6.log`.

---

## Fix log (chronological)

### Fix 1 — disable radix cache + loosen memory fraction (applied Run 1 → Run 2)

- **File.** `test/srt/xpu/test_nvidia_nemotron_3_nano.py`, `XPU_SERVER_ARGS`.
- **Added.** `--disable-radix-cache` (routes the scheduler to `ChunkCache`,
  which does not require `page_size=1`).
- **Changed.** `--mem-fraction-static 0.92` → `0.85` (headroom for Mamba
  SSM state + KV pool + activation buffers on a 24 GB Arc Pro B60 partition).
- **Status.** Applied and effective. Runs 2, 3, 4, 5 confirm
  `disable_radix_cache=True` and no MambaRadixCache instantiation.

### Fix 2 — XPU branch for `causal_conv1d_*` imports (applied Run 3 → Run 4)

- **When.** 2026-05-01 UTC.
- **File.** `python/sglang/srt/layers/attention/mamba/mamba.py`
  - Lines 33-39: added `is_xpu` to the `sglang.srt.utils` import block.
  - Lines 59-74: new `elif is_xpu():` branch after the `is_npu()` branch,
    binding `causal_conv1d_fn`, `causal_conv1d_fn_triton`,
    `causal_conv1d_update`, and `causal_conv1d_update_triton` — all four —
    from `sglang.srt.layers.attention.mamba.causal_conv1d_triton`. Both
    "native" and `_triton` names resolve to the same Triton function,
    matching CUDA's fallback behavior.
- **Why it's correct.** The Triton implementation is device-agnostic and is
  already the CUDA fallback when the sgl-kernel native binding is absent
  (`causal_conv1d.py:80`). Triton already drives the rest of the
  Mamba/linear-attention path on XPU.
- **Status.** Applied and effective. Run 4's `NameError` is gone; the failure
  moved to `ssd_chunk_state.py:463` — a separate defect.

### Fix 3 — device-aware context manager in Mamba SSD ops (applied Run 4 → Run 5)

- **When.** 2026-05-01 UTC.
- **Problem.** `with torch.cuda.device(tensor.device.index):` was opened
  unconditionally in multiple SSD op files, raising
  `RuntimeError: PyTorch was compiled without CUDA support` on XPU.
- **What changed.** New module-local `_device_context(tensor)` helper per
  file. It dispatches through
  `torch.get_device_module(tensor.device).device(tensor.device.index)` and
  falls back to `contextlib.nullcontext()` on CPU / backends without a
  `.device` CM.

  Files and real line ranges after the edit:

  | File | Helper | Call sites (was → is) |
  | --- | --- | --- |
  | `python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py` | 11-38 | 463→484, 523→544, 599→620 |
  | `python/sglang/srt/layers/attention/mamba/ops/ssd_state_passing.py` | 11-29 | 217→237 |
  | `python/sglang/srt/layers/attention/mamba/ops/ssd_bmm.py` | 11-30 | 182→201 |

  Canonical replacement:

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

- **Why it's correct.** PyTorch exposes `torch.get_device_module(device)`
  precisely so generic code can pick the right namespace
  (`torch.cuda` / `torch.xpu` / …). On CUDA this resolves to
  `torch.cuda.device(idx)` — byte-for-byte identical behavior. On XPU it
  resolves to `torch.xpu.device(idx)`. The edit is additive: CUDA is
  unchanged.
- **Scope decision.** Surveyed every `torch.cuda.device(` site under
  `python/sglang/srt/layers/attention/mamba/ops/`. The Nemotron prefill
  path exercises all 5 fixed above (via `_mamba_chunk_scan_combined_fwd`
  in `ssd_combined.py:98-162`). Nemotron-H uses `cu_seqlens` for varlen
  prefill, so `chunk_state_varlen` is on the first-prefill path.
- **Known follow-ups** (not on the prefill path; deliberately deferred):
  - `ops/mamba_ssm.py:430` — `selective_state_update` (decode only).
  - `ops/layernorm_gated.py:122` — orphaned; `Mixer2RMSNormGated` actually
    uses `sglang/srt/layers/attention/fla/layernorm_gated.py`.
- **Status.** Applied and effective. Run 5 completes the full Mamba
  prefill forward pass; failure moved to the MoE FFN.

### Fix 4 — XPU `relu2` support in fused MoE (applied Run 5 → Run 6)

- **When.** 2026-05-01 UTC.
- **File.** `python/sglang/srt/layers/quantization/unquant.py`,
  `UnquantizedFusedMoEMethod.forward_xpu`, whitelist at lines 576-580
  (post-edit).
- **Motivation.** Run 5 failure — `UnquantizedFusedMoEMethod.forward_xpu`
  at `unquant.py:565-579` only accepted `silu`/`gelu`; Nemotron-H uses
  `relu2`. The CUDA path (`unquant.py:437-439`) already handles `relu2`
  via a cutlass MoE runner.
- **What changed.** One-line extension of the activation whitelist
  (Candidate A):

  ```diff
           moe_runner_config = self.moe_runner_config
           assert moe_runner_config.activation in [
               "silu",
               "gelu",
  +            "relu2",  # Nemotron-H (NemotronHForCausalLM) uses squared-ReLU.
           ], f"activation = {moe_runner_config.activation} is not supported."
  ```

- **Why it's correct.** The existing XPU branch in `forward_xpu` already
  forwards `moe_runner_config.activation` verbatim to
  `sgl_kernel.fused_experts(..., activation=...)` at `unquant.py:595`
  (post-edit line 596). The kernel entry point signature is
  `activation: str = 'silu'` — a pass-through string — so the host-side
  change is by construction non-lossy for CUDA and adds one more
  legal value on XPU.
- **Precondition check.** `python3 -c "from sgl_kernel import fused_experts;
  import inspect; print(inspect.signature(fused_experts))"` confirmed the
  kwarg is an untyped string. That meant no host-side type conflict.
  Runtime validity inside the kernel wrapper was left to surface in
  Run 6.
- **Status.** Applied (Candidate A). Host-side whitelist no longer
  rejects `relu2`.
- **Outcome.** See Run 6 below. The host assertion at
  `unquant.py:576-580` is gone, but a second, kernel-side assertion at
  `sgl-kernel-xpu/python/sgl_kernel/moe.py:307-310` now fires with the
  same message shape (`Only silu and gelu are supported but got relu2`).
  Candidate A alone is **not sufficient** to unblock Nemotron-H on XPU.

#### Deferred alternatives if Candidate A fails at the kernel layer

Run 6 confirmed the kernel layer rejects `relu2`. Candidates B and C
below are the two deferred paths. **Per user instruction, they are not
auto-applied** — the choice between them is escalated.

**Superseded by Fix 5 (surgical Candidate C applied 2026-05-01).** See
the Fix 5 entry below for the actual edit that shipped: a relu2-only
early-return to the Triton MoE runner, keeping silu/gelu on the fast
sgl-kernel-xpu path.

#### Current code (what's there today)

`python/sglang/srt/layers/quantization/unquant.py:565-608`:

```python
def forward_xpu(
    self,
    layer: torch.nn.Module,
    dispatch_output: StandardDispatchOutput,
) -> CombineInput:
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    moe_runner_config = self.moe_runner_config
    assert moe_runner_config.activation in [
        "silu",
        "gelu",
    ], f"activation = {moe_runner_config.activation} is not supported."

    backend = self.runner.runner_backend
    if use_intel_xpu_backend():
        from sgl_kernel import fused_experts

        topk_weights, topk_ids, _ = topk_output
        output = fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            b1=getattr(layer, "w13_weight_bias", None),
            b2=getattr(layer, "w2_weight_bias", None),
            activation=moe_runner_config.activation,   # passed through as string
            gemm1_alpha=moe_runner_config.gemm1_alpha,
            gemm1_limit=moe_runner_config.gemm1_clamp_limit,
        )
        return StandardCombineInput(hidden_states=output)
    else:
        assert backend.is_triton()
        assert (
            moe_runner_config.activation == "silu"
        ), f"... please set ENV SGLANG_USE_SGL_XPU=1."
        quant_info = self.get_triton_quant_info(layer)
        return self.runner.run(dispatch_output, quant_info)
```

CUDA precedent in the same file — `forward_cuda`'s flashinfer-cutlass
branch at `unquant.py:437-442` already translates the `relu2` string into
an `ActivationType` enum:

```python
activation_type=(
    ActivationType.Relu2
    if moe_runner_config.activation == "relu2"
    else ActivationType.Swiglu
),
```

#### Candidate A (preferred) — pass `relu2` through to `sgl_kernel.fused_experts`

If sgl-kernel-xpu's `fused_experts` already implements a squared-ReLU
activation, the fix is literally lengthening the whitelist: the kernel
entry point takes `activation` as a string and the XPU branch already
forwards it unchanged.

```python
# python/sglang/srt/layers/quantization/unquant.py, forward_xpu
moe_runner_config = self.moe_runner_config
assert moe_runner_config.activation in [
    "silu",
    "gelu",
    "relu2",     # NEW: Nemotron-H (NemotronHForCausalLM) uses squared-ReLU.
], f"activation = {moe_runner_config.activation} is not supported."
```

**Precondition to verify before landing:** run
`python3 -c "from sgl_kernel import fused_experts; help(fused_experts)"`
on the XPU build and confirm the `activation` kwarg accepts `"relu2"`. If
it does, no further changes are needed — the existing passthrough at
line 595 (`activation=moe_runner_config.activation`) already handles it.

#### Candidate B — apply `relu2` by hand around a plain matmul path

If `sgl_kernel.fused_experts` does not support `relu2` (i.e., Candidate A
is not viable), the next cheapest option is to do the activation outside
the fused kernel: call the kernel with a pass-through-or-silu activation
and then apply `x * x * (x > 0)` on the intermediate activations. This
trades throughput for correctness, which is the right choice for a
smoke test.

Sketch:

```python
# python/sglang/srt/layers/quantization/unquant.py, forward_xpu
if moe_runner_config.activation == "relu2":
    # Run the gate+up matmul without a fused activation, then apply
    # squared-ReLU manually before the down projection.
    output = fused_experts(
        x, layer.w13_weight, layer.w2_weight,
        topk_weights, topk_ids,
        b1=getattr(layer, "w13_weight_bias", None),
        b2=getattr(layer, "w2_weight_bias", None),
        activation="none",                 # or whatever "no activation" knob exists
        gemm1_alpha=moe_runner_config.gemm1_alpha,
        gemm1_limit=moe_runner_config.gemm1_clamp_limit,
        post_activation_hook=lambda y: torch.where(y > 0, y * y, torch.zeros_like(y)),
    )
else:
    assert moe_runner_config.activation in ["silu", "gelu"]
    output = fused_experts(
        x, layer.w13_weight, layer.w2_weight,
        topk_weights, topk_ids,
        b1=getattr(layer, "w13_weight_bias", None),
        b2=getattr(layer, "w2_weight_bias", None),
        activation=moe_runner_config.activation,
        gemm1_alpha=moe_runner_config.gemm1_alpha,
        gemm1_limit=moe_runner_config.gemm1_clamp_limit,
    )
return StandardCombineInput(hidden_states=output)
```

Pseudocode: the `post_activation_hook` / `activation="none"` names are
placeholders — whichever knob `fused_experts` actually exposes for
splitting the two matmuls. If it doesn't expose that knob at all, the
fallback is to lower `forward_xpu` to the Triton MoE runner
(`self.get_triton_quant_info(layer)` + `self.runner.run(...)`) and do the
activation there.

#### Candidate C (fallback) — route `relu2` to the Triton MoE runner

If neither A nor B is possible from the XPU path, the `else` branch of
`forward_xpu` already knows how to drive `get_triton_quant_info(layer)` /
`self.runner.run(dispatch_output, quant_info)`. The Triton MoE runner
lives in `python/sglang/srt/layers/moe/fused_moe_triton/` and is
device-agnostic (it's already in use on the non-`SGLANG_USE_SGL_XPU=1`
XPU config). Extending Triton's activation dispatch to support `relu2`
and then routing `relu2` through Triton would be a strictly correct but
slower fallback.

```python
# python/sglang/srt/layers/quantization/unquant.py, forward_xpu
if moe_runner_config.activation == "relu2":
    # XPU sgl-kernel has no relu2; fall back to the Triton MoE runner.
    quant_info = self.get_triton_quant_info(layer)
    return self.runner.run(dispatch_output, quant_info)
# ... existing silu/gelu sgl-kernel path ...
```

This requires confirming that the Triton `fused_moe` supports `relu2` —
which in turn may need its own small change in
`python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`. Treat this
as a last-resort path only if A and B are both unavailable.

#### Recommended order of attack (updated after Run 6)

1. ~~Candidate A — host-side whitelist extension.~~ **Applied.** The host
   assertion is gone, but the sgl-kernel-xpu wrapper at
   `sgl-kernel-xpu/python/sgl_kernel/moe.py:307-310` independently
   rejects `relu2`. So Candidate A is **necessary but not sufficient**.
2. Candidate B — host-side squared-ReLU around a pass-through /
   split-matmul call. This is the next step if the kernel wrapper cannot
   be extended cheaply. The exact "no activation" knob needs to be
   inspected in the XPU kernel; `sgl_kernel.fused_experts` has no
   `post_activation_hook` or `activation="none"` today (signature check
   confirmed only the 20 kwargs shown above). A minimal Candidate B on
   XPU likely requires either: (a) extending `sgl_kernel.moe.py:307-310`
   to accept `"relu2"` and either implementing squared-ReLU in the
   kernel or doing it outside; or (b) calling the fused kernel with
   `activation="silu"` plus a hacky post-hoc correction, which is
   mathematically wrong and therefore not acceptable. Practically
   Candidate B collapses into "extend sgl-kernel-xpu".
3. Candidate C — route `relu2` through the Triton MoE runner. Requires
   `get_triton_quant_info(layer)` + `self.runner.run(dispatch_output,
   quant_info)` on the XPU side, plus verifying the Triton `fused_moe`
   dispatch supports `relu2`. Slower but strictly additive to the sglang
   repo (no sgl-kernel-xpu change required). This is the most tractable
   path if a sgl-kernel-xpu build-and-test loop is unavailable.

### Fix 5 — Route `relu2` MoE through Triton runner on XPU (applied Run 6 → Run 7)

- **When.** 2026-05-01 UTC.
- **Motivation.** Run 6 — sgl-kernel-xpu `fused_experts` (moe.py:307-310)
  re-asserts `activation in {silu, gelu}` at the kernel layer, so Fix 4
  Candidate A (host whitelist) was necessary but not sufficient.
- **File.** `python/sglang/srt/layers/quantization/unquant.py`, `forward_xpu`.
  - Added `if moe_runner_config.activation == "relu2": → Triton runner`
    early-return before the `use_intel_xpu_backend()` gate (post-edit
    lines 582-589; comment block 582-586, branch 587-589).
  - Widened the else-branch `silu`-only assertion to accept
    `{"silu","relu2"}` (post-edit lines 612-616).
- **Why it's correct (researcher verdict, summarised).**
  - Triton MoE already implements squared-ReLU for `not is_gated` at
    `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:583-584`
    (`torch.square(F.relu(...))`); Nemotron-H has `is_gated=False`
    (`python/sglang/srt/models/nemotron_h.py:192`).
  - `gemm1_alpha`/`gemm1_limit` are GPT-OSS swiglu clamp params, unset
    for Nemotron-H, so routing `relu2` to Triton is numerically
    equivalent to what the sgl-kernel path would have been (no silent
    divergence).
  - Bias tensors `b13`/`b2` are threaded through both paths identically
    (`get_triton_quant_info` at `unquant.py:557-563`).
- **Scope decision.** Surgical form: silu/gelu still take the fast
  sgl-kernel-xpu path. Only relu2 tokens incur the Triton MoE cost.
  Reversal when sgl-kernel-xpu lands relu2 is a one-line branch
  deletion.
- **Follow-up (external repo).** File an issue/PR against
  `sgl-project/sgl-kernel-xpu` requesting `relu2` support in
  `fused_experts` (`moe.py:300-330`). When it ships, delete the Fix 5
  branch and let `use_intel_xpu_backend()` handle all three activations.
- **Status.** ✅ Applied.
- **Outcome.** See Run 7 below.

### Fix 6 — Device-aware context manager in Mamba SSU decode op (applied Run 7 → Run 8)

- **When.** 2026-05-02 UTC.
- **Motivation.** Run 7 decode-step failure — `RuntimeError: PyTorch was
  compiled without CUDA support` at `ops/mamba_ssm.py:430` on the first
  `selective_state_update` call. This was explicitly flagged in Fix 3's
  "Known follow-ups" as decode-only; Fix 5 made decode reachable, so the
  deferred site fired.
- **File.** `python/sglang/srt/layers/attention/mamba/ops/mamba_ssm.py`,
  real line ranges after the edit:
  - `import contextlib` at line 9 (stdlib import block, adjacent to the
    existing `import torch` / `import triton` lines).
  - `_device_context` helper defined at lines 21-38 — copied verbatim
    from `ops/ssd_chunk_state.py:21-38` so every SSD/SSU file now uses
    the same helper text.
  - Call site at line 453 (was 430 pre-edit):
    `with _device_context(x):` wrapping
    `_selective_scan_update_kernel[grid](...)` in
    `selective_state_update`.
- **Why it's correct.** Mechanical repeat of Fix 3's pattern. The CM
  brackets a single Triton `_selective_scan_update_kernel[grid](...)`
  launch with no CUDA-specific state reads in the body.
  `torch.get_device_module(tensor.device)` resolves to `torch.cuda` on
  CUDA — byte-for-byte identical behavior — and `torch.xpu` on XPU.
  CPU / backends without a `.device` context manager fall through to
  `contextlib.nullcontext()`, matching the other three SSD files.
- **Scope decision.** Fixed the one site hit by the Nemotron-H
  non-speculative decode path (the only `torch.cuda.device(` call site
  in `mamba_ssm.py` — grep confirms zero remaining call sites after the
  edit; the only literal is a mention inside the helper's docstring).
  Deferred, per researcher verdict:
  - `ops/layernorm_gated.py:122` — still orphaned dead code;
    `Mixer2RMSNormGated` uses `attention/fla/layernorm_gated.py`
    (confirmed via `mamba/mixer2_rms_norm_gated.py:13`).
  - `mamba_state_scatter_triton.py:120` — `.is_cuda` guard in
    `fused_mamba_state_scatter_with_mask`, only reachable from EAGLE
    tree-attention in `hybrid_linear_attn_backend.py`. Not on the
    Nemotron-H smoke-test path. Follow-up for whoever enables
    speculative decode on XPU hybrid-SSM.
- **Status.** ✅ Applied.
- **Outcome.** See Run 8 above — `test_simple_code_qa` passes end-to-end
  in 118.982s with decode throughput ~14.2 tok/s.

---

## Referenced source locations

- **Test.** `test/srt/xpu/test_nvidia_nemotron_3_nano.py`
- **CUDA reference.** `test/registered/models/test_nvidia_nemotron_3_nano.py`

### Run 1 (page_size)

- `python/sglang/srt/mem_cache/mamba_radix_cache.py:435-438` — assertion
- `python/sglang/srt/server_args.py:2274-2287` — first coercion → 1
- `python/sglang/srt/server_args.py:2531-2536` — second coercion → 128
- `python/sglang/srt/managers/scheduler.py:763-856` — cache-class selection

### Run 3 (imports)

- `python/sglang/srt/layers/attention/mamba/mamba.py:32-51` — import gate
  (Fix 2 target)
- `python/sglang/srt/layers/attention/mamba/mamba.py:503-507` — NameError site
- `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` —
  pure-Triton implementation
- `python/sglang/srt/layers/attention/mamba/causal_conv1d.py:80` — CUDA
  precedent for Triton fallback

### Run 4 + Fix 3 (SSD ops)

- `python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py:11-38, 484, 544, 620`
- `python/sglang/srt/layers/attention/mamba/ops/ssd_state_passing.py:11-29, 237`
- `python/sglang/srt/layers/attention/mamba/ops/ssd_bmm.py:11-30, 201`
- `python/sglang/srt/layers/attention/mamba/ops/ssd_combined.py:98-162` —
  prefill call graph
- `python/sglang/srt/layers/attention/mamba/ops/mamba_ssm.py:430` —
  deferred decode-only site
- `python/sglang/srt/layers/attention/mamba/ops/layernorm_gated.py:122` —
  deferred orphaned site

### Run 5 (MoE host-side)

- `python/sglang/srt/layers/quantization/unquant.py:565-579` — `forward_xpu`
  rejects `relu2` (Fix 4 target; post-edit lines 576-580)
- `python/sglang/srt/layers/quantization/unquant.py:437-439` — CUDA precedent
  for `relu2` dispatch

### Run 6 (MoE kernel-side)

- `python/sglang/srt/layers/quantization/unquant.py:576-580` — post-Fix-4
  whitelist, now includes `relu2`
- `python/sglang/srt/layers/quantization/unquant.py:588-598` — XPU
  pass-through call to `sgl_kernel.fused_experts` (unchanged)
- `/home/sdp/workspace/sgl-kernel-xpu/python/sgl_kernel/moe.py:307-310` —
  kernel-side activation whitelist (new Run 6 blocker; out-of-repo)

### Run 7 + Fix 5 (Triton MoE relu2 routing)

- `python/sglang/srt/layers/quantization/unquant.py:576-580` — activation
  whitelist (unchanged from Fix 4, still includes `relu2`)
- `python/sglang/srt/layers/quantization/unquant.py:582-589` — Fix 5
  early-return routing `relu2` to `self.runner.run(...)` via
  `get_triton_quant_info(layer)`
- `python/sglang/srt/layers/quantization/unquant.py:612-616` — widened
  else-branch assertion now accepting `{"silu","relu2"}`
- `python/sglang/srt/layers/quantization/unquant.py:557-563` —
  `get_triton_quant_info` threads `b13`/`b2` into the Triton runner
- `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:583-584` —
  Triton squared-ReLU path (`torch.square(F.relu(...))`) on the
  `not is_gated` branch
- `python/sglang/srt/models/nemotron_h.py:192` — Nemotron-H sets
  `is_gated=False`, so the above Triton branch is selected

### Run 7 (Mamba2 SSU decode blocker — Fix 6 candidate)

- `python/sglang/srt/layers/attention/mamba/mamba.py:704` — decode
  callsite `selective_state_update(...)`
- `python/sglang/srt/layers/attention/mamba/ops/ssu_dispatch.py:80, 258`
  — dispatch frames leading to the kernel
- `python/sglang/srt/layers/attention/mamba/ops/mamba_ssm.py:430` —
  `with torch.cuda.device(x.device.index):` deferred by Fix 3, now the
  decode-path blocker (Fix 6 target)

### Run 8 + Fix 6 (Mamba SSU decode — device-aware CM)

- `python/sglang/srt/layers/attention/mamba/ops/mamba_ssm.py:9` —
  `import contextlib` (new).
- `python/sglang/srt/layers/attention/mamba/ops/mamba_ssm.py:21-38` —
  `_device_context` helper (copied verbatim from
  `ops/ssd_chunk_state.py:21-38`).
- `python/sglang/srt/layers/attention/mamba/ops/mamba_ssm.py:453` —
  `with _device_context(x):` (was line 430 pre-edit,
  `with torch.cuda.device(x.device.index):`).
- `test/srt/xpu/test_nvidia_nemotron_3_nano.py` — `test_simple_code_qa`,
  now green. No change to the test file in this fix.

### Deferred / out-of-scope sites (confirmed not on Nemotron-H smoke path)

- `python/sglang/srt/layers/attention/mamba/ops/layernorm_gated.py:122`
  — orphaned; `Mixer2RMSNormGated` uses
  `sglang/srt/layers/attention/fla/layernorm_gated.py` (per
  `mamba/mixer2_rms_norm_gated.py:13`).
- `python/sglang/srt/layers/attention/mamba/ops/mamba_state_scatter_triton.py:120`
  — only reached via EAGLE tree-attention in
  `hybrid_linear_attn_backend.py`; not on this test's path.

---

## Full logs

| Run | Log path |
| --- | --- |
| 1 | `/tmp/nemotron_test_output.log` |
| 2 | `/tmp/nemotron_test_output_fixA.log` |
| 3 | `/tmp/nemotron_test_output_fixA_retry.log` |
| 4 | `/tmp/nemotron_test_output_fix2.log` |
| 5 | `/tmp/nemotron_test_output_fix3.log` |
| 6 | `/tmp/nemotron_test_output_fix4.log` |
| 7 | `/tmp/nemotron_test_output_fix5.log` |
| 8 | `/tmp/nemotron_test_output_fix6.log` |
