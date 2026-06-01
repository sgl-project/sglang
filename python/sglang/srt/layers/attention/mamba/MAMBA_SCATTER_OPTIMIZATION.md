# Mamba State Scatter Kernel Optimization

Documentation for the optimization of `fused_mamba_state_scatter_with_mask_kernel`
(`mamba_state_scatter_triton.py`) and its call site in
`hybrid_linear_attn_backend.py`.

This kernel was identified by NCU profiling as the hottest, memory-bound kernel on
the MTP (speculative decoding) verify path when serving
`Sehyo/Qwen3.5-35B-A3B-NVFP4` on an RTX PRO 6000 Blackwell GPU.

---

## The three changes

**Kernel** (`mamba_state_scatter_triton.py`):

1. **Bigger blocks** — `_pick_block_size()` picks `next_pow2(elem_per_entry)` clamped
   to `[1024, 8192]` (was fixed 1024), with `num_warps=8` for blocks ≥4096. Fewer,
   fatter blocks → better latency hiding for a memory-bound kernel.
2. **Layer as inner loop** — `num_layers` dropped from the grid; the kernel now loops
   `for layer in tl.range(num_layers)`. The validity + bounds checks run once per
   request and amortize across all layers instead of re-running per layer. Grid
   shrinks by the layer factor.
3. **Index-set grid dim** — new `num_index_sets` grid dimension with
   `idx_pos = pid_set * set_size + pid_req` and `src_idx = pid_req` (resets per set),
   so concatenated verify+track index arrays scatter in one launch.

**Call site** (`hybrid_linear_attn_backend.py`): `update_mamba_state_after_mtp_verify`
now `torch.cat`s the verify and track index/step tensors once and passes
`num_index_sets=2` (or 1 when no tracking). **4 launches → 2** (one per ssm/conv
buffer). The concatenated indices are reused for both buffers since they're identical.

Net grid reduction: the old grid was `(total_requests, num_layers, blocks)` ≈ 1.23M
tiny 128-thread blocks; the new grid is `(set_size, blocks, num_index_sets)` with the
layer factor removed and far fewer element-blocks per request.

Both files pass syntax check. The remaining validation is a real run on the Blackwell
box — correctness (mamba state matches before/after) and the NCU re-profile to confirm
grid size dropped and eligible-warps/throughput improved.

---

## Why these changes — profiling context

### Expected magnitude

NCU showed the kernel is **memory bound** (90% memory throughput, 9% compute, L2 27%,
L1 16%, L1 hit 6%, L2 hit 0.5%, active warps 10, **eligible warps 0.6**). Launch
config: **Grid Size 1,228,800, Block Size 128, Registers Per Thread 16, Waves Per SM
54.47, # SMs 188** — i.e. an over-subscribed grid of tiny blocks with poor latency
hiding.

A key correction during analysis: the "35ms cudaMemcpyAsync" seen filling the nsys
timeline is **not** bandwidth (only 5.9MB DtoH total, sub-µs per copy). It is a
**host-side synchronization stall** — FlashInfer `plan()` does a synchronizing
`.to("cpu")` of indptr arrays, which forces the host to wait for the entire GPU queue
(tail = the slow scatter kernel) to drain.

From the user's own timeline: memcpy started at 827ms, first scatter kernel at 852ms,
both finished at 858ms. So the scatter is only ~6ms of the ~31ms stall; the remaining
~25ms is legitimate verify-step forward compute that the host syncs behind.
**Expected savings from fixing the kernel: ~4-5ms/step, not the full 35ms.**

### Naming convention check

Per the repo's `.claude/rules/modify-component-must-read.md`, the `speculative-naming`
skill was loaded before modifying spec-decoding code. Conclusion: these changes keep
the generic scatter parameter names `dst_indices_raw` / `step_indices_raw`, so no
`accept`/`correct` renames were needed.

---

## Call-site details

`update_mamba_state_after_mtp_verify` (in `hybrid_linear_attn_backend.py`) originally
called the fused kernel 4 times:

```python
fused_mamba_state_scatter_with_mask(ssm_states, intermediate_state_cache, state_indices_tensor, last_correct_step_indices)
fused_mamba_state_scatter_with_mask(conv_states, intermediate_conv_window_cache, state_indices_tensor, last_correct_step_indices)
# if mamba_track_indices is not None:
fused_mamba_state_scatter_with_mask(ssm_states, intermediate_state_cache, mamba_track_indices, mamba_steps_to_track)
fused_mamba_state_scatter_with_mask(conv_states, intermediate_conv_window_cache, mamba_track_indices, mamba_steps_to_track)
```

Key insight for fusion: the ssm verify (call 1) and ssm track (call 3) share the SAME
dst tensor (`ssm_states`), SAME src (`intermediate_state_cache`), same strides/elem —
only `dst_indices` and `step_indices` differ, and `src_idx = pid_req` for both. Same
for conv (calls 2 & 4). So verify+track can be fused per buffer (4→2 launches).

`state_indices_tensor = self.linear_attn_backend.forward_metadata.mamba_cache_indices[:request_number]`
where `request_number = last_correct_step_indices.shape[0]` (= bs).

Both `last_correct_step_indices` and `mamba_steps_to_track` are `[bs]`-shaped (in
`eagle_worker_v2._mamba_verify_update`, `mamba_steps_to_track` comes from
`torch.where` over `[bs]` masks), and `mamba_track_indices` matches
`state_indices_tensor` length (`bs`). So verify and track sets have a uniform
`set_size = bs` — the fusion via a `num_index_sets` grid dim works cleanly.

---

## Full session history (before compaction)

### Primary request and intent

The overarching goal evolved but converged on: profile and optimize SGLang serving of
the `Sehyo/Qwen3.5-35B-A3B-NVFP4` model on an RTX PRO 6000 Blackwell GPU, originally to
benchmark against vLLM. The final active intent: **apply three specific optimizations
to the `fused_mamba_state_scatter_with_mask_kernel` Triton kernel** which NCU profiling
identified as the hottest, memory-bound kernel on the MTP (speculative decoding) verify
path:
- (1) Increase BLOCK_SIZE from 1024 to 4096/8192
- (2) Collapse the layer grid dimension into an inner kernel loop
- (3) Fuse the 4 separate scatter kernel launches into fewer launches

Earlier (completed) intents: build/run SGLang via Docker, convert a vLLM config to
SGLang flags, modify an offline benchmark script for Nsight, set up nsys/ncu profiling.

### Key technical concepts

- SGLang serving (offline `sgl.Engine` API and `sglang.launch_server`), Docker image
  `lmsysorg/sglang:latest`
- Qwen3-Next hybrid GDN architecture (Gated Delta Network / Mamba linear attention +
  Mamba SSM/conv state caches; ssm_state ~24.73GB)
- NVFP4 quantization (compressed-tensors via llm-compressor), Blackwell sm_120
- MTP (Multi-Token Prediction) speculative decoding; SGLang NEXTN algorithm;
  SGLANG_ENABLE_SPEC_V2; mamba-scheduler-strategy extra_buffer
- FlashInfer attention backend `plan()`/`begin_forward` — computes kernel tile schedule
  on CPU, requires synchronizing DtoH `.to("cpu")` of indptr arrays
- Triton kernels: grid/block sizing, eligible vs active warps, memory-level
  parallelism, CUDA graph capture incompatibility with triton.autotune
- Nsight Systems (nsys) — `--capture-range=cudaProfilerApi`, `--capture-range-end=stop`,
  NVTX ranges, `cudaProfilerStart/Stop`, `--gpu-metrics-devices`,
  `--python-backtrace=cuda`, `--cudabacktrace`, `nsys stats` reports
- Nsight Compute (ncu) — `--kernel-name regex`, `--set full`, `--launch-skip/count`,
  `--target-processes all`
- Docker privileges for profiling: `--cap-add=SYS_ADMIN --security-opt seccomp=unconfined`;
  ERR_NVGPUCTRPERM and host driver `NVreg_RestrictProfilingToAdminUsers=0`
- host-device synchronization stalls (the "35ms cudaMemcpyAsync" being host blocking on
  GPU queue drain, not actual bandwidth)

### Files and code sections

- `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py` (THE FILE
  MODIFIED) — contains `_fused_mamba_state_scatter_with_mask_kernel` (Triton jit) and
  wrapper `fused_mamba_state_scatter_with_mask(dst, src, dst_indices_raw,
  step_indices_raw)`. Original kernel: grid `(total_requests, num_layers,
  triton.cdiv(elem_per_entry, BLOCK_SIZE))`, BLOCK_SIZE=1024, block 128 threads. Each
  block loads step_idx (early-exit if <0), dst_idx, does bounds check, copies
  BLOCK_SIZE elements src→dst for one (req, layer, block). `src_idx = pid_req`.
  Wrapper internals: `elem_per_entry = dst.numel() // (dst.shape[0] * dst.shape[1])`;
  strides via `.stride()`; indices cast `.to(torch.int32).contiguous()`; requires
  dst/src contiguous.

- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` (CALL SITE
  UPDATED) — `update_mamba_state_after_mtp_verify` (was lines 1048-1111) called the
  fused kernel 4 times (see "Call-site details" above).

- `python/sglang/srt/speculative/eagle_worker_v2.py` (caller, investigated) — lines
  1200-1247 prepare `last_correct_step_indices = accept_lens - 1`,
  `mamba_steps_to_track` (via torch.where), and call
  `update_mamba_state_after_mtp_verify`. All on-device tensor ops. Lines 1093-1098:
  `.cpu()` calls are gated by `if batch.has_grammar:` — NOT the source of the stall in
  plain-text benchmarks (ruled out).

- `/Users/kartik.vyas/Downloads/qwen35_offline_benchmark_sglang.py` (CREATED) — uses
  `sgl.Engine` with Option B settings: `mem_fraction_static=0.9`, `context_length`,
  `max_running_requests=128`, `chunked_prefill_size=8192`, `reasoning_parser="qwen3"`,
  `mamba_ssm_dtype="float16"`, `mamba_scheduler_strategy="extra_buffer"`,
  `speculative_algorithm="NEXTN"`, `speculative_num_steps=2`,
  `speculative_num_draft_tokens=2`, `speculative_eagle_topk=1`, `cuda_graph_max_bs=256`.
  Sets `os.environ.setdefault("SGLANG_ENABLE_SPEC_V2", "1")` before import. Has NVTX
  ranges (`nvtx_push/pop`) named `burst{N}_promptlen{L}_max{M}` and
  `cuda_profiler_start/stop` gated by `--profile`. `torch.cuda.synchronize()` around
  timing. Token counts via `o["meta_info"]["completion_tokens"]`. The `if __name__`
  block was edited to add a try/finally with `engine.shutdown()` and `os._exit(0)` to
  bypass the hung atexit hook. On the VM copied to
  `/mnt/extra-disk/sglang/test_sglang_server.py` (and inside container at
  `/scripts/test_sglang_server.py`).

### Errors and fixes

- Dockerfile pip upgrade "Cannot uninstall pip 24.0, RECORD file not found": drop the
  pip upgrade or use `--ignore-installed`. (Approach later abandoned in favor of
  official image.)
- Container name conflict / exit(1): check `docker logs` before retrying, `docker rm`
  stale containers.
- "Unrecognized model in /tmp/llm_model_mtp" (missing config.json): switched
  `--model-path` to HF ID `Sehyo/Qwen3.5-35B-A3B-NVFP4`.
- `--enable-ep-moe is not supported`: dropped the flag (no-op on single GPU; use
  `--ep-size` for multi-GPU).
- Speculative decoding + radix cache + mamba incompatibility
  (`--mamba-scheduler-strategy no_buffer`): Option A `--disable-radix-cache` (matches
  vLLM's enable_prefix_caching=false) or Option B `SGLANG_ENABLE_SPEC_V2=1` +
  `--mamba-scheduler-strategy extra_buffer`. **User chose Option B.**
- Container stopped immediately when run with no command: added `--entrypoint sleep ...
  infinity` to keep alive.
- ERR_NVGPUCTRPERM (GPU counters): drop `--gpu-metrics-devices`, OR set host
  `NVreg_RestrictProfilingToAdminUsers=0` + reload driver, OR add `--cap-add=SYS_ADMIN
  --security-opt seccomp=unconfined`.
- `--gpu-metrics-device` deprecated → `--gpu-metrics-devices` (plural).
- nsys "No reports were generated" — `--capture-range-end=stop-shutdown` waited for
  SGLang's hung atexit `Engine.shutdown` (kill_process_tree not reaped in 60s): changed
  to `--capture-range-end=stop` AND added `os._exit(0)` in script. Report then
  generated successfully.
- Docker file bind-mount error "not a directory": mount the parent directory
  `/mnt/extra-disk/sglang:/scripts:ro` instead of a single file.
- Misdiagnosis correction (important): I initially implied the 35ms memcpy was mostly
  the scatter kernel; user's timeline (memcpy 827→858ms = 31ms, scatter only 852→858ms
  = 6ms) corrected this — scatter is only ~6ms (the tail); the other ~25ms is the
  verify-step forward compute the host syncs behind. Revised expectations: fixing the
  kernel saves ~4-5ms/step, not the full 35ms.

### Problem solving

- Confirmed SGLang DOES load the NVFP4 hybrid-GDN checkpoint (despite no SGLang mention
  on model card).
- Established benchmark fairness considerations (match max-model-len, prefix caching
  settings, warmup, clock pinning).
- Diagnosed the "35ms cudaMemcpyAsync" as a host-side synchronization stall (NOT
  bandwidth: only 5.9MB DtoH total, sub-µs per copy; cudaMemcpyAsync max 38.3ms
  host-blocking). Traced to FlashInfer `plan()` doing synchronizing `.to("cpu")` of
  indptr arrays, which forces the host to wait for the entire GPU queue (tail = the slow
  scatter kernel) to drain.
- Unified conclusion: the scatter kernel is the fixable lever on the critical path.

### nsys stats reference (from the profiled run)

- DtoD 23.6ms / 2789MB; DtoH 9.9ms / 5.9MB / 13808 calls; HtoD 6.3ms / 24MB
- cudaMemcpyAsync 33.36s total / 57368 calls / MAX 38.3ms; cudaStreamSynchronize 1.35s;
  cudaLaunchKernel 183662 calls
- Call stack of the stall: `flashinfer_backend.py:call_begin_forward():1483` →
  `prefill.py:plan():1896` → `libtorch_python.so:torch::autograd::THPVariable_to` →
  libtorch_cpu → libtorch_cuda → libcudart.so
- Benchmark throughput observed: 5778 tok/s
