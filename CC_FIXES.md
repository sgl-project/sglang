# Confidential Computing (CC) perf fixes

Two SGLang-side fixes that recover most of the CC inference-perf gap on B300 (Qwen3.5-397B-A17B-FP8, TP4 EP1). Both auto-enable when CC is detected (`is_confidential_compute()`, NVML `ccFeature != 0`), are **byte-identical off-CC**, and each logs once at INFO so you can confirm it's live in a normal run. There is **no env var to set** — CC detection drives both.

These pair with the FlashInfer-side autotuner timing fix (`flashinfer` branch `cc-autotuner-fixed` / `CC_AUTOTUNER_FIX.md`); together the three close the CC residual from ~70% to ~98% of the cc_off throughput at conc 4.

This branch (`cc-fixes`) sits on `main`, which is pinned to the container release (upstream sglang `v0.5.12`).

## Fix 1 — async D2H copy worker

**Problem.** Under bounce-buffer CC the per-step Device→Host token readback in `GenerationBatchResult.copy_to_cpu` becomes a *synchronous* `cudaMemcpyAsync` that blocks the scheduler thread for ~one decode step at issue (the host destination is Managed/UVM-backed). The scheduler then can't launch the next CUDA graph, so overlap is serialized — ~40–87% loss at high concurrency.

**Fix.** The copy can't be made async under CC, so run the still-blocking copy on a dedicated daemon thread (TRT-LLM #8463 pattern) — non-blocking *to the scheduler thread*, restoring overlap.
- `utils/common.py` — cached `is_confidential_compute()` via NVML `nvmlSystemGetConfComputeState`, fail-safe.
- `managers/overlap_copy_worker.py` — `AsyncD2HCopyWorker` (daemon thread + queue), with explicit teardown.
- `managers/scheduler.py` — gate `enable_cc_async_copy = is_confidential_compute()`; submit to the worker under CC (inline otherwise) at the 3 issue sites.
- `managers/scheduler_output_processor_mixin.py` — wait on `copy_ready_cpu` under CC at the consume sites.

**Marker (INFO).** `Confidential computing detected: offloading the per-step D2H result readback to a worker thread to preserve overlap scheduling.`
**Ref.** sgl-project/sglang#26469.

## Fix 2 — ungate the FlashInfer AR+RMSNorm fusion under CC

**Problem.** cc_off uses FlashInfer's `trtllm` AllReduce+RMSNorm fusion, but under CC it disables itself: `create_allreduce_fusion_workspace` hardcodes `use_symm_dev_mem=True` (symmetric/multicast memory) and its `cuMulticast` preflight fails under CC. The *kernel* is multicast-free — a one-shot **Lamport** AllReduce (verified: zero `multimem` in `trtllm_allreduce_fusion.cuh`). Only the workspace allocator wanted multicast.

**Fix.** Build the fusion on a multicast-free **IPC workspace** instead of disabling it: `trtllm_create_ipc_workspace_for_all_reduce_fusion(use_symm_dev_mem=False)`, skip the `cuMulticast` preflight, and drive `trtllm_allreduce_fusion(use_oneshot, kARResidualRMSNorm)`. Fuses the TP AllReduce + the following RMSNorm into one multicast-free pass — kernel parity with cc_off.
- `layers/flashinfer_comm_fusion.py` — `_cc_ipc_fusion_enabled()` (now just `is_confidential_compute()`), the IPC-workspace init path, the `cc_ipc` forward branch, and IPC-aware cleanup.

**Enabled by default whenever CC is detected** — no env var. (Previously opt-in via `SGLANG_CC_FLASHINFER_FUSION=1`; that gate was removed.)
**Marker (INFO).** `FlashInfer CC IPC fusion workspace initialized (multicast-free) for rank <r>, world_size <w>`.

## Scope / safety

- Both fixes are gated on CC detection; **off-CC runs are unchanged** (no new threads, no fusion-path change, same logs).
- No new required dependencies (`pynvml` is used only for detection and fails safe).
- Experiment harnesses, probes, and analysis docs used to develop these live on the `cc-analysis` branch, intentionally excluded here.
