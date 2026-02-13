# Piecewise CUDA Graph (PCG) Test Report

## Overview

This report documents the systematic testing of SGLang's test suite with **Piecewise CUDA Graph (PCG) enabled** to identify PCG-specific regressions and bugs. All tests were run with `disable_piecewise_cuda_graph=False` (PCG active) across 1-GPU, 2-GPU, and 4-GPU configurations.

**Date:** January 29 – February 9, 2026
**Branch:** main (at time of testing)
**Hardware:** NVIDIA GPU (80GB)

## Methodology

### Test Infrastructure

- **`run.py` / `run_2gpu.py` / `run_4gpu.py`** — Initial bulk runners. Execute all tests in a plan with PCG enabled, collecting logs and results into `plan.json`.
- **`rerun.py` / `rerun_2gpu.py` / `rerun_4gpu.py`** — Selective re-runners. Rerun specific failed tests (by name or `["failed"]`) to confirm failures and reduce noise.
- **`manual_test.py`** — Single-run investigator. Runs confirmed PCG bugs one at a time for interactive debugging. Logs go to `logs_manual/`.
- **`tmux.sh` / `tmux_2gpu.sh` / `tmux_4gpu.sh`** — tmux wrappers for running tests in detached sessions.

### Test Protocol

1. Each test was run **10 times** with a **1200s timeout** per run.
2. Tests were grouped by CI suite (stage-a, stage-b-small, stage-b-large, nightly).
3. PCG was forced enabled via environment — tests that auto-disable PCG (VLM, speculative decoding) were noted.
4. Results were triaged into: **PCG bug**, **not PCG-related**, or **flaky/infrastructure**.

## Results Summary

| Config | Total Tests | Passed | Failed | Pass Rate |
|--------|-------------|--------|--------|-----------|
| 1-GPU  | 156         | 125    | 31     | 80.1%     |
| 2-GPU  | 19          | 15     | 4      | 78.9%     |
| 4-GPU  | 6           | 3      | 3      | 50.0%     |
| **Total** | **181**  | **143**| **38** | **79.0%** |

### 1-GPU Breakdown by Suite

| Suite                    | Tests | Passed | Failed |
|--------------------------|-------|--------|--------|
| stage-a-test-1           | 1     | 1      | 0      |
| stage-b-test-small-1-gpu | 71    | 63     | 8      |
| stage-b-test-large-1-gpu | 67    | 48     | 19     |
| nightly-1-gpu            | 17    | 13     | 4      |

## Triage: Failure Categories

### Category 1: Confirmed PCG Bugs (need fix)

These failures are directly caused by PCG being enabled. They are tracked in `analysis/pcg_bugs.txt` for manual investigation.

| # | Test | GPU | Pass Rate | Summary |
|---|------|-----|-----------|---------|
| 1 | `scheduler/test_retract_decode.py` | 1 | 0/10 | CUDA device-side assert (`vectorized_gather_kernel` index OOB) during decode retraction + re-prefill under PCG |
| 2 | `quant/test_autoround.py` | 1 | 0/10 | `torch.compile` can't trace `sgl_kernel.awq_dequantize` (missing args in fake tensor call); auto-round uses AWQ dequant ops unsupported by dynamo |
| 3 | `core/test_gpt_oss_1gpu.py` | 1 | 2/10 | KV pool memory leak + CUDA device-side assert (likely PCG); `test_mxfp4_20b` produces garbage output |
| 4 | `models/test_embedding_models.py` | 1 | 0/10 | PCG compiles/captures OK but produces wrong embedding dimensions (128 vs 1536); tensor size mismatch in cosine similarity |
| 5 | `openai_server/validation/test_large_max_new_tokens.py` | 1 | 0/10 | Port 24000 conflict — PCG warmup takes ~10s, first server teardown races with second server startup |
| 6 | `attention/test_triton_sliding_window.py` | 1 | 8/10 | `Gemma3ForConditionalGeneration` VLM not detected for PCG auto-disable; `resolve_language_model()` fails on missing `model` attribute |
| 7 | `spec/eagle/test_eagle_infer_b.py` | 1 | 7/10 | `vectorized_gather_kernel` index OOB — same pattern as `test_retract_decode` |
| 8 | `quant/test_torchao.py` | 1 | 9/10 | `vectorized_gather_kernel` index OOB + CUDA device-side assert; same pattern as `test_retract_decode`; flaky |
| 9 | `scheduler/test_abort.py` | 1 | 9/10 | `test_memory_leak` OOM (exit -9) launching second server; PCG warmup consumes extra memory; flaky |
| 10 | `lora/test_lora_tp.py` | 2 | 0/10 | **[HIGH PRIORITY]** CUDA illegal memory access in `pynccl.outplace_all_reduce` during PCG warmup (`torch.compile` + PCG + LoRA + TP=2); crashes at `init_piecewise_cuda_graphs` |
| 11 | `eval/test_text_models_gsm8k_eval.py` | 2 | 0/10 | `torch._dynamo.exc.Unsupported`: `current_blas_handle()` returns int, not Tensor; DeepSeek V2 MLA `bmm_fp8` path incompatible with `torch.compile` |

### Category 2: Not PCG-Related

These tests fail for reasons unrelated to PCG (OOM, private models, pre-existing bugs, infra issues).

| Test | Reason |
|------|--------|
| `spec/eagle/test_eagle_infer_a.py` | EAGLE3 `context_length` mismatch (2048 vs 131072); PCG auto-disabled for speculative decoding |
| `spec/eagle/test_eagle3_basic.py` | OOM during EAGLE3 draft model loading; PCG auto-disabled |
| `spec/eagle/test_eagle_infer_beta.py` | OOM during EAGLE draft model loading; PCG auto-disabled |
| `attention/test_fa3.py` | 404 on `lmsys/sglang-ci-dsv3-test` (private CI model); EAGLE3 context_length mismatch |
| `quant/test_awq.py` | Borderline flaky (`test_mmlu` 0.84375 vs 0.85 threshold); OOM on AWQ+Marlin subtests |
| `vlm/test_vision_chunked_prefill.py` | ROUGE-L 0.8984 vs 0.9 threshold; PCG auto-disabled for VLM; flaky VLM quality |
| `lora/test_lora_qwen3.py` | All 10 runs timeout at 1200s; test too slow (Qwen3-4B + LoRA + HF ref); CUDA graph disabled via torch native attention backend |
| `openai_server/features/test_openai_server_hidden_states.py` | EAGLE3 context_length mismatch (2048 vs 131072) in setUpClass; PCG auto-disabled for spec decode |

### Category 3: Resolved on Rerun

These tests failed in the initial run but **passed 10/10 on rerun**, confirming the original failures were due to environment issues (e.g., leaked processes, stale state).

| Test | Initial | Rerun | Notes |
|------|---------|-------|-------|
| `tokenizer/test_skip_tokenizer_init.py` | 9/10 | 10/10 | Original flaky failure was environment issue |
| `openai_server/features/test_enable_thinking.py` | 0/10 | 10/10 | Original failure was environment issue |
| `lora/test_lora_openai_compatible.py` | 0/10 | 10/10 | Original failure was environment issue |

### Category 4: OOM from Leaked Process

A `test_lora_update` run timed out at 1200s, leaking a server process (PID 202958, 63.71GB GPU memory). The `rerun.py` `subprocess.run()` timeout kills the parent but not the child sglang server. **10 subsequent tests failed purely due to this OOM leak** — they are not PCG-related and need rerun on a clean GPU:

1. `lora/test_multi_lora_backend.py`
2. `models/test_generation_models.py`
3. `models/test_nvidia_nemotron_nano_v2_vl.py`
4. `models/test_vlm_models.py`
5. `moe/test_torch_compile_moe.py`
6. `perf/test_bench_serving_1gpu_large.py`
7. `perf/test_bench_serving_1gpu_part1.py`
8. `scheduler/test_no_chunked_prefill.py`
9. `scheduler/test_no_overlap_scheduler.py`
10. `vlm/test_vlm_input_format.py`

### Category 5: Triaged as Not PCG (marked passed)

These tests failed but were triaged as not PCG-related and marked as passed in plan.json. PCG was either not active or the failures were due to unrelated issues.

| Test | Reason |
|------|--------|
| `perf/test_vlm_perf_5090.py` | CuDNN version check (9.10 < 9.15 required); RTX 5090-specific test |
| `cuda_graph/test_piecewise_cuda_graph_small_1_gpu.py` | `run_bench_one_batch()` API mismatch + 404 on private CI model |
| `hicache/test_hicache_variants.py` | EAGLE OOM in `TestHiCacheEagle` setUpClass; non-EAGLE tests passed |
| `mla/test_flashmla.py` | 404 on private CI model + `KeyError: 'avg_spec_accept_length'` (stale API) |
| `mla/test_mla_deepseek_v3.py` | 404 on private CI model + stale spec decode API |
| `mla/test_mla_flashinfer.py` | 404 on private CI model + stale spec decode API |
| `models/test_vlm_models.py` | Gated HF model (`openbmb/MiniCPM-V-2_6`); PCG not active (VLM auto-disabled) |
| `debug_utils/test_crash_dump.py` | Crash dump `.pkl` not generated; PCG not active (`--skip-server-warmup`) |
| `utils/test_model_file_verifier.py` | Corrupted weights test expects server failure but it doesn't; no PCG |
| `utils/test_request_logger.py` | Logging events not found in stdout; no PCG |
| `utils/test_scheduler_status_logger.py` | Scheduler status events not found; no PCG |

### Category 6: Not Yet Triaged

These tests failed but have not been fully investigated. Some may contain PCG bugs, others may be infrastructure/flaky.

**1-GPU:**
- `lora/test_lora_backend.py` (0/10) — moved to manual investigation
- `lora/test_lora_update.py` (0/10)
- `attention/test_hybrid_attn_backend.py` (0/10)
- `vlm/test_vision_openai_server_a.py` (0/10)
- `perf/test_bench_serving_1gpu_part2.py` (9/10)
- `quant/test_w8a8_quantization.py` (9/10)
- `tokenizer/test_multi_tokenizer.py` (9/10)
- `scheduler/test_routing_key_scheduling.py` (2/10)

**2-GPU:**
- `cuda_graph/test_piecewise_cuda_graph_2_gpu.py` (9/10)
- `eval/test_moe_eval_accuracy_large.py` (9/10)
- `moe/test_glm4_moe_models.py` (9/10)
- `distributed/test_dp_attention.py` (0/10)
- `perf/test_bench_serving_2gpu.py` (0/10)
- `eval/test_vlms_mmmu_eval.py` (0/10)
- `perf/test_text_models_perf.py` (0/10)

**4-GPU:**
- `attention/test_local_attn.py` (0/10)
- `distributed/test_dp_attention_large.py` (0/10)
- `core/test_qwen3_next_deterministic.py` (0/10)
- `vlm/test_encoder_dp.py` (0/10)

## Key Patterns for Triage

These patterns help quickly classify test failures:

| Pattern | Meaning |
|---------|---------|
| "Compiling num tokens" / "Capturing num tokens" in logs | PCG is active |
| `disable_piecewise_cuda_graph=True` in server args | PCG is disabled (test opted out) |
| 404 on `lmsys/sglang-ci-dsv3-test` | Private CI model, not PCG |
| EAGLE3 `context_length` mismatch | Not PCG (auto-disabled for speculative decoding) |
| `resolve_language_model()` `'model'` attribute error | PCG bug — model architecture not supported |
| `vectorized_gather_kernel` index OOB | PCG bug — decode retraction / speculative decoding interaction |
| `torch._dynamo.exc.Unsupported` | PCG bug — op incompatible with `torch.compile` |

## File Structure

```
pcg_test/
├── plan/
│   ├── plan.json          # 1-GPU test plan (156 tests)
│   ├── plan_2gpu.json     # 2-GPU test plan (19 tests)
│   └── plan_4gpu.json     # 4-GPU test plan (6 tests)
├── analysis/
│   ├── pcg_bugs.txt       # Confirmed PCG bugs (manual working doc)
│   ├── plan_1gpu_failures.txt
│   ├── plan_2gpu_failures.txt
│   └── plan_4gpu_failures.txt
├── logs/                  # 1-GPU run logs (10 runs each)
├── logs_2gpu/             # 2-GPU run logs
├── logs_4gpu/             # 4-GPU run logs
├── logs_manual/           # Single-run investigation logs
│   ├── 1gpu/
│   └── 2gpu/
├── run.py                 # Bulk test runner (1-GPU)
├── run_2gpu.py / run_4gpu.py
├── rerun.py               # Selective re-runner (1-GPU)
├── rerun_2gpu.py / rerun_4gpu.py
├── manual_test.py         # Interactive bug investigation
└── REPORT.md              # This file
```

## Known Issues with Test Infrastructure

1. **Process leak on timeout:** `rerun.py`'s `subprocess.run()` timeout kills the parent process but not the child sglang server. This caused a cascading OOM failure affecting 10+ tests. Future runs should use process group kills (`os.killpg`).

## Next Steps

1. Rerun the 10 OOM-leaked tests on a clean GPU to get accurate results.
2. Complete triage of remaining "Not Yet Triaged" tests (Category 6).
3. Fix confirmed PCG bugs, prioritizing:
   - `test_lora_tp` (2-GPU) — CUDA illegal memory access during PCG warmup
   - `test_retract_decode` — CUDA device-side assert in decode retraction
   - `test_embedding_models` — wrong embedding dimensions
   - `vectorized_gather_kernel` index OOB cluster (retract_decode, eagle_infer_b, torchao)
4. Investigate `Gemma3ForConditionalGeneration` detection in PCG auto-disable logic.
