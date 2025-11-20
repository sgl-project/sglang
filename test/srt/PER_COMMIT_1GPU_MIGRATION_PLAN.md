# Per-Commit-1-GPU Tests Migration Plan

## Overview

This document tracks the migration of all 120 tests from `per-commit-1-gpu` suite in `test/srt/run_suite.py` to the new `test/per_commit/` structure with registration decorators.

**Total Tests**: 120 unique tests (122 entries with 2 duplicates noted below)
**Total Estimated Runtime**: ~19,009 seconds (~5.3 hours)

## Migration Strategy

Each PR will:
1. Move tests for ONE feature from `test/srt/` to `test/per_commit/`
2. Add registration decorators to each moved test
3. Remove moved tests from `test/srt/run_suite.py`
4. Preserve original folder structure under `test/per_commit/`

## Migration Order (from smallest/simplest to largest)

1. **rotary_embedding** (1 test, 10s) - ✅ BEST START: Single test, already in subfolder
2. **debug_utils** (1 test, 15s) - Second best: Single test, clean separation
3. **cache** (3 tests, 170s) - Small, cohesive feature
4. **utils** (2 tests, 56s) - Small utility tests
5. **hicache** (2 tests, 520s) - Already in subfolder
6. **rl** (3 tests, 320s) - Already in subfolder
7. **runtime** (3 tests, 1040s) - Core runtime tests (abort, deterministic, retract)
8. **moe** (3 tests, 315s) - MoE-specific tests
9. **scheduler** (3 tests, 394s) - Scheduling tests
10. **tokenization** (3 tests, 385s) - Tokenization tests
11. **vision** (3 tests, 1370s) - Vision/VLM tests
12. **performance** (4 tests, 1344s) - Performance optimization tests
13. **layers/attention/mamba** (4 tests, 99s) - Already in subfolder structure
14. **openai_server/validation** (4 tests, 217s) - Already in subfolder
15. **sampling** (6 tests, 324s) - Sampling/decoding tests
16. **openai_server/features** (5 tests, 539s) - Already in subfolder
17. **quant** (6 tests, 266s) - Already in subfolder (quantization kernels)
18. **attention/backends** (6 tests, 1080s) - Attention backend tests
19. **attention/mla** (6 tests, 1605s) - MLA attention tests
20. **quantization** (6 tests, 468s) - Quantization tests (modelopt, fp8, torchao)
21. **observability** (6 tests, 343s) - Metrics, profiling, monitoring
22. **other** (6 tests, 904s) - Miscellaneous tests to categorize better
23. **lora** (6 tests, 1299s) - Already in subfolder
24. **openai_server/basic** (6 tests, 389s) - Already in subfolder
25. **speculative_decoding** (7 tests, 2039s) - EAGLE and speculative decoding
26. **openai_server/function_call** (2 tests, 180s) - Already in subfolder
27. **openai_server** (2 tests, 131s) - Root-level OpenAI server tests
28. **models** (13 tests, 3187s) - Model tests (longest)

---

## Detailed Test Breakdown by Feature

### 1. rotary_embedding (1 test, ~10s)
**Target**: `test/per_commit/rotary_embedding/`

- `rotary_embedding/test_mrope.py` (10s)

---

### 2. debug_utils (1 test, ~15s)
**Target**: `test/per_commit/debug_utils/`

- `debug_utils/test_tensor_dump_forward_hook.py` (15s)

---

### 3. cache (3 tests, ~170s)
**Target**: `test/per_commit/cache/`

- `test_page_size.py` (60s)
- `test_radix_attention.py` (105s)
- `test_radix_cache_unit.py` (5s)

---

### 4. utils (2 tests, ~56s)
**Target**: `test/per_commit/utils/`

- `test_io_struct.py` (8s)
- `test_utils_update_weights.py` (48s)

---

### 5. hicache (2 tests, ~520s)
**Target**: `test/per_commit/hicache/`

- `hicache/test_hicache_storage.py` (127s)
- `hicache/test_hicache_variants.py` (393s)

---

### 6. rl (3 tests, ~320s)
**Target**: `test/per_commit/rl/`

- `rl/test_fp32_lm_head.py` (30s)
- `rl/test_update_weights_from_disk.py` (210s)
- `rl/test_update_weights_from_tensor.py` (80s)

---

### 7. runtime (3 tests, ~1040s)
**Target**: `test/per_commit/runtime/`

- `test_abort.py` (190s)
- `test_deterministic.py` (400s)
- `test_retract_decode.py` (450s)

---

### 8. moe (3 tests, ~315s)
**Target**: `test/per_commit/moe/`

- `test_fused_moe.py` (80s)
- `test_torch_compile_moe.py` (210s)
- `test_triton_moe_channel_fp8_kernel.py` (25s)

---

### 9. scheduler (3 tests, ~394s)
**Target**: `test/per_commit/scheduler/`

- `test_no_overlap_scheduler.py` (234s)
- `test_priority_scheduling.py` (130s)
- `test_request_queue_validation.py` (30s)

---

### 10. tokenization (3 tests, ~385s)
**Target**: `test/per_commit/tokenization/`

- `test_input_embeddings.py` (38s)
- `test_multi_tokenizer.py` (230s)
- `test_skip_tokenizer_init.py` (117s)

---

### 11. vision (3 tests, ~1370s)
**Target**: `test/per_commit/vision/`

- `test_vision_chunked_prefill.py` (170s)
- `test_vision_openai_server_a.py` (900s)
- `test_vlm_input_format.py` (300s)

---

### 12. performance (4 tests, ~1344s)
**Target**: `test/per_commit/performance/`

- `test_chunked_prefill.py` (410s)
- `test_no_chunked_prefill.py` (108s)
- `test_piecewise_cuda_graph.py` (750s)
- `test_torch_compile.py` (76s)

---

### 13. layers/attention/mamba (4 tests, ~99s)
**Target**: `test/per_commit/layers/attention/mamba/`

- `layers/attention/mamba/test_causal_conv1d.py` (25s)
- `layers/attention/mamba/test_mamba_ssm.py` (50s)
- `layers/attention/mamba/test_mamba_ssm_ssd.py` (20s)
- `test_mamba_unittest.py` (4s)

---

### 14. openai_server/validation (4 tests, ~217s)
**Target**: `test/per_commit/openai_server/validation/`

- `openai_server/validation/test_large_max_new_tokens.py` (41s)
- `openai_server/validation/test_matched_stop.py` (60s)
- `openai_server/validation/test_openai_server_ignore_eos.py` (85s)
- `openai_server/validation/test_request_length_validation.py` (31s)

---

### 15. sampling (6 tests, ~324s)
**Target**: `test/per_commit/sampling/`

- `test_constrained_decoding.py` (150s)
- `test_harmony_parser.py` (20s)
- `test_jinja_template_utils.py` (1s)
- `test_penalty.py` (82s)
- `test_pytorch_sampling_backend.py` (66s)
- `test_reasoning_parser.py` (5s)

---

### 16. openai_server/features (5 tests, ~539s)
**Target**: `test/per_commit/openai_server/features/`

- `openai_server/features/test_enable_thinking.py` (70s)
- `openai_server/features/test_json_mode.py` (120s)
- `openai_server/features/test_openai_server_ebnf.py` (20s)
- `openai_server/features/test_openai_server_hidden_states.py` (240s)
- `openai_server/features/test_reasoning_content.py` (89s)

---

### 17. quant (6 tests, ~266s)
**Target**: `test/per_commit/quant/`

- `quant/test_autoround.py` (60s)
- `quant/test_block_int8.py` (22s)
- `quant/test_fp8_kernel.py` (8s)
- `quant/test_int8_kernel.py` (8s)
- `quant/test_triton_scaled_mm.py` (8s)
- `quant/test_w8a8_quantization.py` (160s)

---

### 18. attention/backends (6 tests, ~1080s)
**Target**: `test/per_commit/attention/backends/`

- `test_fa3.py` (420s)
- `test_hybrid_attn_backend.py` (379s)
- `test_torch_native_attention_backend.py` (123s)
- `test_triton_attention_backend.py` (150s)
- `test_triton_attention_kernels.py` (4s)
- `test_triton_attention_kernels.py` (4s) *(duplicate entry)*

---

### 19. attention/mla (6 tests, ~1605s)
**Target**: `test/per_commit/attention/mla/`

- `test_flashmla.py` (230s)
- `test_mla.py` (180s)
- `test_mla_deepseek_v3.py` (500s)
- `test_mla_flashinfer.py` (302s)
- `test_mla_fp8.py` (93s)
- `test_mla_int8_deepseek_v3.py` (300s)

---

### 20. quantization (6 tests, ~468s)
**Target**: `test/per_commit/quantization/`

- `test_eval_fp8_accuracy.py` (303s)
- `test_fp8_utils.py` (5s)
- `test_modelopt_export.py` (30s)
- `test_modelopt_loader.py` (30s)
- `test_modelopt_loader.py` (30s) *(duplicate entry)*
- `test_torchao.py` (70s)

---

### 21. observability (6 tests, ~343s)
**Target**: `test/per_commit/observability/`

- `test_hidden_states.py` (55s)
- `test_metrics.py` (32s)
- `test_metrics_utils.py` (1s)
- `test_profile_merger.py` (60s)
- `test_profile_merger_http_api.py` (15s)
- `test_start_profile.py` (180s)

---

### 22. other (6 tests, ~904s)
**Target**: `test/per_commit/other/` *(to be recategorized)*

- `test_create_kvindices.py` (2s)
- `test_original_logprobs.py` (41s)
- `test_score_api.py` (310s)
- `test_srt_engine.py` (450s)
- `test_swa_unittest.py` (1s)
- `test_triton_sliding_window.py` (100s)

---

### 23. lora (6 tests, ~1299s)
**Target**: `test/per_commit/lora/`

- `lora/test_lora.py` (150s)
- `lora/test_lora_backend.py` (99s)
- `lora/test_lora_eviction.py` (240s)
- `lora/test_lora_spec_decoding.py` (150s)
- `lora/test_lora_update.py` (600s)
- `lora/test_multi_lora_backend.py` (60s)

---

### 24. openai_server/basic (6 tests, ~389s)
**Target**: `test/per_commit/openai_server/basic/`

- `openai_server/basic/test_openai_embedding.py` (79s)
- `openai_server/basic/test_openai_server.py` (270s)
- `openai_server/basic/test_protocol.py` (10s)
- `openai_server/basic/test_serving_chat.py` (10s)
- `openai_server/basic/test_serving_completions.py` (10s)
- `openai_server/basic/test_serving_embedding.py` (10s)

---

### 25. speculative_decoding (7 tests, ~2039s)
**Target**: `test/per_commit/speculative_decoding/`

- `test_build_eagle_tree.py` (8s)
- `test_eagle_infer_a.py` (750s)
- `test_eagle_infer_b.py` (750s)
- `test_eagle_infer_beta.py` (90s)
- `test_ngram_speculative_decoding.py` (290s)
- `test_speculative_registry.py` (1s)
- `test_standalone_speculative_decoding.py` (150s)

---

### 26. openai_server/function_call (2 tests, ~180s)
**Target**: `test/per_commit/openai_server/function_call/`

- `openai_server/function_call/test_openai_function_calling.py` (60s)
- `openai_server/function_call/test_tool_choice.py` (120s)

---

### 27. openai_server (2 tests, ~131s)
**Target**: `test/per_commit/openai_server/`

- `test_server_args.py` (1s)
- `test_srt_endpoint.py` (130s)

---

### 28. models (13 tests, ~3187s)
**Target**: `test/per_commit/models/`

- `models/test_compressed_tensors_models.py` (42s)
- `models/test_cross_encoder_models.py` (100s)
- `models/test_embedding_models.py` (73s)
- `models/test_encoder_embedding_models.py` (460s)
- `models/test_generation_models.py` (103s)
- `models/test_nvidia_nemotron_nano_v2.py` (160s)
- `models/test_qwen_models.py` (150s)
- `models/test_reward_models.py` (132s)
- `models/test_transformers_models.py` (320s)
- `models/test_vlm_models.py` (741s)
- `test_external_models.py` (155s)
- `test_gpt_oss_1gpu.py` (750s)
- `test_model_hooks.py` (1s)

---

## Registration Decorator Pattern

Each test should add the appropriate registration decorator at the top:

```python
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=<SECONDS>, suite="stage-b-test-small-1-gpu")

import unittest
# ... rest of test
```

**Suite assignment guidelines:**
- Simple unit tests (< 60s, no model loading) → `stage-a-test-2`
- Small model tests → `stage-b-test-small-1-gpu`
- Large model tests → `stage-b-test-large-1-gpu`

---

## Progress Tracker

| # | Feature | Status | PR Link | Tests Migrated | Notes |
|---|---------|--------|---------|----------------|-------|
| 1 | rotary_embedding | ⏳ Pending | - | 0/1 | - |
| 2 | debug_utils | ⏳ Pending | - | 0/1 | - |
| 3 | cache | ⏳ Pending | - | 0/3 | - |
| 4 | utils | ⏳ Pending | - | 0/2 | - |
| 5 | hicache | ⏳ Pending | - | 0/2 | - |
| 6 | rl | ⏳ Pending | - | 0/3 | - |
| 7 | runtime | ⏳ Pending | - | 0/3 | - |
| 8 | moe | ⏳ Pending | - | 0/3 | - |
| 9 | scheduler | ⏳ Pending | - | 0/3 | - |
| 10 | tokenization | ⏳ Pending | - | 0/3 | - |
| 11 | vision | ⏳ Pending | - | 0/3 | - |
| 12 | performance | ⏳ Pending | - | 0/4 | - |
| 13 | layers/attention/mamba | ⏳ Pending | - | 0/4 | - |
| 14 | openai_server/validation | ⏳ Pending | - | 0/4 | - |
| 15 | sampling | ⏳ Pending | - | 0/6 | - |
| 16 | openai_server/features | ⏳ Pending | - | 0/5 | - |
| 17 | quant | ⏳ Pending | - | 0/6 | - |
| 18 | attention/backends | ⏳ Pending | - | 0/6 | - |
| 19 | attention/mla | ⏳ Pending | - | 0/6 | - |
| 20 | quantization | ⏳ Pending | - | 0/6 | - |
| 21 | observability | ⏳ Pending | - | 0/6 | - |
| 22 | other | ⏳ Pending | - | 0/6 | - |
| 23 | lora | ⏳ Pending | - | 0/6 | - |
| 24 | openai_server/basic | ⏳ Pending | - | 0/6 | - |
| 25 | speculative_decoding | ⏳ Pending | - | 0/7 | - |
| 26 | openai_server/function_call | ⏳ Pending | - | 0/2 | - |
| 27 | openai_server | ⏳ Pending | - | 0/2 | - |
| 28 | models | ⏳ Pending | - | 0/13 | - |

**Total Progress**: 0/122 tests migrated (0%)
