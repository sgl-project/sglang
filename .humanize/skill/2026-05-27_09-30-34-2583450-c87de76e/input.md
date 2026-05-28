# Ask Codex Input

## Question

You are performing the first-pass planning analysis for Loop 4 of the Double Sparsity MVP implementation in SGLang (the sglang LLM serving framework).

## Repository Context
SGLang is a high-performance LLM serving framework. This work is on branch dev/double-sparsity-standalone. Key files:
- python/sglang/srt/layers/attention/double_sparsity/ (DS package: config.py, selector.py, selection_kernel.py, cuda_graph.py, page_signature_table.py, page_signature_write.py, page_table_adapter.py, validator.py, calibrate.py, channel_mask.py)
- python/sglang/srt/layers/attention/nsa_backend.py (KV-write sites: set_mla_kv_buffer ~L1383, ~L1583, ~L2108)
- python/sglang/srt/models/deepseek_v2.py (DS attention hook: _select_topk_indices L2060, _bind_double_sparsity_runtime_data L1832)
- python/sglang/srt/model_executor/forward_batch_info.py (ForwardBatch for M2 attachment)
- test/registered/unit/layers/attention/test_double_sparsity_unit.py (150 existing unit tests)
- development/loop4/draft.md (the draft being analyzed)

## Raw Draft Content
Loop 4 makes two main changes:
1. Architecture rotation to token-level signatures at page_size=64 (from page-level). Rename page_signature_table.py→token_label_table.py (shape [L_local, max_tokens, H_local, label_dim]), page_signature_write.py→token_label_write.py (per-token FP8 dequant + channel projection, slot-indexed by out_cache_loc), selection_kernel.py compute_page_scores→compute_token_scores (BGEMV Q_label·K_label_bufferᵀ), selector.py retrieve_topk returns (selected_token_indices: int32[bs, max_top_k_tokens], valid_lengths: int32[bs]).
2. Reach the MVP at Option B operating point (FP8 + flashmla_kv + overlap off + piecewise off): DS-on matches or beats DSA-on TPS at conc=64 with NIAH-Δ ≤ 5pp and MMLU-Δ ≤ 1pp.

Phase A (hard scope, 9 ACs AC-0 through AC-8):
- AC-0: Architecture rotation (file renames, shape changes, validator boot assert top_k==get_dsa_index_topk)
- AC-1/AC-1b: Token-label cache population from KV-write path + chunked-prefill probe
- AC-2: Token-label cache shares KV pool allocator lifetime (no leaks)
- AC-3: Per-request token range mask (excludes cross-request tokens in selector)
- AC-4: DSv3.2 calibration (channel mask generation, NOT committed to git)
- AC-5: Multi-process TP=2 rank sync (all_reduce(SUM) on scores, bit-equal selected_token_indices)
- AC-6: CUDA graph hardware capture at conc=64 (V3.2, Option B, no alloc, 100-step replay)
- AC-7: Short-seq MHA bypass (prefill below threshold skips selector; labels still written; decode runs DS)
- AC-8: End-to-end bench_serving + lightweight quality smoke (20 deterministic prompts from DSA-on reference)

Phase B (stretch, AC-9 through AC-12):
- AC-9: DSA baseline JSON at Option B
- AC-10: Radix cache ON under DS (DEC-2 flip)
- AC-11: Comparator row (DS TPS ≥ DSA TPS at conc=64, P99 TTFT ≤ 1.10×DSA)
- AC-12: NIAH/MMLU quality gate (5pp/1pp)

Hardware: single-node 8×H200 for Phase A+B; TP=2 multiprocess within single node for AC-5.

Operating point (locked): --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv --disable-overlap-schedule --disable-piecewise-cuda-graph --page-size 64, plus --enable-double-sparsity --double-sparsity-config '{"top_k":2048,...}' for DS runs.

Non-goals: piecewise CUDA graphs, overlap scheduler, MTP/EAGLE, GLM-5.1, 128K ISL, FP4, DP Attention, Phase C Triton kernel ports (gated on Phase B profile evidence), explicit chunked-prefill support (Loop 5).

## Task
Critically analyze this draft. Identify what could go wrong, what's missing, and what the strongest implementation path looks like.

Respond in EXACTLY this format (no extra sections):

CORE_RISKS:
- <risk 1>
- <risk 2>
...

MISSING_REQUIREMENTS:
- <missing item 1>
- <missing item 2>
...

TECHNICAL_GAPS:
- <gap 1>
- <gap 2>
...

ALTERNATIVE_DIRECTIONS:
- <alternative 1 with tradeoff>
- <alternative 2 with tradeoff>
...

QUESTIONS_FOR_USER:
- <question 1>
- <question 2>
...

CANDIDATE_CRITERIA:
- <AC suggestion 1>
- <AC suggestion 2>
...

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-05-27_09-30-34
- Tool: codex
