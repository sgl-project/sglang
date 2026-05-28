# Ask Codex Input

## Question

You are reviewing a CANDIDATE PLAN (v1) for Loop 4 of the Double Sparsity MVP on DeepSeek-V3.2 (FP8) in the SGLang LLM serving framework.

## Context
The plan rotates the Double Sparsity (DS) implementation from page-level to token-level label storage, then benchmarks end-to-end at the Option B operating point (FP8 + flashmla_kv + overlap off + piecewise off) on 8xH200.

## Candidate Plan v1 Summary

### Core Architecture Change (AC-0)
- Rename page_signature_table.py → token_label_table.py (shape [L,P,H,D] → [L,T,H,D] where T=KV pool slot count)
- Rename page_signature_write.py → token_label_write.py (per-token FP8 dequant + channel projection, slot-indexed by out_cache_loc)
- selection_kernel.py: compute_page_scores → compute_token_scores (BGEMV Q_label·K_label_bufferT, outputs [bs, max_tokens])
- selector.py: retrieve_topk returns (selected_token_indices: int32[bs, max_top_k], valid_lengths: int32[bs]) sequence-ascending
- config.py: top_k = max tokens per request (default 2048, matching V3.2 index_topk)
- validator.py: boot assert top_k == get_dsa_index_topk(hf_config)
- page_table_adapter.py: NEEDS CONCRETE STRATEGY - current 404 LOC converts page IDs to FlashMLA block table. Token-level replacement must convert selected_token_indices to FlashMLA page_size=64 block table. NOTE: dsa/transform_index.py::transform_index_page_table_decode asserts page_size==1 so cannot be used directly.
- cuda_graph.py DSGraphState: shape update only (selected_indices: [max_bs, max_top_k_tokens])
- All 150 Loop-2 unit tests migrated for shape changes

### M1 Hook (AC-1)
- Hook set_mla_kv_buffer in dsa_backend.py at L1439 (extend/prefill), L1637 (decode), L2162 (TRT-LLM MLA path)
- CORRECTION: draft said nsa_backend.py which is a deprecated re-export shim; live sites are dsa_backend.py

### M2 Range Mask (AC-3)
- Per-request token range mask in ForwardBatch: (req_start[r], req_end[r]) derived from req_to_token_pool.req_to_token, req_pool_indices, seq_lens
- PENDING: Codex v1 flagged that req_to_token is not contiguous under radix/chunked/eviction. Range mask may include foreign slots.

### Calibration (AC-4)
- calibrate.py updated for DSv3.2 MLA layers
- CORRECTION: draft says channel axis=512; code uses qk_nope_head_dim=128 for k_head_dim. Channel axis is 128 (nope dim), not 512 (kv_lora_rank). label_dim=16 per draft's calibration command.

### TP Rank Sync (AC-5)
- Multi-process TP=2 test, all_reduce(SUM) on [max_tokens] scores
- PENDING: Codex v1 asked whether physical KV slot IDs are identical across TP ranks

### CUDA Graph (AC-6)
- Full-graph capture at conc=64, preallocated buffers, zero CPU sync

### End-to-end bench_serving (AC-8)
- 8xH200, Option B operating point, conc 16/32/64, ≥64 requests
- Quality smoke: 20 deterministic prompts vs DSA-on reference

### Phase B (stretch, AC-9 through AC-12)
- DSA baseline JSON, radix cache ON, comparator row, NIAH/MMLU quality gate

## Unresolved Issues from First Codex Pass
1. page_table_adapter strategy: The thin wrapper cannot use dsa/transform_index.py (page_size==1 assertion). What is the correct token-to-block-table conversion for FlashMLA page_size=64?
2. selected_token_indices domain: physical KV slots or logical sequence positions? (affects range mask, TP sync, adapter)
3. Codex suggested AC-11/AC-12 should move to hard scope if the loop goal is 'MVP reached'
4. device_buffer_size=4096 semantics post-rotation

## Your Task
Review this candidate plan for reasonability. Respond in EXACTLY this format:

AGREE:
- <point 1 that is reasonable>
- <point 2>

DISAGREE:
- <point 1> — reason why this is problematic

REQUIRED_CHANGES:
- <must-fix item 1>
- <must-fix item 2>

OPTIONAL_IMPROVEMENTS:
- <non-blocking improvement 1>
- <non-blocking improvement 2>

UNRESOLVED:
- <topic requiring user decision 1>
- <topic requiring user decision 2>

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-05-27_09-50-23
- Tool: codex
