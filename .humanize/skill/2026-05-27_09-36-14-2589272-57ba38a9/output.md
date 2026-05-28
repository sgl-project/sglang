CORE_RISKS:
- Token index domain is underspecified: label storage is physical-slot indexed by `out_cache_loc`, but the DSA transform path expects logical token positions into `req_to_token`; mixing those will silently select wrong KV.
- Token-level memory and score cost are underestimated for concurrency: 4K ISL at conc=64 is roughly 262K live tokens, not a 4K cache, so label HBM and `[bs, max_tokens]` scoring can dominate decode.
- The draft’s “2048 tokens = at most 32 pages” claim is only true for contiguous tokens; worst case is 2048 distinct pages, which can erase the expected FlashMLA-side sparsity win.
- `req_to_token` ownership is not a contiguous range under radix reuse, chunked prefill, eviction, or mixed batches; a range mask can include foreign or stale slots.
- Phase A does not actually prove the stated MVP comparator if radix, DSA baseline JSON, and quality gates remain stretch.
- CUDA graph capture is likely harder than described: current selector/adapter paths allocate tensors, use `torch.topk` outputs, have CPU-sync validation, and expose hard-coded top-k assumptions.

MISSING_REQUIREMENTS:
- Define whether `selected_token_indices` are logical sequence positions or physical KV slots, and make every downstream contract use that same domain.
- Specify token-label cache sizing from the real KV pool capacity, with boot-time HBM estimate and fail-fast budget.
- Replace “per-request range mask” with exact ownership via `req_to_token_pool.req_to_token[req_pool_indices, :seq_len]`, or prove contiguity.
- Add the hot-token/local-window rule from the token-level design: N, merge behavior, ordering, padding, and short-sequence behavior.
- Define stale-label semantics for freed, evicted, reused, radix-hit, and offloaded slots.
- Make write-hook coverage explicit across dense MHA prefill, DSA prefill, decode, chunked prefill, and FP8 paths.
- Require benchmark artifacts to include git SHA, full server args, model revision/path, mask hash, prompt mix, cache-hit setup, warmup, chunked-prefill setting, and radix setting.

TECHNICAL_GAPS:
- `nsa_backend.py` is now a deprecated re-export; live write sites are in `dsa_backend.py` and possibly the MLA memory pool.
- Current `transform_index_page_table_decode` asserts `page_size == 1`, and the fast path hard-codes `TOPK: 2048`; this conflicts with the draft’s `page_size=64` and configurable `top_k` language.
- `device_buffer_size=4096` conflicts with token-level “max_tokens is KV pool slot count” semantics for conc=64.
- Existing table/selector code is page-shaped with `valid_mask[layer,page]`; a rename alone does not address physical-slot validity or exact request ownership.
- Current `DSGraphState` does not preallocate score, top-k, sort, gather, or transformed-index scratch needed by token scoring.
- Calibration needs a stricter V3.2 tensor contract: DSA index dim is 128, while DS label projection says post-RoPE/nope dim 512.
- FP8 label writing must define whether it reads pre-quantized `k`, quantized KV cache rows, or DSA index buffers; those paths have different shapes and scales.

ALTERNATIVE_DIRECTIONS:
- Score in logical-token space by gathering physical label rows through `req_to_token`; tradeoff: extra gather/ragged kernel, but it fixes ownership, radix, chunking, and transform compatibility.
- Put label writes behind `DSATokenToKVPool.set_mla_kv_buffer` or an optional memory-pool callback; tradeoff: tighter pool coupling, but much lower hook-site drift.
- Split Loop 4 into eager token-level bring-up first, then CUDA graph/radix/comparator second; tradeoff: slower MVP claim, but clearer failure isolation.
- Keep per-token labels but dedupe or page-coalesce selected tokens before FlashMLA; tradeoff: less exact sglang-last behavior, but better chance to meet TPS.

QUESTIONS_FOR_USER:
- Should `selected_token_indices` be logical sequence positions or physical KV slots?
- Is `device_buffer_size=4096` still intended after token-level rotation, and what exactly should it bound?
- Are physical KV slot IDs guaranteed identical across TP ranks, or should all-reduce happen in logical-token space?
- Should radix ON and the DSA comparator move from stretch into hard scope for MVP?
- Is the hot-token rule required in Phase A?
- If chunked prefill probe fails, is disabling chunked prefill acceptable for the client SLO run?

CANDIDATE_CRITERIA:
- AC-0: selector emits logical token positions; a non-contiguous physical-slot fixture must transform to the correct KV indices.
- AC-1: one memory-pool-level hook populates labels for all KV-write paths, including dense MHA prefill.
- AC-2: cache sizing derives from KV pool slots, logs GB/rank, and refuses unsafe HBM budgets.
- AC-3: interleaved requests plus stale freed slots cannot be selected.
- AC-5: TP test includes a physical-slot permutation negative case, not only score perturbation.
- AC-6: decode selector runs with preallocated buffers and zero allocation/CPU sync before hardware graph capture.
- AC-8: report selected logical tokens, unique physical tokens, duplicate pages, and effective FlashMLA KV length.
- AC-11/12: make comparator and lightweight quality gates hard if the loop’s stated goal is “MVP reached.”
