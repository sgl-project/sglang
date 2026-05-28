AGREE:
- Rotating labels to be slot-indexed by `out_cache_loc` is directionally reasonable.
- Hooking `set_mla_kv_buffer` in live `dsa_backend.py` sites is the right integration target.

DISAGREE:
- `selected_token_indices` cannot remain ambiguous — physical-slot output makes range masks, TP sync, and sequence ordering fragile.
- The FlashMLA adapter should not be described as token-to-`block_table` for Option B — `flashmla_kv` sparse path consumes flattened physical token `indices`, with `block_table` effectively unused.
- Full `[L,T,H,D]` fp16 token labels are likely too expensive — this is roughly 64x the page-label table and can add tens of GB per rank.
- Channel axis = 128 is not valid for the proposed `set_mla_kv_buffer` hook unless the data source changes — that hook writes MLA latent K with `kv_lora_rank=512`.

REQUIRED_CHANGES:
- Define selector ABI as logical sequence token positions; score in logical position space via `req_to_token`, then convert to physical token slots after top-K.
- Replace range masks over physical slots with `req_to_token`-based ownership/gather logic.
- Specify adapter output for Option B as int32 physical flattened KV token indices, `-1` padded, exactly `get_dsa_index_topk`.
- Add an HBM budget gate for token labels, or change storage to a bounded/quantized/on-demand design.
- Resolve calibration/write-space mismatch: either use 512-d latent K labels, or hook an actual 128-d K source.
- Move radix-cache and NIAH/MMLU quality gates into hard scope if the loop claims “MVP reached.”
- Define or remove `device_buffer_size` after token-level rotation.

OPTIONAL_IMPROVEMENTS:
- Keep `top_k == get_dsa_index_topk(hf_config)` as a boot assert.
- Add a small FlashMLA adapter fixture proving logical tokens → physical token indices under page_size=64 and non-contiguous KV slots.

UNRESOLVED:
- Whether token labels are stored persistently for all KV slots or computed/gathered lazily.
- Whether MVP means Option B-only smoke success or parity against the default DSA + radix + CUDA-graph baseline.
