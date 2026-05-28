# Round 2 Contract

## Mainline Objective

Fix the AC-0 slot-count authority bug (TokenLabelTable sized from request-pool rows, not
physical KV slots), add a committed real-selector regression test for non-contiguous slots,
and wire the AC-1 token-label write hooks at all three `dsa_backend.py` `set_mla_kv_buffer`
call sites so live population flows through the hook.

## Target ACs

- **AC-0**: Architecture rotation fully closed — TokenLabelTable sized from physical KV slot
  capacity (`token_to_kv_pool.size + token_to_kv_pool.page_size`); real regression test for
  non-contiguous physical slots passes.
- **AC-1**: Live token-label cache population — after `forward_extend`, `forward_decode`, and
  the TRT-LLM path, `token_label_table.signatures[layer_id, out_cache_loc]` is non-zero for
  each written slot.

## Blocking Issues (must close this round)

| Issue | File | Fix |
|-------|------|-----|
| `max_tokens` bound from `req_to_token_pool.size` (request rows), not physical KV slot count | `deepseek_v2.py`, `model_runner.py` | Publish `_ds_token_to_kv_pool` from ModelRunner after `init_memory_pool()`; compute `max_tokens = kv_pool.size + kv_pool.page_size` in bind |
| Real non-contiguous slot regression not committed | test file | Add test: `max_tokens=600`, physical slots `[7,64,200,512]`; real scoring path; verify write + retrieve + adapter roundtrip |
| AC-1 hooks not wired at 3 `set_mla_kv_buffer` sites | `dsa_backend.py` | After each KV write: project `k` through `layer.kv_b_proj`, slice K-noPE per head, call `token_label_write` |

## Queued (out of scope this round)

- AC-6 graph helper: `capture_decode_step` missing `req_to_token` (will block AC-6)
- AC-8 observability: `_publish_ds_request_summary` uses page-named metrics (will mislead at AC-8)
- AC-2 stale-slot/lifetime, AC-3 ownership, AC-7 bypass, AC-4/AC-5/AC-6/AC-8/AC-12

## Success Criteria

1. `python -c "from sglang.srt.layers.attention.double_sparsity import TokenLabelTable, token_label_write, retrieve_topk; print('OK')"` prints OK (unchanged from Round 1)
2. `TestAC0RealSlotRegression`: `TokenLabelTable(max_tokens=600)` writes to slots `[7,64,200,512]` without IndexError; real `_compute_logical_token_scores` scoring returns logical `[0,1,2,3]`; adapter maps back to physical `[7,64,200,512]`; writing with `max_tokens=4` raises IndexError (proves old `req_to_token_pool.size` was wrong)
3. AC-1 hook unit test: mock bind with synthetic table + channel mask; call `forward_extend` stub; assert `signatures[layer_id, out_cache_loc]` is non-zero
4. `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` → 150+ passed, 0 failed
