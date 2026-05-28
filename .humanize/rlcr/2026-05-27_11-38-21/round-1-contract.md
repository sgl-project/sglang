# Round 1 Contract

## Mainline Objective

Close AC-0 and AC-13 for real: repair the 5 verified gaps reported by Codex
Round-0 review so that:
- `from sglang.srt.layers.attention.double_sparsity import TokenLabelTable, token_label_write, retrieve_topk` succeeds
- `retrieve_topk_via_labels` scores in logical-position domain when `req_to_token` is provided
- The adapter (`logical_to_physical`) correctly maps logical top-K positions to physical KV slots
- TokenLabelTable is sized from `req_to_token_pool.size` (not `device_buffer_size`)
- DS selector receives projected Q-noPE `[bs, H, 128]`, not latent `q_lora`
- `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` passes all 150 tests

## Target ACs

- **AC-0**: Architecture rotation acceptance criteria — public import, selector domain, bind timing, Q-noPE input, non-contiguous fixture
- **AC-13**: 150-test regression suite fully green

## Blocking Issues (must close this round)

| Issue | File | Fix |
|-------|------|-----|
| `retrieve_topk` not exported | `__init__.py` | add alias + `__all__` entry |
| Selector scores physical slots, returns physical indices | `selection_kernel.py`, `selector.py` | add optional `req_to_token` path for logical-domain scoring |
| DS bind before KV pool exists; falls back to `device_buffer_size` | `deepseek_v2.py`, `model_runner.py` | defer bind to post-`init_memory_pool()`; fail fast if pool unavailable |
| DS path passes latent `q_lora` instead of projected Q-noPE | `forward_mla.py`, `deepseek_v2.py` | disable alt-stream for DS; derive `q_nope = q[..., :qk_nope_head_dim]` after `q_b_proj`; pass through `_select_topk_indices` |
| 3 failing tests + stale `nsa_*` names in test file | test file | targeted test fixes only |

## Queued (out of scope this round)

- AC-1 hook wiring in `dsa_backend.py`
- AC-2 through AC-12 (all remaining ACs)

## Success Criteria

1. `python -c "from sglang.srt.layers.attention.double_sparsity import TokenLabelTable, token_label_write, retrieve_topk; print('OK')"` prints OK
2. Non-contiguous fixture: `req_to_token = [7, 64, 200, 512]`, selector output logical `[0,1,2,3]`, adapter maps to physical `[7,64,200,512]`
3. `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` → 150 passed, 0 failed
4. `_bind_double_sparsity_runtime_data` raises if `_ds_req_to_token_pool` is absent (no silent fallback)
5. `_select_topk_indices` passes `q[..., :qk_nope_head_dim]` to DS selector (not latent `q_lora`)
