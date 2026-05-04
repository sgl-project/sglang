# RelayKV Phase 4.8 Isolated KV Index Resolution Design

## 日付確認

- Design date: **2026-05-03**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. この設計メモの目的

このメモは、RelayKV/SGLang integration の **Phase 4.8: Isolated KV Index Resolution Design** を定義する。

Phase 4.7 では、attention comparison plan から metadata-only の shadow capture result まで作成した。

```text
attention_comparison_plan
→ attention_shadow_capture_result
```

Phase 4.8 では、次の isolated attention comparison に必要になる index resolution の境界を設計する。

中心となる変換は以下。

```text
RelayKV block_id
→ token span
→ request-local token positions
→ req_to_token_pool.req_to_token
→ token_to_kv_pool physical KV index
→ layer-specific K/V tensor reference
```

重要:

```text
Phase 4.8 は設計フェーズ。
まだ KV pool read / tensor read / KV snapshot / attention execution は行わない。
```

---

## 2. 現在の到達点

Phase 4.1〜4.7 までの chain:

```text
attention_handoff_candidate
→ attention_connection_dry_run_result
→ attention_override_noop_result
→ attention_comparison_plan
→ attention_shadow_capture_result
```

Phase 4.7 で許可した counter:

```text
attention_shadow_capture_count > 0
shadow_capture_attempted_count > 0
```

Phase 4.7 時点で 0 維持している counter:

```text
attention_output_captured_count=0
attention_comparison_executed_count=0
attention_override_true_count=0
kv_pool_read_count=0
kv_snapshot_count=0
tensor_read_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
kv_cache_mutation_true_count=0
source_mutated_true_count=0
```

Phase 4.8 でも、これらの read/tensor/override 系 counter はまだ 0 のままにする。

---

## 3. Phase 4.8 の目的

Phase 4.8 の目的は、RelayKV の logical block plan を SGLang の KV index model に接続する前に、変換境界を明文化すること。

設計対象:

```text
1. block_id -> token span
2. token span -> request-local token positions
3. request-local token positions -> req_to_token_pool logical index entries
4. req_to_token entries -> token_to_kv_pool physical KV locations
5. physical KV locations -> layer-specific K/V tensor reference
```

ただし、Phase 4.8 では実 tensor reference を取得しない。

設計としては、以下の staged boundary に分ける。

```text
Stage A: metadata-only block span resolution
Stage B: req_to_token index resolution
Stage C: physical KV index resolution
Stage D: isolated K/V tensor read
Stage E: isolated attention comparison
```

Phase 4.8 で扱うのは **Stage A〜C の設計** まで。

---

## 4. SGLang KV index の責務分離

Phase 4.5 inspection で整理された責務:

```text
req_to_token_pool.req_to_token:
  request 内 logical token index mapping

token_to_kv_pool:
  layer-specific physical KV buffer

out_cache_loc:
  current step write destination
```

RelayKV で注意すべきこと:

```text
RelayKV block_id は SGLang physical KV index ではない。
RelayKV block_id はまず token span に展開する必要がある。
token span は request-local positions として解釈する必要がある。
request-local positions から req_to_token_pool を read-only 参照して physical KV locations へ変換する必要がある。
```

---

## 5. 変換モデル

### 5.1 RelayKV block_id -> token span

RelayKV の block は、少なくとも以下の情報を持つ必要がある。

```text
block_id
start_token
end_token
block_size
request_id
layer_id
```

推奨 schema:

```text
{
  "block_id": 12,
  "token_start": 1536,
  "token_end": 1664,
  "token_count": 128,
  "block_size": 128,
  "request_id": "...",
  "layer_id": 14,
  "span_source": "relaykv_block_metadata"
}
```

注意:

```text
token_end は exclusive として扱う。
```

理由:

```text
Python slice / range semantics と一致し、off-by-one を減らせる。
```

---

### 5.2 token span -> request-local token positions

token span から request-local positions を作る。

```text
positions = range(token_start, token_end)
```

必要な validation:

```text
token_start >= 0
token_end > token_start
token_end <= seq_len
token_count == token_end - token_start
```

この段階では SGLang pool には触れない。

---

### 5.3 request-local token positions -> req_to_token entries

次の段階で、request-local positions を `req_to_token_pool.req_to_token` に対応させる。

概念的には:

```text
req_pool_idx + request-local token position
→ req_to_token_pool.req_to_token[req_pool_idx, position]
→ token index / cache loc
```

ただし、実装前に確認すべき点:

```text
req_to_token_pool.req_to_token の shape
decode/prefill での position semantics
batched request での req_pool_idx handling
padding / unused slot の扱い
extend path と decode path の違い
```

Phase 4.8 ではまだ read しない。

---

### 5.4 req_to_token entries -> token_to_kv_pool physical locations

`req_to_token` で得られる index は、さらに `token_to_kv_pool` の physical KV location と対応する。

概念:

```text
logical token index
→ token_to_kv_pool mapping
→ physical KV pool location
```

確認すべき点:

```text
token_to_kv_pool が layer-specific か shared か
K/V buffer への indexing order
MHA/GQA/MLA で layout が変わるか
decode と extend で参照 path が違うか
```

Phase 4.8 では、この mapping の read-only resolution helper を設計するが、まだ実装しない。

---

### 5.5 physical locations -> layer-specific K/V tensor reference

最終的に isolated comparison では、physical KV locations を使って K/V tensor を読む必要がある。

ただし、これは Phase 4.8 では禁止。

将来許可する段階:

```text
Phase 4.9 or later:
  kv_pool_read_count > 0
  tensor_read_count > 0
```

Phase 4.8 では、この境界だけを定義する。

---

## 6. Stage 分離

### Stage A: metadata-only block span resolution

入力:

```text
attention_shadow_capture_result
relaykv_working_kv_block_ids
full_kv_block_ids
block_metadata_by_id
```

出力候補:

```text
relaykv_kv_index_resolution_plan
```

fields:

```text
event_type="relaykv_kv_index_resolution_plan"
resolution_state="block_span_resolved"
resolution_mode="metadata_only"
request_id
req_pool_idx
seq_len
layer_id
block_id
token_start
token_end
token_count
```

許可:

```text
metadata transform only
```

禁止:

```text
req_to_token_pool read
token_to_kv_pool read
KV tensor read
```

---

### Stage B: req_to_token index resolution

将来の isolated smoke で許可候補。

入力:

```text
block span resolution plan
req_to_token_pool readonly handle
```

出力候補:

```text
relaykv_req_to_token_resolution_result
```

許可候補:

```text
req_to_token_read_count > 0
```

まだ禁止:

```text
token_to_kv_pool read
KV tensor read
attention execution
runtime writeback
scheduler mutation
```

---

### Stage C: physical KV index resolution

入力:

```text
req_to_token_resolution_result
token_to_kv_pool readonly handle
```

出力候補:

```text
relaykv_physical_kv_index_resolution_result
```

許可候補:

```text
token_to_kv_pool_read_count > 0
physical_kv_index_resolved_count > 0
```

まだ禁止:

```text
KV tensor read
attention execution
attention override
runtime writeback
scheduler mutation
```

---

### Stage D: isolated K/V tensor read

将来の isolated comparison smoke。

許可候補:

```text
kv_pool_read_count > 0
tensor_read_count > 0
```

まだ禁止:

```text
attention_override_true_count > 0
runtime_writeback_true_count > 0
scheduler_policy_noop_false_count > 0
source_mutated_true_count > 0
```

---

### Stage E: isolated attention comparison

将来フェーズ。

許可候補:

```text
attention_comparison_executed_count > 0
```

まだ禁止:

```text
attention_override_true_count > 0
production output modification
runtime writeback
scheduler mutation
KV cache mutation
```

---

## 7. Phase 4.8 で作るべき smoke

Phase 4.8 の最小実装は、まだ req_to_token_pool を読まない smoke がよい。

候補名:

```text
Phase 4.8.1 KV Index Resolution Plan Smoke
```

内容:

```text
attention_shadow_capture_result
+ block_metadata_by_id
→ relaykv_kv_index_resolution_plan
```

これは metadata-only で、次を検証する。

```text
block_id が token span に解決できる
token_start/token_end が seq_len 内
working blocks と full blocks の span が作れる
missing block metadata は blocked になる
```

この smoke では以下を 0 維持する。

```text
req_to_token_read_count=0
token_to_kv_pool_read_count=0
kv_pool_read_count=0
tensor_read_count=0
attention_comparison_executed_count=0
attention_override_true_count=0
```

---

## 8. 推奨 helper schema

### 8.1 build_relaykv_kv_index_resolution_plans_for_smoke

候補 signature:

```python
build_relaykv_kv_index_resolution_plans_for_smoke(
    attention_shadow_capture_results,
    block_metadata_by_id=None,
)
```

入力:

```text
attention_shadow_capture_results:
  list/tuple of relaykv_attention_shadow_capture_result

block_metadata_by_id:
  dict
  key candidates:
    block_id
    (request_id, layer_id, block_id)
    "request_id:layer_id:block_id"
```

正常 input:

```text
event_type="relaykv_attention_shadow_capture_result"
shadow_capture_state="metadata_shadow_captured"
shadow_capture_mode="metadata_only"
attention_output_captured=false
attention_comparison_executed=false
attention_override=false
relaykv_working_kv_block_ids non-empty
full_kv_block_ids non-empty
```

正常 output:

```text
event_type="relaykv_kv_index_resolution_plan"
resolution_state="block_span_resolved"
resolution_mode="metadata_only"
source="attention_shadow_capture_result_to_kv_index_resolution_plan"
request_id
req_pool_idx
seq_len
layer_id

relaykv_working_block_spans
full_kv_block_spans
resolved_block_count
missing_block_ids
token_span_count
total_token_count

req_to_token_read=false
token_to_kv_pool_read=false
kv_pool_read=false
tensor_read=false
attention_comparison_executed=false
attention_override=false
runtime_writeback=false
scheduler_policy_noop=true
kv_cache_mutation=false
source_mutated=false
```

blocked reasons:

```text
not_attention_shadow_capture_result
shadow_capture_not_metadata_captured
shadow_capture_not_metadata_only
attention_output_captured_not_allowed
attention_comparison_executed_not_allowed
attention_override_true_not_allowed
no_relaykv_working_kv_blocks
no_full_kv_blocks
missing_block_metadata
invalid_block_span
block_span_out_of_seq_len
```

---

### 8.2 summarize_relaykv_kv_index_resolution_plans_for_smoke

候補 summary fields:

```text
summary_type="relaykv_kv_index_resolution_plan_summary"
total_kv_index_resolution_plans
block_span_resolved_count
blocked_count
error_count

resolved_block_count
missing_block_count
token_span_count
total_token_count

per_request_counts
per_layer_counts
per_resolution_state_counts

req_to_token_read_count
token_to_kv_pool_read_count
kv_pool_read_count
tensor_read_count
attention_comparison_executed_count
attention_override_true_count
runtime_writeback_true_count
scheduler_policy_noop_false_count
kv_cache_mutation_true_count
source_mutated_true_count
```

---

## 9. safety counters

Phase 4.8.1 metadata-only smoke の expected safety:

```text
req_to_token_read_count=0
token_to_kv_pool_read_count=0
kv_pool_read_count=0
kv_snapshot_count=0
tensor_read_count=0
attention_comparison_executed_count=0
attention_override_true_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
kv_cache_mutation_true_count=0
source_mutated_true_count=0
```

将来、Stage B/C で個別に許可する可能性がある counter:

```text
req_to_token_read_count > 0
token_to_kv_pool_read_count > 0
physical_kv_index_resolved_count > 0
```

ただし、許可する際も production output は不変にする。

---

## 10. fallback / blocked design

blocked にする条件:

```text
shadow capture result ではない
metadata_shadow_captured ではない
metadata_only ではない
attention_output_captured=true
attention_comparison_executed=true
attention_override=true
working blocks empty
full blocks empty
block metadata missing
token span invalid
token span out of seq_len
```

blocked output でも安全 flags は必ず 0/false 維持。

```text
req_to_token_read=false
token_to_kv_pool_read=false
kv_pool_read=false
tensor_read=false
attention_comparison_executed=false
attention_override=false
runtime_writeback=false
scheduler_policy_noop=true
kv_cache_mutation=false
source_mutated=false
```

---

## 11. 次フェーズ案

推奨順:

```text
Phase 4.8.1:
  KV Index Resolution Plan Smoke
  - metadata-only
  - block_id -> token span
  - no req_to_token read

Phase 4.8.2:
  Req-to-token Readonly Resolution Design
  - req_to_token_pool read boundary
  - still no token_to_kv_pool / tensor read

Phase 4.8.3:
  Physical KV Index Readonly Resolution Design
  - token_to_kv_pool read boundary
  - still no K/V tensor read

Phase 4.9:
  Isolated K/V Tensor Read Design or Smoke
  - kv_pool_read_count > 0 and tensor_read_count > 0 allowed only here

Phase 4.10:
  Isolated Attention Comparison Smoke
  - attention_comparison_executed_count > 0
  - attention_override_true_count remains 0
```

---

## 12. 結論

Phase 4.8 では、RelayKV block plan を SGLang physical KV index に直接繋がず、まず metadata-only の resolution plan を作るべき。

最小安全経路:

```text
attention_shadow_capture_result
→ relaykv_kv_index_resolution_plan
```

ここではまだ以下を許可しない。

```text
req_to_token_pool read
token_to_kv_pool read
KV pool read
tensor read
attention execution
attention override
runtime writeback
scheduler mutation
```

次に実装するなら、**Phase 4.8.1 KV Index Resolution Plan Smoke** が適切。

---

## 13. commit command

この設計メモを repo に保存する場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/relaykv_phase4_8_isolated_kv_index_resolution_design_2026-05-03.ja.md \
  notes/relaykv_phase4_8_isolated_kv_index_resolution_design_2026-05-03.ja.md

git status --short
git diff --check

git add notes/relaykv_phase4_8_isolated_kv_index_resolution_design_2026-05-03.ja.md
git commit -m "docs: design relaykv isolated kv index resolution"
git push mine relaykv-host-backup-shadow
```
