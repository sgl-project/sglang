# RelayKV Phase 4.8.2 Req-to-token Readonly Resolution Design

## 日付確認

- Design date: **2026-05-04**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. この設計メモの目的

このメモは、RelayKV/SGLang integration の **Phase 4.8.2: Req-to-token Readonly Resolution Design** を定義する。

Phase 4.8.1 では、metadata-only で次の chain まで進んだ。

```text
attention_shadow_capture_result
→ relaykv_kv_index_resolution_plan
```

Phase 4.8.1 の到達点:

```text
RelayKV block_id
→ token span
```

Phase 4.8.2 では、次の段階として、token span を SGLang の `req_to_token_pool.req_to_token` に read-only で対応させる境界を設計する。

中心となる変換:

```text
block span
→ request-local token positions
→ req_to_token_pool.req_to_token readonly resolution
```

重要:

```text
Phase 4.8.2 は設計フェーズ。
まだ token_to_kv_pool は読まない。
まだ KV pool / K/V tensor は読まない。
まだ attention comparison は実行しない。
まだ attention override はしない。
```

---

## 2. 現在の到達点

Phase 4.8.1 でできている chain:

```text
attention_comparison_plan
→ attention_shadow_capture_result
→ relaykv_kv_index_resolution_plan
```

Phase 4.8.1 の pass-flow summary:

```text
block_span_resolved_count=2
resolved_block_count=12
token_span_count=12
total_token_count=1536
```

Phase 4.8.1 で 0 維持できている safety counters:

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

Phase 4.8.2 で初めて許可候補になる counter:

```text
req_to_token_read_count > 0
```

ただし、次はまだ 0 固定。

```text
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

---

## 3. Phase 4.8.2 の目的

Phase 4.8.2 の目的は、RelayKV の token span を SGLang の request-local logical token mapping に接続すること。

設計対象:

```text
relaykv_kv_index_resolution_plan
→ req_to_token readonly resolution result
```

ただし、ここで解決するのは **physical KV index ではない**。

Phase 4.8.2 で得るべきもの:

```text
request_id
req_pool_idx
layer_id
seq_len
block_id
token_start
token_end
request-local token positions
req_to_token entries
```

Phase 4.8.2 で得ないもの:

```text
token_to_kv_pool physical index
K/V tensor reference
attention output
logits
generated token
```

---

## 4. SGLang req_to_token の位置づけ

Phase 4.5 inspection で整理した責務:

```text
req_to_token_pool.req_to_token:
  request 内 logical token index mapping

token_to_kv_pool:
  layer-specific physical KV buffer

out_cache_loc:
  current step write destination
```

Phase 4.8.2 で読む対象は `req_to_token_pool.req_to_token` のみ。

概念的な解決:

```text
req_pool_idx + request-local token position
→ req_to_token_pool.req_to_token[req_pool_idx, position]
→ logical token/cache location entry
```

注意:

```text
この entry はまだ layer-specific K/V tensor location そのものではない。
```

---

## 5. read-only 境界

### 5.1 許可する読み取り

Phase 4.8.2 の smoke で許可する可能性がある読み取り:

```text
req_to_token_pool.req_to_token read-only access
```

許可 counter:

```text
req_to_token_read_count > 0
```

許可する読み取りの条件:

```text
1. input が relaykv_kv_index_resolution_plan
2. resolution_state="block_span_resolved"
3. resolution_mode="metadata_only"
4. req_pool_idx が存在する
5. seq_len が正
6. token spans が seq_len 内
7. req_to_token_pool handle が明示的に渡される
8. read_only=True 相当の smoke helper 内に閉じる
```

### 5.2 禁止する読み取り

Phase 4.8.2 では以下を禁止する。

```text
token_to_kv_pool read
KV pool read
K/V tensor read
KV snapshot
attention execution
attention output capture
attention override
runtime writeback
scheduler mutation
KV cache mutation
source mutation
```

---

## 6. 入力 schema

Phase 4.8.2 helper の入力は、Phase 4.8.1 output を想定する。

候補 signature:

```python
build_relaykv_req_to_token_resolution_results_for_smoke(
    kv_index_resolution_plans,
    req_to_token_table_by_req_pool_idx=None,
    read_req_to_token=False,
)
```

`read_req_to_token` は明示的に opt-in にする。

```text
read_req_to_token=False:
  blocked または dry-run no-read result

read_req_to_token=True:
  supplied table から read-only resolution を行う
```

ただし、実 SGLang object への直接 attribute access はこの段階では避ける。

短期の安全実装:

```text
req_to_token_table_by_req_pool_idx は dict/list/tuple のみ。
実 req_to_token_pool object は Phase 4.8.2 smoke では直接読まない。
```

理由:

```text
poison object / GPU tensor / torch Tensor / production object に触れないため。
```

---

## 7. req_to_token table の表現

Phase 4.8.2 smoke では、実 pool object ではなく、dict/list/tuple の synthetic table を使う。

候補 schema:

```text
req_to_token_table_by_req_pool_idx = {
  7: [1000, 1001, 1002, ...],
  8: [2000, 2001, 2002, ...],
}
```

または:

```text
req_to_token_table_by_req_pool_idx = {
  "7": [1000, 1001, 1002, ...],
}
```

key lookup:

```text
req_pool_idx
str(req_pool_idx)
```

positions lookup:

```text
for position in range(token_start, token_end):
    req_to_token_entry = table[position]
```

validation:

```text
position >= 0
position < len(table)
position < seq_len
entry is int
```

---

## 8. 出力 schema

正常 output 候補:

```text
event_type="relaykv_req_to_token_resolution_result"
resolution_state="req_to_token_resolved"
resolution_mode="readonly_synthetic_table"
source="kv_index_resolution_plan_to_req_to_token_resolution_result"

request_id
req_pool_idx
seq_len
layer_id

relaykv_working_req_to_token_spans
full_kv_req_to_token_spans

resolved_block_count
resolved_token_count
req_to_token_entry_count

req_to_token_read=true
req_to_token_read_count=<number of entries read>
token_to_kv_pool_read=false
kv_pool_read=false
kv_snapshot=false
tensor_read=false
attention_comparison_executed=false
attention_override=false
runtime_writeback=false
scheduler_policy_noop=true
kv_cache_mutation=false
source_mutated=false

blocking_reasons=[]
warning_reasons includes "readonly_req_to_token_resolution"
warning_reasons includes "no_token_to_kv_pool_read"
```

span record schema:

```text
{
  "block_id": 1,
  "token_start": 128,
  "token_end": 256,
  "token_count": 128,
  "request_id": "req-a",
  "req_pool_idx": 7,
  "layer_id": 14,
  "req_to_token_entries": [1128, 1129, ...],
  "entry_count": 128,
  "resolution_source": "synthetic_req_to_token_table"
}
```

注意:

```text
req_to_token_entries は smoke では小さくても良いが、実装時には巨大 list 化によるログ肥大化に注意。
```

推奨:

```text
Phase 4.8.2 smoke では entry list を保持してよい。
実 runtime では preview / count / checksum に切り替える。
```

---

## 9. blocked design

blocked reasons:

```text
not_kv_index_resolution_plan
kv_index_resolution_not_block_span_resolved
kv_index_resolution_not_metadata_only
read_req_to_token_not_enabled
req_pool_idx_missing_or_invalid
seq_len_missing_or_invalid
req_to_token_table_missing
req_to_token_table_for_req_pool_missing
req_to_token_table_not_indexable
token_position_out_of_req_to_token_table
req_to_token_entry_not_int
invalid_block_span
token_span_out_of_seq_len
```

blocked output:

```text
event_type="relaykv_req_to_token_resolution_result"
resolution_state="blocked"
resolution_mode="readonly_synthetic_table"
source="kv_index_resolution_plan_to_req_to_token_resolution_result"

request_id / req_pool_idx / seq_len / layer_id copied when available

relaykv_working_req_to_token_spans=[]
full_kv_req_to_token_spans=[]

resolved_block_count=0
resolved_token_count=0
req_to_token_entry_count=0

req_to_token_read=false
req_to_token_read_count=0
token_to_kv_pool_read=false
kv_pool_read=false
kv_snapshot=false
tensor_read=false
attention_comparison_executed=false
attention_override=false
runtime_writeback=false
scheduler_policy_noop=true
kv_cache_mutation=false
source_mutated=false

blocking_reasons set
```

---

## 10. summary schema

候補 helper:

```python
summarize_relaykv_req_to_token_resolution_results_for_smoke(results)
```

summary fields:

```text
summary_type="relaykv_req_to_token_resolution_result_summary"

total_req_to_token_resolution_results
req_to_token_resolved_count
blocked_count
error_count

resolved_block_count
resolved_token_count
req_to_token_entry_count

per_request_counts
per_layer_counts
per_resolution_state_counts

req_to_token_read_count
token_to_kv_pool_read_count
kv_pool_read_count
kv_snapshot_count
tensor_read_count
attention_comparison_executed_count
attention_override_true_count
runtime_writeback_true_count
scheduler_policy_noop_false_count
kv_cache_mutation_true_count
source_mutated_true_count
```

Expected normal summary in small smoke:

```text
req_to_token_resolved_count=2
resolved_block_count=12
resolved_token_count=1536
req_to_token_entry_count=1536
req_to_token_read_count=1536
token_to_kv_pool_read_count=0
kv_pool_read_count=0
tensor_read_count=0
attention_comparison_executed_count=0
attention_override_true_count=0
```

---

## 11. Phase 4.8.2 smoke 方針

Phase 4.8.2 の最小 smoke は、実 SGLang pool object を使わず synthetic table で行う。

候補名:

```text
scripts/relaykv_req_to_token_resolution_smoke.py
```

pass flow:

```text
relaykv_kv_index_resolution_plan x 2
+ req_to_token_table_by_req_pool_idx
→ relaykv_req_to_token_resolution_result x 2
```

sample:

```text
req-a:
  req_pool_idx=7
  seq_len=768
  table=[1000, 1001, ..., 1767]

req-b:
  req_pool_idx=8
  seq_len=768
  table=[2000, 2001, ..., 2767]
```

expected:

```text
req_to_token_resolved_count=2
resolved_block_count=12
resolved_token_count=1536
req_to_token_entry_count=1536
req_to_token_read_count=1536
```

still zero:

```text
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

---

## 12. 実 SGLang object への接続はまだしない

Phase 4.8.2 では synthetic table までに留める。

実 SGLang object へ接続する場合は、次の段階で別途設計する。

候補:

```text
Phase 4.8.3 Real Req-to-token Readonly Adapter Design
```

または:

```text
Phase 4.8.3 Req-to-token Pool Readonly Adapter Smoke
```

その段階で初めて確認すること:

```text
req_to_token_pool.req_to_token の actual shape
CPU/GPU tensor かどうか
`.tolist()` を禁止するか、限定するか
`.cpu()` を許可しないか
batch slicing をどうするか
log size をどう抑えるか
```

現時点では、production object に触らず、まず pure Python synthetic smoke で schema を固定する。

---

## 13. safety policy

Phase 4.8.2 の安全原則:

```text
Allowed:
  synthetic req_to_token table read
  req_to_token_read_count > 0

Forbidden:
  production req_to_token_pool object read
  token_to_kv_pool read
  KV pool read
  K/V tensor read
  KV snapshot
  attention comparison execution
  attention override
  runtime writeback
  scheduler mutation
  KV cache mutation
  source mutation
```

もし source object が dict/list/tuple/int/str 以外の場合:

```text
blocked にする
attribute access しない
```

---

## 14. 次フェーズ案

推奨順:

```text
Phase 4.8.2:
  Req-to-token Synthetic Readonly Resolution Smoke
  - pure Python table
  - req_to_token_read_count > 0 allowed
  - no production object

Phase 4.8.3:
  Real Req-to-token Pool Readonly Adapter Design
  - actual SGLang object boundary
  - tensor/object read policy

Phase 4.8.4:
  Physical KV Index Readonly Resolution Design
  - token_to_kv_pool_read_count > 0 を設計

Phase 4.9:
  Isolated K/V Tensor Read Design
  - kv_pool_read_count > 0 / tensor_read_count > 0 を設計

Phase 4.10:
  Isolated Attention Comparison Smoke
  - attention_comparison_executed_count > 0
  - attention_override_true_count remains 0
```

---

## 15. 結論

Phase 4.8.2 では、実 SGLang pool object にはまだ触らず、synthetic table によって `req_to_token` resolution schema を固定するのが安全。

最小安全 chain:

```text
relaykv_kv_index_resolution_plan
→ relaykv_req_to_token_resolution_result
```

この段階で初めて許可する counter:

```text
req_to_token_read_count > 0
```

まだ禁止:

```text
token_to_kv_pool_read_count > 0
kv_pool_read_count > 0
tensor_read_count > 0
attention_comparison_executed_count > 0
attention_override_true_count > 0
```

---

## 16. commit command

この設計メモを repo に保存する場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/relaykv_phase4_8_2_req_to_token_readonly_resolution_design_2026-05-04.ja.md \
  notes/relaykv_phase4_8_2_req_to_token_readonly_resolution_design_2026-05-04.ja.md

git status --short
git diff --check

git add notes/relaykv_phase4_8_2_req_to_token_readonly_resolution_design_2026-05-04.ja.md
git commit -m "docs: design relaykv req-to-token readonly resolution"
git push mine relaykv-host-backup-shadow
```
