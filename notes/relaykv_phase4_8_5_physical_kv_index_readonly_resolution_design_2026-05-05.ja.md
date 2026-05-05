# RelayKV Phase 4.8.5 Physical KV Index Readonly Resolution Design

## 日付確認

- Design date: **2026-05-05**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. 目的

このメモは **Phase 4.8.5: Physical KV Index Readonly Resolution Design** を定義する。

直前の Phase 4.8.5-pre では、RelayKV Core-ish field と SGLang Adapter field の分離を入れた。

```text
RelayKV Core-ish:
  logical_block_id
  token_span
  layer_id
  kv_head_group
  kv_class
  decision_state
  fallback_reason
  position_check_state

SGLang Adapter:
  req_pool_idx
  req_to_token_* metadata
  pool_source_path
  engine_block_ref
  token_to_kv_pool_index=None
```

Phase 4.8.5 の目的は、次の境界を設計すること。

```text
req_to_token entry
→ token_to_kv_pool
→ physical KV index metadata
```

ただし、この段階では **K/V tensor は読まない**。

---

## 2. 現在の到達点

Phase 4.8.4.3 までに、実 SGLang server 上で以下を確認した。

```text
ModelRunner.forward
→ SGLANG_RELAYKV_REQ_TO_TOKEN_INSPECTION=1
→ model_runner.req_to_token_pool
→ req_to_token metadata-only summary
```

実 server で確認済み:

```text
env off:
  HTTP 200
  importなし
  hookなし
  logなし

env on:
  HTTP 200
  metadata-only summary emitted
  pool_source_path=model_runner.req_to_token_pool
  production_output_unaffected=true
```

Phase 4.8.5-pre では、schema alignment を追加した。

```text
engine_name="sglang"
adapter_name="sglang"
adapter_metadata
engine_block_ref
token_to_kv_pool_index=None
position_check_state="not_checked_metadata_only"
attention_mask_mode="unknown"
rope_position_consistency="not_checked"
```

---

## 3. Phase 4.8.5 で初めて扱う対象

Phase 4.8.5 では `token_to_kv_pool` を扱い始める。

ただし扱うのは **index metadata** だけ。

```text
req_to_token entry:
  request-local token position から得た logical/cache-side token id

token_to_kv_pool:
  token id から physical KV pool index へ変換する SGLang adapter-side mapping

physical_kv_index:
  SGLang KV pool 内の物理 index metadata
```

ここで得たいもの:

```text
engine_block_ref.token_to_kv_pool_index
engine_block_ref.physical_kv_index
engine_block_ref.physical_kv_index_preview
engine_block_ref.physical_kv_index_count
```

---

## 4. まだ禁止すること

Phase 4.8.5 でも以下は禁止。

```text
K/V tensor read
KV pool read
KV pool snapshot
k_buffer / v_buffer read
host backup copy
attention execution
attention override
scheduler mutation
runtime writeback
KV cache mutation
source mutation
```

禁止 API / 操作:

```text
.cpu()
.tolist()
.item()
.numpy()
direct tensor value read
large full-table dump
unbounded indexing
recursive object traversal
dir()
vars()
repr() of arbitrary runtime objects
```

---

## 5. 許可すること

Phase 4.8.5 で許可するのは、bounded readonly index resolution だけ。

許可:

```text
synthetic token_to_kv_pool table read
fake-object token_to_kv_pool attr read
bounded list/tuple/dict lookup
small preview/count/checksum
metadata-only summary
```

実 runtime object については、最初は metadata inspection までに留める。

```text
type
module
qualname
shape
device
dtype
```

---

## 6. Core / Adapter 境界

RelayKV Core は `physical_kv_index` を直接所有しない。

Core で扱う:

```text
logical_block_id
token_span
layer_id
kv_head_group
kv_class
decision_state
fallback_reason
working_kv_budget
```

SGLang Adapter で扱う:

```text
req_pool_idx
req_to_token entries
token_to_kv_pool mapping
physical_kv_index
cache_position
pool_source_path
engine_block_ref
```

Phase 4.8.5 の payload では、physical index は必ず `engine_block_ref` に入れる。

```text
engine_block_ref:
  req_pool_idx
  cache_position
  token_to_kv_pool_index
  physical_kv_index
  physical_kv_index_preview
  physical_kv_index_count
```

---

## 7. 推奨 helper 設計

### 7.1 Synthetic resolution helper

最初に追加する helper 候補:

```python
build_relaykv_physical_kv_index_resolution_results_for_smoke(
    req_to_token_resolution_results,
    token_to_kv_pool_table=None,
    read_token_to_kv_pool=False,
    max_tokens_per_request=256,
    max_total_tokens=512,
    max_preview_entries=8,
)
```

入力:

```text
relaykv_req_to_token_resolution_result
```

出力:

```text
relaykv_physical_kv_index_resolution_result
```

通常 output:

```text
event_type="relaykv_physical_kv_index_resolution_result"
resolution_state="physical_kv_index_resolved"
resolution_mode="synthetic_token_to_kv_pool_readonly"
engine_name="sglang"
adapter_name="sglang"
engine_block_ref.token_to_kv_pool_index=<bounded preview/list only>
engine_block_ref.physical_kv_index_preview=[...]
engine_block_ref.physical_kv_index_count=N
token_to_kv_pool_read=true
token_to_kv_pool_read_count=N
kv_pool_read=false
kv_snapshot=false
tensor_read=false
attention_comparison_executed=false
attention_override=false
runtime_writeback=false
scheduler_policy_noop=true
kv_cache_mutation=false
source_mutated=false
```

Blocked output:

```text
resolution_state="blocked"
blocked_reason=<reason>
token_to_kv_pool_read=false
token_to_kv_pool_read_count=0
```

### 7.2 Summary helper

```python
summarize_relaykv_physical_kv_index_resolution_results_for_smoke(results)
```

Summary fields:

```text
summary_type="relaykv_physical_kv_index_resolution_summary"
total_results
resolved_count
blocked_count
error_count
resolved_token_count
physical_kv_index_count
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
per_request_counts
per_layer_counts
blocked_reason_counts
```

---

## 8. 入力 table の扱い

Phase 4.8.5.1 では synthetic table のみを使う。

許可 backing:

```text
dict
list
tuple
```

lookup key:

```text
req_to_token_entry
str(req_to_token_entry)
```

または、より SGLang らしく:

```text
token_id -> physical_kv_index
```

禁止:

```text
torch.Tensor
numpy array
cuda-like object
custom object with __getitem__ side effects
unbounded full-table conversion
```

---

## 9. token_to_kv_pool read の counter 方針

Phase 4.8.5 では、初めて以下が nonzero になることを許可する。

```text
token_to_kv_pool_read_count > 0
```

ただし、これは synthetic / fake-object / bounded readonly index read に限る。

引き続き zero のまま:

```text
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

## 10. position / RoPE / attention mask 予約 field

Phase 4.8.5 の output には、まだ実チェックしなくても以下を維持する。

```text
position_check_state="not_checked_metadata_only"
attention_mask_mode="unknown"
rope_position_consistency="not_checked"
is_position_preserving=None
is_contiguous=None
working_token_order=None
```

理由:

```text
physical KV index 解決後、working KV assembly dry-run で token order / position 整合性を検査するため。
```

---

## 11. blocked reason 候補

```text
not_req_to_token_resolution_result
req_to_token_resolution_not_resolved
read_token_to_kv_pool_not_enabled
token_to_kv_pool_table_missing
token_to_kv_pool_table_not_indexable
req_to_token_entries_missing
req_to_token_entry_not_int
token_to_kv_pool_entry_missing
token_to_kv_pool_entry_not_int
token_to_kv_pool_entry_out_of_range
max_tokens_per_request_exceeded
max_total_tokens_exceeded
kv_pool_read_not_allowed
kv_snapshot_not_allowed
tensor_read_not_allowed
attention_override_true_not_allowed
attention_comparison_executed_not_allowed
runtime_writeback_not_allowed
scheduler_mutation_not_allowed
```

---

## 12. smoke 方針

最初の実装 smoke:

```text
scripts/relaykv_physical_kv_index_resolution_smoke.py
```

確認すること:

```text
1. synthetic pass flow
2. dict key / str key lookup
3. bounded preview/count/checksum
4. missing token_to_kv_pool table blocked
5. read_token_to_kv_pool=False blocked
6. non-int entry blocked
7. out-of-range blocked
8. poisoned backing blocked or untouched
9. input non-mutation
10. Core/Adapter schema fields preserved
```

Pass flow expected:

```text
resolved_count > 0
resolved_token_count > 0
token_to_kv_pool_read_count > 0
kv_pool_read_count=0
kv_snapshot_count=0
tensor_read_count=0
attention_override_true_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
source_mutated_true_count=0
```

---

## 13. 段階分け

### Phase 4.8.5.1

```text
Synthetic token_to_kv_pool readonly resolution smoke
```

Allowed files:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_physical_kv_index_resolution_smoke.py
```

### Phase 4.8.5.2

```text
Fake actual token_to_kv_pool adapter payload smoke
```

Allowed files:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_token_to_kv_pool_readonly_adapter_smoke.py
```

### Phase 4.8.5.3

```text
Runtime token_to_kv_pool metadata inspection design
```

Observation only:

```text
type / shape / device / dtype
```

No value read.

### Phase 4.8.5.4

```text
Optional server token_to_kv_pool metadata inspection smoke
```

No index read unless separately enabled.

---

## 14. Phase 4.8.5 の完了条件

Phase 4.8.5 は以下で完了とする。

```text
synthetic token_to_kv_pool index resolution pass
schema alignment maintained
physical_kv_index metadata appears under engine_block_ref
token_to_kv_pool_read_count allowed only in bounded synthetic/fake smoke
KV pool/tensor/attention/runtime/scheduler counters remain zero
```

まだ完了条件に含めないもの:

```text
K/V tensor read
working KV tensor assembly
shadow attention compute
attention backend connection
quality benchmark
```

---

## 15. 次の設計上の注意

Phase 4.8.5 の次は、working KV assembly dry-run に近づく。

その前に必要になる可能性が高いもの:

```text
KV tensor read boundary design
position preserving working token order
layer_id / kv_head_group shape consistency
working KV assembly report
fallback as normal decision state
```

---

## 16. Codex CLI 向け次ステップ

```text
Phase 4.8.5.1: Synthetic token_to_kv_pool readonly resolution smoke

Implement only metrics helper + smoke.
Do not touch runtime/server/model_runner.
Allow token_to_kv_pool_read_count > 0 only for bounded synthetic table.
Keep KV pool/tensor/attention/runtime/scheduler counters zero.
Output physical index metadata under engine_block_ref.
```

---

## 17. commit command

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/relaykv_phase4_8_5_physical_kv_index_readonly_resolution_design_2026-05-05.ja.md \
  notes/relaykv_phase4_8_5_physical_kv_index_readonly_resolution_design_2026-05-05.ja.md

git status --short
git diff --check

git add notes/relaykv_phase4_8_5_physical_kv_index_readonly_resolution_design_2026-05-05.ja.md
git commit -m "docs: design relaykv physical kv index resolution"
git push mine relaykv-host-backup-shadow
```
