# RelayKV Phase 3.6 Actual Host Backup Copy Smoke Design

## 日付確認

この設計メモは **JST 基準で 2026-05-03** の作業記録・設計整理として作成する。

対象 repo:

```text
~/work/sglang-relaykv
```

対象 branch:

```text
relaykv-host-backup-shadow
```

---

## 1. 目的

Phase 3.6 では、初めて `host_backup_copy_executed_count` が 1 以上になり得る smoke を扱う。

ただし、まだ runtime / attention には接続しない。  
copy は **isolated helper / smoke 内に閉じる**。

Phase 3.5 までの到達点:

```text
host backup copy safety boundary design
→ host backup copy readiness
→ host backup copy request schema
→ host backup copy boundary result
→ readonly report integration
→ actual host backup copy readiness
```

現在の ready state:

```text
ready_for_actual_host_backup_copy_smoke_boundary_complete
```

これは「actual host backup copy smoke の設計・isolated helper 実装へ進める」という意味であり、attention や runtime に接続してよいという意味ではない。

---

## 2. Phase 3.6 の安全境界

Phase 3.6 初期で許可するもの:

```text
host_backup_copy_executed=true
host_backup_copy_executed_count > 0
```

Phase 3.6 初期でも禁止するもの:

```text
torch import
KV pool read
GPU tensor read
KV snapshot
attention接続
scheduler decision変更
runtime writeback
KV cache free
RadixTree変更
HiCache prefetch hint
production runtime connection
```

引き続き 0 であるべき counters:

```text
kv_pool_read_count
kv_snapshot_count
attention_override_true_count
runtime_writeback_true_count
scheduler_policy_noop_false_count
kv_cache_mutation_true_count
source_mutated_true_count
```

つまり、Phase 3.6 の pass 状態は次。

```text
copy は実行された
しかし runtime KV pool は読んでいない
snapshot も取っていない
attention には接続していない
scheduler/runtime は変えていない
```

---

## 3. 入力 schema

Phase 3.6 の入力は、Phase 3.5 で作った copy request schema を使う。

```text
event_type="relaykv_host_backup_copy_request"
copy_state="request_ready"
copy_mode="host_backup_copy_boundary"
copy_source="host_backup_candidate"
copy_destination="materialization_result_only"
copy_guard_state="pre_attention_no_runtime_writeback"
```

主要 fields:

```text
request_id
req_pool_idx
seq_len
layer_id
selected_block_ids
materialized_block_ids
retrieved_block_ids
candidate_block_ids
anchor_block_ids
recent_block_ids
materialized_kv_count
materialized_token_count
```

Phase 3.6 初期では tensor は渡さない。  
copy 対象は block id / metadata payload のみとする。

---

## 4. 出力 schema

actual copy smoke の output は、既存の materialization result schema に合わせる。

```text
event_type="relaykv_materialization_result"
materialization_state="host_backup_copy_materialized"
materialization_mode="host_backup_copy"
copy_state="copy_executed"
copy_mode="host_backup_copy_isolated_smoke"
```

主要 fields:

```text
request_id
req_pool_idx
seq_len
layer_id
selected_block_ids
materialized_block_ids
retrieved_block_ids
candidate_block_ids
anchor_block_ids
recent_block_ids
materialized_kv_count
materialized_token_count
source="host_backup_copy_request_to_isolated_materialization_result"
blocking_reasons=[]
warning_reasons=["isolated_smoke_no_runtime_connection"]
```

safety flags:

```text
source_mutated=false
attention_override=false
kv_cache_mutation=false
runtime_writeback=false
scheduler_policy_noop=true
host_backup_copy_executed=true
kv_pool_read=false
kv_snapshot=false
```

---

## 5. Helper 案

追加候補:

```text
build_relaykv_actual_host_backup_copy_results_for_smoke(
    host_backup_copy_requests,
    actual_copy_readiness=None,
    execute_copy=True,
)
```

Phase 3.6 では `execute_copy=True` を許可する。  
ただし、ここでの actual copy は runtime tensor copy ではなく、isolated smoke copy として扱う。

実装上の意味:

```text
copy request payload を host_backup_copy_materialized result へ昇格し、
host_backup_copy_executed=true を立てる isolated smoke copy
```

readiness が false の場合:

```text
copy_state="blocked"
blocking_reasons includes actual_copy_readiness_not_met
host_backup_copy_executed=false
```

`execute_copy=false` の場合:

```text
copy_state="blocked"
blocking_reasons includes execute_copy_required_for_actual_copy_smoke
host_backup_copy_executed=false
```

---

## 6. Summary 案

追加候補:

```text
summarize_relaykv_actual_host_backup_copy_results_for_smoke(results)
```

summary fields:

```text
summary_type="relaykv_actual_host_backup_copy_result_summary"
total_copy_results
host_backup_copy_materialized_count
blocked_count
error_count
materialized_kv_count
materialized_token_count
per_request_counts
per_layer_counts
per_copy_state_counts
source_mutated_true_count
attention_override_true_count
kv_cache_mutation_true_count
runtime_writeback_true_count
scheduler_policy_noop_false_count
host_backup_copy_executed_count
kv_pool_read_count
kv_snapshot_count
```

expected pass:

```text
host_backup_copy_materialized_count > 0
host_backup_copy_executed_count == total_copy_results
kv_pool_read_count == 0
kv_snapshot_count == 0
attention_override_true_count == 0
runtime_writeback_true_count == 0
scheduler_policy_noop_false_count == 0
```

---

## 7. Report integration 方針

Phase 3.6 では、既存の metadata-only report helper をそのまま複雑化しすぎない方がよい。

推奨:

```text
build_relaykv_actual_host_backup_copy_report_for_smoke(...)
```

理由:

```text
Phase 3.5 metadata-only report では host_backup_copy_executed_count > 0 を fail にする。
Phase 3.6 actual-copy report では host_backup_copy_executed_count > 0 を許可する。
安全ルールが違うため、report helper を分けた方がよい。
```

actual copy report で許可するもの:

```text
host_backup_copy_executed_count > 0
```

actual copy report でも fail にするもの:

```text
kv_pool_read_count > 0
kv_snapshot_count > 0
attention_override_true_count > 0
runtime_writeback_true_count > 0
scheduler_policy_noop_false_count > 0
kv_cache_mutation_true_count > 0
source_mutated_true_count > 0
```

---

## 8. Smoke 案

追加候補:

```text
scripts/relaykv_actual_host_backup_copy_smoke.py
```

pass flow:

```text
fake readonly report
→ actual copy readiness pass
→ 2 host backup copy requests
→ actual host backup copy results
→ actual copy summary
```

expected:

```text
host_backup_copy_materialized_count=2
host_backup_copy_executed_count=2
kv_pool_read_count=0
kv_snapshot_count=0
attention_override_true_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
materialized_kv_count=4
```

fail / blocked cases:

```text
readiness missing
readiness blocked
request not ready
wrong event_type
empty materialized_block_ids
execute_copy=false
poison unrelated field
input non-mutation
```

---

## 9. 初期実装での allowed files

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_actual_host_backup_copy_smoke.py
```

禁止領域:

```text
python/sglang/srt/model_executor/forward_batch_info.py
python/sglang/srt/model_executor/model_runner.py
python/sglang/srt/managers/
scheduler.py
schedule_batch.py
attention backend
memory_pool.py
flashinfer
.github/workflows
```

---

## 10. 判断

Phase 3.6 の最初の正しい一歩は、runtime に接続した copy ではない。

正しくは:

```text
isolated actual host backup copy smoke
```

である。

ここで初めて次を許可する。

```text
host_backup_copy_executed_count > 0
```

しかし、次は引き続き 0 のまま維持する。

```text
kv_pool_read_count
kv_snapshot_count
attention_override_true_count
runtime_writeback_true_count
scheduler_policy_noop_false_count
```

これにより、RelayKV は次の順序で安全に進められる。

```text
metadata-only boundary
→ request schema
→ boundary result
→ actual copy smoke
→ actual copy report
→ attention connection design
```

まだ attention 接続には進まない。
