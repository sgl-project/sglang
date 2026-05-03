# RelayKV Phase 4 Attention Connection Safety Boundary Design

## 日付確認

- Design date: **2026-05-03**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. この設計メモの目的

このメモは、RelayKV/SGLang integration の次段階である **Phase 4: Attention Connection** に入る前の安全境界を定義する。

Phase 3.6 までで、RelayKV は以下まで到達した。

```text
runtime observation
→ host backup candidate summary
→ runtime/candidate join
→ dry-run policy
→ metadata-only materialization
→ host backup copy boundary
→ isolated actual host backup copy smoke
→ actual-copy report
→ attention connection readiness: design-only
```

現在の ready state:

```text
ready_for_attention_connection_design_only
```

これは **attention connection の設計に進める** という意味であり、まだ実際に attention backend を置き換える、attention override を有効化する、runtime KV を読む、scheduler を変える、という意味ではない。

---

## 2. Phase 4 の最終目的

Phase 4 の最終目的は、RelayKV が選んだ working KV set を attention 計算に渡すための境界を作ることである。

最終的には次の形を目指す。

```text
RECENT / ANCHOR / RETRIEVED working KV
→ attention input handoff
→ attention comparison
→ optional attention override
```

ただし、Phase 4 初期では、実 attention override は禁止する。

まずは次を作る。

```text
attention connection safety boundary
attention input schema / working KV handoff schema
attention dry-run result
no-op attention override smoke
isolated attention comparison smoke
```

---

## 3. Phase 4 の基本方針

Phase 4 では、以下を順番に進める。

```text
Phase 4.0: attention connection safety boundary design
Phase 4.1: attention input / working KV handoff schema
Phase 4.2: attention connection dry-run result helper
Phase 4.3: no-op attention override smoke
Phase 4.4: isolated attention comparison smoke
Phase 4.5: real attention path connection design
```

重要:

```text
Phase 4.1〜4.4 では、実 attention backend には接続しない。
```

---

## 4. Phase 4 初期で禁止すること

Phase 4.0〜4.4 では、以下を禁止する。

```text
actual attention override
attention backend modification
FlashInfer / FlashAttention path modification
KV pool read
GPU tensor read
KV snapshot
runtime writeback
scheduler decision mutation
KV cache free
RadixTree modification
HiCache prefetch hint
production runtime connection
```

この段階では、attention に渡すための metadata / handoff schema / dry-run result のみを扱う。

---

## 5. Phase 4 初期で許可すること

Phase 4.1〜4.4 で許可するもの:

```text
attention_connection_readiness assessment
working KV handoff metadata result
attention_input_candidate metadata result
attention_connection_dry_run result
attention_override_noop result
attention_comparison_plan metadata result
```

許可される counters:

```text
attention_connection_attempted_count > 0
attention_override_noop_count > 0
```

ただし、次はまだ 0 のまま。

```text
attention_override_true_count=0
kv_pool_read_count=0
kv_snapshot_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
kv_cache_mutation_true_count=0
source_mutated_true_count=0
```

---

## 6. Attention handoff schema

Phase 4.1 で作るべき schema は、実 tensor ではなく metadata-only の handoff である。

候補 event:

```text
event_type="relaykv_attention_handoff_candidate"
handoff_state="handoff_ready"
handoff_mode="metadata_only"
source="actual_host_backup_copy_report_to_attention_handoff_candidate"
```

主要 fields:

```text
request_id
req_pool_idx
seq_len
layer_id

kv_classes_present
recent_block_ids
anchor_block_ids
retrieved_block_ids
candidate_block_ids
materialized_block_ids

working_kv_block_ids
working_kv_block_count
working_kv_token_count

attention_target_layer_id
attention_target_backend="unconnected"
attention_override_allowed=false
```

safety flags:

```text
attention_connection_attempted=false
attention_override=false
kv_pool_read=false
kv_snapshot=false
runtime_writeback=false
scheduler_policy_noop=true
kv_cache_mutation=false
source_mutated=false
```

---

## 7. Attention dry-run result schema

Phase 4.2 で作るべき result:

```text
event_type="relaykv_attention_connection_dry_run_result"
attention_connection_state="dry_run"
attention_connection_mode="metadata_only"
```

主要 fields:

```text
request_id
req_pool_idx
seq_len
layer_id
working_kv_block_ids
working_kv_block_count
working_kv_token_count
attention_target_layer_id
attention_target_backend="unconnected"
```

safety flags:

```text
attention_connection_attempted=true
attention_override=false
attention_override_noop=true
kv_pool_read=false
kv_snapshot=false
runtime_writeback=false
scheduler_policy_noop=true
kv_cache_mutation=false
source_mutated=false
```

この result は「接続できる形の metadata がある」ことを示すだけで、実 attention は実行しない。

---

## 8. No-op attention override schema

Phase 4.3 で作るべき no-op override:

```text
event_type="relaykv_attention_override_noop_result"
attention_connection_state="override_noop"
attention_connection_mode="noop_guarded"
```

意味:

```text
override point に渡せる形式を作る
しかし実際には override しない
```

safety flags:

```text
attention_connection_attempted=true
attention_override=false
attention_override_noop=true
kv_pool_read=false
kv_snapshot=false
runtime_writeback=false
scheduler_policy_noop=true
kv_cache_mutation=false
source_mutated=false
```

blocked reasons:

```text
attention_override_not_allowed_in_phase4_noop
```

---

## 9. Isolated attention comparison smoke の位置づけ

Phase 4.4 では、可能であれば isolated attention comparison smoke に進む。

ただし、これは production attention backend に接続するのではなく、既存の test/smoke helper の中で以下を比較する段階とする。

```text
full metadata plan
vs
RelayKV working KV plan
```

この時点でも、runtime KV pool は読まない。

必要なら fake tensor / synthetic scores / metadata count comparison のみを使う。

Phase 4.4 で許可するかどうかは別 design memo で判断する。

---

## 10. Readiness helper 案

Phase 4.1 の最初に追加する候補:

```text
assess_relaykv_attention_handoff_readiness_for_smoke(report)
```

入力:

```text
relaykv_actual_host_backup_copy_report
```

ready 条件:

```text
report_type="relaykv_actual_host_backup_copy_report"
actual_copy_safety_status="pass"
actual_host_backup_copy_summary_included=true
actual_host_backup_copy_materialized_count > 0
actual_host_backup_copy_executed_count > 0
actual_host_backup_copy_kv_pool_read_count == 0
actual_host_backup_copy_kv_snapshot_count == 0
attention_override_true_count == 0
runtime_writeback_true_count == 0
scheduler_policy_noop_false_count == 0
```

ready state:

```text
ready_for_attention_handoff_metadata_only
```

blocked states:

```text
blocked_not_actual_copy_report
blocked_actual_copy_safety_not_pass
blocked_actual_copy_summary_missing
blocked_no_actual_copy_materialized
blocked_actual_copy_not_executed
blocked_kv_pool_read_observed
blocked_kv_snapshot_observed
blocked_attention_override_observed
blocked_runtime_writeback_observed
blocked_scheduler_mutation_observed
blocked_multiple_reasons
```

---

## 11. Handoff helper 案

Phase 4.1 helper:

```text
build_relaykv_attention_handoff_candidates_for_smoke(
    actual_copy_results,
    attention_handoff_readiness=None,
)
```

入力:

```text
host_backup_copy_materialized results
```

出力:

```text
relaykv_attention_handoff_candidate
```

normal behavior:

```text
materialization_state="host_backup_copy_materialized"
copy_state="copy_executed"
→ handoff_state="handoff_ready"
```

blocked behavior:

```text
readiness missing
readiness blocked
wrong event_type
wrong materialization_state
copy not executed
empty materialized_block_ids
```

normal safety flags:

```text
attention_connection_attempted=false
attention_override=false
kv_pool_read=false
kv_snapshot=false
runtime_writeback=false
scheduler_policy_noop=true
kv_cache_mutation=false
source_mutated=false
```

---

## 12. Summary 案

Phase 4.1 summary:

```text
summarize_relaykv_attention_handoff_candidates_for_smoke(candidates)
```

fields:

```text
summary_type="relaykv_attention_handoff_candidate_summary"
total_handoff_candidates
handoff_ready_count
blocked_count
working_kv_block_count
working_kv_token_count
per_request_counts
per_layer_counts
per_handoff_state_counts

attention_connection_attempted_count
attention_override_true_count
attention_override_noop_count
kv_pool_read_count
kv_snapshot_count
runtime_writeback_true_count
scheduler_policy_noop_false_count
kv_cache_mutation_true_count
source_mutated_true_count
```

expected Phase 4.1 pass:

```text
handoff_ready_count > 0
attention_connection_attempted_count=0
attention_override_true_count=0
kv_pool_read_count=0
kv_snapshot_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
```

---

## 13. Smoke 案

最初の smoke:

```text
scripts/relaykv_attention_handoff_smoke.py
```

pass flow:

```text
actual-copy report pass
→ attention handoff readiness pass
→ 2 host_backup_copy_materialized results
→ 2 attention_handoff_candidate results
→ summary pass
```

expected:

```text
handoff_ready_count=2
working_kv_block_count > 0
attention_connection_attempted_count=0
attention_override_true_count=0
kv_pool_read_count=0
kv_snapshot_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
```

blocked cases:

```text
readiness missing
readiness blocked
wrong event_type
wrong materialization_state
copy not executed
empty materialized_block_ids
poison unrelated field
input non-mutation
```

---

## 14. 初期 allowed files

Phase 4.1 の初期実装では、allowed files を最小化する。

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_attention_handoff_smoke.py
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

## 15. 判断

Phase 4 の最初の正しい一歩は、attention backend に接続することではない。

正しくは:

```text
attention handoff metadata schema
```

を作ることである。

Phase 4.1 では、以下を確認する。

```text
actual copy result から attention に渡す候補 metadata を作れる
ただし attention にはまだ渡さない
```

この段階で許可するのは:

```text
handoff_ready_count > 0
```

まだ許可しないもの:

```text
attention_override_true_count > 0
kv_pool_read_count > 0
kv_snapshot_count > 0
runtime_writeback_true_count > 0
scheduler_policy_noop_false_count > 0
```

次の作業名:

```text
RelayKV Phase 4.1 Attention Handoff Metadata Smoke
```
