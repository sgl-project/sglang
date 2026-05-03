# RelayKV Phase 3.5 Host Backup Copy Safety Boundary Design

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

## 1. このメモの目的

このメモは、RelayKV Phase 3.5 として **host backup copy safety boundary** を固定するための設計メモである。

Phase 3 前半では、以下まで完了した。

```text
dry-run policy event
→ fake materialization result
→ guarded no-op materialization result
→ candidate-event materialization result
→ materialization summary
→ readonly report
→ metadata-only attention readiness
```

ここまででは、まだ実KV操作はしていない。

Phase 3.5 の目的は、実際の host backup copy helper に近づく前に、以下を分離して定義することである。

```text
host backup copy executed
KV pool read
KV snapshot
materialized candidate result
runtime connection
attention connection
```

特に、次の3つを混同しない。

```text
host_backup_copy_executed_count
kv_pool_read_count
kv_snapshot_count
```

---

## 2. 現在の安全境界

現時点でまだ実施していないこと:

```text
KV pool read
host backup copy実行
KV snapshot
attention接続
scheduler decision変更
runtime writeback
KV cache free
RadixTree変更
HiCache prefetch hint
```

Phase 3.5 でも、最初はまだ以下をしない。

```text
attention接続
scheduler decision変更
runtime writeback
KV cache free
RadixTree変更
HiCache prefetch hint
```

ただし、Phase 3.5 では、将来 `host_backup_copy_executed_count` が非ゼロになる段階を設計対象に含める。

---

## 3. 用語定義

### 3.1 candidate-event materialized

Phase 3.3 で実装済み。

意味:

```text
host backup candidate event payload を
metadata-only materialized candidate として扱う。
```

重要:

```text
host_backup_copy_executed=false
kv_pool_read=false
kv_snapshot=false
```

これは実copyではない。

---

### 3.2 guarded no-op

Phase 3.2 で実装済み。

意味:

```text
candidate event を見たが、
copyせずに明示的に no-op で止める。
```

状態:

```text
materialization_state="guarded_noop"
materialization_mode="noop_guarded"
materialized_block_ids=[]
skipped_block_ids=selected_block_ids
```

---

### 3.3 host backup copy materialized

Phase 3.5 以降で扱う予定。

意味:

```text
host backup copy helper が実行され、
copy結果を materialization_result として記録した状態。
```

ただし、この段階でも attention には接続しない。

想定状態:

```text
materialization_state="host_backup_copy_materialized"
materialization_mode="host_backup_copy"
host_backup_copy_executed=true
```

ここで重要なのは、`host_backup_copy_executed=true` になっても、即 attention 接続ではないこと。

---

### 3.4 KV pool read

意味:

```text
SGLang runtime の KV pool から直接 k/v tensor を読むこと。
```

Phase 3.5 初期では禁止。

`host backup copy` が既に CPU backup / host-side buffer から読む設計なら、KV pool read とは分けて扱う。

counter:

```text
kv_pool_read_count
```

---

### 3.5 KV snapshot

意味:

```text
KV pool または host backup から、検証用 snapshot を取ること。
```

snapshot は copy とは別。

counter:

```text
kv_snapshot_count
```

Phase 3.5 では、snapshot を取る場合でも attention へ渡さない。

---

## 4. counter の意味を固定する

### 4.1 host_backup_copy_executed_count

意味:

```text
host backup copy helper が実際に呼ばれ、
candidate KV copy または copy相当処理が実行された回数。
```

非ゼロになる条件:

```text
materialization_mode="host_backup_copy"
materialization_state="host_backup_copy_materialized"
host_backup_copy_executed=true
```

非ゼロにならない条件:

```text
fake_materialized
guarded_noop
candidate_event_materialized
summary-only report
readiness check
```

### 4.2 kv_pool_read_count

意味:

```text
runtime KV pool から直接 k/v を読んだ回数。
```

重要:

```text
host backup copy executed
```

と

```text
KV pool read
```

は同義ではない。

host backup copy が host backup storage から読むなら、`host_backup_copy_executed_count > 0` でも `kv_pool_read_count=0` はあり得る。

逆に、runtime KV pool から直接 materialize するような実装は、Phase 3.5 初期では禁止。

### 4.3 kv_snapshot_count

意味:

```text
検証・比較用に snapshot を作った回数。
```

copy result と snapshot は分ける。

```text
copy result:
  materialization_result として後段候補に使うもの

snapshot:
  smoke / compare / diagnostic 用の観測物
```

---

## 5. Host Backup Copy Input Schema

Phase 3.5 で扱う input は、candidate-event materialization と同じ情報を基礎にする。

最小 input:

```text
event_type="relaykv_host_backup_copy_request"
request_id
req_pool_idx
seq_len
layer_id
selected_block_ids
candidate_block_ids
anchor_block_ids
recent_block_ids
source
readiness_state
```

推奨 fields:

```text
policy_state
materialization_mode_requested
copy_source
copy_destination
copy_guard_state
copy_reason
```

候補値:

```text
materialization_mode_requested="host_backup_copy"
copy_source="host_backup_candidate"
copy_destination="materialization_result_only"
copy_guard_state="pre_attention_no_runtime_writeback"
```

この段階の destination は、まだ attention working KV ではない。

```text
禁止:
copy_destination="attention_working_kv"
copy_destination="runtime_kv_pool"
copy_destination="scheduler_mutation"
```

---

## 6. Host Backup Copy Output Schema

output は既存の materialization_result schema に合わせる。

```text
event_type="relaykv_materialization_result"
materialization_state="host_backup_copy_materialized"
materialization_mode="host_backup_copy"
request_id
req_pool_idx
seq_len
layer_id
selected_block_ids
materialized_block_ids
retrieved_block_ids
skipped_block_ids
fallback_block_ids
anchor_block_ids
recent_block_ids
candidate_block_ids
materialized_kv_count
materialized_token_count
source
blocking_reasons
warning_reasons
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

Phase 3.5 初期の理想は:

```text
host_backup_copy_executed=true
kv_pool_read=false
kv_snapshot=false
attention_override=false
runtime_writeback=false
scheduler_policy_noop=true
```

つまり、host backup copy は実行しても、runtime 側を変えない。

---

## 7. Safety Gate

host backup copy helper を実行する前に、以下を満たす必要がある。

```text
readonly report exists
materialization_summary exists
metadata-only attention readiness was assessed
overall_safety_status="pass"
ready_for_attention_connection_metadata_only=true
blocked_count=0
error_count=0
guarded_noop_count=0
candidate_event_materialized_count > 0 OR fake_materialized_count > 0
```

ただし、この gate は「attention接続OK」ではない。

意味:

```text
host backup copy の安全境界設計に進めるだけの
metadata-only materialization report が揃っている。
```

host backup copy 実行用には、別の readiness を追加する。

候補 helper:

```text
assess_relaykv_host_backup_copy_readiness_for_smoke(report)
```

ready state 候補:

```text
ready_for_host_backup_copy_boundary_smoke
```

---

## 8. Host Backup Copy Readiness Helper 案

追加候補:

```text
assess_relaykv_host_backup_copy_readiness_for_smoke(report)
```

目的:

```text
metadata-only readiness report から、
host backup copy boundary smoke へ進んでよいか判定する。
```

ready 条件:

```text
report_generated_from_readonly_inputs=true
overall_safety_status="pass"
policy_dry_run_included=true
policy_dry_run_total_events > 0
materialization_summary_included=true
materialization_total_results > 0
materialization_result_count > 0
materialized_kv_count > 0
candidate_event_materialized_count > 0
guarded_noop_count == 0
blocked_count == 0
error_count == 0
host_backup_copy_executed_count == 0
kv_pool_read_count == 0
kv_snapshot_count == 0
attention_override_true_count == 0
runtime_writeback_true_count == 0
scheduler_policy_noop_false_count == 0
```

理由:

```text
host backup copy boundary へ進む前は、
まだ host_backup_copy_executed_count は 0 でなければならない。
```

---

## 9. Host Backup Copy Boundary Smoke 案

Phase 3.5 の最初の smoke は、まだ実copyをしない。

目的:

```text
host backup copy request schema
host backup copy readiness
blocked / guarded / noop behavior
counter rules
```

追加候補:

```text
scripts/relaykv_host_backup_copy_boundary_smoke.py
```

最初の helper 候補:

```text
build_relaykv_host_backup_copy_requests_for_smoke(
    candidate_event_materialization_results,
    copy_readiness=None,
)
```

この helper は request を作るだけ。

まだ:

```text
host_backup_copy_executed=false
kv_pool_read=false
kv_snapshot=false
```

---

## 10. その次の helper 案

request schema が固まった後に、no-op boundary result を作る。

候補:

```text
build_relaykv_host_backup_copy_boundary_results_for_smoke(
    host_backup_copy_requests,
    execute_copy=False,
)
```

`execute_copy=False` の場合:

```text
materialization_state="blocked"
または
materialization_state="guarded_noop"
materialization_mode="host_backup_copy_boundary"
host_backup_copy_executed=false
```

`execute_copy=True` は Phase 3.6 以降まで禁止。

---

## 11. Phase 3.5 の禁止事項

Phase 3.5 では、まだ以下は禁止。

```text
actual host backup copy execution
KV pool read
KV snapshot
attention接続
scheduler decision変更
runtime writeback
KV cache free
RadixTree変更
HiCache prefetch hint
```

ただし、Phase 3.5 の設計メモでは、将来 `host_backup_copy_executed=true` になる場合の schema と counter 意味を固定する。

---

## 12. Failure Modes

### 12.1 metadata-only readiness missing

```text
materialization_summary_included=false
```

state:

```text
blocked
```

reason:

```text
materialization_summary_missing
```

### 12.2 no candidate-event materialization

```text
candidate_event_materialized_count == 0
```

state:

```text
blocked
```

reason:

```text
candidate_event_materialization_missing
```

### 12.3 guarded no-op remains

```text
guarded_noop_count > 0
```

state:

```text
blocked
```

reason:

```text
guarded_noop_present
```

### 12.4 previous host backup copy already executed

```text
host_backup_copy_executed_count > 0
```

state:

```text
blocked
```

reason:

```text
host_backup_copy_already_executed
```

### 12.5 KV pool read observed

```text
kv_pool_read_count > 0
```

state:

```text
blocked
```

reason:

```text
kv_pool_read_observed
```

### 12.6 snapshot observed

```text
kv_snapshot_count > 0
```

state:

```text
blocked
```

reason:

```text
kv_snapshot_observed
```

### 12.7 runtime mutation observed

```text
runtime_writeback_true_count > 0
scheduler_policy_noop_false_count > 0
attention_override_true_count > 0
```

state:

```text
blocked
```

reason:

```text
runtime_mutation_observed
```

---

## 13. 次の実装タスク候補

まずは実copyではなく、readiness helper と request schema smoke から始める。

Task:

```text
Add RelayKV host backup copy readiness and boundary request smoke.
```

Allowed files:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_host_backup_copy_boundary_smoke.py
```

Do not touch:

```text
forward_batch_info.py
model_runner.py
scheduler/managers
attention backend
memory_pool.py
flashinfer
.github/workflows
```

追加候補:

```text
assess_relaykv_host_backup_copy_readiness_for_smoke(report)

build_relaykv_host_backup_copy_requests_for_smoke(
    candidate_event_materialization_results,
    copy_readiness=None,
)
```

この段階の確認:

```text
copy readiness pass
copy readiness fail
request schema
blocked request
input non-mutation
poison unrelated field 未アクセス
all safety counters zero
host_backup_copy_executed_count=0
kv_pool_read_count=0
kv_snapshot_count=0
```

---

## 14. 判断

Phase 3.5 では、まだ実copyを始めない。

次の正しい一歩は:

```text
host backup copy readiness
+
host backup copy request schema
+
boundary smoke
```

である。

実KVへ近づく順序は以下。

```text
metadata-only materialization result
→ host backup copy readiness
→ host backup copy request schema
→ guarded boundary result
→ actual host backup copy helper boundary
→ attention connection design
```

ここで焦って actual copy に進むと、`host_backup_copy_executed_count` / `kv_pool_read_count` / `kv_snapshot_count` の意味が曖昧になり、後で safety report が壊れる。

したがって、Phase 3.5 の最初は、copy ではなく **copy boundary の設計と request 化** に留める。
