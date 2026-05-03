# RelayKV Phase 3 Safe Materialization Design Memo

## 日付確認

この設計メモは JST 基準で `2026-05-03` の作業記録・設計整理として作成する。

Branch: `relaykv-host-backup-shadow`  
Repo: `~/work/sglang-relaykv`

---

## 1. このメモの目的

このメモは、RelayKV Phase 3 の **safe materialization** に入る前に、実装境界・schema・状態遷移・failure mode を固定するための設計メモである。

Phase 2 では、以下まで完了した。

```text
metadata
→ join
→ readonly report
→ dry-run policy log
→ dry-run diagnostic flow
→ materialization readiness assessment
```

Phase 3 では、dry-run で選ばれた candidate block を、将来 working KV に載せるための **materialized result** として扱える形にする。

ただし、Phase 3 の最初ではまだ以下をしない。

```text
attention接続
scheduler decision変更
runtime writeback
KV cache free
RadixTree変更
HiCache prefetch hint
```

Phase 3 の目的は、attention へ接続する前に、KV materialization を安全に分離・検証できるようにすることである。

---

## 2. Phase 3 の基本方針

Phase 3 は次の順で進める。

```text
3.0 materialization design memo
3.1 materialization input/output schema 固定
3.2 fake materialization result helper
3.3 no-op guarded materialization smoke
3.4 host backup copy candidate との対応確認
3.5 actual host backup copy helper の安全境界確認
3.6 runtime接続前 readiness gate
```

重要なのは、Phase 3 内でも段階を分けること。

```text
fake result
→ no-op guarded result
→ host backup copy candidate result
→ actual copy helper boundary
→ runtime connection はさらに後
```

---

## 3. Phase 3 で「materialization」と呼ぶもの

RelayKVにおける materialization とは、dry-run policy で選ばれた logical block を、将来 attention working set に渡せる候補として具体化する処理である。

ただし、ここでの materialization は段階的に定義する。

### 3.1 Fake materialization

実KVを読まず、schemaだけを生成する。

```text
selected_block_ids
anchor_block_ids
recent_block_ids
candidate_block_ids

から materialized_result dict を作る
```

用途:

```text
schema確認
summary確認
readiness確認
failure mode確認
input/output non-mutation確認
```

### 3.2 No-op guarded materialization

実KVに触らず、candidate event の payload を materialization result に変換する。

```text
host_backup_candidate_event
→ materialized_candidate_result
```

この段階でも copy はしない。

用途:

```text
candidate event と materialized result の対応確認
applied / fallback / skipped の状態確認
safety flag確認
```

### 3.3 Actual host backup copy materialization

host backup copy から得られる candidate KV を materialized result として扱う。

この段階で初めて実KVに近づく。

ただし、最初は以下を禁止する。

```text
attentionへ渡さない
schedulerを書き換えない
runtime writebackしない
KV cacheをfreeしない
```

### 3.4 Attention materialization

Phase 4 以降。

```text
RECENT + ANCHOR + RETRIEVED
を working KV として attention backend へ渡す
```

Phase 3 ではまだ扱わない。

---

## 4. Phase 3 の安全境界

Phase 3 初期で禁止すること:

```text
attention接続
scheduler decision変更
runtime writeback
KV cache free
RadixTree変更
HiCache prefetch hint
```

Phase 3 初期で慎重に扱うこと:

```text
KV pool read
host backup copy実行
KV snapshot
```

Phase 3.1〜3.3 では原則として以下に留める。

```text
pure helper
dict/list/tuple input
fake payload
no-op payload
summary/report
smoke
```

実KVを読む段階に入る場合は、別途 readiness gate を通す。

---

## 5. Materialization Input Schema

Phase 3 の materialization input は、Phase 2 の dry-run policy event / readonly report を基準にする。

### 5.1 最小 input

```text
event_type="relaykv_materialization_request"
request_id
req_pool_idx
seq_len
layer_id
policy_state
selected_block_ids
anchor_block_ids
recent_block_ids
candidate_block_ids
retrieval_budget_tokens
kv_budget_tokens
layer_budget_policy
source
```

### 5.2 KV class fields

```text
kv_classes_requested:
  RECENT
  ANCHOR
  RETRIEVED
  COLD_CANDIDATE
```

分類:

```text
recent_block_ids:
  RECENT

anchor_block_ids:
  ANCHOR

selected_block_ids:
  RETRIEVED candidate

candidate_block_ids:
  COLD_CANDIDATE
```

### 5.3 Safety fields

```text
source_mutated=false
attention_override=false
kv_cache_mutation=false
runtime_writeback=false
scheduler_policy_noop=true
```

### 5.4 Required relation to dry-run event

materialization request は dry-run event から作る。

対応関係:

```text
dry-run event:
  relaykv_policy_dry_run

materialization request:
  relaykv_materialization_request
```

必須継承:

```text
request_id
req_pool_idx
seq_len
layer_id
selected_block_ids
anchor_block_ids
recent_block_ids
candidate_block_ids
retrieval_budget_tokens
layer_budget_policy
```

---

## 6. Materialization Output Schema

materialization output は、実際に materialize したかどうかに関わらず、同じ schema で返す。

### 6.1 最小 output

```text
event_type="relaykv_materialization_result"
request_id
req_pool_idx
seq_len
layer_id
materialization_state
materialization_mode
selected_block_ids
materialized_block_ids
skipped_block_ids
fallback_block_ids
anchor_block_ids
recent_block_ids
retrieved_block_ids
candidate_block_ids
materialized_kv_count
materialized_token_count
source
```

### 6.2 materialization_state

候補:

```text
noop
guarded_noop
fake_materialized
candidate_event_materialized
host_backup_copy_materialized
skipped
fallback
blocked
error
```

意味:

```text
noop:
  何もしない。schema確認だけ。

guarded_noop:
  safety guard により実copyせず、no-op result を返した。

fake_materialized:
  fake block metadata から materialized result を作った。

candidate_event_materialized:
  host backup candidate event payload から result を作った。
  まだ実copyとは限らない。

host_backup_copy_materialized:
  host backup copy 由来の実KV candidateを result として扱った。

skipped:
  条件不足によりskip。

fallback:
  materializationできず fallback path を使った。

blocked:
  readiness gate / safety check により停止。

error:
  例外または不整合。
```

### 6.3 materialization_mode

候補:

```text
fake
noop_guarded
candidate_event
host_backup_copy
```

Phase 3 初期では以下だけを使う。

```text
fake
noop_guarded
candidate_event
```

`host_backup_copy` は safety boundary が固まってから。

### 6.4 Safety counters

```text
source_mutated_true_count
attention_override_true_count
kv_cache_mutation_true_count
runtime_writeback_true_count
scheduler_policy_noop_false_count
host_backup_copy_executed_count
kv_pool_read_count
kv_snapshot_count
```

Phase 3 初期の期待値:

```text
host_backup_copy_executed_count=0
kv_pool_read_count=0
kv_snapshot_count=0
attention_override_true_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
```

---

## 7. State Transition

Phase 2 から Phase 3 への状態遷移。

```text
relaykv_policy_dry_run
→ relaykv_materialization_request
→ relaykv_materialization_result
→ relaykv_materialization_summary
→ readiness for Phase 4 attention connection
```

Phase 3 初期:

```text
relaykv_policy_dry_run
→ fake materialization request
→ fake materialization result
→ summary
```

次段階:

```text
relaykv_policy_dry_run
→ candidate event materialization request
→ guarded_noop materialization result
→ summary
```

さらに後:

```text
relaykv_policy_dry_run
→ host backup copy materialization request
→ host backup copy materialized result
→ summary
```

---

## 8. Failure Modes

Phase 3 で想定する failure mode。

### 8.1 metadata mismatch

```text
request_id mismatch
req_pool_idx mismatch
layer_id mismatch
seq_len missing
```

state:

```text
blocked
```

blocking reason:

```text
metadata_mismatch
```

### 8.2 no selected blocks

```text
selected_block_ids=[]
```

state:

```text
skipped
```

reason:

```text
no_selected_blocks
```

### 8.3 summary-only unjoinable

```text
join_granularity="summary_only_unjoinable"
```

state:

```text
blocked
```

reason:

```text
summary_only_unjoinable
```

### 8.4 req_pool_idx missing

```text
req_pool_idx is None
```

state:

```text
blocked
```

reason:

```text
req_pool_idx_missing
```

### 8.5 safety counter nonzero

```text
kv_cache_mutation_true_count > 0
runtime_writeback_true_count > 0
attention_override_true_count > 0
scheduler_policy_noop_false_count > 0
```

state:

```text
blocked
```

reason:

```text
safety_counter_nonzero
```

### 8.6 attempted forbidden action

```text
KV pool read
attention override
scheduler mutation
runtime writeback
```

state:

```text
error
```

reason:

```text
forbidden_action_attempted
```

---

## 9. Materialization Summary Schema

materialization results は summary で集計する。

```text
summary_type="relaykv_materialization_summary"
total_materialization_requests
total_materialization_results
materialized_result_count
fake_materialized_count
guarded_noop_count
candidate_event_materialized_count
host_backup_copy_materialized_count
skipped_count
fallback_count
blocked_count
error_count
per_request_counts
per_layer_counts
per_state_counts
per_mode_counts
```

Safety counters:

```text
source_mutated_true_count
attention_override_true_count
kv_cache_mutation_true_count
runtime_writeback_true_count
scheduler_policy_noop_false_count
host_backup_copy_executed_count
kv_pool_read_count
kv_snapshot_count
```

Expected for Phase 3.1〜3.3:

```text
host_backup_copy_executed_count=0
kv_pool_read_count=0
kv_snapshot_count=0
attention_override_true_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
```

---

## 10. Phase 3 Readiness Gate

Phase 2 で追加した readiness helper:

```text
assess_relaykv_readonly_materialization_readiness_for_smoke(report)
```

Phase 3 に入る条件:

```text
ready_for_materialization=true
readiness_state="ready_for_safe_materialization_dry_run_complete"
```

条件:

```text
report_generated_from_readonly_inputs=true
overall_safety_status="pass"
policy_dry_run_included=true
policy_dry_run_total_events > 0
joined_count > 0
join_granularity != "summary_only_unjoinable"
req_pool_idx_missing_count == 0
all safety counters are zero
```

Phase 3 実装helperは、最初にこの readiness result を受け取る設計にする。

---

## 11. 実装ステップ案

### Step 3.1: fake materialization helper

目的:

```text
dry-run policy event から fake materialization result を作る。
```

追加候補:

```text
build_relaykv_fake_materialization_results_for_smoke(...)
summarize_relaykv_materialization_results_for_smoke(...)
```

禁止:

```text
KV pool read
host backup copy
snapshot
attention
scheduler
runtime writeback
```

### Step 3.2: no-op guarded materialization helper

目的:

```text
candidate event から guarded_noop result を作る。
```

状態:

```text
materialization_state="guarded_noop"
materialization_mode="noop_guarded"
```

### Step 3.3: candidate event materialization result

目的:

```text
host backup candidate event payload を materialization result schema に写像する。
```

まだ実copyはしない。

### Step 3.4: host backup copy safety boundary

目的:

```text
actual host backup copy helper を使う前に、
どの関数が何を読むのかを固定する。
```

この段階で初めて `host_backup_copy_executed_count` が 1 以上になる可能性を扱う。

### Step 3.5: runtime connection preflight

目的:

```text
runtime接続前に、readiness / materialization summary / safety counters を確認する。
```

---

## 12. 次のCodexタスク候補

最初の実装タスクは以下。

```text
Add RelayKV fake materialization result helper and smoke.
```

許可ファイル:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_fake_materialization_smoke.py
```

禁止:

```text
ForwardBatch
ModelRunner
scheduler
attention
memory_pool
flashinfer
.github/workflows
```

helper候補:

```text
build_relaykv_fake_materialization_results_for_smoke(
    policy_dry_run_events,
    readiness=None,
)

summarize_relaykv_materialization_results_for_smoke(results)
```

期待:

```text
policy_dry_run event から materialization_result を生成
materialization_state="fake_materialized"
materialization_mode="fake"
selected_block_ids -> retrieved_block_ids / materialized_block_ids
anchor_block_ids / recent_block_ids は保持
safety counters all zero
```

---

## 13. 最重要判断

Phase 3 では、いきなり実KV操作に入らない。

最初にやること:

```text
schema
fake result
summary
readiness gate
guarded no-op
```

その後に初めて host backup copy helper の安全境界へ進む。

Phase 3 の合言葉:

```text
materialize the schema before materializing KV.
```

日本語では:

```text
KVを実体化する前に、materialization結果の形を実体化する。
```
