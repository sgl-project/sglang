# Devlog: RelayKV Phase 4.1〜4.4 Attention Connection Metadata Chain

## 日付確認

- Devlog date: **2026-05-03**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. 今回の区切り

今回の devlog は、RelayKV/SGLang integration の **Phase 4.1〜4.4** をまとめる。

Phase 3.6 までで、RelayKV は isolated actual host backup copy smoke と actual-copy report まで到達していた。

今回の Phase 4.1〜4.4 では、実 attention backend に接続せず、metadata-only のまま以下の chain を構築した。

```text
attention_handoff_candidate
→ attention_connection_dry_run_result
→ attention_override_noop_result
→ attention_comparison_plan
```

この段階で、RelayKV が attention に渡したい working KV set を、実 attention 実行前に安全に観測・比較できる形になった。

---

## 2. 背景

RelayKV は、低VRAM GPUで量子化dense modelを動かす際に、decode-time KV working set を residual VRAM budget 内に収めるための **VRAM-aware memory management layer** である。

ここまでの到達点は以下。

```text
Phase 0: ForwardBatch metadata observation
Phase 1: runtime observation + host backup candidate join
Phase 2: dry-run policy plumbing
Phase 3.0: safe materialization design
Phase 3.1: fake materialization
Phase 3.2: guarded noop materialization
Phase 3.3: candidate-event materialization
Phase 3.4: metadata-only materialization readiness
Phase 3.5: host backup copy boundary
Phase 3.6: isolated actual host backup copy smoke/report
Phase 4.0: attention connection safety boundary design
```

今回の作業は、Phase 4.0 の設計に従い、実 attention path を触る前の metadata chain を閉じるもの。

---

## 3. Phase 4.1: Attention Handoff Metadata Smoke

### 3.1 目的

Phase 4.1 の目的は、actual host backup copy result から、attention に渡す候補 metadata を生成すること。

まだ attention backend には接続しない。

```text
actual host backup copy result
→ attention handoff readiness
→ relaykv_attention_handoff_candidate
```

### 3.2 追加 helper

```text
assess_relaykv_attention_handoff_readiness_for_smoke(report)

build_relaykv_attention_handoff_candidates_for_smoke(
    actual_copy_results,
    attention_handoff_readiness=None,
)

summarize_relaykv_attention_handoff_candidates_for_smoke(candidates)
```

readiness input:

```text
relaykv_actual_host_backup_copy_report
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

### 3.3 Handoff candidate schema

正常系 output:

```text
event_type="relaykv_attention_handoff_candidate"
handoff_state="handoff_ready"
handoff_mode="metadata_only"
source="actual_host_backup_copy_result_to_attention_handoff_candidate"
```

主要 fields:

```text
request_id
req_pool_idx
seq_len
layer_id
recent_block_ids
anchor_block_ids
retrieved_block_ids
candidate_block_ids
materialized_block_ids
working_kv_block_ids = materialized_block_ids
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
attention_override_noop=false
kv_pool_read=false
kv_snapshot=false
runtime_writeback=false
scheduler_policy_noop=true
kv_cache_mutation=false
source_mutated=false
```

### 3.4 Smoke 結果

追加 smoke:

```text
scripts/relaykv_attention_handoff_smoke.py
```

pass flow:

```text
actual-copy report pass
→ attention handoff readiness pass
→ 2 host_backup_copy_materialized results
→ 2 handoff_ready candidates
```

summary:

```text
handoff_ready_count=2
working_kv_block_count=4
attention_connection_attempted_count=0
attention_override_true_count=0
attention_override_noop_count=0
kv_pool_read_count=0
kv_snapshot_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
```

blocked cases:

```text
readiness missing
readiness blocked
wrong event type
wrong materialization state
copy not executed
empty materialized blocks
report not actual-copy report
actual copy safety fail
actual materialized count 0
actual executed count 0
kv pool read observed
kv snapshot observed
attention override observed
runtime writeback observed
scheduler mutation observed
```

---

## 4. Phase 4.2: Attention Connection Dry-run Result Smoke

### 4.1 目的

Phase 4.2 の目的は、handoff candidate を metadata-only の dry-run result に変換すること。

この段階で初めて以下を許可した。

```text
attention_connection_attempted_count > 0
```

ただし、実 attention は実行しない。

```text
attention_handoff_candidate
→ attention_connection_dry_run_result
```

### 4.2 追加 helper

```text
build_relaykv_attention_connection_dry_run_results_for_smoke(
    handoff_candidates,
    execute_attention=False,
)

summarize_relaykv_attention_connection_dry_run_results_for_smoke(results)
```

正常系 input:

```text
event_type="relaykv_attention_handoff_candidate"
handoff_state="handoff_ready"
handoff_mode="metadata_only"
working_kv_block_ids non-empty
execute_attention=False
```

正常系 output:

```text
event_type="relaykv_attention_connection_dry_run_result"
attention_connection_state="dry_run"
attention_connection_mode="metadata_only"
source="attention_handoff_candidate_to_connection_dry_run_result"
attention_target_backend="unconnected"
```

safety flags:

```text
attention_connection_attempted=true
attention_override=false
attention_override_noop=false
kv_pool_read=false
kv_snapshot=false
runtime_writeback=false
scheduler_policy_noop=true
kv_cache_mutation=false
source_mutated=false
```

warning:

```text
metadata_only_attention_connection_dry_run
```

blocked reasons:

```text
execute_attention_not_allowed_in_dry_run_smoke
not_attention_handoff_candidate
handoff_not_ready
handoff_not_metadata_only
no_working_kv_blocks
```

### 4.3 Smoke 結果

追加 smoke:

```text
scripts/relaykv_attention_connection_dry_run_smoke.py
```

pass flow:

```text
2 handoff candidates
→ 2 dry-run results
```

summary:

```text
attention_connection_dry_run_count=2
attention_connection_attempted_count=2
working_kv_block_count=4
attention_override_true_count=0
attention_override_noop_count=0
kv_pool_read_count=0
kv_snapshot_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
```

blocked cases:

```text
execute_attention=True
wrong event type
handoff not ready
handoff mode not metadata_only
empty working kv blocks
```

---

## 5. Phase 4.3: No-op Attention Override Smoke

### 5.1 目的

Phase 4.3 の目的は、dry-run result を override point に渡せる形へ変換すること。

ただし、実 override は行わない。

この段階で初めて以下を許可した。

```text
attention_override_noop_count > 0
```

まだ以下は 0 固定。

```text
attention_override_true_count=0
kv_pool_read_count=0
kv_snapshot_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
```

chain:

```text
attention_connection_dry_run_result
→ attention_override_noop_result
```

### 5.2 追加 helper

```text
build_relaykv_attention_override_noop_results_for_smoke(
    attention_connection_dry_run_results,
    allow_override=False,
)

summarize_relaykv_attention_override_noop_results_for_smoke(results)
```

正常系 input:

```text
event_type="relaykv_attention_connection_dry_run_result"
attention_connection_state="dry_run"
attention_connection_mode="metadata_only"
attention_connection_attempted=true
working_kv_block_ids non-empty
allow_override=False
```

正常系 output:

```text
event_type="relaykv_attention_override_noop_result"
attention_connection_state="override_noop"
attention_connection_mode="noop_guarded"
source="attention_connection_dry_run_result_to_override_noop_result"
attention_target_backend="unconnected"
```

safety flags:

```text
attention_connection_attempted=true
attention_override=false
attention_override_noop=true
attention_override_allowed=false
kv_pool_read=false
kv_snapshot=false
runtime_writeback=false
scheduler_policy_noop=true
kv_cache_mutation=false
source_mutated=false
```

warnings:

```text
attention_override_noop_guarded
no_runtime_attention_backend_connection
```

blocked reasons:

```text
attention_override_not_allowed_in_phase4_noop
not_attention_connection_dry_run_result
attention_connection_not_dry_run
attention_connection_not_metadata_only
attention_connection_not_attempted
no_working_kv_blocks
```

### 5.3 Smoke 結果

追加 smoke:

```text
scripts/relaykv_attention_override_noop_smoke.py
```

pass flow:

```text
2 dry-run results
→ 2 override noop results
```

summary:

```text
attention_override_noop_count=2
attention_connection_attempted_count=2
working_kv_block_count=4
attention_override_true_count=0
kv_pool_read_count=0
kv_snapshot_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
kv_cache_mutation_true_count=0
source_mutated_true_count=0
```

blocked cases:

```text
allow_override=True
wrong event type
not dry_run
not metadata_only
not attempted
empty working blocks
```

---

## 6. Phase 4.4: Attention Comparison Plan Smoke

### 6.1 目的

Phase 4.4 の目的は、metadata 上で以下を比較できる plan を作ること。

```text
Full-KV plan
vs
RelayKV working-KV plan
```

ここでも、実 attention は実行しない。

chain:

```text
attention_override_noop_result
→ attention_comparison_plan
```

### 6.2 追加 helper

```text
build_relaykv_attention_comparison_plans_for_smoke(
    attention_override_noop_results,
    full_kv_block_ids_by_request_layer=None,
)

summarize_relaykv_attention_comparison_plans_for_smoke(plans)
```

`full_kv_block_ids_by_request_layer` は optional。

対応 key:

```text
(request_id, layer_id)
"request_id:layer_id"
request_id
```

明示 key がない場合のみ、metadata から full-KV ids を synthesize する。

synthesize source:

```text
recent_block_ids
anchor_block_ids
candidate_block_ids
retrieved_block_ids
materialized_block_ids
working_kv_block_ids
```

synthesize は `sorted(unique(int values))`。

### 6.3 Comparison plan schema

正常系 output:

```text
event_type="relaykv_attention_comparison_plan"
comparison_state="plan_ready"
comparison_mode="metadata_only"
source="attention_override_noop_result_to_comparison_plan"
```

主要 fields:

```text
full_kv_block_ids
relaykv_working_kv_block_ids
relaykv_working_kv_block_count
full_kv_block_count
reduced_block_count
working_to_full_block_ratio
coverage_block_count
coverage_ratio
missing_from_full_block_ids
full_only_block_ids
```

safety flags:

```text
attention_comparison_executed=false
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

warning:

```text
metadata_only_attention_comparison_plan
```

blocked reasons:

```text
not_attention_override_noop_result
attention_connection_not_override_noop
attention_connection_not_noop_guarded
attention_override_noop_not_true
attention_override_true_not_allowed
no_working_kv_blocks
no_full_kv_blocks
```

### 6.4 Smoke 結果

追加 smoke:

```text
scripts/relaykv_attention_comparison_plan_smoke.py
```

explicit full-KV case:

```text
comparison_plan_ready_count=2
full_kv_block_count=12
relaykv_working_kv_block_count=4
reduced_block_count=8
mean_working_to_full_block_ratio ~= 0.3333333333
mean_coverage_ratio ~= 0.3333333333
```

synthesis fallback case:

```text
full_kv_block_count=12
```

safety counters:

```text
attention_comparison_executed_count=0
attention_connection_attempted_count=2
attention_override_true_count=0
attention_override_noop_count=2
kv_pool_read_count=0
kv_snapshot_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
kv_cache_mutation_true_count=0
source_mutated_true_count=0
```

blocked cases:

```text
wrong event type
state not override_noop
mode not noop_guarded
attention_override_noop=false
attention_override=true
empty working ids
no full-kv ids
```

---

## 7. 実行済み checks

代表的に実行し、pass を確認したもの。

```bash
PYTHONPATH=python .venv/bin/python -m py_compile \\
  python/sglang/srt/relaykv/metrics.py \\
  scripts/relaykv_attention_comparison_plan_smoke.py
```

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_attention_comparison_plan_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_attention_override_noop_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_attention_connection_dry_run_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_attention_handoff_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_actual_host_backup_copy_report_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_actual_host_backup_copy_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_host_backup_copy_boundary_result_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_host_backup_copy_boundary_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_readonly_diagnostic_flow_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_candidate_join_report_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_candidate_event_materialization_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_guarded_noop_materialization_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_fake_materialization_smoke.py
```

diff check:

```bash
git diff --check
```

forbidden-area grep:

```bash
git diff --name-only | grep -E 'forward_batch_info.py|model_runner.py|scheduler.py|schedule_batch.py|attention|flashinfer|memory_pool.py|\\.github/workflows|python/sglang/srt/managers' || true
```

結果:

```text
git diff --check: pass
forbidden-area grep: 出力なし
```

---

## 8. 変更ファイル一覧

Phase 4.1〜4.4 で主に変更されたファイル:

```text
python/sglang/srt/relaykv/metrics.py

scripts/relaykv_attention_handoff_smoke.py
scripts/relaykv_attention_connection_dry_run_smoke.py
scripts/relaykv_attention_override_noop_smoke.py
scripts/relaykv_attention_comparison_plan_smoke.py
```

関連設計メモ:

```text
notes/relaykv_phase4_attention_connection_safety_boundary_design_2026-05-03.ja.md
```

---

## 9. 安全境界

今回の最重要点は、attention connection の metadata chain を作りながら、実 attention backend には一切触れていないこと。

Phase 4.4 時点の pass flow counters:

```text
attention_comparison_executed_count=0
attention_connection_attempted_count=2
attention_override_true_count=0
attention_override_noop_count=2
kv_pool_read_count=0
kv_snapshot_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
kv_cache_mutation_true_count=0
source_mutated_true_count=0
```

まだ実施していないこと:

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

---

## 10. 現在の到達点

現在の RelayKV/SGLang integration phase:

```text
Phase 4.1: attention handoff metadata
Phase 4.2: attention connection dry-run
Phase 4.3: no-op attention override
Phase 4.4: attention comparison plan
```

現在の終点:

```text
Full-KV plan vs RelayKV working-KV plan を metadata-only で比較可能
```

この時点では、RelayKV が attention に渡そうとしている working set の削減率・coverage・full-only blocks を、安全に観測できる。

---

## 11. 次に進むべきこと

次は **Phase 4.5 Real Attention Connection Design**。

ただし、いきなり実装に入らず、まず SGLang の attention path を読み、以下を設計する。

```text
1. SGLang attention path の確認
2. RelayKV working KV を渡す候補位置の洗い出し
3. 実 attention override 前の安全境界
4. comparison-only / shadow-only 接続の可否
5. fallback 条件
6. rollback 条件
```

次に作るべきもの:

```text
RelayKV Phase 4.5 Real Attention Connection Design
```

Phase 4.5 design で特に見るべき対象:

```text
attention backend 呼び出し位置
forward batch と attention metadata の関係
KV pool index / token span / layer の扱い
RadixAttention との境界
HiCache / HiRadixTree との境界
```

まだ実装で許可しないもの:

```text
attention_override_true_count > 0
kv_pool_read_count > 0
kv_snapshot_count > 0
runtime_writeback_true_count > 0
scheduler_policy_noop_false_count > 0
```

---

## 12. commit command

今回の devlog を repo に保存する場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/devlog_2026-05-03_relaykv_phase4_1_4_4_attention_metadata_chain.ja.md \\
  notes/devlog_2026-05-03_relaykv_phase4_1_4_4_attention_metadata_chain.ja.md

git status --short
git diff --check

git add notes/devlog_2026-05-03_relaykv_phase4_1_4_4_attention_metadata_chain.ja.md
git commit -m "docs: add relaykv phase 4 attention metadata devlog"
git push mine relaykv-host-backup-shadow
```
