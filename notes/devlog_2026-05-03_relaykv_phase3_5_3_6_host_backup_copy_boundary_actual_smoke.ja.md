# Devlog: RelayKV Phase 3.5〜3.6 Host Backup Copy Boundary / Isolated Actual Copy

## 日付確認

- Devlog date: **2026-05-03**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. 今回の区切り

今回の devlog は、RelayKV/SGLang 統合における **Phase 3.5〜3.6** のまとまりを記録する。

小さな helper 単位ではなく、以下のフェーズ境界までをまとめる。

```text
Phase 3.5:
  host backup copy safety boundary
  host backup copy request schema
  host backup copy boundary result
  readonly report integration

Phase 3.6:
  isolated actual host backup copy smoke
  actual host backup copy report
  attention connection readiness: design-only
```

この区切りで、RelayKV は metadata-only materialization readiness から一歩進み、**host backup copy を isolated smoke として実行した扱いにできる段階**まで到達した。

ただし、まだ runtime KV pool / attention / scheduler には接続していない。

---

## 2. 背景

RelayKV は、低VRAM GPU上で量子化dense modelを動かす際に、decode-time KV working set を residual VRAM budget 内に収めるための **VRAM-aware memory management layer** として設計している。

現在の SGLang branch では、RelayKV を SGLang の runtime path に直接接続する前に、以下を順に分離して確認している。

```text
runtime observation
→ host backup candidate summary
→ runtime/candidate join
→ dry-run policy
→ metadata-only materialization
→ host backup copy boundary
→ isolated actual copy
```

今回の作業は、このうち **host backup copy boundary** と **isolated actual copy** に相当する。

---

## 3. Phase 3.5: Host Backup Copy Safety Boundary

### 3.1 設計メモ

作成した設計メモ:

```text
notes/relaykv_phase3_5_host_backup_copy_safety_boundary_design_2026-05-03.ja.md
```

目的:

```text
host_backup_copy_executed_count
kv_pool_read_count
kv_snapshot_count
```

を明確に分離し、metadata-only phase ではこれらが 0 のままであることを確認する。

Phase 3.5 の時点では、host backup copy はまだ実行しない。

---

### 3.2 Host backup copy readiness / request smoke

追加・実装した主な helper:

```text
assess_relaykv_host_backup_copy_readiness_for_smoke(report)

build_relaykv_host_backup_copy_requests_for_smoke(
    candidate_event_materialization_results,
    copy_readiness=None,
)

summarize_relaykv_host_backup_copy_requests_for_smoke(requests)
```

追加 smoke:

```text
scripts/relaykv_host_backup_copy_boundary_smoke.py
```

確認内容:

```text
candidate-event materialized result
→ host backup copy readiness
→ relaykv_host_backup_copy_request
→ request summary
```

正常系では:

```text
copy_state="request_ready"
copy_mode="host_backup_copy_boundary"
copy_source="host_backup_candidate"
copy_destination="materialization_result_only"
copy_guard_state="pre_attention_no_runtime_writeback"
```

安全 flags:

```text
host_backup_copy_executed=false
kv_pool_read=false
kv_snapshot=false
attention_override=false
runtime_writeback=false
scheduler_policy_noop=true
```

---

### 3.3 Host backup copy boundary result smoke

追加・実装した主な helper:

```text
build_relaykv_host_backup_copy_boundary_results_for_smoke(
    host_backup_copy_requests,
    execute_copy=False,
)

summarize_relaykv_host_backup_copy_boundary_results_for_smoke(results)
```

追加 smoke:

```text
scripts/relaykv_host_backup_copy_boundary_result_smoke.py
```

目的:

```text
relaykv_host_backup_copy_request
→ relaykv_host_backup_copy_boundary_result
```

を確認する。

ただし、`execute_copy=False` 固定で、copy は実行しない。

正常系:

```text
materialization_state="host_backup_copy_boundary_noop"
materialization_mode="host_backup_copy_boundary"
copy_state="boundary_noop"
```

確認した blocked cases:

```text
execute_copy=True
copy_request_not_ready
not_host_backup_copy_request
no_materialized_blocks
```

この段階でも、以下は 0 のまま。

```text
host_backup_copy_executed_count=0
kv_pool_read_count=0
kv_snapshot_count=0
```

---

### 3.4 Boundary result report integration

拡張した helper:

```text
build_relaykv_readonly_runtime_candidate_join_report_for_smoke(...)
```

追加 optional summary:

```text
host_backup_copy_request_summary=None
host_backup_copy_boundary_result_summary=None
```

追加 flatten fields:

```text
host_backup_copy_request_summary_included
host_backup_copy_request_total
host_backup_copy_request_ready_count
host_backup_copy_request_blocked_count
host_backup_copy_request_materialized_kv_count

host_backup_copy_boundary_result_summary_included
host_backup_copy_boundary_result_total
host_backup_copy_boundary_noop_count
host_backup_copy_boundary_blocked_count
host_backup_copy_boundary_error_count
host_backup_copy_boundary_materialized_kv_count
```

追加 helper:

```text
assess_relaykv_actual_host_backup_copy_readiness_for_smoke(report)
```

ready state:

```text
ready_for_actual_host_backup_copy_smoke_boundary_complete
```

意味:

```text
actual host backup copy smoke の isolated helper 実装に進める
```

ただし、runtime / attention に接続してよいという意味ではない。

更新 smoke:

```text
scripts/relaykv_readonly_diagnostic_flow_smoke.py
scripts/relaykv_runtime_candidate_join_report_smoke.py
```

重要な安全仕様:

metadata-only readonly report では、引き続き次が nonzero なら fail。

```text
host_backup_copy_executed_count
kv_pool_read_count
kv_snapshot_count
```

---

## 4. Phase 3.6: Isolated Actual Host Backup Copy Smoke

### 4.1 設計メモ

作成した設計メモ:

```text
notes/relaykv_phase3_6_actual_host_backup_copy_smoke_design_2026-05-03.ja.md
```

Phase 3.6 の目的:

```text
host_backup_copy_executed_count > 0
```

を初めて許可する。

ただし、許可するのは isolated smoke 内だけであり、以下は引き続き禁止。

```text
KV pool read
KV snapshot
attention接続
scheduler decision変更
runtime writeback
KV cache free
RadixTree変更
HiCache prefetch hint
production runtime connection
```

---

### 4.2 Isolated actual host backup copy smoke

追加・実装した helper:

```text
build_relaykv_actual_host_backup_copy_results_for_smoke(
    host_backup_copy_requests,
    actual_copy_readiness=None,
    execute_copy=True,
)

summarize_relaykv_actual_host_backup_copy_results_for_smoke(results)
```

追加 smoke:

```text
scripts/relaykv_actual_host_backup_copy_smoke.py
```

正常系:

```text
relaykv_host_backup_copy_request
→ relaykv_materialization_result
```

result:

```text
event_type="relaykv_materialization_result"
materialization_state="host_backup_copy_materialized"
materialization_mode="host_backup_copy"
copy_state="copy_executed"
copy_mode="host_backup_copy_isolated_smoke"
source="host_backup_copy_request_to_isolated_materialization_result"
```

安全 flags:

```text
host_backup_copy_executed=true
kv_pool_read=false
kv_snapshot=false
attention_override=false
kv_cache_mutation=false
runtime_writeback=false
scheduler_policy_noop=true
source_mutated=false
```

確認した pass flow:

```text
2 request_ready + ready readiness + execute_copy=True
→ 2 host_backup_copy_materialized results
```

summary expected:

```text
host_backup_copy_materialized_count=2
host_backup_copy_executed_count=2
materialized_kv_count=4
kv_pool_read_count=0
kv_snapshot_count=0
attention/runtime/scheduler mutation counters=0
```

確認した blocked cases:

```text
readiness missing
readiness blocked
execute_copy=False
request not ready
wrong event_type
empty materialized_block_ids
```

---

### 4.3 Actual host backup copy report integration

追加・実装した helper:

```text
build_relaykv_actual_host_backup_copy_report_for_smoke(
    readonly_report,
    actual_host_backup_copy_summary,
)
```

追加した actual-copy report:

```text
report_type="relaykv_actual_host_backup_copy_report"
actual_copy_report_generated_from_isolated_smoke_inputs=true
```

ここでは、metadata-only readonly report とは安全ルールを分けた。

actual-copy report では、次を許可する。

```text
host_backup_copy_executed_count > 0
```

ただし、次が nonzero の場合は fail。

```text
kv_pool_read_count
kv_snapshot_count
source_mutated_true_count
attention_override_true_count
kv_cache_mutation_true_count
runtime_writeback_true_count
scheduler_policy_noop_false_count
```

追加 helper:

```text
assess_relaykv_attention_connection_readiness_for_smoke(report)
```

ready state:

```text
ready_for_attention_connection_design_only
```

これは、attention に接続してよいという意味ではなく、**attention connection design に進める**という意味。

追加 smoke:

```text
scripts/relaykv_actual_host_backup_copy_report_smoke.py
```

確認した pass flow:

```text
readonly report pass
+ actual summary:
    host_backup_copy_materialized_count=2
    host_backup_copy_executed_count=2
    kv_pool_read_count=0
    kv_snapshot_count=0
→ actual_copy_safety_status="pass"
→ attention readiness ready_for_attention_connection_design_only
```

確認した fail cases:

```text
readonly report safety fail
actual summary missing
actual materialized count 0
actual executed count 0
actual blocked count > 0
actual error count > 0
kv_pool_read_count > 0
kv_snapshot_count > 0
attention_override_true_count > 0
runtime_writeback_true_count > 0
scheduler_policy_noop_false_count > 0
```

---

## 5. 実行済み smoke / checks

代表的に実行し、pass を確認したもの:

```bash
PYTHONPATH=python .venv/bin/python -m py_compile   python/sglang/srt/relaykv/metrics.py   scripts/relaykv_actual_host_backup_copy_report_smoke.py
```

```bash
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

禁止領域 grep:

```bash
git diff --name-only | grep -E 'forward_batch_info.py|model_runner.py|scheduler.py|schedule_batch.py|attention|flashinfer|memory_pool.py|\.github/workflows|python/sglang/srt/managers' || true
```

結果:

```text
git diff --check: pass
禁止領域 grep: 出力なし
```

---

## 6. 変更ファイル一覧

Phase 3.5〜3.6 で主に変更されたファイル:

```text
python/sglang/srt/relaykv/metrics.py

scripts/relaykv_host_backup_copy_boundary_smoke.py
scripts/relaykv_host_backup_copy_boundary_result_smoke.py
scripts/relaykv_actual_host_backup_copy_smoke.py
scripts/relaykv_actual_host_backup_copy_report_smoke.py

scripts/relaykv_readonly_diagnostic_flow_smoke.py
scripts/relaykv_runtime_candidate_join_report_smoke.py

notes/relaykv_phase3_5_host_backup_copy_safety_boundary_design_2026-05-03.ja.md
notes/relaykv_phase3_6_actual_host_backup_copy_smoke_design_2026-05-03.ja.md
```

---

## 7. 安全境界

今回の最重要点は、`host_backup_copy_executed_count > 0` を初めて許可した一方で、runtime path への接続はまだ行っていないこと。

Phase 3.6 pass flow:

```text
host_backup_copy_executed_count=2
kv_pool_read_count=0
kv_snapshot_count=0
source_mutated_true_count=0
attention_override_true_count=0
kv_cache_mutation_true_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
```

まだ実施していないこと:

```text
runtime KV pool read
KV snapshot
attention connection
scheduler decision change
runtime writeback
KV cache free
RadixTree変更
HiCache prefetch hint
production runtime connection
```

---

## 8. 現在の到達点

現在の RelayKV/SGLang integration phase:

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
```

現在の終点:

```text
attention connection readiness: ready_for_attention_connection_design_only
```

これは design-only readiness であり、実 attention 接続の実装許可ではない。

---

## 9. 次に進むべきこと

次は **attention connection design** に進む。

ただし、いきなり attention backend へ接続しない。

推奨順序:

```text
1. attention connection safety boundary design memo
2. attention input schema / working KV handoff schema の設計
3. attention connection dry-run result helper
4. no-op attention override smoke
5. isolated attention comparison smoke
6. 実 attention path 接続の検討
```

次の最初の作業は:

```text
RelayKV Phase 4 Attention Connection Safety Boundary Design
```

この段階では、まだ以下を禁止する。

```text
actual attention override
KV pool read
runtime writeback
scheduler decision mutation
```

---

## 10. commit command

今回の devlog を repo に保存する場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/devlog_2026-05-03_relaykv_phase3_5_3_6_host_backup_copy_boundary_actual_smoke.ja.md \
  notes/devlog_2026-05-03_relaykv_phase3_5_3_6_host_backup_copy_boundary_actual_smoke.ja.md

git status --short
git diff --check

git add notes/devlog_2026-05-03_relaykv_phase3_5_3_6_host_backup_copy_boundary_actual_smoke.ja.md
git commit -m "docs: add relaykv phase 3.5 3.6 devlog"
git push mine relaykv-host-backup-shadow
```
