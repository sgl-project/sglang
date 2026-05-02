# RelayKV Host Backup Shadow Runtime Observation Plan

## 日付

2026-05-02 JST

## 対象

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`

## 現在の段階

runtime candidate event payload 互換の構造から、`applied_candidate` の場合だけ read-only snapshot -> host backup copy candidate が通る。

`fallback_candidate` は no-op guard により snapshot / copy を行わない。

まだ以下には進んでいない。

```text
実runtime接続
attention接続
attention override
KV cache free
KV pool書き換え
runtime writeback
scheduler挙動変更
```

## 通過済み smoke

```text
scripts/relaykv_runtime_policy_smoke.py
scripts/relaykv_host_backup_candidate_variation_smoke.py
scripts/relaykv_runtime_observation_readonly_smoke.py
scripts/relaykv_actual_kv_pool_observation_smoke.py
scripts/relaykv_kv_pool_snapshot_smoke.py
scripts/relaykv_host_backup_copy_smoke.py
scripts/relaykv_memory_smoke.py
```

## read-only runtime observation smoke の確認値

`scripts/relaykv_runtime_observation_readonly_smoke.py` は、ForwardBatch / ModelRunner 相当の fake batch 情報から runtime observation 風の candidate event を6件流す。

確認済み summary。

```text
total_candidate_events = 6
applied_candidate_count = 4
fallback_candidate_count = 2
host_backup_copy_executed_count = 4
fallback_candidate_noop_guard_count = 2
per_layer_counts = 0, 1, 2
per_request_counts = rid-a, rid-b, rid-c
per_batch_counts = obs-batch-a
skipped_reason_counts = {"fallback_candidate_noop_guard": 2}
```

## 保持すべき safety invariant

次段階へ進む前後で、以下を維持する。

```text
host_backup_copy_executed_count == applied_candidate_count
fallback_candidate_noop_guard_count == fallback_candidate_count
source_mutated_true_count == 0
attention_override_true_count == 0
kv_cache_mutation_true_count == 0
runtime_writeback_true_count == 0
scheduler_policy_noop_false_count == 0
```

## まだ禁止する変更

```text
attention接続
attention override
attention backend変更
KV cache free
KV pool書き換え
runtime writeback
scheduler挙動変更
ForwardBatch / ModelRunner hot path の無条件変更
memory_pool getter 経由の観測追加
.github/workflows変更
重いmodel download前提の変更
```

## 次に進む条件

次段階は実runtime接続ではなく、default-off read-only observation hook の設計に留める。

進行条件。

```text
default-off
環境変数で完全disable可能
read-only observation のみ
summary log only
no mutation
no writeback
no scheduler decision change
既存smokeがすべて通る
制約grepが空
```

制約grep。

```bash
git diff --name-only | grep -E 'scheduler.py|attention|flashinfer|\.github/workflows' || true
```

## 即停止 / rollback 条件

以下のいずれかが出た場合は、実runtime寄りの接続を止め、直前の read-only smoke 段階へ戻す。

```text
source_mutated_true_count != 0
attention_override_true_count != 0
kv_cache_mutation_true_count != 0
runtime_writeback_true_count != 0
scheduler_policy_noop_false_count != 0
host_backup_copy_executed_count != applied_candidate_count
fallback_candidate で copy 実行が増える
scheduler.py / attention / flashinfer / .github/workflows に意図しない差分が出る
実model downloadやserver起動が必須になる
```

## 次段階の候補

次は実runtime接続ではなく、default-off read-only observation hook の設計比較を行う。

候補地点。

```text
scheduler.py の既存 RelayKV shadow/runtime policy event log 位置
ForwardBatch.init_new()
ModelRunner.forward() / _forward_raw() の手前
```

ただし、いずれも hot path なので、最初はコード変更なしの設計比較に留める。

設計比較では以下を確認する。

```text
どの情報が既存payloadから取れるか
どの情報がread-onlyに観測できるか
default-off guard をどこに置くべきか
summary log only にできるか
既存smokeへ同じ safety invariant を流用できるか
```

## 確認コマンド

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_observation_readonly_smoke.py
git diff --check
git diff --name-only | grep -E 'scheduler.py|attention|flashinfer|\.github/workflows' || true
```
