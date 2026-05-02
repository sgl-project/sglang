# Devlog: RelayKV candidate copy summary / counters

## 日付

2026-05-02 JST

## 対象repo

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`

## 目的

RelayKV runtime host backup copy candidate path の発火状況と安全条件を、request / layer 単位で集計できるようにする。

今回の目的は、実runtime接続前に以下を確認できる観測基盤を完成させること。

```text
applied_candidate が何件あったか
fallback_candidate が何件あったか
host backup copy candidate が何件実行されたか
fallback no-op が何件あったか
source_mutated / attention_override / kv_cache_mutation / runtime_writeback / scheduler_policy_noop が安全条件を満たしているか
```

今回も以下は行わない。

```text
attention接続
KV cache free
KV pool書き換え
runtime writeback
scheduler挙動変更
scheduler.py変更
attention backend変更
.github/workflows変更
実server起動
実model起動
```

## 背景

前段階までに以下を実装・確認済み。

```text
1. budget metadata
2. runtime_policy_state
3. policy counters
4. scheduler shadow candidate event
5. fallback no-op guard
6. runtime policy smoke
7. host backup dry-copy guard
8. fake tensor host backup actual-copy smoke
9. fake KV pool layout snapshot smoke
10. actual KV pool observation smoke
11. flashinfer cache/log workaround
12. read-only KV pool snapshot helper consolidation
13. runtime host backup copy candidate path
```

前段階では、runtime candidate event payload と同じ構造から以下が通った。

```text
applied_candidate:
  read-only snapshot
  host backup copy candidate
  host_backup_copy_executed = true
  copy_equal = true
  source_mutated = false

fallback_candidate:
  snapshotなし
  copyなし
  fallback_candidate_noop_guard = true
```

今回は、この candidate copy event payload を集計する summary / counters を追加した。

## 変更ファイル

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_runtime_policy_smoke.py
```

`git diff`上では以下の付近に変更。

```text
python/sglang/srt/relaykv/metrics.py:161
scripts/relaykv_runtime_policy_smoke.py:11
```

## 追加helper

公開helper。

```text
summarize_host_backup_copy_candidates_for_smoke()
log_host_backup_copy_candidate_summary()
```

内部helper。

```text
_candidate_copy_count_template()
_increment_candidate_copy_counts()
_counter_payload()
```

## summary / counters仕様

candidate copy event payloadだけを読み、集計する。

対象。

```text
runtime host backup copy candidate path のevent payload
```

非対象。

```text
attention結果
KV cache free
KV pool mutation
runtime writeback
scheduler decision
```

集計項目。

```text
total_candidate_events
applied_candidate_count
fallback_candidate_count
snapshot_created_count
snapshot_skipped_count
host_backup_copy_candidate_count
host_backup_copy_executed_count
host_backup_copy_skipped_count
copy_equal_true_count
copy_equal_false_count
source_mutated_true_count
source_mutated_false_count
fallback_candidate_noop_guard_count
attention_override_true_count
attention_override_false_count
kv_cache_mutation_true_count
kv_cache_mutation_false_count
runtime_writeback_true_count
runtime_writeback_false_count
scheduler_policy_noop_true_count
scheduler_policy_noop_false_count
skipped_reason_counts
policy_state_counts
per_layer_counts
per_request_counts
```

## runtime smoke summaryログ例

確認されたsummary。

```text
applied_candidate_count = 1
fallback_candidate_count = 2
host_backup_copy_executed_count = 1
fallback_candidate_noop_guard_count = 2
source_mutated_true_count = 0
attention_override_true_count = 0
kv_cache_mutation_true_count = 0
runtime_writeback_true_count = 0
scheduler_policy_noop_false_count = 0
skipped_reason_counts = {"fallback_candidate_noop_guard": 2}
per_layer_counts["0"].total_candidate_events = 3
per_request_counts["rid-applied-a"].host_backup_copy_executed_count = 1
```

## 安全条件

今回も以下を維持した。

```text
attention差し替えなし
KV cache freeなし
KV pool書き換えなし
runtime writebackなし
scheduler挙動変更なし
scheduler.py未変更
attention backend未変更
.github/workflows差分なし
```

重要なsummary上の安全条件。

```text
source_mutated_true_count = 0
attention_override_true_count = 0
kv_cache_mutation_true_count = 0
runtime_writeback_true_count = 0
scheduler_policy_noop_false_count = 0
```

## 実行済み確認

py_compile。

```bash
.venv/bin/python -m py_compile   python/sglang/srt/relaykv/memory.py   python/sglang/srt/relaykv/metrics.py   scripts/relaykv_runtime_policy_smoke.py   scripts/relaykv_actual_kv_pool_observation_smoke.py   scripts/relaykv_kv_pool_snapshot_smoke.py   scripts/relaykv_host_backup_copy_smoke.py
```

smoke。

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_policy_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_actual_kv_pool_observation_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_kv_pool_snapshot_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_host_backup_copy_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_memory_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_budget_sweep.py
```

diff確認。

```bash
git diff --check
git diff --name-status
git diff --name-status | grep '.github/workflows' || true
```

結果。

```text
py_compile: pass
relaykv_runtime_policy_smoke.py: pass
relaykv_actual_kv_pool_observation_smoke.py: pass
relaykv_kv_pool_snapshot_smoke.py: pass
relaykv_host_backup_copy_smoke.py: pass
relaykv_memory_smoke.py: pass
relaykv_budget_sweep.py: pass
git diff --check: pass
.github/workflows差分: なし
```

## 判断

今回の変更により、runtime candidate event payload から host backup copy candidate を実行し、その発火状況と安全条件をsummaryで確認できるようになった。

到達点。

```text
runtime_policy_state
  -> applied_candidate / fallback_candidate 判定
  -> read-only snapshot
  -> host backup copy candidate
  -> candidate copy summary / counters
```

重要なのは、ここまで進んでもまだ実モデル実行には影響を与えていないこと。

```text
attention未接続
KV freeなし
KV pool mutationなし
runtime writebackなし
scheduler挙動変更なし
```

このため、実runtime接続前の観測基盤として、いったん区切るのに適した状態になった。

## 現在の到達点

SGLang側の RelayKV host backup / snapshot 基盤は以下まで進んだ。

```text
1. budget metadata
2. runtime_policy_state
3. policy counters
4. scheduler shadow candidate event
5. fallback no-op guard
6. runtime policy smoke
7. host backup dry-copy guard
8. fake tensor host backup actual-copy smoke
9. fake KV pool layout snapshot smoke
10. actual KV pool observation smoke
11. flashinfer cache/log workaround
12. read-only KV pool snapshot helper consolidation
13. runtime host backup copy candidate path
14. candidate copy summary / counters
```

## 次スレッドへ移行する理由

ここで次スレッドへ移行するのがよい。

理由。

```text
1. 実runtime接続前の観測基盤が一通り閉じた
2. candidate copyの発火状況と安全条件をsummaryで見られる
3. 次のテーマが「実runtime接続するかどうかの判断」に変わる
4. 会話が長くなっており、次スレッドの方が整理しやすい
```

次スレッドの開始テーマ。

```text
RelayKV SGLang: 実runtime接続前の判断
```

次スレッドで最初に検討すること。

```text
1. runtime host backup copy candidate path を実server smokeへ近づけるか
2. まだ pseudo runtime smoke を増やすか
3. 実runtimeでread-only snapshot/copyだけを行う最小接続点はどこか
4. KV free / attention接続 / writeback に進む前の停止条件をどう置くか
```

まだ禁止するもの。

```text
attention接続
KV cache free
KV pool書き換え
runtime writeback
scheduler挙動変更
```

## commit候補

```bash
cd ~/work/sglang-relaykv

git status --short
git diff --name-status
git diff --name-status | grep '.github/workflows' || true
git diff --check

git add   python/sglang/srt/relaykv/metrics.py   scripts/relaykv_runtime_policy_smoke.py   notes/devlog_2026-05-02_relaykv_candidate_copy_summary_counters_ja.md

git commit -m "Add RelayKV candidate copy summary counters"

git push mine relaykv-host-backup-shadow
```

## 次スレッド用プロンプト

```text
RelayKV / SGLang 実装の続きです。repoは ~/work/sglang-relaykv、branch は relaykv-host-backup-shadow です。

これまでの到達点:
- RelayKV budget metadata
- runtime_policy_state
- policy counters
- scheduler shadow candidate event
- fallback_candidate no-op guard
- runtime policy smoke
- host backup dry-copy guard
- fake tensor host backup actual-copy smoke
- fake KV pool layout snapshot smoke
- actual KV pool observation smoke
- flashinfer cache/log workaround
- read-only KV pool snapshot helper consolidation
- runtime host backup copy candidate path
- candidate copy summary / counters

現在の重要な状態:
- runtime candidate event payload と同じ構造から、applied_candidate のみ read-only snapshot -> host backup copy candidate が通る
- fallback_candidate は no-op
- candidate copy summary / counters で request/layer単位の発火状況と安全条件を集計できる
- source_mutated_true_count=0
- attention_override_true_count=0
- kv_cache_mutation_true_count=0
- runtime_writeback_true_count=0
- scheduler_policy_noop_false_count=0
- scheduler.py、attention backend、.github/workflows は未変更

まだ禁止:
- attention接続
- KV cache free
- KV pool書き換え
- runtime writeback
- scheduler挙動変更

次にやりたいこと:
実runtime接続前の判断をしたい。まず、runtime host backup copy candidate path を実server smokeへ近づけるべきか、まだpseudo runtime smokeで固めるべきかを判断して、最小リスクの次タスクを設計してほしい。

出力順:
1. 見るべきファイル
2. 現在の理解
3. 次の問題点 or 実装ポイント
4. 最有力の次タスク
5. 最小修正案
6. 確認用ログ / smoke
```
