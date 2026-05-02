# Devlog: RelayKV runtime host backup copy candidate path

## 日付

2026-05-01

## 対象repo

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`

## 目的

RelayKV の runtime candidate event 経路に、host backup copy candidate path を追加する。

今回の目的は、scheduler / runtime candidate event と同じpayload構造から、`runtime_policy_state == "applied_candidate"` の場合だけ read-only snapshot -> host backup copy candidate までを通すこと。

ただし、今回も以下は行わない。

```text
attention接続
KV cache free
KV pool書き換え
runtime writeback
scheduler挙動変更
attention backend変更
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
```

前段階で `memory.py` に集約したhelper。

```text
snapshot_mha_kv_pool_readonly_for_smoke()
snapshot_kv_pool_for_host_backup_smoke()
```

確認済みのKV layout。

```text
observed_layout = mha_split_kv

k_buffer:
  [tokens, heads, head_dim]

v_buffer:
  [tokens, heads, head_dim]

snapshot:
  [tokens, 2, heads, head_dim]
```

確認済みの安全条件。

```text
copy_equal = true
source_mutated = false
kv_cache_mutation = false
attention_override = false
scheduler_policy_noop = true
```

## 変更ファイル

```text
python/sglang/srt/relaykv/memory.py
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_runtime_policy_smoke.py
```

## 追加helper

```text
_host_backup_copy_candidate_noop_log()
run_host_backup_copy_candidate_for_smoke()
```

## candidate path仕様

### 基本方針

scheduler / runtime candidate event と同じpayload構造を受ける smoke-only 経路を `memory.py` に追加した。

この経路は、実runtimeのattention結果やscheduler判断には使わない。

```text
runtime candidate event payload
  -> run_host_backup_copy_candidate_for_smoke()
  -> runtime_policy_state を確認
  -> applied_candidate のみ read-only snapshot
  -> snapshot が存在する場合だけ host backup copy helper へ渡す
```

### applied_candidate

`runtime_policy_state == "applied_candidate"` の場合だけ、以下を行う。

```text
1. read-only KV snapshotを作成
2. snapshotが存在する場合だけ copy_host_backup_candidate_for_smoke() に渡す
3. host backup copy candidateとしてcopyを実行
4. copy_equal/source_mutatedなどをログ化
```

期待される状態。

```text
snapshot_created = true
host_backup_copy_candidate = true
host_backup_copy_executed = true
copy_equal = true
source_mutated = false
```

### fallback_candidate

`runtime_policy_state == "fallback_candidate"` の場合は必ずno-op。

```text
snapshot_created = false
host_backup_copy_candidate = false
host_backup_copy_executed = false
fallback_candidate_noop_guard = true
host_backup_copy_skipped_reason = "fallback_candidate_noop_guard"
```

## 重要な表現

今回、`host_backup_copy=false` は維持している。

これは、今回の経路が runtime本経路ではなく、candidate smoke 経路であることを明示するため。

一方で、candidate smoke内では以下が成立する。

```text
host_backup_copy_candidate = true
host_backup_copy_executed = true
```

つまり、意味は以下のように分かれる。

```text
host_backup_copy:
  実runtime本経路としてのhost backup copyではない

host_backup_copy_candidate:
  candidate/smoke経路としてcopy候補を実行した

host_backup_copy_executed:
  candidate/smoke内でcopyが実行された
```

## applied_candidate ログ

確認できたログ。

```json
{
  "runtime_policy_state": "applied_candidate",
  "snapshot_created": true,
  "host_backup_copy_candidate": true,
  "host_backup_copy_executed": true,
  "snapshot_shape": [4, 2, 2, 8],
  "backup_shape": [4, 2, 2, 8],
  "copy_numel": 128,
  "copy_nbytes": 256,
  "copy_equal": true,
  "source_mutated": false,
  "attention_override": false,
  "kv_cache_mutation": false,
  "runtime_writeback": false,
  "scheduler_policy_noop": true
}
```

## fallback_candidate ログ

確認できたログ。

```json
{
  "runtime_policy_state": "fallback_candidate",
  "snapshot_created": false,
  "host_backup_copy_candidate": false,
  "host_backup_copy_executed": false,
  "host_backup_copy_skipped_reason": "fallback_candidate_noop_guard",
  "fallback_candidate_noop_guard": true,
  "attention_override": false,
  "kv_cache_mutation": false,
  "runtime_writeback": false,
  "scheduler_policy_noop": true
}
```

## 安全条件

今回も以下を維持した。

```text
attention差し替えなし:
  attention_override = false

KV cache freeなし:
  GPU KV freeなし

KV pool書き換えなし:
  kv_cache_mutation = false
  source_mutated = false

runtime writebackなし:
  runtime_writeback = false

scheduler挙動変更なし:
  scheduler_policy_noop = true

scheduler.py変更なし
attention backend変更なし
.github/workflows差分なし
```

## 実行済み確認

py_compile。

```bash
.venv/bin/python -m py_compile \
  python/sglang/srt/managers/scheduler.py \
  python/sglang/srt/relaykv/memory.py \
  python/sglang/srt/relaykv/metrics.py \
  scripts/relaykv_runtime_policy_smoke.py \
  scripts/relaykv_actual_kv_pool_observation_smoke.py \
  scripts/relaykv_kv_pool_snapshot_smoke.py \
  scripts/relaykv_host_backup_copy_smoke.py
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

実行時に以下の既存環境由来warningあり。

```text
Can't initialize NVML
torch version warning
```

今回のsmoke検証自体はすべて成功。  
これらは今回の RelayKV host backup copy candidate path の正否には直接関係しない。

## 判断

今回の変更は、standalone smoke から一段進み、runtime candidate event payload と同じ構造から host backup copy candidate を実行できるようにしたもの。

到達点。

```text
以前:
  snapshot/copy は standalone smoke

今回:
  runtime candidate event payload と同じ構造から
  applied_candidate のみ snapshot -> host backup copy candidate が通る
```

重要な点。

```text
scheduler.py変更なし
attention backend変更なし
runtime writebackなし
KV pool mutationなし
fallback_candidate no-op維持
```

このため、まだ実モデル実行には影響を与えない安全な段階に留まっている。

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
```

## 次の候補

次に進むなら以下。

```text
RelayKV candidate copy summary / counters
```

目的。

```text
request/layer単位で、candidate copy path の発火状況と安全条件を集計する。
```

集計したい項目。

```text
applied_candidate count
fallback_candidate count
snapshot_created count
host_backup_copy_candidate count
host_backup_copy_executed count
copy_equal true count
source_mutated false count
fallback_candidate_noop_guard count
attention_override false count
kv_cache_mutation false count
runtime_writeback false count
scheduler_policy_noop true count
```

まだ禁止。

```text
attention接続
KV cache free
KV pool書き換え
runtime writeback
scheduler挙動変更
```

## 次スレッドへ進むタイミング

次スレッドへ進む候補は2つ。

### 候補A: ここで切る

今の状態は、実runtime接続前の安全な節目として非常に切りが良い。

理由。

```text
runtime candidate event payload -> host backup copy candidate まで到達
scheduler.py変更なし
attention未接続
KV freeなし
writebackなし
mutationなし
```

次スレッドで「candidate copy summary / counters」から始めやすい。

### 候補B: candidate copy summary / counters まで終えてから切る

次の小タスクである summary / counters は、まだ実runtime接続前の観測強化なので比較的安全。

これを終えると、次スレッドを「実runtime接続するかどうかの判断」から始められる。

おすすめは候補B。

理由。

```text
1. 次のsummary/countersは小さく安全
2. 実runtime接続前の判断材料が増える
3. 次スレッドを実runtime接続判断から始められる
4. 今の長い流れを、観測基盤完成という形で閉じやすい
```

ただし、会話が重い場合は候補Aで切ってよい。

## commit候補

```bash
cd ~/work/sglang-relaykv

git status --short
git diff --name-status
git diff --name-status | grep '.github/workflows' || true
git diff --check

git add \
  python/sglang/srt/relaykv/memory.py \
  python/sglang/srt/relaykv/metrics.py \
  scripts/relaykv_runtime_policy_smoke.py \
  notes/devlog_2026-05-01_relaykv_runtime_host_backup_copy_candidate_path_ja.md

git commit -m "Add RelayKV host backup copy candidate path"

git push mine relaykv-host-backup-shadow
```
