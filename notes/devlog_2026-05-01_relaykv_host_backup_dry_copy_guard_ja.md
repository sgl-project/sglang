# Devlog: RelayKV host backup dry-copy guard

## 日付

2026-05-01

## 対象repo

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`

## Codex実行環境メモ

今回時点で、Codex は `gpt-5.5 medium` になっている。

このため、以後のCodexタスクでは、従来より複数ファイル横断の設計確認や最小差分実装を任せやすい可能性がある。  
ただし、RelayKV/SGLang側では引き続き以下を必須条件とする。

```text
.github/workflows は触らない
attention backend は触らない
scheduler挙動変更は明示的なタスクまで禁止
KV cache freeは禁止
network downloadは禁止
```

## 目的

RelayKV の host backup dry-copy guard を metadata/log 経路に追加する。

今回の目的は、`applied_candidate` の場合だけ host backup dry-copy の「候補」に進めるようにし、`fallback_candidate` は必ず no-op にすること。

重要: 今回は **actual tensor copy はまだ行わない**。

## 背景

ここまでに SGLang側では以下を実装済み。

```text
1. RelayKV budget metadata
2. runtime_policy_state
3. policy summary counters
4. scheduler shadow candidate event log
5. fallback_candidate no-op guard
6. runtime policy smoke
```

PyTorch側 budget-first 実験の結論は以下。

```text
full KVがVRAMに乗る:
  RelayKV off

VRAM pressureあり:
  RelayKV shadow / applied_candidate

low coverage / high-risk条件:
  partial retrieval強行は危険
  fallback_candidate が必要
```

前回の疑似runtime smokeでは、以下を確認済み。

```text
applied_candidate:
  log only

fallback_candidate:
  no-op guard

kv_cache_mutation:
  false

attention_override:
  false

host_backup_copy:
  false
```

今回はその次の段階として、`applied_candidate` に限って dry-copy candidate をログ上で立てる。

## 変更ファイル

```text
python/sglang/srt/relaykv/memory.py
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_memory_smoke.py
scripts/relaykv_runtime_policy_smoke.py
```

## dry-copy guard仕様

### applied_candidate

`runtime_policy_state == "applied_candidate"` かつ host backup shadow / dry-copy / budget / copy target 条件が揃う場合のみ、dry-copy candidate を立てる。

```text
host_backup_dry_copy_candidate = true
host_backup_dry_copy_guard_ok = true
host_backup_dry_copy_reason = "dry_copy_candidate_metadata_only_no_tensor_copy"
```

ただし、actual copy は実行しない。

```text
host_backup_copy = false
host_backup_copy_executed = false
kv_cache_mutation = false
attention_override = false
scheduler_policy_noop = true
```

### fallback_candidate

`runtime_policy_state == "fallback_candidate"` の場合は、必ず no-op。

```text
host_backup_dry_copy_candidate = false
host_backup_dry_copy_guard_ok = false
host_backup_dry_copy_reason = "runtime_policy_fallback_candidate_noop"
fallback_candidate_noop_guard = true
```

この状態では host backup copy 候補にも進めない。

## applied_candidate ログ例

```json
{
  "runtime_policy_state": "applied_candidate",
  "runtime_policy_action": "dry_copy_candidate_log_only",
  "dry_copy_candidate": true,
  "host_backup_copy": false,
  "host_backup_copy_executed": false,
  "host_backup_copy_skipped_reason": "dry_copy_candidate_metadata_only_no_host_backup_copy",
  "attention_override": false,
  "kv_cache_mutation": false,
  "scheduler_policy_noop": true
}
```

## fallback_candidate ログ例

```json
{
  "runtime_policy_state": "fallback_candidate",
  "runtime_policy_action": "log_only_noop",
  "dry_copy_candidate": false,
  "fallback_candidate_noop_guard": true,
  "host_backup_copy": false,
  "host_backup_copy_executed": false,
  "host_backup_copy_skipped_reason": "fallback_candidate_noop_guard",
  "attention_override": false,
  "kv_cache_mutation": false
}
```

## smoke結果

実行した確認。

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_policy_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_memory_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_budget_sweep.py
```

結果。

```text
relaykv_runtime_policy_smoke.py: ok
relaykv_memory_smoke.py: ok
relaykv_budget_sweep.py: ok
```

candidate summary例。

```json
{
  "candidate_event_counts": {
    "applied_candidate": 1,
    "fallback_candidate": 2
  },
  "noop_guard_counts": {
    "dry_copy_candidate_true": 1,
    "fallback_candidate_noop_guard_true": 2,
    "host_backup_copy_false": 3,
    "host_backup_copy_executed_false": 3,
    "kv_cache_mutation_false": 3,
    "attention_override_false": 3,
    "scheduler_policy_noop_true": 3
  }
}
```

## 確認

```bash
.venv/bin/python -m py_compile   python/sglang/srt/relaykv/memory.py   python/sglang/srt/relaykv/metrics.py   scripts/relaykv_memory_smoke.py   scripts/relaykv_runtime_policy_smoke.py
```

```bash
git diff --name-status
git diff --name-status | grep '.github/workflows' || true
git diff --check
```

確認結果。

```text
py_compile: pass
git diff --check: pass
.github/workflows差分: なし
```

## 変更していないもの

今回も以下は変更していない。

```text
actual host backup tensor copy
attention差し替え
KV cache free
schedulerのスケジューリング挙動
KV cache poolの実挙動
```

特に重要な不変条件。

```text
host_backup_copy_executed = false
attention_override = false
kv_cache_mutation = false
scheduler_policy_noop = true
```

## 判断

今回の変更は、actual host backup copy に進む直前の安全な境界として妥当。

理由。

```text
1. applied_candidate のみ dry-copy candidate に進める
2. fallback_candidate は必ず no-op
3. actual copy はまだ実行しない
4. attentionには接続しない
5. GPU KV freeもしない
6. scheduler挙動も変えない
```

つまり、policy上は「copyしてよい候補」を識別できるようになったが、実際のtensor mutationはまだ起きない。

## 現在の到達点

SGLang側の RelayKV host backup / runtime policy 基盤は以下まで進んだ。

```text
1. budget metadata
2. runtime_policy_state
3. policy counters
4. scheduler shadow candidate event
5. fallback no-op guard
6. runtime policy smoke
7. host backup dry-copy guard
```

## 次の候補

次に進むなら、初めて actual host backup copy smoke に入る。

ただし、条件は厳しくする。

```text
fallback_candidate:
  必ず no-op

applied_candidate:
  actual host backup copy のみ候補
  attention接続は禁止
  KV cache freeは禁止
  scheduler挙動変更は禁止
```

推奨する次タスク。

```text
RelayKV host backup actual-copy smoke

目的:
  applied_candidate の場合だけ、小さなtensor / fake KV payload で host backup copy を実行する。
  ただし attentionには接続しない。
  GPU KV freeもしない。
  scheduler挙動も変えない。

確認:
  copy元/先shape
  dtype
  device
  copy count
  copy bytes
  copy_executed true
  attention_override false
  kv_cache_mutation false または explicit host_backup_only
```

## commit候補

```bash
cd ~/work/sglang-relaykv

git status --short
git diff --name-status
git diff --name-status | grep '.github/workflows' || true
git diff --check

git add   python/sglang/srt/relaykv/memory.py   python/sglang/srt/relaykv/metrics.py   scripts/relaykv_memory_smoke.py   scripts/relaykv_runtime_policy_smoke.py   notes/devlog_2026-05-01_relaykv_host_backup_dry_copy_guard_ja.md

git commit -m "Add RelayKV host backup dry-copy guard"

git push mine relaykv-host-backup-shadow
```
