# Devlog: RelayKV KV pool snapshot smoke

## 日付

2026-05-01

## 対象repo

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`

## 目的

RelayKV host backup copy の次段階として、KV pool layout から read-only snapshot を作成し、host backup copy helper に渡せることを確認する。

今回は **実server / 実model / 実SGLang KV pool は起動しない**。  
fake pool fallback を使い、SGLang の `MHATokenToKVPool` 相当のlayoutで snapshot / copy の形だけを確認する。

## 背景

前段階では、以下を確認済み。

```text
applied_candidate:
  fake KV tensor を CPU host backup buffer に actual copy
  copy_equal = true

fallback_candidate:
  no-op
  host_backup_copy_executed = false

共通:
  attention_override = false
  kv_cache_mutation = false
  scheduler_policy_noop = true
```

今回はその次として、単なるfake tensorではなく、KV pool layout相当の `k_buffer` / `v_buffer` から token範囲を取り出し、RelayKV host backup copy helper に渡す smoke を追加した。

## 変更ファイル

```text
python/sglang/srt/relaykv/memory.py
scripts/relaykv_kv_pool_snapshot_smoke.py
```

## 実装方針

今回は実server / model起動は行わず、fake pool fallback にした。

理由。

```text
1. network download不要を維持するため
2. model起動やGPU状態への依存を避けるため
3. まずはKV layout shape / stack / copyの安全性を確認するため
4. 実KV poolへの接続前に source_mutated=false を確認するため
```

fake pool は SGLang の `MHATokenToKVPool` layout に合わせ、layerごとに以下を持つ形にした。

```text
k_buffer: [tokens, heads, head_dim]
v_buffer: [tokens, heads, head_dim]
```

## snapshot仕様

### applied_candidate

`runtime_policy_state == "applied_candidate"` の場合のみ snapshot を作成する。

処理。

```text
1. k_buffer / v_buffer から指定 token_indices をread-onlyで取り出す
2. K/Vを [tokens, 2, heads, head_dim] にstackする
3. snapshotを copy_host_backup_candidate_for_smoke() に渡す
4. CPU host backup bufferへcopyする
5. copy_equal と source_mutated を確認する
```

### fallback_candidate

`runtime_policy_state == "fallback_candidate"` の場合は必ず no-op。

```text
snapshot_created = false
host_backup_copy_executed = false
fallback_candidate_noop_guard = true
copy_numel = 0
copy_nbytes = 0
```

## applied_candidate ログ例

```json
{
  "runtime_policy_state": "applied_candidate",
  "kv_pool_type": "_FakeMHATokenToKVPool",
  "snapshot_created": true,
  "source_shape": [8, 2, 8],
  "snapshot_shape": [4, 2, 2, 8],
  "backup_shape": [4, 2, 2, 8],
  "token_indices": [2, 3, 4, 5],
  "copy_numel": 128,
  "copy_nbytes": 256,
  "copy_equal": true,
  "source_mutated": false,
  "kv_cache_mutation": false,
  "attention_override": false,
  "scheduler_policy_noop": true
}
```

## fallback_candidate ログ例

```json
{
  "runtime_policy_state": "fallback_candidate",
  "snapshot_created": false,
  "host_backup_copy_executed": false,
  "fallback_candidate_noop_guard": true,
  "copy_numel": 0,
  "copy_nbytes": 0,
  "copy_equal": false,
  "source_mutated": false
}
```

## 確認結果

実行した確認。

```bash
.venv/bin/python -m py_compile   python/sglang/srt/relaykv/memory.py   scripts/relaykv_kv_pool_snapshot_smoke.py
```

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_kv_pool_snapshot_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_host_backup_copy_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_policy_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_memory_smoke.py
```

差分確認。

```bash
git diff --check
git diff --name-status
git diff --name-status | grep '.github/workflows' || true
```

結果。

```text
py_compile: pass
relaykv_kv_pool_snapshot_smoke.py: pass
relaykv_host_backup_copy_smoke.py: pass
relaykv_runtime_policy_smoke.py: pass
relaykv_memory_smoke.py: pass
git diff --check: pass
.github/workflows差分: なし
```

## 変更していないもの

今回も以下は変更していない。

```text
実server起動
実model起動
実SGLang KV pool接続
attention差し替え
KV cache free
KV pool書き換え
runtime writeback
scheduler挙動変更
```

特に重要な不変条件。

```text
source_mutated = false
kv_cache_mutation = false
attention_override = false
scheduler_policy_noop = true
```

## 判断

今回の変更は、fake tensor copy から一段進めて、KV pool layout相当の snapshot / copy を確認するものとして妥当。

到達点。

```text
fake tensor host backup copy:
  完了

fake MHATokenToKVPool layout:
  完了

read-only snapshot:
  完了

snapshot -> host backup copy:
  完了

fallback no-op:
  維持
```

ただし、まだ実SGLang KV poolには接続していない。

このため、次に進む場合もまだ attention / writeback / KV free には進まず、実KV poolを read-only で観測する段階に留めるのが安全。

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
```

## 次の候補

次に進むなら以下。

```text
RelayKV actual KV pool read-only observation smoke
```

目的。

```text
実SGLang KV pool layoutをread-onlyで観測し、
fake pool smokeで想定した shape / token index / K/V layout と一致するか確認する。
```

まだ禁止。

```text
attention接続
KV cache free
KV pool書き換え
runtime writeback
scheduler挙動変更
```

見るべき項目。

```text
実KV pool type
k_buffer / v_buffer の有無
shape
dtype
device
token index mapping
read-only snapshot可否
source_mutated=false
```

## commit候補

```bash
cd ~/work/sglang-relaykv

git status --short
git diff --name-status
git diff --name-status | grep '.github/workflows' || true
git diff --check

git add   python/sglang/srt/relaykv/memory.py   scripts/relaykv_kv_pool_snapshot_smoke.py   notes/devlog_2026-05-01_relaykv_kv_pool_snapshot_smoke_ja.md

git commit -m "Add RelayKV KV pool snapshot smoke"

git push mine relaykv-host-backup-shadow
```
