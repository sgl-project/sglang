# Devlog: RelayKV read-only KV pool snapshot helper consolidation

## 日付

2026-05-01

## 対象repo

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`

## 目的

RelayKV の actual KV pool observation / fake KV pool snapshot smoke で使っていた read-only snapshot 処理を、`memory.py` 側のhelperへ集約する。

今回の目的は、実 `MHATokenToKVPool` と fake pool fallback の両方で、同じ read-only snapshot helper を通せるようにすること。

今回も以下は行わない。

```text
attention接続
KV cache free
KV pool書き換え
runtime writeback
scheduler挙動変更
host backup runtime接続
```

## 背景

前段階で以下を確認済み。

```text
flashinfer import制約:
  FLASHINFER_WORKSPACE_BASE=/tmp/relaykv_flashinfer_cache で回避

実 MHATokenToKVPool:
  import OK
  instantiate OK
  observed_layout = mha_split_kv

layout:
  k_buffer: [tokens, heads, head_dim]
  v_buffer: [tokens, heads, head_dim]
  snapshot: [tokens, 2, heads, head_dim]

確認:
  copy_equal = true
  source_mutated = false
```

前回までは、actual KV pool observation smoke と fake KV pool snapshot smoke の中に、read-only snapshot 処理がやや分散していた。

今回はその処理を `memory.py` に寄せ、今後 runtime host backup copy candidate path に進む前の共通部品として整理した。

## 変更ファイル

```text
python/sglang/srt/relaykv/memory.py
scripts/relaykv_actual_kv_pool_observation_smoke.py
scripts/relaykv_kv_pool_snapshot_smoke.py
```

## 追加・整理したhelper

新規または整理したhelper。

```text
snapshot_mha_kv_pool_readonly_for_smoke()
```

既存helper。

```text
snapshot_kv_pool_for_host_backup_smoke()
```

既存helperは allocator から `kv_pool` を解決し、新helperへ委譲する形にした。

```text
snapshot_kv_pool_for_host_backup_smoke()
  -> allocator から kv_pool 解決
  -> snapshot_mha_kv_pool_readonly_for_smoke() に委譲
```

## helper仕様

対象layout。

```text
split K/V layout
kv_pool.k_buffer
kv_pool.v_buffer
```

想定shape。

```text
k_buffer: [tokens, heads, head_dim]
v_buffer: [tokens, heads, head_dim]
```

入力。

```text
kv_pool
layer_idx
token_indices
runtime_policy_state
```

処理。

```text
1. runtime_policy_state を確認
2. fallback_candidate なら snapshot を作らず no-op
3. applied_candidate なら k_buffer / v_buffer から token_indices を read-only で取得
4. K/Vを [tokens, 2, heads, head_dim] にstack
5. snapshot tensor と metadata を返す
6. source_mutated=false を確認できるmetadataを返す
```

metadataに含める主な項目。

```text
runtime_policy_state
snapshot_created
observed_layout
kv_pool_type
has_k_buffer
has_v_buffer
k_shape
v_shape
k_dtype
v_dtype
k_device
v_device
snapshot_shape
token_indices
source_mutated
kv_cache_mutation
attention_override
scheduler_policy_noop
fallback_candidate_noop_guard
```

## actual KV pool smoke結果

実 `MHATokenToKVPool` で、新helper経由の snapshot / copy が成功。

```json
{
  "kv_pool_type": "MHATokenToKVPool",
  "observed_layout": "mha_split_kv",
  "snapshot_created": true,
  "snapshot_shape": [4, 2, 2, 8],
  "backup_shape": [4, 2, 2, 8],
  "copy_equal": true,
  "source_mutated": false
}
```

## fake KV pool snapshot smoke結果

fake `MHATokenToKVPool` 相当の pool でも、同じhelper経由で成功。

```json
{
  "kv_pool_type": "_FakeMHATokenToKVPool",
  "observed_layout": "mha_split_kv",
  "snapshot_created": true,
  "copy_equal": true,
  "source_mutated": false
}
```

## fallback_candidate no-op

`fallback_candidate` では snapshot を作らない。

期待される状態。

```text
snapshot_created = false
host_backup_copy_executed = false
fallback_candidate_noop_guard = true
source_mutated = false
```

## 確認

実行した確認。

```bash
.venv/bin/python -m py_compile   python/sglang/srt/relaykv/memory.py   scripts/relaykv_actual_kv_pool_observation_smoke.py   scripts/relaykv_kv_pool_snapshot_smoke.py   scripts/relaykv_host_backup_copy_smoke.py
```

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_actual_kv_pool_observation_smoke.py
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
relaykv_actual_kv_pool_observation_smoke.py: pass
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
attention差し替え
KV cache free
KV pool書き換え
runtime writeback
scheduler挙動変更
host backup runtime接続
実server起動
実model起動
```

重要な不変条件。

```text
source_mutated = false
kv_cache_mutation = false
attention_override = false
scheduler_policy_noop = true
```

## 判断

今回の変更は、runtime host backup copy candidate path に進む前の共通部品化として妥当。

到達点。

```text
1. 実 MHATokenToKVPool と fake pool が同一helperで通る
2. split K/V layout の読み取り処理を memory.py に集約
3. [tokens, 2, heads, head_dim] snapshot の形を共通化
4. source_mutated=false を維持
5. fallback_candidate no-op を維持
6. attention / KV free / runtime writeback / scheduler挙動変更なし
```

この段階で、RelayKVは「実KV poolを安全に読む」ための最小helperを持った状態になった。

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
```

## 次の候補

次に進むなら以下。

```text
RelayKV runtime host backup copy candidate path
```

目的。

```text
scheduler / runtime candidate event から、
applied_candidate の場合だけ read-only snapshot -> host backup copy までを候補経路として接続する。
```

ただし、まだ以下は禁止。

```text
attention接続
KV cache free
KV pool書き換え
runtime writeback
scheduler挙動変更
```

最初のruntime candidate pathでは、以下を守る。

```text
applied_candidate:
  read-only snapshot
  host backup copy
  attention未接続
  KV freeなし
  scheduler挙動変更なし

fallback_candidate:
  必ず no-op
```

## commit候補

```bash
cd ~/work/sglang-relaykv

git status --short
git diff --name-status
git diff --name-status | grep '.github/workflows' || true
git diff --check

git add   python/sglang/srt/relaykv/memory.py   scripts/relaykv_actual_kv_pool_observation_smoke.py   scripts/relaykv_kv_pool_snapshot_smoke.py   notes/devlog_2026-05-01_relaykv_readonly_kv_pool_snapshot_helper_ja.md

git commit -m "Consolidate RelayKV read-only KV pool snapshot helper"

git push mine relaykv-host-backup-shadow
```
