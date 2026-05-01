# Devlog: RelayKV actual KV pool observation smoke

## 日付

2026-05-01

## 対象repo

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`

## 目的

RelayKV の host backup / snapshot 経路を、実SGLang KV pool layout に近づけるため、actual KV pool class の import / instantiate 可否を確認する。

ただし、今回も以下は行わない。

```text
attention接続
KV cache free
KV pool書き換え
runtime writeback
scheduler挙動変更
実server起動
実model起動
network download
```

## 背景

前段階では、fake `MHATokenToKVPool` layout を使って以下を確認した。

```text
k_buffer: [tokens, heads, head_dim]
v_buffer: [tokens, heads, head_dim]

snapshot:
  [tokens, 2, heads, head_dim]

applied_candidate:
  read-only snapshot作成
  host backup copy
  copy_equal = true
  source_mutated = false

fallback_candidate:
  snapshot/copyともno-op
```

今回は、SGLang側の実KV pool classである `MHATokenToKVPool` の import / instantiate を試み、実layout観測へ進めるか確認した。

## 変更ファイル

```text
scripts/relaykv_actual_kv_pool_observation_smoke.py
```

## 実KV pool class import結果

`MHATokenToKVPool` の import を試みた。

結果として、この環境では import 中に `flashinfer` が以下のようなpathへ JIT log を書こうとして失敗した。

```text
/home/rinsa/.cache/flashinfer/.../flashinfer_jit.log
```

失敗理由。

```text
Read-only file system
```

そのため、今回のsmokeでは以下のように扱った。

```text
kv_pool_import_ok = false
kv_pool_instantiate_ok = false
```

失敗理由はログ化し、実 `MHATokenToKVPool` と同じ split K/V layout の duck-typed pool に fallback した。

## fallback方針

import失敗を握りつぶさず、以下をログに残す。

```text
kv_pool_import_ok
kv_pool_instantiate_ok
kv_pool_import_error_type
kv_pool_import_error
fallback_used
```

そのうえで、実KV pool layout相当の duck-typed fallback を使う。

fallback layout。

```text
kv_pool_type: _ObservedMHATokenToKVPool
observed_layout: mha_split_kv

k_buffer: [tokens, heads, head_dim]
v_buffer: [tokens, heads, head_dim]
```

## 観測したKV layout

```json
{
  "kv_pool_type": "_ObservedMHATokenToKVPool",
  "observed_layout": "mha_split_kv",
  "has_k_buffer": true,
  "has_v_buffer": true,
  "k_shape": [9, 2, 8],
  "v_shape": [9, 2, 8],
  "k_dtype": "torch.float16",
  "v_dtype": "torch.float16",
  "k_device": "cpu",
  "v_device": "cpu",
  "layer_idx": 0,
  "token_indices": [2, 3, 4, 5]
}
```

## applied_candidate snapshot/copy

```json
{
  "runtime_policy_state": "applied_candidate",
  "snapshot_created": true,
  "snapshot_shape": [4, 2, 2, 8],
  "backup_shape": [4, 2, 2, 8],
  "copy_numel": 128,
  "copy_nbytes": 256,
  "copy_equal": true,
  "source_mutated": false,
  "kv_cache_mutation": false,
  "attention_override": false,
  "scheduler_policy_noop": true
}
```

## fallback_candidate no-op

```json
{
  "runtime_policy_state": "fallback_candidate",
  "snapshot_created": false,
  "host_backup_copy_executed": false,
  "fallback_candidate_noop_guard": true,
  "copy_equal": false,
  "source_mutated": false
}
```

## 確認結果

実行した確認。

```bash
.venv/bin/python -m py_compile   scripts/relaykv_actual_kv_pool_observation_smoke.py
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

今回の結果は、実KV pool観測へ進む前の環境制約を明確にできたという意味で重要。

到達点。

```text
1. MHATokenToKVPool importを試みた
2. flashinfer JIT log書き込みで失敗した
3. Read-only file system が原因
4. 失敗理由をログ化した
5. duck-typed split K/V layoutにfallbackした
6. fallback layoutでread-only snapshotを作成した
7. host backup copyに成功した
8. source_mutated=falseを確認した
9. fallback_candidate no-opを維持した
```

## 現在のlayout理解

現時点のRelayKV側想定layout。

```text
observed_layout = mha_split_kv

k_buffer:
  [tokens, heads, head_dim]

v_buffer:
  [tokens, heads, head_dim]

snapshot:
  [tokens, 2, heads, head_dim]
```

この理解は、fake pool smoke / duck-typed observation smoke の両方で整合している。

## 注意点

`MHATokenToKVPool` の import が失敗した原因は、RelayKV実装そのものではなく、環境側の `flashinfer` cache/log path 書き込み制約。

```text
/home/rinsa/.cache/flashinfer/.../flashinfer_jit.log
Read-only file system
```

実server smokeや実runtime pool接続へ進む前に、この制約を解消または回避する必要がある。

## 次の候補

次に進むなら、以下。

```text
RelayKV flashinfer cache/log path workaround smoke
```

目的。

```text
flashinfer import時の cache/log 書き込み先を /tmp など writable path に逃がせるか確認する。
```

まだ禁止。

```text
attention接続
KV cache free
KV pool書き換え
runtime writeback
scheduler挙動変更
実server接続
```

確認したい項目。

```text
flashinfer cache/log path
writable pathへ変更できるか
MHATokenToKVPool importが通るか
import失敗時の理由が明確か
fallback pathが壊れていないか
```

## commit候補

```bash
cd ~/work/sglang-relaykv

git status --short
git diff --name-status
git diff --name-status | grep '.github/workflows' || true
git diff --check

git add   scripts/relaykv_actual_kv_pool_observation_smoke.py   notes/devlog_2026-05-01_relaykv_actual_kv_pool_observation_smoke_ja.md

git commit -m "Add RelayKV actual KV pool observation smoke"

git push mine relaykv-host-backup-shadow
```
## 追記: flashinfer cache/log path workaround

追加調査により、`flashinfer/jit/env.py` は import 時に `FLASHINFER_WORKSPACE_BASE` を読むことが分かった。

default は `Path.home()` であり、この環境では `/home/rinsa/.cache/flashinfer/.../flashinfer_jit.log` への書き込みが `Read-only file system` で失敗していた。

`XDG_CACHE_HOME` や `FLASHINFER_CACHE_DIR` では回避できず、以下が有効だった。

```bash
FLASHINFER_WORKSPACE_BASE=/tmp/relaykv_flashinfer_cache
