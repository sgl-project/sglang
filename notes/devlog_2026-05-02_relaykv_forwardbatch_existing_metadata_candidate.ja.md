# Devlog: RelayKV ForwardBatch existing metadata payload candidate smoke

Date: 2026-05-02 JST  
Branch: `relaykv-host-backup-shadow`  
Repo: `~/work/sglang-relaykv`

## 今日の目的

RelayKV runtime observation の次段階に向けて、実 `ForwardBatch` / `ModelRunner` / scheduler に接続せず、既存 `ForwardBatch` 属性だけで payload candidate を作れるかを fake smoke で固定する。

この段階では、まだ以下には進まない。

- 実 `ForwardBatch` 接続
- `ForwardBatch.init_new()` 変更
- `ModelRunner.forward` 変更
- scheduler / `ScheduleBatch` / `Req` 変更
- tensor値の読み取り
- `.cpu()` / `.item()` / `.tolist()`
- KV pool 参照
- KV snapshot
- host backup copy
- attention 接続
- runtime writeback

## 背景

前段階までに、2つの事実が分かっている。

### 実server経路で観測済みの `ForwardBatch` metadata

```text
req_pool_indices:
  torch.Tensor
  device=cuda:0
  dtype=torch.int64

seq_lens:
  torch.Tensor
  device=cuda:0
  dtype=torch.int64

seq_lens_cpu:
  torch.Tensor
  device=cpu
  dtype=torch.int64
  shape=torch.Size([1])

rids:
  list

extend_seq_lens_cpu:
  list

extend_prefix_lens_cpu:
  list
```

### すでに固定済みの明示CPU metadata schema

前段階で `build_runtime_observation_cpu_metadata_payloads()` を追加し、明示的な CPU metadata list/tuple から runtime observation payload candidate を作る schema を固定した。

この schema は `req_pool_idx` を含む。

```text
rids
req_pool_indices_cpu
seq_lens_cpu
extend_seq_lens_cpu
extend_prefix_lens_cpu
layer_ids
```

ただし、これはまだ実 runtime には接続していない pure helper である。

## 今回の狙い

今回の目的は、`ForwardBatch.init_new()` を変更せず、既存 `ForwardBatch` 属性だけで payload candidate を作る schema を固定すること。

既存属性として使うもの:

- `rids`
- `seq_lens_cpu`
- `extend_seq_lens_cpu`
- `extend_prefix_lens_cpu`

既存 `ForwardBatch` 上には `req_pool_idx` の CPU list がない前提のため、今回の schema では以下とする。

```text
req_pool_idx = None
req_pool_index = None
```

## 変更したファイル

- `python/sglang/srt/relaykv/observation.py`
- `scripts/relaykv_forward_batch_existing_metadata_payload_candidate_smoke.py`

必要に応じて確認対象:

- `scripts/relaykv_runtime_cpu_metadata_payload_schema_smoke.py`

## 追加 helper

```text
build_runtime_observation_payload_candidates_from_forward_batch_existing_metadata()
```

### 責務

`ForwardBatch-like object` の既存属性だけを読む pure helper。

読む属性:

- `rids`
- `seq_lens_cpu`
- `extend_seq_lens_cpu`
- `extend_prefix_lens_cpu`

この helper は、まだ実 `ForwardBatch` / `ModelRunner` には接続しない。

## 固定した schema

payload candidate の schema:

```text
event_type="runtime_observation_forward_batch_existing_metadata_candidate"
batch_id
request_id
request_index_in_batch
request_index
req_pool_idx=None
req_pool_index=None
seq_len
optional extend_seq_len
optional extend_prefix_len
layer_id
phase
runtime_policy_state
source="forward_batch_existing_cpu_metadata"
source_mutated=false
attention_override=false
kv_cache_mutation=false
runtime_writeback=false
scheduler_policy_noop=true
```

## validation 方針

### 必須

- `rids` は list/tuple のみ
- `seq_lens_cpu` は list/tuple のみ
- `layer_ids` は list/tuple のみ
- `len(rids) == len(seq_lens_cpu)`
- optional `extend_seq_lens_cpu` は None または list/tuple
- optional `extend_prefix_lens_cpu` は None または list/tuple
- optional extend metadata がある場合は request数と長さ一致

### エラー

- list/tuple 以外は `TypeError`
- 長さ不一致は `ValueError`

### 禁止

以下は呼ばない。

- `.cpu()`
- `.item()`
- `.tolist()`
- `int(tensor)`
- `len(tensor-like)`
- `iter(tensor-like)`
- tensor indexing
- numpy conversion

fake smoke では list/tuple のみを許可し、実serverの `seq_lens_cpu` tensor値読み取りにはまだ進まない。

## fake smoke で固定した期待値

新規 smoke:

```text
scripts/relaykv_forward_batch_existing_metadata_payload_candidate_smoke.py
```

fake input:

```text
rids=["rid-a", "rid-b"]
seq_lens_cpu=[128, 256]
extend_seq_lens_cpu=[16, 32]
extend_prefix_lens_cpu=[112, 224]
layer_ids=[0, 14]
```

期待値:

```text
2 requests x 2 layers = 4 payloads

request_id:
  rid-a
  rid-b

request_index_in_batch:
  0
  1

request_index:
  0
  1

seq_len:
  128
  256

extend_seq_len:
  16
  32

extend_prefix_len:
  112
  224

req_pool_idx:
  None

req_pool_index:
  None

safety flags:
  source_mutated=false
  attention_override=false
  kv_cache_mutation=false
  runtime_writeback=false
  scheduler_policy_noop=true
```

追加確認:

- optional extend metadata なしでも pass
- length mismatch は `ValueError`
- poison tensor-like は `TypeError`
- poison tensor-like の `.cpu()` / `.item()` / `.tolist()` / `__iter__()` / `__len__()` / `__getitem__()` は未呼び出し
- 既存 `build_runtime_observation_cpu_metadata_payloads()` は壊れていない

## 確認結果

以下は pass。

```bash
PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/relaykv/observation.py \
  scripts/relaykv_runtime_cpu_metadata_payload_schema_smoke.py \
  scripts/relaykv_forward_batch_existing_metadata_payload_candidate_smoke.py

PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_cpu_metadata_payload_schema_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_forward_batch_existing_metadata_payload_candidate_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_fake_model_runner_forward_observation_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_model_runner_observation_hook_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_observation_summary_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_policy_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_optional_server_observation_smoke.py
```

結果:

- `py_compile`: pass
- `relaykv_runtime_cpu_metadata_payload_schema_smoke.py`: pass
- `relaykv_forward_batch_existing_metadata_payload_candidate_smoke.py`: pass
- `relaykv_fake_model_runner_forward_observation_smoke.py`: pass
- `relaykv_model_runner_observation_hook_smoke.py`: pass
- `relaykv_runtime_observation_summary_smoke.py`: pass
- `relaykv_runtime_policy_smoke.py`: pass
- `relaykv_optional_server_observation_smoke.py`: model env unset の clean skip

追加確認:

```bash
git diff --check
git diff --name-only | grep -E 'scheduler.py|attention|flashinfer|\.github/workflows' || true
```

結果:

- `git diff --check`: pass
- 制約 grep: 出力なし

## 触っていない領域

以下には差分なし。

- 実 `ForwardBatch`
- `ForwardBatch.init_new()`
- `ModelRunner.forward`
- `ModelRunner._forward_raw()`
- scheduler
- `ScheduleBatch`
- `Req`
- attention backend
- `memory_pool.py`
- flashinfer
- `.github/workflows`

## 維持できている安全境界

以下は維持。

- default-off runtime hook
- fake smoke only
- pure helper
- runtime未接続
- tensor値読み取りなし
- GPU同期なし
- KV pool参照なし
- snapshotなし
- host backup copyなし
- attention接続なし
- scheduler変更なし
- runtime writebackなし
- safety flags は全て安全値固定

## 現在の到達点

RelayKV runtime observation payload candidate schema は2本になった。

### A. 明示CPU metadata schema

helper:

```text
build_runtime_observation_cpu_metadata_payloads()
```

特徴:

```text
rids + req_pool_indices_cpu + seq_lens_cpu から payload candidate を作る
req_pool_idx あり
fake list/tuple のみ
runtime未接続
```

用途:

```text
将来、Req / ScheduleBatch から req_pool_idx を read-only metadata として持ち込む場合の本命 schema
```

### B. 既存ForwardBatch metadata schema

helper:

```text
build_runtime_observation_payload_candidates_from_forward_batch_existing_metadata()
```

特徴:

```text
rids + seq_lens_cpu + extend_* から payload candidate を作る
req_pool_idx なし
req_pool_idx=None
fake list/tuple のみ
runtime未接続
```

用途:

```text
ForwardBatch.init_new() を変更せず、既存属性だけで observation candidate を作れるかを見る schema
```

## 判断

今回の変更は安全な前進。

理由:

```text
実ForwardBatch / ModelRunner / scheduler 未接続
ForwardBatch.init_new() 未変更
seq_lens_cpu は fake list/tuple のみ
実server tensor値読み取りなし
req_pool_idx は None
safety flags は安全値固定
```

この段階で、将来接続する場合の payload形だけを固定できた。

## 次の分岐

次からは、実 runtime への接続方針を選ぶ必要がある。

候補:

### 1. req_pool_idx なしで observation-only runtime接続を試す

使う metadata:

```text
rids
seq_lens_cpu
extend_seq_lens_cpu
extend_prefix_lens_cpu
```

利点:

- `ForwardBatch.init_new()` を触らずに進められる可能性がある
- scheduler / Req / ScheduleBatch に触れない
- observation-only のまま前進できる

課題:

- 実 `seq_lens_cpu` は CPU tensor なので、値読み取り方針が必要
- `req_pool_idx` がないため、KV pool / req pool と直接対応しにくい
- host backup copy にはまだ不足

### 2. req_pool_idx を ForwardBatch へ read-only metadata として持ち込む設計に進む

使う metadata:

```text
req.rid
req.req_pool_idx
seq_len
```

利点:

- 将来の host backup copy / KV pool 対応に近い
- 明示CPU metadata schema と対応する

課題:

- `ForwardBatch.init_new()` 変更が必要
- hot path への影響評価が必要
- scheduler / Req / ScheduleBatch 由来 metadata の寿命・整合性を確認する必要がある

## 次段階のおすすめ

まだ host backup copy ではない。

次に進むなら、まずは以下が安全。

```text
req_pool_idxなしの observation-only runtime接続を設計する
```

ただし、すぐに実装する前に以下を決める。

```text
seq_lens_cpu の値読み取りをどこまで許可するか
CPU tensor の .tolist() を env/debug 限定で許可するか
値読み取りを fake smoke のみに留めるか
実server smoke では metadata description のままにするか
```

本命の host backup copy に進むには、最終的には `req_pool_idx` が必要になる可能性が高い。

そのため、次の実験は observation-only と割り切り、`req_pool_idx=None` のまま接続可否を見るのがよい。
