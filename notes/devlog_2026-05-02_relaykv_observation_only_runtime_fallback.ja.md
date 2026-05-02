# Devlog: RelayKV observation-only ForwardBatch existing metadata runtime fallback

Date: 2026-05-02 JST  
Branch: `relaykv-host-backup-shadow`  
Repo: `~/work/sglang-relaykv`

## 今日の目的

RelayKV runtime observation hook に、`req_pool_idx` なしの observation-only fallback を接続し、実server経路で `ForwardBatch` 既存 metadata から summary log が出ることを確認する。

この段階では、まだ以下には進まない。

- `req_pool_indices` の値読み取り
- `seq_lens` CUDA tensor の値読み取り
- `ForwardBatch.init_new()` 変更
- `ModelRunner.forward` 変更
- scheduler / `ScheduleBatch` / `Req` 変更
- KV pool 参照
- KV snapshot
- host backup copy
- attention 接続
- runtime writeback

## 背景

前段階で、以下の pure helper / fake smoke を固定済み。

### 1. 明示CPU metadata schema

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

### 2. 既存ForwardBatch metadata schema

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

実serverでは以下が観測済み。

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

## 今回の実装

### 追加 / 拡張 helper

```text
build_runtime_observation_payload_candidates_from_forward_batch_runtime_existing_metadata()
```

### 責務

`run_model_runner_forward_observation_hook()` の env on 分岐内で、既存 `build_runtime_observation_payloads()` が `TypeError` / `ValueError` になった場合だけ fallback として実行する observation-only helper。

見る属性:

- `rids`
- `seq_lens_cpu`
- `extend_seq_lens_cpu`
- `extend_prefix_lens_cpu`

見ない属性:

- `req_pool_indices`
- `seq_lens`
- KV pool
- memory pool
- scheduler state
- attention state

## CPU tensor `seq_lens_cpu` の扱い

今回初めて、実 runtime hook 内で `seq_lens_cpu` の値読み取りを限定的に許可した。

許可条件:

```text
env on
observation-only helper内
device.type == "cpu"
1D shape
len(rids) と一致
int64-like dtype
```

記録する field:

```text
seq_lens_cpu_value_source="cpu_tensor_observation_only"
```

禁止:

```text
GPU tensor .cpu()
GPU tensor .tolist()
GPU tensor .item()
GPU tensor indexing
req_pool_indices の値読み取り
seq_lens CUDA tensor の値読み取り
```

`seq_lens_cpu` は CPU tensor として既に存在するため、GPU同期回避の観点では `seq_lens` / `req_pool_indices` を読むより安全と判断した。

## payload / summary の仕様

fallback payload source:

```text
source="forward_batch_existing_cpu_metadata_runtime_observation"
```

`req_pool_idx` はまだ持たない。

```text
req_pool_idx=None
req_pool_index=None
req_pool_idx_none=true
```

summary log prefix:

```text
relaykv_runtime_observation_forward_batch_existing_metadata_summary
```

summary に含める主要 field:

```text
forward_pass_id
initial_skip_reason
total_payloads
per_batch_counts
per_layer_counts
per_request_counts
seq_lens_cpu_value_source
req_pool_idx_none
source
source_mutated_true_count
attention_override_true_count
kv_cache_mutation_true_count
runtime_writeback_true_count
scheduler_policy_noop_false_count
```

## 変更したファイル

- `python/sglang/srt/relaykv/observation.py`
- `scripts/relaykv_optional_server_observation_smoke.py`
- `scripts/relaykv_forward_batch_runtime_existing_metadata_observation_smoke.py`

## smoke 結果

以下は pass。

```bash
PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/relaykv/observation.py \
  scripts/relaykv_forward_batch_existing_metadata_payload_candidate_smoke.py \
  scripts/relaykv_forward_batch_runtime_existing_metadata_observation_smoke.py

PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_cpu_metadata_payload_schema_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_forward_batch_existing_metadata_payload_candidate_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_forward_batch_runtime_existing_metadata_observation_smoke.py
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
- `relaykv_forward_batch_runtime_existing_metadata_observation_smoke.py`: pass
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

## 新規 runtime smoke で固定した期待値

`relaykv_forward_batch_runtime_existing_metadata_observation_smoke.py` で確認。

入力:

```text
rids=["rid-a", "rid-b"]
seq_lens_cpu=torch.tensor([128, 256], dtype=torch.int64)
extend_seq_lens_cpu=[16, 32]
extend_prefix_lens_cpu=[112, 224]
layer_ids=[0, 14]
```

期待値:

```text
2 requests x 2 layers = 4 payloads
req_pool_idx=None
req_pool_index=None
seq_lens_cpu_value_source="cpu_tensor_observation_only"
safety counters all zero
```

追加確認:

- GPU tensor-like poison は `TypeError`
- `req_pool_indices` / `seq_lens` poison object は読まれない
- env off 相当では metadata 未読
- `.cpu()` / `.item()` / `.tolist()` / `__iter__()` / `__len__()` / `__getitem__()` の禁止メソッドは呼ばれない

## 任意server smoke 結果

使用 model path:

```text
/home/rinsa/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
```

実行コマンド:

```bash
RELAYKV_OPTIONAL_SERVER_SMOKE_MODEL=/home/rinsa/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 \
RELAYKV_OPTIONAL_SERVER_SMOKE_RUN=1 \
PYTHONPATH=python .venv/bin/python scripts/relaykv_optional_server_observation_smoke.py
```

### env on case

env on では、期待していた fallback summary が出た。

```text
relaykv_runtime_observation_forward_batch_existing_metadata_summary=...
```

検出結果:

```text
forward_completed=true
has_response=true
http_status=200
relaykv_observation_logged=true
relaykv_existing_metadata_summary_logged=true
relaykv_cpu_tensor_value_source_logged=true
relaykv_req_pool_idx_none_logged=true
```

summary の内容:

```text
source="forward_batch_existing_cpu_metadata_runtime_observation"
seq_lens_cpu_value_source="cpu_tensor_observation_only"
req_pool_idx_none=true

source_mutated_true_count=0
attention_override_true_count=0
kv_cache_mutation_true_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
```

実server log では、複数 forward pass に対して summary が出た。

```text
forward_pass_id=1
forward_pass_id=2
...
forward_pass_id=9
```

各 summary は `total_payloads=1`, `per_layer_counts={"0": 1}` の observation-only summary だった。

### env off case

env off では smoke script 上は timeout 扱いになった。

ただし log tail では最終的に `/generate` は `200 OK` まで到達している。

また、RelayKV log は出ていない。

```text
relaykv_observation_logged=false
relaykv_skip_logged=false
relaykv_existing_metadata_summary_logged=false
relaykv_cpu_tensor_value_source_logged=false
relaykv_req_pool_idx_none_logged=false
```

したがって、env off timeout は RelayKV hook 起因ではなく、optional server smoke の timeout閾値または待ち方の問題と判断する。

## 現在の到達点

```text
env off:
  RelayKV observation logなし
  optional smoke timeout判定はやや厳しい

env on:
  実server経路で fallback summary 出力
  CPU tensor seq_lens_cpu を observation-only で読めた
  req_pool_idx=None
  safety counters all safe
  /generate 200 OK
```

## 触っていない領域

以下には差分なし。

- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
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

- env off では helper未実行
- env on の observation hook 内 fallback のみ
- observation-only
- summary log のみ
- `req_pool_idx=None`
- `req_pool_indices` を読まない
- `seq_lens` CUDA tensor を読まない
- GPU tensor `.cpu()` / `.tolist()` / `.item()` なし
- KV pool 参照なし
- snapshotなし
- host backup copyなし
- attention接続なし
- scheduler変更なし
- runtime writebackなし
- hook例外でforwardを止めない

## 判断

今回の変更は、RelayKV runtime observation が「hook到達・skip」から一段進み、実server経路で `ForwardBatch` 既存CPU metadataを使った observation-only summary まで出せたという意味で重要。

特に重要なのは次の点。

```text
seq_lens_cpu:
  CPU tensorとして実ForwardBatch上に存在
  env on observation-only helper内で値を読める

req_pool_idx:
  まだ None
  req_pool_indices は読んでいない
```

これにより、`req_pool_idx` なしであれば、`ForwardBatch.init_new()` を変更せずに runtime observation summary を出せることが確認できた。

ただし、host backup copy / KV pool対応に進むには、最終的に `req_pool_idx` が必要になる可能性が高い。

## 次の分岐

次は2択。

### A. req_pool_idx=None のまま observation-only runtime summary を安定化

内容:

```text
optional server smoke の timeout判定を改善
env off/on の判定を安定化
fallback summaryのログ検出を明確化
```

利点:

- まだ `ForwardBatch.init_new()` に触れない
- runtime observation の足場を安定化できる
- 失敗原因の切り分けが楽になる

課題:

- host backup copy にはまだ進めない
- req_pool_idx は None のまま

### B. req_pool_idx を ForwardBatch へ read-only metadata として持ち込む設計へ進む

内容:

```text
Req.req_pool_idx / req.rid / seq_len を read-only metadata として ForwardBatch に持ち込む設計
```

利点:

- 将来の host backup copy / KV pool対応に近い
- 明示CPU metadata schema と接続できる

課題:

- `ForwardBatch.init_new()` に触る必要がある
- hot path への影響評価が必要
- scheduler / ScheduleBatch / Req 側の整合性確認が必要

## 次段階のおすすめ

次はまだ host backup copy ではない。

おすすめは A。

```text
req_pool_idx=None のまま observation-only runtime summary を安定化
```

理由:

- env on の本命確認はできた
- env off timeout判定が残っている
- ここを安定化してから `ForwardBatch.init_new()` に進む方が安全
- 任意server smoke が安定すれば、以後の read-only metadata持ち込みの回帰確認に使いやすくなる

その後に B へ進む。
