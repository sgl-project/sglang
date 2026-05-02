# Devlog: RelayKV read-only ForwardBatch observation metadata

Date: 2026-05-02 JST  
Branch: `relaykv-host-backup-shadow`  
Repo: `~/work/sglang-relaykv`

## 今日の目的

RelayKV runtime observation hook で、`req_pool_idx` を含む read-only metadata を扱えるようにする。

前段階では、実server env on で `ForwardBatch` 既存 metadata から observation-only summary を出せていたが、`req_pool_idx` はまだなかった。

今回の目的は、`ForwardBatch.init_new()` に最小限の read-only metadata carrier を追加し、runtime observation payload candidate に `req_pool_idx` を含めること。

ただし、この段階ではまだ以下には進まない。

- KV pool 参照
- KV snapshot
- host backup copy
- attention 接続
- runtime writeback
- scheduler decision 変更
- batch 構成変更
- tensor construction 変更
- KV allocation 変更

## 背景

前段階までの到達点:

```text
env off:
  RelayKV observation logなし
  optional server smoke pass

env on:
  existing ForwardBatch metadata fallback summary出力
  source="forward_batch_existing_cpu_metadata_runtime_observation"
  seq_lens_cpu_value_source="cpu_tensor_observation_only"
  req_pool_idx_none=true
  optional server smoke pass
```

既に以下の helper / schema は固定済み。

### 明示CPU metadata schema

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

### 既存ForwardBatch metadata runtime fallback

```text
build_runtime_observation_payload_candidates_from_forward_batch_runtime_existing_metadata()
```

特徴:

```text
rids + seq_lens_cpu + extend_* から payload candidate を作る
req_pool_idx なし
req_pool_idx=None
実server env on で fallback summary確認済み
```

今回、この `req_pool_idx=None` の制約を越えるために、read-only metadata carrier を `ForwardBatch` に追加した。

## 見たファイル

- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/relaykv/observation.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `scripts/relaykv_optional_server_observation_smoke.py`
- 既存 observation / hook smoke 群

## 変更したファイル

- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/relaykv/observation.py`
- `scripts/relaykv_optional_server_observation_smoke.py`
- `scripts/relaykv_forward_batch_readonly_metadata_observation_smoke.py`

## ForwardBatch read-only metadata carrier

追加 field:

```text
relaykv_runtime_observation_metadata
```

作成条件:

```text
SGLANG_RELAYKV_RUNTIME_OBSERVATION == "1"
```

つまり、env off では metadata carrier を作らない。

## ForwardBatch.init_new() への変更

今回 `ForwardBatch.init_new()` に触った。

ただし、変更内容は env-on 時の read-only metadata carrier assignment のみ。

変更していないもの:

- scheduler decision
- batch 構成
- tensor construction
- KV allocation
- req pool allocation
- model forward computation
- attention backend
- memory pool
- runtime writeback

## metadata schema

`relaykv_runtime_observation_metadata` は request ごとの dict list。

必須 field:

```text
request_id
rid
request_index_in_batch
req_pool_idx
seq_len
source
```

optional field:

```text
extend_seq_len
extend_prefix_len
phase
```

source:

```text
source="forward_batch_readonly_runtime_observation_metadata"
```

## metadata 構築元

既存 Python-side 値のみを使用。

読む値:

```text
req.rid
req.req_pool_idx
req.seqlen
batch.extend_seq_lens
batch.extend_prefix_lens
```

読まない値:

```text
req_pool_indices CUDA tensor
seq_lens CUDA tensor
KV pool
memory pool
attention state
scheduler decision state
```

禁止していること:

```text
GPU tensor .cpu()
GPU tensor .tolist()
GPU tensor .item()
GPU tensor indexing
req_pool_indices tensor value read
seq_lens CUDA tensor value read
KV pool read
```

## observation hook fallback順

`run_model_runner_forward_observation_hook()` の fallback順を整理した。

```text
1. 既存 direct payload builder
2. relaykv_runtime_observation_metadata から readonly metadata payload
3. 既存 ForwardBatch existing metadata fallback, req_pool_idx=None
```

つまり、read-only metadata carrier があれば `req_pool_idx` あり summary を優先する。

metadata がない場合は、前段階の `req_pool_idx=None` fallback を維持する。

## readonly metadata summary

readonly metadata 成功時の summary log prefix:

```text
relaykv_runtime_observation_readonly_metadata_summary
```

summary の主要 field:

```text
source="forward_batch_readonly_runtime_observation_metadata"
req_pool_idx_none=false
total_payloads
per_request_counts
per_layer_counts
per_batch_counts
source_mutated_true_count=0
attention_override_true_count=0
kv_cache_mutation_true_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
```

## 新規 smoke

追加:

```text
scripts/relaykv_forward_batch_readonly_metadata_observation_smoke.py
```

確認したこと:

```text
fake ForwardBatch-like に relaykv_runtime_observation_metadata を持たせる
2 requests x 2 layers = 4 payloads
rid-a/rid-b
req_pool_idx 10/11
seq_len 128/256
safety counters all zero
req_pool_idx_none=false
existing metadata fallbackより優先される
metadataがない場合は既存 req_pool_idx=None fallbackへ落ちる
env off相当ではmetadata未読
poison tensor-like req_pool_indices / seq_lens は読まれない
```

## 確認結果

以下は pass。

```bash
PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/relaykv/observation.py \
  python/sglang/srt/model_executor/forward_batch_info.py \
  scripts/relaykv_forward_batch_readonly_metadata_observation_smoke.py

PYTHONPATH=python .venv/bin/python scripts/relaykv_forward_batch_readonly_metadata_observation_smoke.py
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
- `relaykv_forward_batch_readonly_metadata_observation_smoke.py`: pass
- `relaykv_runtime_cpu_metadata_payload_schema_smoke.py`: pass
- `relaykv_forward_batch_existing_metadata_payload_candidate_smoke.py`: pass
- `relaykv_forward_batch_runtime_existing_metadata_observation_smoke.py`: pass
- `relaykv_fake_model_runner_forward_observation_smoke.py`: pass
- `relaykv_model_runner_observation_hook_smoke.py`: pass
- `relaykv_runtime_observation_summary_smoke.py`: pass
- `relaykv_runtime_policy_smoke.py`: pass
- `relaykv_optional_server_observation_smoke.py`: model未設定 clean skip
- `relaykv_optional_server_observation_smoke.py`: local model pass

## optional server smoke local model 結果

local model path:

```text
/home/rinsa/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
```

env off:

```text
/generate 200
RelayKV observation logなし
```

env on:

```text
/generate 200
relaykv_readonly_metadata_summary_logged=true
relaykv_req_pool_idx_present_logged=true
relaykv_req_pool_idx_none_logged=false
```

これにより、実server env on で `req_pool_idx` あり summary が出ることを確認した。

## git確認

```text
git diff --check: pass
制約 grep: 出力なし
```

禁止領域への差分なし。

触っていない禁止領域:

- `scheduler.py`
- `model_runner.py`
- attention backend
- flashinfer
- `memory_pool.py`
- `.github/workflows`

## 現在の到達点

```text
env off:
  relaykv_runtime_observation_metadata は作らない
  RelayKV observation logなし
  /generate 200

env on:
  ForwardBatch.init_new() で read-only metadata carrier 作成
  observation hook が metadata を読む
  req_pool_idxあり summary 出力
  /generate 200
```

fallback:

```text
1. direct payload builder
2. relaykv_runtime_observation_metadata
3. existing ForwardBatch metadata fallback, req_pool_idx=None
```

これで、`req_pool_idx=None` の observation-only段階から、`req_pool_idx` ありの runtime observation段階に進んだ。

## 維持できている安全境界

以下は維持。

- env off では metadata carrier を作らない
- env on の read-only metadata assignment のみ
- scheduler decision は変えない
- batch 構成は変えない
- tensor construction は変えない
- KV allocation は変えない
- `req_pool_indices` CUDA tensor を読まない
- `seq_lens` CUDA tensor を読まない
- GPU tensor `.cpu()` / `.tolist()` / `.item()` なし
- KV pool 参照なし
- snapshotなし
- host backup copyなし
- attention接続なし
- runtime writebackなし
- hook例外で forward を止めない

## 判断

今回の変更は、RelayKV host-backup shadow 系の前段として重要。

理由:

```text
runtime observation hook が、
実server ForwardBatch 上の request_id / req_pool_idx / seq_len を
read-only summaryとして扱えるようになった
```

これにより、今後の host backup copy candidate summary と runtime request metadata を対応づける足場ができた。

ただし、まだ実際の host backup copy や KV pool 参照には進んでいない。

## 次の分岐

次に進むなら、いきなり host backup copy ではなく、まず以下が安全。

```text
read-only metadata と host backup copy candidate summary の join設計
```

目的:

```text
host backup copy candidate event
runtime observation metadata
req_pool_idx
request_id
layer_id
```

を summary 上で対応づけられるか確認する。

まだやらないこと:

```text
KV pool snapshot
host backup copy実行
attention接続
runtime writeback
scheduler decision変更
```

## 次段階のおすすめ

次は以下。

```text
RelayKV read-only observation metadata と
host backup copy candidate summary の join helper / fake smoke
```

理由:

- `req_pool_idx` が runtime metadataに入った
- しかしまだ KV pool / copy に進むには早い
- まず event同士を read-only summaryで対応づける方が安全
- 既存 `summarize_host_backup_copy_candidates_for_smoke()` と接続できる可能性がある
