# Devlog: RelayKV optional server observation smoke timeout stabilization

Date: 2026-05-02 JST  
Branch: `relaykv-host-backup-shadow`  
Repo: `~/work/sglang-relaykv`

## 今日の目的

RelayKV observation-only runtime fallback が実server env on で確認できたため、次の read-only metadata 持ち込み設計に進む前に、任意server smoke の timeout 判定を安定化する。

目的は、今後の回帰確認に使える optional server smoke にすること。

この段階では、まだ以下には進まない。

- `req_pool_idx` 持ち込み
- `ForwardBatch.init_new()` 変更
- scheduler / `ScheduleBatch` / `Req` 変更
- KV pool 参照
- KV snapshot
- host backup copy
- attention 接続
- runtime writeback

## 背景

前段階で、env on の実server経路では以下を確認済み。

```text
relaykv_runtime_observation_forward_batch_existing_metadata_summary が出る
seq_lens_cpu_value_source="cpu_tensor_observation_only"
req_pool_idx_none=true
/generate 200 OK
```

一方で env off case は、log tail 上では後から `/generate 200 OK` が出ているにもかかわらず、optional server smoke script 上では timeout failed になっていた。

このため、RelayKV本体の失敗ではなく、smoke側の timeout / log drain / 判定タイミングが厳しすぎると判断した。

## 今回の変更範囲

変更したファイル:

- `scripts/relaykv_optional_server_observation_smoke.py`

触っていないファイル:

- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `ForwardBatch.init_new()`
- scheduler
- `ScheduleBatch`
- `Req`
- attention backend
- `memory_pool.py`
- flashinfer
- `.github/workflows`
- RelayKV observation helper 本体

## 変更内容

### timeout設定の分離

`/generate` timeout を health/startup timeout から分離した。

追加 env:

```text
RELAYKV_OPTIONAL_SERVER_SMOKE_GENERATE_TIMEOUT
default: 120

RELAYKV_OPTIONAL_SERVER_SMOKE_GENERATE_TIMEOUT_GRACE
default: 20
```

### timeout後の log drain

`/generate` が timeout した場合でも、すぐに失敗扱いせず、少し待ってから server log を drain するようにした。

目的:

```text
timeout後に /generate 200 OK が出ていないか確認する
RelayKV logが出ているか確認する
timeoutの分類を可能にする
```

### result JSON の拡張

追加/整理した field:

```text
generate_timeout
generate_200_logged
timeout_classification
relaykv_existing_metadata_summary_logged
relaykv_cpu_tensor_value_source_logged
relaykv_req_pool_idx_none_logged
top-level timeouts
```

これにより、今後の smoke 失敗時に以下を切り分けやすくなった。

```text
server起動失敗
/generate timeout
/generate 200 OKは後から出た
RelayKV logが誤ってenv offで出た
env onでfallback summaryが出なかった
```

## pass 条件

### env off

env off では RelayKV observation log が一切出ないことを確認する。

期待値:

```text
relaykv_observation_logged=false
relaykv_skip_logged=false
relaykv_existing_metadata_summary_logged=false
relaykv_cpu_tensor_value_source_logged=false
relaykv_req_pool_idx_none_logged=false
/generate 200 OK
```

### env on

env on では observation-only fallback summary が出ることを確認する。

期待値:

```text
relaykv_observation_logged=true
relaykv_existing_metadata_summary_logged=true
relaykv_cpu_tensor_value_source_logged=true
relaykv_req_pool_idx_none_logged=true
forward_completed=true
/generate 200 OK
```

## optional server smoke 結果

### model未設定

model未設定では clean skip を維持。

```text
skipped=true
skip_reason=model_env_unset
```

### local model指定

local model指定で pass / exit 0。

model path:

```text
/home/rinsa/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
```

### env off 結果

```text
forward_completed=true
http_status=200
generate_200_logged=true

relaykv_observation_logged=false
relaykv_skip_logged=false
relaykv_existing_metadata_summary_logged=false
relaykv_cpu_tensor_value_source_logged=false
relaykv_req_pool_idx_none_logged=false
```

env off で RelayKV log は出ていない。

### env on 結果

```text
forward_completed=true
http_status=200
generate_200_logged=true

relaykv_observation_logged=true
relaykv_existing_metadata_summary_logged=true
relaykv_cpu_tensor_value_source_logged=true
relaykv_req_pool_idx_none_logged=true
```

env on で期待どおり observation-only fallback summary が検出された。

検出対象 prefix:

```text
relaykv_runtime_observation_forward_batch_existing_metadata_summary
```

## 既存 smoke 結果

以下はすべて pass。

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_cpu_metadata_payload_schema_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_forward_batch_existing_metadata_payload_candidate_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_forward_batch_runtime_existing_metadata_observation_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_fake_model_runner_forward_observation_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_model_runner_observation_hook_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_observation_summary_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_policy_smoke.py
```

## git確認

```text
git diff --check: pass
制約 grep: 出力なし
```

禁止領域への差分なし。

## 現在の到達点

```text
runtime observation:
  env offでは完全に沈黙
  env onでは実server経路で fallback summary 出力

optional server smoke:
  model未設定 clean skip
  local model指定 pass / exit 0
  env off pass
  env on pass
  /generate 200 OK検出
  timeout診断fieldあり
```

## 維持できている安全境界

以下は維持。

- `req_pool_idx` はまだ持ち込まない
- `ForwardBatch.init_new()` は変更しない
- `ModelRunner.forward` は変更しない
- scheduler は変更しない
- `req_pool_indices` は読まない
- `seq_lens` CUDA tensor は読まない
- GPU tensor `.cpu()` / `.tolist()` / `.item()` はしない
- KV pool 参照なし
- snapshotなし
- host backup copyなし
- attention接続なし
- runtime writebackなし

## 判断

今回の変更で、optional server smoke は次段階の回帰確認に使いやすくなった。

特に重要なのは以下。

```text
env off:
  RelayKV observation logが出ないことを検査可能

env on:
  observation-only fallback summaryが出ることを検査可能

/generate:
  timeoutと200 OKの判定を分離
```

これにより、次に `req_pool_idx` read-only metadata 持ち込みを設計する際、runtime observation の既存挙動が壊れていないか確認しやすくなった。

## 次の分岐

次は大きな設計分岐に進める。

### A. req_pool_idx=None のまま observation-only runtime summary をさらに安定化

現在は既にかなり安定化したため、ここで一旦十分。

### B. req_pool_idx を ForwardBatch へ read-only metadata として持ち込む設計へ進む

次に進むならこちら。

目的:

```text
Req.req_pool_idx / req.rid / seq_len を read-only metadata として ForwardBatch に持ち込む
```

理由:

```text
host backup copy / KV pool対応には req_pool_idx が必要になる可能性が高い
明示CPU metadata schema と接続できる
```

ただし、次段階でもまだ host backup copy には進まない。

次段階でやるべきこと:

```text
ForwardBatch.init_new() への最小 read-only metadata carrier 設計
fake smoke
runtime observation helperへの接続
optional server smokeで回帰確認
```

まだやらないこと:

```text
KV pool snapshot
host backup copy
attention接続
runtime writeback
scheduler decision変更
```
