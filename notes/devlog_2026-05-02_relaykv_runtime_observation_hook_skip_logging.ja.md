# Devlog: RelayKV runtime observation hook skip logging

Date: 2026-05-02 JST  
Branch: `relaykv-host-backup-shadow`  
Repo: `~/work/sglang-relaykv`

## 今日の目的

RelayKV runtime observation hook について、実server / 実モデル経路で以下を確認する。

- env off では RelayKV observation log が出ないこと
- env on では hook に到達すること
- 実 `ForwardBatch` の tensor-like metadata を無理に list 化しないこと
- payload 生成できない場合は skip として明示的に記録すること
- host backup copy / KV snapshot / attention 接続にはまだ進まないこと

## 背景

前段階で `scripts/relaykv_optional_server_observation_smoke.py` を追加し、ローカル既存小モデル限定の任意実server smoke を作成した。

この smoke は以下の安全条件を持つ。

- `RELAYKV_OPTIONAL_SERVER_SMOKE_MODEL` 未設定なら clean skip
- model path が存在しないなら clean skip
- `RELAYKV_OPTIONAL_SERVER_SMOKE_RUN != "1"` なら clean skip
- 実行時は `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
- `SGLANG_RELAYKV_RUNTIME_OBSERVATION=0` と `1` の両方を確認
- model download は必須にしない
- server 起動は標準確認にしない

## 発生していた問題

任意server smoke 自体は env off/on で server forward を壊さず通ったが、結果では以下が false だった。

```text
relaykv_skip_logged=false
relaykv_summary_logged=false
```

このため、実server経路で RelayKV observation hook に到達したかどうかがログ上は判別できなかった。

## 原因の推定

原因は以下。

1. hook skip が debug log だった
2. optional smoke 側の検出文字列と一致していなかった
3. 実 `ForwardBatch` では `req_pool_indices` / `seq_lens` が tensor-like である
4. そのため env on では payload 生成ではなく `TypeError` skip になるのが自然

つまり、今回必要なのは payload 生成ではなく、実server経路で「hook に到達し、同期なしで skip した」ことを明示的に観測することだった。

## 変更したファイル

- `python/sglang/srt/relaykv/observation.py`
- `scripts/relaykv_optional_server_observation_smoke.py`

## 実装内容

### 1. skip log helper の追加

`python/sglang/srt/relaykv/observation.py` に `log_runtime_observation_skip()` を追加した。

env on で observation payload を作れず skip した場合に、以下のような warning log を出す。

```text
relaykv_runtime_observation_skip={...}
```

主な payload:

- `forward_pass_id`
- `reason`

実server経路では以下を検出した。

```text
relaykv_runtime_observation_skip={"forward_pass_id": 1, "reason": "TypeError"}
```

### 2. optional server smoke の log detection 改善

`scripts/relaykv_optional_server_observation_smoke.py` で以下を検出するように変更した。

- `relaykv_runtime_observation_summary`
- `relaykv_runtime_observation_skip`

env off では RelayKV observation log が出たら失敗。

env on では、summary または skip のどちらも検出できなければ失敗。

### 3. readiness check の変更

readiness check は、生成を走らせる `/health_generate` ではなく `/v1/models` に変更した。

目的は、hook 到達確認前の generate / health_generate 由来の副作用や不安定性を減らすこと。

### 4. optional smoke 起動オプションの調整

任意server smoke 起動時に以下を追加した。

- `--disable-cuda-graph`
- `--disable-piecewise-cuda-graph`
- `--disable-overlap-schedule`

目的は、到達確認前の JIT / cuda graph / overlap schedule 系の失敗を減らすこと。

## 確認結果

### 通過した確認

以下は pass。

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_optional_server_observation_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_fake_model_runner_forward_observation_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_model_runner_observation_hook_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_observation_summary_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_policy_smoke.py
```

追加確認:

```bash
PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/relaykv/observation.py \
  scripts/relaykv_optional_server_observation_smoke.py

git diff --check
git diff --name-only | grep -E 'scheduler.py|attention|flashinfer|\.github/workflows' || true
```

結果:

- model未設定 clean skip: pass
- fake forward observation smoke: pass
- model runner observation hook smoke: pass
- runtime observation summary smoke: pass
- runtime policy smoke: pass
- py_compile: pass
- git diff --check: pass
- 制約 grep: 出力なし

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

結果:

- env off case
  - RelayKV observation log なし
  - default-off 維持

- env on case
  - 実server経路で hook 到達
  - `relaykv_runtime_observation_skip` を検出
  - `forward_pass_id=1`
  - `reason="TypeError"`

任意server smoke 全体としては、この環境では `/generate` 後に flashinfer JIT が ninja 不在で落ちるため exit 1 になった。

ただし、RelayKV observation hook の到達確認としては、env on case で skip log を検出できたため前進した。

## 判断

今回の結果は以下のように扱う。

```text
optional server smoke:
  flashinfer JIT / ninja 不在で script 全体は exit 1

RelayKV observation hook:
  env off では observation log なし
  env on では実server経路で hook 到達
  tensor-like metadata により TypeError skip
  skip log 検出済み
```

これは RelayKV hook の失敗ではない。

実 `ForwardBatch` では `req_pool_indices` / `seq_lens` が tensor-like であり、現段階では `.cpu()` / `.item()` / `.tolist()` を使わず skip する方針なので、`TypeError` skip は想定どおり。

## 維持できている安全境界

以下は維持。

- default-off
- payload-only
- summary-only
- skip-safe
- env off では no-op
- env on でも tensor-like metadata は変換せず skip
- hook例外で forward/server を止めない
- model download は必須にしない
- optional smoke は model未設定なら clean skip

## 触っていない領域

以下には差分なし。

- `python/sglang/srt/model_executor/model_runner.py`
- `ForwardBatch.init_new()`
- `ModelRunner._forward_raw()`
- `scheduler.py`
- attention backend
- `memory_pool.py`
- `.github/workflows`
- flashinfer

## まだ禁止すること

次段階でも以下は禁止。

- attention 接続
- attention override
- attention backend 変更
- KV cache free
- KV pool 書き換え
- KV pool snapshot
- host backup copy 実行
- runtime writeback
- scheduler 挙動変更
- `ForwardBatch.init_new()` 変更
- `ModelRunner._forward_raw()` 変更
- `memory_pool.py` 変更
- `.github/workflows` 変更
- tensor `.cpu()` / `.item()` / `.tolist()` 使用

## 現在の到達点

```text
fake hook smoke:
  pass

optional server smoke:
  env off:
    RelayKV observation log なし

  env on:
    実server経路で RelayKV observation hook 到達
    relaykv_runtime_observation_skip 検出
    reason="TypeError"
```

未達:

- 実serverで payload summary を出すこと
- KV snapshot
- host backup copy
- attention 接続
- scheduler 連携
- runtime writeback

## 次段階候補

次もまだ host backup copy 接続ではない。

次に進むなら、実 `ForwardBatch` の tensor-like metadata を同期なしで識別・記録する観測だけを追加する。

候補:

- `req_pool_indices` / `seq_lens` の型名を記録
- `shape` 属性があれば安全に文字列化して記録
- `device` 属性があれば安全に文字列化して記録
- `dtype` 属性があれば安全に文字列化して記録
- `.cpu()` / `.item()` / `.tolist()` は禁止
- payload 化や host backup copy には進まない

目的は、実serverの `ForwardBatch` metadata がどの形で届いているかを、同期なし・read-only で把握すること。

host backup copy / KV snapshot / attention 接続は、その後に再判断する。
