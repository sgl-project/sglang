# Devlog: RelayKV host backup shadow runtime observation

Date: 2026-05-02 JST  
Branch: `relaykv-host-backup-shadow`  
Repo: `~/work/sglang-relaykv`

## 今日の目的

RelayKV host backup shadow を、実runtime / SGLang `ModelRunner.forward()` に近づける前段として、以下を安全に分離・検証する。

- runtime observation payload 化
- observation summary
- default-off runtime observation hook
- fake `ModelRunner.forward()` 到達 smoke
- host backup copy / KV snapshot / attention 接続にはまだ進まない

## 実装・追加したもの

### 1. runtime observation payload builder

追加・更新:

- `python/sglang/srt/relaykv/observation.py`
- `scripts/relaykv_runtime_observation_payload_builder_smoke.py`

追加 helper:

- `build_runtime_observation_payloads()`

責務:

- ForwardBatch 風の batch-like object から request x layer の read-only observation payload を作る。
- KV pool、snapshot、host backup copy、attention、scheduler decision、runtime writeback には触れない。
- `rids` / `req_pool_indices` / `seq_lens` / `layer_ids` は list / tuple のみ受ける。
- tensor-like object に対して `.cpu()` / `.item()` / `.tolist()` は呼ばない。

### 2. runtime observation summary helper

追加・更新:

- `python/sglang/srt/relaykv/observation.py`
- `scripts/relaykv_runtime_observation_summary_smoke.py`

追加 helper:

- `summarize_runtime_observation_payloads(payloads)`
- `log_runtime_observation_summary(summary, logger_=None, prefix="relaykv_runtime_observation_summary")`

summary fields:

- `total_payloads`
- `per_request_counts`
- `per_layer_counts`
- `per_batch_counts`
- `source_mutated_true_count`
- `attention_override_true_count`
- `kv_cache_mutation_true_count`
- `runtime_writeback_true_count`
- `scheduler_policy_noop_false_count`

### 3. ModelRunner.forward default-off runtime observation hook

変更:

- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/relaykv/observation.py`
- `scripts/relaykv_model_runner_observation_hook_smoke.py`

hook位置:

- `ModelRunner.forward()` 内
- `self.forward_pass_id += 1` の直後
- `msprobe_debugger.start()` より前

env:

- `SGLANG_RELAYKV_RUNTIME_OBSERVATION`

挙動:

- unset / `"0"`: off
- `"1"`: payload-only observation

OFF時:

- `os.getenv(...) == "1"` の判定のみ
- RelayKV helper は import しない
- forward 挙動に影響しない

ON時:

- `build_runtime_observation_payloads()`
- `summarize_runtime_observation_payloads()`
- `log_runtime_observation_summary()`

のみ実行候補。

### 4. fake ModelRunner.forward observation smoke

追加:

- `scripts/relaykv_fake_model_runner_forward_observation_smoke.py`

構造:

- `_FakeForwardBatchListLike`
- `_FakeForwardBatchTensorLike`
- `_PoisonTensorLike`
  - `.cpu()` / `.item()` / `.tolist()` が呼ばれたら `AssertionError`
- `_fake_forward_with_observation_hook(...)`
  - `forward_pass_id += 1` 相当
  - RelayKV hook helper 呼び出し
  - hook例外は skip扱い
  - 最後に fake forward result を返す

確認ケース:

- env off
  - `forward_completed=true`
  - `skip_reason="env_disabled"`

- env on + list/tuple
  - `forward_completed=true`
  - `total_payloads=2`
  - safety counters all zero

- env on + tensor-like
  - `forward_completed=true`
  - `skip_reason="TypeError"`
  - sync系メソッド未呼び出し

- env on + synthetic exception
  - `forward_completed=true`
  - `skip_reason="RuntimeError"`

## 通過した smoke

以下すべて pass。

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_fake_model_runner_forward_observation_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_model_runner_observation_hook_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_observation_payload_builder_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_observation_summary_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_observation_readonly_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_host_backup_candidate_variation_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_policy_smoke.py
```

追加確認:

```bash
PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/model_executor/model_runner.py \
  python/sglang/srt/relaykv/observation.py

git diff --check
git diff --name-only | grep -E 'scheduler.py|attention|flashinfer|\.github/workflows' || true
```

結果:

- `py_compile`: pass
- `git diff --check`: pass
- 制約grep: 出力なし

## 維持できている safety invariant

- `source_mutated_true_count=0`
- `attention_override_true_count=0`
- `kv_cache_mutation_true_count=0`
- `runtime_writeback_true_count=0`
- `scheduler_policy_noop_false_count=0`
- tensor-like metadata で `.cpu()` / `.item()` / `.tolist()` 未呼び出し
- hook例外で forward 相当処理を止めない
- fallback / skip は copy に進まない

## 触っていない領域

以下には差分なし。

- `scheduler.py`
- `ForwardBatch.init_new()`
- `ModelRunner._forward_raw()`
- attention backend
- `memory_pool.py`
- `flashinfer`
- `.github/workflows`

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
- scheduler decision 変更
- ForwardBatch hot path 変更
- `_forward_raw()` 変更
- tensor metadata の list 化
- `.cpu()` / `.item()` / `.tolist()` 使用
- model download 必須の smoke
- server 起動必須の標準 smoke

## 現在の結論

RelayKV runtime observation は、以下の段階まで安全に進んだ。

```text
payload builder
  -> summary helper
  -> ModelRunner.forward default-off hook
  -> fake ModelRunner.forward 到達 smoke
```

ただし、まだ host backup copy / KV snapshot / attention 接続には進んでいない。

現段階の hook は、実runtime hook ではあるが、処理内容は payload-only / summary-only / skip-safe に限定されている。

## 次段階候補

優先順:

1. ローカル既存小モデル限定の任意実server smoke 設計
2. env off/on で実server forward が壊れないことだけを確認
3. ON時に tensor-like metadata が skip されることを確認
4. 実server smoke は model download 不要の場合のみ実行
5. host backup copy / KV snapshot への接続はまだ禁止

次に進む場合も、目的はまだ以下に限定する。

```text
env off:
  既存挙動と差分なし

env on:
  observation hook に到達しても forward を止めない
  tensor-like metadata は skip
  payload-only / summary-only
```

host backup copy 接続は、実server smokeで default-off / skip-safe が確認できた後に再検討する。
