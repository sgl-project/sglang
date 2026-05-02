# RelayKV Host Backup Shadow Runtime Observation Plan

## 日付

2026-05-02 JST

## 対象

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`

## 現在の段階

runtime candidate event payload 互換の構造から、`applied_candidate` の場合だけ read-only snapshot -> host backup copy candidate が通る。

`fallback_candidate` は no-op guard により snapshot / copy を行わない。

まだ以下には進んでいない。

```text
実runtime接続
attention接続
attention override
KV cache free
KV pool書き換え
runtime writeback
scheduler挙動変更
```

## 通過済み smoke

```text
scripts/relaykv_runtime_policy_smoke.py
scripts/relaykv_host_backup_candidate_variation_smoke.py
scripts/relaykv_runtime_observation_readonly_smoke.py
scripts/relaykv_actual_kv_pool_observation_smoke.py
scripts/relaykv_kv_pool_snapshot_smoke.py
scripts/relaykv_host_backup_copy_smoke.py
scripts/relaykv_memory_smoke.py
```

## read-only runtime observation smoke の確認値

`scripts/relaykv_runtime_observation_readonly_smoke.py` は、ForwardBatch / ModelRunner 相当の fake batch 情報から runtime observation 風の candidate event を6件流す。

確認済み summary。

```text
total_candidate_events = 6
applied_candidate_count = 4
fallback_candidate_count = 2
host_backup_copy_executed_count = 4
fallback_candidate_noop_guard_count = 2
per_layer_counts = 0, 1, 2
per_request_counts = rid-a, rid-b, rid-c
per_batch_counts = obs-batch-a
skipped_reason_counts = {"fallback_candidate_noop_guard": 2}
```

## 保持すべき safety invariant

次段階へ進む前後で、以下を維持する。

```text
host_backup_copy_executed_count == applied_candidate_count
fallback_candidate_noop_guard_count == fallback_candidate_count
source_mutated_true_count == 0
attention_override_true_count == 0
kv_cache_mutation_true_count == 0
runtime_writeback_true_count == 0
scheduler_policy_noop_false_count == 0
```

## まだ禁止する変更

```text
attention接続
attention override
attention backend変更
KV cache free
KV pool書き換え
runtime writeback
scheduler挙動変更
ForwardBatch / ModelRunner hot path の無条件変更
memory_pool getter 経由の観測追加
.github/workflows変更
重いmodel download前提の変更
```

## 次に進む条件

次段階は実runtime接続ではなく、default-off read-only observation hook の設計に留める。

進行条件。

```text
default-off
環境変数で完全disable可能
read-only observation のみ
summary log only
no mutation
no writeback
no scheduler decision change
既存smokeがすべて通る
制約grepが空
```

制約grep。

```bash
git diff --name-only | grep -E 'scheduler.py|attention|flashinfer|\.github/workflows' || true
```

## 即停止 / rollback 条件

以下のいずれかが出た場合は、実runtime寄りの接続を止め、直前の read-only smoke 段階へ戻す。

```text
source_mutated_true_count != 0
attention_override_true_count != 0
kv_cache_mutation_true_count != 0
runtime_writeback_true_count != 0
scheduler_policy_noop_false_count != 0
host_backup_copy_executed_count != applied_candidate_count
fallback_candidate で copy 実行が増える
scheduler.py / attention / flashinfer / .github/workflows に意図しない差分が出る
実model downloadやserver起動が必須になる
```

## 次段階の候補

次は実runtime接続ではなく、default-off read-only observation hook の設計比較を行う。

候補地点。

```text
scheduler.py の既存 RelayKV shadow/runtime policy event log 位置
ForwardBatch.init_new()
ModelRunner.forward() / _forward_raw() の手前
```

ただし、いずれも hot path なので、最初はコード変更なしの設計比較に留める。

設計比較では以下を確認する。

```text
どの情報が既存payloadから取れるか
どの情報がread-onlyに観測できるか
default-off guard をどこに置くべきか
summary log only にできるか
既存smokeへ同じ safety invariant を流用できるか
```

## 確認コマンド

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_observation_readonly_smoke.py
git diff --check
git diff --name-only | grep -E 'scheduler.py|attention|flashinfer|\.github/workflows' || true
```

## ModelRunner.forward default-off hook 実装済み状態

`ModelRunner.forward()` に default-off / payload-only / skip-safe な read-only observation hook を追加済み。

hook 位置。

```text
python/sglang/srt/model_executor/model_runner.py
ModelRunner.forward()
self.forward_pass_id += 1 の直後
msprobe_debugger.start() より前
```

環境変数。

```text
SGLANG_RELAYKV_RUNTIME_OBSERVATION
unset / "0": off
"1": payload-only observation
```

OFF時は `os.getenv(...) == "1"` の判定のみで、RelayKV observation helper は遅延importされない。

ON時も以下だけを行う。

```text
build_runtime_observation_payloads()
summarize_runtime_observation_payloads()
log_runtime_observation_summary()
```

ON時でも以下は禁止。

```text
KV pool参照
KV pool snapshot
host backup copy
attention接続
scheduler state変更
runtime writeback
tensor.cpu() / tensor.item() / tensor.tolist()
```

実 `ForwardBatch` の `req_pool_indices` / `seq_lens` は tensor のため、hook はこれを list 化せず `TypeError` skip として扱う。

hook 内例外は forward を止めず、debug log に留める。

## hook safety 確認済み結果

確認済み。

```text
git status --short: clean
git diff --check: pass
py_compile model_runner.py observation.py: pass
constraint grep: 出力なし
```

通過済み smoke。

```text
scripts/relaykv_model_runner_observation_hook_smoke.py
scripts/relaykv_runtime_observation_payload_builder_smoke.py
scripts/relaykv_runtime_observation_summary_smoke.py
scripts/relaykv_runtime_observation_readonly_smoke.py
scripts/relaykv_host_backup_candidate_variation_smoke.py
scripts/relaykv_runtime_policy_smoke.py
```

`SGLANG_RELAYKV_RUNTIME_OBSERVATION` の unset / `0` / `1` でも hook smoke は pass。

hook smoke 内で確認済みのケース。

```text
off:
  enabled = false
  skipped = true
  skip_reason = "env_disabled"

on + list/tuple:
  payload生成
  total_payloads = 2
  safety counters = 0

on + tensor-like:
  TypeError skip
  .cpu() / .item() / .tolist() 未呼び出し

hook例外:
  RuntimeError skip
  forward相当処理を止めない
```

## 実server到達確認の位置づけ

実server到達確認は標準確認ではない。

以下を満たす場合だけ、任意の軽量確認として扱う。

```text
ローカルに既存の小モデルがある
追加model downloadが不要
server起動が必須ではない、または明示的な任意確認として扱える
CUDA/GPU必須ではない
CI / workflow には接続しない
失敗しても rollback しやすい
```

実server smoke を通常の必須確認にしてはいけない。

## 実server到達確認へ進む条件

以下をすべて満たすまで、実server到達確認には進まない。

```text
default-off hook smoke が pass
env unset / "0" / "1" で hook smoke が pass
env on でも tensor-like metadata は skip
hook例外で forward を止めない
.cpu() / .item() / .tolist() を呼ばない
KV pool / snapshot / host backup copy に触れない
attention / scheduler / writeback に触れない
既存 RelayKV smoke が pass
制約grepが空
```

## 実server到達確認の即停止 / rollback 条件

以下のいずれかが出た場合は、実server到達確認を止め、default-off hook helper smoke 段階へ戻す。

```text
env off で挙動差分が出る
env on で forward が止まる
tensor-like metadata を list 化しようとする
tensor.cpu() / tensor.item() / tensor.tolist() を呼ぶ
KV pool 参照に進む
KV pool snapshot に進む
host backup copy に進む
attention / scheduler / writeback に触れる
実server smoke が model download 必須になる
server起動が標準確認に組み込まれる
scheduler.py / attention / flashinfer / .github/workflows に意図しない差分が出る
```

## 次段階候補

次段階は以下の順で進める。

```text
A. hook helper smoke 拡張
   - env値、skip理由、ログprefix、例外握りつぶしのケースを増やす
   - server / model 不要

C. server起動なし fake ModelRunner.forward() 到達 smoke
   - fake self / fake forward_batch / _forward_raw 差し替えで hook位置到達のみ確認
   - payload-only / skip-only を維持

B. ローカル既存小モデル限定の任意実server smoke
   - model download 不要の場合だけ
   - 標準確認にはしない
```

host backup copy 接続はまだ禁止。
