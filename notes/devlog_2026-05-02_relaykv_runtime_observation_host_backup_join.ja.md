# Devlog: RelayKV runtime observation と host backup candidate の read-only join

Date: 2026-05-02 JST  
Branch: `relaykv-host-backup-shadow`  
Repo: `~/work/sglang-relaykv`

## 今日の目的

RelayKV の runtime observation metadata と host backup copy candidate event を、read-only summary layer で対応づける。

前段階で、`ForwardBatch` から以下を read-only metadata として観測できるようになった。

```text
request_id / rid
req_pool_idx
seq_len
layer_id
```

今回の目的は、この runtime observation payload と host backup candidate event を、summary上で join できるか確認すること。

まだ以下には進まない。

- host backup copy 実行
- KV pool read
- KV snapshot
- attention 接続
- scheduler decision 変更
- runtime writeback

## 背景

前段階までの到達点:

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

既に以下の flow が確認済み。

```text
ForwardBatch runtime metadata:
  request_id / req_pool_idx / seq_len を read-only observation可能

host backup candidate:
  candidate event summary を既存 helper で集計可能
```

次に必要なのは、両者を `request_id + req_pool_idx + layer_id` で対応づけられるかの確認。

これは、将来の host backup copy / KV working set selection に進む前の metadata plumbing として重要。

## 見たファイル

- `python/sglang/srt/relaykv/metrics.py`
- `python/sglang/srt/relaykv/observation.py`
- `scripts/relaykv_forward_batch_readonly_metadata_observation_smoke.py`

## 変更したファイル

- `python/sglang/srt/relaykv/metrics.py`
- `scripts/relaykv_runtime_observation_host_backup_join_smoke.py`

`relaykv_runtime_observation_host_backup_join_smoke.py` は新規追加。

## 追加 helper

追加した helper:

```text
join_runtime_observation_with_host_backup_candidates_for_smoke(...)
```

目的:

```text
runtime observation payload/event
+
host backup candidate payload/event

を read-only に join し、summary dict を返す。
```

## helper の仕様

### 入力

入力は `dict` / `list` / `tuple` のみ。

torch import はしない。

tensor は読まない。

KV pool / snapshot / host backup copy / attention / scheduler / writeback には接続しない。

### join key

join key:

```text
request_id
req_pool_idx
layer_id
```

candidate 側では、互換性のため以下を扱う。

```text
layer_idx -> layer_id として扱う
```

`req_pool_idx` は以下の別名も見る。

```text
req_pool_idx
req_pool_index
request_pool_idx
```

### summary-only 入力

host backup candidate 側が summary dict だけで per-event detail を持たない場合、詳細は捏造しない。

その場合は以下を返す。

```text
join_granularity="summary_only_unjoinable"
```

これにより、aggregate summary から存在しない join detail を推測してしまう事故を避ける。

## 出力 summary

主な field:

```text
total_runtime_payloads
total_host_backup_candidate_events
joined_count
unmatched_runtime_count
unmatched_candidate_count
per_request_join_counts
per_layer_join_counts
req_pool_idx_joined_count
req_pool_idx_missing_count
join_granularity
```

safety counters:

```text
source_mutated_true_count=0
attention_override_true_count=0
kv_cache_mutation_true_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
```

## 新規 smoke

追加:

```text
scripts/relaykv_runtime_observation_host_backup_join_smoke.py
```

### fake runtime payloads

```text
rid-a / req_pool_idx 10 / layer 0
rid-a / req_pool_idx 10 / layer 14
rid-b / req_pool_idx 11 / layer 0
rid-b / req_pool_idx 11 / layer 14
```

### fake host backup candidate events

```text
matching rid-a layer 0
matching rid-b layer 14
one unmatched candidate
```

### 期待値

```text
joined_count=2
unmatched_runtime_count=2
unmatched_candidate_count=1
per_request_join_counts={"rid-a": 1, "rid-b": 1}
per_layer_join_counts={"0": 1, "14": 1}
req_pool_idx_joined_count=2
safety counters all zero
```

追加で確認したこと:

```text
runtime payload with req_pool_idx missing
candidate event with req_pool_idx missing
req_pool_idx_missing_count=2
poison object の .cpu()/.item()/.tolist()/__iter__/__len__/__getitem__ 未呼び出し
入力 non-mutation
```

## 確認結果

新規 smoke:

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_observation_host_backup_join_smoke.py
```

結果:

```text
pass
```

固定確認値:

```text
joined_count=2
unmatched_runtime_count=2
unmatched_candidate_count=1
per_request_join_counts={"rid-a": 1, "rid-b": 1}
per_layer_join_counts={"0": 1, "14": 1}
req_pool_idx_joined_count=2
req_pool_idx_missing_count=2
safety counters all zero
```

## 既存 smoke

以下はすべて pass。

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_forward_batch_readonly_metadata_observation_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_forward_batch_runtime_existing_metadata_observation_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_host_backup_candidate_variation_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_policy_smoke.py
```

## git確認

```text
git diff --check: pass
制約 grep: 出力なし
```

今回触っていない禁止領域:

- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `scheduler.py`
- `schedule_batch.py`
- attention backend
- flashinfer
- `memory_pool.py`
- `.github/workflows`

## commit / push

今回の変更は push 済み。

commit候補として使った形式:

```bash
git add python/sglang/srt/relaykv/metrics.py \
  scripts/relaykv_runtime_observation_host_backup_join_smoke.py

git commit -m "Add RelayKV runtime observation host backup join smoke"
git push mine relaykv-host-backup-shadow
```

## 現在の到達点

```text
ForwardBatch runtime metadata:
  request_id / req_pool_idx / seq_len を read-only observation可能

host backup candidate:
  candidate event summary を既存 helper で集計可能

今回:
  runtime observation payload と
  host backup candidate event を
  request_id + req_pool_idx + layer_id で join可能
```

ここまでで、RelayKV/SGLang側の metadata plumbing は次の段階に進める状態になった。

```text
1. ForwardBatch から request_id / req_pool_idx / seq_len を観測
2. runtime observation summary に req_pool_idx を載せる
3. host backup candidate event summary を集計
4. runtime observation と candidate event を read-only join
```

## 維持できている安全境界

まだ実施していないこと:

```text
KV pool read
KV snapshot
host backup copy実行
attention接続
scheduler decision変更
runtime writeback
```

今回の join helper も read-only / pure summary helper に留めている。

## 判断

今回の実装は安全な前進。

理由:

```text
変更は metrics.py と新規 smoke のみ
ForwardBatch / ModelRunner / scheduler / attention / memory_pool に触れていない
per-event detailがない summaryから詳細を捏造しない
joinできる場合だけ request_id + req_pool_idx + layer_id で対応づける
```

特に重要なのは以下。

```text
join_granularity="summary_only_unjoinable"
```

これにより、aggregate summary と per-event payload を混同しない。

## 次の分岐

次はまだ host backup copy 実行ではなく、report layer を整えるのが安全。

推奨:

```text
runtime observation summary
+
host backup candidate summary
+
join summary

を同一形式の read-only report にまとめる
```

目的:

```text
env on runtime observation
candidate event summary
join summary
safety counters
```

を一つの summary/report として扱えるようにする。

まだやらないこと:

```text
host backup copy実行
KV pool参照
attention接続
runtime writeback
scheduler decision変更
```

## 次段階のおすすめ

次の実装候補:

```text
RelayKV read-only runtime/candidate/join report helper
```

仮の helper 名:

```text
build_relaykv_readonly_runtime_candidate_join_report_for_smoke(...)
```

期待する内容:

```text
runtime_observation_summary
host_backup_candidate_summary
join_summary
overall_safety_status
report_generated_from_readonly_inputs=true
```

これにより、実copyへ進む前に report / summary layer の見通しを固める。
