# Devlog: RelayKV read-only dry-run policy log

## 日付確認

このdevlogは JST 基準で `2026-05-03` の作業記録として作成する。

Branch: `relaykv-host-backup-shadow`  
Repo: `~/work/sglang-relaykv`

## 今日の目的

RelayKV の metadata plumbing を、実KV操作に進む前の **dry-run policy log** 段階まで進める。

前段階までに、以下は完了している。

```text
ForwardBatch read-only metadata
runtime observation summary
host backup candidate summary
runtime/candidate join summary
readonly runtime candidate join report
```

今回の目的は、read-only metadata / fake block metadata から、RelayKV が将来どの block を選ぶ予定かを **ログだけ** で説明できるようにすること。

まだ以下には進まない。

```text
KV pool read
host backup copy実行
KV snapshot
attention接続
scheduler decision変更
runtime writeback
```

## 背景

RelayKV の開発順序は以下。

```text
metadata
→ dry-run
→ materialization
→ attention
→ quality
→ speed
```

今回の作業は `dry-run` の最初の段階にあたる。

ここで重要なのは、まだ実際の KV を読まないこと。

目的は以下。

```text
request_id
req_pool_idx
seq_len
layer_id
anchor_block_ids
recent_block_ids
candidate_block_ids
selected_block_ids
retrieval_budget_tokens
kv_classes_present
```

を、実行系に接続せずにログとして説明できる状態にすること。

## 変更概要

追加したもの:

```text
RelayKV read-only dry-run policy event builder
RelayKV dry-run policy summary helper
dry-run policy smoke
```

変更ファイル:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_policy_dry_run_smoke.py
```

## 追加 helper

追加された helper は以下。

```text
build_relaykv_policy_dry_run_events_for_smoke(...)
summarize_relaykv_policy_dry_run_events_for_smoke(...)
```

目的:

```text
runtime payloads
+
fake block metadata
+
policy config

から dry-run policy event を生成し、
summary として集計する。
```

この helper は pure Python / read-only / smoke 用である。

## dry-run event schema

生成される event の基本 schema:

```text
event_type="relaykv_policy_dry_run"
request_id
req_pool_idx
seq_len
layer_id
policy_state="dry_run"
kv_budget_tokens
recent_tokens
anchor_tokens
transient_tokens
retrieval_budget_tokens
candidate_block_ids
selected_block_ids
anchor_block_ids
recent_block_ids
kv_classes_present
layer_budget_policy
retrieval_top_k
source="readonly_metadata_policy_dry_run"
```

safety flags:

```text
source_mutated=false
attention_override=false
kv_cache_mutation=false
runtime_writeback=false
scheduler_policy_noop=true
```

## policy config

最小 config:

```text
kv_budget_tokens
recent_tokens
anchor_tokens
transient_tokens
retrieval_top_k
layer_budget_policy
```

今回の smoke では以下を使う。

```text
kv_budget_tokens=1024
recent_tokens=256
anchor_tokens=128
transient_tokens=64
retrieval_top_k=2
layer_budget_policy="uniform"
```

この場合:

```text
retrieval_budget_tokens =
  1024 - 256 - 128 - 64
= 576
```

`retrieval_budget_tokens` が負になる場合は 0 に clamp する。

## 選択ロジック

今回の dry-run policy は、実selection algorithm ではない。

smoke用の最小仕様:

```text
anchor_block_ids:
  fake block metadata からそのまま使う

recent_block_ids:
  fake block metadata からそのまま使う

candidate_block_ids:
  fake block metadata からそのまま使う

selected_block_ids:
  candidate_block_ids の先頭 N 個
  N = retrieval_top_k
```

つまり、目的は selection性能ではなく、schema / budget / safety / join可能性の確認である。

## smoke

追加:

```text
scripts/relaykv_policy_dry_run_smoke.py
```

### runtime payloads

```text
rid-a / req_pool_idx 10 / seq_len 512  / layer 0
rid-a / req_pool_idx 10 / seq_len 512  / layer 14
rid-b / req_pool_idx 11 / seq_len 1024 / layer 0
```

### fake block metadata

```text
rid-a:
  anchor_block_ids=[0]
  recent_block_ids=[7,8]
  candidate_block_ids=[1,2,3,4]

rid-b:
  anchor_block_ids=[0,1]
  recent_block_ids=[15]
  candidate_block_ids=[9,10,11]
```

### 期待値

```text
total_events=3
all event_type are relaykv_policy_dry_run
selected_block_ids for rid-a are [1,2]
selected_block_ids for rid-b are [9,10]
retrieval_budget_tokens=576
kv_classes_present includes RECENT, ANCHOR, RETRIEVED, COLD_CANDIDATE
safety counters all zero
per_request_counts rid-a=2, rid-b=1
per_layer_counts 0=2, 14=1
inputs not mutated
```

追加テスト:

```text
retrieval_top_k larger than candidates
negative retrieval budget clamps to 0
missing optional block metadata produces empty lists, not crash
poison object in unrelated field does not call:
  .cpu()
  .item()
  .tolist()
  .__iter__()
  .__len__()
  .__getitem__()
```

## 確認結果

実行した確認:

```bash
PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/relaykv/metrics.py \
  scripts/relaykv_policy_dry_run_smoke.py

PYTHONPATH=python .venv/bin/python scripts/relaykv_policy_dry_run_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_candidate_join_report_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_observation_host_backup_join_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_host_backup_candidate_variation_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_policy_smoke.py
```

結果:

```text
pass
```

## git確認

```text
git diff --check: pass
制約 grep: 出力なし
```

触っていない禁止領域:

```text
python/sglang/srt/model_executor/forward_batch_info.py
python/sglang/srt/model_executor/model_runner.py
scheduler.py
schedule_batch.py
attention backend
memory_pool.py
flashinfer
.github/workflows
```

## commit / push

今回の変更は commit / push 済み。

commit message:

```text
relaykv: add readonly policy dry-run events
```

push remote:

```text
mine relaykv-host-backup-shadow
```

## 現在の到達点

現在の流れ:

```text
metadata
→ join
→ readonly report
→ dry-run policy log
```

ここまでで、まだ実KVには触らずに以下をログで説明できるようになった。

```text
request_id
req_pool_idx
seq_len
layer_id
anchor_block_ids
recent_block_ids
candidate_block_ids
selected_block_ids
retrieval_budget_tokens
kv_classes_present
```

## 現在位置

```text
Phase 0:
  ForwardBatch metadata観測 完了

Phase 1:
  runtime observation metadata と candidate summary のjoin確認 完了

Phase 2:
  dry-run policy log 開始・最小実装完了
```

## 維持できている安全境界

まだ実施していないこと:

```text
KV pool read
host backup copy実行
KV snapshot
attention接続
scheduler decision変更
runtime writeback
```

dry-run policy helper も、あくまで fake block metadata と readonly runtime payload から event を作るだけである。

## 判断

今回の実装は安全な前進。

理由:

```text
metrics.py と新規 smoke のみ
runtime execution path に接続していない
ForwardBatch / ModelRunner / scheduler / attention に触れていない
KV pool を読んでいない
host backup copy を実行していない
```

特に重要なのは、`selected_block_ids` をログとして出せるようになったこと。

これにより、今後の実 materialization 前に以下を検証できる。

```text
budget計算の形が妥当か
Anchor と Retrieved が別枠で扱えているか
RECENT / COLD_CANDIDATE / RETRIEVED の境界をログで説明できるか
request_id / req_pool_idx / layer_id の対応が崩れていないか
```

## 次の分岐

次はまだ materialization ではなく、dry-run policy events を readonly report と統合するのが安全。

候補:

```text
A. dry-run policy events を readonly runtime candidate join report に統合する
B. dry-run policy summary を独立reportとして安定化する
C. ここで一度、現在の実装を次セッション向けにまとめる
```

おすすめは A。

理由:

```text
runtime observation summary
host backup candidate summary
join summary
dry-run policy summary

を1つの readonly report として扱えると、
実copy前の診断がかなりしやすくなる。
```

## 次の推奨タスク

次の Codex タスク候補:

```text
Add dry-run policy summary into RelayKV readonly runtime/candidate/join report.
```

狙い:

```text
readonly report に policy_dry_run_summary を追加
overall_safety_status に dry-run safety counters も含める
smoke で pass/fail を確認
```

まだやらないこと:

```text
KV pool read
host backup copy実行
attention接続
scheduler変更
runtime writeback
```
