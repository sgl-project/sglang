# Devlog: RelayKV Phase 2 dry-run policy plumbing completion

## 日付確認

このdevlogは JST 基準で `2026-05-03` の作業記録として作成する。

今回は小さなcommitごとの記録ではなく、以下をまとめた **Phase 1後半〜Phase 2完了の区切り** として作成する。

```text
runtime observation / host backup candidate join
readonly report
dry-run policy events
dry-run diagnostic flow
materialization readiness assessment
```

Branch: `relaykv-host-backup-shadow`  
Repo: `~/work/sglang-relaykv`

---

## 1. このdevlogの目的

RelayKV/SGLang統合における metadata plumbing / dry-run policy 段階の完了を記録する。

今回の到達点は以下。

```text
metadata
→ join
→ readonly report
→ dry-run policy log
→ dry-run diagnostic flow
→ materialization readiness assessment
```

これにより、まだ実KVに触らずに、Phase 3 safe materialization へ進める条件を read-only report 上で判定できるようになった。

---

## 2. RelayKVの現在位置

RelayKVは、単純なKV削減アルゴリズムではなく、量子化dense modelを低VRAM GPUに載せた後、残ったVRAM内で decode-time KV working set を制御する **VRAM-aware memory management layer** として扱う。

現在のSGLang側の位置づけ:

```text
Phase 0:
  ForwardBatch metadata observation

Phase 1:
  runtime observation metadata と host backup candidate summary のjoin確認

Phase 2:
  dry-run policy plumbing

Phase 3:
  safe materialization

Phase 4:
  attention connection

Phase 5:
  quality evaluation

Phase 6:
  speed / optimization
```

今回で Phase 2 までを閉じる。

---

## 3. 維持している安全境界

今回までに、以下にはまだ進んでいない。

```text
KV pool read
host backup copy実行
KV snapshot
attention接続
scheduler decision変更
runtime writeback
```

触っていない領域:

```text
ForwardBatch / ModelRunner の追加変更
scheduler / schedule_batch
attention backend
memory_pool
flashinfer
.github/workflows
```

重要な方針:

```text
RelayKVはまず、SGLang実行経路の外側にあるread-only metadata layerとして成立させる。
```

---

## 4. 実装済みの流れ

今回のまとまりで実装・確認した流れは以下。

```text
1. ForwardBatch read-only metadata
2. runtime observation summary
3. host backup candidate summary
4. runtime/candidate join summary
5. readonly runtime candidate join report
6. dry-run policy events
7. dry-run policy summary included in readonly report
8. readonly diagnostic flow smoke
9. readonly materialization readiness assessment
```

---

## 5. runtime/candidate join

### helper

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

join key:

```text
request_id
req_pool_idx
layer_id
```

candidate側の互換:

```text
layer_idx -> layer_id
req_pool_idx / req_pool_index / request_pool_idx
```

summary-only candidate summary の場合は、詳細を捏造しない。

```text
join_granularity="summary_only_unjoinable"
```

これは、aggregate summary から per-event detail を推測してしまう事故を防ぐための重要な安全策。

### smoke

```text
scripts/relaykv_runtime_observation_host_backup_join_smoke.py
```

確認内容:

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

---

## 6. readonly runtime/candidate/join report

### helper

```text
build_relaykv_readonly_runtime_candidate_join_report_for_smoke(...)
```

目的:

```text
runtime_observation_summary
host_backup_candidate_summary
join_summary

を1つの readonly report dict にまとめる。
```

主要field:

```text
report_type="relaykv_readonly_runtime_candidate_join_report"
report_generated_from_readonly_inputs=true
runtime_observation_summary
host_backup_candidate_summary
join_summary
overall_safety_status
```

主要集約field:

```text
total_runtime_payloads
total_host_backup_candidate_events
joined_count
unmatched_runtime_count
unmatched_candidate_count
join_granularity
req_pool_idx_joined_count
req_pool_idx_missing_count
```

safety counters:

```text
source_mutated_true_count
attention_override_true_count
kv_cache_mutation_true_count
runtime_writeback_true_count
scheduler_policy_noop_false_count
```

### safety rule

```text
overall_safety_status="pass"

条件:
  report_generated_from_readonly_inputs=true
  safety counters がすべて 0
```

safety counter が1つでも nonzero なら fail。

---

## 7. dry-run policy events

### helper

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

これは実selection algorithmではなく、schema / budget / safety / join可能性を確認するための最小dry-runである。

### event schema

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

### policy config

最小config:

```text
kv_budget_tokens
recent_tokens
anchor_tokens
transient_tokens
retrieval_top_k
layer_budget_policy
```

例:

```text
kv_budget_tokens=1024
recent_tokens=256
anchor_tokens=128
transient_tokens=64
retrieval_top_k=2
layer_budget_policy="uniform"
```

このとき:

```text
retrieval_budget_tokens = 1024 - 256 - 128 - 64 = 576
```

負の retrieval budget は 0 に clamp する。

### selection behavior for smoke

```text
anchor_block_ids:
  fake block metadata からそのまま

recent_block_ids:
  fake block metadata からそのまま

candidate_block_ids:
  fake block metadata からそのまま

selected_block_ids:
  candidate_block_ids の先頭 N 個
  N = retrieval_top_k
```

### smoke

```text
scripts/relaykv_policy_dry_run_smoke.py
```

確認内容:

```text
total_events=3
selected_block_ids for rid-a are [1,2]
selected_block_ids for rid-b are [9,10]
retrieval_budget_tokens=576
kv_classes_present includes RECENT, ANCHOR, RETRIEVED, COLD_CANDIDATE
safety counters all zero
per_request_counts rid-a=2, rid-b=1
per_layer_counts 0=2, 14=1
inputs not mutated
```

追加確認:

```text
retrieval_top_k larger than candidates
negative retrieval budget clamps to 0
missing optional block metadata produces empty lists
poison object の unsafe method 未呼び出し
```

---

## 8. dry-run policy summary included in readonly report

readonly report helper を拡張し、任意で以下を含められるようにした。

```text
policy_dry_run_summary
```

追加field:

```text
policy_dry_run_included=true/false
policy_dry_run_total_events
policy_dry_run_selected_event_count
```

overall safety rule も拡張した。

```text
overall_safety_status="pass"

条件:
  report_generated_from_readonly_inputs=true
  runtime/candidate/join safety counters all zero
  policy_dry_run_summary がある場合、その safety counters も all zero
```

fail case:

```text
policy_dry_run_summary.kv_cache_mutation_true_count=1
→ overall_safety_status="fail"
```

---

## 9. readonly diagnostic flow smoke

### smoke

```text
scripts/relaykv_readonly_diagnostic_flow_smoke.py
```

目的:

```text
runtime payloads
→ runtime observation summary
→ host backup candidate summary
→ join summary
→ dry-run policy events
→ dry-run policy summary
→ readonly runtime/candidate/join report with policy_dry_run_summary

を fake inputs のみで一貫して流す。
```

まだ real ModelRunner / ForwardBatch には接続しない。

確認内容:

```text
runtime summary total_payloads=4
host backup candidate summary total_candidate_events=4
join summary:
  joined_count=3
  unmatched_runtime_count=1
  unmatched_candidate_count=1
  req_pool_idx_joined_count=3

dry-run policy summary:
  total_events=4
  per_request_counts rid-a=2, rid-b=2
  per_layer_counts 0=2, 14=2
  safety counters all zero

readonly report:
  report_type correct
  policy_dry_run_included=true
  policy_dry_run_total_events=4
  overall_safety_status="pass"
  all safety counters zero
```

追加確認:

```text
policy_dry_run_summary.runtime_writeback_true_count=1
→ overall_safety_status="fail"

summary-only candidate
→ join_granularity="summary_only_unjoinable"
→ report still builds
```

---

## 10. materialization readiness assessment

### helper

```text
assess_relaykv_readonly_materialization_readiness_for_smoke(report)
```

目的:

```text
readonly diagnostic report を見て、
Phase 3 safe materialization に進んでよい状態かを
read-only に判定する。
```

これは materialization を実行するものではない。

### readiness schema

```text
readiness_type="relaykv_readonly_materialization_readiness"
ready_for_materialization
readiness_state
readiness_reasons
blocking_reasons
warning_reasons
observed_join_granularity
observed_overall_safety_status
observed_policy_dry_run_included
observed_policy_dry_run_total_events
observed_joined_count
observed_unmatched_runtime_count
observed_unmatched_candidate_count
observed_req_pool_idx_missing_count
report_generated_from_readonly_inputs
```

safety counters:

```text
source_mutated_true_count
attention_override_true_count
kv_cache_mutation_true_count
runtime_writeback_true_count
scheduler_policy_noop_false_count
```

### ready condition

`ready_for_materialization=true` は以下をすべて満たす場合のみ。

```text
report_generated_from_readonly_inputs=true
overall_safety_status="pass"
policy_dry_run_included=true
policy_dry_run_total_events > 0
joined_count > 0
join_granularity != "summary_only_unjoinable"
req_pool_idx_missing_count == 0
all safety counters are zero
```

### readiness_state

代表的な状態:

```text
ready_for_safe_materialization_dry_run_complete
blocked_safety_counter_nonzero
blocked_not_readonly_report
blocked_overall_safety_not_pass
blocked_policy_dry_run_missing
blocked_no_joined_events
blocked_summary_only_unjoinable
blocked_req_pool_idx_missing
blocked_multiple_reasons
```

### smoke確認

追加確認済み:

```text
ready case:
  ready_for_materialization=true
  readiness_state="ready_for_safe_materialization_dry_run_complete"

fail cases:
  safety nonzero
  summary-only unjoinable
  policy dry-run missing
  req_pool_idx missing
  joined_count=0

input report non-mutation
```

---

## 11. 実行確認

実行済み smoke:

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_readonly_diagnostic_flow_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_candidate_join_report_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_policy_dry_run_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_observation_host_backup_join_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_host_backup_candidate_variation_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_policy_smoke.py
```

結果:

```text
pass
```

py_compile:

```bash
PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/relaykv/metrics.py \
  scripts/relaykv_readonly_diagnostic_flow_smoke.py \
  scripts/relaykv_runtime_candidate_join_report_smoke.py
```

結果:

```text
pass
```

git確認:

```text
git diff --check: pass
制約 grep: 出力なし
```

---

## 12. commit / push 済みの主なまとまり

今回の大きなまとまりに含まれるcommit群:

```text
runtime observation と host backup candidate の read-only join
readonly runtime/candidate/join report
dry-run policy events
policy dry-run summary included in readonly report
readonly diagnostic flow smoke
readonly materialization readiness assessment
```

直近commit:

```text
Add RelayKV readonly materialization readiness smoke
```

push remote:

```text
mine relaykv-host-backup-shadow
```

---

## 13. 現在の到達点

現在の到達点:

```text
metadata
→ join
→ readonly report
→ dry-run policy log
→ dry-run diagnostic flow
→ materialization readiness assessment
```

現在位置:

```text
Phase 0:
  ForwardBatch metadata観測 完了

Phase 1:
  runtime observation metadata と candidate summary のjoin確認 完了

Phase 2:
  dry-run policy plumbing 完了
```

これで、Phase 3 safe materialization に進む前の read-only な検証基盤が整った。

---

## 14. なぜこの区切りが重要か

今回の区切りで、RelayKVは以下を満たすようになった。

```text
request_id / req_pool_idx / seq_len / layer_id を持つ runtime metadata
host backup candidate event
candidate join
policy dry-run selection
readonly report
materialization readiness判定
```

つまり、まだ KV pool に触らずに、以下を説明できる。

```text
どのrequestか
どのreq_pool_idxか
どのlayerか
どのcandidateがあるか
どのblockをdry-runで選ぶか
Anchor / Recent / Retrieved / Cold Candidate がどう見えるか
Phase 3へ進んでよいか
```

この状態は、実KV操作へ進む前の安全なゲートとして機能する。

---

## 15. 次のフェーズ

次は Phase 3。

```text
Phase 3:
  safe KV materialization
```

ただし、いきなり runtime接続するのではなく、次も段階を刻む。

推奨順:

```text
3.0 materialization design memo
3.1 host backup candidate payload schema 固定
3.2 fake materialization result schema
3.3 no-op guarded materialization smoke
3.4 actual host backup copy helper の安全境界確認
3.5 runtime接続前の readiness gate
```

まだ注意すること:

```text
KV pool read は最初から real runtime に接続しない
attention には接続しない
scheduler decision は変えない
runtime writeback はしない
```

---

## 16. 次の推奨タスク

次に進むなら、まず Phase 3 の設計メモを作る。

目的:

```text
safe materialization で何を「読む」とみなし、
何を「copy」とみなし、
どこまでを no-op guarded とするかを固定する。
```

次のCodex実装に進む前に、以下を決める。

```text
materialization input schema
materialization output schema
host backup copy candidate event と materialized result の対応
no-op / guarded / applied の状態定義
safety flags
failure modes
runtime接続禁止境界
```

---

## 17. 最終判断

Phase 2 は完了とみなしてよい。

理由:

```text
readonly report がある
dry-run policy summary がある
diagnostic flow smoke がある
readiness assessment がある
blocking reason を機械的に出せる
KV / attention / scheduler に触れていない
```

次は `metadata → dry-run` から `dry-run → safe materialization` へ進む段階。

ただし、Phase 3 の最初は設計メモから始める。
