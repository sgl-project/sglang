# Devlog: RelayKV Phase 3 前半 — Safe Materialization Schema と Metadata-only Readiness

## 日付確認

この devlog は **JST 基準で 2026-05-03** の作業記録として作成する。  
今回の区切りは、小さな単発コミットではなく、Phase 3 前半のまとまった到達点として扱う。

対象 repo:

```text
~/work/sglang-relaykv
```

対象 branch:

```text
relaykv-host-backup-shadow
```

push remote:

```text
mine
```

---

## 1. 今回の位置づけ

今回の作業は、RelayKV / SGLang 統合における **Phase 3 safe materialization 前半** の完了記録である。

Phase 2 までで、以下の read-only metadata plumbing が完了していた。

```text
ForwardBatch runtime observation metadata
→ host backup candidate summary
→ runtime/candidate join
→ readonly diagnostic report
→ dry-run policy events
→ materialization readiness assessment
```

今回の Phase 3 前半では、実KVを触る前に、materialization の schema・summary・readiness gate を先に固定した。

合言葉:

```text
KVを実体化する前に、materialization結果の形を実体化する。
```

English:

```text
Materialize the schema before materializing KV.
```

---

## 2. 今回完了した範囲

今回完了した Phase 3 前半の範囲:

```text
3.0 safe materialization design memo
3.1 fake materialization result helper + smoke
3.2 guarded no-op materialization helper + smoke
3.3 candidate-event materialization helper + smoke
3.4 materialization summary / readonly report / metadata-only attention readiness integration
```

現在位置:

```text
Phase 0: ForwardBatch metadata観測
  完了

Phase 1: runtime observation metadata と host backup candidate summary のjoin確認
  完了

Phase 2: dry-run policy plumbing
  完了

Phase 3: safe materialization
  3.0 design memo 完了
  3.1 fake materialization 完了
  3.2 guarded no-op materialization 完了
  3.3 candidate-event materialization 完了
  3.4 materialization report / readiness 統合 完了
```

---

## 3. Phase 3.0: Safe Materialization Design Memo

追加した設計メモ:

```text
notes/relaykv_phase3_safe_materialization_design_2026-05-03.ja.md
```

目的:

```text
実KV操作に入る前に、
materialization input schema
materialization output schema
状態定義
failure modes
safety counters
runtime接続禁止境界
を固定する。
```

Phase 3 の重要方針:

```text
fake result
→ no-op guarded result
→ candidate-event materialized result
→ host backup copy safety boundary
→ runtime/attention connection はさらに後
```

この設計により、Phase 3 の初期段階では以下を禁止することを明確化した。

```text
attention接続
scheduler decision変更
runtime writeback
KV cache free
RadixTree変更
HiCache prefetch hint
```

---

## 4. Phase 3.1: Fake Materialization Result

追加 helper:

```text
build_relaykv_fake_materialization_results_for_smoke(
    policy_dry_run_events,
    readiness=None,
)
```

```text
summarize_relaykv_materialization_results_for_smoke(results)
```

役割:

```text
dry-run policy event
→ relaykv_materialization_result
```

ただし、実KVは一切読まない。

通常時の result:

```text
materialization_state="fake_materialized"
materialization_mode="fake"
materialized_block_ids == selected_block_ids
retrieved_block_ids == selected_block_ids
host_backup_copy_executed=false
kv_pool_read=false
kv_snapshot=false
```

確認した smoke:

```text
scripts/relaykv_fake_materialization_smoke.py
```

主な確認内容:

```text
normal fake materialization
selected_block_ids=[] の skipped
readiness blocked
readiness missing warning
poison unrelated field 未アクセス
input non-mutation
safety counters all zero
```

---

## 5. Phase 3.2: Guarded No-op Materialization

追加 helper:

```text
build_relaykv_guarded_noop_materialization_results_for_smoke(
    host_backup_candidate_events,
    readiness=None,
)
```

役割:

```text
host backup candidate event
→ guarded_noop relaykv_materialization_result
```

この段階では、candidate event を見ても materialize せず、明示的に no-op で止める。

通常時の result:

```text
materialization_state="guarded_noop"
materialization_mode="noop_guarded"
materialized_block_ids=[]
retrieved_block_ids=[]
skipped_block_ids=selected_block_ids
host_backup_copy_executed=false
kv_pool_read=false
kv_snapshot=false
```

対応した alias:

```text
req_pool_idx / req_pool_index / request_pool_idx
layer_id / layer_idx
selected_block_ids
copied_block_ids
block_ids
block_id
candidate_block_id
candidate_block_ids
```

確認した smoke:

```text
scripts/relaykv_guarded_noop_materialization_smoke.py
```

主な確認内容:

```text
normal guarded no-op
alias mapping
empty selected
readiness blocked
readiness missing warning
poison unrelated field 未アクセス
input non-mutation
safety counters all zero
```

---

## 6. Phase 3.3: Candidate-event Materialization

追加 helper:

```text
build_relaykv_candidate_event_materialization_results_for_smoke(
    host_backup_candidate_events,
    readiness=None,
)
```

役割:

```text
host backup candidate event
→ candidate_event_materialized relaykv_materialization_result
```

これは実copyではない。  
candidate event payload を **metadata-only materialized candidate** として result schema に昇格する段階である。

通常時の result:

```text
materialization_state="candidate_event_materialized"
materialization_mode="candidate_event"
materialized_block_ids=resolved block ids
retrieved_block_ids=materialized_block_ids unless explicitly provided
host_backup_copy_executed=false
kv_pool_read=false
kv_snapshot=false
```

対応した mapping:

```text
selected_block_ids:
  selected_block_ids
  materialized_block_ids
  retrieved_block_ids
  copied_block_ids
  block_ids
  [block_id]
  [candidate_block_id]
  candidate_block_ids
  else []

materialized_block_ids:
  materialized_block_ids
  else selected_block_ids

retrieved_block_ids:
  retrieved_block_ids
  else materialized_block_ids

candidate_block_ids:
  candidate_block_ids
  else selected_block_ids
```

確認した smoke:

```text
scripts/relaykv_candidate_event_materialization_smoke.py
```

主な確認内容:

```text
normal candidate-event materialization
explicit retrieved_block_ids
alias mapping: block_id, candidate_block_id, block_ids, copied_block_ids
empty selected
readiness blocked
readiness missing warning
poison unrelated field 未アクセス
input non-mutation
safety counters all zero
```

---

## 7. Phase 3.4: Materialization Summary / Readonly Report 統合

拡張した helper:

```text
build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
    ...,
    materialization_summary=None,
)
```

追加した report fields:

```text
materialization_summary
materialization_summary_included
materialization_total_results
materialization_result_count
materialization_fake_count
materialization_guarded_noop_count
materialization_candidate_event_count
materialization_host_backup_copy_count
materialization_blocked_count
materialization_skipped_count
materialization_error_count
materialized_kv_count
materialized_token_count
host_backup_copy_executed_count
kv_pool_read_count
kv_snapshot_count
```

重要な変更:

```text
overall_safety_status が materialization summary 側の safety counters も fold するようになった。
```

metadata-only phase では、以下が非ゼロなら fail とする。

```text
host_backup_copy_executed_count > 0
kv_pool_read_count > 0
kv_snapshot_count > 0
attention_override_true_count > 0
runtime_writeback_true_count > 0
scheduler_policy_noop_false_count > 0
kv_cache_mutation_true_count > 0
source_mutated_true_count > 0
```

これにより、Phase 3 前半の report は **metadata-only safety envelope** を明示的に持つようになった。

---

## 8. Metadata-only Attention Readiness

追加 helper:

```text
assess_relaykv_readonly_attention_readiness_for_smoke(report)
```

これは attention を接続する helper ではない。  
あくまで、Phase 4 attention connection の設計に進めるだけの metadata-only report が揃ったかを判定する。

ready state:

```text
ready_for_attention_connection_metadata_only
```

ready 条件:

```text
report_generated_from_readonly_inputs=true
overall_safety_status="pass"
policy_dry_run_included=true
policy_dry_run_total_events > 0
materialization_summary_included=true
materialization_total_results > 0
materialization_result_count > 0
materialized_kv_count > 0
candidate_event_materialized_count > 0 OR fake_materialized_count > 0
guarded_noop_count == 0
blocked_count == 0
error_count == 0
host_backup_copy_executed_count == 0
kv_pool_read_count == 0
kv_snapshot_count == 0
attention_override_true_count == 0
runtime_writeback_true_count == 0
scheduler_policy_noop_false_count == 0
```

この ready は次を意味する。

```text
attentionを接続してよい
```

ではない。

正しくは次を意味する。

```text
Phase 4 attention接続を設計するための
metadata-only materialization report が揃った。
```

---

## 9. 更新した smoke flow

更新した smoke:

```text
scripts/relaykv_readonly_diagnostic_flow_smoke.py
scripts/relaykv_runtime_candidate_join_report_smoke.py
```

full diagnostic flow:

```text
runtime summary
→ host backup candidate summary
→ runtime/candidate join
→ policy dry-run events
→ materialization readiness
→ candidate-event materialization results
→ materialization summary
→ readonly report
→ metadata-only attention readiness
```

追加した fail cases:

```text
materialization summary missing
guarded_noop_count > 0
materialization blocked
materialization error
host_backup_copy_executed_count > 0
kv_pool_read_count > 0
materialized_kv_count == 0
```

既存の report smoke でも以下を確認した。

```text
materialization_summary なしの既存呼び出しは pass
materialization_summary ありで safety zero は pass
kv_pool_read_count=1 は overall_safety_status fail
```

---

## 10. 確認結果

Codex 実行結果として、以下を確認済み。

```text
py_compile: pass
git diff --check: pass
forbidden-area grep: 出力なし
```

pass した smoke:

```text
scripts/relaykv_readonly_diagnostic_flow_smoke.py
scripts/relaykv_runtime_candidate_join_report_smoke.py
scripts/relaykv_candidate_event_materialization_smoke.py
scripts/relaykv_guarded_noop_materialization_smoke.py
scripts/relaykv_fake_materialization_smoke.py
scripts/relaykv_policy_dry_run_smoke.py
scripts/relaykv_runtime_observation_host_backup_join_smoke.py
scripts/relaykv_host_backup_candidate_variation_smoke.py
scripts/relaykv_runtime_policy_smoke.py
```

未変更確認:

```text
forward_batch_info.py
model_runner.py
scheduler / managers
attention backend
memory_pool.py
flashinfer
.github/workflows
```

---

## 11. 現時点の安全境界

まだ実施していないこと:

```text
KV pool read
host backup copy実行
KV snapshot
attention接続
scheduler decision変更
runtime writeback
KV cache free
RadixTree変更
HiCache prefetch hint
```

今回実装されたのは、あくまで以下である。

```text
metadata-only materialization schema
metadata-only materialization summary
metadata-only readonly report
metadata-only attention readiness
```

したがって、現段階では SGLang runtime の実行挙動を変えない。

---

## 12. 成果

今回の成果は、RelayKV が Phase 4 attention 接続へ進む前に必要な、中間成果物を揃えたこと。

具体的には:

```text
dry-run policy event
→ fake / guarded_noop / candidate-event materialization result
→ materialization summary
→ readonly report
→ metadata-only attention readiness
```

までの流れが smoke で確認できた。

これにより、今後実KVに近づく場合でも、以下を report 上で検出できる。

```text
host backup copy が実行されたか
KV pool を読んだか
snapshot を取ったか
attention override したか
runtime writeback したか
scheduler non-noop が発生したか
```

つまり、Phase 3 前半で **実KV操作に進む前の安全な検査面** ができた。

---

## 13. 次にやること

次はまだ attention 接続ではない。

推奨する次ステップ:

```text
Phase 3.5 host backup copy safety boundary design
```

目的:

```text
actual host backup copy helper を使う前に、
どの関数が何を読み、
どの counter を立て、
どこで止めるかを設計する。
```

次に固定すべき項目:

```text
host backup copy input schema
host backup copy output schema
host_backup_copy_executed_count の意味
kv_pool_read_count との違い
kv_snapshot_count との違い
candidate_event_materialized と host_backup_copy_materialized の境界
failure modes
readiness gate
runtime接続禁止境界
```

まだ避けるべきこと:

```text
attention接続
scheduler変更
runtime writeback
KV cache free
RadixTree変更
HiCache連携
```

---

## 14. commit / push command

この devlog を repo に保存する場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/devlog_2026-05-03_relaykv_phase3_metadata_only_materialization_readiness.ja.md \
  notes/devlog_2026-05-03_relaykv_phase3_metadata_only_materialization_readiness.ja.md

git status --short
git diff --check

git add notes/devlog_2026-05-03_relaykv_phase3_metadata_only_materialization_readiness.ja.md
git commit -m "docs: record relaykv phase3 metadata-only materialization readiness"
git push mine relaykv-host-backup-shadow
```
