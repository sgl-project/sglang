# Devlog: RelayKV read-only runtime/candidate/join report helper

## 日付確認

このdevlogは JST 基準で `2026-05-03` の作業記録として作成する。

Branch: `relaykv-host-backup-shadow`  
Repo: `~/work/sglang-relaykv`

## 今日の目的

RelayKV の metadata plumbing をさらに一段進め、以下を1つの read-only report として扱えるようにする。

```text
runtime observation summary
+
host backup candidate summary
+
runtime/candidate join summary
+
overall_safety_status
```

前段階では、runtime observation payload と host backup candidate event を `request_id + req_pool_idx + layer_id` で join できるところまで確認した。

今回の目的は、その join 結果を含めて、将来の runtime log / smoke / report で扱いやすい summary report 形式へまとめること。

この段階でも、まだ以下には進まない。

- host backup copy 実行
- KV pool read
- KV snapshot
- attention 接続
- scheduler decision 変更
- runtime writeback

## 背景

ここまでの到達点:

```text
1. ForwardBatch read-only metadata
2. runtime observation summary
3. host backup candidate summary
4. runtime/candidate join summary
```

前回追加した helper:

```text
join_runtime_observation_with_host_backup_candidates_for_smoke(...)
```

この helper により、runtime observation payload と candidate event を summary level で対応づけられるようになった。

今回の作業は、その上に report layer を追加するもの。

## 今回の変更

追加した read-only report helper:

```text
build_relaykv_readonly_runtime_candidate_join_report_for_smoke(...)
```

目的:

```text
runtime_observation_summary
host_backup_candidate_summary
join_summary

を1つの report dict にまとめる。
```

## 変更したファイル

- `python/sglang/srt/relaykv/metrics.py`
- `scripts/relaykv_runtime_candidate_join_report_smoke.py`

## report helper の仕様

入力:

```text
runtime_observation_summary: dict
host_backup_candidate_summary: dict
join_summary: dict
```

出力:

```text
report_type="relaykv_readonly_runtime_candidate_join_report"
report_generated_from_readonly_inputs=true
runtime_observation_summary
host_backup_candidate_summary
join_summary
overall_safety_status
```

主要集約 field:

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

## safety status rule

`overall_safety_status` は以下の条件で判定する。

```text
pass:
  report_generated_from_readonly_inputs=true
  かつ safety counters がすべて 0

fail:
  いずれかの safety counter が nonzero
  または readonly input 由来でない
```

これにより、read-only report として安全に扱えるかを1つの field で確認できる。

## missing field 対応

component summary に不足 field があっても、不要に crash しない。

方針:

```text
安全に default できる値は conservative default を使う
不足 field は missing_field_counts / warning 系 field で扱う
```

目的:

```text
smoke / report の初期段階で、summary形式が揺れても診断できるようにする
```

## 新規 smoke

追加:

```text
scripts/relaykv_runtime_candidate_join_report_smoke.py
```

確認内容:

### pass case

fake summaries:

```text
runtime_observation_summary:
  total_payloads=4

host_backup_candidate_summary:
  total_candidate_events=3

join_summary:
  joined_count=2
  unmatched_runtime_count=2
  unmatched_candidate_count=1
  req_pool_idx_joined_count=2
  req_pool_idx_missing_count=0
  join_granularity="per_event"
```

期待値:

```text
report_type correct
report_generated_from_readonly_inputs=true
overall_safety_status="pass"
joined_count=2
unmatched_runtime_count=2
unmatched_candidate_count=1
all safety counters zero
inputs not mutated
```

### fail case

確認内容:

```text
one safety counter nonzero
overall_safety_status="fail"
```

### summary-only unjoinable case

確認内容:

```text
join_granularity="summary_only_unjoinable"
report still builds
overall_safety_status depends only on safety counters
```

## 確認結果

実行した確認:

```bash
PYTHONPATH=python .venv/bin/python -m py_compile   python/sglang/srt/relaykv/metrics.py   scripts/relaykv_runtime_candidate_join_report_smoke.py

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

- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/model_executor/model_runner.py`
- scheduler
- `schedule_batch.py`
- attention backend
- flashinfer
- `memory_pool.py`
- `.github/workflows`

## commit / push

今回の変更は commit / push 済み。

commit message:

```text
relaykv: add readonly runtime candidate join report
```

push remote:

```text
mine relaykv-host-backup-shadow
```

## 現在の到達点

```text
runtime observation summary
+
host backup candidate summary
+
join summary
+
overall_safety_status

を1つの read-only report として扱える
```

RelayKV/SGLang metadata plumbing の段階は以下まで進んだ。

```text
1. ForwardBatch read-only metadata
2. runtime observation summary
3. host backup candidate summary
4. runtime/candidate join summary
5. readonly runtime candidate join report
```

## 維持できている安全境界

まだ実施していないこと:

```text
host backup copy実行
KV pool read
KV snapshot
attention接続
scheduler decision変更
runtime writeback
```

今回の report helper も pure summary / read-only report layer に限定している。

## 判断

今回の実装は安全な前進。

理由:

```text
runtime summary / candidate summary / join summary をまとめるだけ
実runtime stateに触れない
KV poolに触れない
host backup copyを実行しない
attentionに接続しない
scheduler decisionを変えない
```

特に重要なのは、report layer が `overall_safety_status` を持つことで、以後の runtime log や smoke において安全状態を1つの field で見られるようになったこと。

## 次の分岐

次はまだ実copyに進まず、reportを runtime observation hook の log へ接続するかを検討する段階。

候補:

```text
A. read-only report helper を hook smoke / fake runtime path に接続する
B. report helper は smoke専用に留め、次に host backup candidate event の runtime側発生点を整理する
C. ここで metadata plumbing を一度まとめ、設計メモに反映する
```

おすすめは A か C。

ただし、実 copy / KV pool read へ進む前に、以下を確認した方が安全。

```text
runtime hook 上で report を出す場合、
host backup candidate event がどのタイミングで存在するか
```

まだ candidate event が runtime hook と同じ場所に存在しないなら、無理に hook 接続せず、fake smoke / report helper のまま維持する。
