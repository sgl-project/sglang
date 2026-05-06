# Devlog: RelayKV Phase 4.8.5.4 Live token_to_kv_pool Bounded Index Read

## 日付確認

- Devlog date: **2026-05-06**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. この devlog の範囲

この devlog は、RelayKV Phase 4.8.5.4 系の進捗をまとめる。

対象範囲:

```text
Phase 4.8.5.4:
  Live token_to_kv_pool bounded index read design

Phase 4.8.5.4.1:
  Fake/guarded live token_to_kv_pool index read smoke

Phase 4.8.5.4.2:
  ModelRunner default-off live physical-index read hook

Phase 4.8.5.4.3:
  Optional server live physical-index read smoke
```

この範囲で、RelayKV は **live-like / server runtime 上の token_to_kv_pool index read 入口**まで進んだ。

ただし、引き続き以下には触っていない。

```text
KV pool read
KV pool snapshot
K/V tensor read
attention execution
attention override
scheduler mutation
runtime writeback
source mutation
```

---

## 2. 背景

Phase 4.8.5 までに、RelayKV は SGLang runtime 上で次の metadata chain を形成していた。

```text
attention shadow capture result
→ KV index resolution plan
→ req_to_token resolution result
→ token_to_kv_pool readonly resolution
→ physical KV index metadata
→ engine_block_ref
```

Phase 4.8.5.3 系では、実 server 上で `token_to_kv_pool` の metadata-only inspection 入口まで到達した。

Phase 4.8.5.4 の目的は、その次の段階として、実 index read に近い形で以下を確認することだった。

```text
req_to_token resolution result
→ live-like token_to_kv_pool object
→ bounded token_to_kv_pool index read
→ physical_kv_index preview / count / checksum
→ engine_block_ref
```

---

## 3. Phase 4.8.5.4 Design

Design file:

```text
notes/relaykv_phase4_8_5_4_live_token_to_kv_pool_bounded_index_read_design_2026-05-05.ja.md
```

設計上の重要点:

```text
token_to_kv_pool の index read は許可する。
ただし、KV pool / K/V tensor / attention backend には触らない。
```

許可した read:

```text
physical_index = token_to_kv_pool[req_to_token_entry]
```

ただし、以下の制約を付けた。

```text
explicit flag ON の場合のみ
bounded token budget 内のみ
req_to_token resolution result で解決済みの entry のみ
token_to_kv_pool object への index read のみ
output は bounded preview / count / checksum のみ
full physical index list は出力しない
```

新しい env flag:

```text
SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ=1
```

この flag は、既存の metadata-only inspection flag と分離した。

```text
SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INSPECTION=1
```

理由:

```text
metadata inspection と index read はリスク段階が異なるため。
```

---

## 4. Phase 4.8.5.4.1 Fake/guarded live index read smoke

変更ファイル:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_live_token_to_kv_pool_index_read_smoke.py
```

追加 helper:

```text
build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(...)
summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(...)
```

実装内容:

```text
req_to_token resolution result payload
→ fake / guarded live-like token_to_kv_pool object
→ bounded index read
→ physical_kv_index preview / count / checksum
→ engine_block_ref
```

対応した token_to_kv_pool source:

```text
dict
list
tuple
controlled __getitem__ object
```

出力 event:

```text
event_type="relaykv_live_token_to_kv_pool_index_read_result"
resolution_state="physical_kv_index_resolved" | "blocked" | "error"
adapter_mode="live_token_to_kv_pool_bounded_index_read"
decision_state="SHADOW_ONLY"
```

`engine_block_ref` には以下だけを出す。

```text
physical_kv_index_preview
physical_kv_index_count
physical_kv_index_checksum
token_to_kv_pool_index=None
cache_position=None
```

重要:

```text
full physical index list は出さない。
```

確認した blocked / safety cases:

```text
read flag off
missing token_to_kv_pool object
unsupported object
missing index
non-int req_to_token entry
non-int physical index
max_tokens_per_request exceeded
max_total_tokens exceeded
index read failure
poison object
source object mutationなし
schema fields preserved
reserved retrieval metadata fields not computed
```

許可された counter:

```text
token_to_kv_pool_read_count > 0
actual_token_to_kv_pool_read_count > 0
live_token_to_kv_pool_index_read_count > 0
```

維持した zero counters:

```text
req_to_token_read_count=0
actual_req_to_token_pool_read_count=0
kv_pool_read_count=0
kv_snapshot_count=0
tensor_read_count=0
attention_comparison_executed_count=0
attention_override_true_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
kv_cache_mutation_true_count=0
source_mutated_true_count=0
```

---

## 5. Phase 4.8.5.4.2 ModelRunner default-off hook

変更ファイル:

```text
python/sglang/srt/model_executor/model_runner.py
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_fake_model_runner_live_token_to_kv_pool_index_read_smoke.py
```

`metrics.py` には、`ModelRunner.forward` から安全に呼ぶための minimal wrapper を追加した。

追加 wrapper:

```text
run_model_runner_live_token_to_kv_pool_index_read_hook_for_smoke(...)
```

この wrapper の目的:

```text
explicit token_to_kv_pool lookup path を再利用する
req_to_token resolution payloads が無い場合は clean blocked summary を返す
Phase 4.8.5.4.1 の build/summarize helper を呼ぶ
KV pool / tensor / attention / scheduler には触らない
```

`ModelRunner.forward` 側の env flag:

```text
SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ
```

env off:

```text
no lazy import
no hook call
no log emission
forward unchanged
```

env on:

```text
lazy import
bounded live token_to_kv_pool index read hook
relaykv_live_token_to_kv_pool_index_read_summary log
exception-safe
forward unchanged
```

ログ marker:

```text
relaykv_live_token_to_kv_pool_index_read_summary=...
```

fake forward smoke で確認したこと:

```text
env off: no import / no hook / no log
env on with payloads: summary emitted and bounded read counters > 0
env on without payloads: clean blocked summary
missing token_to_kv_pool path: clean blocked summary
hook exception: swallowed
forward return unchanged
poison object: forbidden accessなし
```

---

## 6. Phase 4.8.5.4.3 Optional server smoke

変更ファイル:

```text
scripts/relaykv_optional_server_live_token_to_kv_pool_index_read_smoke.py
```

この smoke は既存の optional server smoke と同じ構造にした。

実行条件:

```text
RELAYKV_OPTIONAL_SERVER_SMOKE_RUN=1
RELAYKV_OPTIONAL_SERVER_SMOKE_MODEL=<local model path>
```

未設定時:

```text
clean skip
exit 0
```

実 server で確認したこと:

```text
env off:
  HTTP 200
  no relaykv_live_token_to_kv_pool_index_read_summary log

env on:
  HTTP 200
  relaykv_live_token_to_kv_pool_index_read_summary log emitted
  response marker unchanged
  safety counters zero
```

今回の real server run では、env on summary は clean blocked になった。

理由:

```text
req_to_token resolution payloads がその server path ではまだ利用可能ではなかったため。
```

これは failure ではなく、今回の pass condition に含めた。

理由:

```text
Phase 4.8.5.4.3 の目的は、
server 上で live index-read hook が default-off / env-on / exception-safe / output-unchanged に動くことを確認すること。
実 resolved path は、req_to_token resolution payload を runtime chain に接続する次フェーズの対象。
```

---

## 7. 検証

Phase 4.8.5.4.1 で通した検証:

```text
py_compile for metrics.py and live index-read smoke
scripts/relaykv_live_token_to_kv_pool_index_read_smoke.py
scripts/relaykv_reserved_retrieval_metadata_fields_smoke.py
scripts/relaykv_token_to_kv_pool_runtime_inspection_smoke.py
scripts/relaykv_token_to_kv_pool_readonly_adapter_smoke.py
scripts/relaykv_physical_kv_index_resolution_smoke.py
scripts/relaykv_req_to_token_resolution_smoke.py
scripts/relaykv_sglang_adapter_schema_alignment_smoke.py
git diff --check
forbidden-path grep empty
```

Phase 4.8.5.4.2 で通した検証:

```text
py_compile for model_runner.py and fake forward smoke
scripts/relaykv_fake_model_runner_live_token_to_kv_pool_index_read_smoke.py
scripts/relaykv_live_token_to_kv_pool_index_read_smoke.py
scripts/relaykv_fake_model_runner_token_to_kv_pool_inspection_smoke.py
scripts/relaykv_model_runner_token_to_kv_pool_inspection_hook_smoke.py
scripts/relaykv_token_to_kv_pool_runtime_inspection_smoke.py
scripts/relaykv_physical_kv_index_resolution_smoke.py
scripts/relaykv_req_to_token_resolution_smoke.py
git diff --check
forbidden-path grep empty
```

Phase 4.8.5.4.3 で通した検証:

```text
py_compile for optional server live index-read smoke
clean skip path
real optional server smoke with local model
scripts/relaykv_fake_model_runner_live_token_to_kv_pool_index_read_smoke.py
scripts/relaykv_live_token_to_kv_pool_index_read_smoke.py
scripts/relaykv_optional_server_token_to_kv_pool_inspection_smoke.py
scripts/relaykv_fake_model_runner_token_to_kv_pool_inspection_smoke.py
scripts/relaykv_token_to_kv_pool_runtime_inspection_smoke.py
scripts/relaykv_physical_kv_index_resolution_smoke.py
scripts/relaykv_req_to_token_resolution_smoke.py
git diff --check
forbidden-path grep empty
```

Optional server real run:

```text
RELAYKV_OPTIONAL_SERVER_SMOKE_MODEL=/home/rinsa/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
RELAYKV_OPTIONAL_SERVER_SMOKE_RUN=1
```

---

## 8. 安全不変条件

Phase 4.8.5.4 で初めて許可したもの:

```text
token_to_kv_pool bounded index read
```

許可 counter:

```text
token_to_kv_pool_read_count > 0
actual_token_to_kv_pool_read_count > 0
live_token_to_kv_pool_index_read_count > 0
```

ただし、以下は常に 0 を維持する。

```text
req_to_token_read_count=0
actual_req_to_token_pool_read_count=0
kv_pool_read_count=0
kv_snapshot_count=0
tensor_read_count=0
attention_comparison_executed_count=0
attention_override_true_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
kv_cache_mutation_true_count=0
source_mutated_true_count=0
```

このフェーズの境界:

```text
OK:
  token_to_kv_pool[req_to_token_entry] を bounded に読む

NG:
  req_to_token live value/index read
  k_buffer[physical_index]
  v_buffer[physical_index]
  kv_pool snapshot
  attention backend input replacement
  scheduler decision change
```

---

## 9. 現在の到達点

現時点で、RelayKV の Phase 4 chain は次の段階まで到達した。

```text
ForwardBatch runtime observation
→ host backup candidate summary join
→ dry-run policy
→ safe materialization metadata
→ host backup copy boundary
→ attention handoff metadata
→ attention shadow capture metadata
→ KV index resolution plan
→ req_to_token resolution result
→ token_to_kv_pool metadata inspection
→ fake/guarded live token_to_kv_pool bounded index read
→ ModelRunner default-off live index-read hook
→ optional server live index-read hook smoke
```

ただし、実 server path ではまだ `req_to_token resolution payloads` が接続されていないため、env-on live index-read hook は clean blocked になる。

これは次の実装ポイントである。

---

## 10. まだ未実施のこと

未実施:

```text
server runtime path への req_to_token resolution payload 接続
live server 上での resolved physical index read
KV pool read
K/V tensor read
working KV assembly
shadow attention compute
real attention override
scheduler integration
runtime writeback
residual VRAM budget enforcement
```

また、今回追加した retrieval-critical reserved metadata fields は、あくまで schema/log の将来受け皿であり、profiling 本体はまだ実装していない。

---

## 11. 次の推奨ステップ

次は以下が自然。

```text
Phase 4.8.6:
  Runtime req_to_token resolution payload bridge for live index read
```

目的:

```text
ModelRunner.forward / runtime hook path で、
既存の req_to_token resolution result payload を live token_to_kv_pool index-read hook に渡せるようにする。
```

ただし、引き続き以下を禁止する。

```text
live req_to_token value read
KV pool read
tensor read
attention override
scheduler mutation
runtime writeback
```

実装方針:

```text
1. まず design memo
2. 次に fake bridge smoke
3. 次に ModelRunner default-off bridge hook
4. 最後に optional server smoke
```

期待する chain:

```text
req_to_token resolution result payload
→ live token_to_kv_pool index-read hook
→ physical_kv_index preview/count/checksum
→ server summary log
```

---

## 12. commit commands

Phase 4.8.5.4.3 の optional server smoke が未 commit の場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

git status --short
git diff --check

git add scripts/relaykv_optional_server_live_token_to_kv_pool_index_read_smoke.py

git commit -m "relaykv: add optional live index-read server smoke"
git push mine relaykv-host-backup-shadow
```

この devlog を repo に保存する場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/devlog_2026-05-06_relaykv_phase4_8_5_4_live_token_to_kv_pool_index_read.ja.md \
  notes/devlog_2026-05-06_relaykv_phase4_8_5_4_live_token_to_kv_pool_index_read.ja.md

git status --short
git diff --check

git add notes/devlog_2026-05-06_relaykv_phase4_8_5_4_live_token_to_kv_pool_index_read.ja.md
git commit -m "docs: add relaykv phase 4.8.5.4 devlog"
git push mine relaykv-host-backup-shadow
```
