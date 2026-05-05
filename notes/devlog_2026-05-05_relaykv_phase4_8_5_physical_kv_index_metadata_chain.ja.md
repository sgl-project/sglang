# Devlog: RelayKV Phase 4.8.5 Physical KV Index Metadata Chain

## 日付確認

- Devlog date: **2026-05-05**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. この devlog の範囲

この devlog は、以下の Phase 4.8.5 系の進捗をまとめる。

```text
Phase 4.8.5-pre:
  SGLang Adapter Schema Alignment

Phase 4.8.5:
  Physical KV Index Readonly Resolution Design

Phase 4.8.5.1:
  Synthetic token_to_kv_pool readonly resolution smoke

Phase 4.8.5.2:
  Fake actual token_to_kv_pool adapter payload smoke

Phase 4.8.5.3:
  Runtime token_to_kv_pool metadata inspection smoke

Phase 4.8.5.3.1:
  ModelRunner token_to_kv_pool inspection wrapper fake-object smoke

Phase 4.8.5.3.2:
  ModelRunner default-off token_to_kv_pool inspection hook wiring

Phase 4.8.5.3.3:
  Optional server token_to_kv_pool metadata inspection smoke
```

この範囲で、RelayKV Phase 4 は以下の metadata chain を形成した。

```text
attention shadow capture result
→ KV index resolution plan
→ req_to_token resolution result
→ token_to_kv_pool readonly resolution
→ physical KV index metadata
→ engine_block_ref
```

---

## 2. 目的

Phase 4.8.5 系の目的は、SGLang runtime における logical token / block から physical KV index へ進むための metadata path を、安全に段階化することだった。

ただし、この段階ではまだ以下を行わない。

```text
KV tensor read
KV pool snapshot
K/V buffer read
attention execution
attention override
working KV assembly
scheduler mutation
runtime writeback
```

RelayKV は引き続き **read-only / metadata-only / shadow-only** の範囲に留める。

---

## 3. Phase 4.8.5-pre: SGLang Adapter Schema Alignment

まず、SGLang runtime 固有の payload を RelayKV Core から切り離すため、adapter schema alignment を行った。

主な方針:

```text
RelayKV Core:
  engine-independent fields

SGLang Adapter:
  SGLang-specific runtime metadata
```

追加・整理した主な概念:

```text
engine_name="sglang"
adapter_name="sglang"
engine_request_id
logical_sequence_id
logical_block_id
token_span
layer_id
kv_head_group
decision_state
fallback_reason
adapter_metadata
engine_block_ref
```

重要な設計判断:

```text
logical_block_id と physical engine_block_ref を分離する。
```

これにより、RelayKV Core は SGLang の `req_to_token`, `token_to_kv_pool`, `ForwardBatch`, `RadixAttention` などに直接依存しない構造になった。

---

## 4. Phase 4.8.5: Physical KV Index Readonly Resolution Design

Physical KV index resolution の設計メモを作成した。

Design file:

```text
notes/relaykv_phase4_8_5_physical_kv_index_readonly_resolution_design_2026-05-05.ja.md
```

この設計では、次の chain を明確化した。

```text
logical_block_id
→ token_span
→ req_to_token entry
→ token_to_kv_pool entry
→ physical KV index
→ engine_block_ref
```

ただし、実 runtime で physical index を読む前に、以下の段階を設ける方針にした。

```text
1. synthetic table
2. fake actual object
3. runtime metadata-only inspection
4. ModelRunner default-off hook
5. optional server metadata smoke
```

---

## 5. Phase 4.8.5.1: Synthetic token_to_kv_pool readonly resolution smoke

Synthetic table を使って、`token_to_kv_pool` lookup の read-only resolution を実装した。

変更ファイル:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_physical_kv_index_resolution_smoke.py
```

追加 helper:

```text
build_relaykv_physical_kv_index_resolution_results_for_smoke(...)
summarize_relaykv_physical_kv_index_resolution_results_for_smoke(...)
```

確認したこと:

```text
req_to_token resolution result
→ synthetic token_to_kv_pool table
→ bounded readonly lookup
→ physical KV index metadata
→ engine_block_ref
```

この段階では、`token_to_kv_pool_read_count > 0` を初めて許可した。

ただし許可範囲は以下に限定した。

```text
synthetic bounded table only
dict/list/tuple only
bounded preview only
no runtime object access
no KV pool read
no tensor read
```

安全 counters:

```text
token_to_kv_pool_read_count > 0  # synthetic bounded table only

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

## 6. Phase 4.8.5.2: Fake actual token_to_kv_pool adapter payload smoke

次に、synthetic table ではなく fake object の `.token_to_kv_pool` attr 境界を controlled に確認した。

変更ファイル:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_token_to_kv_pool_readonly_adapter_smoke.py
```

追加 helper:

```text
build_relaykv_token_to_kv_pool_readonly_adapter_payloads_for_smoke(...)
summarize_relaykv_token_to_kv_pool_readonly_adapter_payloads_for_smoke(...)
```

許可した境界:

```text
getattr(obj, "token_to_kv_pool", None)
```

禁止したこと:

```text
recursive search
dir()
vars()
repr()
arbitrary object scan
unbounded table dump
KV pool read
tensor read
attention override
scheduler mutation
runtime writeback
```

確認した chain:

```text
req_to_token resolution result
→ fake object getattr(obj, "token_to_kv_pool", None)
→ bounded token_to_kv_pool readonly lookup
→ physical KV index metadata
→ engine_block_ref
```

安全 counters:

```text
token_to_kv_pool_read_count > 0
actual_token_to_kv_pool_read_count > 0

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

## 7. Phase 4.8.5.3: Runtime token_to_kv_pool metadata inspection smoke

次に、runtime-like object 上で `token_to_kv_pool` の metadata-only inspection を実装した。

変更ファイル:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_token_to_kv_pool_runtime_inspection_smoke.py
```

追加 helper:

```text
build_relaykv_token_to_kv_pool_runtime_inspection_payloads_for_smoke(...)
summarize_relaykv_token_to_kv_pool_runtime_inspection_payloads_for_smoke(...)
run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(...)
```

観測対象:

```text
token_to_kv_pool object / attr
```

観測する metadata:

```text
type
module
qualname
shape
device
dtype
```

この段階では **value/index read は禁止**。

確認したこと:

```text
runtime-like object
→ explicit path lookup only
→ token_to_kv_pool metadata observation
→ type / module / qualname / shape / device / dtype
→ no value/index read
```

安全 counters:

```text
token_to_kv_pool_read_count=0
actual_token_to_kv_pool_read_count=0
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

## 8. Phase 4.8.5.3.1: ModelRunner token_to_kv_pool inspection wrapper fake-object smoke

次に、ModelRunner / ForwardBatch 風の fake object から、明示 path のみで `token_to_kv_pool` を探す wrapper を確認した。

変更ファイル:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_model_runner_token_to_kv_pool_inspection_hook_smoke.py
```

確認した explicit path:

```text
model_runner.token_to_kv_pool
model_runner.token_to_kv_pool_allocator.token_to_kv_pool
model_runner.token_to_kv_pool_allocator
model_runner.kv_pool_allocator.token_to_kv_pool
model_runner.memory_pool.token_to_kv_pool
forward_batch.token_to_kv_pool
```

重要な修正:

```text
explicit-path lookup error を terminal blocked outcome にした。
```

これにより、想定外 path を探索し続ける危険を下げた。

また、観測 source path を以下に記録するようにした。

```text
adapter_metadata.token_to_kv_pool_source_path
```

確認した chain:

```text
fake ModelRunner / fake ForwardBatch
→ explicit path lookup
→ token_to_kv_pool metadata-only inspection
→ adapter_metadata.token_to_kv_pool_source_path
→ all read/mutation counters zero
```

---

## 9. Phase 4.8.5.3.2: ModelRunner default-off token_to_kv_pool inspection hook wiring

次に、実 `ModelRunner.forward` に default-off の metadata-only hook を wiring した。

変更ファイル:

```text
python/sglang/srt/model_executor/model_runner.py
scripts/relaykv_fake_model_runner_token_to_kv_pool_inspection_smoke.py
```

env flag:

```text
SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INSPECTION
```

挙動:

```text
unset / not "1":
  no RelayKV import
  no hook call
  no log emission
  forward behavior unchanged

"1":
  lazy import
  metadata-only inspection
  compact summary log
  exception-safe
  forward output unchanged
```

ログ marker:

```text
relaykv_token_to_kv_pool_runtime_inspection_summary
```

確認した chain:

```text
ModelRunner.forward
→ SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INSPECTION == "1" のときだけ
→ lazy import
→ metadata-only token_to_kv_pool inspection
→ relaykv_token_to_kv_pool_runtime_inspection_summary log
→ forward output unchanged
```

安全境界:

```text
env off:
  no import / no hook / no log

env on:
  metadata-only
  exception-safe
  no production output mutation
```

---

## 10. Phase 4.8.5.3.3: Optional server token_to_kv_pool metadata inspection smoke

最後に、実 server を起動して env off / on の差分を確認する optional smoke を追加した。

変更ファイル:

```text
scripts/relaykv_optional_server_token_to_kv_pool_inspection_smoke.py
```

この smoke は既存の optional server smoke と同じ形式にした。

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

確認したこと:

```text
env off:
  HTTP 200
  no token_to_kv inspection import/log/hook

env on:
  HTTP 200
  relaykv_token_to_kv_pool_runtime_inspection_summary log emitted
  metadata-only summary parsed
  response marker unchanged
  safety counters zero
```

確認した chain:

```text
ModelRunner.forward
→ env off: no import / no hook / no log
→ env on: token_to_kv_pool metadata-only inspection
→ optional server /generate HTTP 200
→ relaykv_token_to_kv_pool_runtime_inspection_summary log
→ response marker unchanged
→ safety counters zero
```

---

## 11. 検証

各フェーズで以下を実施した。

```text
py_compile
new smoke
regression subset
git diff --check
forbidden-file grep
```

Optional server smoke は、提供された local model path と以下の env で通過した。

```text
RELAYKV_OPTIONAL_SERVER_SMOKE_RUN=1
RELAYKV_OPTIONAL_SERVER_SMOKE_MODEL=/home/rinsa/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
```

---

## 12. 重要な安全不変条件

Phase 4.8.5 系全体で維持した安全境界:

```text
KV pool read = 0
KV snapshot = 0
tensor read = 0
K/V buffer read = 0
attention execution = 0
attention override = 0
runtime writeback = 0
scheduler mutation = 0
KV cache mutation = 0
source mutation = 0
```

Phase 4.8.5.1 / 4.8.5.2 では、bounded synthetic/fake object flow に限って以下を許可した。

```text
token_to_kv_pool_read_count > 0
actual_token_to_kv_pool_read_count > 0
```

Phase 4.8.5.3 以降の runtime metadata inspection では、再び以下をゼロにした。

```text
token_to_kv_pool_read_count=0
actual_token_to_kv_pool_read_count=0
```

理由:

```text
runtime object では、まず metadata-only inspection に限定し、
live token_to_kv_pool index read は次の明示フェーズに分離するため。
```

---

## 13. 現在の到達点

現時点で、RelayKV は SGLang 上で以下を metadata-only に観測できる段階に到達した。

```text
ForwardBatch request metadata
req_pool_idx
seq_len
runtime observation payload
host backup candidate summary
dry-run policy event
safe materialization metadata
host backup copy boundary
attention handoff metadata
attention comparison plan
attention shadow capture metadata
KV index resolution plan
req_to_token metadata / bounded fake resolution
token_to_kv_pool metadata / bounded fake resolution
physical KV index metadata under engine_block_ref
```

特に Phase 4.8.5 系で、以下の2本が揃った。

```text
req_to_token:
  fake object adapter
  runtime metadata inspection
  ModelRunner default-off hook
  optional server smoke

token_to_kv_pool:
  synthetic resolution
  fake object adapter
  runtime metadata inspection
  ModelRunner default-off hook
  optional server smoke
```

これにより、次の段階で live runtime object から physical index を読む前の安全境界が整った。

---

## 14. まだ未実施のこと

まだ実施していないこと:

```text
live runtime token_to_kv_pool index read
live runtime physical KV index resolution
KV tensor read
KV pool snapshot
working KV assembly
shadow attention compute
real attention override
scheduler integration
runtime writeback
residual VRAM budget enforcement
```

RelayKV はまだ SGLang の attention path を変更していない。

---

## 15. 次の推奨フェーズ

次は以下が自然。

```text
Phase 4.8.5.4:
  Live token_to_kv_pool bounded index read design

Phase 4.8.5.4.1:
  Fake/guarded live-token-to-physical-index read smoke

Phase 4.8.5.4.2:
  ModelRunner default-off live physical-index read hook

Phase 4.8.5.4.3:
  Optional server live physical-index read smoke
```

ただし、ここからは runtime object の index read を許可し始めるため、Phase 4.8.5.1 / 4.8.5.2 より一段リスクが高い。

推奨方針:

```text
1. design memo を先に作る
2. read count を明示的に分離する
3. bounded token budget を入れる
4. index read は token_to_kv_pool のみ
5. KV pool / tensor read はまだ禁止
6. optional server smoke でも response marker unchanged を維持
```

---

## 16. commit command

Phase 4.8.5.3.3 の optional server smoke がまだ未 commit の場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

git status --short
git diff --check

git add scripts/relaykv_optional_server_token_to_kv_pool_inspection_smoke.py

git commit -m "relaykv: add optional token-to-kv-pool server smoke"
git push mine relaykv-host-backup-shadow
```

この devlog を repo に保存する場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/devlog_2026-05-05_relaykv_phase4_8_5_physical_kv_index_metadata_chain.ja.md \
  notes/devlog_2026-05-05_relaykv_phase4_8_5_physical_kv_index_metadata_chain.ja.md

git status --short
git diff --check

git add notes/devlog_2026-05-05_relaykv_phase4_8_5_physical_kv_index_metadata_chain.ja.md
git commit -m "docs: add relaykv phase 4.8.5 devlog"
git push mine relaykv-host-backup-shadow
```
