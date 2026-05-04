# RelayKV Devlog: Phase 4.8 KV Index / Req-to-token / Runtime Inspection Chain

## 日付確認

- Devlog date: **2026-05-04**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. 概要

この devlog は、RelayKV/SGLang integration の Phase 4.8 系の進捗をまとめる。

対象範囲:

```text
Phase 4.8.1  Isolated KV Index Resolution Plan Smoke
Phase 4.8.2  Req-to-token Synthetic Readonly Resolution Smoke
Phase 4.8.3  Real Req-to-token Pool Readonly Adapter Design
Phase 4.8.3.1 Req-to-token Readonly Adapter Payload Smoke
Phase 4.8.4  Actual Req-to-token Pool Readonly Adapter Design
Phase 4.8.4.1 Actual Req-to-token Pool Adapter Fake-object Smoke
Phase 4.8.4.2 Optional Runtime Req-to-token Adapter Inspection Design
Phase 4.8.4.2.1 Runtime Req-to-token Metadata Inspection Smoke
Phase 4.8.4.3 Optional Server Runtime Inspection Design
Phase 4.8.4.3.1 Server Hook Wrapper Fake-object Smoke
Phase 4.8.4.3.2 ModelRunner default-off hook wiring
Phase 4.8.4.3.3 Optional Server Req-to-token Inspection Smoke
```

今回の大きな到達点は、RelayKV の metadata-only chain が、実 SGLang server 上で `model_runner.req_to_token_pool` を metadata-only に観測できるところまで進んだこと。

ただし、まだ以下は行っていない。

```text
req_to_token value/index read
token_to_kv_pool read
KV pool read
KV snapshot
K/V tensor read
attention comparison execution
attention override
scheduler mutation
runtime writeback
```

---

## 2. 現在の chain

現在の chain:

```text
attention_handoff_candidate
→ attention_connection_dry_run_result
→ attention_override_noop_result
→ attention_comparison_plan
→ attention_shadow_capture_result
→ relaykv_kv_index_resolution_plan
→ relaykv_req_to_token_resolution_result
→ relaykv_req_to_token_readonly_adapter_payload
→ req_to_token runtime metadata inspection
→ optional server req_to_token metadata summary
```

Phase 4.8 系で伸びた部分:

```text
relaykv_kv_index_resolution_plan
→ synthetic req_to_token read
→ readonly adapter payload
→ fake actual req_to_token_pool attr boundary
→ runtime metadata inspection
→ ModelRunner default-off hook
→ optional server inspection
```

---

## 3. Phase 4.8.1: KV Index Resolution Plan Smoke

追加/変更:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_kv_index_resolution_plan_smoke.py
```

追加 helper:

```text
build_relaykv_kv_index_resolution_plans_for_smoke(...)
summarize_relaykv_kv_index_resolution_plans_for_smoke(...)
```

目的:

```text
attention_shadow_capture_result
→ relaykv_kv_index_resolution_plan
```

RelayKV working blocks / full KV blocks を token span metadata に解決する。

特徴:

```text
metadata-only
block_id → token_start/token_end
no req_to_token read
no KV pool read
no tensor read
```

Pass-flow summary:

```text
block_span_resolved_count=2
resolved_block_count=12
token_span_count=12
total_token_count=1536
```

Safety counters:

```text
req_to_token_read_count=0
token_to_kv_pool_read_count=0
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

## 4. Phase 4.8.2: Req-to-token Synthetic Readonly Resolution Smoke

追加/変更:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_req_to_token_resolution_smoke.py
```

追加 helper:

```text
build_relaykv_req_to_token_resolution_results_for_smoke(...)
summarize_relaykv_req_to_token_resolution_results_for_smoke(...)
```

目的:

```text
relaykv_kv_index_resolution_plan
+ synthetic req_to_token table
→ relaykv_req_to_token_resolution_result
```

この段階では実 SGLang object は読まず、dict/list の synthetic table だけを読む。

Pass-flow summary:

```text
req_to_token_resolved_count=2
resolved_block_count=12
resolved_token_count=1536
req_to_token_entry_count=1536
req_to_token_read_count=1536
```

Safety counters:

```text
token_to_kv_pool_read_count=0
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

## 5. Phase 4.8.3 / 4.8.3.1: Req-to-token Readonly Adapter Payload

Design memo:

```text
notes/relaykv_phase4_8_3_real_req_to_token_pool_readonly_adapter_design_2026-05-04.ja.md
```

追加/変更:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_req_to_token_readonly_adapter_smoke.py
```

追加 helper:

```text
build_relaykv_req_to_token_readonly_adapter_payloads_for_smoke(...)
summarize_relaykv_req_to_token_readonly_adapter_payloads_for_smoke(...)
```

目的:

```text
relaykv_kv_index_resolution_plan
+ bounded dict/list backing
→ relaykv_req_to_token_readonly_adapter_payload
```

特徴:

```text
bounded preview/count/checksum only
no full req_to_token_entries in output
dedup by (block_id, token_start, token_end)
req_pool_idx / str(req_pool_idx) lookup support
```

Pass-flow summary:

```text
adapter_payload_ready_count=2
requested_block_count=4
requested_token_count=256
read_token_count=256
preview_entry_count=16
truncated_preview_count=2
req_to_token_read_count=256
```

Safety counters:

```text
token_to_kv_pool_read_count=0
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

## 6. Phase 4.8.4 / 4.8.4.1: Actual Req-to-token Pool Adapter Fake-object Smoke

Design memo:

```text
notes/relaykv_phase4_8_4_actual_req_to_token_pool_readonly_adapter_design_2026-05-04.ja.md
```

追加/変更:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_actual_req_to_token_pool_adapter_smoke.py
```

追加 helper:

```text
build_relaykv_actual_req_to_token_pool_adapter_payloads_for_smoke(...)
summarize_relaykv_actual_req_to_token_pool_adapter_payloads_for_smoke(...)
```

目的:

```text
relaykv_kv_index_resolution_plan
+ FakeReqToTokenPool(req_to_token=...)
→ relaykv_req_to_token_readonly_adapter_payload
```

この段階では、実 runtime object は読まないが、`.req_to_token` attr 境界を controlled に確認した。

確認済み:

```text
getattr(req_to_token_pool, "req_to_token", None) のみ
dict lookup by req_pool_idx / str(req_pool_idx)
safe top-level list/tuple row lookup
unsupported/tensor-like/cuda-like backing blocked
full entries not returned
poison unrelated field untouched
input non-mutation
```

Pass-flow summary:

```text
adapter_payload_ready_count=2
requested_block_count=4
requested_token_count=256
read_token_count=256
preview_entry_count=16
truncated_preview_count=2
req_to_token_read_count=256
actual_req_to_token_pool_read_count=256
actual_req_to_token_pool_read_true_count=2
```

Safety counters:

```text
token_to_kv_pool_read_count=0
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

## 7. Phase 4.8.4.2 / 4.8.4.2.1: Runtime Req-to-token Metadata Inspection Smoke

Design memo:

```text
notes/relaykv_phase4_8_4_2_optional_runtime_req_to_token_adapter_inspection_design_2026-05-04.ja.md
```

追加/変更:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_req_to_token_runtime_inspection_smoke.py
```

追加 helper:

```text
build_relaykv_req_to_token_runtime_inspection_payloads_for_smoke(...)
summarize_relaykv_req_to_token_runtime_inspection_payloads_for_smoke(...)
```

目的:

```text
req_to_token_pool
→ req_to_token attr metadata inspection
```

この段階では、値も index も読まない。

観測する metadata:

```text
type
module
qualname
shape
device
dtype
```

Pass-flow summary:

```text
metadata_observed_count=1
req_to_token_attr_present_count=1
actual_req_to_token_pool_inspection_count=1
req_to_token_attr_observed_count=1
```

Safety counters:

```text
req_to_token_read_count=0
actual_req_to_token_pool_read_count=0
token_to_kv_pool_read_count=0
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

## 8. Phase 4.8.4.3.1: Server Hook Wrapper Fake-object Smoke

追加/変更:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_model_runner_req_to_token_inspection_hook_smoke.py
```

追加 helper:

```text
run_model_runner_req_to_token_runtime_inspection_hook_for_smoke(model_runner, forward_batch=None)
```

目的:

```text
ModelRunner-like object / ForwardBatch-like object
→ explicit req_to_token_pool path lookup
→ req_to_token metadata-only inspection
```

固定した探索 path:

```text
model_runner.req_to_token_pool
model_runner.token_to_kv_pool_allocator.req_to_token_pool
model_runner.memory_pool.req_to_token_pool
forward_batch.req_to_token_pool
```

特徴:

```text
no recursive search
no dir/vars/repr
no value/index read
missing pool becomes clean blocked payload
poison table confirms no value/index/tensor read
```

Safety counters:

```text
req_to_token_read_count=0
actual_req_to_token_pool_read_count=0
token_to_kv_pool_read_count=0
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

## 9. Phase 4.8.4.3.2: ModelRunner default-off hook wiring

追加/変更:

```text
python/sglang/srt/model_executor/model_runner.py
scripts/relaykv_fake_model_runner_req_to_token_inspection_smoke.py
```

目的:

```text
ModelRunner.forward
→ SGLANG_RELAYKV_REQ_TO_TOKEN_INSPECTION == "1"
→ lazy import
→ req_to_token metadata-only inspection
→ exception-safe summary log
```

Env behavior:

```text
env unset / not "1":
  no import
  no hook
  no log
  forward unchanged

env "1":
  metadata-only hook
  no value/index read
  no forward output mutation
```

確認済み:

```text
env off: hook not run, no log, forward returns normally
env on: metadata summary emitted
missing pool: blocked summary, no crash
poison req_to_token table: no value/index read
```

Safety counters:

```text
req_to_token_read_count=0
actual_req_to_token_pool_read_count=0
token_to_kv_pool_read_count=0
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

## 10. Phase 4.8.4.3.3: Optional Server Req-to-token Inspection Smoke

追加/変更:

```text
scripts/relaykv_optional_server_req_to_token_inspection_smoke.py
```

目的:

```text
実 server
→ ModelRunner.forward
→ SGLANG_RELAYKV_REQ_TO_TOKEN_INSPECTION=1
→ model_runner.req_to_token_pool
→ req_to_token metadata-only summary
```

重要な到達点:

```text
実 server 上で model_runner.req_to_token_pool が見つかった
```

Optional server result:

```text
env off:
  HTTP 200
  importなし
  hookなし
  logなし

env on:
  HTTP 200
  metadata-only summary emitted
  pool_source_path=model_runner.req_to_token_pool
  production_output_unaffected=true
```

Safety counters:

```text
req_to_token_read_count=0
actual_req_to_token_pool_read_count=0
token_to_kv_pool_read_count=0
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

## 11. 実行確認

主な確認:

```text
py_compile: pass
new smoke scripts: pass
requested regression subset: pass
optional server smoke: pass
git diff --check: pass
forbidden-area grep: empty
```

代表的 regression chain:

```text
relaykv_actual_req_to_token_pool_adapter_smoke.py
relaykv_req_to_token_readonly_adapter_smoke.py
relaykv_req_to_token_resolution_smoke.py
relaykv_kv_index_resolution_plan_smoke.py
relaykv_attention_shadow_capture_smoke.py
relaykv_attention_comparison_plan_smoke.py
relaykv_attention_override_noop_smoke.py
relaykv_attention_connection_dry_run_smoke.py
relaykv_attention_handoff_smoke.py
```

Forbidden-area grep remained empty for each bounded task except explicitly allowed `model_runner.py` hook wiring.

---

## 12. 現時点の意味

Phase 4.8 系で、RelayKV は次の段階まで安全に到達した。

```text
RelayKV block/token span metadata
→ req_to_token mapping schema
→ bounded adapter payload
→ actual req_to_token_pool attr boundary
→ runtime metadata inspection
→ ModelRunner default-off hook
→ optional server metadata-only confirmation
```

この時点で分かったこと:

```text
1. SGLang server path で model_runner.req_to_token_pool が参照可能。
2. default-off env hook で production output を変えずに metadata-only inspection できる。
3. req_to_token の shape/device/dtype は、値を読まずに観測できる。
4. token_to_kv_pool / KV pool / tensor / attention へ進む前の安全境界ができた。
```

---

## 13. まだ進んでいない領域

まだ禁止・未実装:

```text
req_to_token value/index read from live runtime
token_to_kv_pool read
physical KV index resolution
KV pool read
K/V tensor read
attention comparison execution
attention backend override
scheduler decision mutation
runtime writeback
RadixTree / HiCache interaction
```

---

## 14. 次にやること

推奨 next phase:

```text
Phase 4.8.5 Physical KV Index Readonly Resolution Design
```

目的:

```text
req_to_token entry
→ token_to_kv_pool
→ physical KV index metadata
```

ただし、次も設計から入る。

安全方針:

```text
まず design memo
次に fake-object smoke
次に metadata-only runtime inspection
最後に optional server smoke
```

次の境界では `token_to_kv_pool` が入るため、Phase 4.8.4 より危険度が上がる。

特に禁止を明確にする必要がある。

```text
KV tensor read はまだ禁止
KV pool snapshot も禁止
attention execution も禁止
```

---

## 15. commit command

今回の devlog を repo に保存する場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/devlog_2026-05-04_relaykv_phase4_8_req_to_token_runtime_inspection.ja.md \
  notes/devlog_2026-05-04_relaykv_phase4_8_req_to_token_runtime_inspection.ja.md

git status --short
git diff --check

git add notes/devlog_2026-05-04_relaykv_phase4_8_req_to_token_runtime_inspection.ja.md
git commit -m "docs: add relaykv phase 4.8 req-to-token devlog"
git push mine relaykv-host-backup-shadow
```
