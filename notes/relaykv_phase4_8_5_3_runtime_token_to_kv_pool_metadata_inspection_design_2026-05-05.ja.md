# RelayKV Phase 4.8.5.3 Runtime token_to_kv_pool Metadata Inspection Design

## 日付確認

- Design date: **2026-05-05**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. 目的

このメモは **Phase 4.8.5.3: Runtime token_to_kv_pool Metadata Inspection Design** を定義する。

直前までに完了したこと:

```text
Phase 4.8.5.1:
  synthetic token_to_kv_pool readonly resolution smoke

Phase 4.8.5.2:
  fake actual token_to_kv_pool adapter payload smoke
```

ここまでで、RelayKV は synthetic/fake object 上で次の chain を確認した。

```text
req_to_token resolution result
→ token_to_kv_pool readonly lookup
→ physical KV index metadata
→ engine_block_ref
```

Phase 4.8.5.3 の目的は、実 runtime object に近い境界で `token_to_kv_pool` を **metadata-only** に観測すること。

ただし、この段階でも **token_to_kv_pool の index 値は読まない**。

---

## 2. このフェーズで観測するもの

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

必要なら補助 metadata:

```text
source_path
metadata_observed
attr_present
attr_access_error
inspection_state
blocked_reason
```

---

## 3. まだ禁止すること

Phase 4.8.5.3 では、以下を引き続き禁止する。

```text
token_to_kv_pool value/index read
req_to_token value/index read
KV pool read
KV pool snapshot
K/V tensor read
k_buffer / v_buffer read
attention execution
attention override
scheduler mutation
runtime writeback
KV cache mutation
source mutation
```

禁止 API / 操作:

```text
.cpu()
.tolist()
.item()
.numpy()
direct tensor value read
unbounded indexing
full table dump
recursive object traversal
dir()
vars()
repr() of arbitrary runtime objects
```

---

## 4. 許可すること

許可するのは、明示 path に対する `getattr` と metadata extraction のみ。

許可 path 候補:

```text
model_runner.token_to_kv_pool
model_runner.token_to_kv_pool_allocator.token_to_kv_pool
model_runner.token_to_kv_pool_allocator
model_runner.kv_pool_allocator.token_to_kv_pool
model_runner.memory_pool.token_to_kv_pool
forward_batch.token_to_kv_pool
```

ただし、探索は **明示 path のみ**。

禁止:

```text
recursive search
dir(model_runner)
vars(model_runner)
repr(model_runner)
all attributes scan
```

---

## 5. Core / Adapter 境界

このフェーズは **RelayKV SGLang Adapter** の責務。

Core-ish fields:

```text
engine_name="sglang"
adapter_name="sglang"
engine_request_id
logical_sequence_id
logical_block_id
token_span
layer_id
kv_head_group
kv_class
decision_state
fallback_reason
position_check_state
attention_mask_mode
rope_position_consistency
```

SGLang adapter fields:

```text
adapter_metadata:
  token_to_kv_pool_source_path
  token_to_kv_pool_type
  token_to_kv_pool_module
  token_to_kv_pool_qualname
  token_to_kv_pool_shape
  token_to_kv_pool_device
  token_to_kv_pool_dtype

engine_block_ref:
  token_to_kv_pool_index=None
  physical_kv_index=None
  cache_position=None
```

重要:

```text
physical_kv_index はまだ runtime では解決しない。
```

---

## 6. 推奨 helper 設計

### 6.1 Runtime metadata inspection helper

追加 helper 候補:

```python
build_relaykv_token_to_kv_pool_runtime_inspection_payloads_for_smoke(
    token_to_kv_pool_object=None,
    *,
    enabled=False,
    source_path=None,
    allow_value_read=False,
)
```

出力 event:

```text
event_type="relaykv_token_to_kv_pool_runtime_inspection_payload"
inspection_state="metadata_observed" | "blocked" | "error"
inspection_mode="runtime_metadata_only"
engine_name="sglang"
adapter_name="sglang"
```

metadata output:

```text
token_to_kv_pool_attr_present
token_to_kv_pool_source_path
token_to_kv_pool_type
token_to_kv_pool_module
token_to_kv_pool_qualname
token_to_kv_pool_shape
token_to_kv_pool_device
token_to_kv_pool_dtype
```

safety output:

```text
token_to_kv_pool_read=false
token_to_kv_pool_read_count=0
actual_token_to_kv_pool_read_count=0
kv_pool_read=false
kv_pool_read_count=0
kv_snapshot=false
kv_snapshot_count=0
tensor_read=false
tensor_read_count=0
attention_comparison_executed=false
attention_override=false
runtime_writeback=false
scheduler_policy_noop=true
kv_cache_mutation=false
source_mutated=false
```

### 6.2 Summary helper

```python
summarize_relaykv_token_to_kv_pool_runtime_inspection_payloads_for_smoke(payloads)
```

Summary fields:

```text
summary_type="relaykv_token_to_kv_pool_runtime_inspection_summary"
total_payloads
metadata_observed_count
blocked_count
error_count
token_to_kv_pool_attr_present_count
actual_token_to_kv_pool_inspection_count
token_to_kv_pool_read_count
actual_token_to_kv_pool_read_count
kv_pool_read_count
kv_snapshot_count
tensor_read_count
attention_comparison_executed_count
attention_override_true_count
runtime_writeback_true_count
scheduler_policy_noop_false_count
kv_cache_mutation_true_count
source_mutated_true_count
blocked_reason_counts
source_path_counts
```

---

## 7. ModelRunner wrapper 設計

Phase 4.8.5.3 の smoke では、まだ `model_runner.py` は触らない。

まず fake wrapper helper を metrics.py に置く。

候補 helper:

```python
run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(
    model_runner,
    forward_batch=None,
)
```

探索 path は明示固定。

```text
model_runner.token_to_kv_pool
model_runner.token_to_kv_pool_allocator.token_to_kv_pool
model_runner.token_to_kv_pool_allocator
model_runner.kv_pool_allocator.token_to_kv_pool
model_runner.memory_pool.token_to_kv_pool
forward_batch.token_to_kv_pool
```

missing の場合:

```text
inspection_state="blocked"
blocked_reason="token_to_kv_pool_missing"
```

attr access failure の場合:

```text
inspection_state="blocked"
blocked_reason="token_to_kv_pool_attr_access_failed"
```

---

## 8. blocked reason 候補

```text
inspection_not_enabled
token_to_kv_pool_missing
token_to_kv_pool_attr_missing
token_to_kv_pool_attr_access_failed
token_to_kv_pool_value_read_not_allowed
token_to_kv_pool_index_read_not_allowed
kv_pool_read_not_allowed
kv_snapshot_not_allowed
tensor_read_not_allowed
attention_override_true_not_allowed
attention_comparison_executed_not_allowed
runtime_writeback_not_allowed
scheduler_mutation_not_allowed
source_mutation_not_allowed
```

---

## 9. smoke 方針

新規 smoke:

```text
scripts/relaykv_token_to_kv_pool_runtime_inspection_smoke.py
```

確認すること:

```text
1. direct token_to_kv_pool object metadata observed
2. model_runner.token_to_kv_pool path observed
3. nested allocator path observed
4. memory_pool path observed
5. forward_batch.token_to_kv_pool path observed
6. missing path blocked cleanly
7. attr access failure blocked cleanly
8. poison object value/index read not triggered
9. no dir/vars/repr needed
10. safety counters remain zero
11. schema alignment fields are present
12. input/source objects are not mutated
```

Pass-flow expected:

```text
metadata_observed_count > 0
token_to_kv_pool_attr_present_count > 0
token_to_kv_pool_read_count=0
actual_token_to_kv_pool_read_count=0
kv_pool_read_count=0
kv_snapshot_count=0
tensor_read_count=0
attention_override_true_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
source_mutated_true_count=0
```

---

## 10. runtime hook は次フェーズ

Phase 4.8.5.3 では、まだ `ModelRunner.forward` に env hook を追加しない。

次フェーズ候補:

```text
Phase 4.8.5.3.1:
  ModelRunner token_to_kv_pool inspection wrapper fake-object smoke

Phase 4.8.5.3.2:
  ModelRunner default-off token_to_kv_pool inspection hook wiring

Phase 4.8.5.3.3:
  Optional server token_to_kv_pool metadata inspection smoke
```

env 名候補:

```text
SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INSPECTION=1
```

default off:

```text
unset / not "1":
  no import
  no hook
  no log

"1":
  metadata-only inspection
  exception-safe
  no production output mutation
```

---

## 11. 完了条件

Phase 4.8.5.3 は以下で完了。

```text
token_to_kv_pool runtime-like object metadata inspection pass
explicit path lookup only
no value/index read
no KV pool/tensor read
no attention/runtime/scheduler mutation
schema alignment maintained
```

まだ完了条件に含めないもの:

```text
actual runtime hook
optional server smoke
token_to_kv_pool index read from live object
KV tensor read
working KV assembly
shadow attention compute
```

---

## 12. Codex CLI 向け次ステップ

```text
Phase 4.8.5.3: Runtime token_to_kv_pool metadata inspection smoke

Implement only metrics helper + smoke.
No model_runner changes.
No runtime/server hook.
Use explicit object paths only.
Observe type/module/qualname/shape/device/dtype only.
Do not read token_to_kv_pool value or index.
Keep all read/mutation counters zero.
```

---

## 13. commit command

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/relaykv_phase4_8_5_3_runtime_token_to_kv_pool_metadata_inspection_design_2026-05-05.ja.md \
  notes/relaykv_phase4_8_5_3_runtime_token_to_kv_pool_metadata_inspection_design_2026-05-05.ja.md

git status --short
git diff --check

git add notes/relaykv_phase4_8_5_3_runtime_token_to_kv_pool_metadata_inspection_design_2026-05-05.ja.md
git commit -m "docs: design relaykv token-to-kv-pool runtime inspection"
git push mine relaykv-host-backup-shadow
```
