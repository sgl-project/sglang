# RelayKV Phase 4.8.4.2 Optional Runtime Req-to-token Adapter Inspection Design

## 日付確認

- Design date: **2026-05-04**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. 目的

このメモは、RelayKV/SGLang integration の **Phase 4.8.4.2: Optional Runtime Req-to-token Adapter Inspection Design** を定義する。

直前の Phase 4.8.4.1 では、fake object に対して `.req_to_token` attr 境界を確認した。

```text
relaykv_kv_index_resolution_plan
+ FakeReqToTokenPool(req_to_token=...)
→ relaykv_req_to_token_readonly_adapter_payload
```

Phase 4.8.4.2 の目的は、実 SGLang runtime に入る前に、default-off の inspection 境界を設計すること。

この段階では **値を読まない**。

見るのは metadata のみ。

```text
type
module
qualname
shape
device
dtype
```

---

## 2. 許可すること

Phase 4.8.4.2 で許可するのは、default-off の inspection payload だけ。

```text
req_to_token_pool object の存在確認
req_to_token attr の存在確認
req_to_token の type/module/qualname
shape/device/dtype の metadata 取得
```

許可候補 counter:

```text
actual_req_to_token_pool_inspection_count > 0
req_to_token_attr_observed_count > 0
```

---

## 3. 禁止すること

まだ以下は禁止。

```text
req_to_token value read
req_to_token index read
.cpu()
.tolist()
.item()
.numpy()
token_to_kv_pool read
KV pool read
KV snapshot
tensor value read
attention comparison execution
attention override
runtime writeback
scheduler mutation
KV cache mutation
source mutation
```

安全 counter は維持する。

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

## 4. runtime hook 方針

runtime path に入れる場合は、必ず専用 env で default-off にする。

候補 env:

```text
SGLANG_RELAYKV_REQ_TO_TOKEN_INSPECTION=1
```

ただし、最初は production runtime hook ではなく、isolated smoke で `ForwardBatch`-like object / fake model runner context を使う。

推奨順:

```text
1. isolated fake runtime inspection smoke
2. optional server inspection smoke
3. actual adapter read design
```

---

## 5. helper 方針

追加候補 helper:

```python
build_relaykv_req_to_token_runtime_inspection_payloads_for_smoke(
    forward_batch_like,
    req_to_token_pool=None,
    inspect_req_to_token=False,
)
```

summary helper:

```python
summarize_relaykv_req_to_token_runtime_inspection_payloads_for_smoke(payloads)
```

入力制約:

```text
inspect_req_to_token=False なら blocked/skip
req_to_token_pool が None なら blocked
許可する getattr は "req_to_token" のみ
値 read はしない
arbitrary method call しない
dir/vars/repr を使わない
```

---

## 6. output schema

正常 payload 候補:

```text
event_type="relaykv_req_to_token_runtime_inspection_payload"
inspection_state="metadata_observed"
inspection_mode="runtime_metadata_only"
source="runtime_req_to_token_pool_metadata_inspection"

req_to_token_pool_type
req_to_token_pool_module
req_to_token_pool_qualname

req_to_token_attr_present=true
req_to_token_type
req_to_token_module
req_to_token_qualname
req_to_token_shape
req_to_token_device
req_to_token_dtype

actual_req_to_token_pool_inspection=true
actual_req_to_token_pool_inspection_count=1
req_to_token_attr_observed=true
req_to_token_attr_observed_count=1

req_to_token_read=false
actual_req_to_token_pool_read=false
token_to_kv_pool_read=false
kv_pool_read=false
kv_snapshot=false
tensor_read=false
attention_comparison_executed=false
attention_override=false
runtime_writeback=false
scheduler_policy_noop=true
kv_cache_mutation=false
source_mutated=false

blocking_reasons=[]
warning_reasons=[
  "metadata_only_runtime_inspection",
  "no_req_to_token_value_read",
  "no_token_to_kv_pool_read"
]
```

blocked payload は同じ safety flags を false/zero にして、`inspection_state="blocked"` とする。

---

## 7. blocked reasons

```text
inspect_req_to_token_not_enabled
req_to_token_pool_missing
req_to_token_attr_missing
req_to_token_attr_access_failed
unsupported_runtime_object
req_to_token_value_read_not_allowed
token_to_kv_pool_read_not_allowed
kv_pool_read_not_allowed
tensor_read_not_allowed
attention_comparison_executed_not_allowed
attention_override_true_not_allowed
```

---

## 8. summary schema

```text
summary_type="relaykv_req_to_token_runtime_inspection_payload_summary"

total_runtime_inspection_payloads
metadata_observed_count
blocked_count
error_count

req_to_token_attr_present_count
actual_req_to_token_pool_inspection_count
req_to_token_attr_observed_count

req_to_token_read_count
actual_req_to_token_pool_read_count
token_to_kv_pool_read_count
kv_pool_read_count
kv_snapshot_count
tensor_read_count
attention_comparison_executed_count
attention_override_true_count
runtime_writeback_true_count
scheduler_policy_noop_false_count
kv_cache_mutation_true_count
source_mutated_true_count
```

---

## 9. smoke 方針

候補 script:

```text
scripts/relaykv_req_to_token_runtime_inspection_smoke.py
```

pass flow:

```text
FakeReqToTokenPool(req_to_token=FakeReqToTokenTable(shape=(16, 1024), device="cuda:0", dtype="torch.int32"))
→ runtime inspection payload
```

Fake table は metadata attr だけを持つ。

```text
shape
device
dtype
```

値 access されると例外を投げる poison object を使い、値 read が起きないことを確認する。

expected:

```text
metadata_observed_count=1
req_to_token_attr_present_count=1
actual_req_to_token_pool_inspection_count=1
req_to_token_attr_observed_count=1

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

## 10. 次フェーズ

```text
Phase 4.8.4.2:
  Optional Runtime Req-to-token Adapter Inspection Design

Phase 4.8.4.2.1:
  Runtime Req-to-token Metadata Inspection Smoke
  - fake runtime object only
  - metadata only
  - no value/index read

Phase 4.8.4.3:
  Optional Server Runtime Inspection Smoke
  - default-off
  - shape/device/dtype only

Phase 4.8.5:
  Physical KV Index Readonly Resolution Design
  - token_to_kv_pool read boundary
```

---

## 11. commit command

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/relaykv_phase4_8_4_2_optional_runtime_req_to_token_adapter_inspection_design_2026-05-04.ja.md \
  notes/relaykv_phase4_8_4_2_optional_runtime_req_to_token_adapter_inspection_design_2026-05-04.ja.md

git status --short
git diff --check

git add notes/relaykv_phase4_8_4_2_optional_runtime_req_to_token_adapter_inspection_design_2026-05-04.ja.md
git commit -m "docs: design relaykv runtime req-to-token inspection"
git push mine relaykv-host-backup-shadow
```
