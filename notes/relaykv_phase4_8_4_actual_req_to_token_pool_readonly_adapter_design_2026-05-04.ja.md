# RelayKV Phase 4.8.4 Actual Req-to-token Pool Readonly Adapter Design

## 日付確認

- Design date: **2026-05-04**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. この設計メモの目的

このメモは、RelayKV/SGLang integration の **Phase 4.8.4: Actual Req-to-token Pool Readonly Adapter Design** を定義する。

Phase 4.8.3.1 では、実 SGLang object には触れず、dict/list backing だけで bounded adapter payload を作成した。

```text
relaykv_kv_index_resolution_plan
→ relaykv_req_to_token_readonly_adapter_payload
```

Phase 4.8.3.1 の到達点:

```text
adapter_payload_ready_count=2
requested_block_count=4
requested_token_count=256
read_token_count=256
preview_entry_count=16
truncated_preview_count=2
req_to_token_read_count=256
```

Phase 4.8.4 の目的は、次に実 SGLang object へ接続する場合の **actual req_to_token_pool readonly adapter boundary** を設計すること。

重要:

```text
Phase 4.8.4 は設計フェーズ。
まだ actual object adapter smoke は実装しない。
まだ token_to_kv_pool は読まない。
まだ KV pool / K/V tensor は読まない。
まだ attention comparison は実行しない。
まだ attention override はしない。
```

---

## 2. 現在の安全 chain

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
```

Phase 4.8.3.1 で許可済み:

```text
req_to_token_read_count > 0
```

ただし、これは dict/list backing からの bounded read であり、actual SGLang object ではない。

Phase 4.8.4 で設計する許可候補:

```text
actual_req_to_token_pool_read_count > 0
req_to_token_read_count > 0
```

まだ 0 固定:

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

## 3. actual object に進む理由

Phase 4.8.3.1 で adapter payload schema は固定できた。

次に必要なのは、SGLang runtime の実 metadata から次の値を安全に得られるか確認すること。

```text
req_to_token_pool.req_to_token[req_pool_idx, token_position]
```

これにより、RelayKV block span が SGLang の request-local token mapping に一致しているかを確認できる。

ただし、この段階でも physical KV index には進まない。

```text
OK:
  req_to_token_pool.req_to_token readonly bounded access

NG:
  token_to_kv_pool access
  K/V tensor access
  attention backend access
```

---

## 4. actual adapter の設計原則

### 4.1 default-off

actual object adapter は必ず default-off。

候補 env:

```text
SGLANG_RELAYKV_REQ_TO_TOKEN_ADAPTER_SMOKE=1
```

または既存 prefix に合わせて:

```text
SGLANG_RELAYKV_RUNTIME_OBSERVATION=1
SGLANG_RELAYKV_REQ_TO_TOKEN_ADAPTER=1
```

推奨:

```text
専用 env を追加する。
```

理由:

```text
runtime observation metadata と actual req_to_token read は危険度が違うため。
```

### 4.2 production output 不変

actual adapter は:

```text
read-only
bounded
exception-safe
no writeback
no scheduler mutation
no attention override
```

adapter failure は production output に影響させない。

### 4.3 runtime path へ直接入れない

初回は server/runtime path に直接接続しない。

推奨:

```text
isolated smoke script
```

で fake/minimal object を使って interface だけ確認する。

actual runtime hook はさらに後。

---

## 5. actual req_to_token_pool access policy

許可する attribute は 1 つだけ。

```text
req_to_token_pool.req_to_token
```

禁止:

```text
dir(req_to_token_pool)
vars(req_to_token_pool)
repr(req_to_token_pool) の詳細ログ
arbitrary getattr
method call
mutation
```

実装候補:

```python
req_to_token = getattr(req_to_token_pool, "req_to_token", None)
```

ただし、これは actual adapter helper 内だけで行う。

ログに出す情報は限定。

```text
type name
shape if safely available
device if safely available
dtype if safely available
```

値全体は出さない。

---

## 6. tensor handling policy

### 6.1 Torch import

Phase 4.8.4 の actual adapter では、torch import を helper 本体に追加しない方が安全。

理由:

```text
metrics.py の pure helper 群をなるべく torch-free に保つため。
```

方針:

```text
duck typing で shape/device/dtype を観測する。
明示的な torch.Tensor 判定はしない。
```

ただし、GPU tensor と疑われる場合は read を blocked にする。

### 6.2 GPU tensor read

原則禁止。

禁止操作:

```text
.cpu()
.tolist()
.item()
.numpy()
```

blocked reason:

```text
req_to_token_tensor_device_not_allowed
gpu_tensor_read_not_allowed
```

GPU tensor らしさの判定:

```text
hasattr(value, "device") and "cuda" in str(value.device).lower()
```

この程度の metadata access は許容候補。ただし値 read はしない。

### 6.3 CPU tensor read

CPU tensor も default blocked。

候補 flag:

```text
allow_cpu_tensor_read=False
```

将来 `allow_cpu_tensor_read=True` を使う場合も、bounded slice のみにする。

Phase 4.8.4 初期設計では:

```text
actual torch Tensor value read はしない。
```

### 6.4 Python list/tuple backing

actual adapter smoke では、actual pool object の `req_to_token` が list/tuple の場合だけ read を許可する。

つまり初回 actual adapter smoke では:

```text
FakeReqToTokenPool(req_to_token=list/tuple)
```

を使う。

これは actual attribute boundary を確認するためで、まだ実 tensor read ではない。

---

## 7. indexing policy

SGLang の `req_to_token` は概念上:

```text
req_to_token[req_pool_idx, token_position]
```

である。

Python list/tuple backing では、以下の形を許可する。

### 7.1 nested table

```text
req_to_token[req_pool_idx][position]
```

例:

```python
req_to_token = {
  7: [1000, 1001, ...],
  8: [2000, 2001, ...],
}
```

または:

```python
req_to_token = [
  [...],
  [...],
]
```

### 7.2 dict key lookup

dictの場合:

```text
req_pool_idx
str(req_pool_idx)
```

### 7.3 list row lookup

list/tuple の場合:

```text
req_pool_idx must be int
0 <= req_pool_idx < len(req_to_token)
```

ただし、実 SGLang の req_pool_idx は大きい可能性があるため、fake smoke では dict の方が安全。

---

## 8. bounded read policy

actual adapter でも Phase 4.8.3.1 と同じ制限を使う。

推奨初期値:

```text
max_tokens_per_request=256
max_blocks_per_request=4
max_total_tokens=512
max_preview_entries=8
```

制限超過時:

```text
blocked
```

理由:

```text
actual object に対する read を初期段階から小さく保つため。
```

---

## 9. output schema

actual adapter payload は Phase 4.8.3.1 の schema と互換にする。

追加 fields:

```text
actual_req_to_token_pool_read=true
actual_req_to_token_pool_read_count=<count>
req_to_token_source="actual_pool_attr"
req_to_token_backing_type
req_to_token_shape
req_to_token_device
req_to_token_dtype
```

正常 output:

```text
event_type="relaykv_req_to_token_readonly_adapter_payload"
adapter_state="adapter_payload_ready"
adapter_mode="actual_pool_readonly_bounded_preview"
source="kv_index_resolution_plan_to_actual_req_to_token_readonly_adapter_payload"

request_id
req_pool_idx
seq_len
layer_id

requested_block_count
requested_token_count
read_token_count
preview_entry_count
preview_entries
entry_count
entry_min
entry_max
entry_checksum
truncated_preview

req_to_token_read=true
req_to_token_read_count=<read_token_count>
actual_req_to_token_pool_read=true
actual_req_to_token_pool_read_count=<read_token_count>

token_to_kv_pool_read=false
token_to_kv_pool_read_count=0
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
warning_reasons includes "actual_req_to_token_pool_readonly_adapter"
warning_reasons includes "bounded_preview_only"
warning_reasons includes "no_token_to_kv_pool_read"
warning_reasons includes "preview_only_no_full_entries_logged"
```

---

## 10. blocked reasons

Actual adapter specific blocked reasons:

```text
req_to_token_pool_missing
req_to_token_attr_missing
req_to_token_backing_not_supported
req_to_token_tensor_device_not_allowed
cpu_tensor_read_not_allowed
gpu_tensor_read_not_allowed
actual_req_to_token_pool_read_not_enabled
```

Common blocked reasons:

```text
not_kv_index_resolution_plan
kv_index_resolution_not_block_span_resolved
kv_index_resolution_not_metadata_only
req_pool_idx_missing_or_invalid
seq_len_missing_or_invalid
requested_block_count_exceeds_limit
requested_token_count_exceeds_limit
total_requested_token_count_exceeds_limit
token_position_out_of_req_to_token_table
req_to_token_entry_not_int
invalid_block_span
token_span_out_of_seq_len
token_to_kv_pool_read_not_allowed
kv_pool_read_not_allowed
tensor_read_not_allowed
attention_comparison_executed_not_allowed
attention_override_true_not_allowed
```

blocked output:

```text
event_type="relaykv_req_to_token_readonly_adapter_payload"
adapter_state="blocked"
adapter_mode="actual_pool_readonly_bounded_preview"

actual_req_to_token_pool_read=false
actual_req_to_token_pool_read_count=0
req_to_token_read=false
req_to_token_read_count=0
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
blocking_reasons set
```

---

## 11. summary schema

既存 summary を拡張する。

追加 fields:

```text
actual_req_to_token_pool_read_count
actual_req_to_token_pool_read_true_count
per_adapter_mode_counts
per_req_to_token_source_counts
```

維持する safety fields:

```text
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

## 12. Phase 4.8.4 smoke 方針

最小 smoke 候補:

```text
scripts/relaykv_actual_req_to_token_pool_adapter_smoke.py
```

Allowed files:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_actual_req_to_token_pool_adapter_smoke.py
```

Smokeでは real SGLang runtime object ではなく、actual attr boundary を持つ fake object を使う。

例:

```python
class FakeReqToTokenPool:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token
```

これは「実 object adapter boundary」の smoke であり、まだ optional server smoke ではない。

Pass flow:

```text
relaykv_kv_index_resolution_plan x 2
+ FakeReqToTokenPool(req_to_token={7: [...], 8: [...]})
→ relaykv_req_to_token_readonly_adapter_payload x 2
```

expected:

```text
adapter_payload_ready_count=2
actual_req_to_token_pool_read_count=256
req_to_token_read_count=256
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

## 13. actual runtime へ進む前の条件

actual runtime path に入る前に必要な条件:

```text
1. fake actual pool adapter smoke pass
2. bounded read limits pass
3. unsupported backing blocks safely
4. CUDA-like device blocks safely
5. CPU tensor-like backing blocks by default
6. no full entries in logs
7. no token_to_kv_pool read
8. no KV/tensor read
9. no attention execution
10. no production mutation
```

runtime hook へ進む場合は、専用 env で default-off。

候補:

```text
SGLANG_RELAYKV_ACTUAL_REQ_TO_TOKEN_ADAPTER=1
```

ただし、これは次フェーズ以降。

---

## 14. Phase 4.8.4 の結論

Phase 4.8.4 では、actual req_to_token_pool へ接続する前に、次の境界を設計する。

```text
real/fake req_to_token_pool object
→ .req_to_token attr only
→ bounded preview/count/checksum
→ relaykv_req_to_token_readonly_adapter_payload
```

許可候補:

```text
actual_req_to_token_pool_read_count > 0
req_to_token_read_count > 0
```

まだ禁止:

```text
token_to_kv_pool_read_count > 0
kv_pool_read_count > 0
kv_snapshot_count > 0
tensor_read_count > 0
attention_comparison_executed_count > 0
attention_override_true_count > 0
runtime_writeback_true_count > 0
scheduler_policy_noop_false_count > 0
kv_cache_mutation_true_count > 0
source_mutated_true_count > 0
```

---

## 15. 推奨次フェーズ

推奨順:

```text
Phase 4.8.4:
  Actual Req-to-token Pool Readonly Adapter Design
  - this memo

Phase 4.8.4.1:
  Actual Req-to-token Pool Adapter Fake-object Smoke
  - FakeReqToTokenPool(req_to_token=dict/list)
  - bounded preview/count/checksum
  - no real SGLang runtime object
  - no torch import

Phase 4.8.4.2:
  Optional Runtime Req-to-token Adapter Inspection Design
  - default-off runtime inspection
  - only metadata / shape / device / dtype at first

Phase 4.8.5:
  Physical KV Index Readonly Resolution Design
  - token_to_kv_pool read boundary

Phase 4.9:
  Isolated K/V Tensor Read Design

Phase 4.10:
  Isolated Attention Comparison Smoke
```

---

## 16. commit command

この設計メモを repo に保存する場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/relaykv_phase4_8_4_actual_req_to_token_pool_readonly_adapter_design_2026-05-04.ja.md \
  notes/relaykv_phase4_8_4_actual_req_to_token_pool_readonly_adapter_design_2026-05-04.ja.md

git status --short
git diff --check

git add notes/relaykv_phase4_8_4_actual_req_to_token_pool_readonly_adapter_design_2026-05-04.ja.md
git commit -m "docs: design relaykv actual req-to-token adapter"
git push mine relaykv-host-backup-shadow
```
