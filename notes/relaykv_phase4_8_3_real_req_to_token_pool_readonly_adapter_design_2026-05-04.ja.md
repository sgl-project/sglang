# RelayKV Phase 4.8.3 Real Req-to-token Pool Readonly Adapter Design

## 日付確認

- Design date: **2026-05-04**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. この設計メモの目的

このメモは、RelayKV/SGLang integration の **Phase 4.8.3: Real Req-to-token Pool Readonly Adapter Design** を定義する。

Phase 4.8.2 では synthetic table を使って、次の chain まで進んだ。

```text
relaykv_kv_index_resolution_plan
→ relaykv_req_to_token_resolution_result
```

Phase 4.8.2 で初めて許可した counter:

```text
req_to_token_read_count=1536
```

ただし Phase 4.8.2 では、実 SGLang object には触れていない。

Phase 4.8.3 の目的は、synthetic table ではなく、実 SGLang の `req_to_token_pool.req_to_token` を read-only に参照する場合の境界を設計すること。

重要:

```text
Phase 4.8.3 は設計フェーズ。
まだ実 SGLang object adapter を実装しない。
まだ token_to_kv_pool は読まない。
まだ KV pool / K/V tensor は読まない。
まだ attention comparison は実行しない。
まだ attention override はしない。
```

---

## 2. 現在の到達点

現在の安全 chain:

```text
attention_handoff_candidate
→ attention_connection_dry_run_result
→ attention_override_noop_result
→ attention_comparison_plan
→ attention_shadow_capture_result
→ relaykv_kv_index_resolution_plan
→ relaykv_req_to_token_resolution_result
```

Phase 4.8.2 pass-flow summary:

```text
req_to_token_resolved_count=2
resolved_block_count=12
resolved_token_count=1536
req_to_token_entry_count=1536
req_to_token_read_count=1536
```

Phase 4.8.2 で 0 維持できている safety counters:

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

Phase 4.8.3 でも、これらはまだ 0 固定。

---

## 3. Phase 4.8.3 の中心課題

実 `req_to_token_pool.req_to_token` に触れる場合、synthetic table よりリスクが上がる。

主なリスク:

```text
GPU tensor read
.cpu() による同期
.tolist() による巨大転送
巨大ログ化
production object への attribute access
shape / dtype / device 差異
batch dimension の取り違え
decode / extend path の position semantics の違い
poison object による予期しない副作用
```

Phase 4.8.3 では、これらの境界を設計し、次の smoke で何を許可するかを固定する。

---

## 4. 設計方針

Phase 4.8.3 の安全方針:

```text
1. 実 SGLang object を直接 helper に渡すのではなく、まず adapter 境界を作る。
2. adapter は read-only。
3. adapter は default-off。
4. adapter は小さい bounded slice だけ読む。
5. adapter は full table を .tolist() しない。
6. adapter は GPU tensor の .cpu() を原則禁止。
7. adapter は log に full entries を出さず preview/count/checksum にする。
8. adapter failure はすべて blocked/fallback にする。
9. production output には影響しない。
```

---

## 5. 推奨する adapter 分離

### 5.1 pure schema helper

既存の Phase 4.8.2 helper:

```python
build_relaykv_req_to_token_resolution_results_for_smoke(...)
```

これは pure Python synthetic table 用として維持する。

### 5.2 real object adapter

新規に別 helper を設計する。

候補名:

```python
build_relaykv_req_to_token_readonly_adapter_payloads_for_smoke(...)
```

または:

```python
extract_relaykv_req_to_token_readonly_slices_for_smoke(...)
```

責務:

```text
real req_to_token_pool object
+ kv_index_resolution_plan
→ small bounded synthetic table-like payload
```

その後、既存 synthetic helper に渡す。

推奨 chain:

```text
real req_to_token_pool object
→ readonly adapter payload
→ synthetic-table compatible req_to_token resolution
```

これにより、実 object access と schema resolution を分離できる。

---

## 6. adapter の入力

候補 signature:

```python
build_relaykv_req_to_token_readonly_adapter_payloads_for_smoke(
    kv_index_resolution_plans,
    req_to_token_pool=None,
    max_tokens_per_request=256,
    allow_cpu_tensor_read=False,
    allow_gpu_tensor_read=False,
)
```

ただし、初回 smoke では実 pool object ではなく fake object / fake table object でよい。

短期安全実装:

```text
Phase 4.8.3 smoke:
  fake req_to_token_pool object の attr を読むのではなく、
  adapter input は dict か、明示 wrapper のみ。

Phase 4.8.4:
  actual SGLang req_to_token_pool object を読むか検討。
```

理由:

```text
実 object access は思ったより副作用や同期コストが高い可能性がある。
まず adapter schema と safety counters を固定する。
```

---

## 7. req_to_token_pool object access policy

実 object に触る場合、許可する attribute は限定する。

許可候補:

```text
req_to_token_pool.req_to_token
```

禁止:

```text
dir(obj)
vars(obj)
repr(obj) の詳細ログ
arbitrary getattr
method call
mutation
```

許可する getattr は明示的に 1 回だけ:

```text
getattr(req_to_token_pool, "req_to_token", None)
```

ただし、これも Phase 4.8.3 では設計のみ。

---

## 8. tensor read policy

### 8.1 GPU tensor

原則:

```text
GPU tensor read は Phase 4.8.3 では禁止。
```

禁止:

```text
tensor.cpu()
tensor.tolist()
tensor.item()
tensor.numpy()
```

理由:

```text
GPU同期と巨大転送が起きるため。
```

もし `req_to_token` が CUDA tensor の場合:

```text
blocked reason:
  req_to_token_tensor_device_not_allowed
```

### 8.2 CPU tensor

CPU tensor についても慎重に扱う。

Phase 4.8.3 では初回は CPU tensor も禁止または明示 opt-in。

候補:

```text
allow_cpu_tensor_read=False default
```

`allow_cpu_tensor_read=True` の場合のみ、小さい bounded slice で読む。

ただし、初回 smoke は torch import なしを維持するのが安全。

### 8.3 Python list/tuple

最初に許可するのは Python list/tuple のみ。

```text
req_to_token backing table が list/tuple:
  bounded index read allowed
```

---

## 9. bounded read policy

実 object adapter で最も重要なのは read bound。

必要な制約:

```text
max_tokens_per_request
max_blocks_per_request
max_total_tokens
max_preview_entries
```

推奨初期値:

```text
max_tokens_per_request=256
max_blocks_per_request=4
max_total_tokens=512
max_preview_entries=8
```

超過時:

```text
blocked または truncated_preview
```

Phase 4.8.3 では、実 resolution に必要な full entries を返すより、まず preview/count/checksum へ寄せる。

---

## 10. payload schema

adapter payload の候補:

```text
event_type="relaykv_req_to_token_readonly_adapter_payload"
adapter_state="adapter_payload_ready"
adapter_mode="readonly_bounded_preview"
source="kv_index_resolution_plan_to_req_to_token_readonly_adapter_payload"

request_id
req_pool_idx
seq_len
layer_id

requested_token_count
read_token_count
preview_entry_count
preview_entries
entry_count
entry_min
entry_max
entry_checksum

req_to_token_read=true
req_to_token_read_count=<bounded count>
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
warning_reasons includes "readonly_bounded_req_to_token_adapter"
warning_reasons includes "no_token_to_kv_pool_read"
```

注意:

```text
preview_entries は最大 max_preview_entries。
full entries を log に出さない。
```

---

## 11. full entries と preview の扱い

Phase 4.8.2 synthetic helper は full `req_to_token_entries` を span に保持していた。

実 object adapter では、full entries を保持しない方が安全。

推奨:

```text
synthetic smoke:
  full req_to_token_entries allowed

real adapter:
  preview_entries + entry_count + checksum only
```

理由:

```text
実 request では seq_len が大きくなり、log が巨大化するため。
```

将来、physical KV index resolution へ進むには full entries が必要になる可能性がある。

その場合は、log とは別に ephemeral in-memory payload と summary log を分ける。

```text
internal_payload:
  bounded full entries

log_payload:
  preview/count/checksum only
```

Phase 4.8.3 ではまだ設計のみ。

---

## 12. blocked design

blocked reasons:

```text
not_kv_index_resolution_plan
kv_index_resolution_not_block_span_resolved
kv_index_resolution_not_metadata_only
req_to_token_pool_missing
req_to_token_attr_missing
req_to_token_backing_not_supported
req_to_token_tensor_device_not_allowed
cpu_tensor_read_not_allowed
gpu_tensor_read_not_allowed
requested_token_count_exceeds_limit
req_pool_idx_missing_or_invalid
seq_len_missing_or_invalid
token_position_out_of_req_to_token_table
req_to_token_entry_not_int
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
adapter_mode="readonly_bounded_preview"

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

## 13. summary schema

候補:

```python
summarize_relaykv_req_to_token_readonly_adapter_payloads_for_smoke(payloads)
```

summary fields:

```text
summary_type="relaykv_req_to_token_readonly_adapter_payload_summary"

total_req_to_token_adapter_payloads
adapter_payload_ready_count
blocked_count
error_count

requested_token_count
read_token_count
preview_entry_count

per_request_counts
per_layer_counts
per_adapter_state_counts

req_to_token_read_count
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

## 14. Phase 4.8.3 smoke 方針

Phase 4.8.3 の最小 smoke は、実 SGLang object ではなく、**adapter-compatible fake object** で行うか、さらに安全に dict/list に限定する。

候補名:

```text
scripts/relaykv_req_to_token_readonly_adapter_smoke.py
```

推奨初回:

```text
input backing:
  dict/list only

not yet:
  real SGLang req_to_token_pool object
  torch Tensor
  CUDA tensor
```

pass flow:

```text
relaykv_kv_index_resolution_plan x 2
+ req_to_token backing list
→ relaykv_req_to_token_readonly_adapter_payload x 2
```

expected:

```text
adapter_payload_ready_count=2
req_to_token_read_count > 0
token_to_kv_pool_read_count=0
kv_pool_read_count=0
tensor_read_count=0
attention_comparison_executed_count=0
attention_override_true_count=0
```

---

## 15. 実 SGLang object へ進む条件

実 `req_to_token_pool.req_to_token` を読む前に必要な条件:

```text
1. adapter payload smoke が pass
2. bounded read limit が実装済み
3. preview/count/checksum log schema が固定済み
4. GPU tensor read が default blocked
5. CPU tensor read も default blocked
6. token_to_kv_pool read はまだ blocked
7. runtime path default-off
8. exception が output に影響しない
```

実 object adapter は、optional server smoke とは分ける。

候補:

```text
Phase 4.8.4 Actual Req-to-token Pool Readonly Adapter Smoke
```

これは `SGLANG_RELAYKV_*` env で default-off にする。

---

## 16. Phase 4.8.3 の結論

Phase 4.8.3 では、synthetic table から実 object へ一気に進まない。

最小安全経路:

```text
relaykv_kv_index_resolution_plan
→ req_to_token readonly adapter payload
→ preview/count/checksum
```

許可候補:

```text
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

## 17. 推奨次フェーズ

推奨順:

```text
Phase 4.8.3:
  Req-to-token Readonly Adapter Design
  - this memo

Phase 4.8.3.1:
  Req-to-token Readonly Adapter Payload Smoke
  - dict/list backing only
  - bounded preview/count/checksum
  - no torch import
  - no actual SGLang object

Phase 4.8.4:
  Actual Req-to-token Pool Readonly Adapter Design/Smoke
  - actual object
  - default-off
  - GPU/CPU tensor handling carefully blocked or bounded

Phase 4.8.5:
  Physical KV Index Readonly Resolution Design
  - token_to_kv_pool read boundary

Phase 4.9:
  Isolated K/V Tensor Read Design

Phase 4.10:
  Isolated Attention Comparison Smoke
```

---

## 18. commit command

この設計メモを repo に保存する場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/relaykv_phase4_8_3_real_req_to_token_pool_readonly_adapter_design_2026-05-04.ja.md \
  notes/relaykv_phase4_8_3_real_req_to_token_pool_readonly_adapter_design_2026-05-04.ja.md

git status --short
git diff --check

git add notes/relaykv_phase4_8_3_real_req_to_token_pool_readonly_adapter_design_2026-05-04.ja.md
git commit -m "docs: design relaykv req-to-token readonly adapter"
git push mine relaykv-host-backup-shadow
```
