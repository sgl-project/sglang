# RelayKV Phase 4.8.5.4 Live token_to_kv_pool Bounded Index Read Design

## 日付確認

- Design date: **2026-05-05**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. 目的

このメモは **Phase 4.8.5.4: Live token_to_kv_pool bounded index read** の設計を定義する。

直前までに、RelayKV は SGLang 上で以下を完了した。

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

Phase 4.8.5.4 の目的は、実 runtime 上の `token_to_kv_pool` から **bounded な index read** を行い、logical token / req_to_token entry から physical KV index metadata へ進めることである。

ただし、この段階でも **KV pool / K/V tensor / attention backend には触らない**。

---

## 2. 現在の chain

現時点の metadata chain:

```text
attention shadow capture result
→ KV index resolution plan
→ req_to_token resolution result
→ token_to_kv_pool metadata inspection
→ physical KV index metadata under engine_block_ref
```

Phase 4.8.5.4 で目指す chain:

```text
req_to_token resolution result
→ live token_to_kv_pool object
→ bounded token_to_kv_pool index read
→ physical_kv_index preview/count/checksum
→ engine_block_ref
```

---

## 3. なぜこのフェーズが必要か

SGLang の decode path では、RelayKV が最終的に working KV を組み立てるには、少なくとも以下の対応関係が必要になる。

```text
logical token span
→ req_to_token entry
→ token_to_kv_pool entry
→ physical KV slot/index
```

これまでの Phase 4.8.5.1 / 4.8.5.2 は synthetic/fake object でこの流れを確認した。

Phase 4.8.5.3 系では、実 server 上で `token_to_kv_pool` の metadata-only inspection 入口まで確認した。

次に必要なのは、実 runtime object から **bounded に index を読む**段階である。

---

## 4. このフェーズで許可すること

Phase 4.8.5.4 で初めて許可すること:

```text
live token_to_kv_pool object の bounded index read
```

ただし、許可範囲は以下に限定する。

```text
- env flag が明示 ON の場合のみ
- bounded token budget 内のみ
- req_to_token resolution result で解決済みの entry のみ
- token_to_kv_pool object への index read のみ
- output は bounded preview / count / checksum のみ
- full physical index array は出力しない
```

想定 read:

```text
physical_index = token_to_kv_pool[req_to_token_entry]
```

または runtime object の型に応じて、adapter 内で許可された minimal indexing のみ。

---

## 5. 引き続き禁止すること

Phase 4.8.5.4 でも以下は禁止する。

```text
KV pool read
KV pool snapshot
K/V tensor read
k_buffer / v_buffer read
attention execution
attention comparison execution
attention override
scheduler mutation
runtime writeback
KV cache mutation
source mutation
```

禁止 API:

```text
.cpu()
.tolist()
.item()
.numpy()
full tensor dump
full table dump
unbounded slice
recursive object traversal
dir()
vars()
repr() of arbitrary runtime objects
```

重要:

```text
token_to_kv_pool の index read は許可するが、
KV pool / tensor / attention はまだ許可しない。
```

---

## 6. env flag 設計

既存の metadata inspection flag:

```text
SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INSPECTION=1
```

Phase 4.8.5.4 では、index read を metadata inspection より危険な操作として分離する。

推奨 flag:

```text
SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ=1
```

default off:

```text
unset / not "1":
  no live index read
  no additional value/index access
  no production behavior change
```

ON:

```text
"1":
  bounded token_to_kv_pool index read
  compact metadata summary log
  exception-safe
  forward output unchanged
```

---

## 7. input / source

入力は既存の `req_to_token_resolution_result` 系 payload を想定する。

必要 fields:

```text
event_type="relaykv_req_to_token_resolution_result"
resolution_state="req_to_token_resolved"
req_to_token entries or bounded preview metadata
logical_block_id
token_span
layer_id
kv_head_group
engine_request_id
adapter_metadata
engine_block_ref
```

実 runtime では、以下を明示 path で取得する。

```text
model_runner.token_to_kv_pool
model_runner.token_to_kv_pool_allocator.token_to_kv_pool
model_runner.token_to_kv_pool_allocator
model_runner.kv_pool_allocator.token_to_kv_pool
model_runner.memory_pool.token_to_kv_pool
forward_batch.token_to_kv_pool
```

ただし、探索は明示 path のみ。

---

## 8. helper 設計

### 8.1 live bounded read helper

候補 helper:

```python
build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
    req_to_token_resolution_results,
    token_to_kv_pool_object,
    *,
    read_token_to_kv_pool_index=False,
    max_tokens_per_request=256,
    max_total_tokens=1024,
    source_path=None,
)
```

出力 event:

```text
event_type="relaykv_live_token_to_kv_pool_index_read_result"
resolution_state="physical_kv_index_resolved" | "blocked" | "error"
adapter_mode="live_token_to_kv_pool_bounded_index_read"
```

成功時:

```text
engine_block_ref:
  token_to_kv_pool_index=None or bounded preview only
  physical_kv_index_preview=[...bounded...]
  physical_kv_index_count=N
  physical_kv_index_checksum=...
  cache_position=None
```

重要:

```text
full physical index list は出さない。
```

### 8.2 ModelRunner wrapper helper

候補 helper:

```python
run_model_runner_live_token_to_kv_pool_index_read_hook_for_smoke(
    model_runner,
    forward_batch=None,
    req_to_token_resolution_results=None,
)
```

この helper は次の責務だけ持つ。

```text
1. explicit path lookup で token_to_kv_pool object を見つける
2. req_to_token_resolution_results を受け取る
3. bounded read helper に渡す
4. summary を返す
```

まだ scheduler / attention / KV tensor には触らない。

---

## 9. output schema

Core/Adapter fields:

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
decision_state="SHADOW_ONLY"
fallback_reason
position_check_state
attention_mask_mode
rope_position_consistency
adapter_metadata
engine_block_ref
```

adapter_metadata:

```text
token_to_kv_pool_source_path
token_to_kv_pool_type
token_to_kv_pool_shape
live_index_read_enabled
max_tokens_per_request
max_total_tokens
truncated_preview
```

engine_block_ref:

```text
physical_kv_index_preview
physical_kv_index_count
physical_kv_index_checksum
token_to_kv_pool_index=None
cache_position=None
```

---

## 10. safety counters

このフェーズでは、以下が pass flow で >0 になってよい。

```text
token_to_kv_pool_read_count > 0
actual_token_to_kv_pool_read_count > 0
live_token_to_kv_pool_index_read_count > 0
```

ただし、以下は必ず 0。

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

理由:

```text
req_to_token resolution は既存 payload を使う。
このフェーズで live req_to_token を新たに読まない。
```

---

## 11. blocked reason 候補

```text
live_index_read_not_enabled
not_req_to_token_resolution_result
req_to_token_resolution_not_resolved
req_to_token_entries_missing
req_to_token_entry_not_int
token_to_kv_pool_object_missing
token_to_kv_pool_source_missing
token_to_kv_pool_attr_missing
token_to_kv_pool_attr_access_failed
token_to_kv_pool_object_not_indexable
token_to_kv_pool_index_read_failed
token_to_kv_pool_entry_missing
token_to_kv_pool_entry_not_int
max_tokens_per_request_exceeded
max_total_tokens_exceeded
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

## 12. smoke 設計

### 12.1 まず fake/guarded live object smoke

新規 smoke 候補:

```text
scripts/relaykv_live_token_to_kv_pool_index_read_smoke.py
```

確認内容:

```text
1. read flag off blocked
2. explicit token_to_kv_pool object pass
3. model_runner explicit path pass
4. forward_batch explicit path pass
5. bounded preview / count / checksum
6. dict int-key lookup
7. dict str-key lookup if needed
8. list/tuple lookup
9. unsupported object blocked
10. attr access failure blocked
11. missing index blocked
12. non-int physical index blocked
13. max_tokens_per_request blocked
14. max_total_tokens blocked
15. poison object does not trigger forbidden methods
16. source object not mutated
17. full physical index list not emitted
18. schema fields preserved
```

Safety expectation:

```text
token_to_kv_pool_read_count > 0
actual_token_to_kv_pool_read_count > 0
live_token_to_kv_pool_index_read_count > 0

kv_pool_read_count=0
kv_snapshot_count=0
tensor_read_count=0
attention_override_true_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
source_mutated_true_count=0
```

### 12.2 次に ModelRunner default-off hook

次フェーズ候補:

```text
Phase 4.8.5.4.2:
  ModelRunner default-off live physical-index read hook
```

env:

```text
SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ=1
```

default off では:

```text
no import
no hook
no log
forward unchanged
```

### 12.3 最後に optional server smoke

次フェーズ候補:

```text
Phase 4.8.5.4.3:
  Optional server live physical-index read smoke
```

pass condition:

```text
env off:
  HTTP 200
  no live index read log

env on:
  HTTP 200
  bounded physical index summary log
  response marker unchanged
  KV/tensor/attention counters zero
```

---

## 13. この設計の重要境界

このフェーズは **physical KV index metadata を読む段階**であって、まだ **KV tensor を読む段階ではない**。

境界:

```text
OK:
  token_to_kv_pool[req_to_token_entry] を bounded に読む

NG:
  k_buffer[physical_index]
  v_buffer[physical_index]
  kv_pool snapshot
  attention backend input replacement
```

この境界を守ることで、Phase 4.9 / Phase 5 の shadow attention compare に進む前に、index path の安全性だけを確認できる。

---

## 14. 完了条件

Phase 4.8.5.4 の完了条件:

```text
live-like token_to_kv_pool object から bounded index read ができる
physical_kv_index_preview/count/checksum が engine_block_ref に出る
full physical index list は出ない
read budget が効く
KV pool / tensor / attention / scheduler / runtime mutation はゼロ
schema alignment が維持される
```

---

## 15. 次の Codex CLI 向け要約

```text
Goal:
Phase 4.8.5.4: live-like token_to_kv_pool bounded index read smoke.

Implement metrics helper + smoke only first.
Do not touch model_runner.py yet.
Use existing req_to_token_resolution_result payloads.
Allow bounded token_to_kv_pool index read only when explicit flag is true.
Emit physical_kv_index_preview/count/checksum under engine_block_ref.
Do not emit full physical index list.
Keep KV pool/tensor/attention/scheduler/runtime/source mutation counters zero.
```

---

## 16. commit command

この design memo を repo に保存する場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/relaykv_phase4_8_5_4_live_token_to_kv_pool_bounded_index_read_design_2026-05-05.ja.md \
  notes/relaykv_phase4_8_5_4_live_token_to_kv_pool_bounded_index_read_design_2026-05-05.ja.md

git status --short
git diff --check

git add notes/relaykv_phase4_8_5_4_live_token_to_kv_pool_bounded_index_read_design_2026-05-05.ja.md
git commit -m "docs: design relaykv live token-to-kv-pool index read"
git push mine relaykv-host-backup-shadow
```
