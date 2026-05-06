# RelayKV Phase 4.8.6 Runtime req_to_token Resolution Payload Bridge Design

## 日付確認

- Design date: **2026-05-06**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. 目的

このメモは **Phase 4.8.6: Runtime req_to_token resolution payload bridge for live index read** の設計を定義する。

直前の Phase 4.8.5.4 では、以下が完了した。

```text
req_to_token resolution result
→ live-like token_to_kv_pool object
→ bounded token_to_kv_pool index read
→ physical_kv_index preview / count / checksum
→ engine_block_ref
```

また、`ModelRunner.forward` には default-off の live index-read hook が入った。

```text
SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ=1
→ relaykv_live_token_to_kv_pool_index_read_summary=...
```

ただし、optional server smoke では env-on summary が clean blocked になった。

理由:

```text
server runtime path で req_to_token resolution result payloads が live index-read hook に渡っていないため。
```

Phase 4.8.6 の目的は、この missing bridge を安全に追加することである。

---

## 2. 現在の問題

現在の server path:

```text
ModelRunner.forward
→ live token_to_kv_pool index-read hook
→ token_to_kv_pool object lookup
→ req_to_token resolution payloads unavailable
→ clean blocked
```

期待する次の path:

```text
ModelRunner.forward
→ existing runtime metadata / candidate payload source
→ req_to_token resolution result payload bridge
→ live token_to_kv_pool index-read hook
→ bounded physical index read
→ relaykv_live_token_to_kv_pool_index_read_summary
```

重要:

```text
この Phase では live req_to_token value/index read を新規に行わない。
既存または synthetic/fake-safe の req_to_token resolution result payload を bridge するだけ。
```

---

## 3. なぜ bridge が必要か

Phase 4.8.5.4.3 optional server smoke は、live index-read hook が server 上で安全に起動することを確認した。

しかし、hook が physical index を解決するには、入力として以下が必要になる。

```text
req_to_token resolution result payload
```

この payload が無いと、hook は `token_to_kv_pool[req_to_token_entry]` に進めない。

したがって Phase 4.8.6 では、runtime path 上で以下を接続する。

```text
req_to_token resolution result payloads
→ live token_to_kv_pool index-read hook
```

---

## 4. 安全境界

Phase 4.8.6 でも禁止すること:

```text
live req_to_token value/index read
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

許可すること:

```text
existing req_to_token resolution result payloads を hook に渡す
payload shape / count / schema を検証する
payload が無い場合は clean blocked にする
```

この Phase の主語は **bridge** であり、**new live req_to_token reader** ではない。

---

## 5. env flag

既存 flag:

```text
SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ=1
```

Phase 4.8.6 では、bridge 自体も default-off にする。

候補 flag:

```text
SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE=1
```

ただし、実装を簡素化するなら、最初は以下でもよい。

```text
SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ=1
```

推奨:

```text
bridge 専用 flag を追加する。
```

理由:

```text
index-read hook の起動と req_to_token payload bridge を別々に制御できる。
```

想定挙動:

```text
bridge flag off:
  no bridge payload injection
  live index-read hook は clean blocked のまま

bridge flag on:
  existing req_to_token resolution result payloads を hook に渡す
```

---

## 6. source 設計

Phase 4.8.6 で参照してよい source は以下に限定する。

```text
forward_batch.relaykv_runtime_observation_metadata
forward_batch.relaykv_req_to_token_resolution_results
forward_batch.relaykv_runtime_observation_payloads
model_runner.relaykv_req_to_token_resolution_results
explicit test-provided payloads
```

ただし、実 runtime でまだ `forward_batch.relaykv_req_to_token_resolution_results` のような field が無い場合は、まず fake object / smoke でこの field を使う。

注意:

```text
ForwardBatch class への恒久 field 追加は、この Phase では避ける。
まずは getattr-based optional bridge に留める。
```

---

## 7. helper 設計

### 7.1 bridge helper

候補:

```python
build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
    *,
    forward_batch=None,
    model_runner=None,
    explicit_payloads=None,
    bridge_enabled=False,
)
```

責務:

```text
1. bridge_enabled=False なら blocked summary
2. explicit_payloads があれば最優先で使う
3. forward_batch / model_runner の明示 attr だけを見る
4. payload schema を浅く検証する
5. valid payloads だけ返す
6. invalid / missing は clean blocked にする
```

見てよい attr:

```text
forward_batch.relaykv_req_to_token_resolution_results
forward_batch.relaykv_req_to_token_resolution_payloads
model_runner.relaykv_req_to_token_resolution_results
model_runner.relaykv_req_to_token_resolution_payloads
```

禁止:

```text
recursive search
dir()
vars()
repr()
req_to_token_pool read
req_to_token tensor/list indexing
```

### 7.2 live index-read wrapper update

既存 wrapper:

```python
run_model_runner_live_token_to_kv_pool_index_read_hook_for_smoke(...)
```

この wrapper に bridge payload source を渡せるようにする。

候補引数:

```python
bridge_req_to_token_resolution_payloads=None
bridge_enabled=False
```

または wrapper 内で bridge helper を呼ぶ。

推奨:

```text
wrapper 内で bridge helper を呼ぶ。
```

理由:

```text
ModelRunner.forward 側の変更を小さくできる。
```

---

## 8. output schema

bridge result event:

```text
event_type="relaykv_req_to_token_resolution_payload_bridge_result"
bridge_state="bridged" | "blocked" | "error"
bridge_mode="runtime_payload_bridge"
```

fields:

```text
payload_count
valid_payload_count
blocked_payload_count
bridge_source_path
blocked_reason
```

live index-read summary 側には、以下を含める。

```text
req_to_token_resolution_bridge_enabled
req_to_token_resolution_bridge_state
req_to_token_resolution_bridge_payload_count
req_to_token_resolution_bridge_source_path
```

---

## 9. safety counters

Phase 4.8.6 で追加する counter 候補:

```text
req_to_token_resolution_bridge_payload_count
req_to_token_resolution_bridge_valid_count
req_to_token_resolution_bridge_blocked_count
```

引き続き常に 0:

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

resolved index-read path では以下が >0 になってよい。

```text
token_to_kv_pool_read_count > 0
actual_token_to_kv_pool_read_count > 0
live_token_to_kv_pool_index_read_count > 0
```

---

## 10. blocked reason 候補

```text
req_to_token_resolution_bridge_not_enabled
req_to_token_resolution_bridge_source_missing
req_to_token_resolution_bridge_payloads_missing
req_to_token_resolution_bridge_payloads_not_list
req_to_token_resolution_bridge_payload_invalid
req_to_token_resolution_bridge_payload_empty
req_to_token_resolution_bridge_no_valid_payloads
live_index_read_not_enabled
token_to_kv_pool_object_missing
token_to_kv_pool_object_not_indexable
token_to_kv_pool_index_read_failed
```

---

## 11. 実装ステップ

### Phase 4.8.6.1 Fake bridge smoke

変更候補:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_req_to_token_resolution_payload_bridge_smoke.py
```

確認:

```text
bridge off blocked
explicit payload bridged
forward_batch attr bridged
model_runner attr bridged
invalid payload blocked
empty payload blocked
no recursive traversal
source object not mutated
live index-read helper can consume bridged payloads
```

### Phase 4.8.6.2 ModelRunner wrapper update

変更候補:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_fake_model_runner_req_to_token_bridge_live_index_read_smoke.py
```

確認:

```text
env off unchanged
index-read on + bridge off clean blocked
index-read on + bridge on + payloads resolved
index-read on + bridge on + no payloads clean blocked
forward return unchanged
```

### Phase 4.8.6.3 Optional server bridge smoke

変更候補:

```text
scripts/relaykv_optional_server_req_to_token_bridge_live_index_read_smoke.py
```

期待:

```text
env off:
  HTTP 200
  no bridge / no live-index summary

index-read on, bridge off:
  HTTP 200
  live-index summary emitted
  clean blocked

index-read on, bridge on:
  HTTP 200
  bridge summary emitted
  live-index summary emitted
  either resolved or clean blocked depending on payload availability
  response marker unchanged
```

注意:

```text
server path に payload source がまだない場合、bridge on でも clean blocked でよい。
その場合は次フェーズで payload production を設計する。
```

---

## 12. 完了条件

Phase 4.8.6 の完了条件:

```text
req_to_token resolution payload bridge helper がある
fake object で payload bridge が確認できる
live index-read hook が bridged payload を consume できる
payload missing の場合は clean blocked
server optional smoke で bridge flag の off/on 差分が確認できる
forward output unchanged
KV/tensor/attention/scheduler/runtime mutation counters zero
```

---

## 13. 次にまだ残ること

Phase 4.8.6 の後に残るもの:

```text
server runtime path で req_to_token resolution payload を実際に生成する
live server 上で resolved physical index read を通す
KV pool read / K/V tensor read はまだ先
shadow attention compare はさらに先
```

次候補:

```text
Phase 4.8.7:
  Runtime req_to_token resolution payload production
```

---

## 14. Codex CLI 向け要約

```text
Goal:
Phase 4.8.6.1 fake req_to_token resolution payload bridge smoke.

Implement bridge helper + smoke only.
Do not touch model_runner.py yet.
Do not read live req_to_token values.
Do not read KV pool/tensors/attention.
Bridge existing explicit/fake req_to_token resolution result payloads into the live token_to_kv_pool index-read helper.
Payload missing/invalid must cleanly block.
Preserve schema and safety counters.
```

---

## 15. commit command

この design memo を repo に保存する場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/relaykv_phase4_8_6_req_to_token_resolution_payload_bridge_design_2026-05-06.ja.md \
  notes/relaykv_phase4_8_6_req_to_token_resolution_payload_bridge_design_2026-05-06.ja.md

git status --short
git diff --check

git add notes/relaykv_phase4_8_6_req_to_token_resolution_payload_bridge_design_2026-05-06.ja.md
git commit -m "docs: design relaykv req-to-token payload bridge"
git push mine relaykv-host-backup-shadow
```
