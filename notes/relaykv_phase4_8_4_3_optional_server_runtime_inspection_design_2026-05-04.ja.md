# RelayKV Phase 4.8.4.3 Optional Server Runtime Inspection Design

## 日付確認

- Design date: **2026-05-04**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. 目的

このメモは **Phase 4.8.4.3: Optional Server Runtime Inspection Design** を定義する。

直前の Phase 4.8.4.2.1 では、fake runtime object に対して metadata-only inspection を確認した。

```text
req_to_token_pool
→ req_to_token attr metadata inspection
```

Phase 4.8.4.3 の目的は、実 SGLang server path のどこに、default-off の metadata-only inspection hook を置くかを設計すること。

この段階でも **値は読まない**。

---

## 2. 現在の到達点

Phase 4.8.4.2.1 で確認済み:

```text
metadata_observed_count=1
req_to_token_attr_present_count=1
actual_req_to_token_pool_inspection_count=1
req_to_token_attr_observed_count=1
```

安全 counter:

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

## 3. Phase 4.8.4.3 で許可すること

許可するのは、実 server runtime 上での metadata-only inspection だけ。

```text
req_to_token_pool object の存在確認
req_to_token_pool.req_to_token attr の存在確認
req_to_token の type / module / qualname
shape / device / dtype の metadata 取得
summary log
```

許可候補 counter:

```text
actual_req_to_token_pool_inspection_count > 0
req_to_token_attr_observed_count > 0
```

---

## 4. 禁止すること

まだ禁止:

```text
req_to_token value read
req_to_token index read
actual_req_to_token_pool_read_count > 0
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

---

## 5. 推奨 hook 位置

最初の hook は **ModelRunner.forward の early metadata hook** が安全。

理由:

```text
既に SGLANG_RELAYKV_RUNTIME_OBSERVATION=1 の default-off hook がある
ForwardBatch metadata observation と同じ安全思想で扱える
attention backend / scheduler / KV pool mutation に入る前に限定できる
exception を握りつぶして production output に影響させない設計にできる
```

候補:

```text
python/sglang/srt/model_executor/model_runner.py
```

既存の runtime observation hook の近くに、専用 env が有効な場合だけ lazy import する。

---

## 6. req_to_token_pool の取得候補

実 server path で `req_to_token_pool` を得る候補は、ModelRunner 周辺の runtime object から read-only に探す。

ただし Phase 4.8.4.3 では設計だけで、実装時は最小探索にする。

候補:

```text
self.req_to_token_pool
self.token_to_kv_pool_allocator.req_to_token_pool
self.memory_pool.req_to_token_pool
forward_batch.req_to_token_pool
```

方針:

```text
実装時に arbitrary recursive search はしない。
候補 path を明示的に順番に見る。
見つからなければ clean skip。
```

ログは path 名だけに留める。

---

## 7. env 設計

専用 env:

```text
SGLANG_RELAYKV_REQ_TO_TOKEN_INSPECTION=1
```

off の場合:

```text
importなし
hook実行なし
ログなし
```

on の場合:

```text
lazy import
metadata-only inspection
exception-safe
summary log only
```

既存 env `SGLANG_RELAYKV_RUNTIME_OBSERVATION` とは分離する。

理由:

```text
runtime observation metadata と req_to_token_pool inspection は危険度が違う。
ユーザーが個別にON/OFFできるべき。
```

---

## 8. runtime helper 方針

既存 helper:

```python
build_relaykv_req_to_token_runtime_inspection_payloads_for_smoke(...)
summarize_relaykv_req_to_token_runtime_inspection_payloads_for_smoke(...)
```

server hook 用 wrapper 候補:

```python
run_model_runner_req_to_token_runtime_inspection_hook_for_smoke(
    model_runner,
    forward_batch=None,
)
```

責務:

```text
env on 時だけ呼ばれる
req_to_token_pool 候補 path を明示的に見る
見つかれば metadata-only helper に渡す
summary log を出す
失敗しても forward を止めない
```

---

## 9. log schema

log prefix 候補:

```text
relaykv_req_to_token_runtime_inspection_summary
```

summary に含める:

```text
summary_type
metadata_observed_count
blocked_count
req_to_token_attr_present_count
actual_req_to_token_pool_inspection_count
req_to_token_attr_observed_count
req_to_token_read_count
actual_req_to_token_pool_read_count
token_to_kv_pool_read_count
kv_pool_read_count
tensor_read_count
attention_override_true_count
runtime_writeback_true_count
scheduler_policy_noop_false_count
source_mutated_true_count
pool_source_path
```

値本体は log しない。

---

## 10. optional server smoke 方針

候補 script:

```text
scripts/relaykv_optional_server_req_to_token_inspection_smoke.py
```

動作:

```text
env off:
  server request succeeds
  inspection log absent

env on:
  server request succeeds
  inspection summary log appears if pool path is found
  or clean skip summary appears if pool path is unavailable
```

実 server では path 差異があり得るので、初回は **found 必須にしない**。

期待:

```text
HTTP 200
production output unchanged
no exception
off時 import/log なし
on時 metadata-only summary or clean skip
safety counters all zero except inspection counters
```

---

## 11. blocked / skip 方針

clean skip reasons:

```text
req_to_token_inspection_env_off
req_to_token_pool_not_found
req_to_token_attr_missing
req_to_token_attr_access_failed
```

安全上 blocked:

```text
req_to_token_value_read_not_allowed
token_to_kv_pool_read_not_allowed
kv_pool_read_not_allowed
tensor_read_not_allowed
attention_override_true_not_allowed
runtime_writeback_not_allowed
scheduler_mutation_not_allowed
```

---

## 12. 実装順

推奨順:

```text
1. metrics.py に server hook wrapper helper を追加
2. fake ModelRunner-like object smoke を追加
3. model_runner.py に default-off lazy hook を追加
4. optional server smoke を追加
```

ただし、変更範囲が広がるため、実装は分割する。

### Phase 4.8.4.3.1

```text
Server hook wrapper fake-object smoke
Allowed files:
- metrics.py
- scripts/relaykv_model_runner_req_to_token_inspection_hook_smoke.py
```

### Phase 4.8.4.3.2

```text
ModelRunner default-off hook wiring
Allowed files:
- metrics.py
- model_runner.py
- scripts/relaykv_fake_model_runner_req_to_token_inspection_smoke.py
```

### Phase 4.8.4.3.3

```text
Optional server inspection smoke
Allowed files:
- scripts/relaykv_optional_server_req_to_token_inspection_smoke.py
```

---

## 13. Phase 4.8.4.3 の結論

Phase 4.8.4.3 では、実 server runtime path への接続を急がず、まず hook 位置と env 境界を固定する。

最初に許可するのは metadata-only inspection。

```text
req_to_token_pool.req_to_token
→ type / shape / device / dtype metadata only
```

まだ禁止:

```text
req_to_token value/index read
token_to_kv_pool read
KV pool read
tensor read
attention execution/override
runtime/scheduler mutation
```

---

## 14. commit command

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/relaykv_phase4_8_4_3_optional_server_runtime_inspection_design_2026-05-04.ja.md \
  notes/relaykv_phase4_8_4_3_optional_server_runtime_inspection_design_2026-05-04.ja.md

git status --short
git diff --check

git add notes/relaykv_phase4_8_4_3_optional_server_runtime_inspection_design_2026-05-04.ja.md
git commit -m "docs: design relaykv optional server req-to-token inspection"
git push mine relaykv-host-backup-shadow
```
