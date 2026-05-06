# RelayKV Devlog: Phase 4.8.6〜4.8.7 req_to_token payload bridge / production chain

- Date basis: **2026-05-06 JST**
- Target repo: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`
- Phase range: **Phase 4.8.6〜Phase 4.8.7**
- Status: **completed as smoke / optional-server summary-chain milestone**

## 1. 今回の目的

Phase 4.8.5.4 までで、RelayKV は runtime 上の `token_to_kv_pool` を bounded に index read する安全な smoke-only 経路まで到達していた。
ただし optional server smoke では、live index-read hook が clean blocked になっていた。

主な理由は以下。

```text
server runtime path に req_to_token resolution payload source がまだ存在しない
```

今回の Phase 4.8.6〜4.8.7 では、この gap を埋めるために、以下の3段階を安全に追加・検証した。

```text
req_to_token resolution payload bridge
→ runtime req_to_token payload producer
→ optional server producer / bridge / live index-read summary chain
```

重要な境界として、今回もまだ以下には進んでいない。

```text
live req_to_token value/index read
KV pool read
K/V tensor read
attention execution / override
scheduler mutation
runtime writeback
```

## 2. Phase 4.8.6.1: Fake req_to_token resolution payload bridge smoke

### 変更ファイル

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_req_to_token_resolution_payload_bridge_smoke.py
```

### 実装内容

`metrics.py` に smoke-only の req_to_token resolution payload bridge を追加した。

代表 helper:

```python
build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(...)
```

bridge は以下の優先順位で payload source を見る。

```text
explicit_payloads
→ forward_batch.relaykv_req_to_token_resolution_results
→ forward_batch.relaykv_req_to_token_resolution_payloads
→ model_runner.relaykv_req_to_token_resolution_results
→ model_runner.relaykv_req_to_token_resolution_payloads
```

挙動:

- `bridge_enabled=False` では clean blocked
- explicit payloads が最優先
- forward_batch / model_runner は指定された RelayKV attr のみ浅く確認
- valid payload のみ preserving
- invalid payload は blocked count に計上
- recursive traversal / `dir()` / `vars()` / arbitrary `repr()` は禁止
- req_to_token_pool / KV pool / tensor / attention / scheduler には触れない

### 検証

以下が確認された。

- bridge off blocked
- explicit payload bridged
- forward_batch attr bridged
- model_runner attr bridged
- source priority: explicit > forward_batch > model_runner
- missing / empty / invalid payload blocked
- mixed valid/invalid payloads は valid subset を保持し invalid を count
- poison object で forbidden traversal が発火しない
- source object not mutated
- bridged payloads が live token_to_kv_pool index-read helper に渡せる
- safety counters zero

## 3. Phase 4.8.6.2: ModelRunner live index-read wrapper bridge integration

### 変更ファイル

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_model_runner_req_to_token_bridge_live_index_read_smoke.py
```

### 実装内容

既存の live token_to_kv_pool index-read wrapper を拡張し、bridge flag が有効な場合だけ req_to_token payload bridge を consume できるようにした。

新 flag:

```text
SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE=1
```

挙動:

```text
SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE unset / not "1":
  previous behavior preserved
  no bridged payload consumption

SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE == "1":
  bridge helper invoked
  valid req_to_token resolution payloads consumed
  live token_to_kv_pool index-read helper receives bridged payloads
  resolved or clean blocked summary emitted
```

live-index summary には以下の bridge metadata を追加した。

```text
req_to_token_resolution_bridge_enabled
req_to_token_resolution_bridge_state
req_to_token_resolution_bridge_payload_count
req_to_token_resolution_bridge_valid_count
req_to_token_resolution_bridge_source_path
req_to_token_resolution_bridge_blocked_reason
```

### 検証

以下が確認された。

- bridge env off では従来挙動を維持
- bridge env on + explicit payloads で resolved smoke path
- bridge env on + forward_batch attr payloads で resolved smoke path
- bridge env on + model_runner attr payloads で resolved smoke path
- invalid / missing payloads は clean blocked
- source priority 維持
- source objects not mutated
- no `model_runner.py` edit in this phase
- safety counters zero

## 4. Phase 4.8.6.3: Optional server bridge smoke

### 変更ファイル

```text
scripts/relaykv_optional_server_req_to_token_bridge_live_index_read_smoke.py
```

### 実装内容

optional server smoke を追加し、server runtime で bridge flag が live index-read wrapper に届き、summary に bridge metadata が出ることを確認した。

検証ケース:

```text
Case 1: all off
  SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ unset
  SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE unset

Case 2: index-read on / bridge off
  SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ=1

Case 3: index-read on / bridge on
  SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ=1
  SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE=1
```

### 実 server run 結果

```text
Case 1:
  HTTP 200
  no live-index summary

Case 2:
  HTTP 200
  live-index summary emitted
  clean blocked

Case 3:
  HTTP 200
  live-index summary emitted
  bridge metadata present
  clean blocked bridge state
  reason: no req_to_token payload source in server path

Response marker unchanged across cases.
```

この時点で、bridge flag と server summary metadata の到達は確認できた。
ただし、server path に req_to_token payload source が無いため、resolved physical index にはまだ進んでいない。

## 5. Phase 4.8.7.1: Runtime req_to_token resolution payload production smoke

### 変更ファイル

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_runtime_req_to_token_resolution_payload_production_smoke.py
```

### 実装内容

smoke-only の runtime req_to_token resolution payload producer を追加した。

代表 helper:

```python
build_relaykv_runtime_req_to_token_resolution_payloads_for_smoke(...)
```

目的:

```text
safe runtime metadata
+ explicit fake req_to_token entries
→ relaykv_req_to_token_resolution_result payload
→ bridge helper
→ live token_to_kv_pool index-read helper
→ resolved physical index smoke path
```

producer は以下のみを source とする。

```text
runtime observation metadata
kv_index_resolution_plan metadata
explicit fake req_to_token entries from smoke
```

禁止事項:

```text
req_to_token_pool read
live req_to_token value/index read
token_to_kv_pool access in producer
KV pool read
K/V tensor read
attention execution / override
scheduler mutation
runtime writeback
```

### 検証

以下が確認された。

- production off blocked
- explicit fake entries から resolved payload 生成
- runtime observation metadata + explicit fake entries で schema-preserving payload 生成
- kv_index_resolution_plan metadata preservation
- max_tokens_per_request / max_total_tokens guard
- invalid entry / missing entry blocked
- source object not mutated
- poison object で forbidden traversal が発火しない
- produced payloads が bridge helper を通過
- bridged payloads が fake token_to_kv_pool で live index-read helper により resolved smoke path へ進む
- safety counters zero

## 6. Phase 4.8.7.2: ModelRunner default-off runtime payload production hook

### 変更ファイル

```text
python/sglang/srt/model_executor/model_runner.py
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_fake_model_runner_runtime_req_to_token_payload_production_smoke.py
```

### 実装内容

`ModelRunner.forward` に default-off の smoke-only payload production hook を追加した。

新 flag:

```text
SGLANG_RELAYKV_RUNTIME_REQ_TO_TOKEN_PAYLOAD_PRODUCTION=1
```

log marker:

```text
relaykv_runtime_req_to_token_payload_production_summary=...
```

hook の env off 挙動:

```text
SGLANG_RELAYKV_RUNTIME_REQ_TO_TOKEN_PAYLOAD_PRODUCTION unset / not "1":
  no lazy import
  no hook call
  no log
  no attr write
  forward unchanged
```

hook の env on 挙動:

```text
SGLANG_RELAYKV_RUNTIME_REQ_TO_TOKEN_PAYLOAD_PRODUCTION == "1":
  lazy import smoke wrapper
  build smoke-only req_to_token resolution payloads
  attach relaykv_req_to_token_resolution_payloads to forward_batch if possible
  otherwise attach to model_runner
  emit relaykv_runtime_req_to_token_payload_production_summary
  swallow exceptions
  forward unchanged
```

attaching は以下の RelayKV smoke-only attr に限定した。

```text
forward_batch.relaykv_req_to_token_resolution_payloads
model_runner.relaykv_req_to_token_resolution_payloads
```

### 検証

以下が確認された。

- env off: forward sentinel unchanged, no lazy import / hook / log / attr write
- env on + explicit fake entries on forward_batch: payload attached and bridge-consumable
- env on + explicit fake entries on model_runner: payload attached and bridge-consumable
- env on + missing / invalid fake entries: clean blocked summary
- hook exception swallowed, forward unchanged
- producer does not read live req_to_token / token_to_kv_pool / KV pool / tensors
- integration consume stepで bridge + live index-read helper が resolved smoke path へ進む
- safety counters zero

## 7. Phase 4.8.7.3: Optional server payload production + bridge + live index-read smoke

### 変更ファイル

```text
scripts/relaykv_optional_server_runtime_req_to_token_payload_production_bridge_live_index_read_smoke.py
```

### 実装内容

optional server smoke を追加し、server 上で producer / bridge / live index-read wrapper の3段 summary chain が安全に出ることを確認した。

検証ケース:

```text
Case 1: all off
  no producer summary
  no live-index summary

Case 2: producer only
  SGLANG_RELAYKV_RUNTIME_REQ_TO_TOKEN_PAYLOAD_PRODUCTION=1
  producer summary emitted
  no live-index summary

Case 3: producer + bridge + index-read
  SGLANG_RELAYKV_RUNTIME_REQ_TO_TOKEN_PAYLOAD_PRODUCTION=1
  SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE=1
  SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ=1
  producer summary emitted
  live-index summary emitted
  bridge metadata present
```

### 実 server run 結果

```text
Case 1:
  HTTP 200
  no RelayKV summaries

Case 2:
  HTTP 200
  producer summary emitted
  no live-index summary
  clean blocked producer summary

Case 3:
  HTTP 200
  producer summary emitted
  live-index summary emitted
  bridge metadata present
  clean blocked live-index summary
  reason: no explicit fake req_to_token entries were available in the real server path

Response marker unchanged across cases.
```

これにより、server上で以下の summary chain が確認できた。

```text
ModelRunner payload production hook
→ req_to_token payload bridge
→ live token_to_kv_pool index-read wrapper
→ summary chain emitted safely
```

## 8. 実行済み validation

各 phase で以下の validation が通った。

代表的な実行済み checks:

```text
py_compile
scripts/relaykv_req_to_token_resolution_payload_bridge_smoke.py
scripts/relaykv_model_runner_req_to_token_bridge_live_index_read_smoke.py
scripts/relaykv_optional_server_req_to_token_bridge_live_index_read_smoke.py
scripts/relaykv_runtime_req_to_token_resolution_payload_production_smoke.py
scripts/relaykv_fake_model_runner_runtime_req_to_token_payload_production_smoke.py
scripts/relaykv_optional_server_runtime_req_to_token_payload_production_bridge_live_index_read_smoke.py
scripts/relaykv_live_token_to_kv_pool_index_read_smoke.py
scripts/relaykv_fake_model_runner_live_token_to_kv_pool_index_read_smoke.py
scripts/relaykv_req_to_token_resolution_smoke.py
scripts/relaykv_physical_kv_index_resolution_smoke.py
scripts/relaykv_sglang_adapter_schema_alignment_smoke.py
scripts/relaykv_reserved_retrieval_metadata_fields_smoke.py
git diff --check
forbidden-path grep empty
```

optional server smoke では、local Qwen2.5-3B-Instruct model path を用いた real server run も通過した。

## 9. Safety invariants

今回の範囲では、以下を引き続き禁止・未実施として維持した。

```text
live req_to_token value/index read
req_to_token_pool direct access in producer
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
recursive object traversal
dir() / vars() / arbitrary repr()
.cpu() / .tolist() / .item() / .numpy()
```

producer summary / live-index summary において、以下の forbidden counters は zero を維持した。

```text
req_to_token_read_count
actual_req_to_token_pool_read_count
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

resolved smoke-only consume step では、fake token_to_kv_pool を対象にした bounded index-read のため、以下のみ許容した。

```text
token_to_kv_pool_read_count > 0
actual_token_to_kv_pool_read_count > 0
live_token_to_kv_pool_index_read_count > 0
```

## 10. 現在の到達点

Phase 4.8.6〜4.8.7 の完了により、RelayKV/SGLang integration は以下まで到達した。

```text
read-only runtime observation
→ host backup candidate join
→ dry-run policy
→ safe materialization boundary
→ attention metadata chain
→ req_to_token readonly inspection
→ token_to_kv_pool readonly inspection
→ live token_to_kv_pool bounded index-read smoke
→ req_to_token payload bridge
→ runtime req_to_token payload producer
→ optional server producer / bridge / live-index summary chain
```

ただし、まだ実 server 上では resolved physical index には到達していない。

現在の blocker:

```text
real server path には explicit fake req_to_token entries / produced req_to_token entries がまだ無い
```

これは failure ではなく、今回の phase の設計通り。
今回の目的は、server 上で producer / bridge / index-read の summary chain が安全に接続されることの確認だった。

## 11. 次の選択肢

次は大きく2つの進め方がある。

### Option A: Phase 4.8.7.4 server runtime metadata から safe entries を作る入口を追加

目的:

```text
server runtime metadata
→ smoke-safe req_to_token entries production
→ bridge
→ live token_to_kv_pool index-read
→ resolved physical index summary
```

ただし、この段階からは live runtime metadata の扱いが一段危険になるため、以下を慎重に分ける必要がある。

- metadata-only source
- explicit bounded entries
- req_to_token_pool value read をまだ禁止するか、限定解禁するか
- source mutation / tensor read / KV pool read の zero invariant

### Option B: Phase 4.9 / Phase 5 に進む前の design consolidation

現在の boundary を整理し、次にどこで req_to_token value read を限定解禁するか、あるいは既存 metadata だけで payload を作るかを設計する。

推奨は Option B を短く挟み、その後 Option A に入ること。

## 12. Commit command block

今回の phase 群で未 commit のファイルがある場合の commit / push 例。

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

git status --short
git diff --check

# Phase 4.8.6.1
git add \
  python/sglang/srt/relaykv/metrics.py \
  scripts/relaykv_req_to_token_resolution_payload_bridge_smoke.py

git commit -m "relaykv: add req-to-token payload bridge smoke"
git push mine relaykv-host-backup-shadow

# Phase 4.8.6.2
git add \
  python/sglang/srt/relaykv/metrics.py \
  scripts/relaykv_model_runner_req_to_token_bridge_live_index_read_smoke.py

git commit -m "relaykv: bridge req-to-token payloads into live index read"
git push mine relaykv-host-backup-shadow

# Phase 4.8.6.3
git add scripts/relaykv_optional_server_req_to_token_bridge_live_index_read_smoke.py

git commit -m "relaykv: add optional req-to-token bridge server smoke"
git push mine relaykv-host-backup-shadow

# Phase 4.8.7.1
git add \
  python/sglang/srt/relaykv/metrics.py \
  scripts/relaykv_runtime_req_to_token_resolution_payload_production_smoke.py

git commit -m "relaykv: add runtime req-to-token payload production smoke"
git push mine relaykv-host-backup-shadow

# Phase 4.8.7.2
git add \
  python/sglang/srt/model_executor/model_runner.py \
  python/sglang/srt/relaykv/metrics.py \
  scripts/relaykv_fake_model_runner_runtime_req_to_token_payload_production_smoke.py

git commit -m "relaykv: wire runtime req-to-token payload production hook"
git push mine relaykv-host-backup-shadow

# Phase 4.8.7.3
git add scripts/relaykv_optional_server_runtime_req_to_token_payload_production_bridge_live_index_read_smoke.py

git commit -m "relaykv: add optional payload production bridge server smoke"
git push mine relaykv-host-backup-shadow
```

この devlog 自体を repo に保存する場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/devlog_2026-05-06_relaykv_phase4_8_6_4_8_7_req_to_token_payload_bridge_production.ja.md \
  notes/devlog_2026-05-06_relaykv_phase4_8_6_4_8_7_req_to_token_payload_bridge_production.ja.md

git status --short
git diff --check

git add notes/devlog_2026-05-06_relaykv_phase4_8_6_4_8_7_req_to_token_payload_bridge_production.ja.md
git commit -m "docs: add relaykv req-to-token bridge production devlog"
git push mine relaykv-host-backup-shadow
```
