# RelayKV Host Backup Dry-Copy Design

## 日付

2026-04-29

## 対象

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`
- phase: MVP-1b-2 design

## 目的

RelayKV MVP-1b-2 では、host backup dry-copy の最小プロトタイプを設計する。

ただし、MVP-1b-2 の初期段階では **GPU KVをfreeしない**、**attentionに使わない**、**swap-in/swap-outしない**。

目的は、cold KV をCPUへ退避コピーできるか、どの程度の時間・host memoryを使うかを観測すること。

## 現在の到達点

### MVP-0

```text
server args / skeleton / shadow plan / no-op / profile metadata 完了
```

### MVP-1a

```text
KV memory estimate / budget sweep / smoke test 完了
```

### MVP-1b-0

```text
host backup shadow flags / config 完了
```

### MVP-1b-1

```text
host backup candidate metadata log / runtime verification 完了
```

確認済み:

```text
seq_len = 2535
planned_resident_tokens = 1024
planned_cold_tokens = 1511
planned_resident_kv_mib = 28.0
planned_cold_kv_mib = 41.316
host_backup_candidate_kv_mib = 41.316
host_backup_budget_ok = true / false depending on max_mib
host_backup_would_copy = false
host_backup_reason = metadata_only_no_tensor_copy
```

## MVP-1b-2 の原則

### やること

```text
cold KV candidate を CPU へ dry-copy する
copy time を測る
host backup bytes を測る
layer別 / request別にログする
```

### まだやらないこと

```text
GPU KVをfreeしない
attentionでCPU backupを使わない
GPUへのswap-inをしない
resident mappingをattention kernelへ渡さない
schedulerの割当・evict・batch挙動を変えない
```

## 追加する明示flag案

既存:

```text
--relaykv-host-backup-shadow
--relaykv-host-backup-max-mib
```

MVP-1b-2 で追加する候補:

```text
--relaykv-host-backup-dry-copy
```

意味:

```text
host backup candidate KV をCPUへcopyする。
ただしGPU側KVは保持し続け、attentionには使用しない。
```

安全条件:

```text
default: false
--enable-relaykv が必要
--relaykv-mode shadow が必要
--relaykv-host-backup-shadow が必要
```

つまり dry-copy は以下が揃ったときだけ動く。

```text
enable_relaykv == true
relaykv_mode == "shadow"
host_backup_shadow == true
host_backup_dry_copy == true
host_backup_budget_ok == true
```

## copy対象

初期実装では、shadow plan上の cold tokens を対象にする。

```text
cold token range:
  [0, seq_len) から resident対象を除いた範囲
```

ただし、現行plannerは主に以下を出している。

```text
anchor_pages
recent_page_range
planned_resident_tokens
planned_cold_tokens
```

MVP-1b-2では、まず copy対象を以下のように単純化する。

```text
host_backup_candidate_range:
  [anchor_pages_end, recent_page_start)
```

page_size=1 の現状では token index と page index が一致する。

例:

```text
seq_len = 2535
anchor_pages = [0, 1, 2, 3]
recent_page_range = [1767, 2535]

candidate cold range:
  [4, 1767)

candidate tokens:
  1763 tokens
```

注意:

現行の `planned_cold_tokens = 1511` は `seq_len - planned_resident_tokens` であり、anchor/recentの重複やresident budget配分と完全一致しない可能性がある。

そのため、MVP-1b-2前に planner の token range schema を明確化する。

## 推奨: range schema の追加

dry-copy前に、shadow planへ以下を追加する。

```text
resident_anchor_ranges
resident_recent_ranges
cold_candidate_ranges
```

例:

```json
{
  "resident_anchor_ranges": [[0, 4]],
  "resident_recent_ranges": [[1767, 2535]],
  "cold_candidate_ranges": [[4, 1767]]
}
```

この range schema を使えば、copy対象が明確になる。

## copyタイミング

初期実装は **prefill shadow log hook のタイミング** に限定する。

ただし SGLang は chunked prefill で同じ request が複数batchに分かれる。

既に重複ログ抑制が入っているため、dry-copyも同様に request_id 単位で一度だけにする。

```text
same request_id:
  dry-copy once
```

## chunked prefillとの関係

現状ログ例:

```text
Prefill batch, #new-token: 2048, #pending-token: 487
Prefill batch, #new-token: 487, #pending-token: 0
```

注意点:

1回目のprefill時点で、全seq_len相当のKVがGPU上に存在するとは限らない。

したがってMVP-1b-2のcopyタイミングは慎重に扱う。

### 安全な方針

まずは **metadata-only copy target discovery** を行う。

次に dry-copy するなら、`pending-token == 0` 相当、つまり requestのprefillが完了したタイミングに限定する。

もしscheduler hookで pending-token を安定して取れない場合は、dry-copy実装を延期し、copy target discoveryに留める。

## layer単位かtoken range単位か

初期dry-copyは layer単位で loop し、各layerの cold token range をCPUへcopyする想定。

概念:

```text
for layer in layers:
  k_cold = k_cache[layer][cold_ranges]
  v_cold = v_cache[layer][cold_ranges]
  host_backup[layer] = (k_cold.cpu(), v_cold.cpu())
```

ただし、SGLangの実際の paged KV layout は単純な `[seq, head, dim]` ではない可能性がある。

そのため、最初の実装では必ず実レイアウトを観察する。

## host buffer形式

初期案:

```python
RelayKVHostBackupRecord:
  request_id: str
  seq_len: int
  cold_ranges: list[tuple[int, int]]
  num_layers: int
  dtype: str
  total_bytes: int
  copy_time_ms: float
  tensors: optional
```

ただし最初の段階では `tensors` は保持しなくてもよい。

MVP-1b-2a:

```text
host backup record metadata only
```

MVP-1b-2b:

```text
actual CPU tensors stored in debug-only structure
```

## max_mib超過時の挙動

```text
host_backup_budget_ok == false:
  dry-copyしない
  log only
  generation continues normally
```

ログ:

```json
{
  "host_backup_dry_copy_attempted": false,
  "host_backup_dry_copy_skipped": true,
  "host_backup_skip_reason": "host_backup_budget_exceeded"
}
```

## copy failure時のfallback

copy failure は generation failure にしない。

```text
try:
  dry-copy
except Exception:
  log warning
  continue generation
```

ログ:

```json
{
  "host_backup_dry_copy_success": false,
  "host_backup_error": "...",
  "host_backup_fallback": "continue_without_host_backup"
}
```

## ログ項目

dry-copy前の target discovery:

```text
host_backup_candidate_tokens
host_backup_candidate_kv_bytes
host_backup_candidate_kv_mib
host_backup_max_mib
host_backup_budget_ok
host_backup_would_copy
host_backup_reason
cold_candidate_ranges
```

dry-copy時:

```text
host_backup_dry_copy_enabled
host_backup_dry_copy_attempted
host_backup_dry_copy_success
host_backup_dry_copy_skipped
host_backup_skip_reason
host_backup_copy_time_ms
host_backup_actual_bytes
host_backup_actual_mib
host_backup_num_layers
host_backup_dtype
```

## 実装ステップ

### MVP-1b-2a: range schema / target discovery

```text
- shadow plan に resident/cold ranges を追加
- host backup candidate range をログする
- tensorには触らない
```

### MVP-1b-2b: dry-copy flags / guards

```text
- --relaykv-host-backup-dry-copy を追加
- configに追加
- 条件判定だけ入れる
- tensorには触らない
```

### MVP-1b-2c: layout observation

```text
- SGLang KV cache layout を安全に観察
- shape / dtype / device だけログ
- tensor copyはしない
```

### MVP-1b-2d: explicit CPU copy prototype

```text
- dry-copy flagがtrueのときだけCPU copy
- GPU KVはfreeしない
- attentionには使わない
- copy time / bytesをログ
```

## 推奨する次タスク

いきなりCPU copyではなく、まず次を行う。

```text
MVP-1b-2a:
  Add RelayKV cold/resident range schema and host backup copy target discovery logs.
```

理由:

```text
copy対象が曖昧なままtensor copyに進むと危険
chunked prefillとの整合性がまだ未確認
SGLangのKV layoutへの依存を最小化できる
```

## 次のCodexタスク案

```text
RelayKV MVP-1b-2a として、cold/resident range schema と host backup copy target discovery log を追加してください。

重要:
- KV tensor は絶対に動かさない
- CPU copy しない
- GPU KV を free しない
- attention / scheduler behavior は変えない
- .github/workflows は触らない
- 既存の shadow plan / memory estimate / host backup candidate log を壊さない

実装内容:
1. ShadowPlan に以下のrange metadataを追加する
   - resident_anchor_ranges
   - resident_recent_ranges
   - cold_candidate_ranges
2. page_size=1 の現状では token/page indexを同一として扱ってよい
3. host backup log に cold_candidate_ranges を追加する
4. range計算と planned_cold_tokens の関係が分かる reason を追加する
5. scripts/relaykv_memory_smoke.py を更新して schema guard に range fields を追加する
6. tensor copy は絶対にしない
```
