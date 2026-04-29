# Devlog: RelayKV SGLang MVP-1b-2c+ Token Pool Mapping Observation

## 日付

2026-04-29

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`
- remote: `mine/relaykv-host-backup-shadow`

## 今回の目的

RelayKV MVP-1b-2c+ として、request logical token range から SGLang KV pool index への mapping observation を追加し、実サーバーで確認した。

この段階でも、CPU copy は行っていない。

## 守った制約

- KV tensor は copy しない
- CPU copy はしない
- GPU KV は free しない
- tensor content は読まない
- attention kernel は変更しない
- scheduler の実挙動は変えない
- 通常生成結果は変えない
- `.github/workflows` は触らない

## 実行条件

```text
model: Qwen/Qwen2.5-1.5B-Instruct
mode: shadow
resident_budget_tokens: 1024
recent_window: 768
anchor_pages: 4
host_backup_shadow: true
host_backup_max_mib: 0.0
host_backup_dry_copy: true
```

## 観測された KV layout

```text
kv_layout_observed: true
kv_layout_object_type: MHATokenToKVPool
kv_layout_k_shape: [160824, 2, 128]
kv_layout_v_shape: [160824, 2, 128]
kv_layout_dtype: torch.bfloat16
kv_layout_device: cuda:0
kv_layout_num_layers_observed: 28
kv_layout_reason: ok
```

## 観測された token pool mapping

```text
kv_pool_mapping_observed: true
kv_pool_mapping_object_type: ReqToTokenPool
kv_pool_mapping_shape: [2512, 32772]
kv_pool_mapping_dtype: torch.int32
kv_pool_mapping_device: cuda:0
kv_pool_mapping_reason: ok
```

request全体:

```text
request_pool_indices_count: 2535
request_pool_indices_preview_head: [15, 16, 17, 18, 19, 20, 21, 22]
request_pool_indices_preview_tail: [0, 0, 0, 0, 0, 0, 0, 0]
```

cold range:

```text
cold_candidate_ranges: [[4, 1767]]
cold_range_pool_mapping_supported: true
cold_range_pool_mapping_reason: ok
cold_range_pool_indices_count: 1763
cold_range_pool_indices_preview: [19, 20, 21, 22, 23, 24, 25, 26]
```

## 解釈

`ReqToTokenPool` を通して、request logical token index から pool index への対応が観測できた。

今回の例では:

```text
logical token 0 -> pool index 15
logical token 1 -> pool index 16
logical token 2 -> pool index 17
logical token 3 -> pool index 18
logical token 4 -> pool index 19
```

したがって、cold candidate range `[4, 1767]` は先頭側では pool index `[19, 20, 21, ...]` に対応している。

`cold_range_pool_indices_count = 1763` は `1767 - 4 = 1763` と一致しており、range-based copy target discovery と mapping observation が整合している。

## 注意点

`request_pool_indices_preview_tail` が `[0, 0, 0, 0, 0, 0, 0, 0]` になっている。

これは、観測タイミングが chunked prefill の途中、または req_to_token_pool 上で末尾側のmappingがまだ完全に埋まっていない可能性を示す。

ログにも chunked prefill が出ている。

```text
Prefill batch, #new-token: 2048, #pending-token: 487
Prefill batch, #new-token: 487, #pending-token: 0
```

したがって、実CPU copyへ進む前に以下を確認する必要がある。

```text
- pending-token が 0 になった後のmapping
- tail側pool indicesが0でなくなるタイミング
- copy target rangeが全て有効pool indexに変換できること
```

## dry-copy guard状態

```text
host_backup_dry_copy: true
host_backup_dry_copy_guard_ok: true
host_backup_dry_copy_would_run: true
host_backup_dry_copy_reason: guard_only_no_tensor_copy
host_backup_would_copy: false
```

意味:

```text
dry-copy可能条件は満たしているが、MVP-1b-2c+ は mapping observation のみで、実copyはしない。
```

## API確認

```text
POST /v1/chat/completions HTTP/1.1" 200 OK
```

通常生成経路は壊れていない。

## 現在の到達点

```text
MVP-0:
  server args / skeleton / shadow plan / no-op / profile metadata 完了

MVP-1a:
  KV memory estimate / budget sweep / smoke test 完了

MVP-1b-0:
  host backup shadow flags / config 完了

MVP-1b-1:
  host backup candidate metadata log / runtime verification 完了

MVP-1b-2a:
  cold/resident range schema / copy target discovery log 完了

MVP-1b-2b:
  dry-copy flag / guard-only path / runtime verification 完了

MVP-1b-2c:
  KV layout observation / runtime verification 完了

MVP-1b-2c+:
  token pool mapping observation / runtime verification 完了
```

## 次にやること

次は、いきなりCPU copyに入る前に、**prefill completion aware mapping observation** を追加するのが安全。

理由:

```text
request_pool_indices_preview_tail が 0 であり、chunked prefill途中のmappingを見ている可能性がある。
```

次に確認したいこと:

```text
- pending-token == 0 のタイミングで同じrequestのmappingを観測できるか
- copy target rangesのpool indicesが全て nonzero / valid か
- valid pool indices count と copy_target_tokens が一致するか
```

## 次の推奨タスク

```text
MVP-1b-2c++:
  Add prefill-complete aware token pool mapping observation.

- tensor copy しない
- CPU copy しない
- GPU KV free しない
- attention変更しない
- pending-token が残っている場合は actual copy readiness を false にする
- mapping_valid_count / mapping_zero_count をログする
```

## CPU dry-copyに進む条件

最低限、以下が揃ってから。

```text
host_backup_dry_copy_guard_ok == true
kv_layout_observed == true
kv_pool_mapping_observed == true
cold_range_pool_mapping_supported == true
mapping_zero_count == 0
pending_tokens == 0
copy_target_tokens == valid_pool_indices_count
```
