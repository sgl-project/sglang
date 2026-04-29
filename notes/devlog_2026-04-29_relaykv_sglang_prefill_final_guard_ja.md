# Devlog: RelayKV SGLang Dry-Copy Final Guard on Prefill Completion

## 日付

2026-04-29

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`
- remote: `mine/relaykv-host-backup-shadow`

## 今回の目的

RelayKV host backup dry-copy の最終実行条件に、`prefill_complete_for_request == true` を追加した。

背景として、前段の mapping observation では copy target range 自体は ready になっていたが、request全体では chunked prefill が未完了だった。

そのため、CPU dry-copy prototype に進む前の安全条件として、request全体の prefill 完了を final guard に含めた。

## 守った制約

今回も以下は行っていない。

- KV tensor copy
- CPU copy
- GPU KV free
- attention kernel変更
- scheduler挙動変更
- 通常生成結果変更
- `.github/workflows` 変更

## 確認ログ

実サーバーで以下を確認。

```text
mapping_ready_for_copy: true
mapping_valid_count: 1763
mapping_zero_count: 0
mapping_invalid_count: 0

prefill_complete_for_request: false
prefill_pending_tokens: 487

host_backup_dry_copy_guard_ok: true
host_backup_dry_copy_would_run: true

host_backup_dry_copy_final_guard_ok: false
host_backup_dry_copy_final_guard_reason: prefill_not_complete
```

## 判定

期待通り。

copy target range `[4, 1767]` 自体は mapping ready だが、request全体の prefill はまだ完了していない。

そのため、final guard は以下で止まる。

```text
host_backup_dry_copy_final_guard_ok: false
host_backup_dry_copy_final_guard_reason: prefill_not_complete
```

## 重要な意味

これにより、将来CPU dry-copyを実装する際、chunked prefill途中で誤ってcopyするリスクを下げられる。

現状の関係:

```text
mapping_ready_for_copy:
  copy target range のpool indexが有効かどうか

prefill_complete_for_request:
  request全体のprefillが完了しているかどうか

host_backup_dry_copy_final_guard_ok:
  実copyに進んでよいかどうか
```

今回のログでは:

```text
mapping_ready_for_copy = true
prefill_complete_for_request = false
final_guard_ok = false
```

これは安全側の挙動として正しい。

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

MVP-1b-2c++:
  prefill completion final guard / runtime verification 完了
```

## 次にやること

次にCPU dry-copy prototypeへ進む前に、prefill完了後のhook位置を探す。

目的:

```text
prefill_complete_for_request == true
prefill_pending_tokens == 0
request_pool_indices_count == seq_len
mapping_zero_count == 0
host_backup_dry_copy_final_guard_ok == true
```

になるタイミングを観測する。

## 次の推奨タスク

```text
MVP-1b-2c+++
Find or add a prefill-complete observation hook.

- CPU copyしない
- GPU KV freeしない
- attention変更しない
- pending-token == 0 のタイミングで mapping readiness を再評価する
- final_guard_ok が true になるかログする
```

## CPU dry-copyに進む最小条件

```text
host_backup_dry_copy_guard_ok == true
kv_layout_observed == true
kv_pool_mapping_observed == true
cold_range_pool_mapping_supported == true
mapping_ready_for_copy == true
prefill_complete_for_request == true
prefill_pending_tokens == 0
request_pool_indices_count == seq_len
host_backup_dry_copy_final_guard_ok == true
```
