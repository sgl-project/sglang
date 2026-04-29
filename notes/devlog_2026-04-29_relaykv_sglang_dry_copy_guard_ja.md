# Devlog: RelayKV SGLang MVP-1b-2b Host Backup Dry-Copy Guard

## 日付

2026-04-29

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`
- remote: `mine/relaykv-host-backup-shadow`

## 今回の目的

RelayKV MVP-1b-2b として、host backup dry-copy の明示flagと guard-only path を追加した。

この段階では、dry-copy の実行条件を metadata/log として判定するだけで、実際の tensor copy はまだ行わない。

## 重要な制約

今回も以下を守る。

- KV tensor は動かさない
- CPU copy はしない
- GPU KV は free しない
- attention kernel は変更しない
- scheduler の実挙動は変えない
- 通常生成結果は変えない
- `.github/workflows` は触らない

## 追加したserver arg

```text
--relaykv-host-backup-dry-copy
```

意味:

```text
host backup dry-copy path を明示的に有効化するためのflag。
ただし MVP-1b-2b では guard-only で、実copyはしない。
```

## 追加した想定config項目

```text
host_backup_dry_copy: bool
```

## 追加した想定ログ項目

```text
host_backup_dry_copy
host_backup_dry_copy_guard_ok
host_backup_dry_copy_would_run
host_backup_dry_copy_reason
```

## guard条件

dry-copy guard は以下を満たす場合のみ OK。

```text
relaykv_enabled == true
mode == "shadow"
host_backup_shadow == true
host_backup_dry_copy == true
host_backup_budget_ok == true
host_backup_copy_target_tokens > 0
```

ただし、MVP-1b-2b では guard がOKでも実copyはしない。

期待されるログ:

```text
host_backup_dry_copy_guard_ok: true
host_backup_dry_copy_would_run: true
host_backup_dry_copy_reason: guard_only_no_tensor_copy
```

## disabled / budget exceeded の挙動

### dry-copy flagなし

```text
host_backup_dry_copy: false
host_backup_dry_copy_guard_ok: false
host_backup_dry_copy_would_run: false
host_backup_dry_copy_reason: dry_copy_disabled
```

### budget exceeded

```text
host_backup_budget_ok: false
host_backup_dry_copy_guard_ok: false
host_backup_dry_copy_would_run: false
host_backup_dry_copy_reason: host_backup_budget_exceeded
```

## コミット

ユーザーにより commit / push 済み。

想定コミットメッセージ:

```text
Add RelayKV host backup dry-copy guard
```

## 次に確認すること

実サーバーで以下3ケースを確認する。

### Case A: dry-copy flagなし

起動:

```bash
PYTHONPATH=python python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --enable-relaykv \
  --relaykv-mode shadow \
  --relaykv-resident-budget-tokens 1024 \
  --relaykv-recent-window 768 \
  --relaykv-anchor-pages 4 \
  --relaykv-log-interval 1 \
  --relaykv-host-backup-shadow \
  --relaykv-host-backup-max-mib 0
```

期待:

```text
host_backup_dry_copy: false
host_backup_dry_copy_guard_ok: false
host_backup_dry_copy_would_run: false
host_backup_dry_copy_reason: dry_copy_disabled
```

### Case B: dry-copy flagあり / budget OK

起動:

```bash
PYTHONPATH=python python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --enable-relaykv \
  --relaykv-mode shadow \
  --relaykv-resident-budget-tokens 1024 \
  --relaykv-recent-window 768 \
  --relaykv-anchor-pages 4 \
  --relaykv-log-interval 1 \
  --relaykv-host-backup-shadow \
  --relaykv-host-backup-max-mib 0 \
  --relaykv-host-backup-dry-copy
```

期待:

```text
host_backup_budget_ok: true
host_backup_dry_copy: true
host_backup_dry_copy_guard_ok: true
host_backup_dry_copy_would_run: true
host_backup_dry_copy_reason: guard_only_no_tensor_copy
```

### Case C: dry-copy flagあり / budget exceeded

起動:

```bash
PYTHONPATH=python python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --enable-relaykv \
  --relaykv-mode shadow \
  --relaykv-resident-budget-tokens 1024 \
  --relaykv-recent-window 768 \
  --relaykv-anchor-pages 4 \
  --relaykv-log-interval 1 \
  --relaykv-host-backup-shadow \
  --relaykv-host-backup-max-mib 16 \
  --relaykv-host-backup-dry-copy
```

期待:

```text
host_backup_budget_ok: false
host_backup_dry_copy: true
host_backup_dry_copy_guard_ok: false
host_backup_dry_copy_would_run: false
host_backup_dry_copy_reason: host_backup_budget_exceeded
```

### request

```bash
curl -sS -i http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/relaykv_long_payload.json
```

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
  dry-copy flag / guard-only path 完了
```

## 次の推奨

次は実サーバーで dry-copy guard の3ケースを確認する。

その後、MVP-1b-2c として SGLang KV layout observation に進む。

ただし layout observation でも、まだ tensor copy はしない。

MVP-1b-2c の候補:

```text
- KV cache object の存在場所を確認
- shape / dtype / device のみログ
- request_id と token ranges の対応を確認
- tensor content は読まない
- CPU copy はしない
```
