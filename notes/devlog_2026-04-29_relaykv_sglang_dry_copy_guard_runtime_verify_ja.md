# Devlog: RelayKV SGLang MVP-1b-2b Dry-Copy Guard Runtime Verification

## 日付

2026-04-29

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`
- remote: `mine/relaykv-host-backup-shadow`

## 今回の目的

RelayKV MVP-1b-2b で追加した host backup dry-copy guard-only path について、実サーバーで3ケースを確認した。

この確認でも、実際のCPU copyは行っていない。

## 共通条件

```text
model: Qwen/Qwen2.5-1.5B-Instruct
mode: shadow
resident_budget_tokens: 1024
recent_window: 768
anchor_pages: 4
seq_len: 2535
planned_resident_tokens: 1024
planned_cold_tokens: 1511
planned_resident_kv_mib: 28.0
planned_cold_kv_mib: 41.316
host_backup_copy_target_ranges: [[4, 1767]]
host_backup_copy_target_tokens: 1763
```

## Case A: dry-copy flagなし

条件:

```text
--relaykv-host-backup-shadow
--relaykv-host-backup-max-mib 0
# --relaykv-host-backup-dry-copy なし
```

確認ログ:

```text
host_backup_dry_copy: false
host_backup_dry_copy_guard_ok: false
host_backup_dry_copy_would_run: false
host_backup_dry_copy_reason: dry_copy_disabled
host_backup_budget_ok: true
host_backup_would_copy: false
```

判定:

```text
OK
```

## Case B: dry-copy flagあり / budget OK

条件:

```text
--relaykv-host-backup-shadow
--relaykv-host-backup-max-mib 0
--relaykv-host-backup-dry-copy
```

確認ログ:

```text
host_backup_dry_copy: true
host_backup_dry_copy_guard_ok: true
host_backup_dry_copy_would_run: true
host_backup_dry_copy_reason: guard_only_no_tensor_copy
host_backup_budget_ok: true
host_backup_would_copy: false
```

判定:

```text
OK
```

意味:

```text
dry-copyを実行可能な条件は満たしている。
ただしMVP-1b-2bはguard-onlyなので、実copyはしない。
```

## Case C: dry-copy flagあり / budget exceeded

条件:

```text
--relaykv-host-backup-shadow
--relaykv-host-backup-max-mib 16
--relaykv-host-backup-dry-copy
```

確認ログ:

```text
host_backup_dry_copy: true
host_backup_dry_copy_guard_ok: false
host_backup_dry_copy_would_run: false
host_backup_dry_copy_reason: host_backup_budget_exceeded
host_backup_budget_ok: false
host_backup_would_copy: false
```

判定:

```text
OK
```

意味:

```text
host backup candidate は 41.316 MiB。
max_mib は 16.0 MiB。
budget exceeded のため dry-copy guard は false。
```

## OpenAI互換API確認

3ケースすべてで以下を確認。

```text
POST /v1/chat/completions HTTP/1.1" 200 OK
```

通常生成経路は壊れていない。

## 安全条件

3ケースすべてで以下を維持。

```text
host_backup_would_copy: false
```

また、実copyしないことが以下で明示されている。

```text
dry_copy_disabled
guard_only_no_tensor_copy
host_backup_budget_exceeded
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
  dry-copy flag / guard-only path / runtime verification 完了
```

## 次にやること

次は MVP-1b-2c として、SGLang KV layout observation に進む。

ただし、まだ tensor copy はしない。

目的:

```text
- KV cache object の存在場所を特定
- shape / dtype / device のみログ
- request_id と token ranges の対応を確認
- copy可能性を判断する
```

やらないこと:

```text
- tensor contentを読む
- CPU copyする
- GPU KVをfreeする
- attentionに使う
- scheduler挙動を変える
```

## 次のCodexタスク候補

```text
RelayKV MVP-1b-2c として、host backup dry-copy の前準備として KV layout observation を追加してください。

重要:
- KV tensor はまだcopyしない
- CPU copyしない
- GPU KVをfreeしない
- attention / scheduler behavior は変えない
- .github/workflows は触らない

実装内容:
- dry-copy guardがtrueのときだけ、KV cache object のshape/dtype/deviceを安全に観察する
- 取得できない場合はwarning reasonをログしてno-op
- tensor contentは読まない
- copy対象rangeとKV layoutの対応可否をmetadataとしてログする
```
