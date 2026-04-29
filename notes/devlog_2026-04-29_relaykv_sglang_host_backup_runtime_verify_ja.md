# Devlog: RelayKV SGLang MVP-1b-1 Runtime Verification

## 日付

2026-04-29

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`
- remote: `mine/relaykv-host-backup-shadow`

## 今回の目的

RelayKV MVP-1b-1 で追加した host backup candidate metadata log について、実サーバー上で以下を確認した。

1. `host_backup_max_mib = 0.0` の unlimited case
2. `host_backup_max_mib = 16.0` の budget too small case
3. OpenAI互換APIが引き続き `200 OK` を返すこと
4. CPU copy / GPU free / attention変更はまだ行っていないことをログ上の metadata-only reason で確認

## 実行条件

共通条件:

```text
model: Qwen/Qwen2.5-1.5B-Instruct
relaykv_mode: shadow
resident_budget_tokens: 1024
recent_window: 768
anchor_pages: 4
seq_len: 2535
planned_resident_tokens: 1024
planned_cold_tokens: 1511
planned_resident_kv_mib: 28.0
planned_cold_kv_mib: 41.316
```

## Case 1: unlimited host backup metadata

起動条件:

```text
--relaykv-host-backup-shadow
--relaykv-host-backup-max-mib 0
```

確認ログ:

```text
host_backup_shadow: true
host_backup_max_mib: 0.0
host_backup_candidate_tokens: 1511
host_backup_candidate_kv_bytes: 43323392
host_backup_candidate_kv_mib: 41.316
host_backup_planned: true
host_backup_budget_ok: true
host_backup_would_copy: false
host_backup_reason: metadata_only_no_tensor_copy
```

判定:

```text
OK
```

理由:

```text
host_backup_max_mib = 0.0 は unlimited 扱いなので、
candidate 41.316 MiB に対して budget_ok = true で正しい。
ただし MVP-1b-1 は metadata-only なので would_copy = false で正しい。
```

## Case 2: host backup budget too small

起動条件:

```text
--relaykv-host-backup-shadow
--relaykv-host-backup-max-mib 16
```

確認ログ:

```text
host_backup_shadow: true
host_backup_max_mib: 16.0
host_backup_candidate_tokens: 1511
host_backup_candidate_kv_bytes: 43323392
host_backup_candidate_kv_mib: 41.316
host_backup_planned: true
host_backup_budget_ok: false
host_backup_would_copy: false
host_backup_reason: metadata_only_no_tensor_copy
```

判定:

```text
OK
```

理由:

```text
candidate 41.316 MiB > max 16.0 MiB なので、
budget_ok = false で正しい。
MVP-1b-1 は metadata-only なので would_copy = false で正しい。
```

## OpenAI互換API確認

両caseで以下を確認。

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
```

## 重要な安全条件

現時点では、以下はまだ未実装であり、意図通り。

```text
CPU tensor copy: なし
GPU KV free: なし
attentionでhost backup利用: なし
swap-in / swap-out: なし
resident mapping実適用: なし
```

ログ上も以下で metadata-only であることを明示できている。

```text
host_backup_would_copy: false
host_backup_reason: metadata_only_no_tensor_copy
```

## 次にやること

次は MVP-1b-2 dry-copy に入る前に、設計メモを追加する。

推奨ファイル:

```text
notes/relaykv_host_backup_dry_copy_design_ja.md
```

設計メモで固定すべき項目:

```text
- copy対象
- copyタイミング
- chunked prefillとの関係
- layer単位かtoken range単位か
- host buffer形式
- max_mib超過時の挙動
- copy failure時のfallback
- GPU KVはfreeしない
- attentionには使わない
- dry-copy用の明示flag
- ログ項目
```

## 次の推奨方針

MVP-1b-2 もいきなりproduction実装ではなく、以下の順で分ける。

### MVP-1b-2a: design note only

```text
dry-copyの設計メモだけ追加
```

### MVP-1b-2b: copy target discovery

```text
実tensorには触らず、copy対象のlayer/token rangeを確定してログする
```

### MVP-1b-2c: explicit dry-copy prototype

```text
明示flag付きでCPU copy
GPU KVはfreeしない
attentionには使わない
copy time / host bytesのみlog
```
