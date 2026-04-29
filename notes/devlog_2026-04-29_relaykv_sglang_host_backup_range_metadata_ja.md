# Devlog: RelayKV SGLang MVP-1b-2a Host Backup Range Metadata

## 日付

2026-04-29

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`
- remote: `mine/relaykv-host-backup-shadow`

## 今回の目的

RelayKV MVP-1b-2a として、host backup dry-copy に進む前の copy target discovery を行うため、cold/resident range schema を追加した。

この段階でも、tensor copy はまだ行わない。

## 重要な制約

今回も以下を守る。

- KV tensor は動かさない
- CPU copy はしない
- GPU KV は free しない
- attention kernel は変更しない
- scheduler の実挙動は変えない
- 通常生成結果は変えない
- `.github/workflows` は触らない

## 追加した想定range metadata

ShadowPlan / shadow log に以下の range metadata を追加した。

```text
resident_anchor_ranges
resident_recent_ranges
cold_candidate_ranges
host_backup_copy_target_ranges
host_backup_copy_target_tokens
host_backup_copy_target_reason
```

## range schema の意味

### resident_anchor_ranges

anchor pages/tokens として常時 resident 扱いにする範囲。

例:

```json
[[0, 4]]
```

### resident_recent_ranges

recent window として resident 扱いにする範囲。

例:

```json
[[1767, 2535]]
```

### cold_candidate_ranges

anchor / recent に含まれない middle range。

host backup dry-copy に進む場合の候補範囲。

例:

```json
[[4, 1767]]
```

### host_backup_copy_target_ranges

MVP-1b-2a 時点では、実copy対象ではなく「将来copyするとしたら対象になるrange」を示す。

## planned_cold_tokens と range tokens の注意

`planned_cold_tokens` は budget-based な値。

一方、`cold_candidate_ranges` は range-based な値。

そのため、以下が完全一致しない可能性がある。

```text
planned_cold_tokens
sum(cold_candidate_ranges)
```

この不一致は例外にせず、metadata reason として扱う方針。

理由:

```text
- resident budget は anchor + recent + retrieval 等の将来構成に依存する
- cold range は dry-copy候補の構造情報
- MVP-1b-2a では copy対象の明確化が目的であり、実tensor操作はしない
```

## 想定確認コマンド

```bash
cd ~/work/sglang-relaykv
source .venv/bin/activate

python -m compileall python/sglang/srt/relaykv python/sglang/srt/managers/scheduler.py

PYTHONPATH=python python scripts/relaykv_memory_smoke.py

git diff --name-status | grep '.github/workflows' || true
```

## 実サーバー確認案

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

別ターミナル:

```bash
curl -sS -i http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/relaykv_long_payload.json
```

期待するログ項目:

```text
resident_anchor_ranges
resident_recent_ranges
cold_candidate_ranges
host_backup_copy_target_ranges
host_backup_copy_target_tokens
host_backup_copy_target_reason
host_backup_would_copy: false
host_backup_reason: metadata_only_no_tensor_copy
```

## コミット

ユーザーにより commit / push 済み。

想定コミットメッセージ:

```text
Add RelayKV host backup range metadata
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
```

## 次にやること

次は実サーバーで range metadata が期待通り出るか確認する。

確認後、MVP-1b-2b として dry-copy flag / guard を追加する。

ただしまだ tensor copy は行わない。

## 次の推奨タスク

```text
MVP-1b-2b:
  Add explicit host backup dry-copy flag and guard-only path.

- --relaykv-host-backup-dry-copy を追加
- configに host_backup_dry_copy を追加
- dry-copy実行条件を判定してログする
- 条件を満たしても、まだ tensor copy はしない
- host_backup_dry_copy_would_run をログする
```
