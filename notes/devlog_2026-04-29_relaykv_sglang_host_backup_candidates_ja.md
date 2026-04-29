# Devlog: RelayKV SGLang MVP-1b-1 Host Backup Candidate Metadata Log

## 日付

2026-04-29

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`
- remote: `mine/relaykv-host-backup-shadow`

## 今回の目的

RelayKV MVP-1b-1 として、host backup shadow 用の candidate metadata log を追加した。

この段階では、planned cold KV を「host backup するとしたら対象になるKV」として metadata 上で扱うだけであり、実際の tensor copy はまだ行わない。

## 重要な制約

今回も KV 実体には触らない。

- KV tensor は動かさない
- CPU copy はしない
- GPU KV は free しない
- attention kernel は変更しない
- scheduler の実挙動は変えない
- 通常生成結果は変えない
- `.github/workflows` は触らない

## 前提

MVP-1b-0 で以下の server args / config が追加済み。

```text
--relaykv-host-backup-shadow
--relaykv-host-backup-max-mib
```

## 今回追加した想定ログ項目

`host_backup_shadow` が true のとき、planned cold KV を host backup candidate としてログする。

想定項目:

```text
host_backup_candidate_tokens
host_backup_candidate_kv_bytes
host_backup_candidate_kv_mib
host_backup_max_mib
host_backup_budget_ok
host_backup_would_copy
host_backup_reason
```

## 判定仕様

### `host_backup_max_mib == 0.0`

0.0 は unlimited として扱う。

```text
host_backup_budget_ok = true
host_backup_would_copy = false
host_backup_reason = metadata_only_no_tensor_copy
```

この段階では `would_copy` は false のままにする。理由は、MVP-1b-1 は metadata-only であり、実際のCPU copyはまだしないため。

### `host_backup_max_mib > 0`

`planned_cold_kv_mib` と比較する。

```text
planned_cold_kv_mib <= host_backup_max_mib:
  host_backup_budget_ok = true

planned_cold_kv_mib > host_backup_max_mib:
  host_backup_budget_ok = false
```

どちらの場合も、今回の段階では `host_backup_would_copy = false`。

## 期待される意味

MVP-1b-1 により、将来CPU host backupを行う場合の対象量を request単位で見られるようになる。

例:

```text
planned_cold_tokens
planned_cold_kv_mib
host_backup_candidate_tokens
host_backup_candidate_kv_mib
host_backup_budget_ok
```

これにより、実copy前に以下を評価できる。

- cold KV がどのくらい大きいか
- host backup の候補量がどの程度か
- host backup budget を超えるか
- requestごとに backup 可否を metadata で判定できるか

## 確認コマンド

```bash
cd ~/work/sglang-relaykv
source .venv/bin/activate

git status --short
git log --oneline --decorate --max-count=10

python -m compileall python/sglang/srt/relaykv python/sglang/srt/managers/scheduler.py

PYTHONPATH=python python -m sglang.launch_server --help | grep -A 60 -i relaykv

PYTHONPATH=python python scripts/relaykv_memory_smoke.py

git diff --name-status | grep '.github/workflows' || true
```

## 実サーバー確認案

### unlimited host backup metadata

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
host_backup_candidate_kv_mib = planned_cold_kv_mib
host_backup_budget_ok = true
host_backup_would_copy = false
```

### budget too small

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
  --relaykv-host-backup-max-mib 16
```

期待:

```text
planned_cold_kv_mib ≈ 41.316
host_backup_max_mib = 16
host_backup_budget_ok = false
host_backup_would_copy = false
```

### request

```bash
curl -sS -i http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/relaykv_long_payload.json
```

## コミット

ユーザーにより commit / push 済み。

想定コミットメッセージ:

```text
Log RelayKV host backup candidates
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
  host backup candidate metadata log 完了
```

## 次にやること

次は、実サーバーで host backup candidate metadata log を確認する。

確認後、次の候補は2つ。

### A. MVP-1b-1を締める

- unlimited case
- budget-too-small case
- disabled/mode off no-op
- smoke test

### B. MVP-1b-2 dry-copy の設計だけ追加

ただし、まだ実装に入る前に、copy対象・copyタイミング・保存先・メモリ上限・例外時挙動を設計メモに固定する。

## 推奨

次はまず実サーバー確認。

その後、MVP-1b-2 に進む前に、以下の設計メモを追加するのが安全。

```text
notes/relaykv_host_backup_dry_copy_design_ja.md
```

設計メモで決めること:

- copy対象は cold KV のどの範囲か
- prefill後copyか、chunked prefillごとか
- layer単位か token range単位か
- host buffer の形式
- max_mib 超過時の挙動
- copy failure時の no-op fallback
- GPU KVはfreeしない方針
- attentionには使わない方針
