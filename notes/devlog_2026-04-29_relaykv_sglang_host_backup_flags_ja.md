# Devlog: RelayKV SGLang MVP-1b-0 Host Backup Shadow Flags

## 日付

2026-04-29

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`
- remote: `mine/relaykv-host-backup-shadow`

## 今回の目的

RelayKV MVP-1b-0 として、host backup shadow のための明示flagとconfig項目を追加した。

この段階では、host backup はまだ metadata / log schema の準備のみ。

## 重要な制約

今回も runtime のKV実体には触らない。

- KV tensor は動かさない
- CPU copy はしない
- GPU KV は free しない
- attention kernel は変更しない
- scheduler の実挙動は変えない
- 通常生成結果は変えない
- `.github/workflows` は触らない

## 追加した想定server args

```text
--relaykv-host-backup-shadow
--relaykv-host-backup-max-mib
```

意味:

```text
--relaykv-host-backup-shadow:
  host backup shadow planning/logging を有効化する。
  ただし tensor copy はしない。

--relaykv-host-backup-max-mib:
  host backup shadow planning 用の上限メタデータ。
  0.0 は unlimited。
```

## 追加した想定config項目

```text
host_backup_shadow: bool
host_backup_max_mib: float
```

## shadow log schema への追加項目

```text
host_backup_shadow
host_backup_max_mib
host_backup_planned
host_backup_reason
```

## 期待される挙動

### host_backup_shadow=false

既存の MVP-1a と同じ。

```text
shadow plan log
KV memory estimate log
host backup planned: false
```

### host_backup_shadow=true

metadata上は host backup shadow を計画するが、まだ tensor copy はしない。

```text
shadow plan log
KV memory estimate log
host_backup_shadow: true
host_backup_planned: true or false
host_backup_reason: ...
```

## 確認コマンド

```bash
cd ~/work/sglang-relaykv
source .venv/bin/activate

python -m compileall python/sglang/srt/relaykv python/sglang/srt/managers/scheduler.py

PYTHONPATH=python python -m sglang.launch_server --help | grep -A 50 -i relaykv

PYTHONPATH=python python scripts/relaykv_memory_smoke.py

git diff --name-status | grep '.github/workflows' || true
```

## コミット

ユーザーにより実装・commit / push 済み。

想定コミットメッセージ:

```text
Add RelayKV host backup shadow flags
```

## 次にやること

次は **MVP-1b-1: host backup candidate metadata log** に進む。

まだ tensor copy はしない。

目的:

```text
cold token ranges
estimated host backup bytes
would-copy layers
would-copy dtype
host backup budget check
```

をログに追加する。

## MVP-1b-1 の方針

### やること

- shadow plan の cold tokens / cold ranges を host backup candidate として扱う
- planned_cold_kv_mib を host backup candidate bytes としてログする
- `host_backup_max_mib` が 0.0 なら unlimited
- `host_backup_max_mib > 0` の場合は planned_cold_kv_mib と比較して guard 判定をログする
- host backup が可能かどうかを metadata上で `would_backup` として出す

### まだやらないこと

- CPU tensor copy
- GPU KV free
- attentionでのhost backup利用
- swap-in / swap-out
- resident mappingの実適用

## 次のCodexタスク候補

```text
RelayKV MVP-1b-1 として host backup candidate metadata log を追加してください。

重要:
- KV tensor は動かさない
- CPU copy しない
- GPU KV を free しない
- attention / scheduler behavior は変えない
- .github/workflows は触らない
- 既存の shadow plan / memory estimate log を壊さない

実装内容:
- host_backup_shadow が true のときだけ、planned cold KV を host backup candidate としてログする
- host_backup_candidate_tokens
- host_backup_candidate_kv_mib
- host_backup_max_mib
- host_backup_budget_ok
- host_backup_would_copy
- host_backup_reason
- cold token range / recent range との関係が分かるメタデータ
- copyは絶対にしない
```
