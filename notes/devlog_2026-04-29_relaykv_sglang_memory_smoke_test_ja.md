# Devlog: RelayKV SGLang MVP-1a+ Memory Estimate Smoke Test

## 日付

2026-04-29

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-memory-mvp0`
- remote: `mine/relaykv-memory-mvp0`

## 今回の目的

RelayKV MVP-1a+ として、KV memory estimate の計算結果を軽量 smoke test で確認できるようにした。

これにより、実サーバーを起動しなくても、Qwen2.5-1.5B 相当の fake profile / fake plan で以下を検証できる。

- `kv_bytes_per_token`
- logical / resident / cold KV bytes
- MiB換算
- shadow log schema の必須項目

## 重要な制約

今回も runtime 挙動は変えない。

- KV tensor は動かさない
- CPU copy はしない
- GPU KV は free しない
- attention kernel は変更しない
- scheduler の実挙動は変更しない
- 通常生成結果は変えない
- `.github/workflows` は触らない

## 追加された確認スクリプト

```text
scripts/relaykv_memory_smoke.py
```

実行コマンド:

```bash
cd ~/work/sglang-relaykv
source .venv/bin/activate

PYTHONPATH=python python scripts/relaykv_memory_smoke.py
```

## 期待値

Qwen2.5-1.5B 相当の fake metadata:

```text
num_layers = 28
num_key_value_heads = 2
head_dim = 128
kv_dtype_bytes = 2
```

式:

```text
kv_bytes_per_token
= num_layers * 2(K/V) * num_key_value_heads * head_dim * kv_dtype_bytes
= 28 * 2 * 2 * 128 * 2
= 28672 bytes/token
```

fake shadow plan:

```text
seq_len = 2535
planned_resident_tokens = 1024
planned_cold_tokens = 1511
```

期待される memory estimate:

```text
logical_kv_mib = 69.316
planned_resident_kv_mib = 28.0
planned_cold_kv_mib = 41.316
```

## schema guard

shadow log dict の必須項目を軽く確認する。

必須項目例:

```text
relaykv_enabled
mode
seq_len
planned_resident_tokens
planned_cold_tokens
estimated_resident_ratio
model_arch
attention_type
relaykv_profile_supported
num_layers
head_dim
kv_dtype_bytes
kv_bytes_per_token
logical_kv_mib
planned_resident_kv_mib
planned_cold_kv_mib
kv_memory_estimate_reason
```

## コミット

ユーザーにより commit / push 済み。

コミットメッセージ想定:

```text
Add RelayKV memory estimate smoke test
```

## 現在の状態

MVP-1a は以下まで完了。

```text
[x] server args
[x] skeleton
[x] shadow plan log
[x] runtime short/long verification
[x] disabled no-op
[x] mode off no-op
[x] Qwen2.5 GQA profile metadata
[x] KV memory estimate logging
[x] budget sweep 512 / 1024 / 2048
[x] memory estimate smoke test
[x] shadow log schema guard
```

## 次にやること

次は MVP-1b に入る前に、現在のブランチを区切る。

推奨:

1. `git status --short`
2. `git log --oneline --decorate --max-count=10`
3. 必要なら GitHub上の branch / commit を確認
4. MVP-1b 用にブランチを分ける

## 次ブランチ案

```bash
cd ~/work/sglang-relaykv

git status --short
git switch relaykv-memory-mvp0
git pull --ff-only

git switch -c relaykv-host-backup-shadow
```

## MVP-1b の推奨スコープ

MVP-1b は host backup dry-copy だが、いきなり本格copyに入らず、明示flag付きで段階化する。

### MVP-1b-0: flag / config only

```text
--relaykv-host-backup-shadow
--relaykv-host-backup-max-mib
```

まだ tensor copy しない。

### MVP-1b-1: metadata-only host backup candidate log

```text
cold token ranges
estimated host backup bytes
would-copy layers
would-copy dtype
```

まだ tensor copy しない。

### MVP-1b-2: explicit dry-copy

```text
明示flag付きでCPUへcopy
GPU KVはfreeしない
attentionには使わない
copy time / host bytesのみlog
```

## 次のCodexタスク候補

```text
RelayKV MVP-1b-0 として host backup shadow 用の明示flagとconfig項目だけ追加してください。

- KV tensor は動かさない
- CPU copy しない
- GPU KV を free しない
- attention / scheduler behavior は変えない
- .github/workflows は触らない
- 既存の shadow plan / memory estimate log は壊さない
- 追加するのは server args / config / log schema の準備だけ
```
