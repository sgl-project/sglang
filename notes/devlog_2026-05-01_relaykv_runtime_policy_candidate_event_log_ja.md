# Devlog: RelayKV runtime policy candidate event log

## 日付

2026-05-01

## 対象repo

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`

## 目的

RelayKV の `runtime_policy_state` を、既存の shadow runtime log 経路に接続し、`applied_candidate` / `fallback_candidate` の発火を request 単位で観測できるようにする。

今回の目的は **観測ログの追加** であり、実KV退避、attention差し替え、host backup copy、schedulerのスケジューリング挙動変更は行わない。

## 背景

PyTorch側の budget-first 実験では、以下の方針が明確になった。

```text
full KVがVRAMに乗る:
  RelayKV off

VRAM pressureがある:
  RelayKV shadow / applied_candidate

low coverage / high-risk条件:
  partial retrieval強行は危険
  fallback_candidate が必要
```

SGLang側では前段階として、以下を追加済み。

```text
1. RelayKV budget metadata
2. runtime_policy_state
3. policy summary counters
4. applied_candidate / fallback_candidate の集計
```

今回は次の段階として、candidate event を runtime log に接続した。

## 変更ファイル

```text
python/sglang/srt/managers/scheduler.py
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_memory_smoke.py
```

## 実装内容

既存の RelayKV shadow runtime log 経路に、candidate 専用 event log と同一呼び出し内 summary log を接続した。

対象になる state は以下。

```text
applied_candidate
fallback_candidate
```

`off` / `shadow` は通常の plan log / summary 対象であり、candidate event log は出さない。

## runtime log例

`fallback_candidate` の例。

```json
{
  "runtime_policy_state": "fallback_candidate",
  "request_id": null,
  "request_index": 0,
  "layer_idx": null,
  "full_kv_fits": false,
  "budget_pressure": true,
  "available_kv_budget_tokens": 1024,
  "estimated_full_kv_tokens": 2535,
  "resident_budget_tokens": 1024,
  "coverage_ratio": 0.4039447731755424,
  "risk_level": "high",
  "policy_reason": "budget_pressure_high_risk_low_coverage_ratio; budget_policy=anchor_budget_clipped_after_recent_window",
  "runtime_policy_action": "log_only_noop",
  "fallback_candidate_noop_guard": true,
  "applied_candidate_log_only": false,
  "scheduler_policy_noop": true,
  "kv_cache_mutation": false,
  "attention_override": false,
  "host_backup_copy": false
}
```

重要なのは以下。

```text
runtime_policy_action: log_only_noop
fallback_candidate_noop_guard: true
scheduler_policy_noop: true
kv_cache_mutation: false
attention_override: false
host_backup_copy: false
```

`scheduler.py` に触っているが、現時点では scheduler の実行判断やスケジューリング挙動は変更していない。

## layer_idxについて

`layer_idx` は planner / event payload には含める。

ただし、現時点の runtime hook は scheduler request 単位であり、layer hook にはまだ接続していない。  
そのため、現runtime logでは `layer_idx: null` になり得る。

これは現段階では正常。

```text
現段階:
  scheduler request単位のcandidate event log

未接続:
  layer hook単位のpolicy event
```

## summary例

`scripts/relaykv_memory_smoke.py` のsummary。

```json
{
  "policy_state_counts": {
    "applied_candidate": 1,
    "fallback_candidate": 2,
    "off": 2,
    "shadow": 1
  },
  "relaykv_policy_applied_candidate_count": 1,
  "relaykv_policy_fallback_candidate_count": 2
}
```

`scripts/relaykv_budget_sweep.py` のsummaryは前回同様。

```json
{
  "policy_state_counts": {
    "applied_candidate": 31,
    "fallback_candidate": 65,
    "off": 192,
    "shadow": 0
  }
}
```

## 変更していないもの

```text
実KV退避
attention差し替え
host backup copy
schedulerのスケジューリング挙動
KV cache poolの実挙動
partial retrievalの実適用
```

`applied_candidate` も今回は log only。

`fallback_candidate` は no-op guard として明示した。

```text
fallback_candidate:
  runtime_policy_action = log_only_noop
  fallback_candidate_noop_guard = true

applied_candidate:
  applied_candidate_log_only = true
```

## 確認

実行した確認。

```bash
.venv/bin/python -m py_compile   python/sglang/srt/managers/scheduler.py   python/sglang/srt/relaykv/metrics.py   scripts/relaykv_memory_smoke.py
```

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_memory_smoke.py
```

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_budget_sweep.py
```

差分確認。

```bash
git diff --name-status
git diff --name-status | grep '.github/workflows' || true
git diff --check
```

確認結果。

```text
py_compile: pass
relaykv_memory_smoke.py: pass
relaykv_budget_sweep.py: pass
.github/workflows差分: なし
git diff --check: pass
```

実行時に既存環境由来の warning は出た。

```text
Can't initialize NVML
torch version warning
```

ただし、対象チェック自体は通過。

## 判断

今回の変更は、applied mode 前の観測基盤として妥当。

理由。

```text
1. runtime_policy_state を scheduler shadow経路で観測できる
2. applied_candidate / fallback_candidate だけをcandidate eventとして見られる
3. fallback_candidate は明示的にno-op guard
4. applied_candidate もまだlog only
5. 実KV経路、attention、scheduler挙動を変えていない
```

## 現在の到達点

SGLang側の RelayKV policy 観測基盤は以下まで進んだ。

```text
1. budget metadata
2. runtime_policy_state
3. policy summary counters
4. candidate event runtime log
5. fallback no-op guard
```

## 次の候補

次に進む場合の候補は2つ。

### A. applied_candidate発火率の実server smoke

目的。

```text
実server runtime上で applied_candidate / fallback_candidate が
どれくらい発火するかを見る
```

まだKVは動かさない。

見る項目。

```text
request単位
batch条件
budget条件
candidate発火率
fallback率
```

### B. host backup dry-copy guard

目的。

```text
fallback_candidate は必ずno-opのまま
applied_candidate の場合だけ dry-copy guard を通す
```

ただし、まだattentionには接続しない。

推奨順は A → B。

理由。

```text
fallback_candidate が多い場合、
実copyに進む前にpolicy条件を調整すべきだから。
```

## commit候補

```bash
cd ~/work/sglang-relaykv

git status --short
git diff --name-status
git diff --name-status | grep '.github/workflows' || true
git diff --check

git add   python/sglang/srt/managers/scheduler.py   python/sglang/srt/relaykv/metrics.py   scripts/relaykv_memory_smoke.py   notes/devlog_2026-05-01_relaykv_runtime_policy_candidate_event_log_ja.md

git commit -m "Log RelayKV runtime policy candidate events"

git push mine relaykv-host-backup-shadow
```
