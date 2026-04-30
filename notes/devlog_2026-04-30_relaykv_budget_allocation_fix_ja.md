# Devlog: RelayKV SGLang Budget Planner Allocation Fix

## 日付

2026-04-30

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`

## 今回の目的

RelayKV を VRAM-constrained KV working set manager として扱うため、budget planner metadata の allocation fields を修正した。

前回ログでは以下が不自然だった。

```text
anchor_blocks: 4
anchor_budget_tokens: 4
retrieval_budget_tokens: 8
```

これは SGLang 内部の `page_size=1` と RelayKV 論理 block size を混同していた可能性があった。

今回、RelayKV budget 用の `budget_block_size=128` を導入し、anchor / retrieval budget を token budget として明確化した。

## 確認条件

```text
kv_working_budget_tokens: 1024
recent_window_tokens: 768
anchor_blocks: 4
budget_block_size: 128
retrieval_top_k_requested: 8
seq_len: 2535
```

## 確認ログ

```text
kv_working_budget_source: explicit_working_budget_tokens
kv_working_budget_tokens: 1024
resident_budget_tokens: 1024

recent_window_tokens: 768
anchor_blocks: 4
budget_block_size: 128
anchor_budget_tokens: 256

retrieval_budget_tokens: 0
retrieval_block_budget: 0
retrieval_top_k_requested: 8
retrieval_top_k_effective: 0

budget_overflow: true
budget_policy_reason: anchor_budget_clipped_after_recent_window
```

## 解釈

計算は期待通り。

```text
total = 1024
recent = 768
requested_anchor = 4 * 128 = 512
remaining_after_recent = 256
anchor_budget_tokens = min(512, 256) = 256
retrieval_budget_tokens = 1024 - 768 - 256 = 0
retrieval_block_budget = 0
retrieval_top_k_effective = 0
```

したがって、1024 token budget の小さい制約下では:

```text
recent full window:
  768 tokens

anchor:
  requested 512 tokens
  clipped to 256 tokens

retrieval:
  0 tokens
```

となる。

## resident/cold planning

```text
seq_len: 2535
planned_resident_tokens: 1024
planned_cold_tokens: 1511
estimated_resident_ratio: 0.4039447731755424
```

working KV budget が 1024 tokens のため、2535 tokens のうち 1024 tokens を resident、1511 tokens を cold として扱う計画になっている。

これは budget-first の挙動として正しい。

## API確認

```text
POST /v1/chat/completions HTTP/1.1" 200 OK
```

通常生成経路は壊れていない。

## 現在の評価

```text
budget metadata output:
  OK

explicit token budget handling:
  OK

budget_block_size:
  OK

anchor budget allocation:
  OK

retrieval budget allocation:
  OK

small budget overflow / clipping reason:
  OK

KV tensor / CPU copy / GPU free / attention変更:
  なし
```

## 次に確認すること

次は 512MiB budget case でも allocation が期待通りになるか確認する。

期待値:

```text
kv_bytes_per_token = 28672
available_kv_budget_mib = 512
kv_working_budget_tokens ≈ 18724

recent_window_tokens = 768
anchor_blocks = 4
budget_block_size = 128
anchor_budget_tokens = 512
retrieval_budget_tokens = 18724 - 768 - 512 = 17444
retrieval_block_budget = floor(17444 / 128) = 136
retrieval_top_k_requested = 8
retrieval_top_k_effective = 8
budget_overflow = false
```

## 次のタスク候補

1. 512MiB / 1024MiB / 2048MiB の実サーバーbudget log確認
2. budget smoke test を表形式で出す
3. budget sweep script を追加する
4. その後、PyTorch側RelayKV評価にも同じbudget fieldsを導入する
