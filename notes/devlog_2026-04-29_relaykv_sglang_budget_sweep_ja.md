# Devlog: RelayKV SGLang MVP-1a Budget Sweep Verification

## 日付

2026-04-29

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-memory-mvp0`
- remote: `mine/relaykv-memory-mvp0`

## 目的

RelayKV MVP-1a の metadata-only KV memory estimate logging について、`resident_budget_tokens` を変えたときに `planned_resident_kv_mib` が線形に変化することを確認した。

この確認でも、KV tensor は動かしていない。

## 固定条件

入力:

```text
seq_len = 2535
```

model/profile:

```text
model_arch = Qwen2ForCausalLM
attention_type = gqa
num_layers = 28
num_attention_heads = 12
num_key_value_heads = 2
head_dim = 128
kv_dtype_bytes = 2
kv_bytes_per_token = 28672
```

logical KV:

```text
logical_kv_bytes = 72,683,520
logical_kv_mib = 69.316
```

## Budget sweep 結果

| resident_budget_tokens | planned_resident_tokens | planned_cold_tokens | planned_resident_kv_mib | planned_cold_kv_mib | estimated_resident_ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 512 | 2023 | 14.0 | 55.316 | 0.2019723866 |
| 1024 | 1024 | 1511 | 28.0 | 41.316 | 0.4039447732 |
| 2048 | 2048 | 487 | 56.0 | 13.316 | 0.8078895464 |

## 確認ログ: budget 512

```text
relaykv_shadow_plan_prefill={
  "anchor_pages": [0, 1, 2, 3],
  "attention_type": "gqa",
  "estimated_resident_ratio": 0.2019723865877712,
  "head_dim": 128,
  "kv_bytes_per_token": 28672,
  "kv_dtype_bytes": 2,
  "kv_memory_estimate_reason": "ok",
  "logical_kv_bytes": 72683520,
  "logical_kv_mib": 69.316,
  "mode": "shadow",
  "model_arch": "Qwen2ForCausalLM",
  "num_attention_heads": 12,
  "num_key_value_heads": 2,
  "num_layers": 28,
  "page_size": 1,
  "planned_cold_kv_bytes": 58003456,
  "planned_cold_kv_mib": 55.316,
  "planned_cold_tokens": 2023,
  "planned_resident_kv_bytes": 14680064,
  "planned_resident_kv_mib": 14.0,
  "planned_resident_tokens": 512,
  "reason": "Qwen2.5-style standard full attention is supported for shadow planning",
  "recent_page_range": [2151, 2535],
  "relaykv_enabled": true,
  "relaykv_profile_supported": true,
  "resident_budget_tokens": 512,
  "seq_len": 2535
}
```

## 確認ログ: budget 2048

```text
relaykv_shadow_plan_prefill={
  "anchor_pages": [0, 1, 2, 3],
  "attention_type": "gqa",
  "estimated_resident_ratio": 0.8078895463510848,
  "head_dim": 128,
  "kv_bytes_per_token": 28672,
  "kv_dtype_bytes": 2,
  "kv_memory_estimate_reason": "ok",
  "logical_kv_bytes": 72683520,
  "logical_kv_mib": 69.316,
  "mode": "shadow",
  "model_arch": "Qwen2ForCausalLM",
  "num_attention_heads": 12,
  "num_key_value_heads": 2,
  "num_layers": 28,
  "page_size": 1,
  "planned_cold_kv_bytes": 13963264,
  "planned_cold_kv_mib": 13.316,
  "planned_cold_tokens": 487,
  "planned_resident_kv_bytes": 58720256,
  "planned_resident_kv_mib": 56.0,
  "planned_resident_tokens": 2048,
  "reason": "Qwen2.5-style standard full attention is supported for shadow planning",
  "recent_page_range": [999, 2535],
  "relaykv_enabled": true,
  "relaykv_profile_supported": true,
  "resident_budget_tokens": 2048,
  "seq_len": 2535
}
```

## 判定

期待通り。

```text
budget 512:
  planned_resident_kv_mib = 14.0 MiB

budget 1024:
  planned_resident_kv_mib = 28.0 MiB

budget 2048:
  planned_resident_kv_mib = 56.0 MiB
```

`planned_resident_kv_mib` は `resident_budget_tokens` に対して線形に変化している。

## 意味

MVP-1a では、まだ実際のVRAM削減はしていない。

ただし、RelayKVが将来的に GPU resident KV を `resident_budget_tokens` に制限した場合のメモリ予算効果を、request単位で MiB として観測できるようになった。

この段階で確認できた価値:

```text
logical KV量
resident予定KV量
cold予定KV量
resident/cold ratio
budget変更時の線形性
```

## MVP-1a 完了判定

MVP-1a は完了とみなせる。

完了条件:

```text
[x] KV memory estimate fields are emitted in shadow log
[x] Qwen2.5 GQA metadata is used
[x] bytes_per_token formula is consistent
[x] budget 512 / 1024 / 2048 sweep matches expected values
[x] OpenAI-compatible API still returns 200 OK
[x] No KV tensor movement
[x] No CPU KV copy
[x] No GPU KV free
[x] No attention behavior change
```

## 次の推奨

次はすぐに host backup dry-copy へ進まず、MVP-1a+ として schema / test / guard を固めるのが安全。

推奨順:

1. `memory estimate` の純関数 smoke test を追加
2. shadow log schema を安定化
3. unsupported profile の挙動を明確化
4. その後、MVP-1b host backup dry-copy behind explicit flag

## 次のCodexタスク候補

```text
RelayKV MVP-1a+ として、KV memory estimate の純関数 smoke test と log schema guard を追加してください。

- KV tensor は動かさない
- CPU copy しない
- GPU KV を free しない
- attention / scheduler behavior は変えない
- .github/workflows は触らない
- 既存の shadow log の計算式をテスト可能にする
- Qwen2.5-1.5B相当の fake profile / fake plan で:
  - kv_bytes_per_token = 28672
  - 1024 resident tokens = 28.0 MiB
  - 1511 cold tokens = 41.316 MiB
  を確認する
```
