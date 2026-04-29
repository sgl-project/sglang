# Devlog: RelayKV SGLang MVP-1a KV Memory Estimate Logging

## 日付

2026-04-29

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-memory-mvp0`
- remote: `mine/relaykv-memory-mvp0`

## 今回の目的

RelayKV MVP-1a として、shadow plan に対して metadata-only の KV memory estimate logging を追加した。

この段階では、KV tensor はまだ動かさない。

## 守った制約

- KV tensor は動かさない
- CPU copy はしない
- GPU KV は free しない
- attention kernel は変更しない
- scheduler の実挙動は変更しない
- 通常生成結果を変えない
- RelayKV disabled / mode off では no-op
- `.github/workflows` は触らない

## 実装内容

既存の `relaykv_shadow_plan_prefill` ログに、KV memory estimate のメタデータを追加した。

追加された主な項目:

```text
num_layers
head_dim
kv_dtype_bytes
kv_bytes_per_token
logical_kv_bytes
planned_resident_kv_bytes
planned_cold_kv_bytes
logical_kv_mib
planned_resident_kv_mib
planned_cold_kv_mib
kv_memory_estimate_reason
```

## 確認ログ

長文リクエストで以下を確認。

```text
relaykv_shadow_plan_prefill={
  "anchor_pages": [0, 1, 2, 3],
  "attention_type": "gqa",
  "estimated_resident_ratio": 0.4039447731755424,
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
  "planned_cold_kv_bytes": 43323392,
  "planned_cold_kv_mib": 41.316,
  "planned_cold_tokens": 1511,
  "planned_resident_kv_bytes": 29360128,
  "planned_resident_kv_mib": 28.0,
  "planned_resident_tokens": 1024,
  "reason": "Qwen2.5-style standard full attention is supported for shadow planning",
  "recent_page_range": [1767, 2535],
  "relaykv_enabled": true,
  "relaykv_profile_supported": true,
  "request_id": "b9e67bdc192149f8b89623d640e77c99",
  "resident_budget_tokens": 1024,
  "seq_len": 2535
}
```

OpenAI互換APIも成功。

```text
POST /v1/chat/completions HTTP/1.1" 200 OK
```

## 式の確認

Qwen2.5-1.5B-Instruct 相当の observed metadata:

```text
num_layers = 28
num_key_value_heads = 2
head_dim = 128
kv_dtype_bytes = 2
```

KV bytes per token:

```text
kv_bytes_per_token
= num_layers * 2(K/V) * num_key_value_heads * head_dim * kv_dtype_bytes
= 28 * 2 * 2 * 128 * 2
= 28672 bytes/token
```

長文リクエスト:

```text
seq_len = 2535
planned_resident_tokens = 1024
planned_cold_tokens = 1511
```

memory estimate:

```text
logical_kv_bytes = 2535 * 28672 = 72,683,520 bytes = 69.316 MiB
planned_resident_kv_bytes = 1024 * 28672 = 29,360,128 bytes = 28.0 MiB
planned_cold_kv_bytes = 1511 * 28672 = 43,323,392 bytes = 41.316 MiB
```

## 現時点の意味

この段階では、実際のVRAM削減はまだ行っていない。

ただし、RelayKVが将来的に GPU resident KV を budget 内に抑えた場合に、どの程度の logical KV を resident/cold に分ける設計になるかを MiB 単位で観測できるようになった。

今回の例では:

```text
logical KV: 69.316 MiB
planned resident KV: 28.0 MiB
planned cold KV: 41.316 MiB
```

つまり、shadow plan 上は logical KV のうち約 40.4% を resident、約 59.6% を cold として扱う計画になっている。

## 次にやること

MVP-1a の締めとして、budget sweep を行う。

目的:

```text
resident_budget_tokens を変えたときに planned_resident_kv_mib が線形に変わることを確認する
```

確認対象:

```text
budget 512:
  planned_resident_kv_mib ≒ 14.0 MiB

budget 1024:
  planned_resident_kv_mib = 28.0 MiB

budget 2048:
  planned_resident_kv_mib ≒ 56.0 MiB
```

これが確認できれば、MVP-1a はかなりきれいに完了とみなせる。

## 次回コマンド案

### budget 512

```bash
PYTHONPATH=python python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --enable-relaykv \
  --relaykv-mode shadow \
  --relaykv-resident-budget-tokens 512 \
  --relaykv-recent-window 384 \
  --relaykv-anchor-pages 4 \
  --relaykv-log-interval 1
```

### budget 2048

```bash
PYTHONPATH=python python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --enable-relaykv \
  --relaykv-mode shadow \
  --relaykv-resident-budget-tokens 2048 \
  --relaykv-recent-window 1536 \
  --relaykv-anchor-pages 4 \
  --relaykv-log-interval 1
```

### request

```bash
curl -sS -i http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/relaykv_long_payload.json
```

## 次の実装候補

Budget sweep 後、次に進むなら MVP-1b ではなく、まず MVP-1a+ として以下を固めるのが安全。

```text
- memory estimate をテスト可能な純関数として整理
- unit-like smoke test を追加
- profile unsupported時の挙動を明確化
- log schema を安定化
```

その後に MVP-1b:

```text
host backup dry-copy behind explicit flag
```

ただし MVP-1b でも、最初は GPU KV を free しない。
