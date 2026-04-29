# Devlog: RelayKV SGLang MVP-0.3 No-op Verification and Model Profile Metadata

## 日付

2026-04-29

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-memory-mvp0`
- remote: `mine/relaykv-memory-mvp0`

## 今回の目的

RelayKV MVP-0 系の安全性確認として、以下を行った。

1. RelayKV無効時の no-op 確認
2. `--enable-relaykv --relaykv-mode off` 時の no-op 確認
3. `--enable-relaykv --relaykv-mode shadow` 時の shadow plan log 確認
4. 長文入力で `planned_cold_tokens > 0` になることの確認
5. model/attention profile metadata を shadow log に追加する方向で実装を進めた

## 確認結果

### RelayKV disabled

起動時に `--enable-relaykv` を付けない状態で、長文リクエストを送信。

ログ:

```text
Prefill batch, #new-seq: 1, #new-token: 2048, #cached-token: 0, token usage: 0.02, #running-req: 0, #queue-req: 0, #pending-token: 487, cuda graph: True
Prefill batch, #new-seq: 1, #new-token: 487, #cached-token: 0, token usage: 0.02, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True
INFO: 127.0.0.1 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

結果:

```text
200 OK
relaykv_shadow_plan_prefill は出ない
no-op OK
```

### RelayKV enabled + mode off

起動時に `--enable-relaykv --relaykv-mode off` を指定して、同じ長文リクエストを送信。

ログ:

```text
Prefill batch, #new-seq: 1, #new-token: 2048, #cached-token: 0, token usage: 0.02, #running-req: 0, #queue-req: 0, #pending-token: 487, cuda graph: True
Prefill batch, #new-seq: 1, #new-token: 487, #cached-token: 0, token usage: 0.02, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True
INFO: 127.0.0.1 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

結果:

```text
200 OK
relaykv_shadow_plan_prefill は出ない
no-op OK
```

### RelayKV enabled + mode shadow

短文・長文ともに `relaykv_shadow_plan_prefill` が出ることを確認。

短文:

```text
seq_len: 36
resident_budget_tokens: 1024
planned_resident_tokens: 36
planned_cold_tokens: 0
estimated_resident_ratio: 1.0
```

長文:

```text
seq_len: 2535
resident_budget_tokens: 1024
planned_resident_tokens: 1024
planned_cold_tokens: 1511
estimated_resident_ratio: 0.4039447731755424
recent_page_range: [1767, 2535]
anchor_pages: [0, 1, 2, 3]
```

結果:

```text
short input:
  all resident
  planned_cold_tokens = 0

long input:
  resident budget respected
  planned_cold_tokens > 0
```

## 実装メモ

MVP-0.3 では、shadow planning の前段に model/attention profile の浅い判定を追加する方針。

想定する追加メタデータ:

```text
model_arch
attention_type
relaykv_profile_supported
reason
```

初期方針:

- Qwen2.5系の standard MHA/GQA full attention を supported として扱う
- 判定に必要な情報が不足する場合は、unknown-but-shadow-ok または conservative warning
- unsupported attention type の場合は、KVには触らず warning log のみにする
- shadow log の可読性を上げるため、profile情報を含める

## 安全条件

今後も以下を守る。

- `.github/workflows` は触らない
- KV tensor は動かさない
- attention kernel は変更しない
- KV cache の中身・配置・evict・swap は変更しない
- scheduler の実挙動を変えない
- RelayKV無効時/mode off時は完全no-op

## 次にやること

次は、profile metadata 実装後の実サーバー確認。

確認対象:

1. `--enable-relaykv --relaykv-mode shadow` で profile付きshadow logが出る
2. profile項目がログに含まれる
3. Qwen2.5系が supported として扱われる
4. RelayKV無効時/mode off時は引き続き log が出ない
5. 通常生成が 200 OK のまま

## 確認コマンド案

### compile

```bash
cd ~/work/sglang-relaykv
source .venv/bin/activate

python -m compileall python/sglang/srt/relaykv python/sglang/srt/managers/scheduler.py
```

### help

```bash
PYTHONPATH=python python -m sglang.launch_server --help | grep -A 30 -i relaykv
```

### shadow server

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
  --relaykv-log-interval 1
```

### request

```bash
curl -sS -i http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/relaykv_long_payload.json
```

### expected log fields

```text
relaykv_shadow_plan_prefill
planned_resident_tokens
planned_cold_tokens
estimated_resident_ratio
model_arch
attention_type
relaykv_profile_supported
reason
```

## コミット後確認

```bash
git status --short
git log --oneline --decorate --max-count=8
git push
```
