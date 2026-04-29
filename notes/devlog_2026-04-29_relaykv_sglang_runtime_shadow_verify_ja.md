# Devlog: RelayKV SGLang MVP-0.2 Shadow Log Runtime Verification

## 日付

2026-04-29

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-memory-mvp0`
- remote: `mine/relaykv-memory-mvp0`

## 今回の目的

RelayKV MVP-0.1 で追加した shadow-only resident/cold plan logging が、実際の SGLang server 起動後に出ることを確認した。

## 確認したこと

### 1. 短文リクエスト

短文入力で `relaykv_shadow_plan_prefill` が出た。

ログ:

```text
relaykv_shadow_plan_prefill={
  "anchor_pages": [0, 1, 2, 3],
  "estimated_resident_ratio": 1.0,
  "mode": "shadow",
  "page_size": 1,
  "planned_cold_tokens": 0,
  "planned_resident_tokens": 36,
  "recent_page_range": [0, 36],
  "relaykv_enabled": true,
  "request_id": "ebdf9405c1b946c3886b22dc449b8eb1",
  "resident_budget_tokens": 1024,
  "seq_len": 36
}
```

意味:

```text
logical context: 36 tokens
resident budget: 1024 tokens
planned resident: 36 tokens
planned cold: 0 tokens
resident ratio: 1.0
```

短文では全tokenがresident budget内に収まるため、cold tokens は 0 で正常。

### 2. 長文リクエスト

長文入力で `planned_cold_tokens > 0` になることを確認した。

ログ:

```text
relaykv_shadow_plan_prefill={
  "anchor_pages": [0, 1, 2, 3],
  "estimated_resident_ratio": 0.4039447731755424,
  "mode": "shadow",
  "page_size": 1,
  "planned_cold_tokens": 1511,
  "planned_resident_tokens": 1024,
  "recent_page_range": [1767, 2535],
  "relaykv_enabled": true,
  "request_id": "037bf0a80e4a4730b8bb147c0885eab1",
  "resident_budget_tokens": 1024,
  "seq_len": 2535
}
```

意味:

```text
logical context: 2535 tokens
resident budget: 1024 tokens
planned resident: 1024 tokens
planned cold: 1511 tokens
resident ratio: 約40.4%
```

長文では logical context が resident budget を超えるため、cold tokens が発生して正常。

### 3. OpenAI互換APIの疎通

`POST /v1/chat/completions` が `200 OK` を返すことを確認した。

```text
INFO: 127.0.0.1 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

通常生成は壊れていない。

## 途中で発生した問題

### 問題

長文payload作成時に、別ターミナルで `python` コマンドが存在せず、payloadファイルが作成できなかった。

```text
Command 'python' not found
```

その結果、curlで空bodyが送られ、SGLang側で以下の400が出た。

```text
Field required
```

### 対処

別ターミナルでも venv を有効化する方針にした。

```bash
cd ~/work/sglang-relaykv
source .venv/bin/activate
```

または `.venv` の Python を直接使う。

```bash
/home/rinsa/work/sglang-relaykv/.venv/bin/python
```

## 観察事項

長文入力では、同じ request_id に対して `relaykv_shadow_plan_prefill` が複数回出た。

理由は、SGLang の chunked prefill / pending token 処理で、1 request の prefill が複数batchに分かれるためと考えられる。

ログ例:

```text
Prefill batch, #new-token: 2048, #pending-token: 463
Prefill batch, #new-token: 463, #pending-token: 0
```

このため、同一 request_id の shadow plan log は初回だけ出すように重複抑制を追加した。

## 現在の状態

ユーザーにより、重複抑制の変更はコミット済み。

## 次にやること

次は MVP-0.3 として、RelayKV shadow planning の安全性を上げる。

候補:

1. RelayKV無効時の no-op 確認
2. `--relaykv-mode off` 時の no-op 確認
3. model profile の浅い判定を追加
4. MHA/GQA対象モデルのみ shadow対象として明示
5. unsupported attention type は warning log のみにする

次に進むなら、最優先は no-op 確認。

理由:

- `--enable-relaykv` なしでは完全に通常SGLangと同じ挙動にしたい
- `--relaykv-mode off` でもログや副作用が出ないことを確認したい
- ここを固めてから profile 判定へ進む方が安全

## 次回確認コマンド案

### RelayKV有効・shadow

```bash
PYTHONPATH=python ./.venv/bin/python -m sglang.launch_server \
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

### RelayKV無効

```bash
PYTHONPATH=python ./.venv/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 30000
```

### RelayKV mode off

```bash
PYTHONPATH=python ./.venv/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --enable-relaykv \
  --relaykv-mode off
```

### リクエスト

```bash
curl -sS -i http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/relaykv_long_payload.json
```

期待:

```text
shadow:
  relaykv_shadow_plan_prefill が出る

disabled:
  relaykv_shadow_plan_prefill が出ない

mode off:
  relaykv_shadow_plan_prefill が出ない
```
