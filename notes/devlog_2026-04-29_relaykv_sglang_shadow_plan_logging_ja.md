# Devlog: RelayKV SGLang MVP-0.1 Shadow Plan Logging

## 日付

2026-04-29

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-memory-mvp0`
- remote: `mine/relaykv-memory-mvp0`

## 今回の目的

RelayKV MVP-0 の server args / skeleton に続いて、MVP-0.1 として SGLang 実行中に shadow-only の resident/cold plan をログ出力できる最小hookを追加した。

重要方針は以下。

- KV tensor は動かさない
- attention kernel は変更しない
- KV cache の配置、evict、swap、resident化はまだ実装しない
- scheduler の実挙動を変えない
- `--enable-relaykv` かつ `--relaykv-mode shadow` のときだけ動く
- RelayKV無効時は no-op

## コミット

```text
281de0433 Log RelayKV shadow resident plans
```

## 変更ファイル

```text
python/sglang/srt/managers/scheduler.py
```

変更量:

```text
1 file changed, 26 insertions(+)
```

## 実装内容

`scheduler.py` に RelayKV shadow logging hook を追加した。

主な処理は以下。

1. scheduler 側で RelayKV config を参照できるようにする
2. prefill/request 付近で seq_len 相当の情報を使う
3. `make_shadow_plan(...)` を呼ぶ
4. `log_shadow_plan(...)` で resident/cold plan をログ出力する
5. 条件に合わない場合は no-op

期待されるログ内容の例:

```json
{
  "relaykv_enabled": true,
  "mode": "shadow",
  "request_id": "...",
  "seq_len": 4096,
  "page_size": 1,
  "resident_budget_tokens": 1024,
  "planned_resident_tokens": 1024,
  "planned_cold_tokens": 3072,
  "anchor_pages": [0, 1, 2, 3],
  "recent_page_range": [3072, 4096],
  "estimated_resident_ratio": 0.25
}
```

## 確認済み事項

ユーザー側で以下を確認済み。

```bash
git status
```

結果:

```text
On branch relaykv-memory-mvp0
Your branch is up to date with 'mine/relaykv-memory-mvp0'.

nothing to commit, working tree clean
```

push結果:

```text
To https://github.com/rinsakamo/sglang.git
   494a14eff..281de0433  relaykv-memory-mvp0 -> relaykv-memory-mvp0
```

## 現在の状態

現在の作業ブランチは clean。

```text
relaykv-memory-mvp0
```

入っている主なコミット:

```text
281de0433 Log RelayKV shadow resident plans
6bcf3fed2 Add gitignore
066dd2b0e Add RelayKV SGLang devlogs
a7e9e2c2d Document RelayKV SGLang memory management design
```

## 注意点

今回 `scheduler.py` を触っている。

`scheduler.py` は request / batch / seq_len に近い情報を持つため、shadow-only logging hook の場所としては妥当。

ただし、今後も以下は避ける。

- KV cache本体の変更
- `token_to_kv_pool` / `req_to_token_pool` の変更
- eviction / allocation / swap の実装
- request list / batch の実挙動変更
- attention対象の変更

## 次にやること

次は MVP-0.2 として、実サーバー起動時に shadow log が本当に出るかを確認する。

最小確認方針:

1. Qwen2.5系の小さなモデルで SGLang server を起動
2. `--enable-relaykv --relaykv-mode shadow` を付ける
3. 短いOpenAI互換APIリクエストを投げる
4. `relaykv` のshadow plan logが出るか確認
5. 生成結果が通常通り返ることを確認
6. RelayKV無効時はログが出ないことを確認

## 次回用コマンド案

### サーバー起動

```bash
cd ~/work/sglang-relaykv

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

### 別ターミナルから疎通確認

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Say hello in one short sentence."}
    ],
    "max_tokens": 16
  }'
```

### ログ確認

サーバーログに `relaykv` / `shadow` / `planned_resident_tokens` / `planned_cold_tokens` などが出るか確認する。

## 次の実装候補

実ログ確認後、次に進むなら以下。

### MVP-0.2

- shadow logの出力位置と頻度の調整
- request単位でログが読めるように整形
- `seq_len` / `request_id` の信頼性確認

### MVP-0.3

- model profileの浅い判定を追加
- MHA/GQA想定モデルだけ対象にする
- unsupported attention type は shadowでも明示ログにする

### MVP-1

- host backup shadow
- GPU側はまだfreeしない
- resident/cold copy計画とメモリ見積もりだけ検証する
