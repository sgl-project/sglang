# Devlog: SGLang main切り直し後のRelayKV設計メモ追加

## 日付

2026-04-29

## 対象repo

SGLang fork / SGLang作業ブランチ

## 背景

SGLangの `main` から作業ブランチを切り直したため、実装に入る前にRelayKVのSGLang統合方針を `notes/` に設計メモとして追加する。

今回の方針は、RelayKVをActive-KV Switchではなく、Memory Management中心で実装すること。

## 今回作るもの

```text
notes/relaykv_sglang_memory_management_design_ja.md
```

## 設計メモの要点

RelayKVは、GPU resident KVを予算内に制限し、resident KVが実質的なattention対象になる設計を優先する。

主目的は、OpenClaw / OpenAI互換API backendとしてSGLang with RelayKVを動かし、long-running agent session / 長寿命チャットでVRAMを増やし続けずに運用すること。

## 重要な設計判断

1. RelayKVはMax Context Sizeを直接拡張する技術ではない。
2. RelayKVはfull KV servingが現実的でない長寿命sessionで、GPU resident KV budgetを守る技術。
3. 品質保証はfull KV一致ではなく、task-level service guarantee。
4. full KV fallbackは本命ではない。
5. 実用時はbudget-preserving fallbackを使う。
6. SGLangのRadix / HiCacheは置き換えない。
7. SGLang cacheがshared prefixを救い、RelayKVがthread-local long-tail KVを救う。
8. HiCacheよりHiSparse思想に寄せる。
9. 現行HiSparseはNSA/MLA寄りなので、まずMHA/GQA向けのHiSparse-like resident KV managerを作る。
10. RelayKV Coreはattention type非依存にし、最初のbackendだけMHA/GQAにする。
11. 将来はNSA/MLA、SWA、Qwen3.5系のFull/Gated Attention層へ拡張する。
12. DeltaNet/GDN/linear stateはRelayKV本体の対象外にし、Total VRAM Budget Guardとして別途扱う。
13. Frozen Contextはガワだけ残し、実装はQwen3.5 9B Q4級の実用化が見えてからでよい。

## MVP-0スコープ

最初の実装対象は以下。

```text
- SGLang fork
- Qwen2.5系
- MHA/GQA full attention
- --enable-relaykv
- --relaykv-mode shadow
- KV tensorはまだ動かさない
- resident/cold planのshadow loggingだけ実装する
```

## MVP-0の目的

SGLang上で、requestごとの以下をログ出力する。

```text
- logical KV
- planned resident KV
- planned cold KV
- anchor pages
- recent pages
- estimated resident ratio
```

実際のKV移動やattention差し替えはしない。

## 追加予定ファイル

```text
python/sglang/srt/relaykv/
  __init__.py
  config.py
  profile.py
  budget.py
  planner.py
  metrics.py
```

## 触る候補

```text
python/sglang/srt/server_args.py
python/sglang/srt/model_executor/model_runner.py
python/sglang/srt/managers/scheduler.py
```

## 起動イメージ

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --enable-relaykv \
  --relaykv-mode shadow \
  --relaykv-resident-budget-tokens 1024 \
  --relaykv-recent-window 768 \
  --relaykv-anchor-pages 4
```

## 想定ログ

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

## 完了条件

- 設計メモが `notes/relaykv_sglang_memory_management_design_ja.md` に入っている
- SGLangが `--enable-relaykv` 付きで起動する準備に進める
- MVP-0ではKV tensorを動かさない方針が明文化されている

## 次の作業

1. 設計メモをrepoの `notes/` に配置する。
2. commitする。
3. `--enable-relaykv` / `--relaykv-mode shadow` 等のserver args追加へ進む。
4. `python/sglang/srt/relaykv/` にconfig/planner/metricsのskeletonを作る。
5. Qwen2.5系で通常生成が壊れずshadow logだけ出ることを確認する。

## コミット案

```bash
git add notes/relaykv_sglang_memory_management_design_ja.md

git commit -m "Document RelayKV SGLang memory management design"
```
