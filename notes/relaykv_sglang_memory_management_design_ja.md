# RelayKV SGLang Memory Management Design

## 日付

2026-04-29

## 目的

RelayKVをSGLangに統合するにあたり、まずは **Memory Management中心** の設計として整理する。

本メモは、SGLang `main` から切り直した新ブランチで、最初に `notes/` へ追加する設計メモである。

## 基本方針

RelayKVは、Active-KV Switchを主機能にするのではなく、GPU resident KV を予算内に制限する **decode-time KV memory manager** として実装する。

つまり、attention対象だけを別途switchするのではなく、GPU上にresidentなKVを制限し、resident KVが実質的なattention対象になる設計を優先する。

```text
RelayKV Core
= GPU resident KV budget manager
+ CPU cold KV tier
+ selected page/block swap-in
+ budget-preserving fallback
```

## RelayKVの位置づけ

RelayKVは、モデルのhard max contextを直接拡張する技術ではない。

RelayKVが狙うのは、full KV serving が現実的でない長寿命agent sessionにおいて、logical contextを保持しつつ、GPU resident KVを予算内に収めることである。

品質保証は full KV一致ではなく、task-level service guarantee とする。

## 想定する実用構成

OpenClaw / OpenAI互換API backend として SGLang with RelayKV を動かす。

```text
OpenClaw / agent frontend
  ↓ OpenAI-compatible API
SGLang server
  ↓
SGLang with RelayKV
  ↓
local model
```

OpenClaw側は、通常のOpenAI互換API backendとしてSGLangを見る。
RelayKVはSGLang内部のruntime memory policyとして隠蔽する。

## SGLang cacheとの関係

SGLangのRadix / HiCacheは置き換えない。

役割分担は以下とする。

```text
SGLang Radix / HiCache:
  shared prefix reuse
  prefix hit / storage hit
  tree node管理

RelayKV:
  thread-local tail KV の resident / cold 管理
```

複数スレッド・複数sessionでは、SGLang cacheが共有prefixを救い、RelayKVが共有できないthread-local long-tail KVを救う。

## HiCache / HiSparseとの関係

通常HiCacheはprefix reuseと階層cache管理が中心であり、RelayKVが欲しいGPU resident KV制御とは粒度が異なる。

HiSparseは以下の点でRelayKVに近い。

- logical indexとdevice hot buffer indexを分ける
- host-side KV backupを持つ
- selected token/pageをGPU hot bufferへswap-inする
- `top_k`, `device_buffer_size`, `host_to_device_ratio` のような設定を持つ

ただし現行HiSparseはNSA/MLA寄りである。
RelayKVでは、まずMHA/GQA向けにHiSparse-likeなresident KV managerを作る。

## Attention type対応

RelayKV Coreはattention type非依存の抽象にする。
最初のbackendだけMHA/GQAにする。

想定構成:

```text
python/sglang/srt/relaykv/
  __init__.py
  config.py
  profile.py
  budget.py
  planner.py
  metrics.py
  backends/
    base.py
    mha_gqa.py
    nsa_mla.py
    swa.py
    unsupported.py
```

初期対応:

```text
対応:
  MHA / GQA standard paged full attention

後回し:
  NSA / MLA
  SWA
  Mamba / DeltaNet / GDN / linear state
  multimodal特殊token
```

## Qwen3.5 9B Q4への考え方

実用価値の基準は、Qwen3.5 9B Q4級のローカル実用モデルで動かせることである。

ただしQwen3.5系はDeltaNet/GDN + Full/Gated Attentionのhybrid architectureなので、RelayKV単体で全VRAMを保証するとは言わない。

正確には以下の分担にする。

```text
RelayKV:
  Full/Gated Attention層のresident KVを保証

SGLang hybrid/mamba state budget:
  DeltaNet/GDN/linear stateを管理

Total VRAM Budget Guard:
  weights
  + attention KV
  + DeltaNet/GDN state
  + workspace
  + safety margin
```

初期実装はQwen2.5系MHA/GQAで進め、Qwen3.5 9B Q4は後段のfeasibility targetにする。

## Fallback方針

full KV fallbackは本命ではない。

RelayKVが価値を出す領域では、そもそもfull KV baselineがVRAMに乗らない可能性が高い。
そのため、実用時はbudget-preserving fallbackを採用する。

例:

```text
normal:
  recent 1024
  anchor 512
  retrieved 2048
  transient 512

safe:
  recent 2048
  anchor 512
  retrieved 1024
  transient 512

panic:
  recent 3072
  anchor 512
  retrieved 512
  transient 0
```

## Frozen Context

Frozen Contextは有望だが、RelayKV MVPでは実装しない。

ガワまたは設計メモだけ残す。
実装はQwen3.5 9B Q4級で長寿命sessionの実用化が見えてからでよい。

役割:

```text
Cold KV:
  同一session内で復元可能なKV

Frozen Context:
  RAM圧迫時・session間継続用の意味圧縮
```

## MVP-0: shadow planning

最初のMVPはshadow planningのみとする。

やること:

```text
- --enable-relaykv
- --relaykv-mode shadow
- --relaykv-resident-budget-tokens
- --relaykv-recent-window
- --relaykv-anchor-pages
- resident/cold planのログ出力
- actual KV tensorは動かさない
```

想定ファイル:

```text
python/sglang/srt/relaykv/
  __init__.py
  config.py
  profile.py
  budget.py
  planner.py
  metrics.py
```

SGLang hook候補:

```text
python/sglang/srt/server_args.py
python/sglang/srt/model_executor/model_runner.py
python/sglang/srt/managers/scheduler.py
```

起動イメージ:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --enable-relaykv \
  --relaykv-mode shadow \
  --relaykv-resident-budget-tokens 1024 \
  --relaykv-recent-window 768 \
  --relaykv-anchor-pages 4
```

出すログ例:

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

## MVP-0完了条件

- SGLangが `--enable-relaykv` 付きで起動する
- Qwen2.5系で通常生成が壊れない
- shadow logが出る
- logical / resident / cold tokensが計算できる
- KV tensorはまだ触らない

## 以後のPhase

### Phase 1: MVP-0 shadow planning

resident/cold計画を出すのみ。

### Phase 2: profile / backend abstraction

RelayKVBackend抽象を作り、MHA/GQA backendだけ実装する。

### Phase 3: host backup shadow

prefill後のresident対象外KVをCPUにcopyする。ただしGPU側はfreeしない。

### Phase 4: resident mapping prototype

logical indexとresident GPU indexを分ける。

### Phase 5: apply-budget single request

resident KVだけでdecodeできるかをQwen2.5系で試す。

### Phase 6: Qwen2.5評価

long table lookup / structured lookup / long prose QA / multi-turn threadで評価する。

### Phase 7: Qwen3.5 9B Q4 feasibility

Full/Gated Attention層のKV削減余地、DeltaNet/GDN state、VRAM breakdownを計測する。

## Non-goals

- Radix cacheを置き換えない
- HiCacheを置き換えない
- attention kernelを最初から改造しない
- full KV一致を保証しない
- hard max contextを直接拡張しない
- DeltaNet/GDN stateをRelayKV本体では管理しない
- multimodal初期対応しない
- Frozen ContextをMVPで実装しない

## 次にやること

1. この設計メモをrepoの `notes/relaykv_sglang_memory_management_design_ja.md` に追加する。
2. SGLang forkで `relaykv-memory-mvp0` ブランチを作る。
3. `--enable-relaykv` / `--relaykv-mode shadow` 等のserver argsを追加する。
4. `python/sglang/srt/relaykv/` にMVP-0 skeletonを作る。
5. Qwen2.5系で通常生成を壊さずshadow logだけ出す。

## 最初に絶対守ること

MVP-0ではKV tensorを絶対に動かさない。

まずはSGLang上でRelayKVのresident/cold計画だけを観測する。
