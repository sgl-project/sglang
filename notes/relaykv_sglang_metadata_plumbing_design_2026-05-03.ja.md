# RelayKV SGLang Metadata Plumbing Design Memo

## 日付確認

この設計メモは JST 基準で `2026-05-03` の作業記録・設計整理として作成する。

Branch: `relaykv-host-backup-shadow`  
Repo: `~/work/sglang-relaykv`

## 0. このメモの目的

このメモは、RelayKV の大枠設計と、現在の SGLang 実装フェーズを接続するための整理である。

今回の整理では、添付の `relaykv_ideas_summary_2026-05-03.ja.md` の内容を前提に、現在の実装を以下の位置づけに固定する。

```text
RelayKV:
  低VRAM環境で decode-time KV working set を固定残VRAM予算内に収める
  VRAM-aware memory management layer

現在のSGLang実装:
  実KV操作前の read-only metadata plumbing
```

まだ以下には進まない。

```text
KV pool read
KV snapshot
host backup copy実行
attention接続
scheduler decision変更
runtime writeback
semantic tree
KV-BVH
precision promotion
```

## 1. RelayKV の基本定義

RelayKV は、単なる KV cache 削減アルゴリズムではない。

RelayKV は、量子化 dense model を低VRAM GPU に載せた後、残った少ない VRAM 内で decode-time KV working set を制御する memory management layer である。

目的は以下。

```text
KVを雑に削る:
  NG

残VRAM内でattention対象にするKVを制御する:
  OK
```

一文で表すと以下。

```text
RelayKV is a VRAM-aware decode-time KV working-set manager
that combines layer-wise structured KV selection,
Anchor KV protection,
cold KV retrieval,
and optional precision promotion
under a fixed residual VRAM budget.
```

日本語では以下。

```text
RelayKVは、量子化dense modelを低VRAM GPUに載せた後、
残ったVRAM内で、RECENT / ANCHOR / RETRIEVED KVを制御する
decode-time KV working-set managerである。
```

## 2. 実用ターゲット

当面の主要ターゲット:

```text
RTX 3060 12GB 単体
dense系の量子化モデル
Qwen3.5-9B Q4 クラス
将来: Qwen3.6 9B〜12B dense Q4 クラス
長期: Qwen3.6 27B dense + 強力な重み/KV量子化
```

前提:

```text
ローカルLLMユーザーは、
VRAMに収まる最大級の量子化モデルを載せ、
残ったVRAMで context / KV cache を伸ばそうとする。
```

そのため、RelayKV はモデルロード後の残VRAMを前提にした working KV budget manager として設計する。

## 3. 正式用語: Anchor KV

RelayKV では **Anchor KV** を正式名称にする。

`Sink KV` は独立した主要分類にはしない。

必要な場合のみ、Anchor KV の下位概念として以下を使う。

```text
Sink-like Anchor
```

理由:

```text
RelayKVで守りたいもの:
  attention sink token だけではない

守りたいもの:
  system prompt
  user instruction
  output format
  table headers
  JSON schema
  code definitions
  section headings
  initial tokens
```

整理:

```text
Anchor KV
├─ Semantic Anchor
│  ├─ system prompt
│  ├─ user instruction
│  ├─ output format
│  └─ task constraints
│
├─ Structural Anchor
│  ├─ table headers
│  ├─ JSON schema
│  ├─ code definitions
│  └─ section headings
│
└─ Sink-like Anchor
   ├─ BOS / initial tokens
   ├─ attention sink tokens
   └─ learned/synthetic sink tokens
```

短期実装では以下で十分。

```text
KVClass.ANCHOR
```

将来、必要なら以下を追加する。

```text
anchor_type = "semantic" | "structural" | "sink_like"
```

## 4. KV分類

### 4.1 短期の論理分類

短期の論理分類:

```text
KVClass.RECENT
KVClass.ANCHOR
KVClass.TRANSIENT
KVClass.RETRIEVED
KVClass.COLD_CANDIDATE
```

ただし、短期の実装classとして `TRANSIENT` を必ず物理化する必要はない。

短期実装では以下でよい。

```text
実装class:
  RECENT
  ANCHOR
  RETRIEVED
  COLD_CANDIDATE

TRANSIENT:
  decode-state / log / metadata concept
```

### 4.2 RECENT

直近トークンから作られたKV。

役割:

```text
文体の連続性
直前の係り受け
生成中の局所整合性
次トークン予測の安定性
```

扱い:

```text
削減対象ではなく保護対象
GPU resident 優先
full precision 優先
常に working attention set に含める
```

### 4.3 ANCHOR

長距離にわたって生成を安定させる基準点となるKV。

含むもの:

```text
system prompt
冒頭指示
出力フォーマット
表の列名
JSON schema
コード定義
section heading
attention sink 的な初期token
```

扱い:

```text
retrieval top-k とは別枠で保持
B_anchor として独立予算を持つ
retrieved KV に押し出されない
```

### 4.4 TRANSIENT

decode中に新しく発生するKV。

役割:

```text
現在生成中の文脈を保つ
recent window に合流する前の一時KV
next-step apply / replacement / gating の観測対象
```

短期扱い:

```text
物理classにしなくてよい
decode-state / log / metadata concept として扱う
生成後は recent window に合流
recent window外に出たら cold candidate へ移す
```

状態遷移:

```text
new decode KV
→ transient
→ recent
→ cold candidate
```

### 4.5 RETRIEVED

cold側にあるKVのうち、現在のdecode stepで必要と判断され、working KVへ戻されたKV。

役割:

```text
長距離参照
needle情報の復元
表・コード・文書内の該当箇所参照
RAG的な内部記憶参照
```

扱い:

```text
RelayKV selection の主対象
B_retrieval の範囲内で選ぶ
block単位で扱う
将来的には layer/head-aware にする
```

流れ:

```text
Cold KV blocks
→ score
→ top-k select
→ retrieved KV
→ working KV
```

### 4.6 COLD_CANDIDATE

論理的には保持されているが、現在のdecode stepでは attention 対象ではないKV。

役割:

```text
full KV の論理的保存先
retrieval候補
CPU / compressed store / HiCache などに置く対象
```

扱い:

```text
GPU attention対象ではない
必要になったら retrieved KV へ昇格
完全dropよりも、まずは cold candidate として保持する方が安全
```

### 4.7 中長期分類

中期で追加:

```text
KVClass.HEAVY_HITTER
KVClass.COMPRESSED_COLD
```

長期で追加:

```text
KVClass.HIERARCHICAL_REMOTE
KVClass.DROPPED
```

HEAVY_HITTER:

```text
何度もretrievedされるblock
累積attentionやretrieval頻度が高いblock
anchor-likeに準常駐させる候補
```

COMPRESSED_COLD:

```text
TurboQuant / PolarQuant / INT4 / FP8 などで圧縮保存されたcold KV
RelayKV本体は圧縮形式を直接知らない
ColdKVStoreがblock_idから必要なKVを返す
```

HIERARCHICAL_REMOTE:

```text
GPU / CPU / SSD / remote storage など階層的に置かれるKV
HiCache / HiRadixTree / CPU offload / L3 storage と接続する長期対象
```

DROPPED:

```text
metadata only化や完全dropされたKV
初期実装では原則ここまで進まない
```

## 5. Working KV Budget Model

RelayKVでは、decode時にattention対象になるKVを working KV と呼ぶ。

予算モデル:

```text
B_total_working_kv =
  B_recent
+ B_anchor
+ B_transient
+ B_retrieval
```

意味:

```text
B_recent:
  品質下限を守る局所文脈枠

B_anchor:
  system / schema / format / sink-like token を守る枠

B_transient:
  decode中の一時KV安全枠

B_retrieval:
  cold candidate から戻すKV枠
```

将来的には layer ごとに拡張する。

```text
B_recent[layer]
B_anchor[layer]
B_retrieval[layer]
```

さらに GQA / kv_head_group を考慮して拡張する。

```text
B_retrieval[layer][kv_head_group]
```

## 6. Layer × Block Mask 仮説

各層で必要なKV blockは一様ではない。

`layer × block` のmaskとして見る。

```text
M[layer, block] ∈ {0, 1}
```

仮説:

```text
浅い層:
  広く必要

中間層:
  絞れる可能性が高い

深い層:
  少数で足りる場合もある
  ただしタスクによってfragileになる可能性あり
```

予算方針候補:

```text
uniform:
  全層同じbudget

pyramid:
  浅い層を厚く、深い層を薄く

hourglass:
  浅い層と深い層を厚く、中間層を薄く

hard_layer_heavy:
  実測で壊れやすい層を厚く

adaptive:
  実測fragilityから動的に補正
```

現時点では、これは実装ではなく設計・ログ項目として入れる。

## 7. ゲームエンジン的カリング / KV-BVH / Semantic Tree

RelayKVは、3Dゲームの visibility culling / LOD / streaming と似ている。

対応関係:

```text
Frustum culling:
  query方向に明らかに関係ないblockを除外

Occlusion culling:
  anchor/recentで説明済みのblockを低優先化

LOD / Mipmapping:
  chunk summary → block summary → full KV の粗密探索

BVH:
  KV blockを階層index化

Temporal coherence:
  直前stepのretrieved blockを再利用

Early-Z:
  score上限で勝てないblockを早期棄却

Streaming:
  必要KVだけCPU/hostからGPUへ上げる
```

これらは有望だが、現フェーズでは実装しない。

扱い:

```text
Phase 6 optimization / Phase 7 research expansion
```

Semantic Tree は RadixTree とは別物として整理する。

```text
RadixTree:
  prefix reuseの木

Semantic Tree:
  query / key / block summary の意味的近さの木

RelayKV:
  Semantic Treeを使って COLD_CANDIDATE から RETRIEVED 候補を絞る
```

Semantic Tree は RECENT や ANCHOR の代わりではない。

```text
Semantic Tree:
  COLD_CANDIDATE から RETRIEVED を探す補助index
```

## 8. Precision Promotion / 選択的詳細化

将来案:

```text
COLD_CANDIDATE:
  low-bit / compressed / low-detail

RETRIEVED:
  selected blockをdequantizeまたはworking precisionへ昇格

RECENT / ANCHOR:
  high precision寄りで保護

HEAVY_HITTER:
  頻繁に使うblockを高精度準常駐
```

注意:

```text
2bit KVをfp16へdequantizeしても失われた情報は戻らない。
真の詳細化には、高精度backup copyかresidual / error correction情報が必要。
```

RelayKVでは、将来的に以下を分けて扱う。

```text
selection:
  どのblockをworking setに入れるか

precision selection:
  どの精度でworking setに入れるか
```

現フェーズでは実装しない。

## 9. SGLang RadixTree / HiCache との統合方針

RelayKV は SGLang の RadixTree / RadixAttention を置き換えない。

役割分担:

```text
RadixTree / RadixAttention:
  prefix cache manager

HiCache / HiRadixTree:
  KV residency manager

RelayKV:
  decode-time working-set selector
  residual VRAM budget controller
```

短期方針:

```text
RadixTreeを変更しない
KV pool readしない
attention接続しない
scheduler decision変更しない
runtime writebackしない
ForwardBatch read-only metadata observationを続ける
request_id / req_pool_idx / seq_len と host backup copy candidate summary をjoinできる形にする
```

中期方針:

```text
RadixTree / HiRadixTree の node span / token span / residency metadata をread-only参照
RelayKV block_id と RadixTree node span を対応づける
```

長期方針:

```text
RelayKVがHiCacheにprefetch hintを出す
RelayKV score / Anchor / Heavy-Hitterをeviction hintにする
RadixTree-backed hierarchical KV cullingを検討する
```

## 10. 現在のSGLang実装フェーズ

現在は **ForwardBatch read-only metadata plumbing** フェーズである。

既に確認済みの流れ:

```text
1. ForwardBatch read-only metadata
2. runtime observation summary
3. host backup candidate summary
4. runtime/candidate join summary
5. readonly runtime candidate join report
```

現在の目的:

```text
request_id の観測
req_pool_idx の観測
seq_len の観測
runtime observation metadata と host backup copy candidate summary の join
read-only report化
```

まだやらないこと:

```text
KV pool read
host backup copy実行
attention接続
scheduler decision変更
runtime writeback
```

一番重要な境界:

```text
RelayKVはまず、SGLang実行経路の外側にあるread-only metadata layerとして成立させる。
```

## 11. 現在までの実装済み要素

### 11.1 ForwardBatch read-only metadata

env-on only で `ForwardBatch` 上に read-only metadata carrier を持たせる。

```text
relaykv_runtime_observation_metadata
```

含む情報:

```text
request_id / rid
request_index_in_batch
req_pool_idx
seq_len
extend_seq_len
extend_prefix_len
phase
source="forward_batch_readonly_runtime_observation_metadata"
```

安全条件:

```text
env offでは作らない
GPU tensorを読まない
KV poolを読まない
scheduler decisionを変えない
attentionへ接続しない
```

### 11.2 runtime observation summary

runtime observation hook は env-on の時だけ動く。

主な summary:

```text
relaykv_runtime_observation_readonly_metadata_summary
```

確認済み:

```text
req_pool_idx が summary に載る
req_pool_idx_none=false
/generate 200
```

### 11.3 host backup candidate summary

host backup copy candidate event を集計する helper がある。

```text
summarize_host_backup_copy_candidates_for_smoke(...)
log_host_backup_copy_candidate_summary(...)
```

ここではまだ実copyしない。

扱うのは candidate event payload / summary のみ。

### 11.4 runtime/candidate join summary

追加済み helper:

```text
join_runtime_observation_with_host_backup_candidates_for_smoke(...)
```

join key:

```text
request_id
req_pool_idx
layer_id
```

candidate側互換:

```text
layer_idx -> layer_id
req_pool_idx / req_pool_index / request_pool_idx
```

summary-only 入力の場合:

```text
join_granularity="summary_only_unjoinable"
```

これは、aggregate summary から存在しない詳細を捏造しないための安全措置。

### 11.5 readonly runtime candidate join report

追加済み helper:

```text
build_relaykv_readonly_runtime_candidate_join_report_for_smoke(...)
```

目的:

```text
runtime_observation_summary
host_backup_candidate_summary
join_summary

を1つのreport dictにまとめる。
```

report:

```text
report_type="relaykv_readonly_runtime_candidate_join_report"
report_generated_from_readonly_inputs=true
overall_safety_status
```

overall safety rule:

```text
pass:
  readonly input由来
  safety counters all zero

fail:
  safety counter nonzero
  または readonly input由来ではない
```

## 12. 開発フェーズ全体

```text
Phase 0:
  ForwardBatch metadata観測

Phase 1:
  runtime observation metadata と candidate summary のjoin確認

Phase 2:
  dry-run policy
  どのblockを選ぶかをログだけで出す

Phase 3:
  safe KV materialization
  host backup copy / candidate KVを安全に取り出す

Phase 4:
  attention接続
  RECENT + ANCHOR + RETRIEVED をworking KVとして使う

Phase 5:
  品質評価
  mean_abs_diff / top-k overlap / divergence / task accuracyを見る

Phase 6:
  高速化
  semantic tree / KV-BVH / temporal reuse / LOD / compressed-domain scoring / HiCache prefetch

Phase 7:
  research expansion
  precision promotion / adaptive budget / native sparse kernels など
```

合言葉:

```text
まず正しく動く RelayKV。
次に品質が保てる RelayKV。
最後に速い RelayKV。
```

現在位置:

```text
Phase 1 の終盤
```

次は以下。

```text
Phase 2:
  dry-run policy log
```

## 13. 今すぐ metadata/log 設計に入れてよい項目

実装はまだ先でも、ログ・metadata schema に予約してよい項目。

### KV class

```text
kv_class:
  RECENT
  ANCHOR
  RETRIEVED
  COLD_CANDIDATE
```

`TRANSIENT` は以下として扱う。

```text
decode_state / log concept
```

### Budget

```text
available_kv_budget_mib
kv_working_budget_tokens
working_kv_tokens
recent_tokens
anchor_tokens
transient_tokens
retrieved_tokens
cold_candidate_tokens
```

### Layer policy

```text
layer_budget_policy:
  uniform
  pyramid
  hourglass
  hard_layer_heavy
  adaptive
```

### Retrieval metadata

```text
selected_block_ids
candidate_block_ids
score_margin
retrieval_top_k
layer_id
request_id
req_pool_idx
seq_len
```

### Precision / residency reserved fields

```text
precision_level
source_precision
working_precision
promotion_reason
promotion_step
residency_level
```

### Future index fields

```text
semantic_index_id
semantic_tree_depth
nodes_visited
nodes_pruned
```

### Heavy-hitter fields

```text
retrieval_reuse_count
last_retrieved_step
retrieval_frequency
anchor_score
heavy_hitter_score
```

### GQA / head-aware budget

```text
kv_head_group_budget
```

## 14. 今は実装しないもの

以下は有望だが、現フェーズでは入れない。

```text
semantic tree
KV-BVH
LOD retrieval
temporal reuse optimization
compressed-domain scoring
learned eviction
HiCache prefetch hint
RadixTree-backed culling
adaptive precision promotion
native sparse attention kernel
agent memory KV reuse
```

これらは `Phase 6 optimization / Phase 7 research expansion` として扱う。

## 15. 次の実装候補: dry-run policy log

ここまでで metadata plumbing は以下まで成立した。

```text
ForwardBatch metadata
runtime observation summary
host backup candidate summary
join summary
readonly report
```

次は、まだ実copyに進まず、dry-run policy log を追加するのが安全。

### 目的

```text
request_id / req_pool_idx / layer_id / seq_len ごとに、
どのKV blockを selected / candidate / anchor / recent として扱う予定かを
ログだけで出す。
```

### まだやらない

```text
KV pool read
host backup copy実行
attention接続
scheduler変更
runtime writeback
```

### dry-run policy log の出力イメージ

```text
event_type="relaykv_policy_dry_run"
request_id
req_pool_idx
seq_len
layer_id
policy_state="dry_run"
kv_budget_tokens
recent_tokens
anchor_tokens
retrieval_budget_tokens
candidate_block_ids
selected_block_ids
anchor_block_ids
kv_classes_present
source="readonly_metadata_policy_dry_run"
```

safety flags:

```text
source_mutated=false
attention_override=false
kv_cache_mutation=false
runtime_writeback=false
scheduler_policy_noop=true
```

### dry-run の価値

dry-run policy log により、実copy前に以下を確認できる。

```text
request_id / req_pool_idx / layer_id の対応が崩れていないか
budget計算の形が妥当か
AnchorとRetrievedが別枠で扱えているか
RECENT / COLD_CANDIDATE の境界をログで説明できるか
```

## 16. 次にCodexへ渡すなら

次のCodexタスクは、まだ実runtime接続ではなく、pure helper + smoke に限定する。

推奨タスク:

```text
Add RelayKV read-only dry-run policy event builder and smoke.
```

許可ファイル候補:

```text
python/sglang/srt/relaykv/metrics.py
scripts/relaykv_policy_dry_run_smoke.py
```

禁止:

```text
ForwardBatch
ModelRunner
scheduler
attention
memory_pool
flashinfer
.github/workflows
```

expected helper:

```text
build_relaykv_policy_dry_run_events_for_smoke(...)
summarize_relaykv_policy_dry_run_events_for_smoke(...)
```

狙い:

```text
readonly report / runtime metadata / fake block metadata から
selected_block_ids / anchor_block_ids / candidate_block_ids を
ログだけで作る。
```

## 17. 最重要判断

今は研究アイデアを大量に実装へ入れる段階ではない。

今やること:

```text
ForwardBatch metadata observation
runtime observation metadata
host backup copy candidate summary
join確認
readonly report
dry-run policy log
```

まだやらないこと:

```text
KV pool read
attention接続
scheduler変更
runtime writeback
semantic tree
KV-BVH
precision promotion
```

開発順序:

```text
metadata
→ dry-run
→ materialization
→ attention
→ quality
→ speed
```

この順番を崩さない。
