# RelayKV Phase 4.5: SGLang Attention Path Inspection

## 1. 日付確認

- 対象日付: 2026-05-03
- タイムゾーン: JST / Japan time
- 本メモは Phase 4.5 の inspection/report のみを目的とし、runtime code / attention backend / scheduler / KV pool / RadixTree / HiCache の実装変更は行っていない。

## 2. 見たファイル

主に確認したファイル:

- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/layers/radix_attention.py`
- `python/sglang/srt/layers/attention/base_attn_backend.py`
- `python/sglang/srt/layers/attention/flashinfer_backend.py`
- `python/sglang/srt/layers/attention/triton_backend.py`
- `python/sglang/srt/layers/attention/flashattention_backend.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/mem_cache/hicache_storage.py`
- `python/sglang/srt/mem_cache/hiradix_cache.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/models/llama.py`
- `python/sglang/srt/models/transformers.py`

重要クラス・関数・メソッド:

- `ForwardBatch`
- `ForwardBatch.init_new(...)`
- `ModelRunner.forward_decode(...)`
- `ModelRunner.forward_extend(...)`
- `ModelRunner.forward_split_prefill(...)`
- `ModelRunner.forward(...)`
- `RadixAttention.forward(...)`
- `AttentionBackend.forward(...)`
- `FlashInferAttnBackend.init_forward_metadata(...)`
- `FlashInferAttnBackend.forward_extend(...)`
- `FlashInferAttnBackend.forward_decode(...)`
- `FlashAttentionBackend.init_forward_metadata(...)`
- `TritonAttnBackend.init_forward_metadata(...)`
- `TritonAttnBackend.forward_extend(...)`
- `TritonAttnBackend.forward_decode(...)`
- `ReqToTokenPool.alloc(...)`
- `ReqToTokenPool.write(...)`
- `KVCache.set_kv_buffer(...)`
- `MHATokenToKVPool.get_kv_buffer(...)`
- `RadixCache.match_prefix(...)`
- `RadixCache.insert(...)`
- `RadixCache.cache_finished_req(...)`
- `HiRadixCache.match_prefix(...)`
- `HiRadixCache.prefetch_from_storage(...)`
- `Scheduler._prefetch_kvcache(...)`

補足:

- 指定の `python/sglang/srt/mem_cache/hicache_controller.py` はこのブランチには存在しなかった。
- 実際の階層キャッシュ制御は `python/sglang/srt/managers/cache_controller.py` と `python/sglang/srt/mem_cache/hiradix_cache.py`、および `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` に分散している。

## 3. attention backend 呼び出しの入口

観察できた主経路は以下。

1. `ModelRunner` が `ForwardBatch` を組み立てる。
2. `ModelRunner.forward_decode(...)` / `forward_extend(...)` / `forward_split_prefill(...)` が `self.attn_backend.init_forward_metadata(forward_batch)` を呼ぶ。
3. その後 `self.model.forward(input_ids, positions, forward_batch, ...)` を呼ぶ。
4. モデル層では `LlamaAttention.forward(...)` のような attention module が `q, k, v` を作り、`self.attn(q, k, v, forward_batch)` を呼ぶ。
5. `self.attn` は `RadixAttention` であり、最終的に `forward_batch.attn_backend.forward(...)` に渡す。
6. `AttentionBackend.forward(...)` が `forward_mode` を見て `forward_decode(...)` または `forward_extend(...)` に振り分ける。

入口の具体例:

- `python/sglang/srt/models/llama.py`
  - `LlamaAttention.forward(...)`
  - `attn_output = self.attn(q, k, v, forward_batch)`
- `python/sglang/srt/models/transformers.py`
  - HuggingFace 側変換経路でも `self_attn.forward(query, key, value, forward_batch=forward_batch)` に集約

backend 受け口:

- `FlashInferAttnBackend.forward_extend(...)` / `forward_decode(...)`
- `TritonAttnBackend.forward_extend(...)` / `forward_decode(...)`
- `FlashAttentionBackend.forward_extend(...)` / `forward_decode(...)`

prefill/decode 差分:

- `decode`
  - `forward_mode.is_decode_or_idle()`
  - 既存 prefix を `req_to_token_pool.req_to_token[...]` から page/index table 化して参照
  - `out_cache_loc` には通常 1 token/step の書き込み先が入る
- `extend/prefill`
  - `extend_prefix_lens` / `extend_seq_lens` / `extend_start_loc` を使う
  - prefix 部分は既存 KV を参照しつつ、extend 部分は `out_cache_loc` に新規 KV を保存
  - backend により paged / ragged / unified kernel の選択差がある

## 4. ForwardBatch から attention に渡る主要 metadata

`ForwardBatch` に含まれ、attention path で実際に参照される主要 metadata:

- request id
  - `rids: Optional[List[str]]`
  - `ForwardBatch.init_new(...)` で `batch.reqs` から構築
- `req_pool_indices`
  - request ごとの `req_to_token_pool` 行インデックス
- `seq_lens`
  - request ごとの現在 sequence length
- `seq_lens_cpu`
  - backend metadata 計算で最大長や CPU 側 planner に利用
- `positions`
  - decode では `clamp_position(batch.seq_lens)`、extend では `compute_position(...)` で計算
- `out_cache_loc`
  - 今回の token の KV 書き込み先 index
- `out_cache_loc_swa`
  - hybrid SWA 時の変換済み location
- `extend_seq_lens`
- `extend_prefix_lens`
- `extend_start_loc`
- `encoder_out_cache_loc`
  - cross attention 用
- `req_to_token_pool`
  - request -> token/KV index 対応表
- `token_to_kv_pool`
  - layer 別の物理 KV バッファ保持先
- `attn_backend`
  - 実際に kernel planning / execution を受ける backend
- `spec_info`
  - speculative decoding / verify 系 metadata
- `attn_cp_metadata`
  - context parallel 用 metadata
- `forward_mode`
  - decode / extend / target_verify / draft_extend / idle などの mode flag
- `cross_attention_custom_mask`
  - 一部 backend planner に渡る

RelayKV 観点で特に重要なのは以下。

- request 識別子: `rids`, `req_pool_indices`
- sequence 位置: `seq_lens`, `positions`, `extend_*`
- 現在 step の書き込み先: `out_cache_loc`
- prefix 参照表: `req_to_token_pool.req_to_token`
- layer 物理 KV 実体: `token_to_kv_pool.get_kv_buffer(layer_id)`
- backend planner 状態: `forward_batch.attn_backend.forward_metadata`

## 5. KV index / pool flow

観察できた flow は以下。

1. request 単位
   - `ReqToTokenPool` が `req_pool_idx` を払い出す。
   - `req_to_token_pool.req_to_token[req_pool_idx, pos]` が、その request の token position ごとの KV index を持つ。

2. step 単位
   - scheduler / batch 準備側が `out_cache_loc` を作る。
   - これは「今回生成または extend する token を、物理 KV pool のどこへ保存するか」の index 列。

3. layer 単位
   - backend の `forward_extend(...)` / `forward_decode(...)` が `token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, k, v, ...)` を呼ぶ。
   - 同じ `out_cache_loc` を layer ごとの K/V buffer に対して使う。
   - 実データは layer ごとに別 buffer、index 空間は token location として共通。

4. 参照時
   - decode/prefix attention は `req_to_token_pool.req_to_token[req_pool_indices, :max_seq_len]` から page/index table を作る。
   - backend はそれを使って `token_to_kv_pool.get_kv_buffer(layer_id)` の K/V にアクセスする。

layer-specific と request-specific の境界:

- request-specific
  - `req_pool_idx`
  - `seq_lens`
  - `positions`
  - `req_to_token_pool.req_to_token[row, :]`
  - `out_cache_loc`
- layer-specific
  - `layer.layer_id`
  - `token_to_kv_pool.get_key_buffer(layer_id)`
  - `token_to_kv_pool.get_value_buffer(layer_id)`
  - quant scale や sliding window 設定など layer 属性

要点:

- `req_to_token_pool` は論理 index table
- `token_to_kv_pool` は物理 layer buffer
- `out_cache_loc` は「今回の write index」
- backend metadata は `req_to_token_pool` と `seq_lens` から read path を計画する

## 6. RadixAttention / RadixTree boundary

責務の見え方:

- `RadixAttention`
  - attention layer の共通フロント
  - `forward_batch.attn_backend.forward(...)` への橋渡し
  - attention kernel 選択の論理境界
- `RadixCache` / RadixTree
  - prefix cache 木構造
  - token prefix と KV index 群の対応管理
  - `match_prefix(...)`, `insert(...)`, `cache_finished_req(...)`, `evict(...)`
  - request 完了時に `req_to_token_pool.req_to_token[...]` から KV index を tree に取り込む

RelayKV が短期で置き換えてはいけないもの:

- `RadixAttention.forward(...)` 自体の backend dispatch
- `RadixCache.match_prefix(...)`
- `RadixCache.insert(...)`
- `RadixCache.cache_finished_req(...)`
- `RadixCache.evict(...)`
- `req_to_token_pool.write(...)` による prefix index の更新

RelayKV が将来 read-only 参照し得る metadata:

- `req.prefix_indices`
- `req.last_node`
- `req.req_pool_idx`
- `req_to_token_pool.req_to_token[...]`
- `RadixCache.match_prefix(...)` の結果長
- page-aligned prefix 長

理由:

- RadixTree は scheduler の prefix reuse と eviction の整合性中心にある。
- ここを RelayKV が差し替えると、Phase 4.5 の安全境界を超えて scheduler policy / cache mutation を発生させる。

## 7. HiCache / HiRadixTree boundary

責務の見え方:

- `HiCacheStorage`
  - host/storage への汎用 KV I/O 抽象
  - `batch_exists_v2`, `batch_get_v2`, `batch_set_v2`
- `HiRadixCache`
  - `RadixCache` を拡張し、device/host/storage の階層管理を持つ
  - `match_prefix(...)` で device hit と host hit を分離
  - `prefetch_from_storage(...)` で storage 先読みを起動
- `Scheduler._prefetch_kvcache(...)`
  - request enqueue 時に HiCache prefetch を起動する入口

RelayKV が短期で変更してはいけないもの:

- `HiRadixCache.match_prefix(...)`
- `HiRadixCache.prefetch_from_storage(...)`
- `HiRadixCache.attach_storage_backend(...)`
- `HiRadixCache.detach_storage_backend(...)`
- `HiCacheStorage.batch_get_v2(...)` / `batch_set_v2(...)`
- scheduler の HiCache prefetch 呼び出し

prefetch/eviction hint の扱い:

- これは将来フェーズの対象であり、Phase 4.5 では future-only とみなすべき。
- 理由は、hint であっても storage path / host pool / eviction timing に波及し、`kv_pool_read_count` や `runtime_writeback_true_count` を動かす危険があるため。

RelayKV が read-only で参照し得る metadata:

- `host_hit_length`
- `last_host_node`
- `last_hash`
- `prefix_keys`
- `prefetch_threshold` や policy 名

ただし短期では、これらも runtime で消費せず「設計候補として文書化」に止めるべき。

## 8. RelayKV hook candidate classification

### Candidate A: `ForwardBatch.init_new(...)` 後の read-only metadata 収集

- 分類: metadata-only hook
- location: `python/sglang/srt/model_executor/forward_batch_info.py`
- needed inputs:
  - `rids`
  - `req_pool_indices`
  - `seq_lens`
  - `positions`
  - `out_cache_loc`
  - `extend_*`
  - `forward_mode`
- output 影響: なし
- safety risk: 低
- implementation cost: 低
- recommendation: 最も安全。Phase 4.6 設計の出発点として推奨

### Candidate B: `ModelRunner.forward_*` の backend init 直後

- 分類: shadow-only hook
- location: `python/sglang/srt/model_executor/model_runner.py`
- needed inputs:
  - `forward_batch`
  - `forward_batch.attn_backend.forward_metadata`
  - `positions`
  - `out_cache_loc`
- output 影響: hook 自体は無しにできるが、場所が execution 直前なので事故時の影響範囲は広い
- safety risk: 中
- implementation cost: 中
- recommendation: 設計候補にはなるが、最初の接続点にはしない方がよい

### Candidate C: `RadixAttention.forward(...)` 入口での shadow capture

- 分類: shadow-only hook
- location: `python/sglang/srt/layers/radix_attention.py`
- needed inputs:
  - `layer.layer_id`
  - `forward_batch`
  - `q.shape`, `k.shape`, `v.shape`
  - backend 種別
- output 影響: 実装次第で無しにできる
- safety risk: 中
- implementation cost: 中
- recommendation: 将来の layer-by-layer 観測には有用だが、Phase 4.7 以降に限定

### Candidate D: backend `init_forward_metadata(...)` の planner result 参照

- 分類: comparison-only hook
- location:
  - `python/sglang/srt/layers/attention/flashinfer_backend.py`
  - `python/sglang/srt/layers/attention/triton_backend.py`
  - `python/sglang/srt/layers/attention/flashattention_backend.py`
- needed inputs:
  - `req_pool_indices`
  - `seq_lens`
  - planner が作った `page_table` / `kv_indices` / `cu_seqlens_*`
- output 影響: なしで済むが backend 内部差分が大きい
- safety risk: 中
- implementation cost: 中〜高
- recommendation: backend 共通抽象が弱いので、比較専用の設計対象に止める

### Candidate E: backend `forward_extend(...)` / `forward_decode(...)` の直前比較

- 分類: comparison-only hook
- location:
  - `FlashInferAttnBackend.forward_extend/decode`
  - `TritonAttnBackend.forward_extend/decode`
  - `FlashAttentionBackend.forward_extend/decode`
- needed inputs:
  - `layer.layer_id`
  - `out_cache_loc`
  - `req_to_token_pool` 由来 table
  - `token_to_kv_pool.get_kv_buffer(layer_id)`
  - optional `q/k/v`
- output 影響: 比較専用なら避けられるが、極めて実行パス近接
- safety risk: 高
- implementation cost: 高
- recommendation: Phase 4.8 でも guard を強く掛けるべき。Phase 4.5 では未着手維持

### Candidate F: backend 内で KV source を差し替える path

- 分類: real override hook
- location: 各 backend の `forward_extend(...)` / `forward_decode(...)`
- needed inputs:
  - RelayKV が供給する per-layer KV source
  - page/index mapping
  - seq metadata
  - cache write/read consistency
- output 影響: あり
- safety risk: 最高
- implementation cost: 最高
- recommendation: 現時点で禁止。実 attention override はまだ実装しない

## 9. 最小安全経路

### Phase 4.6 isolated attention comparison design

- 編集候補を `notes/` と非 runtime の設計文書に限定
- `ForwardBatch` 由来 metadata と backend planner metadata の比較対象を定義
- 比較結果は counters ではなく report/plan のみで整理

### Phase 4.7 shadow-only connection smoke

- runtime 出力非介入
- `ForwardBatch` または `ModelRunner` の既存 read-only observation 境界のみ使用
- backend / scheduler / KV pool への write 禁止

### Phase 4.8 comparison-only smoke

- 実 attention 出力は不変更
- 比較対象は layer ごとの metadata / index planning のみ
- `attention_comparison_executed_count` を初めて増やすなら、完全隔離フラグと fail-closed が必要

### Phase 4.9 guarded override design

- override 設計だけ先行
- 実装は別 phase
- 条件:
  - backend ごとの planner 差分吸収方針
  - layer-local KV source contract
  - prefix/decode 両系統の整合条件
  - rollback / noop fallback / output equivalence guard

## 10. 結論

- 現時点では real attention override を実装すべきではない。
- 最も安全な次アクションは、`ForwardBatch` と backend planner metadata の対応表を作る Phase 4.6 の isolated comparison design である。
- 次に安全に触れられるファイルは runtime 外の文書ファイルのみ。
- runtime 側で次に比較的安全な調査候補を挙げるなら、将来フェーズで `python/sglang/srt/model_executor/forward_batch_info.py` と `python/sglang/srt/model_executor/model_runner.py` の read-only observation 境界だけを検討対象にするのが妥当。
- attention backend / scheduler / KV pool / RadixTree / HiCache を直接 edit する段階ではない。

## 付記: safety boundary との整合

今回の inspection では以下を変更していない。

- `attention_comparison_executed_count`
- `attention_override_true_count`
- `kv_pool_read_count`
- `kv_snapshot_count`
- `runtime_writeback_true_count`
- `scheduler_policy_noop_false_count`
- `kv_cache_mutation_true_count`
- `source_mutated_true_count`

したがって、Phase 4.1〜4.4 の metadata-only safety boundary は維持されている。
