# RelayKV Phase 4.6 Isolated Attention Comparison Design

## 日付確認

- Design date: **2026-05-03**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. この設計メモの目的

このメモは、RelayKV/SGLang integration の **Phase 4.6: Isolated Attention Comparison Design** を定義する。

Phase 4.5 の SGLang attention path inspection により、attention 実行入口と KV index / pool の責務分離が見えた。

確認された大枠:

```text
ModelRunner
→ model.forward(...)
→ 各 model の self_attn
→ RadixAttention.forward(...)
→ backend.forward_extend / backend.forward_decode
```

KV の責務分離:

```text
req_to_token_pool.req_to_token = request 内 logical token index mapping
token_to_kv_pool = layer-specific physical KV buffer
out_cache_loc = current step write destination
```

Phase 4.6 の目的は、実 output に影響を与えずに、将来の **Full-KV attention result vs RelayKV working-KV attention result** 比較を isolated / comparison-only で行うための安全境界を設計すること。

重要:

```text
Phase 4.6 はまだ実装フェーズではない。
Phase 4.6 では real attention override を許可しない。
Phase 4.6 では production output へ影響する変更を許可しない。
```

---

## 2. 現在の到達点

Phase 4.1〜4.4 で metadata-only chain は閉じている。

```text
attention_handoff_candidate
→ attention_connection_dry_run_result
→ attention_override_noop_result
→ attention_comparison_plan
```

Phase 4.5 で SGLang attention path inspection を実施し、hook candidate を分類した。

現時点での推奨順:

```text
1. ForwardBatch.init_new(...) 後の metadata-only hook
2. ModelRunner / RadixAttention 周辺の shadow-only hook
3. backend init_forward_metadata(...) の comparison-only hook
4. backend forward_extend/decode(...) の isolated comparison
5. real override hook はまだ禁止
```

---

## 3. Phase 4.6 の設計方針

Phase 4.6 は、実 attention backend を変更する前に、comparison-only attention をどのように isolated に扱うかを決める。

中心となる問い:

```text
Full-KV attention result
vs
RelayKV working-KV attention result

を比較するには、
どこで KV pool read / tensor read / attention execution を許可する必要があるか。
```

ただし、Phase 4.6 ではこの許可を実装しない。

設計上、次の境界を分離する。

```text
metadata-only zone:
  これまで通り安全。KV pool read なし。attention execution なし。

isolated comparison zone:
  将来、限定条件下で KV pool read / tensor read / attention execution を許可する候補。
  production output には使わない。

guarded override zone:
  RelayKV attention result を実 output に使う候補。
  Phase 4.6 では扱わない。
```

---

## 4. Phase 4.6 で許可しないこと

Phase 4.6 design では、以下をまだ許可しない。

```text
attention_override_true_count > 0
production output modification
scheduler decision mutation
runtime writeback
KV cache free
RadixTree modification
HiCache prefetch hint
attention backend source replacement
```

さらに、設計メモ段階では以下も実行しない。

```text
KV pool read
GPU tensor read
KV snapshot
actual attention execution
FlashInfer / FlashAttention / Triton backend modification
```

Phase 4.6 の成果物は design memo のみ。

---

## 5. 将来の isolated comparison で許可し得る counter

Phase 4.6 では実行しないが、Phase 4.7 以降の isolated comparison smoke で許可候補になる counter は以下。

```text
attention_comparison_executed_count > 0
kv_pool_read_count > 0
tensor_read_count > 0
```

ただし、isolated comparison でも以下は 0 固定。

```text
attention_override_true_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
kv_cache_mutation_true_count=0
source_mutated_true_count=0
```

重要:

```text
kv_pool_read_count > 0 は comparison-only isolated smoke でのみ許可。
production runtime path ではまだ許可しない。
```

---

## 6. 比較対象の定義

### 6.1 Full-KV attention plan

Full-KV plan は、SGLang 標準 attention が参照する KV set を baseline として扱う。

metadata-only phase では、これは `full_kv_block_ids` として表現されている。

将来の isolated comparison では、Full-KV side の attention result は以下のどちらかで扱う。

```text
A. 既存 SGLang attention output を shadow capture する
B. isolated helper で Full-KV indices を使って別途 attention を計算する
```

短期では A の方が安全。

理由:

```text
既存 output を変えない
backend 内の二重実行を避けられる
Full-KV baseline が production path と一致しやすい
```

### 6.2 RelayKV working-KV attention plan

RelayKV side は、Phase 4.4 の comparison plan にある以下を使う。

```text
relaykv_working_kv_block_ids
relaykv_working_kv_block_count
working_to_full_block_ratio
coverage_ratio
missing_from_full_block_ids
full_only_block_ids
```

将来の isolated comparison では、この block ids を actual KV indices に展開する必要がある。

必要になる可能性が高い情報:

```text
request_id
req_pool_idx
layer_id
seq_len
block_id -> token span mapping
token span -> req_to_token index mapping
req_to_token index -> token_to_kv_pool physical index mapping
```

---

## 7. block_id から actual KV index への変換境界

RelayKV の logical block と SGLang の physical KV pool index は同じではない。

分離すべき変換:

```text
RelayKV block_id
→ token span
→ request-local token positions
→ req_to_token_pool.req_to_token entries
→ token_to_kv_pool physical KV locations
→ layer-specific K/V tensors
```

Phase 4.6 で設計する必要があること:

```text
1. block_id -> token span mapping をどこで持つか
2. request_id / req_pool_idx / layer_id をどう join するか
3. req_to_token_pool を read-only で参照する境界
4. token_to_kv_pool を read-only で参照する境界
5. K/V tensor materialization をどこで isolated に行うか
```

短期方針:

```text
metadata-only では block_id / token span / req_pool_idx まで。
physical KV index read は isolated comparison smoke まで遅らせる。
```

---

## 8. hook candidate 再分類

Phase 4.5 inspection の結果を踏まえ、Phase 4.6 では hook candidate を以下に再分類する。

### 8.1 ForwardBatch 後段 metadata-only hook

推奨度: 高

用途:

```text
RelayKV comparison plan の生成
request_id / req_pool_idx / seq_len の観測
block plan の横置き
```

利点:

```text
runtime output に影響しない
attention backend に触れない
scheduler に触れない
既に実績がある
```

欠点:

```text
actual attention result comparison はできない
KV tensor には触れない
```

判断:

```text
Phase 4.6 でも引き続き安全な主経路。
```

---

### 8.2 ModelRunner 周辺 shadow-only hook

推奨度: 中〜高

用途:

```text
既存 Full-KV output を shadow capture する候補
RelayKV comparison plan と runtime observation を join する候補
```

利点:

```text
backend に入る前後の情報を見られる可能性
runtime output には影響させずに shadow log 可能
```

リスク:

```text
forward path に import / hook を置くと性能・安定性に影響し得る
既存の optional observation hook と同じ default-off 方針が必要
```

判断:

```text
Phase 4.7 shadow-only connection smoke の候補。
```

---

### 8.3 RadixAttention.forward(...) 入口 hook

推奨度: 中

用途:

```text
attention layer 単位で comparison plan を見る候補
layer_id / forward_batch / backend dispatch の直前情報を取る候補
```

利点:

```text
attention 入口に近い
layerごとの情報が取りやすい可能性
```

リスク:

```text
RadixAttention は prefix cache / attention abstraction の重要経路
不用意に触ると SGLang semantics に影響し得る
```

判断:

```text
Phase 4.7 以降の shadow-only hook 候補。
Phase 4.6 では触らない。
```

---

### 8.4 backend init_forward_metadata(...) hook

推奨度: 中

用途:

```text
backend-specific metadata が確定する地点で comparison plan を join する候補
```

利点:

```text
decode/prefill backend metadata に近い
KV index / paged attention metadata に近い可能性
```

リスク:

```text
backendごとの差異が大きい
FlashInfer / Triton / FlashAttention で扱いが変わる可能性
```

判断:

```text
Phase 4.8 comparison-only smoke の候補。
```

---

### 8.5 backend forward_decode / forward_extend hook

推奨度: 低〜中

用途:

```text
実 attention comparison を行う最終候補
```

利点:

```text
実 attention execution に最も近い
Full-KV vs RelayKV attention を比較しやすい可能性
```

リスク:

```text
高リスク
backend-specific
性能影響大
tensor shape / dtype / device safety が必要
```

判断:

```text
Phase 4.6 では設計対象のみ。
実装はまだしない。
```

---

### 8.6 backend 内 KV source 差し替え

推奨度: 現時点では禁止

用途:

```text
RelayKV working-KV を実 output に使う real override
```

判断:

```text
Phase 4.6 では禁止。
Phase 4.9 guarded override design 以降。
```

---

## 9. isolated comparison の候補設計

### 9.1 Shadow capture first

最小安全経路は、既存 attention output を変えずに Full-KV side を capture すること。

```text
existing Full-KV attention output
→ shadow capture metadata
→ RelayKV comparison plan と join
```

この時点では RelayKV side attention はまだ計算しない。

許可される可能性がある counter:

```text
attention_shadow_capture_count > 0
```

まだ 0 固定:

```text
attention_comparison_executed_count=0
attention_override_true_count=0
kv_pool_read_count=0
```

### 9.2 Isolated RelayKV attention compare

次段階で、RelayKV working-KV side の isolated attention を計算する。

```text
RelayKV working block ids
→ token spans
→ physical KV indices
→ gather K/V tensors
→ isolated attention
→ compare with Full-KV output
```

この段階で初めて、限定的に以下を許可する可能性がある。

```text
kv_pool_read_count > 0
tensor_read_count > 0
attention_comparison_executed_count > 0
```

ただし、以下は 0 固定。

```text
attention_override_true_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
kv_cache_mutation_true_count=0
source_mutated_true_count=0
```

---

## 10. comparison metrics

将来の isolated comparison で記録すべき metrics。

```text
mean_abs_diff
max_abs_diff
cosine_similarity
topk_logit_overlap
top1_same
first_divergence_step
working_to_full_block_ratio
coverage_ratio
missing_from_full_block_count
full_only_block_count
```

ただし、attention output の比較だけなら logits までは見えない可能性がある。

段階分け:

```text
attention-output level:
  mean_abs_diff
  max_abs_diff
  cosine_similarity

logit/output level:
  topk_logit_overlap
  top1_same
  first_divergence_step
```

Phase 4.6 では metrics schema のみ設計する。

---

## 11. fallback / rollback 条件

isolated comparison から guarded override へ進むには、fallback 条件が必須。

候補:

```text
RelayKV working_kv_block_count == 0
coverage_ratio below threshold
missing_from_full_block_count too high
mean_abs_diff above threshold
max_abs_diff above threshold
cosine_similarity below threshold
shape mismatch
dtype mismatch
device mismatch
backend unsupported
layer unsupported
request mode unsupported
prefill path
multi-request batching unsupported
```

fallback 動作:

```text
Use standard SGLang Full-KV attention output.
Do not mutate RelayKV state.
Do not write back KV.
Do not change scheduler decision.
```

rollback 条件:

```text
any exception in RelayKV comparison path
any nonzero source_mutated
any runtime writeback attempt
any scheduler mutation attempt
unexpected attention_override_true before guarded override phase
```

---

## 12. Phase 4.6 deliverable

Phase 4.6 の deliverable はこの設計メモ。

次に作るべきものは、実装ではなく、より具体的な smoke design。

候補:

```text
Phase 4.7 Attention Shadow Capture Design
```

または、最小実装 smoke として:

```text
Phase 4.7 Attention Shadow Capture Metadata Smoke
```

ただし、最初の実装もまだ default-off / metadata-only に近い形がよい。

---

## 13. 推奨次ステップ

推奨順:

```text
Phase 4.7:
  Attention Shadow Capture Metadata Smoke
  - existing output を変えない
  - attention_shadow_capture_count > 0 を許可
  - attention_override_true_count=0
  - kv_pool_read_count=0

Phase 4.8:
  Isolated KV Index Resolution Design
  - block_id -> token span -> req_to_token -> physical KV index の read-only 境界を設計

Phase 4.9:
  Isolated Attention Comparison Smoke
  - 限定的に kv_pool_read_count > 0 / attention_comparison_executed_count > 0 を許可
  - output は使わない

Phase 5:
  Guarded Attention Override Design
```

---

## 14. commit command

この設計メモを repo に保存する場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/relaykv_phase4_6_isolated_attention_comparison_design_2026-05-03.ja.md \
  notes/relaykv_phase4_6_isolated_attention_comparison_design_2026-05-03.ja.md

git status --short
git diff --check

git add notes/relaykv_phase4_6_isolated_attention_comparison_design_2026-05-03.ja.md
git commit -m "docs: design relaykv isolated attention comparison"
git push mine relaykv-host-backup-shadow
```
