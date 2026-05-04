# RelayKV Phase 4.5 Real Attention Connection Design

## 日付確認

- Design date: **2026-05-03**
- Date basis: **JST / Japan time**
- Repository: `~/work/sglang-relaykv`
- Branch: `relaykv-host-backup-shadow`

---

## 1. この設計メモの目的

このメモは、RelayKV/SGLang integration の **Phase 4.5: Real Attention Connection Design** を定義する。

Phase 4.1〜4.4 では、実 attention backend に触れず、metadata-only で以下の chain を閉じた。

```text
attention_handoff_candidate
→ attention_connection_dry_run_result
→ attention_override_noop_result
→ attention_comparison_plan
```

Phase 4.5 では、次に実 attention path を読むための設計境界を作る。

重要:

```text
Phase 4.5 は実装フェーズではなく、実 attention 接続前の設計・調査フェーズ。
```

この段階では、まだ `attention_override_true_count > 0` を許可しない。

---

## 2. 現在の到達点

現時点の安全な到達点:

```text
Full-KV plan vs RelayKV working-KV plan を metadata-only で比較可能
```

Phase 4.4 pass flow の安全 counters:

```text
attention_comparison_executed_count=0
attention_connection_attempted_count=2
attention_override_true_count=0
attention_override_noop_count=2
kv_pool_read_count=0
kv_snapshot_count=0
runtime_writeback_true_count=0
scheduler_policy_noop_false_count=0
kv_cache_mutation_true_count=0
source_mutated_true_count=0
```

これは、RelayKV の working set を attention に渡す前段 schema まではできているが、実際の attention execution / override にはまだ到達していないことを示す。

---

## 3. Phase 4.5 の目的

Phase 4.5 の目的は、SGLang の attention path を読み、RelayKV を接続する場合の最小安全経路を決めること。

具体的には次を明確にする。

```text
1. SGLang attention backend の呼び出し位置
2. ForwardBatch / attention metadata / req_to_token_pool / token_to_kv_pool の関係
3. decode step で attention が参照する KV index の流れ
4. RadixAttention / RadixTree との境界
5. HiCache / HiRadixTree との境界
6. RelayKV が入れる候補位置
7. RelayKV が入ってはいけない位置
8. shadow-only / comparison-only 接続の最小実装方針
9. rollback / fallback 条件
```

---

## 4. 重要な設計原則

RelayKV は短期的には SGLang の prefix cache / residency manager を置き換えない。

役割分担:

```text
RadixTree / RadixAttention:
  prefix KV reuse
  shared prefix cache
  KV cache index management
  request prefix sharing

HiCache / HiRadixTree:
  GPU / CPU / L3 residency management

RelayKV:
  decode-time working-set selector
  residual VRAM budget controller
  RECENT / ANCHOR / RETRIEVED / COLD_CANDIDATE の working set selection
```

短期原則:

```text
Do not modify RadixTree.
Do not modify RadixAttention semantics.
Do not change scheduler decisions.
Do not free KV cache.
Do not write back runtime KV.
Do not issue HiCache prefetch hints.
```

---

## 5. Phase 4.5 で見るべきファイル候補

Codex に調査させるべきファイル候補。

```text
python/sglang/srt/model_executor/model_runner.py
python/sglang/srt/model_executor/forward_batch_info.py

python/sglang/srt/layers/attention/
python/sglang/srt/layers/attention/base_attn_backend.py
python/sglang/srt/layers/attention/flashinfer_backend.py
python/sglang/srt/layers/attention/triton_backend.py
python/sglang/srt/layers/attention/flashattention_backend.py

python/sglang/srt/mem_cache/
python/sglang/srt/mem_cache/memory_pool.py
python/sglang/srt/mem_cache/radix_cache.py
python/sglang/srt/mem_cache/hicache_storage.py
python/sglang/srt/mem_cache/hicache_controller.py

python/sglang/srt/managers/
```

ただし、Phase 4.5 調査では変更しない。

推奨コマンド:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

grep -R "class .*Attention" -n python/sglang/srt/layers/attention python/sglang/srt/models | head -80
grep -R "forward_batch" -n python/sglang/srt/layers/attention python/sglang/srt/models | head -120
grep -R "req_to_token_pool\|token_to_kv_pool\|out_cache_loc\|positions" -n python/sglang/srt | head -160
grep -R "RadixAttention\|RadixTree\|radix" -n python/sglang/srt | head -160
grep -R "HiCache\|HiRadix\|hicache" -n python/sglang/srt | head -160
```

---

## 6. 実 attention 接続前の禁止事項

Phase 4.5 では、以下を禁止する。

```text
attention_override_true_count > 0
actual attention override
attention backend modification
KV pool read
GPU tensor read
KV snapshot
runtime writeback
scheduler decision mutation
KV cache free
RadixTree modification
HiCache prefetch hint
production runtime connection
```

Phase 4.5 は read-only source inspection / design memo / optional metadata-only helper まで。

---

## 7. 接続候補の分類

RelayKV の attention 接続候補は、いきなり本番 override ではなく、以下に分ける。

### 7.1 Design-only

```text
code path を読む
接続候補位置を特定する
変更しない
```

現在ここ。

### 7.2 Shadow-only

```text
既存 attention はそのまま実行
RelayKV は metadata/comparison plan だけを作る
実 output には影響しない
```

許可できる可能性があるもの:

```text
attention_shadow_plan_count > 0
```

まだ禁止:

```text
attention_override_true_count > 0
```

### 7.3 Comparison-only

```text
Full-KV attention と RelayKV working-KV attention の比較を行う
ただし model output には使わない
```

注意:

この段階から、KV pool read / tensor read / attention execution が関係し始める可能性がある。

そのため、Phase 4.5 で直接入るのではなく、別設計として分ける。

候補名:

```text
Phase 4.6 Isolated Attention Comparison Design
```

### 7.4 Guarded override

```text
RelayKV working-KV attention を実 output に使う
fallback 条件が満たされない場合のみ
```

これは Phase 4 後半または Phase 5 で検討すべき。

---

## 8. 推奨される最小安全経路

現時点で最も安全な次ステップは、実 attention 接続ではない。

推奨経路:

```text
Phase 4.5:
  SGLang attention path inspection report

Phase 4.6:
  isolated attention comparison design
  KV pool read / tensor read の許可境界を別に定義

Phase 4.7:
  shadow-only connection smoke
  production output には影響させない

Phase 4.8:
  comparison-only smoke
  full vs relaykv attention compare をログ化

Phase 4.9:
  guarded override design
```

---

## 9. Phase 4.5 の deliverable

Phase 4.5 で作るべきものは、コード変更ではなく調査レポート。

ファイル候補:

```text
notes/relaykv_phase4_5_sglang_attention_path_inspection_2026-05-03.ja.md
```

記載すべき内容:

```text
1. 見たファイル
2. attention backend 呼び出しの入口
3. decode/prefill の分岐
4. ForwardBatch から attention に渡る主要 metadata
5. KV pool index の流れ
6. RadixAttention / RadixTree の責務
7. HiCache / HiRadixTree の責務
8. RelayKV が metadata-only で横に置ける位置
9. RelayKV が実 tensor に触るなら必要な境界
10. 最小の次フェーズ案
```

---

## 10. Codex 調査プロンプト方針

Codex には、まず実装をさせず、調査だけさせる。

重要な制約:

```text
Do not edit files.
Do not run formatting.
Do not modify generated files.
Do not touch runtime code.
Output an inspection report only.
```

調査結果として欲しいもの:

```text
- file/path/function/class
- call flow
- relevant fields
- where KV indices are resolved
- where backend dispatch happens
- candidate hook points
- why each candidate is safe/unsafe
- recommended next phase
```

---

## 11. 期待する調査観点

### 11.1 attention backend dispatch

確認したいこと:

```text
どの class/method が attention backend を呼ぶか
prefill/decode で path が違うか
FlashInfer / Triton / FlashAttention backend の抽象境界
forward_batch がどこまで渡るか
```

### 11.2 KV index / pool flow

確認したいこと:

```text
req_to_token_pool の役割
token_to_kv_pool の役割
out_cache_loc の役割
positions の役割
decode step の query がどの KV index を見るか
```

### 11.3 RadixAttention boundary

確認したいこと:

```text
RadixAttention が prefix cache reuse に関与する場所
RadixTree node span と token span の扱い
decode-time working set selection と干渉する可能性
```

### 11.4 HiCache boundary

確認したいこと:

```text
HiCache が GPU/CPU/L3 residency を決める場所
RelayKV が prefetch hint を出す可能性
短期では触らないべき境界
```

### 11.5 RelayKV hook candidate

候補を分類する。

```text
metadata-only hook
shadow-only hook
comparison-only hook
real override hook
```

それぞれについて:

```text
必要な入力
出力影響の有無
fallback 可能性
安全性
実装コスト
```

---

## 12. Phase 4.5 での判断

Phase 4.5 では、まだ attention 接続を実装しない。

正しい判断は:

```text
SGLang attention path を読んで、RelayKV の hook point を分類する。
```

次に進む条件:

```text
attention backend dispatch の入口が分かる
decode path で KV index がどう渡るか分かる
RadixAttention / HiCache と RelayKV の境界が説明できる
shadow-only / comparison-only の候補位置が1つ以上ある
```

---

## 13. 次の作業名

次の作業名:

```text
RelayKV Phase 4.5 SGLang Attention Path Inspection
```

次に Codex へ依頼するべき内容:

```text
コード変更なしで、SGLang attention path を調査して notes に inspection report を作る。
```

---

## 14. commit command

この設計メモを repo に保存する場合:

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

mkdir -p notes
cp /mnt/data/relaykv_phase4_5_real_attention_connection_design_2026-05-03.ja.md \
  notes/relaykv_phase4_5_real_attention_connection_design_2026-05-03.ja.md

git status --short
git diff --check

git add notes/relaykv_phase4_5_real_attention_connection_design_2026-05-03.ja.md
git commit -m "docs: design relaykv real attention connection phase"
git push mine relaykv-host-backup-shadow
```
