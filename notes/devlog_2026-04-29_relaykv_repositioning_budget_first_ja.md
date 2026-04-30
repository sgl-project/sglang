# Devlog: RelayKV Repositioning as VRAM-Constrained KV Working Set Manager

## 日付

2026-04-29

## 背景

RelayKV の開発方針を更新した。

これまでの実装は、SGLang上で以下を段階的に確認してきた。

```text
MVP-0:
  server args / skeleton / shadow plan / no-op / profile metadata

MVP-1a:
  KV memory estimate / budget sweep / smoke test

MVP-1b:
  host backup flags / candidate metadata / range metadata / KV layout observation / token pool mapping observation / dry-copy guard
```

これらにより、SGLang内部のKV poolやmappingを安全に観測できるようになった。

一方で、RelayKVの実用価値を考えると、次に深掘りすべきなのは実CPU copyではなく、**残VRAM制約下でのKV working set budget管理** である。

## 新しい位置づけ

RelayKV は、単なる KV cache 削減プロトタイプではなく、**VRAM制約を意識したKV working set manager** として位置づける。

```text
RelayKV
= decode時にattention対象となるworking KVを
  残VRAM予算内に収めるためのmanager
```

## 実用上の主要ターゲット

```text
RTX 3060 12GB単体
dense系の量子化モデル
当面の本命: Qwen3.5-9B Q4クラス
将来: Qwen3.6 9B〜12B dense Q4クラス
長期: Qwen3.6 27B dense + 強力な重み/KV量子化
```

## 中心前提

ローカルLLMユーザーは、多くの場合、VRAMに収まる最大級の量子化モデルを載せ、その残りでcontext / KV cacheを伸ばそうとする。

そのため、RelayKVはモデルロード後に残る少ないKV用VRAM予算を最大限活用する仕組みとして設計する。

## 設計目標

```text
- decode時にattention対象となるworking KVを固定予算内に収める
- recent + anchor + retrieved cold blocks の三層構成を基本にする
- full KVは論理的・cold storage的には保持しても、常にGPU attention対象にしない
- TurboQuant / PolarQuant系 compressed KV store と将来接続できるようにする
```

## 評価方針の変更

従来の `seq_len` / `coverage_ratio` 中心の評価に加え、残VRAM制約下でのKV予算管理を明示的に評価する。

主な評価項目:

```text
available_kv_budget_mib
kv_working_budget_tokens
recent_window
anchor_blocks
retrieval_top_k
working_ratio
mean_abs_diff
top5_overlap
first_divergence_step
task_accuracy
same_first_code
```

見るべき問いは、単に「何tokenまで伸ばせるか」ではない。

```text
モデルでVRAMをかなり使った後、
残り512MB / 1GB / 2GB程度のKV予算で、
どのworking KV構成なら品質を保てるか
```

である。

## 実装優先度の変更

次は実CPU copyへ進まず、budget-firstに切り替える。

新しい優先度:

```text
1. KV budget mode を追加または明確化する
2. recent full window を安定保持する
3. anchor blocks を always-on memory として扱う
4. 残り予算内で retrieved cold blocks を選択する
5. RTX 3060 12GB相当の residual KV制約を模擬して評価する
6. budget behavior が明確になってから scoring改善に進む
```

## TurboQuant / PolarQuantとの関係

TurboQuant / PolarQuantは、RelayKVと競合しない。

```text
TurboQuant / PolarQuant:
  モデル重みやKV cacheの表現サイズを小さくする

RelayKV:
  圧縮後も残るdecode時のKV working setを、残VRAM内に収める
```

RelayKVはcold KVの保存形式を決め打ちしない。

短期:

```text
ColdKVStore.get_block(block_id) -> fp16/bf16 K/V block
```

将来:

```text
TurboQuant / PolarQuant compressed KV
  -> selected blockだけdequantize
  -> working KVへ投入
```

## 作業再構築

### 一時停止

```text
actual CPU dry-copy 実装は一旦停止
```

理由:

```text
copy自体よりも、どのworking setを残VRAM内に収めるかが先
```

### 継続利用する成果

以下は引き続き有用。

```text
shadow planner
memory estimate
host backup candidate metadata
range metadata
KV layout observation
token pool mapping observation
prefill final guard
```

### 次に行う作業

```text
RelayKV budget mode / budget planner metadata
```

## 次の具体タスク

### 設計メモ

```text
notes/relaykv_budget_mode_design_ja.md
```

### 実装

budget planner metadataを追加する。

```text
available_kv_budget_mib
kv_working_budget_tokens
recent_window_tokens
anchor_budget_tokens
retrieval_budget_tokens
retrieval_top_k_requested
retrieval_top_k_effective
budget_overflow
budget_policy_reason
```

### smoke test

```text
512MiB / 1024MiB / 2048MiB のbudgetで
working budget tokens と allocation が期待通り変化することを確認
```

## 次のCodexタスク候補

```text
RelayKV budget mode / budget planner metadata を追加してください。

- KV tensorは動かさない
- CPU copyしない
- GPU KV freeしない
- attention変更しない
- scheduler挙動変更しない
- .github/workflowsは触らない
- 既存のshadow log / memory estimate / host backup metadataを壊さない

目的:
available_kv_budget_mib を起点に、recent / anchor / retrieval のworking KV budget配分をmetadataとしてログする。
```
