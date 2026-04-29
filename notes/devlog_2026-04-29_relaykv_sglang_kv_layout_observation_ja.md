# Devlog: RelayKV SGLang MVP-1b-2c KV Layout Observation

## 日付

2026-04-29

## 対象repo / branch

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-host-backup-shadow`
- remote: `mine/relaykv-host-backup-shadow`

## 今回の目的

RelayKV MVP-1b-2c として、host backup dry-copy の前準備として SGLang の KV layout metadata を観察した。

この段階でも、CPU copy は行っていない。

## 守った制約

- KV tensor は copy しない
- CPU copy はしない
- GPU KV は free しない
- tensor content は読まない
- attention kernel は変更しない
- scheduler の実挙動は変えない
- 通常生成結果は変えない
- `.github/workflows` は触らない

## 実行条件

```text
model: Qwen/Qwen2.5-1.5B-Instruct
mode: shadow
resident_budget_tokens: 1024
recent_window: 768
anchor_pages: 4
host_backup_shadow: true
host_backup_max_mib: 0.0
host_backup_dry_copy: true
```

## 確認ログ

```text
kv_layout_observed: true
kv_layout_object_type: MHATokenToKVPool
kv_layout_k_shape: [160824, 2, 128]
kv_layout_v_shape: [160824, 2, 128]
kv_layout_dtype: torch.bfloat16
kv_layout_device: cuda:0
kv_layout_num_layers_observed: 28
kv_layout_range_mapping_supported: true
kv_layout_range_mapping_reason: page_size_1_metadata_only
kv_layout_reason: ok
```

## 解釈

SGLang上で、Qwen2.5-1.5B-Instruct の KV pool は以下のように観測された。

```text
object type: MHATokenToKVPool
K shape: [160824, 2, 128]
V shape: [160824, 2, 128]
dtype: torch.bfloat16
device: cuda:0
layers observed: 28
```

`[160824, 2, 128]` は、おそらく token pool capacity × KV heads × head dim の形。

今回の model profile と一致している。

```text
num_key_value_heads = 2
head_dim = 128
dtype = bfloat16
num_layers = 28
```

## range metadata

同じログで以下も確認。

```text
resident_anchor_ranges: [[0, 4]]
resident_recent_ranges: [[1767, 2535]]
cold_candidate_ranges: [[4, 1767]]
host_backup_copy_target_ranges: [[4, 1767]]
host_backup_copy_target_tokens: 1763
host_backup_copy_target_reason: metadata_only_no_tensor_copy_range_token_mismatch
```

planned cold tokens は 1511。

```text
planned_cold_tokens: 1511
host_backup_copy_target_tokens: 1763
```

この mismatch は想定内。

```text
planned_cold_tokens:
  budget-based

host_backup_copy_target_tokens:
  range-based middle span
```

## dry-copy guard 状態

```text
host_backup_dry_copy: true
host_backup_dry_copy_guard_ok: true
host_backup_dry_copy_would_run: true
host_backup_dry_copy_reason: guard_only_no_tensor_copy
host_backup_would_copy: false
```

意味:

```text
dry-copyを実行可能な条件は満たしている。
ただし MVP-1b-2c ではまだ guard / observation only のため、実copyはしない。
```

## API確認

```text
POST /v1/chat/completions HTTP/1.1" 200 OK
```

通常生成経路は壊れていない。

## 現在の到達点

```text
MVP-0:
  server args / skeleton / shadow plan / no-op / profile metadata 完了

MVP-1a:
  KV memory estimate / budget sweep / smoke test 完了

MVP-1b-0:
  host backup shadow flags / config 完了

MVP-1b-1:
  host backup candidate metadata log / runtime verification 完了

MVP-1b-2a:
  cold/resident range schema / copy target discovery log 完了

MVP-1b-2b:
  dry-copy flag / guard-only path / runtime verification 完了

MVP-1b-2c:
  KV layout observation / runtime verification 完了
```

## 次にやること

次は **MVP-1b-2d: explicit CPU dry-copy prototype** を検討できる段階に入った。

ただし、いきなり本格化せず、最初のdry-copyは以下に限定する。

```text
- 明示flagがtrue
- guard_okがtrue
- budget_okがtrue
- layout_observedがtrue
- range_mapping_supportedがtrue
- 1 requestにつき1回
- CPU copyするだけ
- GPU KVはfreeしない
- attentionには使わない
- copy time / copied bytes / success/failureのみlog
```

## 重要な注意点

`MHATokenToKVPool` の shape が `[160824, 2, 128]` として観測されたが、実際の request token index と pool index の対応はまだ未確認。

したがって MVP-1b-2d に進む前に、以下を確認する必要がある。

```text
- request logical token index -> token pool index の対応
- req_to_token_pool の参照方法
- copy target ranges を token pool index に変換できるか
- chunked prefill完了後に対象KVが存在するか
```

## 推奨する次ステップ

本当のCPU copy前に、次を挟むのが安全。

### MVP-1b-2c+ token pool mapping observation

```text
- req_to_token_pool / token_to_kv_pool の関係を観察
- request_idに対応する token indices を取得できるか確認
- logical range -> pool indices の変換可否をログ
- tensor content は読まない
- CPU copy はしない
```

その後に MVP-1b-2d dry-copy prototype へ進む。

## 次のCodexタスク候補

```text
RelayKV MVP-1b-2c+ として、request logical token range から KV pool index への mapping observation を追加してください。

重要:
- KV tensor はcopyしない
- CPU copyしない
- GPU KVをfreeしない
- tensor contentを読まない
- attention / scheduler behavior は変えない
- .github/workflows は触らない

実装内容:
- dry-copy guard true かつ kv_layout_observed true のときだけ動く
- req_to_token_pool などから request token indices を観察
- logical token range -> pool index range/list の変換可否をmetadata logする
- 取得できない場合は reason を出してno-op
```
