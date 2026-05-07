# RelayKV / SGLang 探索フェーズ closeout devlog

- Date basis: JST
- Devlog date: 2026-05-07
- Scope: SGLang fork / RelayKV engine-adapter exploration closeout
- Repository: https://github.com/rinsakamo/sglang
- Local dir: `~/work/sglang-relaykv`
- Primary closeout branch: `relaykv-host-backup-shadow`
- Related branches: `relaykv-memory-mvp0`, `relaykv-v0`

## 0. Date confirmation

この devlog は JST 基準で 2026-05-07 の closeout 記録として作成する。

この ChatGPT 実行環境からは `~/work/sglang-relaykv` が見えなかったため、実ローカル repo に対する `git status` / smoke 実行は未実行。以下は、これまでの会話で報告された実装・検証結果に基づく closeout 記録である。ローカルでは末尾の確認コマンドを実行してから commit する。

## 1. Closeout decision

SGLang 側は、現時点で「RelayKV を engine adapter として差し込める見込みを確認した探索フェーズ」として一旦クローズする。

ここから先の以下の領域には、現フェーズでは進まない。

- KV pool mutation
- attention backend connection
- scheduler decision change
- runtime writeback
- real prefetch / materialization
- RadixTree / RadixAttention の改変

理由は、SGLang adapter 側の metadata plumbing / runtime observation / source bridge / bounded-read smoke で、engine adapter として必要な観測・接続境界の見込みは十分確認できた一方、RelayKV policy の品質・勝ち筋がまだ HF Transformers / PyTorch 側で十分に詰め切れていないためである。

今後は、SGLang の深い runtime 接続に進む前に、HF Transformers / PyTorch 側で RelayKV policy の品質、実用 baselines への優位性、固定 working KV budget 下での挙動を再評価する。

## 2. Branch positioning

### relaykv-v0

位置づけ: early exploration branch。

初期の SGLang decode path / Triton attention 近傍に踏み込んだ探索ブランチ。RelayKV が SGLang attention path に接続できる可能性を見るための早期実験としては有用だったが、attention override に踏み込みすぎているため、今後の mainline base にはしない。

扱い:

- 証跡・参考実装として残す。
- 今後の土台にはしない。
- attention backend 接続の参考にする場合も、直接継承せず設計だけ参照する。

### relaykv-memory-mvp0

位置づけ: smaller memory planning / shadow plan base。

RelayKV を memory management / shadow planning として扱う方向に寄せた比較的 clean な土台候補。今後 SGLang adapter に戻る場合、`relaykv-host-backup-shadow` よりも軽い再開 base として検討可能。

扱い:

- 将来の clean adapter 再開候補。
- 現時点では SGLang 側実装をこれ以上進めない。
- HF 側で policy 勝ち筋が見えてから再評価する。

### relaykv-host-backup-shadow

位置づけ: rich evidence / exploration branch。

ForwardBatch read-only observation、runtime observation metadata、host backup copy candidate summary、candidate event summary、req_to_token / token_to_kv_pool 周辺の bounded read / bridge / optional server smoke など、SGLang adapter 境界の証拠が最も多い探索ブランチ。

ただし、smoke / helper / metadata join が増えて重くなっているため、次の mainline base にはしない。

扱い:

- SGLang adapter feasibility の証拠保管ブランチ。
- closeout devlog はこのブランチに置く。
- 将来の再開時は、このブランチから必要な helper / smoke / schema 知見を抽出し、clean branch に移植する。

## 3. Confirmed items

### 3.1 ForwardBatch read-only metadata observation

SGLang runtime の `ForwardBatch` 周辺から、read-only の観測 metadata を取得する方向性を確認した。

確認済みの観測対象:

- request id 系 metadata
- `req_pool_idx`
- `seq_len`
- positions 周辺 metadata
- runtime observation payload / summary

重要な点として、この段階では `ForwardBatch` の runtime metadata を読むだけで、attention、scheduler、KV pool、runtime writeback には接続しない方針を維持した。

### 3.2 runtime observation metadata source bridge

runtime observation metadata を、後段の RelayKV helper / summary で利用できる payload として橋渡しする方向を確認した。

確認済み:

- observation marker / summary
- optional server smoke での clean skip path
- env off 時の no-op 挙動
- response marker unchanged

### 3.3 host backup copy candidate summary

host backup copy candidate 側の summary / counters を整備し、runtime observation metadata と candidate summary の join に進める見込みを確認した。

確認済み:

- candidate summary
- candidate copy event summary
- applied / fallback / skipped reason counters
- policy state counters
- per-layer / per-request counters
- safety counters

### 3.4 runtime observation metadata と candidate summary の join

runtime observation metadata と host backup copy candidate summary を join し、RelayKV adapter 側で request-level / layer-level の candidate 情報を扱う見込みを確認した。

この join は、将来の RelayKV Core / SGLang Adapter 境界で、engine-specific runtime observation を engine-independent RelayKV metadata に寄せるための足場になる。

### 3.5 candidate copy event summary / counters

candidate copy event payload を読み、以下を集計できることを確認した。

- total candidate events
- applied / fallback count
- snapshot / copy count
- no-op count
- safety flag true / false count
- skipped reason counts
- policy state counts
- per-layer counts
- per-request counts

この段階でも、実 KV pool mutation / runtime writeback / scheduler decision change には接続していない。

### 3.6 optional server smoke / clean skip path

optional server smoke 群で、重い server 実行を任意化し、env / model がない場合は clean skip する運用を確認した。

確認済みの性質:

- `RELAYKV_OPTIONAL_SERVER_SMOKE_RUN` がない場合は clean skip
- `RELAYKV_OPTIONAL_SERVER_SMOKE_MODEL` がない場合は clean skip
- 実 server smoke 実行時も response marker unchanged を確認
- env off 時は RelayKV summary を出さない
- observation only / bounded read / bridge / live-index などの段階的 case split が可能

### 3.7 real req_to_token_pool bounded read / scalar tensor conversion

実 server runtime 上で `req_to_token_pool` に到達し、CUDA 上の 0-d scalar tensor 的な値を、明示 flag 下の `.item()` conversion に限定して Python int として扱えることを確認した。

確認済み:

- `model_runner.req_to_token_pool.req_to_token` source path に到達
- indexed result は `Tensor`, shape `[]`, dtype `torch.int32`, device `cuda:0`
- default では conversion 無効
- `SGLANG_RELAYKV_REAL_REQ_TO_TOKEN_POOL_SCALAR_ITEM_CONVERSION=1` のときのみ scalar-like / one-element-like への `.item()` conversion を許可
- resolved_count > 0
- scalar_tensor_item_conversion_succeeded_count > 0
- `req_to_token_payload_attached=true`
- response marker unchanged

### 3.8 req_to_token payload bridge

実 server runtime 上で、resolved req_to_token payload を bridge し、後段の live token_to_kv_pool index-read helper に渡すところまで確認した。

確認済み:

- `req_to_token_resolution_bridge_enabled=true`
- `req_to_token_resolution_bridge_state="bridged"`
- `req_to_token_resolution_bridge_payload_count=1`
- `req_to_token_resolution_bridge_valid_count=1`
- `req_to_token_resolution_bridge_source_path="forward_batch.relaykv_req_to_token_resolution_payloads"`
- response marker unchanged

### 3.9 live token_to_kv_pool source lookup

実 server runtime 上で、bridged req_to_token payload から live token_to_kv_pool index-read path に進み、`model_runner.token_to_kv_pool` source path までは到達した。

確認済み:

- old blocker `token_to_kv_pool_read_not_performed_after_bridged_req_to_token_payload` は解消
- `token_to_kv_pool_source_path="model_runner.token_to_kv_pool"`
- `token_to_kv_pool_lookup_error=false`
- response marker unchanged

現時点の blocker:

- `token_to_kv_pool_object_not_indexable_after_bridged_req_to_token_payload`

これは、source path 自体は見つかっているが、現 helper が想定する `dict/list/tuple/__getitem__` 的 interface では直接 index できないことを意味する。

### 3.10 KV pool object interface inspection smoke

metadata-only の `token_to_kv_pool` object/interface inspector を追加し、fake smoke では以下を確認済み。

確認済み:

- type / module / qualname
- shape / dtype / device の shallow metadata
- `__getitem__` presence
- selected method-name presence
- selected shallow attr-name presence
- candidate next source paths

禁止維持:

- token_to_kv_pool values read
- indexing into token_to_kv_pool
- KV pool read / snapshot
- K/V buffer read
- `.cpu()` / `.tolist()` / `.item()` / `.numpy()`
- `dir()` / `vars()` / `repr()`
- recursive traversal
- mutation

optional server smoke では、現時点で `SGLANG_RELAYKV_TOKEN_TO_KV_POOL_OBJECT_INTERFACE_INSPECTION=1` を ModelRunner.forward から呼ぶ default-off hook が未配線であることを確認した。

確認済みの分岐:

- `inspection_hook_wired=false`
- no interface-inspection summary emitted
- response marker unchanged

## 4. Closeout boundary

この closeout 時点で、以下は明示的に実装しない。

- RadixTree / RadixAttention の改変
- KV pool mutation
- attention backend connection
- scheduler decision change
- runtime writeback
- real prefetch
- real materialization
- working KV assembly の実 runtime 接続
- K/V buffer read
- KV pool snapshot
- attention comparison execution in server runtime
- production behavior change

RelayKV の SGLang integration は、現時点では read-only metadata observation / source bridge / bounded index-read smoke までに限定する。

## 5. Why stop SGLang here

SGLang 側で進めるほど、次の実装対象は以下のように runtime critical path に近づく。

- token_to_kv_pool object の内部 accessor
- physical KV index 解決
- KV pool / K/V tensor 参照
- working KV materialization
- attention backend connection
- scheduler / memory manager との調停

しかし RelayKV policy そのものの品質、つまり固定 working KV budget 下で FullKV / sliding window / truncation / recent+anchor / RAG-like baseline に対して十分な実用価値が出るかは、まだ HF Transformers / PyTorch 側で再評価する余地が大きい。

したがって、SGLang 側は「engine adapter として差し込める見込みを確認した探索フェーズ」として閉じる。先に HF / PyTorch で policy の勝ち筋を確認し、その後に SGLang / vLLM adapter に戻る方が手戻りが少ない。

## 6. Remaining unimplemented work

SGLang 側に戻る場合の未実装項目:

1. `SGLANG_RELAYKV_TOKEN_TO_KV_POOL_OBJECT_INTERFACE_INSPECTION=1` の default-off ModelRunner hook wiring
2. 実 server 上の `model_runner.token_to_kv_pool` actual type / shallow attrs / candidate source paths の確認
3. token_to_kv_pool object の安全な明示 accessor / path の特定
4. bridged req_to_token payload から physical KV index への read-only 解決
5. safe working-KV assembly dry-run
6. shadow attention compare
7. gated attention connection
8. residual VRAM budget control
9. SGLang adapter / RelayKV Core schema の整理
10. vLLM adapter への横展開

ただし、1〜4 も HF 側で policy 勝ち筋が見えた後に再開する。

## 7. Next phase: HF Transformers / PyTorch policy evaluation

次は SGLang runtime ではなく、HF Transformers / PyTorch 側で RelayKV policy の品質検証に戻る。

重点:

- Qwen2.5-Coder-7B Q4 / Qwen3-8B Q4 など、FullKV baseline が成立する程度のモデルで評価する
- 小型モデルで FullKV 自体が失敗する問題を避ける
- coding / structured retrieval / long context lookup 系タスクで評価する
- theoretical reference と practical baseline を分ける
  - reference: FullKV
  - practical baseline: sliding window, truncation, recent+anchor, summary compression, RAG-like retrieval, engine default compression
- RelayKV は FullKV 完全一致ではなく、固定 VRAM budget 下で practical baseline を上回ることを狙う
- SGLang は HF 側で勝ち筋が見えた後、engine adapter として戻る

## 8. Suggested local verification commands before commit

この ChatGPT 実行環境では `~/work/sglang-relaykv` が見えなかったため未実行。ローカル環境で以下を実行してから commit する。

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

git status
git branch -vv
git log --oneline --decorate -n 10
git remote -v
```

軽い smoke の例:

```bash
PYTHONPATH=python .venv/bin/python scripts/relaykv_sglang_adapter_schema_alignment_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_req_to_token_resolution_payload_bridge_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_live_token_to_kv_pool_index_read_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_token_to_kv_pool_object_interface_inspection_smoke.py
```

optional server smoke は重いため、closeout commit 前の必須確認にはしない。実行する場合のみ以下のように任意で行う。

```bash
RELAYKV_OPTIONAL_SERVER_SMOKE_MODEL=/path/to/local/model \
RELAYKV_OPTIONAL_SERVER_SMOKE_RUN=1 \
PYTHONPATH=python .venv/bin/python \
  scripts/relaykv_optional_server_token_to_kv_pool_object_interface_inspection_smoke.py
```

## 9. Commit / push

```bash
cd ~/work/sglang-relaykv
git switch relaykv-host-backup-shadow

git status --short
git diff --check

git add notes/devlog_2026-05-07_relaykv_sglang_closeout_ja.md

git commit -m "Document RelayKV SGLang closeout"
git push -u mine $(git branch --show-current)
```

## 10. Handoff summary

SGLang exploration confirmed that RelayKV can be structured as an external engine adapter with read-only runtime observation, metadata bridge, bounded index-read smoke, and optional server validation. The remaining SGLang work now approaches engine-critical runtime behavior, so it should pause until HF Transformers / PyTorch policy quality demonstrates a practical win under fixed KV working-set budgets.

SGLang should be revisited later as an adapter target, not as the place to continue policy discovery.
