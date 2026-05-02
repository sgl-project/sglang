# Devlog: RelayKV ForwardBatch CPU metadata inventory on observation skip

Date: 2026-05-02 JST  
Branch: `relaykv-host-backup-shadow`  
Repo: `~/work/sglang-relaykv`

## 今日の目的

RelayKV runtime observation hook で、実 `ForwardBatch` の tensor-like metadata が skip される場合に、既存 `forward_batch` 属性だけを使って CPU 側 metadata inventory を観測する。

この段階では、まだ以下には進まない。

- `ForwardBatch.init_new()` 変更
- `ScheduleBatch` / `Req` / scheduler 変更
- tensor値の読み取り
- `.cpu()` / `.item()` / `.tolist()`
- KV pool 参照
- KV snapshot
- host backup copy
- attention 接続
- runtime writeback

## 背景

前段階で、実server経路では以下が確認済みだった。

```text
SGLANG_RELAYKV_RUNTIME_OBSERVATION=1:
  RelayKV runtime observation hook に到達
  実 ForwardBatch の req_pool_indices / seq_lens は torch.Tensor
  payload builder は list/tuple 以外を拒否
  TypeError skip
  relaykv_runtime_observation_skip を検出
  metadata_description を検出
```

観測された実 `ForwardBatch` metadata:

```text
req_pool_indices:
  type=torch.Tensor
  shape=torch.Size([1])
  device=cuda:0
  dtype=torch.int64

seq_lens:
  type=torch.Tensor
  shape=torch.Size([1])
  device=cuda:0
  dtype=torch.int64
```

また、調査で以下が分かった。

- `ForwardBatch.rids` は list として存在する可能性が高い
- `ForwardBatch.seq_lens_cpu` は CPU tensor として渡っている
- `extend_seq_lens_cpu` / `extend_prefix_lens_cpu` は prefill/extend では list のまま存在する可能性がある
- `req_pool_idx` は `Req` 側にはあるが、今回は `ForwardBatch.init_new()` を触らないため対象外

## 変更したファイル

- `python/sglang/srt/relaykv/observation.py`
- `scripts/relaykv_optional_server_observation_smoke.py`
- `scripts/relaykv_fake_model_runner_forward_observation_smoke.py`
- `scripts/relaykv_model_runner_observation_hook_smoke.py`

## 実装内容

### 1. `forward_batch_cpu_metadata_description` の追加

`relaykv_runtime_observation_skip` に、既存 `forward_batch` 属性だけから作る `forward_batch_cpu_metadata_description` を追加した。

対象:

- `rids`
- `seq_lens_cpu`
- `extend_seq_lens_cpu`
- `extend_prefix_lens_cpu`

各 field では、既存 `metadata_description` と同じく以下だけを記録する。

- `type_name`
- `type_module`
- `type_qualname`
- `is_list_or_tuple`
- `list_or_tuple_len`
  - list/tuple の場合のみ
- `has_shape`
- `shape_repr`
- `has_device`
- `device_repr`
- `has_dtype`
- `dtype_repr`

### 2. 値読み取り・GPU同期を避ける方針

以下は禁止のまま。

- `.cpu()`
- `.item()`
- `.tolist()`
- `int(tensor)`
- `len(tensor)`
- `iter(tensor)`
- tensor indexing
- numpy conversion

tensor-like object には `len()` / iteration / indexing を呼ばない。

list/tuple の `len()` だけは許可する。

### 3. poison object smoke の拡張

fake / hook smoke の poison object を拡張し、以下が呼ばれたら失敗することを確認した。

- `.cpu()`
- `.item()`
- `.tolist()`
- `__iter__()`
- `__len__()`
- `__getitem__()`

対象を `req_pool_indices` だけでなく、`seq_lens_cpu` / `extend_seq_lens_cpu` などの CPU metadata inventory 側にも広げた。

## env 動作

### env off

`SGLANG_RELAYKV_RUNTIME_OBSERVATION` が unset または `0` の場合:

- `enabled=false`
- `skip_reason=env_disabled`
- RelayKV observation log なし
- `metadata_description` なし
- `forward_batch_cpu_metadata_description` なし

### env on + tensor-like metadata

`SGLANG_RELAYKV_RUNTIME_OBSERVATION=1` の場合:

- hook 到達
- `req_pool_indices` / `seq_lens` は tensor-like のため payload builder が `TypeError` skip
- `relaykv_runtime_observation_skip` を出す
- `metadata_description` を含める
- `forward_batch_cpu_metadata_description` を含める
- payload summary は出ない
  - `relaykv_summary_logged=false` は想定内

## 確認結果

以下は pass。

```bash
PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/relaykv/observation.py \
  scripts/relaykv_optional_server_observation_smoke.py \
  scripts/relaykv_fake_model_runner_forward_observation_smoke.py \
  scripts/relaykv_model_runner_observation_hook_smoke.py

PYTHONPATH=python .venv/bin/python scripts/relaykv_optional_server_observation_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_fake_model_runner_forward_observation_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_model_runner_observation_hook_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_observation_summary_smoke.py
PYTHONPATH=python .venv/bin/python scripts/relaykv_runtime_policy_smoke.py

git diff --check
git diff --name-only | grep -E 'scheduler.py|attention|flashinfer|\.github/workflows' || true
```

結果:

- `py_compile`: pass
- optional server smoke model未設定 clean skip: pass
- fake model runner forward observation smoke: pass
- model runner observation hook smoke: pass
- runtime observation summary smoke: pass
- runtime policy smoke: pass
- `git diff --check`: pass
- 制約 grep: 出力なし

## 任意server smoke: ninja インストール後の再実行

ninja インストール後、ローカルモデル指定の optional server smoke を再実行した。

使用 model path:

```text
/home/rinsa/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
```

実行コマンド:

```bash
RELAYKV_OPTIONAL_SERVER_SMOKE_MODEL=/home/rinsa/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 \
RELAYKV_OPTIONAL_SERVER_SMOKE_RUN=1 \
PYTHONPATH=python .venv/bin/python scripts/relaykv_optional_server_observation_smoke.py
```

### env on case

env on case では以下を確認した。

```text
forward_completed=true
http_status=200
relaykv_skip_logged=true
relaykv_metadata_description_logged=true
relaykv_cpu_metadata_description_logged=true
```

実server観測:

```text
req_pool_indices:
  torch.Tensor
  device=cuda:0
  dtype=torch.int64

seq_lens:
  torch.Tensor
  device=cuda:0
  dtype=torch.int64

seq_lens_cpu:
  torch.Tensor
  device=cpu
  dtype=torch.int64
  shape=torch.Size([1])

rids:
  list

extend_seq_lens_cpu:
  list

extend_prefix_lens_cpu:
  list
```

つまり、実server経路で hook 到達、skip log、metadata description、CPU metadata inventory まで確認できた。

### env off case

env off case では以下。

```text
relaykv_observation_logged=false
relaykv_skip_logged=false
```

RelayKV hook log は出ていない。

server は起動し `/v1/models` までは 200 だったが、`/generate` が timeout し、script が SIGTERM した。

これは RelayKV hook 起因ではない可能性が高い。

## 判断

今回の到達点:

```text
env off:
  RelayKV observation log なし
  default-off 維持

env on:
  実server経路で hook 到達
  TypeError skip
  metadata_description 検出
  forward_batch_cpu_metadata_description 検出
```

重要な観測:

```text
seq_lens_cpu:
  ForwardBatch 上に CPU tensor として存在

rids:
  list として存在

extend_seq_lens_cpu / extend_prefix_lens_cpu:
  list として存在
```

これにより、少なくとも `seq_lens` については GPU tensor を `.cpu()` しなくても、CPU側 metadataを使える可能性が高まった。

一方、`req_pool_idx` はまだ ForwardBatch 上の CPU値としては未確認。

`req_pool_idx` は `Req` 側には存在するが、`ForwardBatch.init_new()` へ metadata として持ち込むかどうかは次段階の設計論点。

## 触っていない領域

以下には差分なし。

- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/schedule_batch.py`
- attention backend
- `memory_pool.py`
- flashinfer
- `.github/workflows`
- `ForwardBatch.init_new()`
- `ModelRunner._forward_raw()`

## 維持できている安全境界

以下は維持。

- default-off
- payload-only
- summary-only
- skip-safe
- env off では CPU metadata inventory を作らない
- env on でも tensor-like metadata は変換せず skip
- tensor-like object に対して値を読まない
- GPU 同期しない
- hook 例外で forward/server を止めない
- model download は必須にしない
- optional smoke は model未設定なら clean skip

## まだ未達

- request単位の payload 化
- `req_pool_idx` の CPU metadata としての取得
- `seq_lens_cpu` の値読み取り方針の決定
- KV pool 参照
- KV snapshot
- host backup copy
- attention 接続
- scheduler 連携
- runtime writeback

## 次段階の論点

ここから先は、「値を読まない観測」から「CPU値を使う設計」に境界が変わる。

次に設計すべきこと:

1. `rids` はそのまま runtime observation payload に使えるか
2. `seq_lens_cpu` の値読み取りを env/debug 限定で許可するか
3. `req_pool_idx` を `Req` 側から `ForwardBatch` に read-only metadata として渡すべきか
4. `ForwardBatch.init_new()` を触るリスクをどう抑えるか
5. host backup copy に最低限必要な key は何か
6. `req_pool_indices.cpu()` を使う案と、`Req.req_pool_idx` を事前に持ち込む案の差分
7. 実装前に fake smoke でどの形を固定するか

## 次段階候補

次はまだ host backup copy 接続ではない。

おすすめは、以下の設計メモまたは最小 fake helper から始めること。

```text
RelayKV runtime observation minimal CPU metadata design:
  - rids: ForwardBatch既存listを利用
  - seq_lens_cpu: CPU tensorとして存在。値読み取り可否を設計
  - req_pool_idx: Req側からForwardBatchへ read-only list として持ち込む案を検討
  - host backup copyに必要な最小metadataを定義
```

実装へ進む場合でも、最初は `ForwardBatch.init_new()` 変更ではなく、設計メモと fake smoke で expected schema を固定するのが安全。
