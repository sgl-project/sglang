# Devlog: RelayKV runtime ForwardBatch metadata description

Date: 2026-05-02 JST  
Branch: `relaykv-host-backup-shadow`  
Repo: `~/work/sglang-relaykv`

## 今日の目的

RelayKV runtime observation hook について、実server経路で `ForwardBatch` metadata がどの形で届いているかを、同期なし・read-only で観測する。

この段階では、まだ以下には進まない。

- KV pool 参照
- KV snapshot
- host backup copy
- attention 接続
- scheduler 変更
- runtime writeback
- tensor値の読み取り
- tensor の list 化

## 背景

前段階で、実server経路において env on の場合に RelayKV runtime observation hook へ到達し、以下の skip log を検出できた。

```text
relaykv_runtime_observation_skip={"forward_pass_id": 1, "reason": "TypeError"}
```

この `TypeError` は、実 `ForwardBatch` の `req_pool_indices` / `seq_lens` が list/tuple ではなく tensor-like object として届き、payload builder が list/tuple 以外を拒否するために発生していた。

この挙動自体は安全設計どおり。

今回の目的は、payload 化や値取得に進まず、skip 時に型・shape・device・dtype だけを記録すること。

## 変更したファイル

- `python/sglang/srt/relaykv/observation.py`
- `scripts/relaykv_optional_server_observation_smoke.py`
- `scripts/relaykv_fake_model_runner_forward_observation_smoke.py`
- `scripts/relaykv_model_runner_observation_hook_smoke.py`

## 実装内容

### 1. runtime metadata description の追加

skip log に `metadata_description` を追加した。

記録対象:

- `rids`
- `req_pool_indices`
- `seq_lens`
- `layer_ids`

記録する情報:

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

### 2. 禁止した操作

metadata description では、以下を行わない。

- `value.cpu()`
- `value.item()`
- `value.tolist()`
- `int(tensor)`
- `len(tensor)`
- `iter(tensor)`
- tensor indexing
- shape 各要素の `int()` 変換
- numpy conversion
- KV pool 参照
- snapshot
- host backup copy
- attention 接続
- scheduler 変更
- runtime writeback

list/tuple の場合にのみ `len(value)` を許可する。

tensor-like object では `shape` / `device` / `dtype` の属性を安全に文字列化するだけに留める。

### 3. poison object による確認

fake / hook smoke に、以下が呼ばれたら失敗する poison object を追加・拡張した。

- `.cpu()`
- `.item()`
- `.tolist()`
- `__iter__()`
- `__len__()`
- `__getitem__()`

この poison object を使い、metadata description が tensor-like object に対して値読み取り・同期・iteration・indexing を行わないことを確認した。

## env 動作

### env off

`SGLANG_RELAYKV_RUNTIME_OBSERVATION=0` または unset の場合:

- RelayKV observation helper は実行しない
- metadata description は作らない
- `env_disabled` skip のみ
- 実serverでは RelayKV observation log なし

### env on + list/tuple metadata

- payload builder が payload を作る
- summary helper が summary を出す
- safety counters は all zero

### env on + tensor-like metadata

- payload builder は list/tuple 以外を拒否
- `TypeError` skip
- `relaykv_runtime_observation_skip` に `metadata_description` を含める
- 値読み取りや GPU 同期は行わない

## 実server任意 smoke 結果

使用 model path:

```text
/home/rinsa/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
```

実server経路の env on case で、以下を確認した。

```text
relaykv_skip_logged=true
relaykv_metadata_description_logged=true
```

観測された metadata:

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

任意server smoke 全体としては、この環境では `/generate` 後に flashinfer JIT が ninja 不在で落ちるため exit 1 になる。

ただし、RelayKV observation hook 到達と metadata description log は確認できた。

この flashinfer JIT / ninja 不在の失敗は、RelayKV hook 起因ではない。

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

## 触っていない領域

以下には差分なし。

- `python/sglang/srt/model_executor/model_runner.py`
- `scheduler.py`
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
- env off では metadata description を作らない
- env on でも tensor-like metadata は変換せず skip
- tensor-like object に対して値を読まない
- GPU 同期しない
- hook 例外で forward/server を止めない
- model download は必須にしない
- optional smoke は model未設定なら clean skip

## 現在の到達点

```text
ModelRunner.forward default-off hook:
  実装済み

fake forward smoke:
  pass

optional server smoke:
  env off:
    RelayKV observation logなし

  env on:
    hook到達
    TypeError skip
    metadata_description検出

実ForwardBatch:
  req_pool_indices:
    torch.Tensor
    shape=torch.Size([1])
    device=cuda:0
    dtype=torch.int64

  seq_lens:
    torch.Tensor
    shape=torch.Size([1])
    device=cuda:0
    dtype=torch.int64
```

## まだ未達

- 実serverで payload summary を出すこと
- request単位の payload 化
- tensor値取得
- KV pool 参照
- KV snapshot
- host backup copy
- attention 接続
- scheduler 連携
- runtime writeback

## 判断

今回の変更は、host backup copy 接続前に必要な runtime 観測として有効。

実serverの `ForwardBatch` metadata は `torch.Tensor` として届くため、request単位の payload 化に進むには、どこかで `req_pool_indices` / `seq_lens` の実値が必要になる可能性が高い。

ただし、それを `.cpu()` / `.item()` / `.tolist()` で読むと同期コストが発生しうる。

したがって、次段階では実装に進む前に、値取得方針を設計する必要がある。

## 次段階の設計論点

次に検討すべき論点:

1. `req_pool_indices` / `seq_lens` の値を読むか
2. 読む場合、どこで読むか
3. debug/env限定で同期を許容するか
4. scheduler側または request管理側にCPU metadataが既にあるか
5. ForwardBatch生成前の情報を使えるか
6. 値読み取りを実server smoke限定にするか
7. 将来のhost backup copy接続に必要な最小metadataは何か

## 次段階候補

次もまだ host backup copy 接続ではない。

候補:

```text
A. req_pool_indices / seq_lens の値取得方針を設計する
B. ForwardBatch 生成前後のCPU metadata経路を調査する
C. debug/env限定で tensor値を読む optional observation mode を設計する
```

おすすめは B。

理由は、いきなり `.cpu()` / `.tolist()` を許可するより、SGLang内部に既存のCPU側 request metadata があるかを探す方が安全なため。

host backup copy / KV snapshot / attention 接続は、その後に再判断する。
