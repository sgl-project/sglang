# HiCache Engram storage

HiCache provides a unified storage interface for immutable Engram tables.
Models supply integer row IDs and interpret the returned bytes; storage backends
own the tables and perform the reads. `HiCacheEngramStorage` defines this contract
through `initialize`, `lookup` and `close`, independently of the KV page/key
interface. Engram storage does not require `--enable-hierarchical-cache` or a
host KV pool.

The current backend, `MooncakeEngramStorage`, uses Mooncake EngramStore to own
local host snapshots and read rows on the caller's CUDA stream. Enable it with
`--enable-hi-engram local`. Omitting this option preserves each model's native
implementation. `remote` is a reserved mode and fails when the Engram component is constructed:
Mooncake's remote table API does not yet support CUDA stream/graph lookup.

## Model integration

Currently supported: **Qwen4-Exp**, including Qwen3.8-Flash-Next PLE N-gram tables.
Additional models can use the same byte-row storage contract.

For vocab-parallel tables, `HiCacheVocabParallelEmbedding` preserves checkpoint
loading, TP shard masking, dtype conversion and the embedding/prefetch interface.
It delegates row access to `HiCacheEngramStorage` through
`sglang.srt.mem_cache.engram.create_storage`. These tensor and model concerns
stay outside `MooncakeEngramStorage`, so the storage backend does not depend on
embedding dtypes or distributed model state. Qwen4-Exp selects the embedding
class directly from `enable_hi_engram`; hashing, scales, projections, request
history, collectives and prefetch scheduling remain in the model.

## Usage

Install the CUDA-enabled wheel from
[Mooncake #4701](https://github.com/kvcache-ai/Mooncake/pull/4701), which supplies
`mooncake.engram.EngramStore` and `mooncake._engram_cuda`. The ordinary Mooncake KV
backend does not provide this table API. Native execution does not import it.

Start a supported model with local HiCache Engram storage:

```bash
python -m sglang.launch_server \
  --model-path /models/Qwen3.8-Flash-Next-FP8 \
  --tp-size 4 --ep-size 4 \
  --ple-offload-embedding \
  --enable-hi-engram local \
  --hi-engram-config '{"local_dir": "/dev/shm/sglang-engram"}'
```

Local HiCache Engram storage keeps the table on the host even when native PLE
offloading is disabled. `--ple-offload-embedding` enables Qwen4-Exp's existing
side-stream prefetch. `--ple-offload-backend` and `--ple-offload-dir` configure the
native offloader. HiCache accepts storage options as a JSON object through
`--hi-engram-config`. Currently supported: `local_dir`, which defaults to
`/dev/shm/sglang-engram` when the configuration or key is omitted. Malformed JSON,
unknown keys and invalid directory values are rejected when the Engram component
is constructed, before allocating or loading its embedding weights.

## Ownership and ordering

The first eager lookup publishes an immutable copy of the loaded CPU weights.
Each table and TP rank owns an anonymous snapshot (`persistent=False`), so
worker exit leaves no named table files and instances cannot reuse stale weights.
After successful publication, the adapter releases the checkpoint parameter's
CPU storage; only the snapshot retains the table data. Initialization temporarily
holds both copies. CUDA lookup pins the snapshot in full. This is not an
on-demand host page cache; a disk directory does not eliminate the RAM requirement.

Lookup accepts int32/int64 IDs and returns byte rows on the same device. ID -1
produces zeros for masked shards. CPU lookup is synchronous; CUDA lookup enqueues
on the caller's current stream. Consumers on another stream must wait, as Qwen's
prefetch path already does. Warm up every table and CUDA lookup shape before graph
capture. Destroy graphs and finish outstanding GPU work before closing storage.

Tables cannot be updated after publication. Restart the engine when changing PLE
weights; live weight reload is not supported for this backend.

## Validation

Validation results and experiment settings are recorded in the PR. The standalone
benchmark and integration-check scripts are maintained with the experiment
artifacts outside the SGLang source tree.

Compare native and HiCache execution in fresh processes with identical inputs,
TP/EP, host offloading, KV capacity and graph settings. Check output tokens and
logprobs under controlled scheduling; measure latency and throughput separately.
