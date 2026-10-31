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
implementation. `--enable-hi-engram remote` selects node-sharded CPU tables
with DeepEP GPU-initiated RDMA reads.

## Model integration

Supported adapters: **Qwen4-Exp** (including Qwen3.8-Flash-Next PLE N-gram tables)
and **DeepSeek V4.1 Engram**.
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

## Node-sharded remote configuration

Use the Mooncake `feat/engram-sharded-deepep` branch together with this branch;
the base PR's local-only wheel does not implement `initialize`. Configure its
optional DeepEP/DeepJIT/NCCL build dependencies before startup.

For two nodes with eight GPU callers each, append these options to the otherwise
unchanged model launch configuration (set `shard_id` to 1 on the second node):

```bash
--enable-hi-engram remote \
--hi-engram-config '{"local_dir":"/dev/shm/sglang-engram","shard_id":0,"num_shards":2,"num_callers":8,"master_addr":"http://master-host:8080/metadata","session_id":"unique-model-deployment","max_lookup_rows":786432,"rdma_device":"mlx5_bond_{local_rank}","read_policy":"auto"}'
```

A storage shard is **one node**, not a TP/DP/CP rank. The adapter divides complete
hash heads across nodes, loads only the node's physical row range, and passes
the head descriptors and packed CPU rows into EngramStore initialization. Local
caller processes share the final CPU allocation. Checkpoint parameter copies
are released after publication; startup can temporarily hold additional copies.
`caller_id` defaults to the CUDA device index and must be dense in
`[0,num_callers)` on each node. Override it when visible-device numbering differs.
The `rdma_device` template expands `{local_rank}` to the current CUDA device.
Both nodes must provide the same caller IDs, layouts, capacities and session ID.

DeepSeek retains its original `row_start/rows` **computational** ownership and
CP/DP ID/result collectives. Its weight loader separately uses the **storage**
head range; `_owned_rows` still masks every row outside the computational range.
Qwen retains its vocabulary shard mask, collectives, scales and prefetch. Its
checkpoint shard loader uses the storage range only when remote mode is active.
A request can therefore select rows owned by its computational rank while the
corresponding bytes live partly on this node and partly on another node.

All bootstrap, local sharing, peer discovery and readiness waits are inside
EngramStore initialization. The lookup interface is unchanged. Qwen publishes
on its first eager lookup; DeepSeek publishes in `finish_load`. Both must finish
initialization before CUDA graph capture. Remote lookups require CUDA inputs;
local mode retains the CPU lookup overload. Engram weights are immutable after
publication. This change does not add a storage-aware model collective or change
CP/DP scheduling.

`max_lookup_rows` counts all submitted IDs (including masks), not request batch
size. Include speculative draft positions, CP/DP gathered IDs and graph padding
when sizing it. Optional tuning keys are `opt_max_rows` (default 12288),
`native_qps` (15), `opt_qps` (32), and `timeout` (300 seconds). Auto chooses flat
for small submitted row counts and native for larger counts. Tune the crossover
for the actual head placement before drawing throughput conclusions.

Validation for this branch consists of adapter placement/loading checks, packed
FP8 dequantization against the native gather kernel, and Mooncake's real two-node
RDMA/CUDA-graph tests. Full-model remote accuracy and serving throughput have not
been measured on this branch. The historical local results above do not validate
remote inference.
