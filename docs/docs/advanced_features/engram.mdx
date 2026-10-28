---
title: "Engram Tables"
metatags:
    description: "SGLang Engram tables: choose where DeepSeek-V4.1's Engram n-gram tables live (GPU or pinned host memory), how many copies there are, and how a rank reads rows another rank holds, with --engram-config."
---

DeepSeek-V4.1 adds an **Engram** memory to two of its layers: the preceding tokens are hashed
into two large static FP8 tables, and the rows they select are gated into the hidden state. The
tables are only ever looked up, never computed over, and for DeepSeek-V4.1-Flash they hold about
189 GiB together, more than any other block of weights. Where they live decides how much memory
is left for the KV cache, so SGLang lets you choose it with one argument, `--engram-config`.

Every choice below returns the same rows; only memory and speed change.

## Usage

`--engram-config` takes a JSON object. Each key answers one question, and omitted keys keep their
defaults:

| Key | Question | Values |
|---|---|---|
| `placement` | Where do the rows live? | `gpu` (default): device memory. `cpu`: pinned host memory, which the GPU reads in place; the HBM goes to the KV cache. |
| `replication` | How many copies are there? | `none` (default): one table, its rows split across the TP group. `per_node`: each node holds the whole table, mapped by all of its ranks. |
| `lookup` | How does a rank get rows another rank holds? Only with `replication` `none`. | `auto` (default): a collective SGLang picks for each step. `all_reduce`, `reduce_scatter`, `all_to_all`: that collective (see below). `peer_read`: each rank reads its tokens' rows straight from the owning GPU's shard over NVLink, with no collective. |

The resolved configuration is logged at startup as `DeepSeek-V4.1 Engram tables: ...`.

```bash
# Default: rows split across the TP group in GPU memory.
sglang serve --model-path deepseek-ai/DeepSeek-V4.1-Flash --tp-size 4

# Rows in host memory, one shard per rank: frees the HBM for the KV cache.
sglang serve --model-path deepseek-ai/DeepSeek-V4.1-Flash --tp-size 4 \
  --engram-config '{"placement": "cpu"}'

# Rows in GPU memory, read from the owning GPU without a collective.
sglang serve --model-path deepseek-ai/DeepSeek-V4.1-Flash --tp-size 8 --attn-dp-size 8 \
  --engram-config '{"lookup": "peer_read"}'
```

## Choosing a configuration

- **Small TP groups that are short of KV cache** (TP2, TP4, DP attention over 2 GPUs): use
  `{"placement": "cpu"}`. On GPU, each rank would hold half or a quarter of the 189 GiB.
- **Wide TP groups** (8 GPUs and more, including DP attention across nodes): the shard per GPU is
  small, and `{"lookup": "peer_read"}` removes the collective that each Engram layer otherwise
  runs per step, a cost that grows with the group size.
- **Latency-bound serving with memory to spare**: the default keeps every lookup in device memory.

## How `auto` picks the collective

Under DP attention, every rank needs rows for its own tokens, but each table is split across the
TP group. The lookup gathers every rank's hash ids, looks up rows, and exchanges them:

- `all_reduce`: every rank looks up the rows it holds for all tokens, zero for the rest, and an
  all-reduce sums the buffers; each rank keeps its own slice.
- `reduce_scatter`: the same lookup, summed with a reduce-scatter that hands each rank only its own
  tokens' rows.
- `all_to_all`: each rank looks up only the hash columns it holds, for all tokens, and sends each
  rank its part. It moves `1 / tp_size` of the bytes the other two move, and needs the 24 hash
  columns to divide over the TP group (DP 2, 3, 4, 6, 8 and 12).

On CUDA, with one DP rank per TP rank and no context parallelism, `auto` uses `all_to_all` for
eager prefill batches whenever the columns divide, and elsewhere keeps the collective it used
before (a reduce-scatter for decode batches, an all-reduce otherwise). Measured on GB300, B300 and
H200 at DP 2-8 with host tables, from 512 to 65,536 tokens per rank, `all_to_all` was the fastest
of the three in 134 of 144 shapes, and in all 96 from 8,192 tokens per rank up, where it was 8-62%
faster than the all-reduce. Below that, another choice was ahead by at most 85 us. Every
choice returns the same rows.

An explicit `all_reduce`, `reduce_scatter` or `all_to_all` applies to the same eager prefill
lookups; decode batches and CUDA-graph replays keep their collective, and without DP attention the
lookup is always an all-reduce.

## Requirements and limits

- **`placement` `cpu`**: host RAM for the whole table across the node's ranks, and a longer load.
  With `replication` `none`, each rank maps its shard with 2 MiB transparent huge pages and logs
  the coverage it got; without them, lookups are several times slower.
- **`replication` `per_node`**: one shared memory file per TP group, which the other ranks open
  through `/proc`. The TP group must sit on one node (`--nnodes 1`) and the ranks must share a PID
  namespace. This mapping usually gets no huge pages unless shmem transparent huge pages are on.
- **`lookup` `reduce_scatter` or `all_to_all`**: DP attention on CUDA; `all_to_all` also needs the
  hash columns to divide over `--tp-size`.
- **`lookup` `peer_read`**: CUDA with peer access between the GPUs (POSIX handles within a node,
  fabric handles across an NVLink domain). Each GPU holds `1 / tp_size` of the table. With TP1
  there is no peer, and the setting has no effect.
- `--engram-config` applies to DeepSeek-V4.1 only; a DeepSeek-V4 checkpoint without Engram layers
  rejects it.

## Deprecated environment variables

Earlier releases chose the placement with environment variables. They still work when
`--engram-config` is not given, with a deprecation warning that prints the equivalent JSON, and are
ignored when it is:

| Environment variables | `--engram-config` |
|---|---|
| `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` | `{"placement": "cpu"}` |
| plus `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank` | `{"placement": "cpu"}` |
| plus `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared` | `{"placement": "cpu", "replication": "per_node"}` |
| `SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS=1` | `{"lookup": "peer_read"}` |

`SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT` used to default to `shared`; a host table set by
`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` alone now gets one shard per rank, which is backed by
huge pages.
