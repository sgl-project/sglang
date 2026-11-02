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
| `lookup` | How does a rank get rows another rank holds? Only with `replication` `none`. | `auto` (default): `peer_read` for GPU tables whose TP group is one NVLink domain, otherwise a collective SGLang picks for each step. `all_reduce`, `reduce_scatter`, `all_to_all`: that collective (see below). `peer_read`: each rank reads its tokens' rows straight from the owning GPU's shard over NVLink, with no collective. |

The configuration is logged at startup as `DeepSeek-V4.1 Engram tables: ...`, and the lookup
`auto` resolves to, once the model is built, as `Engram lookup auto -> ...`.

```bash
# Default: rows split across the TP group in GPU memory; over NVLink, each rank reads its
# tokens' rows from the owning GPU.
sglang serve --model-path deepseek-ai/DeepSeek-V4.1-Flash --tp-size 4

# Rows in host memory, one shard per rank: frees the HBM for the KV cache.
sglang serve --model-path deepseek-ai/DeepSeek-V4.1-Flash --tp-size 4 \
  --engram-config '{"placement": "cpu"}'
```

## Choosing a configuration

- **Small TP groups that are short of KV cache** (TP2, TP4, DP attention over 2 GPUs): use
  `{"placement": "cpu"}`. On GPU, each rank would hold half or a quarter of the 189 GiB.
- **Wide TP groups and memory to spare** (8 GPUs and more, including DP attention across nodes):
  the default. The shard per GPU is small, and over NVLink each rank reads its rows in place,
  with no collective.

## How `auto` picks

With the tables in GPU memory, CUDA, and a TP group that is one NVLink domain (one node with
full NVLink, or one multi-node NVLink fabric such as GB300 NVL72), `auto` reads the rows in
place, as `peer_read` does, and logs `Engram lookup auto -> peer_read`. Measured with
one Engram layer, it was faster than every collective, for prefill (1K to 32K tokens per rank)
and decode (1 to 1,024 tokens per rank, in CUDA graphs):

- DP attention 8 on GB300 (two nodes), B300 and H200: 2.8-7.5x faster than `all_to_all`.
- DP attention 16 and 32 on GB300: 6-75x faster than the all-reduce and reduce-scatter, the only
  collectives there.
- Plain TP 2 and 8 on B300, TP 8 on GB300: up to 3.7x faster than the all-reduce, and never
  slower.

Otherwise (host tables, or GPUs without NVLink between them) it picks a collective.

Under DP attention, every rank needs rows for its own tokens, but each table is split across the
TP group. The lookup gathers every rank's hash ids, looks up rows, and exchanges them:

- `all_reduce`: every rank looks up the rows it holds for all tokens, zero for the rest, and an
  all-reduce sums the buffers; each rank keeps its own slice.
- `reduce_scatter`: the same lookup, summed with a reduce-scatter that hands each rank only its own
  tokens' rows.
- `all_to_all`: each rank looks up only the hash columns it holds, for all tokens, and sends each
  rank its part. It moves `1 / tp_size` of the bytes the other two move, and needs the 24 hash
  columns to divide over the TP group (DP 2, 3, 4, 6, 8 and 12).

On CUDA, with one DP rank per TP rank and no context parallelism, `auto` uses `all_to_all`
whenever the columns divide, for prefill and decode batches alike, in CUDA graphs too; only a
prefill batch of uneven per-rank counts captured in a CUDA graph keeps the earlier collective.
Measured on GB300, B300 and H200 at DP 2-8 with host tables:

- Prefill, 512 to 65,536 tokens per rank, uniform or skewed: `all_to_all` was the fastest of the
  three in 134 of 144 shapes, and in all 96 from 8,192 tokens per rank up, where it was 8-62%
  faster than the all-reduce. Below that, another choice was ahead by at most 85 us.
- Decode under CUDA graphs, 16 to 1,280 tokens per rank: from DP 4, `all_to_all` was up to 48%
  faster than the reduce-scatter decode used before, and at most 12 us behind at 64 tokens per rank
  or fewer; at DP 2 it was at most 20 us behind and up to 53 us ahead.

Every choice returns the same rows. `all_reduce` keeps the collectives used before `auto` (an
all-reduce for prefill, a reduce-scatter for decode), and an explicit `reduce_scatter` or
`all_to_all` applies to prefill and decode alike. Without DP attention the lookup is always an
all-reduce.

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
  fabric handles across an NVLink domain); `auto` takes it only over NVLink. Each GPU holds
  `1 / tp_size` of the table. With TP1 there is no peer, and the setting has no effect.
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
