Status: implemented

# DCP + HiCache L3: matching engines

**TL;DR:** Store each distinct dense MLA DCP shard once in file-backed L3. Equivalent ranks share a key, one rank writes each shard, and every rank reads its own shard. Engines reuse KV only with matching TP/DCP sizes and cache format.

## Supported configuration

The first stage supports dense MLA through a single `MLATokenToKVPoolHost`, shared `file` storage, `page_first` layout, `kernel` transfers, BF16 KV, `write_through`, `wait_complete`, and `cache` host memory mode. PP, DP, and other CP sizes are 1; attention DP is disabled. Startup and runtime attachment enforce these limits before starting storage workers. Runtime policy changes enforce the same limits.

The H200 reference is `deepseek-ai/DeepSeek-V2-Lite-Chat` with FlashInfer MLA and `ag_rs`. Both engines use the same model revision, tokenizer, KV-affecting settings, TP size, DCP size, physical page size, dtype, and layout. DCP size divides TP size.

One shared storage directory belongs to one exact model revision and its KV-affecting settings. Operators use different directories for incompatible models/settings; this stage does not fingerprint model weights.

## Shards and ownership

```text
TP=4, DCP=2

rank 0 ---- write ----> shard 0 <---- read ---- ranks 0, 2
rank 1 ---- write ----> shard 1 <---- read ---- ranks 1, 3

one logical page -> two objects
```

MLA shares its compressed KV representation across TP ranks. DCP splits its token positions. The lowest TP rank holding each token slice writes it: with contiguous DCP groups, TP ranks `0..DCP_size-1` are writers. All ranks read. The controller and file evictor use the same writer decision; replicas do not independently account for or evict shared files.

The general identity is `(KV partition, DCP token slice)`. MLA has one shared KV partition, hence exactly `DCP_size` objects per logical page. GQA implementation is outside this stage: different KV-head sets are different partitions, so the general rule does not imply only `DCP_size` GQA objects. For example, the regular contiguous mapping at TP=8/DCP=2 with two KV heads has writers 0/1/4/5 and four objects.

DCP=1 preserves existing behavior: MLA writes on TP rank 0; GQA writes its existing rank-specific objects.

## Storage keys and publication

Use the existing full-prefix logical page hash. Add TP size, DCP size/rank, logical page size, dtype, and host layout for DCP objects. Equivalent replicas and matching engines resolve to the same key. TP rank, DCP group, engine ID, and GPU address are not part of that identity. Different supported formats/topologies cannot alias; DCP=1 keeps its existing keys.

Each object contains one rank's local rows for one logical page across all layers. The file backend writes a unique temporary file and publishes with atomic replacement. Concurrent engines may both write a missing object, but the final key stores one complete object. This guarantees unique persistent objects, not exactly-once I/O across engines.

## Logical tokens and local rows

For DCP size `D` and physical page size `P`, the logical page contains `L=P*D` tokens.

```text
--page-size 64, DCP=2

logical page:        128 tokens
rank-local payload:   64 rows
logical start 256 -> local rows 128..191
```

The controller, prefix hashes, completion counts, and host allocations use logical tokens. Generic MLA host-page access validates alignment/range and translates the page start once (`local_start = logical_start / D`). Dummy read buffers contain local bytes. Nonadjacent page operations preserve their order. Zero-copy page metadata remains guarded under DCP.

## Completion and shared-prefix agreement

Each backup acknowledges its local work. A replica can acknowledge a skipped write before the shard writer finishes; this does not claim global persistence. A writer holds its host buffer until its write completes.

Lookup and read completion use the existing minimum-prefix reductions. In the supported layout, the attention TP group covers every DCP shard and replica. Missing shards shorten the shared usable prefix. The existing read path preserves one acknowledgment per batch when a page is absent, including eviction after lookup.

```text
rank 0: A B C
rank 1: A [missing B]
all ranks reuse A; the model computes the rest
```

The final acknowledgment lets the scheduler release unused host pages after I/O stops. No KV gather or cross-engine collective is needed.

## Verification and code pointers

The release gate uses separate writer and reader processes with initially empty GPU/host caches. It compares deterministic output token IDs against a cold run and checks per-rank L3-restored tokens, final object counts, and byte sizes. It covers TP/DCP=2/2, 4/2, 4/4, a missing middle shard, concurrent writers, and runtime attachment.

CPU tests cover shared keys, every shard at DCP=2/4, reordered page round trips, writer/eviction ownership, delayed backup lifetimes, and real four-process prefix agreement with bounded timeouts. See [commands and measured evidence](evidence/feature-00-dcp-l3.md).

| Responsibility | Code |
| --- | --- |
| Support checks | `arg_groups/hicache_hook.py`, `mem_cache/unified_cache/storage_attachment.py` |
| Keys and writer decision | `mem_cache/hicache_storage.py` |
| Logical page translation | `mem_cache/pool_host/mla.py` |
| Backup/read workers and reductions | `managers/cache_controller.py`, `mem_cache/hybrid_cache/hybrid_cache_controller.py` |
| File accounting and eviction | `mem_cache/storage/file/lru_file_evictor.py` |
| Fresh-engine H200 test | `test/manual/hicache/test_dcp_l3.py` |

Runtime code paths above are relative to `python/sglang/srt/`.

## Boundaries

This stage does not reshard between different DCP or TP sizes. GQA/MHA with DCP, sparse/hybrid/multiple-pool models, speculative decoding, quantized KV, other L3 backends, zero-copy storage, PP/DP/other-CP combinations, and prefill/decode disaggregation are outside scope. Cross-engine locking, distributed quota coordination, and performance targets are outside the release gate.
