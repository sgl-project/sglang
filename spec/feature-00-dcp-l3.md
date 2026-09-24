Status: proposed

# DCP + HiCache L3: matching engines

**TL;DR:** Store each distinct MLA DCP shard once in L3, with one writer per shard in each engine. Ranks holding equivalent shards share a storage key. Another engine with the same TP/DCP sizes and cache format reads the corresponding shards. A prefix is reused only as far as every rank successfully loads it.

## Supported contract

Stage 1 supports dense MLA through `MLATokenToKVPoolHost`, the `file` L3 backend on shared storage, `page_first` host layout, `kernel` host/device transfers, BF16 KV, `write_through`, and `wait_complete` prefetch. The reference model is `deepseek-ai/DeepSeek-V2-Lite-Chat` on H200 with FlashInfer MLA and `ag_rs` DCP communication.

An engine is one serving instance, including its GPU ranks. Both engines use the same model revision, tokenizer and KV-affecting model settings, TP size, DCP size, physical page size, and host layout. DCP size divides TP size. PP, attention DP, and other context parallel groups have size 1.

The shared storage directory belongs to one exact model revision and its KV-affecting settings. Operators use separate directories for different revisions/settings; stage 1 does not fingerprint model weights. GPU addresses, process IDs, job IDs, and engine IDs are never part of a cache key.

The release gate is a new engine with empty GPU and host caches reading pages written by a separate engine. Repeated requests to one warm engine do not establish cross-engine L3 support.

## Save and load workflow

For TP=2 and DCP=2, one eight-token example page is split by token position:

```text
                  shared L3                     fresh engine B
engine A
rank 0: 0 2 4 6 -> prefix X / rank 0 shard ----> rank 0: 0 2 4 6
rank 1: 1 3 5 7 -> prefix X / rank 1 shard ----> rank 1: 1 3 5 7
                    two different objects

save: GPU -> local host rows -> own L3 object
load: own L3 object -> local host rows -> GPU
```

For each DCP rank, the lowest TP rank holding that shard is its writer. With the supported contiguous DCP groups, writers are TP ranks `0..DCP_size-1` in the first DCP group. All TP ranks read their corresponding DCP-shard key directly into their own host buffers.

```text
TP=4, DCP=2       write once per shard       read on every rank

TP rank 0 -----> L3 shard 0 -------------> TP ranks 0 and 2
TP rank 1 -----> L3 shard 1 -------------> TP ranks 1 and 3
TP ranks 2, 3: skip storage writes

one logical page -> 2 stored objects, not 4
```

Exactly `DCP_size` objects represent each logical page, regardless of how many DCP groups the TP group contains. The first DCP group performs backup I/O; the others still acknowledge their skipped backup operations and release their local resources through the normal completion path. A skipped backup acknowledgment does not claim that the writer has finished.

MLA remains the data format. Write ownership is determined separately from whether a model uses MLA. The controller and file eviction/accounting code use the same writer selection: one owner per distinct DCP shard in each engine. Readers outside the first DCP group do not independently account for or evict those shared files.

## Writer selection rule

Identify a shard by the KV data it contains and its token-position slice, rather than by the GPU storing it. For each stored pool/layer range, ranks can share an object only when both match:

```text
shard = (KV partition, DCP rank)
writer = lowest TP rank holding that shard

save: only the writer issues storage I/O
load: every rank reads the key for its shard
```

For dense MLA, the KV partition is the shared compressed representation, so only DCP rank varies. Compute writer selection once from the parallel layout and use it for backup and file ownership. There is no per-page rank election or KV gathering.

The same rule explains GQA, although GQA implementation remains outside stage 1: its KV partition is the ordered set of KV head IDs in the stored data. Derive that set from the model's actual KV-head mapping, including replicated heads; two ranks with different head IDs cannot share a key. If one object spans multiple layers, the mapping must match across every included layer.

| Layout | Equivalent ranks | Selected writers | Objects per logical page |
| --- | --- | --- | --- |
| MLA, TP=4/DCP=2 | 0/2 and 1/3 | 0, 1 | 2 |
| GQA, 2 KV heads, TP=4/DCP=2 | None | 0, 1, 2, 3 | 4 |
| GQA, 2 KV heads, TP=8/DCP=2 | 0/2, 1/3, 4/6, 5/7 | 0, 1, 4, 5 | 4 |

The GQA examples assume the regular contiguous head mapping. Use the actual mapping as the source of truth rather than a rule that always chooses the first DCP group.

## Storage identity

Use the existing full-prefix page hash, computed over logical tokens. All ranks use the same hash for the same logical page; they do not hash only their local token positions.

For DCP objects, the storage key also identifies:

| Field | Purpose |
| --- | --- |
| Model identity in the shared directory | Keeps unrelated model data apart |
| TP size | Limits reuse to the matching TP configuration supported in stage 1 |
| DCP size and DCP rank | Identifies the distinct token-position shard shared by equivalent TP ranks |
| Logical page size, KV dtype, host layout | Identifies how to interpret the bytes |

Example identity, independent of the exact filename spelling:

```text
(model, prefix_hash, TP=4, DCP=2, dcp_rank=1,
 logical_page=128, dtype=bf16, layout=page_first)
```

`exists`, `get`, `set`, metadata lookup, and eviction use the same identity. TP rank and DCP group identity are not part of the key: TP ranks 1 and 3 in this example resolve to the same object. Equal configurations in different engines produce equal keys. A different DCP or TP size produces a different key. DCP=1 objects cannot be mistaken for DCP shards.

Each object contains exactly one rank's local rows for one full logical page, across the model's layers. Publish complete objects using the file backend's atomic file replacement. Concurrent writers of the same key use independent temporary files and publish complete objects.

Within an engine, selected writers eliminate duplicate backup I/O. Across engines, identical keys give one persistent object per shard; an existing valid object can skip the write. Two engines racing on a missing key may both perform I/O and create temporary files before publishing to the same final path. Stage 1 guarantees unique stored objects, not exactly-once I/O across engines.

## Logical tokens and local rows

Let `D` be DCP size and `P` the number of physical rows per rank per page. The logical page contains `L = P * D` tokens. In the reference setup, `--page-size 64`, DCP=2 gives `P=64`, `L=128`.

```text
controller / prefix hashes / token counts: 128 logical tokens per page
rank 0 host buffer and L3 payload:          64 local rows per page
rank 1 host buffer and L3 payload:          64 local rows per page
```

The controller keeps logical indices and logical page counts. At the generic L3 page-access boundary, an aligned logical page start `s` maps to local row `s / D`; the accessor reads or writes `P` rows. A page beginning at logical slot 256 maps to local rows 128..191 for DCP=2/P=64. Validate page alignment and translate exactly once on both save and restore.

Preserve each page's order when operations contain nonadjacent host pages. Allocate dummy read buffers in local bytes. Never divide the prefix hash length, completion token count, or host allocation/free accounting by `D`. Host allocations remain logical and page-aligned.

## Cross-rank completion

Reuse the controller's existing minimum-prefix reductions for both lookup and actual read completion. Their groups must include all TP ranks participating in the request, and therefore all its DCP shards.

```text
rank 0 successfully reads: page A, page B, page C
rank 1 successfully reads: page A, [missing B]
agreed reusable prefix:   page A only, on every rank
```

Missing objects, eviction between lookup and read, short reads, and failed writes must not produce a partial-shard cache hit. Every rank still emits the matching completion acknowledgments so peers do not hang. Reuse the common complete prefix and compute the remaining tokens normally. Release unused host allocations after transfers stop using them.

Keep host pages alive until local backup I/O finishes. Backup completion acknowledges local work; it is not a claim that every shard is stored. Readers decide usability through cross-rank lookup and read results. No cross-engine collective is needed.

Startup and runtime storage attachment apply the same support checks. Unsupported DCP/L3 combinations fail with a specific error before starting storage workers.

## Implementation map

Paths below link to the inspected base commit `9f6fc553322e897a22f30320dae750f4a40a0b84`.

| Code | Required change |
| --- | --- |
| [HiCache argument checks](https://github.com/sgl-project/sglang/blob/9f6fc553322e897a22f30320dae750f4a40a0b84/python/sglang/srt/arg_groups/hicache_hook.py#L110) | Allow exactly the supported DCP/L3 configuration; keep errors for other combinations. |
| [Storage config and file backend](https://github.com/sgl-project/sglang/blob/9f6fc553322e897a22f30320dae750f4a40a0b84/python/sglang/srt/mem_cache/hicache_storage.py#L26) | Carry shard/layout information and apply it consistently to file keys and read sizes. |
| [Controller attach/config](https://github.com/sgl-project/sglang/blob/9f6fc553322e897a22f30320dae750f4a40a0b84/python/sglang/srt/managers/cache_controller.py#L520) | Separate MLA format from write ownership, select the first DCP group as writers, preserve skipped-backup acknowledgments, and validate runtime attachment. |
| [MLA host page access](https://github.com/sgl-project/sglang/blob/9f6fc553322e897a22f30320dae750f4a40a0b84/python/sglang/srt/mem_cache/pool_host/mla.py#L998) | Accept logical page starts and translate to local rows for generic file reads/writes. |
| [File eviction ownership](https://github.com/sgl-project/sglang/blob/9f6fc553322e897a22f30320dae750f4a40a0b84/python/sglang/srt/mem_cache/storage/file/lru_file_evictor.py#L66) | Assign each shard's accounting and eviction to its selected writer within the engine. |
| [Lookup agreement](https://github.com/sgl-project/sglang/blob/9f6fc553322e897a22f30320dae750f4a40a0b84/python/sglang/srt/managers/cache_controller.py#L1198) and [read completion agreement](https://github.com/sgl-project/sglang/blob/9f6fc553322e897a22f30320dae750f4a40a0b84/python/sglang/srt/managers/cache_controller.py#L1326) | Verify group coverage, matching acknowledgments after errors, and logical token units. |

## Acceptance tests

| Test | Required result |
| --- | --- |
| Key identity | Equivalent shards across TP ranks and matching engines give the same key; different DCP ranks, TP/DCP sizes, page sizes, dtypes, or layouts cannot alias. |
| Page round trip | Distinct per-token/per-layer values survive save/load exactly for every rank at DCP=2 and 4, including nonzero page starts and reordered, nonadjacent pages; payload bytes equal `P * layers * KV_width * dtype_bytes`. |
| Ownership | At TP=4/DCP=2, only ranks 0 and 1 write/account/evict; ranks 2 and 3 complete skipped backups without storage I/O or leaked resources. Assert exactly two objects per logical page and total payload bytes `2 * P * layers * KV_width * dtype_bytes`, the same as TP=2/DCP=2. |
| Equivalent replicas | Give equivalent MLA ranks the same logical page at different local buffer addresses; verify equal shard identity, equal saved payload, one selected writer, and correct restore into each reader's allocation. |
| Fresh engine | Engine A writes a multi-page prefix and completes every selected writer's I/O; fresh B reuses a nonzero, page-aligned L3 prefix with empty L1/L2 and generates the same deterministic token IDs as a cold run. Test TP/DCP=2/2, 4/2, and 4/4; at 4/2 verify ranks 0/2 and 1/3 restore from their respective shared keys. |
| Incomplete storage | Delete a middle-page shard, truncate one object, and inject a read failure after successful lookup in separate runs. All ranks agree on the available complete prefix, compute the rest correctly, release unused allocations, and finish within the test timeout. |
| Concurrency | Two matching engines writing the same prefix leave complete readable objects; subsequent readers restore every shard. |
| Guards and regressions | Startup and runtime attach reject unsupported combinations; DCP L1/L2 host-pool tests and existing DCP=1 MLA/GQA file-storage tests pass. |

Use prompts spanning at least four logical pages and exceeding the prefetch threshold. Record actual L3-restored tokens and per-rank payload sizes; total `cached_tokens` alone is insufficient. Wait for backup completion with a bounded condition, rather than a fixed sleep. A separate fresh reader is required even if flush-based tests also pass.

The baseline demonstrates the guards and existing local page geometry; it does not demonstrate DCP/L3 restore. See [reproducible evidence](evidence/feature-00-dcp-l3.md).

## Implementation phases and review gates

Complete and verify each phase before committing it. The user authorized completion of the remaining plan without approval pauses. Each review includes an uncommitted diff, exact test commands and results, and the small expected/actual example below.

```text
1. keys + writer selection
          |
2. logical page -> local bytes
          |
3. controller backup + ownership
          |
4. missing shards + error completion
          |
5. enable supported configuration + fresh-engine proof
```

Phases 1-3 are committed (`1cdb6b1`, `5ba812b`, `866d936`). Phase 4 passes real four-rank CPU failure tests and is complete. Phase 5 remains. The feature remains proposed until the full release gate passes. See the evidence log for phase-local results and reproduction commands.

Startup rejection stays in place through phase 4. Component tests call the relevant storage/pool/controller APIs directly; do not add a production flag to bypass the guard. Runtime attachment must not expose the unfinished path. Phase 5 changes the support checks only when the integrated tests pass.

### Phase 1: define shared keys and select writers

**Diff:** Add the DCP/layout information to storage configuration, derive the dense-MLA writer once, and use the shared shard identity in file keys. Keep data transfer and worker behavior for later commits. Avoid a general GQA ownership framework in this MLA implementation.

**Verify:** CPU tests construct all rank configurations for TP/DCP=2/2, 4/2, and 4/4. Assert key equality for replicas, separation for different shards/formats, and identical keys in two engines. Check existing DCP=1 MLA/GQA file key behavior with its tests.

**Human review:** Inspect this exact TP=4/DCP=2 result, including the rule used by every key operation:

```text
TP rank       0       1       2       3
DCP rank      0       1       0       1
key           K0      K1      K0      K1
writer?       yes     yes     no      no
```

**Gate:** The identity and ownership table passes without reading or writing model KV. Startup still rejects DCP/L3.

### Phase 2: make one page round-trip correctly

**Diff:** Update generic MLA host-page reads/writes to translate logical page starts once and validate alignment. Use local-sized read buffers. Retain guards on storage paths outside the supported generic file path.

**Verify:** CPU tests fill real host pools with distinguishable token/layer values, save to temporary files, and restore into a different allocation. Cover every DCP rank at sizes 2 and 4, nonzero page starts, reordered pages, and unchanged neighboring rows. Run the existing DCP L1/L2 host-pool tests, replacing the generic accessor's old rejection assertion with the new round-trip contract.

**Human review:** For DCP=2, logical start 256 and logical page size 128 select local rows 128..191. With 2 layers, width 12, and BF16, the object is exactly 3,072 bytes; restored values match exactly.

**Gate:** File/pool round trips pass without a running server or background controller workers. No controller token count or allocator accounting changes units.

### Phase 3: wire backup ownership into the controller

**Diff:** Replace the rank-0-only MLA backup decision with the selected writer rule. Apply the same ownership to file accounting/eviction. Preserve skipped-operation acknowledgments and host-buffer lifetimes. Add or retain runtime-attach rejection until the final enablement phase.

**Verify:** Controller tests at TP=4/DCP=2 process the same page on all four ranks. Assert storage calls only from ranks 0/1, two final objects, all four local acknowledgments, and successful reads by ranks 2/3. Delay a writer to verify its buffer remains live; skipped replicas complete without claiming global persistence. Check allocation counts after completion and eviction ownership for both shard keys.

**Human review:** Inspect a four-row report containing rank, selected writer, storage-call count, acknowledgment count, and remaining allocation count. It must show writes `[1, 1, 0, 0]` and one acknowledgment per rank for one backup operation.

**Gate:** Controller-to-file save/restore works with real temporary files and controlled transfer completion. There is no KV gather, extra replica object, or host-memory leak.

### Phase 4: prove missing shards and errors cannot hang ranks

**Diff:** Verify and, where needed, fix lookup/read exception handling and completion acknowledgment ordering. Retain the existing minimum-prefix reductions and ensure their groups cover all TP ranks. Only change synchronization that fails the contract.

**Verify:** A multi-process CPU test uses real distributed reductions and the controller completion path with bounded timeouts. At TP=4/DCP=2, test a missing middle-page shard, truncated payload, lookup exception, read exception after a hit, and failed/delayed backup. All processes must finish, agree on the complete prefix, and release unused allocations. Keep successful page reads on other ranks in flight during at least one failure case.

**Human review:** For pages A/B/C with shard 1 missing at B, all four ranks report only A as reusable. With logical page size 128, each reports 128 restored tokens, not 64 or 384. Review both the agreed result and process completion; absence of a Python exception alone is insufficient.

**Gate:** Failure cases finish within their timeout with equal prefix counts and no retained allocations. The test cannot pass by mocking away the cross-rank reductions.

### Phase 5: enable the feature and prove cross-engine reuse

**Diff:** Allow the supported configuration in startup and runtime-attach checks. Add the H200 end-to-end test using the same server launch and request helpers as existing HiCache tests. Include specific rejection tests for the out-of-scope combinations and run DCP=1 file-storage regressions.

**Verify:** For TP/DCP=2/2, 4/2, and 4/4, start engine A, write a prefix spanning at least four logical pages, and wait for every selected writer's completion. Stop A and start fresh B against the same storage directory. Compare B against a cold run with the same topology and deterministic settings. Also test concurrent matching engines and repeat the missing-shard case through actual inference. Each individual A/B run keeps TP and DCP sizes fixed.

**Human review:** Inspect a compact result table for each topology: logical pages saved, final objects, payload bytes, L3-restored tokens per rank, generated token IDs versus cold run, and failure-case completion. At TP=4/DCP=2, four saved logical pages must give eight KV objects, not sixteen. The fresh reader must show positive, page-aligned L3 restore with initially empty L1/L2.

**Gate:** All acceptance tests pass, exact commands and real results are recorded in the evidence log, and the spec matches the implementation. Only then mark the feature implemented and remove the remaining-work items.

### Review artifacts

Keep phase-local test changes beside the code they verify. Unit and distributed test files introduced by a phase must have an exact runnable command in the evidence log. Keep raw logs and measurements in evidence, not in the feature contract. No passing results should be claimed for phases not run.

- [ ] Complete phase 5 with H200 cross-engine evidence and mark the spec implemented.

## Boundaries

Stage 1 does not reshard between different DCP sizes or reuse across different TP sizes.

Mooncake, NIXL, other L3 backends, zero-copy storage paths, GQA/MHA with DCP, sparse or hybrid models, speculative decoding, quantized KV, PP/attention-DP/other-CP combinations, and prefill/decode disaggregation are outside this first stage. Cross-engine write locking, distributed storage quota coordination, and performance targets are also outside the release gate.
