# Unified L3 KV Layout — Canonical Shard Grid (manifest-free v1)

> Status: proposal, 2026-08-03 (rev 2, after adversarial review against the tree).
> Successor iteration of the unified-L3 thread
> ([`DESIGN_unified_l3_kv_cache.md`](DESIGN_unified_l3_kv_cache.md),
> [`DESIGN_unified_l3_kv_layout.md`](DESIGN_unified_l3_kv_layout.md),
> [`DESIGN_unified_l3_kv_layout_simple.md`](DESIGN_unified_l3_kv_layout_simple.md)).
> Keeps that family's two load-bearing ideas — topology-free identity and
> model-global coordinates — and drops the coordinator / manifest / transactional
> metadata plane from v1 by making the shard key-space *deterministically
> enumerable* from static config. Cells here are forward-compatible with that
> ABI's "chunks", so the full artifact protocol can be layered on later without
> rewriting data.

> **Implementation update (rev 15).** The implemented design has evolved past
> this document in two ways. (1) `layer_partition` is a single integer — the
> layer unit — with the model's trailing remainder forming a short final
> cell (only the last PP stage may end off-unit; boundary lists no longer
> exist). (2) There are no separate layout-pinned zero-copy schemes: ANY
> partition knob (`head_group` and/or `layer_partition`, rank-replicated
> included) selects the cell adapter, whose objects always carry the
> layout-neutral canonical byte order (`object_layout = "cell-v1"`). The
> adapter skips conversion per slab when the pool view is already
> canonical-contiguous — MLA on `page_first_direct` stays fully zero-copy
> (no staging arena allocated). Where sections below pin `page_head` or
> `page_first_direct` as *requirements*, read them as the zero-copy fast
> cases of the one adapter path. Current source of truth:
> `PR_unified_kv_l3.md` and `hicache_key_scheme.plan_canonical_cells`.

## 0. Direct answers to the two framing questions

**Q1 — "single key across parallel configs + metadata to distinguish"?**
Half yes. One *logical* key per page (the existing chained content hash — it is
already topology-free), but **not** one object whose metadata describes the
writer's parallel config. Writer-config-in-metadata is the same coupling as key
suffixes, moved one level down: every reader must still understand every
producer topology, two producers with different configs still produce
incompatible objects under one name, and shared *mutable* metadata across
independent rank-writers reintroduces exactly the write race the current
per-rank suffixes exist to avoid (`hicache_storage.py:383-386`). No store we
target even has a mutable metadata plane (mooncake: none; nixl: none; hf3fs: a
side server keyed per rank).

Instead: name sub-objects by **canonical model-global coordinates**
(layer-range × kv-head-range × token-range), a shard space that never mentions
rank, world size, TP/PP/CP. The only metadata is one **static, immutable
namespace descriptor** that is *distributed out-of-band as configuration* —
never per-page manifests, never per-object writer provenance, and never a
store-resident object as source of truth (§2.1 explains why the store cannot be
authoritative for layout). Everything dynamic ("which cells exist") is answered
by the store's own `exists()`, which every backend already implements and the
prefetch path already batches.

**Q2 — "we cannot allgather KV before saving"?**
Agreed, and the current L3 write path already has this property: no KV-data
collective exists anywhere in `cache_controller.py` / `hiradix_cache.py` — every
rank backs up its own shard, and the only collectives are small control-plane
gloo ops (MIN/MAX allreduces of scalar counts, barriers, PP point-to-point of
small control tensors; `cache_controller.py:1065-1071`,
`hiradix_cache.py:996-1021`, `:1581-1591`, `:260-303`). No KV bytes ever cross
ranks for L3. What the suffix scheme wastes is the *read* side of that
property: shard-local writes exist but are only legible to a bit-identical
topology. The fix is not to add communication — it is to name the shards
canonically so that:

- **Save**: each rank puts exactly the bytes it already holds locally.
  Zero inter-GPU traffic, cross-node PP/TP included. Same as today.
- **Load**: a reader with a *different* topology fetches exactly the canonical
  cells covering the rectangle it owns. "Resharding" degenerates into key
  selection — it costs nothing beyond the L3 fetch that was happening anyway.
  No NCCL, no post-load shuffle.

(The one place an allgather exists today is inside prefill-CP *compute* —
`cp_allgather_and_save_kv_cache`, `layers/utils/cp_utils.py:395-426` — which
replicates KV into every rank's pool for the forward pass itself. That is a
pool-layout decision upstream of L3; the `kv_reshard` branch removes it. This
design works with either: replicated pools elect one writer per cell, sharded
pools write home-only.)

## 1. Why unify at all: the suffix scheme is already seven inconsistent schemes

Every backend hand-rolls its own topology encoding today, and they disagree:

| Backend | MHA key encodes | PP | CP | Model isolation |
|---|---|---|---|---|
| file | `_{model}_{tp_rank}_{tp_size}` | `_{pp_size}_{pp_rank}` | `_cp{r}_{s}` | yes (served_model_name) |
| mooncake | `_{tp_rank}` (**no tp_size**) | `_{pp_rank}` (**no pp_size**) | **dead fields, no suffix** | yes (served_model_name) |
| nixl | `_{model}_{tp_rank}_{tp_size}` | **none** | none | yes |
| simm / umbp | `_{tp_rank}` | `_{pp_rank}` | none | **none** |
| eic | prefix w/ tp_rank+size+layout | **none** | none | yes |
| hf3fs | per-rank namespace (no size) | **none** | none | **none** |

(`hicache_storage.py:376-386`, `mooncake_store.py:571-593`,
`hicache_nixl.py:108-113`, `hicache_simm.py:193-211`, `umbp_store.py:794-810`,
`eic_storage.py:315-324`, `storage_hf3fs.py:200-224`.)

Concrete live hazards, before any new feature:

1. **Mooncake TP=8 writer / TP=4 reader**: keys embed rank but not size, so
   every reader rank 0–3 finds the writer's same-numbered object with *half*
   the heads it expects. Mooncake's get accepts any positive byte count
   (`_batch_postprocess`, `mooncake_store.py:1021-1028` checks `res > 0`, not
   `res == expected`), so the short read "succeeds" and the upper half of each
   rank's heads is stale garbage — silent corruption. (The reverse direction,
   TP=4 writer → TP=8 reader, degrades cleanly: reader ranks 4–7 miss, the
   attn-TP MIN-allreduce revokes the prefetch fleet-wide. The file backend is
   also not vulnerable — it raises on short read, `hicache_storage.py:473-474`.)
2. **PP on nixl/eic/hf3fs**: stages holding different layers collide on the
   same key. Mooncake encodes `pp_rank` but not `pp_size`, so PP2-stage-0
   (layers 0–39) is legible to a PP4-stage-0 reader (expects layers 0–19).
3. **CP on any backend except file**: `attn_cp_rank/size` are read into
   attributes (`mooncake_store.py:560-569`) and never used — all CP ranks race
   on one key. No server-args guard exists.
4. **PP prefetch divergence**: hit-length agreement spans only attn TP/CP
   gloo groups (`cache_controller.py:324-346`); PP stages can disagree on
   `storage_hit_length` with no gate.
5. **Split-heads breaks every side pool**: with `should_split_heads`,
   `mha_suffix` becomes a Python *list* (`mooncake_store.py:588-593`) and
   `_get_hybrid_page_component_keys` f-string-interpolates it
   (`mooncake_store.py:758, 772-773, 791-792`), producing keys like
   `hash_['2', '3']_conv_0`. Same-topology peers agree on the malformed key so
   it works by accident; the cross-TP reuse the mechanism exists for silently
   never hits for mamba/draft/SWA components.

Two orthogonal gaps compound these: the *naming* gap (above) and a *detection*
gap — geometry mismatches are silent because gets don't validate byte counts.
The grid fixes naming; §7's guards fix detection independently (and are worth
shipping even under the legacy scheme).

## 2. Identity

```text
logical page id   = chained SHA-256 over token ids (unchanged; mem_cache/utils.py:106-112)
namespace id (ns) = digest of the canonical descriptor encoding (below)
cell key          = {ns}:{page_hash}:{component}:L{i}:H{j}
```

> Implementation note (branch `unified_kv_l3`): v1 encodes the layer
> coordinate as the **absolute layer range** `L{start}-{end}` instead of a
> grid index `L{i}`. Uneven PP partitions (61-layer models at any pp_size)
> then attach without any divisibility constraint — differing partitions
> derive disjoint keys and miss instead of colliding — and `Lg` enters the
> descriptor only when uniform-grid layer fan-out lands. The two forms are
> equivalent where `Lg` divides every stage; the range form is a superset.
>
> Second implementation note (the **cell adapter**): when both fleet grids
> are declared (`head_group` + `layer_partition`) on a sharded-KV pool, the
> implementation switches the namespace to layout-neutral canonical cells —
> `(head, layer, token, dim)` per K/V half, the `page_head_layer_direct`
> byte order — gathered/scattered through pinned store-registered staging
> arenas (one per IO direction). The `object_layout` identity becomes the
> constant `cell-v1`, so any host layout interops (layer_first included), and
> the layout *mandates* in section 4.4 apply only to the zero-copy
> single-axis schemes. Cost: one host memcpy per direction on the async
> backup/prefetch threads; this is the section 4.4 "optional new layout"
> trade taken in staging-copy form, with the byte order pinned so a future
> zero-copy `page_head_layer_direct` layout is drop-in compatible.

- `component` is the `PoolName` axis. **In v1 only `kv` is supported** — the
  side pools are *not* orthogonal to sharding (§8) and models that have them
  are guarded out.
- `L{i}` = i-th canonical **layer group**, `H{j}` = j-th canonical **kv-head
  group** (§3). MLA-family (rank-replicated latent) has exactly one H group.
- A cell object holds **both K and V** of its rectangle (two fragments via the
  scatter-gather put path where available, §4.4) — fusing them removes the
  K-written-V-missing splice window that separate `_k`/`_v` objects would
  reopen (§6.2).
- No rank, world size, tp/pp/cp/dp anywhere in any durable name.

### 2.1 The namespace descriptor is configuration, not store state

```yaml
ukv_namespace_v1:
  model_id: deepseek-ai/DeepSeek-V4        # + optional weights digest
  representation: {dtype: bf16, mla: true, page_size: 64, layout_version: 1}
  grid: {layer_group: 8, head_group: 1, token_granule: page}
  numerics_id: <kernel/build ABI digest>   # REQUIRED — see §6.2
```

The descriptor is a frozen `msgspec.Struct` with a canonical serialization
(sorted fields, defaults resolved — the encoding is ABI, since `ns` is its
digest; a YAML-ish sketch is shown for readability only, per
`.claude/rules/no-dataclasses.md`). It reaches every deployment **out-of-band**
— a file/URI passed at launch — the same way model configs do. Two deployments
share cache iff they were given the same descriptor bytes; then and only then
they compute the same `ns` and land in the same keyspace.

This deliberately gives up in-store fail-fast for *mismatched* descriptors: a
fleet launched with a different descriptor computes a different `ns` and
silently partitions into a disjoint keyspace. That is safe (never corrupt) but
must be observable, so: each process logs its computed `ns` at attach, exports
it as a metrics label, and best-effort appends its digest to a per-model epoch
index object so operators can see accidental splits. What is *checked* at
attach is local compatibility (§2.2).

Why not store the descriptor authoritatively under a well-known key and verify
against it? Because no target store can hold authoritative config: mooncake's
"write-once" is a client-side check-then-put with exists-mapped-to-success
(`batch_set_v1`, `mooncake_store.py:1079-1093` — a TOCTOU, not a guarantee),
the master can evict any object under pressure, `clear()` →
`store.remove_all()` (`mooncake_store.py:1274-1275`) deletes it while fleets
run, and the file backend is plain last-writer-wins. A descriptor that can be
evicted and re-created differently while old-grid cells still exist is a
corruption generator, not a safety check. Config distribution is the one
channel that is already durable, versioned, and reviewed.

### 2.2 Attach-time compatibility check (local, always enforced)

Attach fails fast unless the local topology **tiles the grid**:

- PP/CP-layer-split: the rank's layer range is a whole number of layer groups;
- TP: the rank's kv-head shard is a whole number of head groups
  (`local_heads % Hg == 0`, i.e. `tp_size` divides `tp_lcm` implied by Hg);
- pool layout supports cell buffer metas (§4.4);
- `page_size`, dtype, and model config match the descriptor.

A PP2 stage attaching to an `Lg=80` (full-model) namespace is *refused* — not
silently degraded — with the remedy in the error message (regrid: §9).

### 2.3 Detection guards (required, independent of the grid)

- Every get validates `bytes_read == expected cell size` (today: `res > 0`,
  `mooncake_store.py:1021-1028`). One line; converts every residual
  geometry-confusion scenario in this doc from silent corruption to a loud
  error, and fixes hazard #1 even for the legacy scheme.
- `_page_backup` / `_batch_postprocess` treat key-already-exists as success
  (today a lost put race poisons the whole 128-page batch via break-on-failure,
  `cache_controller.py:1197-1202`) — required for idempotent election (§4.2).

## 3. The canonical grid

All KV bytes of one logical page form the model-global tensor
`K/V[global_layer, token_in_page, global_kv_head, head_dim]` (coordinate
convention identical to `DESIGN_unified_l3_kv_cache.md` §5.1). The grid cuts it
into **cells** along the two dimensions parallel configs actually shard:

| Grid axis | Sharded by | Canonical unit | How to pick |
|---|---|---|---|
| `head_group` (Hg) | TP (MHA/GQA) | `num_kv_heads / tp_lcm_size` heads | subsumes the existing `should_split_heads`/`tp_lcm_size` mechanism (`mooncake_store.py:580-593`) into the descriptor — no more free-text extra-config knob |
| `layer_group` (Lg) | PP, CP-layer-split | contiguous layer range; **must divide `num_layers`** (no ragged tail) | must divide every deployed stage/shard size; `Lg = num_layers` when the fleet runs one partition |
| `token_granule` (Tg) | CP page variants, DCP | **whole page** in v1 | sub-page residue classes are the defined v2 extension (§8) |

A cell = one dense rectangle `(layer group i) × (all page tokens) ×
(head group j)`, K+V fused into one object (§2, §4.4).

**Grid selection is a fleet policy knob, and fleet-matched grids cost nothing
new.** Sizing (fp16, page 64, head_dim 128; "today" = current per-rank mooncake
objects for the same fleet):

| Model / fleet | Grid | cells/page (fleet-wide) | cell size | today's objects |
|---|---|---|---|---|
| 70B GQA (80L, 8 kvh), TP∈{4,8} | Hg=1, Lg=80 | 8 | 2.5 MiB (K+V) | 16 × 1.25 MiB (k,v split) — **same bytes** |
| same fleet + PP2 | Hg=1, Lg=40 | 16 | 1.25 MiB | 32 × 640 KiB — same bytes |
| MLA 61L (DeepSeek), any TP | Lg=61 | 1 | ~4.3 MiB | 1 × 4.3 MiB — identical |
| portability maximum (prior ABI) | Lg=8†, Hg=1 | 160 | 128 KiB | — (10× object count) |

† illustrative only — violates the divisibility rule for 61-layer models and
the fragment floor (§4.4) for most layouts; fine grids below the deployed
topology are gated on the group-exists dependency (§5.2).

For grids that match the deployed topologies (the recommended regime), cells
are **byte-identical to today's objects** — the design changes names, not
object population. Amplification appears only when a grid is finer than every
deployed config, and §4.4/§5.2 bound how fine v1 allows.

### 3.1 Worked example: TP2 × PP2 × CP2 prefill, TP4 × PP1 decode

GQA 70B (80 layers, 8 KV heads, head_dim 128, bf16, page 64). Fleet: prefill
TP2×PP2×CP2 (8 GPUs/replica), decode TP4×PP1. Descriptor derivation:
`Hg = 8 / lcm(2,4) = 2` heads → `H0..H3`; `Lg = gcd(40, 80) = 40` → `L0,L1`;
`Tg` = page. 8 cells/page × 2.5 MiB (K+V fused) = the page's 20 MiB.

Every object name for one page (hash `e3b0…41`, namespace digest `7c9d…`):

```text
7c9d…:e3b0…41:kv:L0:H0     layers 0–39,  heads 0–1, K+V, 2.5 MiB
7c9d…:e3b0…41:kv:L0:H1     layers 0–39,  heads 2–3
7c9d…:e3b0…41:kv:L0:H2     layers 0–39,  heads 4–5
7c9d…:e3b0…41:kv:L0:H3     layers 0–39,  heads 6–7
7c9d…:e3b0…41:kv:L1:H0     layers 40–79, heads 0–1
7c9d…:e3b0…41:kv:L1:H1     …
7c9d…:e3b0…41:kv:L1:H2
7c9d…:e3b0…41:kv:L1:H3
```

No rank, size, tp/pp/cp anywhere — the "suffix" is the canonical coordinate.
Writer map on the prefill side: rank (pp=p, tp=t) materializes cells
`L{p} × {H2t, H2t+1}`; the CP pair resolves per CP mode — replicated-at-rest
(zigzag): elected writer `cp = hash(P) % 2` per cell; page-range sharded
(`kv_reshard`): the home rank of page P writes, the peer writes nothing.
Either way exactly one GPU writes each of the 8 cells, with zero
communication.

Reader map on the decode side (TP4, no PP, no CP): rank t′ owns heads
`[2t′, 2t′+2)` = exactly `H{t′}` and all layers, so it fetches
`…:kv:L0:H{t′}` and `…:kv:L1:H{t′}` — two gets per page, no reshard, no
collective. Contrast with today's names for the same deployment, where this
read is impossible: mooncake writes `{hash}_{tp_rank}_{pp_rank}_k/_v` (no
tp/pp *size*, CP ranks race on the same key — §1 hazards 1–3), and the file
backend writes `{hash}_{model}_{t}_2_2_{p}_cp{c}_2.bin`, unreadable by any
other topology. (MLA variant of this example: the H axis collapses — 2 cells
per page, `…:kv:L0` and `…:kv:L1`, elected writer per CP/TP replica set.)

## 4. Write path (unchanged shape, new names)

Per rank, exactly as today (`cache_controller.py:1186-1227` backup thread,
zero-copy `batch_set_v1`), except the key builder emits cell keys:

**4.1 Owned-cell enumeration** (init-time, static): from the rank's
`(attn_tp_rank/size, pp layer range, cp mode)` compute the cells it
materializes locally. TP=4 rank 1 with Hg at lcm 8 owns H2,H3; PP2 stage 1 with
Lg=40 owns L1; layer-split CP rank r owns its layer range's L groups.

**4.2 Replicated cells elect one writer, deterministically, without
communication**: `writer = replicas[page_hash % len(replicas)]`.

- Covers MLA-under-TP (today rank 0 always writes — the "todo: load balancing"
  at `cache_controller.py:464-469` becomes hash-spread; healthy-case math:
  512 pages over 8 ranks = 64 ± 7.5 pages each, ≈6.5× faster drain than
  rank0-writes-all) and replicated-pool prefill CP.
- Election is safe *only together with* the exists-is-success fix (§2.3): a
  failover double-write must be absorbed, and duplicate payloads are
  byte-identical only within one namespace (`numerics_id` pinned, §6.2).
- **Known regression vs rank0-always, accepted with mitigation**: page hashes
  are i.i.d. mod N, so one *stalled* writer rank punches holes every ~N pages,
  and consecutive-prefix hit semantics (`cache_controller.py:1034-1041`)
  collapse a 512-page hit to ~N pages even though (N−1)/N of the bytes landed.
  Mitigation is the existing retry path: pages past the truncated
  `completed_tokens` stay unmarked in the radix tree and are re-attempted by
  the normal backup policy; a per-attempt salt in the election
  (`hash(page_hash, attempt)`) lets retries route around a persistently slow
  rank. Note also the ack-tail coupling: completion becomes gated on the
  slowest of N writers instead of rank 0 alone (MIN-ack machinery,
  `hiradix_cache.py:996-1021`).
- Per-op accounting must change with election: a rank's `completed_tokens` is
  the longest prefix whose *owned* cells all succeeded (sparse ownership means
  the existing "skip-all, ack-full-length" MLA shortcut
  (`cache_controller.py:1222-1224`) and the unconditional ack enqueue must be
  reworked, not just the key builder).

**4.3 Sharded pools write home-only** (`kv_reshard`-style page-contiguous CP,
page-stripe draft): a rank puts full-depth cells only for pages it homes — the
KV-MMU doc's "permanent-tier writes are home-only" rule. The page's cells at
that rank are complete because the home rank holds all layers × its heads of
that page.

**Write-traffic accounting, scoped honestly**: on the *file backend* with a
replicated pool, today each CP rank persists a full copy under `_cp{r}_{s}`
suffixes → election reduces persisted capacity by cp_size×. On *mooncake*
persisted capacity is already ~1× (all CP ranks target one key; the client-side
exists-check dedups late writers) — election removes up-to-cp_size× *wire*
traffic in the common lockstep race window, not stored bytes. For sharded pools
there is no duplicate to remove. Cross-rank backup skew also means partially
written pages are a *steady state*, not just a crash artifact (per-rank backup
threads have no cross-rank put gating): those bytes are unreadable by every
topology until the laggard catches up, a capacity/hit-rate tax that bounding
backup-queue skew across the attn group keeps small.

**4.4 Physical mapping (zero-copy)**:

- Head-group cells require the `page_head` host layout —
  `(2, page, head, page_size, layer, head_dim)`, `pool_host/mha.py:147-155` —
  where a (head-range × all owned layers) slice is one contiguous range per
  K/V half (`get_split_heads_page_buffer_meta` pointer math,
  `mha.py:509-526`). The **default layout is `page_first`**
  (`server_args.py:2601`), under which a 1-head cell shatters into
  `layers × page_size` 256 B fragments — unusable. So `canonical-grid` for MHA
  *mandates* `page_head`, normalized in `_handle_hicache` exactly like the
  existing mooncake `layer_first` rewrite (`_resolve_storage_layout_compatibility`,
  `server_args.py:7187-7206`), and the `page_head` L2 device↔host transfer path
  must be benchmarked against `page_first` before default-on.
- Fused K+V cells = 2 fragments per object via the scatter-gather put path
  (`batch_put_from_multi_buffers`, call site `mooncake_store.py:1294-1298`) —
  mooncake-only today; backends without multi-buffer store split `:k`/`:v`
  objects and accept the splice caveat (§6.2), or copy through the flat-page
  path (file backend).
- Layer-sliced cells (Lg < the rank's layer count) fragment `page_head` into
  `heads_in_group × page_size` strided pieces (64 × 2 KiB at Lg=8/fp16 —
  message-rate-bound, the real throughput cliff). A layer-sliced grid therefore
  **requires** a `page_head_layer` layout
  (`(2, page, head, layer, page_size, head_dim)`) — one pointer per cell when
  Hg is a single head, `heads_in_group` pointers otherwise. Not optional.
- **Fragment floor guard**: refuse grids whose contiguous fragment size falls
  below ~64 KiB. At ≥128 KiB objects with a few hundred per batch, mooncake
  overhead is a survivable ~1.5–2×; below the floor it is message-rate-bound.
- Asymmetric-K/V MHA models (e.g. MiMo-V2) hard-reject `page_head`
  (`mha.py:1026-1030`) — excluded from head-group cells (§8).

## 5. Read path

Prefetch (`_storage_hit_query`, `cache_controller.py:1023-1045`) changes only
in what it enumerates:

**5.1** Rank computes page hashes (unchanged) and expands each page to **the
cell keys covering its own rectangle** — `lcm/tp_size` head groups × its layer
groups. It checks/fetches *only its own cells*.

**5.2** `batch_exists` over cell keys; a page hits iff all cells this rank
needs exist; hit length = consecutive fully-covered pages. Cost, 32k-token
request (512 pages, `STORAGE_BATCH_SIZE=128`):

| Config | keys/page (per rank) | keys/request | vs today |
|---|---|---|---|
| TP8 reader, fleet grid Hg=1/Lg=80, fused KV | 1 | 512 | 0.5× (k,v were 2) |
| TP4 reader, same grid | 2 | 1,024 | parity with split-heads |
| no-PP reader of Lg=40 grid | 2 | 1,024 | 2× |
| TP8 reader, Lg=8 fine grid | 10 | 5,120 | 10× |

Supported v1 boundary: **cells-per-rank-per-page ≤ 4** unless the store grows a
group-membership existence query. Mooncake's `ReplicateConfig.group_ids`
(`mooncake_store.py:723-732`) is today a put-side placement tag only — there is
no group-exists RPC in the client API (`_batch_exist` is per-key,
`mooncake_store.py:1315-1316`), so that optimization is an **upstream
dependency**, not an available escape hatch.

**5.3** Cross-rank agreement must span **every rank that computes its own hit
length — including PP stages**, at **all four** reconciliation points, with a
true full-group reduce (gather-min-broadcast or a gloo group over
attn×CP×PP), *not* the existing `_pp_sync` pattern
(`hiradix_cache.py:260-303`), which is one-directional PP0→PPn and cannot
express "stage 1 is missing its cells":

1. exists-time hit count (`_all_reduce_prefetch_groups`,
   `cache_controller.py:1068`) — today attn-only;
2. prefetch termination votes (`can_terminate_prefetch` MAX,
   `hiradix_cache.py:1581-1591`) — today attn-only;
3. completed-token clamp (`_sync_and_clamp_prefetch_result` MIN,
   `hiradix_cache.py:1679`) — today attn-only; this is the point that
   reconciles exists→get races (a cell evicted between exists and get surfaces
   as a get failure and must clamp *all* stages, or PP stages permanently
   disagree on the radix prefix);
4. the timeout decision (`is_prefetch_timeout`, `hiradix_cache.py:1568`) —
   per-rank wall clock today; stages can revoke divergently.

Cost note: a blocking PP-wide reduce in the prefetch thread couples stages
across pipeline skew (stage 0 waits for stage N−1 to reach the same op). The
reduce is a scalar and prefetch is already asynchronous to the forward pass,
but the added latency per hit query is pipeline-depth-dependent and must be
measured; PP fleets that can't pay it should run PP-private namespaces
(honest fallback, still suffix-free within each stage set).

**5.4** `batch_get_v1` lands cells via the same buffer metas as writes, with
the §2.3 size assertion; L2→L1 load-back unchanged.

Cross-topology reads fall out with no extra machinery:

| Writer → Reader | Reader fetches | Extra comm |
|---|---|---|
| TP8 → TP4 (MHA) | 2 head-group cells per page | none |
| TP4 → TP8 | 1 cell (lcm-granular by construction) | none |
| PP2 → PP4 | its half of the writer stage's L groups | none |
| CP replicated / page-sharded → anything | full-depth cells | none |
| DP replicas | identical cell space (attn-group scoped, as today) | none |
| MLA any-TP → any-TP | the single latent cell | none |

## 6. Consistency without manifests

What the prior ABI doc buys with plans/receipts/`CommitAndPublish` is atomic
multi-writer publication. V1 deliberately relaxes to **reader-scoped
completeness** and states the residual anomalies:

**6.1 Missing cells, not wrong bytes.** A torn page (writer died or lags
between cells) manifests as missing cells → shorter hits for readers that need
them. This claim holds only with §2.3's size assertion (the store itself does
not enforce it) and rests on one assumption about the external store: puts are
atomically visible (never a partial object). That is a property of Mooncake's
put pipeline, asserted here as a requirement on any qualifying backend, not
something this repo's code enforces.

**6.2 Cross-producer assembly.** Different cells of one page may come from
different producer processes — by design (a TP8 fleet and a TP4 fleet elect
different writers for the same replicated cell; MHA fleets write disjoint
head cells of shared pages). Three containments: (a) `numerics_id` is
**mandatory** in the descriptor — same kernels/build/dtype or no shared
namespace; (b) K and V of a rectangle live in one fused object, so no
K-from-A/V-from-B splice within a head-group (backends stuck with split `:k`/`:v`
objects retain that window and must document it); (c) the residual — cells of
one page from producers whose bf16 outputs differ at rounding level due to
batch-shape-dependent reduction order — is *already accepted today*: DP
replicas race idempotent writes of identical keys under the current scheme.
This is an honest tolerance statement, not a proof; fleets that can't accept it
need the manifest layer (per-producer publication units), which is exactly the
v2 upgrade path.

**6.3 Eviction is the dominant anomaly source, not crashes.** The store evicts
cells independently (mooncake master-side LRU; `group_ids` give
group-*coalesced*, best-effort eviction — a group is considered only once every
member's lease has expired, and members are then evicted together *except*
pinned/in-use ones (`try_evict_group_or_object`,
`master_service.cpp:6651-6709`) — useful, but not all-or-nothing atomicity).
Consequences and requirements:

- One evicted cell truncates every reader chain at that page
  (consecutive-prefix semantics) — an availability cliff that grows with
  cells/page; another reason for the §5.2 fineness cap. The store-side ask:
  group-aware eviction (evict a page's group together, oldest-chain-first), or
  accept and monitor the truncation rate.
- Evictions between exists and get surface as get failures → handled by the
  §5.3 clamp, which is why it must span PP.
- The **file backend's** two local subsystems assume per-rank key ownership and
  break under shared keys: the positive-existence `MetadataCache`
  (`hicache_storage.py:327-357`; a peer's eviction leaves a permanent false
  positive under `ttl=-1` → permanent hit cliff at that page) and
  `LRUFileEvictor` (per-suffix accounting, `hicache_storage.py:420-431`; N
  suffix-free evictors double-count and race deletes). Under `canonical-grid`:
  invalidate metadata-cache entries on get failure, forbid `ttl=-1`, and make
  eviction single-owner (only one elected rank evicts, mirroring §4.2).

**6.4 Descriptor lifecycle.** Because the descriptor is config (§2.1), there is
nothing in-store to evict or race. `clear()`/flush semantics must become
namespace-scoped (`remove_all()` today nukes every namespace and epoch —
unacceptable once namespaces share a store).

When stronger guarantees become necessary (multi-tenant trust, codec/quantized
variants, measured torn-page or mixing rates worth fencing), the full artifact
ABI layers on: v1 cells are exactly its dense-rectangle chunks; manifests are
additive metadata over already-written objects. Nothing stored under v1 moves.

## 7. Implementation shape

1. **`CanonicalKeyBuilder`** beside `hicache_storage.py`: owns the descriptor
   (msgspec, canonical encoding, digest), grid math, owned/needed-cell
   enumeration, writer election, and the attach compatibility check. KV-cell
   key construction leaves the backends; v1 backends are **mooncake (+ file
   for development)** — nixl/eic/hf3fs/simm/umbp reject `canonical-grid` until
   they grow cell-granular IO (their v0 paths are untouched).
2. **Host-pool cell metas**: generalize `get_split_heads_page_buffer_meta`
   (`pool_host/mha.py:490-536`) to
   `get_cell_buffer_meta(indices, layer_groups, head_groups)`; MLA variant is
   the existing whole-page meta. Add `page_head_layer` when layer-sliced grids
   are needed (§4.4).
3. **Controller**: cell enumeration in `_storage_hit_query` / `_page_backup`;
   election replaces `backup_skip` *plus* the sparse-ownership ack/completed-
   tokens rework (§4.2); the four sync points extended per §5.3.
4. **Flag**: `--hicache-storage-key-scheme {rank-suffix, canonical-grid}`
   (descriptive values; avoids colliding with `--hicache-mem-layout`), an
   annotated `NS("memory")` field read via the memory config bag; the
   descriptor arrives as `--hicache-storage-namespace-descriptor <path/URI>`.
   Layout normalization (`page_head` for MHA head-group grids) and the guards
   below live in `_handle_hicache` next to the existing DCP guard
   (`server_args.py:7128-7139`).
5. **Guards to ship regardless of scheme** (they fix live v0 hazards):
   get-size assertion (§2.3); exists-is-success in `_page_backup` (§2.3);
   reject attn-CP + non-file backends under `rank-suffix` (hazard 3); reject
   PP>1 on nixl/eic/hf3fs under `rank-suffix` (hazard 2); fix the split-heads
   list-interpolation keys (hazard 5).
6. **Guards for `canonical-grid` v1**: models with mamba/SWA/indexer/DSV4/draft
   side pools rejected (§8); NSA CP rejected (sub-page slices, §8); asymmetric
   K/V MHA rejected (§4.4); grids finer than cells-per-rank-per-page > 4
   rejected (§5.2); fragment floor (§4.4).

## 8. Explicitly out of v1 (and the defined path in)

- **Side pools / hybrid models** (mamba `_temporal`/`_conv`, SWA, indexer,
  DSV4 pools, spec-decode draft). Not orthogonal: mamba states shard on
  cat([q,k,v]) sub-blocks and GDN value-head axes
  (`memory_pool.py:843-846, 1082-1102`), Kimi conv on a per-slot axis — none
  nameable by kv-head groups; the draft pool is a *different model* (own layer
  count/head count/MLA-ness, `mooncake_store.py:761-774`) whose identity the
  descriptor's single `model_id` does not pin; SWA/state components use
  `TRAILING_PAGES` hit policy (`hicache_storage.py:82-89`), not the
  ALL_PAGES semantics §5 describes. V2 path: per-component grid descriptors
  (component → shard-axis descriptor + shard count, reusing the sub-block
  slicing PD transfer already implements, e.g. `get_state_conv_shard_groups`)
  plus a `draft_model_id` field, and policy-aware hit accounting.
- **DCP / NSA within-page interleave** (`pos % dcp_size` ownership,
  `layers/dcp/layout.py:29-51`): a rank's shard of a page is a strided residue
  class, not a dense rectangle. Already hard-blocked for L3
  (`server_args.py:7128-7139`); stays blocked. V2 path: token-granule
  descriptors `T{r mod d}` + local re-striding during the L3→L2 copy — the
  operation `build_dcp_token_transfer_plan` already performs for P/D
  (`disaggregation/common/utils.py:139-198`). Still no collectives.
- **Quantized / mixed-dtype variants**: new representation ⇒ new namespace in
  v1; the manifest layer is the long-term answer.
- **Partial-page tails, intra-page hits**: unchanged from today.
- **`RadixKey.extra_key`** (LoRA / multimodal salt) never enters the page hash
  today — decide whether it becomes a descriptor field or a hash-chain input
  before broad cross-deployment sharing. (Open question; applies to v0
  equally.)

## 9. Migration and grid evolution

The blunt truth first: **changing the grid means a new namespace, and a new
namespace is a cold cache** for every deployment that moves. The common fleet
event — adding a PP2 deployment to a live `Lg=80` namespace — forces exactly
this. V1 policy and the softening paths:

1. **Pre-provision at bring-up**: choose the finest grid any *anticipated*
   deployment needs (the cost of running Lg=40 under a no-PP fleet is one
   extra cell per page — §3 shows fleet-adjacent grids stay in the same object
   size class). This is the recommended posture; the descriptor forces the
   conversation at fleet-design time instead of incident time.
2. **Read-old-fallback (v1.5)**: a coarse cell *contains* its finer same-axis
   sub-cells, but not contiguously — under the `page_head` cell layout a
   finer-layer half is contiguous only per (head, token), i.e. ~`page_size`
   fragments per head — which is exactly why this path repacks through a
   bounded staging buffer rather than splitting bytes in place. A reader that
   misses in the new epoch may look up the old epoch and repack through that
   staging buffer
   (zero-copy relaxed only for migration reads), write-through to the new
   epoch, and let the old epoch drain by LRU. Per-namespace hit/occupancy
   metrics make drain progress visible.
3. **Old-epoch GC**: bulk delete by key prefix or group ids
   (`mooncake_store.py:723-732`) once drained — requires namespace-scoped
   delete (§6.4) instead of `remove_all()`.

## 10. Rejected alternatives

| Alternative | Why not |
|---|---|
| **Status quo** (per-config suffixes) | No cross-topology reuse by construction; seven divergent implementations with live TP/PP/CP corruption hazards (§1); file-backend replicated CP persists cp_size× copies. |
| **Literal Q1: one key, one object, writer-config metadata** | Readers must understand every producer topology; different-config producers collide under one name or need read-modify-write on shared metadata (race); no target store has a mutable metadata plane. |
| **One key, composite object, ranks write at offsets** | Needs partial-write/read-at-offset and object preallocation sized before the first writer; breaks the write-once/idempotency model backends rely on; eviction of the composite is all-or-nothing across topologies. |
| **Gather-then-write (allgather or rank-0 gather)** | The thing Q2 forbids: cp/tp_size× NVLink/IB traffic plus a write hotspot; also loses information — a gathered page must be re-split for any sharded reader. |
| **Full artifact ABI now** (manifests, coordinator, transactional metadata) | Right end-state for multi-tenant/durable tiers; but v1 needs none of its guarantees to beat the status quo, and it demands an etcd-class metadata service before the first byte lands. Grid cells are its chunks; adopt incrementally when §6's tolerances stop being acceptable. |
| **Megatron-Core dist-ckpt / PyTorch DCP mechanism** (per-artifact `.metadata` index) | Same *principle* (topology-free identity + model-global rectangles), different *mechanism* — index-based, not name-based. Transplanted to a cache it either puts a coordinator collective on the serving path or converges to a weaker rebuild of the manifest ABI. See Appendix A. |

## Appendix A. Comparison: Megatron-Core dist-checkpointing / PyTorch DCP

(Read from source: `megatron/core/dist_checkpointing/` and
`torch/distributed/checkpoint/`; citations are to those trees.)

**MCore validates the core principle.** `ShardedTensor` identity is a
topology-free `key` string plus model-global geometry (`global_shape`,
`global_offset`); TP/PP/DP enter only as offsets and a `replica_id`
(`mapping.py:81-91`). A TP=4 writer and TP=2 reader declare the same
`(key, global_shape)` with different fragmentations, and load intersects the
reader's rectangles with stored chunks (`planner_helpers.py:280-400`,
sweep-line + `_shards_get_overlap_region_wrt_saved_tensor`). Save moves no
tensor data between ranks — only plan metadata is gathered. That is exactly
this design's "topology is a runtime placement, not identity."

**But the mechanism is index-based where the grid is name-based.** In DCP,
object names are meaningless (`__{rank}_{n}.distcp`); the authority is one
central pickled `.metadata` per artifact mapping `(fqn, chunk offset)` →
`(file, byte offset, length)` (`metadata.py:113-150`,
`filesystem.py:762-798`), committed by an atomic tmp+fsync+rename on the
coordinator after gathering all ranks' `WriteResult`s. Everything DCP is good
at flows from that index; everything that breaks in an L3 cache flows from it
too:

| Cache requirement | DCP/MCore behavior |
|---|---|
| pages seal continuously (~10³/s/node) | commit protocol (plan gather → write → result gather → rename) runs once per *artifact*; object stores have no rename |
| content-addressed identity (cross-job dedup) | path + per-save uuid; two identical saves are unrelated artifacts |
| longest-prefix `batch_exists` at request latency | no probe path — locate artifact, unpickle *entire* `.metadata`, then look up |
| store LRU-evicts individual objects | a dangling index entry is a load *exception*, not a miss; single-pickle index has no CAS/merge |
| a prefix of pages is useful | fully-parallel load raises on any missing shard (`fully_parallel.py:280-295`) |
| many independent engines share one namespace | save is a collective with a coordinator; cross-job writes have no story |

Shrinking the artifact to one page to fix these yields: immutable cell blobs +
one small atomically-published index object per page — i.e. **the manifest
plane of `DESIGN_unified_l3_kv_cache.md`, rebuilt worse** (pickle instead of
canonical CBOR, rename-commit instead of digest-addressed manifests, path
identity instead of content identity). So "use the MCore mechanism" is not a
third option; it is the §10 artifact-ABI row, and the grid's §6 tolerances are
exactly what it would buy back.

**Worth adopting from MCore/DCP into this design:**

1. `ShardedTensor`-style rectangle declaration as the *internal* API of
   `CanonicalKeyBuilder` — `from_rank_offsets((axis, rank, fragmentations))`
   is precisely owned-cell enumeration, and a `validate_sharding_integrity`
   analogue (overlap/gap check over all ranks' declared rectangles) is the
   right debug-mode attach test.
2. `replica_id` / `is_main_replica` vocabulary for writer election — and an
   upgrade path for §4.2: MCore's greedy size-balanced distribution
   (`exchange_utils.py:117-170`) is deterministic given the replica map; it
   needs an `all_gather_object` only because checkpoint shard sets are
   dynamic. KV cell ownership is static SPMD knowledge, so the same
   deterministic greedy assignment is computable with **zero** communication —
   hash election is its degenerate case, weighted election a drop-in upgrade.
3. Fully-parallel *load* (read-once + interconnect broadcast,
   `exchange_utils.py:461-542`) as an optional read-amplification
   optimization for replicated cells: an MLA cell is needed identically by all
   attn-TP ranks; one rank could GET and NVLink-broadcast, cutting L3 read
   traffic by the TP degree. Optional because it reintroduces a collective on
   the load path — if adopted, it must ride the existing prefetch batch
   boundary, not create a new lockstep point.
4. The async save pipeline internals (staged D2H overlap, size-balanced write
   binning, deprioritized writer process) as engineering reference for the
   backup thread.

**Where DCP/MCore is genuinely stronger** — the honest cost of name-based:
arbitrary uneven rectangles with no divisibility constraint
(`ChunkStorageMetadata` is raw offsets+sizes; MCore allows irregular grids and
`allow_shape_mismatch`); true atomic publication (no torn-page semantics at
all); and **no grid-evolution problem** — every artifact self-describes its
sharding, so differently-chunked writers coexist forever, where the grid pays
§9's namespace-migration cost. Those three are precisely what the grid trades
away for indexless probes, computable names, per-object eviction, and a
coordinator-free write path.

## Appendix B. IO fragmentation: survey of chunked-storage designs and the packing options

(Sources: Zarr v3 sharding-indexed spec, TensorStore OCDBT docs, Orbax
optimized-checkpointing guide, HF safetensors docs; local source reads of
LMCache @`622e1464` and Mooncake; citations below.)

### B.1 The universal pattern

Every system that stores fine-grained tensor chunks on object stores converges
on **pack many logical chunks into one large physical object + keep a
byte-range index somewhere**. They differ only in where the index lives:

| System | Pack unit | Index location | Writers per pack | Read path (cold) |
|---|---|---|---|---|
| Zarr v3 `sharding_indexed` | fixed grid of inner chunks per shard | **footer inside the shard** (fixed-size `(offset, nbytes)` array) | one | 2 range GETs (1 with cached index) |
| TensorStore **OCDBT** | log-structured ~2 GiB data files | **distributed B+tree** + CAS-committed manifest | many (conflict-free data plane; contention only at manifest commit) | 3–4 GETs cold, ~1 range GET amortized |
| HF sharded safetensors | ~5 GB pack-by-size files | external `index.json` + self-describing per-file header | one (write-once artifact) | ~3 round trips |
| PyTorch DCP (App. A) | per-rank `.distcp` files | central `.metadata` pickle | collective, coordinator | unpickle index + range reads |
| **LMCache** | fused K+V, all layers, **256-token chunk**, per TP rank | **none — name-addressed** (chunk-hash keys) | one per chunk | 1 GET per chunk |
| **This design (v1)** | one cell (page × layer-group × head-group), K+V fused | **none — name-addressed** | one per cell | 1 GET per cell |

Two name-addressed systems exist because they gave up different things:
LMCache avoids fragmentation by **coarsening the token granule to 256** and
fusing all layers + K/V into one blob — but bakes `world_size` and `worker_id`
(global TP rank) into every key (`lmcache/utils.py:399-456`), so there is *no
cross-TP reuse at all* (grep confirms zero resharding code; the only exception
is an MLA rank-0 hack, `token_database.py:112-121`). The grid keeps
cross-topology reuse and controls fragmentation by grid coarseness instead.
LMCache also validates two of this doc's mechanisms in production: fused K+V
objects, and NCCL-broadcast-on-retrieve for MLA replicas
(`cache_engine.py:856-921`) — Appendix A's read-once+broadcast option.

### B.2 What Mooncake can actually do (changes the option space)

From source (`Mooncake/mooncake-store/`):

- **Range/offset reads are first-class** for memory replicas:
  `get_into_ranges` reads arbitrary `(src_offset, size)` fragments of many
  objects into registered buffers in one call
  (`store_py.cpp:2710-2733`, `transfer_task.cpp:1256-1289` — RDMA reads at
  `buffer_address_ + src_offset`). Disk replicas degrade to full-read+scatter
  (`real_client.cpp:3229-3294`). In-tree precedent: **Engram** stores
  embedding tables as one object per head and range-reads rows at computed
  offsets (`engram_store.cpp:75-133`) — pack + computed-offset reads is a
  supported pattern, not a hack.
- **Partial/offset writes do not exist**: puts are whole-object, single-writer,
  two-phase (`PutStart`→transfer→`PutEnd`); a concurrent `PutStart` on the same
  key gets `OBJECT_ALREADY_EXISTS` (`master_service.cpp:3148-3156`). But
  `put_parts` assembles one object from many local spans
  (`real_client.cpp:1856-1946`) — a writer can build a pack zero-copy from
  scattered per-cell host pointers.
- **Small objects have hard costs**: ~0.3–0.6 KB master metadata per key
  (`master_service.h:856-922`); eviction is an O(total-objects) scan per cycle
  (`master_service.cpp:6737-6772`); and the default OFFSET allocator caps
  **~1M live allocations per mounted segment**
  (`offset_allocator.h:141-142`) — 64 KiB objects on a 64 GiB segment hit that
  cap exactly. Packing attacks precisely these three costs.
- **`group_ids` provide group-atomic eviction**, not just placement: one
  unexpired lease protects the whole group, and eviction takes the group
  together (`try_evict_group_or_object`, `master_service.cpp:6651-6709`);
  `batch_is_exist` refreshes leases group-wide (`master_service.cpp:2168-2170`)
  — exists-probing a prefix keeps it alive.

### B.3 The packing options for the grid, in order of preference

**Option 0 — fleet-matched grid (v1 default, no fragmentation by
construction).** §3's review-verified result stands: for grids matching the
deployed topologies, cells are byte-identical to today's per-rank mooncake
objects (0.6–4.3 MiB). Fragmentation only appears for grids finer than every
deployed config (capped by §5.2/§4.4 guards) or small-model + small-page
corners.

**Option 1 — coarsen the token granule (Tg > 1 page).** The LMCache answer,
already an axis of the grid. Object size and exists-RPC count scale by B,
object count by 1/B. Cost: hit and dedup resolution coarsen to B pages
(LMCache measured overlap-overwrite waste and drops up-to-255-token tails;
same trade here). No new mechanism needed — this is a descriptor knob.

**Option 2 — chain-segment packs with computed offsets (v1.5, mooncake-only).**
Keep per-page *read* resolution while writing B pages per object, exploiting
range reads. One rank owns its cell for all pages of a chain segment
(consecutive pages `[kB, (k+1)B)`), so the pack has a **single writer**
(`put_parts` from B per-page spans; no offset-write needed):

- **Key**: `{ns}:{page_hash of page kB}:kv:L:H:seg{B}` — keyed by the segment's
  *first* page hash, which any reader that reaches the segment can compute
  (keying by the last page's hash would make partial-segment readers unable to
  form the key).
- **Divergence handling**: first-page keying means two chains sharing the
  prefix but diverging inside the segment collide by design. The pack therefore
  embeds a fixed-size header: B × 32-byte page hashes at computed offsets (the
  Zarr footer pattern, with content hashes instead of byte extents — sizes are
  uniform, so extents stay computable). A reader range-reads the header,
  compares against its own computed chain hashes, and treats the first
  mismatching page as the end of the hit. First-writer-wins means a divergent
  branch cannot store its own tail until the next segment boundary — dedup
  coarsens to segment granularity on the *write* side while reads stay
  page-granular for the stored branch. For long-shared-prefix workloads
  (the CP target) most segments are fully shared.
- **Eviction/lease**: per segment — B-page aligned holes instead of random
  single-cell holes, and one lease covers B pages (`group_ids` can bind the
  L/H sibling packs of the same segment into one eviction unit).
- **Costs**: memory-replica-only for true range reads (disk tier degrades to
  full-segment reads); header adds one small fragment per get (coalesceable
  with the data read when the needed pages are contiguous from the header);
  partial tail segments follow the same rule as today's partial pages (not
  stored until sealed, or stored short with size-prefix validation).

**Option 3 — an OCDBT-class index tier.** If per-page dedup *and* packing
*and* cross-branch storage must all coexist, that is a mutable
logical→physical index over immutable packed data — the manifest plane again.
Appendix C gives the concrete design sketch (architecture, commit model,
eviction/GC, escalation triggers) and the mooncake-specific implementation
choice.

### B.4 CP placement compatibility (Options 1 and 2 require contiguous ownership)

Both packing options assume the writer locally holds B *consecutive* pages of
the chain. That holds for zigzag/interleave prefill CP (KV replicated at rest
— the elected writer holds everything; elect per segment) and for contiguous
page-range CP sharding (`kv_reshard`-style — align segment boundaries to the
range partition), but **breaks under per-page round-robin striping**
(`dsa_cache_page_stripe.py`): pages p and p+1 always home on different ranks,
so no B≥2 granule has a single writer, and gathering is forbidden. Note the
hashes are never the constraint — token ids are SPMD-global, every rank
computes every page hash; only the KV bytes are striped. Today the conflict is
vacuous in practice (page-stripe targets DSA/MLA-family models whose cells are
already 1–4.3 MiB single-H blobs — no fragmentation problem), but the rule
should be explicit:

- **Fix A (preferred): block round-robin.** Stripe CP homes in blocks of B
  pages with B = the pack/granule size — `CpAlignedPagedTokenToKVPoolAllocator`
  already allocates atomic cp-aligned groups; generalize the group to
  `cp × B` pages. Balance at 4–8-page blocks is indistinguishable for
  long-prefill workloads. Descriptor rule: deployed stripe units must be a
  multiple of Tg / the segment size — the token-axis analog of the Hg
  divisibility rule.
- **Fix B: canonical residue-class packs.** Declare a canonical stripe-class
  count `d` (LCM of deployed cp sizes); pack = one residue class's pages
  within a W-page window, key `{ns}:{window-start hash}:kv:L:H:T{j}` — a rank
  under cp with `cp | d` owns `d/cp` classes, exactly like TP ranks owning
  multiple head groups. Mismatched readers fetch several class packs and
  re-stride locally during the L3→L2 copy (`build_dcp_token_transfer_plan`
  precedent). Costs: windows d× larger for the same pack size, a local
  re-stride for mismatched topologies, one more frozen descriptor constant.
  This is the same mechanism as §8's v2 DCP extension — one design serves
  both.
- **Stopgap without packing:** `group_ids` can bind a window of per-page cells
  into one eviction/lease unit — fixes the availability half of fragmentation
  (eviction holes, lease churn) while leaving per-object master metadata and
  the allocation-count cap unaddressed.

The recommendation: ship Option 0 with the §4.4/§5.2 guards; expose Option 1
as the descriptor knob it already is; prototype Option 2 behind the mooncake
backend when a fleet actually hits the fine-grid regime — adopting Fix A's
block-striping rule at the same time if striped CP is in that fleet; treat
Option 3 as the same decision as Appendix A's manifest ABI.

## Appendix C. Option 3 concretized — the index tier design sketch

> Rev 2 after adversarial review. This is the escalation path, specified far
> enough to know what it costs and when to take it. Nothing here is v1 work.
> The honest headline up front: Option 3 buys packing with per-page dedup and
> a self-describing physical layer — it does **not** buy page-level atomic
> publication for multi-writer topologies, and it introduces one new
> corruption channel (stale pack pointers) that needs two dedicated guards.

### C.1 What changes and what doesn't

Logical identity is untouched: cells keep their canonical keys
(`{ns}:{page_hash}:kv:L{i}:H{j}...`), the descriptor, the hash chain, the
owned-cell enumeration. What changes is the **physical mapping**: instead of
`logical key = one store object`, an index maps
`logical key → (pack_id, offset, length)`, and cells live packed inside large
immutable **pack objects**. Because v1 cells are exactly the prior ABI's
chunks, this appendix is an implementation profile of
`DESIGN_unified_l3_kv_cache.md`'s manifest plane — not a third architecture.

```text
              index tier (mutable)                 data tier (immutable)
  logical key ───────────────► (pack_id, off, len) ────► range-read into pack
  {ns}:{hash}:kv:L0:H3                                    pack = 64–256 MiB,
                                                          single writer, log-
                                                          structured append
```

One consequence worth stating first: the index only ever resolves keys the
reader can *name*. Enumerability (§5.1) survives untouched — which also means
Option 3 does **not** unlock writer-chosen arbitrary rectangles. A reader
cannot form the key of a rectangle it doesn't know exists; supporting uneven
shards would require a coordinate-range overlap query on the probe path — the
Appendix A planner, with exactly the probe cost that disqualified it. The
divisibility rule relaxes only to *descriptor-versioned regular grids*
(multi-extent index entries per cell), never to free-form extents.

### C.2 Data plane: per-writer log-structured packs

Each rank appends the cells of a backup batch into its own uniquely-named pack
(`{ns}:pack:{writer_uuid}:{seq}`), sealed write-once. No writer ever contends
with another on a data object — the OCDBT/Orbax data-plane rule. Three
disciplines, all load-bearing:

- **Pack by eviction cohort**: a pack should hold cells that die together —
  same chain window, same age band — so eviction is one DELETE and repacking
  stays rare. Packing a backup batch (up to `STORAGE_BATCH_SIZE` = 128
  consecutive pages of hot chains) approximates this — **but only if writer
  election is window-aligned**. §4.2's per-page hash election is
  *incompatible* with cohort packing: page hashes are i.i.d. mod N, so a
  rank's elected pages are a pseudo-random ~1/N stride of the window, one
  window's cells scatter across N writers' packs, and evicting any one pack
  permanently removes every ~Nth page of a still-hot chain — the §4.2
  stalled-writer cliff replayed at every eviction, forever. Under Option 3,
  election moves to **per chain window** (hash of the window-start page,
  window = pack cohort), exactly as Option 2 already prescribes for segments.
- **Dedup before seal**: cross-fleet sharing means two fleets routinely back
  up the same hot prefix; first-publisher-wins per logical key would otherwise
  fill the loser's pack with dead-on-arrival bytes that its few winning cells
  then pin via the pack lease — a permanent capacity leak. One
  `batch_query` of the index for the batch's keys before building the pack
  drops already-published cells for the cost of one RPC.
- **64–256 MiB targets**, not OCDBT's 2 GiB: pack size is the eviction
  quantum — too large and one eviction punches a huge hole; too small and
  per-pack overhead returns.

Cost honesty on the build path: `put_parts` assembles one object from many
spans but is **not zero-copy** — it allocates a contiguous client staging
buffer of the total size and memcpys every span into it before transfer
(`real_client.cpp:1885-1904`). A pack build therefore costs one full staging
copy plus a pack-sized registered-buffer allocation; true gather-on-transfer
would be a new client capability.

### C.3 Index plane: mooncake-native, not literal OCDBT

The key realization from the Mooncake source: **OCDBT exists to put a mutable
index on a dumb object store (S3/GCS). Mooncake is not a dumb store — its
master is already a sharded, mutable, batch-RPC metadata service** holding
per-key `ObjectMetadata` (`master_service.h:856-922`) with group routing,
leases, and O(1) batched lookups. The natural Option 3 on mooncake is
**master-native indirection**: extend the master entry with an optional
`(pack_id, offset, length)` pointer (multi-extent list for fragmented cells).
Reads become: pointer resolution via an **extended batch-query-class RPC**
(same round-trip shape as today's `batch_is_exist`, but a new/extended RPC —
`BatchExistKey` returns only booleans) → `get_into_ranges` on packs. Two round
trips, matching today's exists+get, with data-object count reduced by the
packing factor.

Required master extensions — new capabilities, not configuration:

1. the pointer field and multi-extent entry schema;
2. an **atomic multi-key publish RPC** (today's batch mutations are per-key
   loops — `BatchPutEnd` etc.; group co-sharding and the single-shard lock
   give the building blocks, but nothing multi-key-atomic exists);
3. **replica-less entries as first-class**: the exists/lease/eviction paths
   are all gated on "has a COMPLETE replica" (`master_service.cpp:2168-2175`),
   so today a pointer entry would read as a miss, refresh no lease, and be
   skipped by group eviction — each gate needs a pointer-entry branch.

Fallbacks, for completeness:

- **Pointer objects** (zero store changes): each index entry as a tiny object
  whose value is the pointer record. Rejected as steady state: every pointer
  still consumes a master entry *and* an allocation node, relieving neither
  cost. But note the honest symmetry: **master-native entries don't relieve
  per-key master memory much either** — the ~0.3–0.6 KB/key is dominated by
  key string, map node, lease, and group refs, all of which remain. What
  master-native indirection actually relieves is the **allocation-node cap**
  (`offset_allocator.h:141-142`) and data-object count. If per-key master
  memory is the binding pressure, Option 3 is the wrong tool — coarsen Tg or
  use Option 2 segments.
- **Literal OCDBT-on-mooncake** (store-neutral deployments): B+tree nodes and
  numbered manifests as objects, using create-if-absent
  (`OBJECT_ALREADY_EXISTS`) as the commit primitive. Caveat from source: this
  is a *first-`PutStart`-wins lock with a ~30 s discard window*
  (`put_start_discard_timeout_sec_`, `master_service.cpp:3144-3170`), not a
  CAS on committed values — a winner that crashes between `PutStart` and
  `PutEnd` stalls the manifest chain for the window, during which losers
  cannot distinguish committed from in-flight and have nothing to rebase on.
  Safety holds (the late `PutEnd` fails `ILLEGAL_CLIENT`); liveness needs a
  reader-side timeout+retry protocol. Justified only where no master exists to
  extend.

### C.4 Commit model: per-writer atomicity — and only that

A writer seals its pack, then publishes all of that pack's index entries in
one atomic multi-key master update (capability 2 above). Be precise about what
this buys:

- **Within one writer**: multi-page, multi-cell all-or-nothing. A writer's
  batch is never half-visible; index and data lifecycles stay coherent.
- **Across writers: nothing changes.** Under TP/PP a page's cells come from
  *different* writer processes by design, and writers publish independently
  (the Orbax lesson — no cross-writer coordination on the hot path). A lagging
  rank still yields §6.1 torn pages, published straight through the
  per-writer-atomic index; cross-producer per-key first-publisher-wins still
  allows page P to assemble H0 from fleet A and H1 from fleet B (§6.2c is
  *not* closed). True page-level atomic publication for multi-writer
  topologies requires a cross-writer commit — the Appendix A coordinator —
  which is precisely what this design declines to put on the serving path.
  Readers therefore keep v1's reader-scoped completeness semantics unchanged.
- **An availability regression to manage**: publish-or-nothing means a writer
  crashing after its pack bytes land but before publish loses the *whole
  batch*, where v1's per-object writes keep the consecutive prefix that
  landed. Mitigation: publish in sub-batches (e.g. per 16–32 pages), trading
  a little atomicity granularity for bounded loss.

### C.5 Eviction, the stale-pointer race, and GC

- **Eviction unit = the pack**, LRU-ordered by pack lease; resolving any live
  cell refreshes the pack (the group-coalesced lease machinery generalizes
  *once* pointer entries pass the replica gates — capability 3). Evicting a
  pack deletes its index entries in the same master operation, so *subsequent*
  lookups miss cleanly.
- **The resolve→read race is Option 3's one new corruption channel.** A reader
  resolves `(pack_id, offset, length)`, the pack is evicted, and its segment
  space is reallocated to a new object before the RDMA range read fires: the
  read returns bytes of an *unrelated live object* — at exactly the requested
  length, so **the §2.3 size assertion does not protect range reads**. Two
  hard requirements: (1) **lease-through-read** — pointer resolution grants a
  pack lease whose term bounds the resolve→read window (and the window must be
  the prefetch-batch latency if resolutions from the exists phase are reused
  at fetch time); (2) **per-cell checksums embedded in packs** (Option 2's
  header pattern generalized), verified on read.
- **§5.3's four reconciliation points remain fully load-bearing** and matter
  *more*: one pack eviction invalidates thousands of resolved pointers
  mid-batch at once, and PP stages resolve independently — without the
  completed-token clamp spanning PP, divergent revocation becomes more likely
  per eviction event, not less.
- **GC**: dedup-before-seal (C.2) bounds dead bytes at the source; packs whose
  entry count reaches zero are collected by TTL; a minimal whole-pack rewrite
  (re-publish survivors into a fresh pack, idempotent by first-publisher-wins)
  is re-admitted as the *rare* path for measured mixed-pack leakage — cohort
  packing plus dedup-before-seal should keep it rare, but "compaction never"
  was not defensible.

### C.6 What it buys, what it costs, when to take it

**Buys** (relative to v1 + Options 1/2): packing with per-page dedup *and*
cross-branch storage simultaneously; per-writer multi-page atomic publish;
coherent pack-scoped eviction; and a softer grid-evolution story — **scoped
honestly**: head-axis refinements are index-only re-grids (head groups are
outermost in the cell layout, so finer H cells are contiguous sub-ranges;
multi-extent entries cover them); layer-axis refinements of `page_head`-era
packs and all coarsenings are **background repacks** (finer-layer slices are
~`page_size`-fragment strided — see §9.2), and token-granule changes likewise.
Better than §9's cold cache; not free.

**Costs**: an index write per backup batch and a resolution per lookup
(extended RPC); the three master extensions in C.3 — schema, atomic multi-key
publish, replica-less entry gates; one staging copy per pack build;
lease-through-read on the read path; per-cell checksums; and the loss of pure
name-addressed simplicity — backends without a master-class metadata service
participate only via literal-OCDBT with its liveness caveat. Per-key master
*memory* is explicitly not relieved (entry count is unchanged from v1; only
the allocation-node cap and object count improve).

**Escalation triggers** — take Option 3 when measurement shows any of:

1. a fleet genuinely needs cells-per-rank-per-page > 4 (fine grids beyond the
   §5.2 cap);
2. divergent-branch storage inside segments matters (Option 2's
   first-writer-tail loss is measurable);
3. master allocation-count or eviction-scan pressure at target capacity
   (but not per-key memory pressure — see C.3);
4. a trust boundary requires atomic publication **and** the namespace is
   single-writer-per-page (MLA election, Option-2 segments) — for
   multi-writer pages, atomic publication is Appendix A's coordinator, not
   Option 3.

Until one of those is real, Options 0–2 are strictly simpler and sufficient.
