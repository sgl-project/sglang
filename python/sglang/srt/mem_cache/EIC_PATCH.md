# EIC patch ledger

ep_main is periodically rebuilt from upstream and drops downstream merges, so EIC
lives as one patch commit on `cklxx/eic-patch`. After every ep_main refresh:

```
git fetch origin && git rebase --onto origin/ep_main HEAD~1
```
A maintained reference diff also lives on `cklxx/eic-patch-diff` (see its
README): for a large refresh, branch from origin/ep_main and `git apply --3way`
that diff; resolve only the ep_main-owned files below. The EIC-owned new files
apply untouched.

Conflicts only happen in the ep_main-owned files below. For each touch point this
ledger records what it does, why, where it came from, and which test turns red if
it is lost. PR numbers refer to bytedance-iaas/sglang; their descriptions carry
the measurements. The pre-squash history is on `cklxx/eic-main` (PR #737).

Unit tests: `test/registered/unit/mem_cache/test_eic_hicache_regression.py`
(`EICReg` below), `test_registry.py`, `test_forward_pass_metrics.py`.
Serving benchmark snapshots: `benchmark/hicache/eic_snapshots/`.

## Rules for editing this patch

- ep_main's fix is authoritative. When ep_main already solves a problem, or
  already has a field or metric for it, EIC adapts inside its own files and the
  patch does not touch ep_main. See "Replaced by ep_main mechanisms".
- Touch an ep_main file only for EIC. Every hunk is gated on `enable_eic_cache`,
  an EIC-only hook, or an EIC class, so non-EIC behavior matches ep_main.
- Add a row here in the same commit as the code.
- Squash back to one commit before pushing.

## New files (EIC-owned, no conflict surface)

| File | Role |
|---|---|
| `scripts/eic_integration_check.py` | post-deploy EIC integration check (#709); standalone, no runtime import |
| `managers/eic_cache_controller.py` | EIC write/load threads and queues on top of `HiCacheController` |
| `mem_cache/eic_hiradix_cache.py` | `EICHiRadixCache` / `EICPagedHiRadixCache`: remote match, async load-admit gate, PP verdicts |
| `mem_cache/eic_memory_pool.py` | EIC client and host pools (MHA/MLA/NSA/DSv4) |
| `mem_cache/eic_chunk_cache.py` | `EICChunkCache` / `EICSWAChunkCache` for `--disable-radix-cache` |
| `mem_cache/eic_pp_reconcile.py` | Cross-PP load-length reconciler |

Upstream APIs these subclass or call, the usual source of silent breakage after
a refresh: `HiCacheController.__init__`, `HiRadixCache`,
`RadixCache.cache_finished_req`, `build_deepseek_v4_hicache_stack`,
`CacheInitParams`, `Req.extend_range`. Three contracts broke only on a real
server, not in unit tests:

- `RadixCache.insert` unpacks `(prefix_len, last_node)` from `_insert_helper`
  (#27058); the EIC override must return the same pair.
- `HostPoolGroup.load_to_device_per_layer` does not move indices;
  `HiCacheController.move_indices` does it for the kernel IO backend. The DSv4
  `device_writeback` repeats that move, or the load kernel rejects CPU indices.
- `Req.extend_range` is None until `PrefillAdder.add_one_req` sets it from
  `prefix_indices`; `_finalize_load_admit` must not read or set it.


## Refresh fixes for the pool/ServerArgs refactor (f4c61f324b)

The 2026-09-15 refresh crossed the host-pool and ServerArgs refactors; these were
not text conflicts but API-contract breaks found by review, all in EIC-owned files:

- DSv4 load/backup drives pools per entry. `HostPoolGroup` no longer has
  `load_to_device_per_layer`/`backup_from_device_all_layer`, and the per-pool
  methods take no `pool_transfers=`. `EICDeepSeekV4TokenToKVPoolHost` iterates
  `_build_pool_transfers` and calls each `entry.host_pool.{backup,load}` with the
  entry's device pool and `layer_mapper`, matching `l2_transfer.py`.
- Req fields moved into `ReqKvInfo`: use `req.kv.req_pool_idx` and
  `req.kv.cache_protected_len` (no top-level forwarding properties).
- `req.fill_ids` is now `req.get_fill_ids()` (bounded by `extend_range.end`).
- `last_matched_prefix_len` was removed upstream; the EIC finalize no longer sets it.
- Test imports updated: `unified_cache.component_type.ComponentType`,
  `unified_cache.unified_tree_core.UnifiedTreeNode`.

## Touch points in ep_main files

### Enablement

| Where | Change | Why | From | Guard |
|---|---|---|---|---|
| `server_args.py` fields | `enable_eic_cache`, `disable_eic_shared` | CLI flags (generated from annotated fields) | #481 | `test_registry.py` EIC cases |
| `server_args.py` post-init | `enable_eic_cache` forces `enable_hierarchical_cache` | `--enable-eic-cache` alone must be enough | #481 | none |
| `arg_groups/hicache_hook.py` `handle_hicache` | When `enable_eic_cache`, declare `enable_hierarchical_cache=True` before the hicache skip test | After the ServerArgs resolution pipeline refactor there is no `__post_init__` body to force it in; resolution writes go through `declare_resolution` | #767 (moved on refresh) | registry EIC cases |
| `mem_cache/registry.py` `default_radix_cache_factory` | EIC chunk cache under `disable_radix_cache`; `EICHiRadixCacheBuilder` ahead of the hybrid/DSA arms; sets `dedup_aliased_swa` on an SWA allocator | Nested under the hierarchical arm it was unreachable for hybrid-SWA/DSA models: the flag was ignored and the first request died on `ongoing_load_admit` | #737 | `test_eic_hi_radix_cache_when_hierarchical_and_eic`, `test_eic_chunk_cache_when_chunked_prefill_disable_radix_and_eic`, `test_eic_swa_chunk_cache_when_hybrid_swa_and_eic`, `test_eic_marks_swa_allocator_for_alias_dedup` |
| `managers/scheduler.py` `__init__` | `self.enable_eic_cache` = flag AND tree cache is `EICHiRadixCache` | EIC bookkeeping must follow what was built, not the flag. `EICChunkCache` (PD decode-save path) was admitted too but has no admission hooks, so aborting a waiting req there raised `AttributeError` on `release_load_admit` | #737, #767 | `test_scheduler_eic_gate_admits_only_caches_with_admission_hooks` |

### Scheduling

| Where | Change | Why | From | Guard |
|---|---|---|---|---|
| `scheduler.py` `_get_new_batch_prefill_raw` | `match_from_remote(waiting_queue)` before the PP-divergent early returns | Its PP all_reduce hangs unless every stage calls it in lockstep | #523, #670 | `EICReg.test_match_from_remote_*` |
| `scheduler.py` candidate loop | Skip `init_next_round_input` while a load-admit is in flight; gate candidates on `check_load_back_progress` (every candidate when PP>1) | Load-back is async; re-matching a mutated tree double-counts; PP>1 needs symmetric per-rid reports | #670, #741 | `EICReg.test_pp1_*`, `test_two_stage_*`, `test_gate_refuses_load_back_without_pool_headroom` |
| `scheduler.py` candidate loop | After `init_next_round_input`, skip a req whose matched prefix runs through an in-flight load-back (`prefix_loading`) | `load_back` publishes `node.value` before the DMA acks; adopting those slots reads KV still landing, and a failed load frees the tail under the req, whose insert then re-links freed slots into the tree (a page both free and cached). Same PR: `cache_unfinished_req` keeps a chunked req private while its prefix loads, `_match_prefix_helper` stops at a resident node under an evicted gap, and `batch_get` re-gets only the failed keys of a partial mget once | #768 | `test_failed_load_under_same_prefix_req_keeps_pool_invariant`, `test_chunk_insert_through_inflight_load_keeps_pool_invariant`, `test_partial_mget_refetches_only_failed_keys` |
| `scheduler.py` `_abort_on_queued_limit`, `_abort_on_waiting_timeout`, `abort_request` | `release_load_admit(rid)` | Aborted or preempted requests leak admit locks otherwise | #670 | `EICReg.test_release_tombstones_and_drops_straggler_verdicts` |
| `scheduler.py` idle check | `ongoing_load_admit` must be empty | PP-symmetric in-flight marker; the rank0-only maps would desync idle | #670 | none |
| `managers/schedule_policy.py` `PrefillAdder` | `enable_eic_cache`; under EIC do not subtract `host_hit_length` and skip `init_load_back` | ep_main's `host_hit_length` means "host hit not loaded yet". The EIC gate has to fold the load into `prefix_indices` before admission to keep PP stages at one clamped length, so neither step applies. | #670 | none |
| `distributed/communication_tags.py` | `HIRADIX_PP_VERDICT` | Verdict stream must not share a FIFO tag with num_ready | #670 | `EICReg.test_pp_bcast_from_first_is_nonblocking_isend` |

### SWA memory

| Where | Change | Why | From | Guard |
|---|---|---|---|---|
| `schedule_batch.py` `maybe_evict_swa` | Extend-phase eviction also when the cache exposes `eic_swa_extend_eviction()` | EIC holds no SWA in the tree; without it the SWA pool fills during chunked prefill | #661 | none |
| `schedule_batch.py` → `common.free_swa_out_of_window_slots` | `release_cache_protected_prefix` from `swa_evict_release_prefix` | EIC restores prefix SWA from host, so the out-of-window prefix can be released | #655, #710 | `EICReg.test_swa_evict_release_prefix_reaches_free_swa` |
| `common.py` `free_swa_out_of_window_slots` | Free from `req.prefix_indices` for the EIC-loaded span | Loaded indices are not in `req_to_token` (zeros there), so the free was silently dropped | #710 | `EICReg.test_swa_evict_release_prefix_reaches_free_swa` |
| `allocator/swa.py` `free_swa` | Class field `dedup_aliased_swa = False`; when set, drop the reserved page (`< page_size`) and `unique` | EIC aliases FULL spans onto reused SWA slots; double free otherwise | #661 | `EICReg.test_free_swa_skips_reserved_page_and_dedups_aliases` |

### Host pools and L3 storage

| Where | Change | Why | From | Guard |
|---|---|---|---|---|
| `hybrid_cache/hybrid_pool_assembler.py` | `device_indexed` clamps DSv4 host pages to device+1 | EIC host pages are device-indexed; `ratio × device` pinned unreachable RSS and OOMed TP8 | #744 | `EICReg.test_device_indexed_host_pages_ignore_hicache_ratio`, `test_eic_calls_the_assembler_with_its_current_signature` |
| `mem_cache/radix_cache.py` `TreeNode` | `content_hash = None` | Every EIC hash path reads it on nodes `RadixCache` creates | #523 | `EICReg.test_real_tree_node_carries_content_hash` |
| `storage/eic/eic_storage.py` | v2 multi-pool API (`register_mem_host_pool_v2`, `batch_{exists,get,set}_v2`), pp-scoped key prefix, logical anchor, mexist cardinality guard | Unified radix EIC L3 for DSv4 pools | #637, #660 | `EICReg.test_batch_exists_impl_failed_batch_keeps_cardinality`, `test_unified_tree_node_exposes_storage_hash_helpers` |

## Replaced by ep_main mechanisms

| Was in the patch | ep_main mechanism now used | EIC side |
|---|---|---|
| `get_num_allocatable_reqs`: PP=1 uses `max_running_requests` (#721) | Scheduler defaults `pp_max_micro_batch_size` to `max_running_requests // pp_size` when unset | None. Deployments must stop passing `--pp-max-micro-batch-size 1` with PP=1; that flag caused prefill batch 1. |
| `hiradix_cache.py`: disable the eic storage tier under PP>1 (#670) | #746 adds `pp_group` to the L3 prefetch sync groups, so PP stages agree on L3 hit length | None |
| `Req.eic_loaded_len`, `PrefillAdder.log_hit_eic_tokens`, `sglang:eic_cache_hit_rate`, EIC fields in `metrics_reporter` (#523, #767) | Per-request `cached_tokens_details` and `sglang:cached_tokens_total{cache_source}` | `_finalize_load_admit` sets `req.host_hit_length = req.storage_hit_length = loaded`, so EIC load-backs count as `storage`; the metric label is `storage_unknown` because EIC registers no `storage_backend` (same path as PD decode). Guard: `EICReg.test_finalize_*` |
| `free_swa` read `get_global_server_args().enable_eic_cache` | `test_legacy_global_ratchet` forbids new `get_global_server_args` call sites, and allocators are built without published config in unit tests | The registry sets `allocator.dedup_aliased_swa` in its EIC branch |
| `eic_memory_pool` used `get_attn_tensor_model_parallel_{rank,world_size}` | `get_parallel().attn_tp_{rank,size}` (`test_parallel_adoption_ratchet`) | Direct swap |

## Deliberately not carried from bytedance/deepseek_v4

| Item | Reason |
|---|---|
| `scripts/eic_gate.py` (#760) | Pre-benchmark gate only; `scripts/eic_integration_check.py` (#709, carried here) is the deployment check and the gate duplicates it |
| Min host pages `device+1` on the non-EIC DSv4 path | Upstream change; alters non-EIC sizing when `hicache_ratio < 1` |
| `free_swa` negative/out-of-range index filter (#573) | Draft-extend padding fix, not EIC |
| `allow_radix_cache_insert_once` | No setter anywhere |
| `last_matched_prefix_len` / `eic_prefix_len` in the non-EIC `init_load_back` branch | ep_main refactored that branch; with EIC on the branch is unreachable |
| `release_kv_cache(is_decode=...)` | No caller passes `is_decode` on either branch |
| `/enable_eic`, `/disable_eic` (#481) | On ep_main they only reported state; the working runtime toggle (flush and rebuild pools) exists only on poc_weizhong. They misreported under `EICChunkCache`. `/server_info` already returns `enable_eic_cache`. A real toggle should follow ep_main's `/hicache/storage-backend` attach/detach. |

## Dead or misleading config keys

- `load_remote_threshold`: removed in #763. The remote probe budget is `eic_check_max_num` (default 2048).
- `save_decode_cache` in `EICHiRadixCache`: read, never used.
- `eic_thread_num`: read, no effect on the client. Set `--eic_client_default_io_thread_num` in `eic_flag_file`.
- `enable_kvset_gpu_direct`: forced off in `eic_memory_pool.py`, with a warning if a
  ConfigMap still sets it. A CUDA write pool takes the server down: the client
  segfaults on its own IO thread (`be::Engine::Transmit`) and the other ranks then
  fail the write-path `all_reduce`. Measured in the 2026-09-12 snapshot, "Write-path
  config A/B". eic 1.5.2 gives the remote write path no CUDA handling to fall back on:
  `Client::register_memory` (`src/client/api/eic.cc:124`) discards the `MemoryInfo` it
  is given (`cuda_id` included) and registers every region as byterpc
  `HolderType::UserNormal` — byterpc v1.1 has no CUDA holder type — and nothing under
  `src/client/io/remote_engine` reads `memory_info` at all; the client's CUDA staging
  (`CudaDeepCopyIOBuf`) exists only on the local cache path. Reads take a different
  path, one-sided RDMA into the device buffer, which is why `enable_kvget_gpu_direct`
  is safe and this one is not. The exact faulting frame is not pinned down: the
  benchmark's flag file is gone, so whether the write was on the plain `KvSet` or the
  two-phase `KvSetGDR` path at the time is unproven. Re-enable only against a client
  that can register CUDA regions for the remote write path.
- The client's flag file wins over `eic_trans_type`. `InitClientInstance`
  (`src/client/api/eic_inner.cc:350`) forces both `kv_get_enable_gdr` and
  `kv_set_enable_gdr` true for `EIC_TRANSPORT_RDMA` (`eic_trans_type: 2`), then parses
  the flag file a few lines later, so the file's value is the one that takes effect.
  Production runs `--eic_client_kv_set_enable_gdr=false` with
  `--eic_client_kv_get_enable_gdr=true` (confirmed in prefill startup logs), i.e. the
  GDR write path is off and only the GDR read path is live.

## Open risks

- ep_main #750 appends MTP draft SWA layers to the DSv4 host pool, and
  `EICDeepSeekV4TokenToKVPoolHost._build_page_specs` reads them. With MTP on,
  EIC page bytes may change, and data written by an older build becomes
  layout-incompatible. Not yet run with MTP.
