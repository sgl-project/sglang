# KVCR direct linker

Offload GPU KV pages straight into KVCR-owned DRAM and restore them straight
back into GPU allocations, with no SGLang host pool. Optionally reuse prefixes
held by a peer worker that a router (or an explicit hint) names.

```
Local offload:   SGLang GPU -> KVCR DRAM (this rank)
Local restore:   KVCR DRAM  -> SGLang GPU
Peer restore:    source KVCR DRAM -> target KVCR DRAM -> target GPU
```

The peer path hydrates the local KVCR tier first so residency is confirmed and
held before a GPU cache hit is admitted. It still includes the target-side DRAM
hop; it is not a remote-to-GPU transfer.

## Requirements

- The `kvcr` package with typed framework memory regions
  (`kvcr.config.FrameworkMemoryRegion`), i.e. the commit that adds
  `KVCRBackendConfigs.framework_regions`, and `nixl==1.3.2` with the UCX
  plugin. Install them into the same environment as SGLang:

  ```bash
  uv pip install --no-deps nixl==1.3.2 nixl-cu13==1.3.2
  uv pip install --no-deps -e /path/to/kvcr
  ```

- A dense MHA or MLA model, DSA (DeepSeek V3 layout), or DeepSeek V4. CUDA
  only; pipeline parallelism, Mamba, HiSparse, and DP without attention DP are
  rejected at startup.

- UCX able to register GPU memory. On hosts with InfiniBand devices but no
  GPUDirect RDMA peer memory (`nvidia_peermem`), set
  `UCX_TLS=cuda_copy,cuda_ipc,sm,tcp` for same-node transfers; otherwise
  registration fails at startup with `ibv_reg_mr ... Bad address`.

## Launch

```bash
bash examples/runtime/kvcr_direct_linker/launch_worker.sh \
  --model Qwen/Qwen3-0.6B --port 30000 --control-port 25000 --dram-gib 8
```

The script passes:

```
--enable-unified-cache-external-linker
--unified-cache-external-linker-backend kvcr
--hicache-storage-backend-extra-config '{
  "local_dram_bytes_per_worker": <bytes>,
  "enable_remote_hint": true,
  "control_port": 25000,
  "control_advertise_host": "<host peers dial>",
  "preparation_deadline_ms": 2000
}'
```

`local_dram_bytes_per_worker` is the total for this worker; each scheduler
rank on the node receives an equal share and logs its resolved per-pool
allocation at startup (`KVCRDirectLinker rank=... page_capacity=...`).
Every rank binds `control_port + engine_global_attention_rank`, so reserve one
port per rank starting at the base.

Other options (all optional): `operation_timeout_ms`, `abandon_timeout_ms`,
`max_inflight_prepare_requests`, `max_inflight_prepare_bytes`,
`max_prepare_bytes_per_request`, `max_inflight_offload_bytes`,
`max_abandoned_bytes`, `fetch_chunk_pages`, `policy` (`lru` or `fifo`),
`pin_local_dram`, `poll_interval_ms`, `stats_log_interval_s`.

## Explicit-hint peer reuse check

`peer_reuse_check.py` drives two workers without a router:

1. sends a prompt to the source worker so its KV is offloaded into the source
   KVCR tier,
2. flushes the target worker's cache,
3. sends the same prompt to the target with a `kv_hints` envelope naming the
   source's control endpoint and the page hashes, and
4. compares the target's greedy output (and the served `cached_tokens`) against
   the control run.

```bash
python examples/runtime/kvcr_direct_linker/peer_reuse_check.py \
  --source http://host-a:30000 --source-control tcp://host-a:25000 \
  --target http://host-b:30001 --model Qwen/Qwen3-0.6B --page-size 64
```

The hint envelope the router sends has one `kv.fetch@1.0` action:

```json
{"protocol_version": "0.1", "message_id": "...", "actions": [
  {"action_id": "...", "action_type": "kv.fetch", "action_version": "1.0",
   "payload": {"source_control_endpoint": "tcp://host-a:25000",
               "block_hashes": [<int64 event hashes>]}}]}
```

Block hashes are the 64-bit event hashes SGLang publishes in `BlockStored`
events (the leading 16 hex characters of each page's storage hash). The full
storage key stays authoritative; a hint only selects candidates, and a page
becomes a hit only after KVCR confirms and claims it locally.

## Observability

Each rank logs a `KVCRDirectLinker stats` line every `stats_log_interval_s`
seconds with, among others: `queried_pages`, `prepared_pages`,
`peer_prepared_pages`, `admitted_pages`, `restored_pages`,
`gpu_restore_bytes`, `offload_bytes`, `prepare_latency_s_max`,
`prepare_deadlines`, `miss_*` reasons, `offload_inflight_bytes_hwm`,
`outstanding_claims`, `late_completions`, `uncertain_loads`, `dram_bytes`,
and `descriptor_build_s_sum`. Keys are never used as metric labels.
