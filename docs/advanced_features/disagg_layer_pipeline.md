# Layer-Pipelined KV Transfer for PD Disaggregation

This page documents `--enable-disagg-layer-pipeline`, an opt-in
optimization for the [Prefill–Decode disaggregation](./pd_disaggregation.md)
path that overlaps KV cache RDMA transfer with the prefill server's
remaining-layer compute. It is currently supported on the **Mooncake**
transfer backend.

## What it does

Without layer pipelining the prefill server runs its entire forward
pass, then ships the full KV cache for the request to the decode
server in one or more chunks. Compute and RDMA run **strictly
sequentially**, so total end-to-end latency is `compute_time +
transfer_time`.

With layer pipelining a forward hook on `RadixAttention` fires after
every layer's KV bytes are committed to the pool. The hook batches
adjacent layers into **layer groups** (configurable via
`--disagg-layer-group-size`, default 4) and enqueues one RDMA transfer
per group. While the prefill server is still computing the later
layers, Mooncake is already shipping the earlier groups' KV. End-to-end
latency becomes approximately `max(compute_time, transfer_time)` plus
the cost of the trailing aux/state finalizer.

## When to enable

LP gives the most benefit when:

- **Prompts are long** (≥ a few thousand tokens). Short prompts'
  compute finishes before the first layer group is even ready to ship,
  leaving no room to overlap. The default
  `--disagg-layer-pipeline-min-prefill-len=2048` skips LP for shorter
  requests automatically.
- **RDMA bandwidth, not compute, is the bottleneck** between prefill
  and decode. Long-context or high-batch prefill stresses the network;
  LP folds that transfer time under compute.
- The transfer backend is **Mooncake**. Enabling LP with a non-Mooncake
  backend is rejected at startup; leave LP disabled for NIXL / Mori.

LP gives little or no benefit when:

- Prompts are short or batches are small (compute dominates).
- Network is faster than compute (transfer was never the bottleneck).
- The deployment uses speculative decoding with a draft model AND the
  draft KV is large relative to main — see "Draft KV" below.

## Flags

| Flag | Default | Notes |
|---|---|---|
| `--enable-disagg-layer-pipeline` | off | Master toggle. Both prefill and decode must enable it for LP to engage. |
| `--disagg-layer-group-size` | 4 | Layers per RDMA group. Must match between prefill and decode (mismatch logs a warning at handshake and silently falls back to legacy). |
| `--disagg-layer-pipeline-min-prefill-len` | 2048 | Request total length (tokens) below which LP is skipped per-request. |
| `--disaggregation-cp-transfer-shard-mode` | `page` | `layer` mode partitions LAYER GROUPS across CP ranks (vs the default page-shard); only meaningful with NSA prefill CP where every CP rank holds identical KV. |

The env vars `SGLANG_DISAGG_LAYER_PIPELINE`,
`SGLANG_DISAGG_LAYER_GROUP_SIZE`, and
`SGLANG_DISAGG_LAYER_PIPELINE_MIN_PREFILL_LEN` also work; they override
the CLI flags when set. `SGLANG_DISAGGREGATION_CP_TRANSFER_SHARD_MODE`
exists for back-compat but is deprecated in favor of the CLI flag.

## Expected speedup

A rough rule of thumb for a long-prompt 1P2D deployment on H200 +
RDMA: with `--disagg-layer-group-size=4` LP folds 30–50% of the
transfer wall-clock under compute on the prefill side, depending on
how close compute and transfer times are. The maximum theoretical
saving is `min(compute_time, transfer_time)` per request; LP can't
beat the bound. If your profiling shows transfer time ≪ compute time
(or vice versa), the benefit will be small.

## Metrics

Two Prometheus metrics are exported only when LP is enabled:

- `sglang:kv_transfer_layer_group_chunks_total` (Counter) — number
  of LP chunks shipped per request. Stays 0 when LP is off.
- `sglang:kv_transfer_layer_group_ms` (Histogram) — per-group
  enqueue→RDMA-completion wall-clock. Histogram buckets cover up to
  5000 ms for long-context groups.

When LP is disabled the metrics remain registered but not exported.

## Troubleshooting

### "LP is enabled on both sides but I see no speedup"

Check the prefill server startup log for a handshake warning:

```
WARNING ... Layer-pipeline group-size mismatch with decode session ...
```

A mismatch in `--disagg-layer-group-size` between prefill and decode,
or a `kv_cache_dtype` mismatch, silently disables LP and falls back to
the legacy fan-out. Both sides must agree on group size for LP to
engage.

### "I set --disaggregation-cp-transfer-shard-mode=layer but transfer still page-shards"

`layer` mode only engages when **all four** conditions hold:

1. `--enable-disagg-layer-pipeline` is on.
2. All CP ranks participate in transfer (e.g. NSA prefill CP).
3. `attn_cp_size > 1`.
4. The shard mode resolves to `layer`.

If any of these fails, the helper degrades to page-shard silently
(this is the safe default — non-CP0 ranks would otherwise skip pages
that no one covers).

### "Where can I see if LP fired for a request?"

Look at `_hook_enqueued_chunks` on the sender's request trace, or
inspect the `sglang:kv_transfer_layer_group_chunks_total` counter
delta. Per-request LP attribution lives in the request's time_stats
under `transfer_latency_ms`.

### Debug-only environment variables

These flags exist for development; they are NOT for production. A
startup `WARNING` lists any that are set:

| Env | Effect | Risk |
|---|---|---|
| `SGLANG_DISAGG_LAYER_PIPELINE_HOOK_NOOP=1` | Hook fires but skips RDMA, marks success. | **Silent KV corruption on decode.** |
| `SGLANG_DISAGG_LAYER_PIPELINE_VERIFY_KV=1` | `cuda.synchronize()` + log at every hook fire. | Massive slowdown. |
| `SGLANG_DISAGG_LAYER_PIPELINE_HOOK_TIMING=1` | Records per-fire hook latency. | Minor overhead. |
| `SGLANG_DISAGG_LAYER_PIPELINE_HASH_LOG=1` | CRC32 sampling of KV bytes per chunk. | Per-layer D2H copy. |
| `SGLANG_DISAGG_KV_HASH_VERIFY=1` | Per-request KV hash logs on both sides. | Per-layer D2H copy. |

## Limitations

- **Mooncake only.** The LP flag requires Mooncake; NIXL / Mori should
  run with LP disabled and use the regular transfer path.
- **No HiSparse support.** Startup fails if
  `--enable-disagg-layer-pipeline` and `--enable-hisparse` are both set.
- **No draft KV pipelining** — the draft pool's KV is shipped in a
  single trailing chunk by `send_draft_kv` after the main forward
  completes. For models with proportionally large draft KV (e.g.
  EAGLE-3) this is a small fixed overhead per request.
- **Group size must match** between prefill and decode — see
  Troubleshooting.
- **Short requests bypass LP** below
  `--disagg-layer-pipeline-min-prefill-len`.

## See also

- [PD Disaggregation overview](./pd_disaggregation.md) — base mechanism.
- [Forward hooks](./forward_hooks.md) — the underlying hook framework.
