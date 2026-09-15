# Diffusion weight-cache recovery

Two explicit native transformer adapters are supported: Wan2.1 T2V 1.3B and
the original Qwen-Image (not Edit, Layered or 2512). Both require CUDA,
single GPU/rank/node, bf16, resident non-FSDP weights, FA attention and eager
execution. Other components use their ordinary loaders. Unsupported resolved
configurations and missing/incompatible owners are errors, not disk fallback.

## Run

Use the same immutable checkpoint and installed code for both processes:

```bash
python -m sglang.multimodal_gen.runtime.weight_cache.daemon \
  --model-path /path/to/published/model

sglang generate --model-path /path/to/published/model \
  --weight-cache-mode client \
  --prompt 'A small boat on a lake' \
  --width 832 --height 480 --num-frames 9 --num-inference-steps 4
```

Leave the owner running when restarting the client. Optional
`--weight-cache-socket /private/directory/owner.sock` must match in both commands.
The parent directory must be owned by the current user and mode 0700. Otherwise
the short socket locator is derived from the physical GPU and compatibility
digest under `$XDG_RUNTIME_DIR/sglang_diffusion_weight_cache` (or `/tmp`).
`SGLANG_DIFFUSION_WEIGHT_CACHE_DIR` overrides that runtime directory.

An HF cache snapshot uses its pinned revision and file stat receipt. A local
checkpoint requires a content manifest covering exactly `model_index.json`,
the transformer config, its shard index (if present), and consumed weight files:

```bash
python -m sglang.weight_cache_common.checkpoint /path/to/published/model \
  model_index.json transformer/config.json \
  transformer/diffusion_pytorch_model.safetensors.index.json \
  transformer/diffusion_pytorch_model-00001-of-00002.safetensors \
  transformer/diffusion_pytorch_model-00002-of-00002.safetensors
```

Use the filenames actually present in your checkpoint. Publication is trusted:
do not edit a snapshot or installed native package in place. Python source is
hashed across the installed `sglang` package; editing it requires restarting the
owner. Development-only weak/unverified identity switches explicitly relax this
contract and are not production defaults.

## Reuse boundary

| Responsibility | Implementation |
| --- | --- |
| IPC serialization/import, UUID mapping | Existing SRT `TorchIpcTransportBackend`, serializer and Torch reduction patch |
| Message framing/cap, environment stamp, stale cleanup | Existing SRT `weight_cache.protocol` |
| Parameter/buffer traversal, registration, producer watchdog | `weight_cache_common`, also consumed by SRT |
| Weight loading and finalization | Existing diffusion `TransformerLoader` and `ComponentLoader`, using frozen decisions |
| Pipeline component materialization | Existing `ComposedPipelineBase` load loop, including uncached components |
| Runtime initialization | Shared diffusion worker/owner bootstrap |
| New diffusion-specific logic | Component adapter, prepared pipeline, compatibility/execution plans, strict admission and owner/client orchestration |

Storage aliases and exact object ties require a component state manifest on top
of the existing tensor transport. The component layer does not implement Torch
CUDA handle creation/reconstruction or a second serializer.

Both adapters reuse the same ordinary loader, meta constructor, common admission
checks and fingerprint mechanics. Qwen's packed text QKV is imported in its
ordinary finalized layout. RoPE frequencies, modulation caches and the small
`timestep_zero` constant are process-local derived state, not shared weights.

## Lifecycle and current limits

- Launcher preflight consumes no IPC handles. Each worker rechecks the admitted
  generation; its watchdog starts before requesting any handles.
- A same-user Unix peer is authenticated with kernel credentials. Fetches bind
  the full compatibility plan, producer PID/start identity and generation nonce.
- The owner retains every finalized allocation. SIGTERM stops admission and
  terminates/drains actual fetching consumers before releasing ownership.
  Unexpected owner death kills attached consumers; recovery requires restarting
  them. It is not safe for a consumer to keep running without its owner.
- Each delivery serializes fresh counted sends. Abandoned/fatal consumers may
  retain Torch bookkeeping until owner exit. Non-refundable delivery and storage
  export budgets bound this; restart the owner after draining when exhausted.
- LoRA, weight-update and sleep/release/resume APIs are rejected for shared weights.
  `expandable_segments` allocations are unsupported by this transport.
- Package imports can already initialize CUDA outside planning. Cached workers
  use `spawn`; the new preparation path itself does not construct model tensors.
- Initial coverage does not include quantized weights, multiple devices/nodes,
  alternate attention, automatic disk fallback, or in-place cache replacement.

Component import latency is not total service readiness: Python startup,
distributed setup, uncached text encoder/VAE loading and offload setup remain.

## Verification

The standalone GPU test uses the existing diffusion HTTP server manager and
video validators, with five ordinary starts and five cached worker restarts for
each warmup mode. It checks output parity, mutation rejection, owner death,
graceful drain, stale-file recovery and missing-owner failure:

```bash
pytest python/sglang/multimodal_gen/test/single_test_file/test_weight_cache_1_gpu.py -v -s
```

The Qwen acceptance test compares 4/20-step 1024×1024 PNGs byte for byte,
checks the complete finalized transformer before/after inference, rejects
mutation APIs and covers both graceful and abrupt owner loss plus restart:

```bash
pytest python/sglang/multimodal_gen/test/single_test_file/test_weight_cache_qwen_image_1_gpu.py -v -s
```

Use `SGLANG_WEIGHT_CACHE_QWEN_TEST_MODEL` for a local Qwen snapshot. Its default
is the pinned original Qwen-Image revision, not an automatically selected variant.

`SGLANG_WEIGHT_CACHE_TEST_MODEL` can point to a local published mirror. The
default uses a pinned HF revision. Readiness samples and median/p90 for both
`/liveness` and `/health` are written to the pytest temporary output directory.
The initial regression threshold is not a speedup claim; speedup must be
established separately for the target model, storage and host.

For startup performance, use the separate paired benchmark from the repo root:

```bash
python test/manual/bench_diffusion_weight_cache_startup.py \
  --model-path /path/to/pinned/Wan/snapshot \
  --output-dir /tmp/wan-startup-run
```

For Qwen, add `--model-kind qwen-image` and use the original pinned Qwen snapshot.
This generates images instead of videos and uses 1024×1024 server warmup. The
Qwen benchmark disables post-warmup automatic residency changes in both arms
(`SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY=1`), retaining the same initial automatic
placement. Otherwise the cache's free-VRAM advantage could change uncached text
encoder placement and confound the startup comparison. Synthetic warmup itself
still runs in the `server` arm; this is a controlled-placement measurement.
Because the owner also remains present in the ordinary arm, this benchmark needs
VRAM for two resident DiTs plus the uncached components and activations. This
extra ordinary copy is an A/B measurement requirement, not a cache-client
deployment requirement.

The output directory must not exist. This runs five ordinary/cache pairs for
each warmup mode, alternates pair order, keeps an owner present in both modes,
checks identical resolved placement and byte-exact generated images/videos, and probes
both HTTP readiness endpoints every 50 ms. It saves raw samples, per-start logs,
outputs, median/p90 and paired deltas; owner startup is reported separately.
This is a warm-file recovery benchmark, without page-cache eviction or artificial
I/O throttling. It does not assert a speedup merely because a regression limit
passes. Use an idle GPU/host and do not edit installed Python code during a run.

Cache logs separately report launcher preparation/identity/admission and worker
identity/manifest/watchdog/meta construction/fetch/mapping/finalization. Worker
component import excludes uncached pipeline loading, so it is not end-to-end
startup. A small model with subsecond ordinary DiT loading has little startup
time for a DiT-only cache to remove; do not extrapolate a large-model speedup
from its functional recovery result.
