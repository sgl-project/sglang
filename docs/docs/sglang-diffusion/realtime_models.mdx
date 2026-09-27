---
title: "Realtime and Causal Video Models"
metatags:
    description: "Deploy session-based realtime and request-based causal video models with SGLang Diffusion."
---

Realtime and causal video pipelines generate video incrementally and reuse state
across chunks. This differs from offline diffusion pipelines, which denoise one
bounded latent sequence and release all request state when generation finishes.

## Execution Modes

SGLang Diffusion exposes two related but distinct modes:

| Mode | Lifetime | Interface | Examples |
| --- | --- | --- | --- |
| Realtime session | State persists until the client disconnects or the session ends | `/v1/realtime_video/generate` WebSocket | LingBot World, SANA-WM realtime |
| Request-based causal generation | State is reused across chunks within one request, then released | Standard video generation API | LongLive 2.0, batch-streaming SANA-WM |

The realtime server retains model-specific state such as the causal self-attention
KV cache, cross-attention cache, decoder history, and pending control events.
State is isolated per session and is not reused by unrelated requests.

<Note>
A causal DiT is not automatically a realtime session model. The pipeline must
also register a realtime adapter and implement the WebSocket session lifecycle.
</Note>

## Supported Realtime Pipelines

| Model family | Pipeline | Live controls | QVG KV-cache quantization |
| --- | --- | --- | --- |
| LingBot World | `LingBotWorldCausalDMDPipeline` | Camera actions and prompt updates | Supported |
| SANA-WM | `SanaWMRealtimePipeline` | Camera actions | Not supported |

Use the model cookbooks for launch commands, request schemas, and control-token
details:

- [LingBot World](/cookbook/diffusion/LingBot-World/LingBot-World)
- [LingBot World 2.0](/cookbook/diffusion/LingBot-World/LingBot-World-2.0)
- [SANA-WM](/cookbook/diffusion/SANA-WM/SANA-WM)

For the complete model list, see
[Supported Models](./compatibility_matrix).

## Causal Cache Controls

Realtime requests can override two model defaults:

- `realtime_causal_sink_size`: amount of stable prefix history retained as an attention sink
- `realtime_causal_kv_cache_num_frames`: recent causal history retained in the rolling KV-cache window

Larger windows preserve more history but increase resident memory and attention
work. These fields are request/session controls; supported ranges and defaults
remain model-specific.

For supported LingBot World deployments, the server-level
`--kv-cache-quant {off,int4,int2}` option compresses completed cache chunks.
It is disabled by default and is lossy when enabled. Start with `int4`; use
`int2` only when the additional memory reduction is worth the larger quality
risk.

See [Causal KV-Cache Quantization](./quantization#causal-kv-cache-quantization)
for installation, storage policy, tuning options, memory/latency tradeoffs, and
current limitations.

<Warning>
QVG KV-cache quantization currently supports only the LingBot realtime
sliding-window-and-sink path. It does not apply to SANA-WM realtime, LongLive 2.0
pinned sinks, global sinks, or dynamically growing caches.
</Warning>

## Deployment Considerations

- Keep `--kv-cache-quant off` when bit-exact BF16 cache behavior is required.
- Benchmark a representative session length. Short clips may not exercise cold-cache packing and can hide both its memory benefit and packing overhead.
- Treat sequence parallelism as model-specific. Follow the model cookbook and the [Sequence Parallelism](./ring_sp_performance) guide instead of assuming one mesh is best for every realtime pipeline.
- Realtime WebSocket clients must send an initialization message before control events. The exact MessagePack schema and output encoding are documented in each model cookbook.
