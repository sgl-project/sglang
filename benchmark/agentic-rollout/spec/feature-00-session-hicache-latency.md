Status: implemented

# Synthetic session workload

**TL;DR:** A portable HTTP client compares full-history requests, ordinary sessions, and streaming sessions using independent synthetic conversations. Server launch settings control feature ablations.

```text
generate -> seeded tool delay -> append synthetic result -> repeat
finish configured turns -> close
```

The client runs exactly `--conversations` conversations, each with `--turns` sequential generations. Each conversation uses fixed input/output lengths and a distinct prefix. Actual output token IDs remain in its history. Conversations run concurrently, each with sequential turns; tool waits release concurrency slots. Client-side DP sticky routing is enabled by default: every turn of conversation `i` routes to rank `i % dp_size`. `--disable-dp-sticky-routing` omits the routing hint. All modes stream responses for timing.

The client records resolved server settings, its arguments/version, exact request timings, streamed token-count increments, output hashes, session events, and one-second raw metrics. Request errors stop the run and trigger bounded cleanup; evidence survives. Metrics errors remain visible. Missing measurements remain unavailable.

Separate plots show time and turn number, TTFT p50/p95, observed output throughput, average token timing, token-weighted GPU/CPU/total cache hit, running/queued requests, occupancy, and available transfer rates. Independent metrics exporters remain separate. Several tokens arriving together are not individual token-gap samples.

Cache-pressure claims require observed restores and measured capacity. Public smoke success does not establish slowdown reproduction. Exact commands and measured results belong in [evidence](evidence/feature-00-session-hicache-latency.md).

## Boundaries

Synthetic text and simulated tool delays only. No trainer, training-stack dependency, real tool execution, engine changes, server orchestration, or private deployment details in the shared client. Features are tested only when supported by the chosen server; no silent compatibility fallback.


## Explorer packaging (2026-09-17)

Keep the client, plotter, shared Prometheus parser, viewer assets and tests in benchmark/agentic-rollout/. The explorer embeds all assets/data without a frontend dependency. Timeline contains only sampling, tool call and client wait using exact timestamps; missing old phases remain unavailable. Engine Metrics retains 15 panels with L1/L2 in KV Usage, KV Tokens and Cache Hit Rate. Exporters remain separately selectable. Missing/reset/idle counter intervals are gaps. Preserve failed-run status and partial measurements. Verify with CPU tests and both saved 16-conversation Qwen recordings; no new GPU run is required.
