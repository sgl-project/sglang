# Mixed prefill/decode and repeated-prefix serving-boundary benchmark

A compact benchmark that stresses **mixed long-prefill, long-decode, and
repeated-prefix** traffic under an increasing client-concurrency ladder and
reports where the latency tail breaks down. It is a thin driver on top of
`sglang.bench_serving`: each (workload, phase, concurrency, rep) cell is one
`python -m sglang.bench_serving` run, and the driver aggregates the per-cell
results into a boundary table.

Tracking issue: https://github.com/sgl-project/sglang/issues/27406

## Workloads

| Workload | Prompt / output tokens | Stresses |
|---|---|---|
| `balanced_2k` | ~2048 / 128 | continuity baseline |
| `long_decode` | ~1024 / 512 | decode / memory bandwidth |
| `long_prefill_8k` | ~8192 / 64 | prefill / KV-cache allocation |
| `repeated_prefix` | ~2048 / 128 (long shared prefix) | RadixAttention prefix reuse |
| `agentic_session` | multi-turn: ~3072 prefix + ~512/turn / 256 | cache retention under interleaved sessions |

`repeated_prefix` uses the `generated-shared-prefix` dataset so a long identical
prefix is actually reused across requests (a real prefix-cache hit), rather than
a short prefix that the cache would barely benefit from.

`agentic_session` models interleaved agent sessions: each session is a
multi-turn conversation (`--gsp-num-turns`) with a unique ~3072-token session
prefix, a ~512-token appended suffix per turn (tool output), and a client-side
tool-call pause before each turn — mostly short (`--agentic-gap-short-s`,
default 0.5s), occasionally long (`--agentic-gap-long-s`, default 15s with
probability `--agentic-gap-long-prob` 0.15). A session holds its concurrency
slot during pauses, like a live agent that is idle mid-tool-call. Turn `k+1`
either hits the session's prefix in the radix cache or re-prefills it after
eviction, so the per-turn p95 TTFT spread is the cache-retention signal; the
N ladder plus gaps control eviction pressure. Session count auto-scales as
`2×N` (override with `--agentic-sessions`); turns default to 6
(`--agentic-turns`). This workload runs on the `sglang-oai-chat` backend
(multi-turn requires a chat backend) with real assistant responses carried
across turns.

## Phases

- **`scaling`** — requests issued as fast as the server accepts them
  (`--request-rate inf`); measures peak throughput and the saturation tail.
- **`ttft`** — paced arrivals (`--request-rate` finite, default = concurrency
  `N`) so time-to-first-token attributes to prefill rather than queueing.

## Launch the server

```bash
# default (RadixAttention prefix cache on)
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 30000

# radix cache OFF (for the prefix-cache on/off comparison)
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 30000 --disable-radix-cache

# different chunked-prefill size (prefill-boundary sweep)
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 30000 --chunked-prefill-size 4096
```

## Run the benchmark

```bash
# full sweep: 4 workloads x {scaling,ttft} x N in {1,2,4,8,16,32} x 3 reps
python3 bench_boundary.py --model Qwen/Qwen2.5-7B-Instruct --port 30000

# a quick smoke slice
python3 bench_boundary.py --model Qwen/Qwen2.5-7B-Instruct --port 30000 \
    --workloads long_decode,long_prefill_8k --phases scaling \
    --concurrency 1,8,32 --reps 1 --num-prompts 64

# agentic sessions only (cache-retention boundary; scaling phase is the
# primary signal -- gaps already pace the turns within each session)
python3 bench_boundary.py --model Qwen/Qwen2.5-7B-Instruct --port 30000 \
    --workloads agentic_session --phases scaling --concurrency 1,8,32 --reps 3

# print the planned bench_serving commands without running
python3 bench_boundary.py --port 30000 --dry-run
```

### Compare server configs (e.g. radix cache on vs off)

Run the driver once per server configuration, with a distinct
`--server-config-label`, pointing at the same `--output-dir`:

```bash
# against the default server
python3 bench_boundary.py --port 30000 --server-config-label radix_on

# restart the server with --disable-radix-cache, then:
python3 bench_boundary.py --port 30000 --server-config-label radix_off
```

Each run writes `raw_<label>.jsonl`, `boundary_cells_<label>.jsonl`, and
`boundary_summary_<label>.jsonl` so the two configs can be compared directly.

## Output

Per (phase, workload, N) cell, aggregated as the mean over reps:

- **decode_tps** — aggregate decode throughput across the `N` concurrent streams
- **p50/p95 wall latency** — end-to-end request latency
- **p50/p95 TTFT** — time to first token
- **p95×N1** — p95 wall-latency multiplier vs `N=1` (the tail-breakdown signal)
- **fail** — failed request count

```
phase    workload            N  decode_tps  p50_wall  p95_wall  p95xN1  p50_ttft  p95_ttft  fail
scaling  long_decode        32      2288.0    ...       ...      1.36x    ...       ...        0
scaling  long_prefill_8k    32       680.0    ...       ...      4.03x    ...       ...        0
...
```

## Reference numbers

From a complete run on `Qwen/Qwen2.5-7B-Instruct`, single A100 80GB-class GPU,
3 reps, `N` swept to 32. `decode_tps` is the aggregate decode throughput across
the `N` concurrent streams in a cell, averaged over the 3 reps; `p95×N1` is
p95(wall latency at `N`) / p95(wall latency at `N=1`).

| Phase | Workload | decode tok/s (N=32) | p95 wall-latency mult vs N=1 |
|---|---|---:|---:|
| scaling | `long_decode` | 2288 | 1.36x |
| scaling | `long_prefill_8k` | 680 | 4.03x |
| scaling | `repeated_prefix` | 1474 | 2.25x |
| ttft | `long_decode` | 1719 | 1.38x |
| ttft | `long_prefill_8k` | 461 | 4.82x |
| ttft | `repeated_prefix` | 1236 | 2.12x |

## How to read the boundary

The signal is workload shape, not a single throughput ranking. **Long-decode**
traffic stays comparatively stable through `N=32` (p95 multiplier ~1.36–1.38x).
**Long-prefill** and **repeated-prefix** traffic cross a much sharper tail
boundary (p95 multiplier ~2–5x) — these are the shapes operators tune with
chunked-prefill size, prefix caching, and scheduling knobs. Running the same
sweep against `--disable-radix-cache` and different `--chunked-prefill-size`
values isolates how much each knob moves the boundary.
