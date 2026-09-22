Status: implemented

# Validation evidence

## Packaging checks — 2026-09-17

17 CPU tests pass: all three request modes and both stream formats, default DP routing, verified-single-worker omitted ranks, strict multiworker ranks, context/output validation, aborted/truncated streams, timeouts and cleanup. Measurement tests cover occupancy, per-label resets, missing/idle counters, token-weighted denominators, separate exporters, old/interrupted recordings, partial failed requests and safe self-contained HTML.

~~~bash
uv run --no-project --with aiohttp --with transformers --with prometheus-client \
  python -m unittest discover -s benchmark/agentic-rollout/tests -v
~~~

Viewers were rebuilt from the two saved Qwen runs below, each with 16 conversations and 96 requests. Both tabs, all 16 lanes, zoom, request selection, legends and static plots were checked. Embedded-asset tests pass, and the browser renders with external resource loading blocked by the page policy. Direct file-URL navigation was blocked by the browser automation policy, so that path was not manually verified. Packaging does not rerun GPU inference. Raw recordings, generated HTML and machine-specific orchestration stay outside this PR.

## Saved Qwen comparison — 2026-09-17

One H200, Qwen/Qwen3-1.7B, a downstream SGLang serving build at revision 41da06adca698c0032f75000445087f0922dbe43. These saved runs are not GPU validation against current upstream main. Ordinary sessions, session-aware cache, HiCache ratio 1.6, write-through, direct I/O, layer-first layout and resolved FA3 attention. One fresh-server run per page size.

Launch command, with executable and port normalized for portability:

~~~bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-1.7B --host 127.0.0.1 --port 30000 \
  --enable-session-radix-cache --enable-hierarchical-cache \
  --hicache-ratio 1.6 --hicache-write-policy write_through \
  --hicache-io-backend direct --hicache-mem-layout layer_first \
  --mem-fraction-static 0.5 --max-total-tokens 32768 --context-length 8192 \
  --cuda-graph-max-bs-decode 8 --enable-metrics --stream-interval 1 \
  --page-size "$PAGE"

uv run benchmark/agentic-rollout/simulate.py \
  --tokenizer Qwen/Qwen3-1.7B --mode ordinary \
  --conversations 16 --concurrency 16 --turns 6 \
  --initial-tokens 1024 --tool-tokens 256 --output-tokens 128 \
  --tool-delay 1 3 --start-spread 1 --seed 101 \
  --disable-dp-sticky-routing --output-dir "results/page$PAGE"
~~~

The recorded client disabled routing hints because this single-worker server omitted its response rank. The packaged client now accepts that omission with default sticky routing only for verified DP=1; this adjustment is CPU-tested, not GPU-rerun.

| Measurement | Page 1 | Page 64 |
| --- | ---: | ---: |
| Successful requests | 96/96 | 96/96 |
| GPU / CPU capacity (tokens) | 32,768 / 52,429 | 32,768 / 52,480 |
| Duration | 18.54 s | 14.96 s |
| Output throughput | 662.8 tokens/s | 821.1 tokens/s |
| Turns 4–5 TTFT p95 | 2,502.49 ms | 20.73 ms |
| Turns 4–5 mean time per output token | 10.56 ms | 2.33 ms |
| Last observed eviction / restore counters | 92,200 / 76,088 | 66,112 / 49,984 |

Limitations: one short run per setting, no reverse-order repeat. Client inputs, delays and lengths matched, but server-generated random seeds differed; 85/96 output hashes matched. This validates the client and exposes cache pressure; it does not establish a general page-size speedup or a kernel-level cause. Counters are last scraped values and may miss the tail.
