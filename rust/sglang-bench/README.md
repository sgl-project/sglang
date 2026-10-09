# sglang-bench

The serving benchmark client, ported from `python/sglang/benchmark/serving.py`.

## Why

The Python client drives every request and parses every response stream on one
asyncio thread, and it builds a fresh aiohttp session, and so a fresh
connection, per request. Both bite as the request count grows: past some
point the client, not the server, sets the measured throughput, and the
benchmark reports the client's limit. Here each request is a Tokio task on a
shared pooled HTTP client, so stream parsing spreads across
`--worker-threads` cores and connections are reused.

What that is worth was measured on one host against a mock server, 96 output
tokens per request at concurrency 256, two repeats each:

| Requests | Python | Rust |
|---|---|---|
| 500 | 1.8 s, 27.1k tok/s | 1.7 s, 28.9k tok/s |
| 2500 | 106.8 s, 2.2k tok/s | 13.2 s, 18.2k tok/s |

So: no difference at 500 requests, and 8x at 2500, with the cliff somewhere
between. Mean TTFT follows the same shape, 196 ms against 153 ms at 500
requests and 446 ms against 144 ms at 2500.

**Treat those figures as a direction, not a specification.** The mock is a
Python `ThreadingHTTPServer` sharing the host's CPUs with the client, so it
caps both clients and punishes the Python one's connection churn harder than
a real server might. An earlier run of a different workload on the same setup
showed 2x rather than 8x. The client-side CPU drawn per request is comparable
between the two, which says the Python client here is blocked rather than
compute-bound, so where your own cliff falls depends on your server. Measure
it on the server you care about before quoting a number.

## Usage

Python loads it as the `sglang.srt.rust_extensions._bench` extension module,
the same way the Rust server is loaded. Two entry points:

```bash
# Its own command line. Imports no model tooling, so it starts fast.
python -m sglang.benchmark.rust_client \
  --backend sglang --dataset-name random \
  --num-prompts 20000 --random-input-len 1024 --random-output-len 512 \
  --max-concurrency 512

# Or from the existing entry point, for a caller already on that command line.
python -m sglang.benchmark.serving --rust-client \
  --backend sglang --dataset-name random --num-prompts 20000
```

The Rust client owns the flag definitions, so `--help` on the first form is
authoritative. Flag names, defaults, the printed report, and the result
`.jsonl` keys are the Python client's, so an existing command line runs
unchanged and anything reading the result file keeps working.
`--worker-threads` and `--disable-retokenize` are the additions; through
`serving.py` the first is spelled `--rust-worker-threads`.

In process:

```python
from sglang.benchmark.rust_client import run_rust_benchmark

result = run_rust_benchmark(args)  # args as `serving.py` parses them
```

`--rust-client` refuses a configuration the Rust client does not implement
rather than measuring something else; `serving.py` reports it as a usage
error.

### Building

A wheel build picks the crate up from `[package.metadata.sglang]`, as it does
for the other extensions. In a source tree the loader compiles and caches it
on first use, which costs about a minute once and roughly 8 seconds of
workspace fingerprinting per run after that; a wheel's bundled module skips
both. To stage it by hand:

```bash
cd rust && cargo build --release -p sglang-bench --features python
cp target/release/libsglang_bench_core.so \
  ../python/sglang/srt/rust_extensions/_bench.cpython-312-aarch64-linux-gnu.so
```

The PyO3 bindings sit behind the non-default `python` feature, so `cargo test`
and `cargo clippy` build the client core without Python.

## Covered

- **Backends**: `sglang` / `sglang-native` (`/generate`), `sglang-oai`,
  `vllm`, `lmdeploy` (`/v1/completions`), and `sglang-oai-chat`, `vllm-chat`,
  `lmdeploy-chat` (`/v1/chat/completions`), streaming and not.
- **Datasets**: `random`, `random-ids`, `sharegpt`.
- Poisson request pacing (`--request-rate`), a concurrency cap, warmup, cache
  flush, profiler start/stop, the readiness probe, `--extra-request-body`,
  custom headers, and API-key auth.
- The full metric set, including the per-second peak-throughput and
  peak-concurrency windows, with NumPy's percentile interpolation and
  population standard deviation so the numbers line up with the Python
  script's.

## Not covered

Flags for these are absent rather than accepted and ignored, because a
silently dropped flag would report a benchmark nobody ran.

- **Backends**: `trt`, `gserver`, `truss`, and the embedding backends.
- **Datasets**: `image`, `mmmu`, `mooncake`, `agentic-trace`,
  `generated-shared-prefix`, `custom`, `openai`, `longbench-v2`,
  `speed-bench`, and trace-timestamp replay.
- Multi-turn replay, LoRA request distributions, `--cache-report`,
  `--plot-throughput`, PD-separated profiling, and the per-stage profiler
  options.

Reach for the Python script for any of those. Both write the same result
shape, so a mixed set of runs stays comparable.

## Parity

Counted quantities match exactly. Against one mock server with
`--random-range-ratio 1.0` (which fixes every length, making the two clients'
different random draws irrelevant), both clients report identical
`completed`, `total_input_tokens`, `total_output_tokens`,
`total_output_tokens_retokenized` and `accept_length`, and their result files
have identical key sets.

Sampling does not match: the two use different random number generators, so
the same `--seed` draws different prompts. A run reproduces itself, and
distributions match, but individual prompts do not.

Latency figures are not expected to match either, and that is the point: the
Python client's TTFT on the run above is 2.4x the Rust client's on the same
server, because its single parse thread is queueing.

## Tests

`cargo test -p sglang-bench` covers the wire parsers against recorded frame
sequences, the statistics against NumPy's definitions, dataset sampling, and
argument and config validation. CI runs them through the workspace
`cargo test` in `test/registered/rust/test_run_rust_tests.py`.

`test/registered/unit/benchmark/test_rust_client.py` covers the Python
boundary (config translation and the refusals) with the extension mocked, so
it needs no compiled artifact.
