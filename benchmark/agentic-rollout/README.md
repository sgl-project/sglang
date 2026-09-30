# Rollout Simulator and Explorer

Run independent conversations against an existing SGLang server, with simulated tool waits and growing history. The client runs on any machine with Python 3.11+ and network access to the server. `uv run` installs the client dependencies; it does not need SGLang, CUDA, a trainer, or a dataset locally.

```text
prompt -> generate -> tool delay -> append tool result -> repeat
finish configured turns -> close
```

`--conversations N --turns T` runs exactly N conversations of T turns each. `--concurrency` limits in-flight generation requests; tool waits release those slots.

Inputs are readable synthetic text, tokenized, repeated, and trimmed to exact lengths. A distinct conversation prefix limits cross-conversation reuse. Model outputs remain in the history as exact token IDs. The client inserts a tool step after every generation; it does not parse or execute model-produced tool calls. The seed controls tool delays and synthetic inputs, not bitwise reproducibility of GPU outputs.

## Quick start

Example server command (GPU required; validation status is in the evidence note):

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B --host 0.0.0.0 --port 30000 \
  --enable-session-radix-cache --enable-hierarchical-cache \
  --hicache-ratio 1.6 --hicache-write-policy write_through \
  --enable-metrics --stream-interval 1
```

From the repository root (or use an absolute script path from another directory):

```bash
uv run benchmark/agentic-rollout/simulate.py \
  --base-url http://SERVER:30000 --tokenizer Qwen/Qwen3-0.6B \
  --mode ordinary --conversations 8 --concurrency 8 \
  --initial-tokens 2048 --tool-tokens 256 --output-tokens 128 \
  --turns 8 --tool-delay 1 3 --seed 1 \
  --output-dir results/ordinary

uv run benchmark/agentic-rollout/explore.py results/ordinary --output rollout.html

uv run benchmark/agentic-rollout/plot.py results/ordinary \
  --output ordinary.png
```

The tokenizer must match the server; use a fixed local tokenizer directory for reproducible runs. `--trust-remote-code` is explicit. The defaults are a small functional workload, not a guarantee of cache pressure. For a smoke test, use `--turns 2 --initial-tokens 128 --tool-tokens 32 --output-tokens 16`.

## Compare features

| Client mode | History sent on each continuation |
| --- | --- |
| `full-history` | All previous input and output token IDs, plus the new result |
| `ordinary` | Session/request IDs and only the new result |
| `streaming` | Streaming-session/request IDs and only the new result |

All modes use HTTP streaming for timing. The server must support the selected session mode. For servers that gate streaming sessions, launch with `--enable-streaming-session` for that mode. Unsupported requests fail; the client never substitutes a different mode.

Restart the server between runs. Hold the model, workload, seed, rank assignment, and all unrelated launch settings fixed. Change one feature at a time: session mode, `--enable-session-radix-cache`, `--enable-hierarchical-cache`, `--hicache-ratio`, `--hicache-write-policy`, or `--page-size`. Supported combinations depend on the serving version and attention backend. Keep the exact server command and wheel/source revision with each result. The client records resolved `/server_info`, but cannot read a remote machine's wheel checksum or environment.

By default, client-side DP sticky routing assigns conversation `i` to `i % dp_size`, using the server's reported size. Every turn stays on that rank. A missing response rank is accepted only when the server reports exactly one DP worker. Use `--disable-dp-sticky-routing` to let the server choose ranks. Choose conversation and concurrency counts divisible by DP size. Tool waits release concurrency slots. The client sends `capacity_of_str_len=1000` as an unused placeholder required by older servers; it does not change engine KV capacity.

A larger pressure workload:

```bash
uv run benchmark/agentic-rollout/simulate.py \
  --base-url http://SERVER:30000 --tokenizer /path/to/matching/tokenizer \
  --mode ordinary --conversations 256 --concurrency 256 \
  --initial-tokens 8192 --tool-tokens 2048 --output-tokens 128 \
  --turns 56 --tool-delay 1 3 --seed 1 \
  --output-dir results/page1-a
```

Check measured GPU/host capacities first. Final history per conversation is `initial + (turns - 1) * (tool + output) + output`. Increase conversations until the live histories exceed GPU capacity while fitting host capacity, then freeze the workload. Verify actual CPU restores; a large configured workload alone is not evidence of cache pressure. Use fresh output directories and servers for each comparison; repeat in reverse order to check repeatability.

## Read results

The client writes `manifest.json`, `requests.jsonl`, `sessions.jsonl`, and raw `metrics.jsonl`. It stops on request failures, records partial evidence, and attempts session cleanup. A close HTTP response confirms submission, not synchronous release of every cache allocation. Metrics scrape errors are recorded separately; missing metrics do not invalidate successful client timing.

The plotter accepts multiple directories and writes elapsed-time and per-turn plots, per-run CSV/summary files, and a configuration sidecar for reviewing differences. Failed runs are labeled and must be excluded from clean performance comparisons. It retains their successful requests for diagnosis.

TTFT starts when the client submits HTTP work and ends at the first generated-token event. Local semaphore waiting and tool delays are separate. Token timing uses observed stream events; several tokens can arrive together, so it is not a kernel timing or a distribution of individual token gaps. Output rate counts token increments in their arrival windows. Cache-hit fractions are token-weighted; missing GPU/host breakdowns stay unavailable.

By default, scrape the generation endpoint's `/metrics` once per second. Repeat `--metrics-url http://NODE:PORT/metrics` for **distinct** distributed exporters. Do not list the same exporter twice through aliases. Plots keep exporters separate and show maximum reported occupancy across ranks; raw labels remain intact. Only exported ranks are observed. A flat total hit rate can conceal a shift from GPU hits to more expensive CPU restores.

A candidate slowdown is three consecutive 30-second windows with p95 TTFT at least twice a preceding stable period while total cache hit stays within five percentage points. Inspect context lengths and queueing before attributing it to HiCache. Compare matching turns as well as elapsed time; faster runs advance through the workload sooner. The plotter exposes measurements rather than automatically declaring a cause.

## Offline Explorer

Open the generated rollout.html directly in a browser. Data, CSS and JavaScript are embedded; no server, CDN, Grafana or frontend build is needed. Keep sibling source files together when running explore.py.

- **Timeline:** one conversation per lane, grouped by worker. Sampling spans HTTP submission to completion/failure; tool call spans the simulated delay **before that turn**; wait spans client queue entry to submission. Select a phase or use the request picker, including failed requests. Drag to zoom, or use Window/Position.
- **Engine Metrics:** 15 panels with client distributions in two-second windows and server samples at recorded scrape times. Select one exporter by endpoint; exporters are never combined. Click legends to hide series.
- **L1/L2:** GPU occupied = active + evictable; CPU occupied = host used. Each usage percentage divides by its own capacity; capacities are dashed. Hit fractions use deltas of prefill_effective_tokens_total: device, host and storage hits plus uncached input form the denominator; total hits include storage.
- Missing counters, changed label sets, resets and zero hit denominators are gaps. Explicit host-hit zero remains 0%. Occupancy aggregates matching label sets inside the selected exporter. Mamba usage shows the maximum reported rank; speculative acceptance length shows an unweighted rank mean.
- Older recordings without exact tool/wait timestamps mark those phases unavailable. Failed runs retain measured phases, token events and available TTFT. Interrupted runs are labeled incomplete.

The output contains model identifiers, endpoint labels and recorded errors. Review it before sharing. Raw recordings and generated viewers should stay outside Git.

## CPU tests

```bash
uv run --no-project --with aiohttp --with transformers --with prometheus-client \
  python -m unittest discover -s benchmark/agentic-rollout/tests -v
```

See [the spec](spec/feature-00-session-hicache-latency.md) and [validation evidence](spec/evidence/feature-00-session-hicache-latency.md).
