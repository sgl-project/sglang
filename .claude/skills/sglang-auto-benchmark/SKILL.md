---
name: sglang-auto-benchmark
description: Run SGLang auto benchmark searches with tiered server-flag sweeps, canonical dataset preparation, ShareGPT auto-download, custom-data conversion/validation, SLA or fixed-QPS benchmarking, CSV export, and optional second-stage speculative/EAGLE tuning. Use when the user wants an AI-operated benchmark workflow rather than a one-off bench_serving command.
---

# SGLang Auto Benchmark

This skill is for repeatable, AI-driven SGLang performance tuning.

The preferred workflow is:
- start from a mostly pure-TP baseline command,
- move the rest of the performance knobs into `search_space`,
- let auto benchmark search and compare candidates under the target SLA.

The implementation lives in:
- `python -m sglang.auto_benchmark`
- canonical dataset loader in `python -m sglang.bench_serving --dataset-name autobench`
- cookbook-derived LLM reference configs in `.claude/skills/sglang-auto-benchmark/references/cookbook-llm/`

## Preconditions

- SGLang can already launch and serve the target model in this environment.
- The model path exists, or the model is otherwise launchable.
- The goal is clear:
  - benchmark a fixed QPS list, or
  - search the maximum QPS that satisfies `max_ttft_ms` / `max_tpot_ms`.

If those are not true yet, fix them before running a large search.

Environment consistency check:
- if the benchmark will run from a remote repo copy or ad-hoc synced workspace,
  verify that the remote `python/sglang/bench_serving.py` matches the local
  feature level needed by auto benchmark before launching a long run
- at minimum, run a preflight such as `PYTHONPATH=<repo>/python python3 -m
  sglang.bench_serving --help` and confirm that the dataset choices include
  `autobench`
- if `autobench` is missing remotely, do not start the benchmark; sync
  `python/sglang/bench_serving.py` and any required dataset modules first

## Remote Run Logging

If the benchmark is executed on a remote machine, the progress bar output must be
mirrored back to a local file for humans to watch.

Scope note:
- use the remote-log mirroring workflow only when the benchmark is running in a
  different machine or a different remote container than the one the agent is
  actively operating in
- if the agent itself is already running inside the target container where auto
  benchmark is executing, do not add a separate log-return loop just for parity;
  inspect the live log files and result files directly in the current container
- in other words, "remote container" needs mirrored local logs, while "current
  container" should use direct local inspection

Required behavior:
- start the remote run with a persistent terminal/session log, for example with
  `script -q -f <log> -c "<cmd>"`; on Linux containers that use util-linux
  `script`, prefer the explicit `-c` form instead of BSD-style positional
  command arguments
- continuously sync a cleaned version of that remote session log back to a local
  `progress.log`; this local `progress.log` should already have terminal control
  sequences removed, because `script` + `tqdm` progress bars will otherwise leave
  ANSI cursor-control bytes and carriage-return redraws that look like garbled text
- if the benchmark itself is executed inside a remote container, the cleaned local
  `progress.log` must be refreshed automatically at least once every 30
  seconds while the run is active; do not rely on one-off manual polling
- implement the sync loop as a dedicated local script file checked into neither
  git nor the benchmark config; avoid fragile one-line `nohup zsh -lc '...'`
  command strings with heavy nested quoting
- prefer running the sync loop inside a long-lived local session such as a
  dedicated `tmux` pane, `screen`, or the agent's own persistent PTY session;
  detached child processes started from short-lived command runners can be
  reaped unexpectedly, so plain `nohup ... &` is not the most stable default
- immediately after starting the sync loop, verify that `progress.log` is
  actually updating by checking its timestamp or size twice across a short wait;
  if it is not changing, treat that as a broken sync setup and fix it before
  telling the user that live log mirroring is working
- tell the user the local log path up front
- keep final result files synced back locally after the run ends
- when scenario-level or top-level markdown summaries are produced, sync those
  `summary.md` / `SUMMARY.md` files back locally as first-class result artifacts
  rather than leaving them only on the remote machine

This is important because long searches can run for hours, and people need a
stable local file they can tail without logging into the remote box. The final
local run folder should also be self-contained enough for someone to review the
benchmark outcome without re-entering the remote environment.

Recommended cleanup pipeline for the local mirrored log:

```bash
perl -pe 's/\e\[[0-9;?]*[ -\/]*[@-~]//g; s/\r/\n/g; s/\x08//g;' raw_progress.log \
  > progress.log
```

Recommended remote-container sync pattern:

```bash
cat > sync_progress.sh <<'EOF'
#!/bin/zsh
set -euo pipefail
while true; do
  ssh <remote-host> "tail -n 200 <remote-progress-log>" > raw_progress.log
  perl -pe 's/\e\[[0-9;?]*[ -\/]*[@-~]//g; s/\r/\n/g; s/\x08//g;' raw_progress.log \
    > progress.log
  sleep 15
done
EOF
chmod +x sync_progress.sh
```

Run that script from a long-lived local session, for example:

```bash
tmux new-session -d -s autobench-sync './sync_progress.sh'
```

Use a persistent local background job, `tmux` pane, `screen`, or equivalent
long-lived sync process so that humans can watch the cleaned local log in real
time. Use `sleep 15` by default for long runs unless there is a specific need
for tighter polling, and keep the cleaned local `progress.log` within the
required 30-second refresh window while the run is active.

At the end of the run, make sure the local artifact set includes any generated:
- `results.jsonl`
- `results.csv`
- `summary.md`
- `SUMMARY.md`
- `scenario_summary.jsonl`
- `scenario_summary.csv`

Required health check after starting the sync script:

```bash
stat -f '%m %z' progress.log
sleep 5
stat -f '%m %z' progress.log
```

If the timestamp and size both stay unchanged while the remote benchmark is known
to be producing new output, the sync loop is broken. Fix the script before
continuing.

Do not make the cleaned log optional. The default local progress artifact should
be the cleaned `progress.log` that humans actually read.

## Most Important Rule

If the user wants the best command for a **real production or real workload scenario**, the benchmark must use **their real request distribution**.

That means:
- real prompt lengths,
- real output lengths,
- real multi-turn patterns,
- real tool / reasoning / sampling settings,
- real prefix-sharing behavior if it exists.

`sharegpt`, `random`, and `generated-shared-prefix` are useful for sanity checks and broad tuning, but they are not a substitute for the user’s real traffic.

The cookbook reference configs now default to `random` because it is portable and immediately runnable, but that should still be treated as a fallback benchmark shape rather than the final answer for a real deployment.

## Supported Dataset Kinds

The current implementation intentionally keeps the dataset surface small:

- `sharegpt`
  - Supports auto-download when no file path is provided.
  - Will be prepared into canonical autobench JSONL on disk before benchmarking.
- `custom`
  - Supports two cases:
    - old `bench_serving` custom conversation JSONL,
    - already-converted canonical autobench JSONL.
- `random`
  - Uses SGLang’s existing synthetic/random benchmark path.
  - This is the default dataset mode in the cookbook reference configs.
  - `input_len` and `output_len` can be lists of equal length.
  - Each aligned pair becomes one full benchmark scenario, not a cartesian product.
  - Example:

```yaml
dataset:
  kind: random
  scenario_names: [chat, summarization]
  input_len: [1000, 8000]
  output_len: [1000, 1000]
```

  - The workflow will run one full search for `1000 -> 1000` and one full search for `8000 -> 1000`.
- `generated-shared-prefix`
  - Uses SGLang’s existing shared-prefix synthetic generator.

Everything is normalized into one canonical autobench JSONL file before the benchmark loop starts.

## Canonical Dataset Format

Canonical format is JSONL, one request per line.

Minimal rows:

```json
{"prompt": "Write a summary of this document.", "output_len": 256}
{"prompt": [{"role": "user", "content": "Summarize this document."}], "output_len": 256}
{"prompt": ["first turn", "follow-up turn"], "output_len": 128}
```

Optional fields:

```json
{
  "prompt": [{"role": "user", "content": "Use the weather tool."}],
  "output_len": 256,
  "extra_request_body": {"temperature": 0.0, "top_p": 0.95},
  "image_data": ["file:///tmp/example.png"],
  "timestamp": 1710000000,
  "routing_key": "group-a",
  "metadata": {"source": "custom-upload"}
}
```

Compatibility:
- legacy `messages`
- legacy `prompt_origin`
- legacy `param_send`
- legacy `system + content`

## ShareGPT Auto-Prepare

`sharegpt` does not need a full path.

Example:

```bash
python3 -m sglang.auto_benchmark convert \
  --kind sharegpt \
  --tokenizer /path/to/tokenizer \
  --num-prompts 200 \
  --output /tmp/sharegpt.autobench.jsonl
```

This will:
- auto-download ShareGPT through the existing SGLang cache path when needed,
- convert it into canonical autobench JSONL,
- save it to the requested output path.

## Custom User Data Workflow

When the user uploads custom data:

1. Inspect a few raw rows first.
2. Decide whether the file is:
   - already canonical autobench JSONL,
   - old `bench_serving` custom format,
   - or an unsupported custom schema that must be transformed manually.
3. If manual transformation is needed:
   - map it into canonical JSONL,
   - never hallucinate missing turns or answers,
   - never keep the final assistant answer as part of the benchmark prompt if that answer is the target completion,
   - preserve per-request generation settings in `extra_request_body`.
4. Run:

```bash
python3 -m sglang.auto_benchmark validate \
  --dataset-path /path/to/converted.autobench.jsonl \
  --tokenizer /path/to/tokenizer
```

5. Manually inspect at least 3 converted rows and confirm:
   - prompt shape is correct,
   - final assistant answer was not accidentally left in the prompt,
   - `output_len` is sensible,
   - request extras were preserved.

## Search Tiers

`search.tier` controls search breadth.

- Tier 1
  - Fastest and smallest sweep.
  - Best for smoke tests, config validation, and quickly checking whether a model can run at all.
  - Uses a very small subset of the search space and mainly does one-at-a-time changes on top of the baseline.
  - Lowest search cost, but also the easiest to miss a better configuration.
- Tier 2
  - Recommended default.
  - Good balance between coverage and runtime.
  - Runs a small cartesian search on the first few high-priority keys, then expands the rest one at a time.
  - Usually the right choice for everyday tuning when you want meaningful search without waiting too long.
- Tier 3
  - Largest search space.
  - Runs the full cartesian product of the provided search space.
  - Search time is the longest by far.
  - Only use it when the search space is already tightly bounded and you intentionally want the most exhaustive sweep.
  - This is the best chance of finding the strongest config, but it is also the easiest way to turn a benchmark into a multi-hour or multi-day run.

`search.max_candidates` still applies at all tiers, including tier 3.
When it is set together with tier 3, the workflow still enumerates the full cartesian order conceptually, but only keeps the first `max_candidates` unique candidates after deduplication.
That makes it useful as a safety valve, but it also means tier 3 is no longer truly exhaustive unless you remove the cap or raise it high enough.

If `search.max_candidates` is omitted, the workflow now defaults to `8`.
Set it to `null` only when you intentionally want an unbounded sweep.

The reference configs now default to tier 2 with `search.max_candidates: 8`.

## Interrupt And Resume

Long searches may need to be stopped and resumed later.

Use:

```yaml
search:
  tier: 2
  resume: true
```

Behavior:
- every completed trial is appended to `live_results.jsonl`
- if the process receives `SIGINT` or `SIGTERM`, it will first save partial
  `results.jsonl`, `results.csv`, and `summary.md`
- on the next run with the same config and `search.resume: true`, completed
  trials are reused and only unfinished trials are executed
- resume works per scenario directory, so it is safest to keep the same
  `benchmark.output_dir`

Notes:
- resume assumes the candidate order and dataset are unchanged
- for maximum safety, reuse the same prepared dataset or keep the same dataset
  seed/config
- `SIGKILL` cannot be handled gracefully, so only the already-written
  `live_results.jsonl` can be reused after a hard kill

YAML key order matters. Put the most important search keys first.

## What Is Tunable

This workflow is not limited to attention backend tuning.

`server.base_flags` and `server.search_space` are passed directly to `sglang.launch_server`, so in practice any valid server CLI flag can be set or searched.

There is also a small convenience layer for parallel search:

- `server.parallel.tp`
- `server.parallel.pp_size`

When `server.parallel` is used and `dp_size` is not set explicitly, the workflow auto-derives:

`dp_size = visible_gpus / (tp_size * pp_size)`

Visible GPU count is inferred from `server.env.CUDA_VISIBLE_DEVICES` by default, or from `server.parallel.gpu_count` if you set it explicitly.

The most important performance-related groups are:

- Kernel / backend
  - `attention_backend`
  - `prefill_attention_backend`
  - `decode_attention_backend`
  - `sampling_backend`
  - `grammar_backend`
- Batching / scheduling
  - `max_running_requests`
  - `max_queued_requests`
  - `chunked_prefill_size`
  - `prefill_max_requests`
  - `max_prefill_tokens`
  - `schedule_conservativeness`
  - `num_continuous_decode_steps`
  - `stream_interval`
- Memory / cache
  - `max_total_tokens`
  - `page_size`
  - `disable_radix_cache`
- Parallel / distributed execution
  - `tp_size`
  - `pp_size`
  - `dp_size`
  - `ep_size`
  - `load_balance_method`
  - `enable_dp_attention`
  - `enable_mixed_chunk`
  - `disable_overlap_schedule`
- Runtime / CUDA graph
  - keep CUDA graph enabled by default for performance benchmarking
  - `cuda_graph_max_bs`
  - `disable_cuda_graph_padding`
  - `enable_cudagraph_gc`
- Optional speculative / EAGLE stage
  - `speculative_num_steps`
  - `speculative_eagle_topk`
  - `speculative_num_draft_tokens`
  - `speculative_attention_mode`
  - `speculative_draft_attention_backend`
  - `speculative_accept_threshold_single`
  - `speculative_accept_threshold_acc`

For cookbook-derived reference configs, keep `mem_fraction_static` and
`schedule_policy` pinned to the cookbook baseline unless the user explicitly
asks to search them. They are useful knobs, but they add a lot of search width
for relatively low validation value in the default workflow.

Do not put these into the default search space:
- `mem_fraction_static`
- `schedule_policy`
- `enable_hierarchical_cache`
- `hicache_ratio`
- `hicache_size`
- `enable_lmcache`

Those features are not treated as standard auto-benchmark sweep knobs in this workflow.

Budget guardrails for the default workflow:
- use `dataset.num_prompts: 80` unless the user asks for a heavier study
- prefer a coarse QPS search tolerance
- keep `benchmark.qps.max_rounds <= 5`
- keep `search.max_duration_hours <= 12`

## Base Tuning Before EAGLE

Never start by tuning EAGLE first.

Use this order:

1. Tune the non-speculative base server first.
2. Find the best normal config for the target dataset and SLA.
3. Only if the user explicitly asks for speculative/EAGLE tuning, and provides the required draft model or equivalent assets, run the second-stage speculative search.

Do not put `disable_cuda_graph` into the default search space. For normal performance tuning, CUDA graph should stay enabled unless the user is debugging compatibility issues.

When a candidate OOMs, keep it in the final result table as a failed row and add a hint such as:
- increase GPU count, or
- use GPUs with larger memory.

## Running The Workflow

Prepare a dataset explicitly:

```bash
python3 -m sglang.auto_benchmark convert \
  --kind custom \
  --path /path/to/data.jsonl \
  --tokenizer /path/to/tokenizer \
  --output /tmp/data.autobench.jsonl
```

Run from config:

```bash
python3 -m sglang.auto_benchmark run --config /path/to/config.yaml
```

Outputs:
- prepared canonical dataset JSONL
- per-run `results.jsonl`
- summary `results.csv`
- per-candidate server logs

## Config Template

Standalone example (uses ShareGPT as dataset, a good starting point for non-cookbook models):
- `references/qwen3-32b.yaml`

Cookbook-derived configs live in `references/cookbook-llm/`.
They default to synthetic `random` traffic and are runnable out of the box.
See `references/cookbook-llm/README.md` for the full list.

Representative picks from that folder:
- `references/cookbook-llm/llama-3.1-70b-instruct.yaml`
- `references/cookbook-llm/llama-3.3-70b-instruct.yaml`
- `references/cookbook-llm/llama-4-scout-17b-16e-instruct.yaml`
- `references/cookbook-llm/llama-4-maverick-17b-128e-instruct-fp8.yaml`
- `references/cookbook-llm/minimax-m2.5.yaml`
- `references/cookbook-llm/minimax-m2.1.yaml`
- `references/cookbook-llm/deepseek-v3.yaml`
- `references/cookbook-llm/deepseek-v3.1.yaml`
- `references/cookbook-llm/deepseek-v3.2.yaml`
- `references/cookbook-llm/deepseek-r1-0528.yaml`
- `references/cookbook-llm/qwen3-235b-a22b.yaml`
- `references/cookbook-llm/qwen35-397b-a17b-fp8.yaml`
- `references/cookbook-llm/mistral-small-4-119b-2603.yaml`
- `references/cookbook-llm/kimi-k2-instruct.yaml`

All reference configs use Hugging Face repo IDs by default.
Replace `model_path` and `tokenizer` with local paths when the weights are already on disk.

## What To Report Back

After a run, summarize:
- which tier was used,
- which dataset kind was used,
- whether the dataset was synthetic or real user traffic,
- best base config,
- best QPS that satisfied SLA,
- whether speculative tuning was skipped or run,
- paths to:
  - prepared dataset JSONL
  - `results.jsonl`
  - `results.csv`
  - key server logs
