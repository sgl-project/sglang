---
name: sglang-prod-incident-triage
description: Replay-first debug flow for SGLang serving problems. Use when a live or recent server shows health-check failures, latency or throughput regressions, queue growth, timeouts, distributed stalls, crash dumps, wrong outputs after deploys, or PD/EP/HiCache issues, and the job is to turn the problem into a replay plus the right next debug tool.
---

# SGLang Serving Debug

## Overview

Use this skill to turn a live serving problem into a debug path you can replay.

Use one loop:

- collect a baseline bundle
- save the failing request or crash dump
- replay on a clean target
- only then switch tools

Do not start with profiling.

This skill should work with more focused skills instead of re-implementing them:

- `debug-cuda-crash` when replay plus coredump points to a CUDA crash path
- `debug-distributed-hang` when the problem is clearly a TP/PP/DP/EP hang
- `llm-torch-profiler-analysis` when the issue is already narrowed to a
  compute-side path

Three examples are included:

- TTFT spike with low queue time
- replay-first CUDA crash flow
- request-shaped distributed hang flow

## Output Contract

Return:

- problem class
- what was checked
- strongest signal so far
- current best guess
- what was ruled out
- next step
- production risk

## When To Use It

- `/health` or `/health_generate` is unhealthy
- latency or throughput regressed under serving load
- queue size grows while health still looks green
- one request class times out or hangs
- the server crashes only after some requests
- outputs changed after a deploy, topology change, or weight switch
- one older commit is known-good and a newer commit is known-bad

## Workflow

### 1. Collect a baseline bundle

If a live server is reachable, collect a read-only bundle before anything more
intrusive:

```bash
python3 scripts/incident_artifact_tool.py collect-bundle \
  --base-url http://127.0.0.1:30000 \
  --outdir /tmp/incident_bundle

python3 scripts/incident_artifact_tool.py summarize-bundle \
  /tmp/incident_bundle
```

If the server is protected:

```bash
python3 scripts/incident_artifact_tool.py collect-bundle \
  --base-url http://127.0.0.1:30000 \
  --token "$SGLANG_BEARER_TOKEN" \
  --outdir /tmp/incident_bundle
```

The bundle script collects:

- `/health`
- `/health_generate`
- `/model_info`
- `/server_info`
- `/v1/loads?include=all`
- `/v1/loads?include=core,queues,disagg,spec`
- `/metrics`
- `/hicache/storage-backend` on a best-effort basis

Use the summary for a quick read on:

- health vs. active health state
- topology and runtime flags
- point-in-time queue and token usage
- TTFT / E2E / queue-time heuristics from Prometheus metrics

If the summary says the bundle was captured while the server was idle, recollect
it during traffic or move quickly to dump plus replay.

If no live server is reachable, start from the best dump or log already available:

- crash dump
- request dump
- logs
- CUDA coredump
- OTel trace
- torch profile

### 2. Save the failing request

Read [references/decision-tree.md](references/decision-tree.md) only if the
problem class is still unclear:

- server down or unhealthy
- latency or throughput regression
- wrong output or behavior regression
- intermittent timeout or hang

Then preserve the request payload that actually triggers the problem:

- crash path: use `--crash-dump-folder`
- non-crash path: enable request dump or save the exact trigger request

Do not jump straight from a live symptom to low-level debugging without first
saving something you can replay.

### 3. Replay on a clean target

Read [references/endpoints-and-signals.md](references/endpoints-and-signals.md)
when you need help reading the baseline bundle or the replay target.

Read [references/replay-trace-profile.md](references/replay-trace-profile.md)
when you need the replay, trace, profile, or bisect paths.

Standard order:

1. collect baseline bundle
2. capture request dump or crash dump
3. restart a clean debug target if needed
4. replay the same issue
5. collect replay-time logs and dumps

### 4. Only go deeper after replay

#### Replay

Use replay when:

- a crash dump exists
- a request dump exists
- the problem depends on request shape or workload mix

If a crash dump exists, summarize it first:

```bash
python3 scripts/incident_artifact_tool.py summarize-dump \
  --input-file /path/to/crash_dump.pkl
```

Then replay:

```bash
python3 /path/to/sglang/scripts/playground/replay_request_dump.py \
  --input-file /path/to/crash_dump.pkl \
  --host 127.0.0.1 \
  --port 30000 \
  --parallel 128
```

If `safe_pickle_load` blocks a locally captured trusted dump, use:

```bash
python3 scripts/replay_trusted_request_dump.py \
  --input-file /path/to/request_dump.pkl \
  --host 127.0.0.1 \
  --port 30000 \
  --parallel 1
```

If replay indicates a CUDA crash path, restart the same build with coredumps
enabled before reproducing again:

```bash
SGLANG_CUDA_COREDUMP=1 \
SGLANG_CUDA_COREDUMP_DIR=/tmp/sglang_cuda_coredumps \
python -m sglang.launch_server \
  --model-path ... \
  --crash-dump-folder /tmp/sglang_crash_dump \
  ...
```

Then inspect the generated coredump:

```bash
cuda-gdb "$(which python3)" \
  -ex "target cudacore /tmp/sglang_cuda_coredumps/cuda_coredump_<host>.<pid>.<ts>"
```

For a replay-first crash example, read
[references/case-studies.md](references/case-studies.md).

#### OTel trace

Use tracing when:

- request-stage timing is unclear
- router vs. worker attribution is unclear
- PD prefill/decode transfer may be implicated

If tracing was enabled at startup, you can change the level without restart:

```bash
curl "http://127.0.0.1:30000/set_trace_level?level=1"
curl "http://127.0.0.1:30000/set_trace_level?level=2"
```

#### Torch profile

Use profiling when:

- the issue is already narrowed to compute-side ownership
- replay already reproduces the problem
- metrics and loads do not explain the regression

At that point, switch to `llm-torch-profiler-analysis`. Do not duplicate
its profiling workflow here.

For a low-noise latency example, read
[references/case-studies.md](references/case-studies.md).

#### Distributed hang

If this looks like a collective stall, save the failing request, replay it on a
clean target, collect the replay-time bundle and stacks, then switch to
`debug-distributed-hang`.

For an example of that flow, read
[references/case-studies.md](references/case-studies.md).

#### Regression between two commits

If one commit is known-good and another is known-bad, build a deterministic
harness before doing deeper manual debugging:

1. choose a stable reproducer: request replay, benchmark command, or correctness check
2. make the harness return `0` on good behavior and non-zero on bad behavior
3. run `git bisect start <bad> <good>`
4. run `git bisect run <harness>`
5. return here only after a candidate commit is isolated

Prefer replay-backed bisect when the regression depends on request shape or
long-running serving state.

### 6. Switch tools when the boundary is clear

Switch tools once the fault class is clear:

- `llm-torch-profiler-analysis` for kernel and overlap attribution
- `debug-distributed-hang` for collective or rank-divergence hangs
- `debug-cuda-crash` for CUDA crash reproduction and kernel API logging

Do not switch tools before collecting the first bundle unless the user already has
decisive logs or dumps.

## References

Load only what the current step needs:

- [references/decision-tree.md](references/decision-tree.md)
  - problem classes, tool switch points, return shape
- [references/endpoints-and-signals.md](references/endpoints-and-signals.md)
  - endpoint behavior, auth notes, field reading
- [references/replay-trace-profile.md](references/replay-trace-profile.md)
  - request dump, crash dump, replay, trace, profiler step, bisect
- [references/case-studies.md](references/case-studies.md)
  - compact examples for replay-first CUDA crash, latency, and distributed-hang triage

## Scripts

- [scripts/incident_artifact_tool.py](scripts/incident_artifact_tool.py)
  - collect a read-only live bundle
  - summarize a collected bundle into a compact debug note
  - summarize a trusted request dump or crash dump before replay
- [scripts/replay_trusted_request_dump.py](scripts/replay_trusted_request_dump.py)
  - replay a trusted request dump when `safe_pickle_load` blocks stock replay

If a live bundle was collected, include its path.

If replay, trace, or profiling was chosen, say why bundle plus dump were not enough.
