# Replay, Trace, Profile, and Bisect

Use this reference after the first live checks. The goal is to turn the problem
into something repeatable.

## Save Requests

### Request dump

```bash
python3 -m sglang.srt.managers.configure_logging \
  --url http://127.0.0.1:30000 \
  --dump-requests-folder /tmp/sglang_request_dump \
  --dump-requests-threshold 100
```

Use this when:

- the problem is intermittent
- you need the real request shape
- you do not want to restart the server

### Crash dump

If the server already runs with:

```bash
--crash-dump-folder /tmp/crash_dump
```

SGLang saves recent requests before a crash. Treat that dump as the best
starting point.

Summarize it first:

```bash
python3 scripts/incident_artifact_tool.py summarize-dump \
  --input-file /path/to/crash_dump.pkl
```

Current crash-dump tests show at least:

- `server_args`
- `requests`
- `launch_command`

## Replay

Use the stock replay tool:

```bash
python3 scripts/playground/replay_request_dump.py \
  --input-file /path/to/crash_dump.pkl \
  --host 127.0.0.1 \
  --port 30000 \
  --parallel 128
```

Or replay a folder:

```bash
python3 scripts/playground/replay_request_dump.py \
  --input-folder /path/to/request_dump_dir \
  --file-number 10 \
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

If that happens, the allowlist is the problem, not the dump.

Use replay before profiling when:

- the issue depends on workload mix
- it only appears after some number of requests
- you need to compare two builds on the same traffic

## CUDA Restart-And-Replay

If replay points to a CUDA crash path, restart the same build with coredumps:

```bash
SGLANG_CUDA_COREDUMP=1 \
SGLANG_CUDA_COREDUMP_DIR=/tmp/sglang_cuda_coredumps \
python -m sglang.launch_server \
  --model-path ... \
  --crash-dump-folder /tmp/sglang_crash_dump \
  ...
```

Then inspect the coredump:

```bash
cuda-gdb "$(which python3)" \
  -ex "target cudacore /tmp/sglang_cuda_coredumps/cuda_coredump_<host>.<pid>.<ts>"
```

Good first commands:

- `where`
- `info cuda kernels`
- `x/10i <pc>`

Use the coredump to find the failing kernel, not automatically the root-cause
kernel.

See:

- [moe-shared-oob-case-study.md](moe-shared-oob-case-study.md)

## Trace

Tracing must be enabled at startup:

```bash
python -m sglang.launch_server \
  --enable-trace \
  --otlp-traces-endpoint localhost:4317 \
  ...
```

Optional router command:

```bash
python -m sglang_router.launch_router \
  --enable-trace \
  --otlp-traces-endpoint localhost:4317 \
  ...
```

Useful environment variables:

```bash
export SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS=500
export SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE=64
```

If tracing is already enabled, change the level without restart:

```bash
curl "http://127.0.0.1:30000/set_trace_level?level=1"
curl "http://127.0.0.1:30000/set_trace_level?level=2"
curl "http://127.0.0.1:30000/set_trace_level?level=3"
```

Use tracing for:

- router vs. worker delay
- tokenizer / scheduler / detokenizer timing
- PD transfer timing
- request timing across processes

If you already have OTEL JSON or JSONL, convert it for timeline inspection:

```bash
python3 scripts/convert_otel_2_perfetto.py \
  --input /tmp/otel_trace.json \
  --output /tmp/sglang_trace_perfetto.json
```

## Torch Profiler

Switch to `llm-torch-profiler-analysis` when:

- replay already reproduces the issue
- metrics and loads do not explain it
- the problem now looks compute-side

This skill should decide when to profile, not duplicate the profiler workflow.

## Bisect

If one commit is known-good and a newer commit is known-bad:

1. build a deterministic harness from the problem
2. prefer replay-based harnesses when the failure depends on request mix
3. use `git bisect run <harness>`
4. only then go back to trace or profile if needed

Example:

```bash
git bisect start <bad> <good>
git bisect run bash ./repro_or_check.sh
```

## Common Paths

### Crash

1. crash dump
2. summarize dump
3. replay
4. CUDA coredump plus `cuda-gdb`
5. `debug-cuda-crash` or narrower instrumentation

### TTFT regression

1. baseline metrics and loads
2. request dump
3. replay the slow request
4. trace if stage ownership is unclear
5. `llm-torch-profiler-analysis` if it still looks compute-side

See:

- [ttft-prefill-not-queue-case-study.md](ttft-prefill-not-queue-case-study.md)

### Distributed hang

1. healthy baseline bundle
2. save the trigger request
3. replay on a clean target
4. collect replay-time bundle and stacks
5. identify the NCCL or collective path
6. switch to `debug-distributed-hang`

See:

- [communication-hang-case-study.md](communication-hang-case-study.md)

### Throughput regression after deploy

1. compare `server_info`
2. compare `/metrics` and `/v1/loads`
3. replay stable workload
4. bisect if one older commit is known-good
5. profile only if compute still looks suspicious
