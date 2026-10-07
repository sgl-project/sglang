# SGLang First Checks

Use this reference when the problem class is still unclear and you need a fast
starting point.

## Default Order

1. classify the symptom
2. collect the fastest useful signal
3. save the failing request or dump
4. replay before you profile

Do not start with `torch.profiler` unless the issue is already clearly
compute-side.

If one commit is known-good and another is known-bad, turn the problem into a
stable `git bisect run <harness>` first.

## Problem Classes

### Server down or unhealthy

Check:

- `/health`
- `/health_generate`
- `/server_info`
- recent stderr/stdout
- crash dump status if `--crash-dump-folder` is enabled

Likely directions:

- startup or weight-load failure
- deadlock or blocked scheduler
- CUDA crash or OOM
- auth or routing mismatch

### High latency or low throughput

Check:

- `/v1/loads?include=all`
- `/metrics`
- `/server_info`
- the exact request shape or benchmark command

Likely directions:

- queueing or capacity pressure
- cache hit rate collapse
- PD or EP topology mismatch
- speculative decoding disabled or ineffective
- kernel or backend regression

### Wrong output or behavior regression

Check:

- exact request and expected output
- `/model_info`
- `/server_info`
- current weights or recent config change

Likely directions:

- wrong weights or wrong revision
- chat template, parser, or tool config drift
- multimodal preprocessing drift
- quantization or kernel correctness bug

### Timeout or hang

Check:

- `/health`
- `/health_generate`
- `/v1/loads?include=all`
- request dumps if enabled
- per-rank logs
- OTel trace if already enabled

Likely directions:

- distributed divergence or collective hang
- queue starvation or retraction storm
- PD transfer stall
- storage or HiCache backend stall

## Quick Paths

### TTFT spike

Start with:

- `/v1/loads?include=all`
- `/metrics`
- `/server_info`

Watch for:

- `num_waiting_reqs` growth
- `token_usage` saturation
- `cache_hit_rate` drop
- PD queue buildup

If queue pressure does not explain the slowdown, save the slow request and
replay it.

### Throughput collapse

Start with:

- `/v1/loads?include=all`
- `/metrics`
- benchmark reproduction if available

Watch for:

- low `gen_throughput`
- queue growth
- low cache hit rate
- speculative metrics collapse
- PD transfer or decode prealloc queues backing up

### Crash after some requests

Start with:

- crash dump folder
- stderr/stdout
- request dump folder if available

Then replay the crash dump or recent request dump.

### Regression between two commits

Start with:

- known-good commit
- known-bad commit
- one stable pass/fail harness

Best move:

- `git bisect run <harness>`

### One request class fails

Start with:

- exact request payload
- request dump if available
- smallest reproduction request

Typical categories:

- multimodal edge case
- parser or structured output bug
- model-specific kernel path
- tool-call formatting issue

## When To Switch Tools

### Use replay when

- a crash dump or request dump already exists
- the issue depends on request shape or workload mix
- you need one stable reproducer before going deeper

### Use OTel trace when

- request-stage timing is unclear
- router vs. worker ownership is unclear
- PD boundaries may be involved

### Use torch profiler when

- replay already reproduces the issue
- queueing and routing are mostly ruled out
- you need kernel-level attribution

At that point, switch to `llm-torch-profiler-analysis`.

### Use lower-level debug paths when

- replay plus trace still leave ambiguity
- the problem looks like a specific crash, hang, or correctness bug

## What To Return

- problem class
- what was checked
- strongest signal so far
- current best guess
- what was ruled out
- next step
- production risk
